#!/usr/bin/env python3
"""
Klasör bazlı toplu tahmin scripti.
Bir klasördeki tüm görseller için YOLOv8 sınıflandırma tahmini yapar ve CSV'ye kaydeder.
"""

import argparse
import csv
import sys
import warnings
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings('ignore')

try:
    from ultralytics import YOLO
except ImportError:
    print("HATA: ultralytics kütüphanesi bulunamadı!")
    print("Lütfen kurun: pip install ultralytics>=8.2.0")
    sys.exit(1)


def infer_true_label(image_path):
    """
    Görsel path'inden true label'ı çıkar.
    
    Args:
        image_path: Görsel dosya yolu
        
    Returns:
        int or None: 1 (positive), 0 (negative), veya None (bulunamadı)
    """
    path = Path(image_path)
    
    # Path'teki tüm parent klasörleri kontrol et
    for parent in path.parents:
        parent_name = parent.name.lower()
        
        if parent_name == 'positive':
            return 1
        elif parent_name == 'negative':
            return 0
    
    # Bulunamadı
    return None


def find_images(input_dir, recursive=True):
    """
    Klasördeki tüm görsel dosyalarını bul.
    
    Args:
        input_dir: Giriş klasörü
        recursive: Alt klasörleri de tara
        
    Returns:
        list: Görsel dosya yolları
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"HATA: Klasör bulunamadı: {input_dir}")
        sys.exit(1)
    
    # Desteklenen formatlar
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    
    images = []
    for ext in image_extensions:
        if recursive:
            images.extend(input_path.rglob(ext))
        else:
            images.extend(input_path.glob(ext))
    
    # Sırala (tutarlılık için)
    images = sorted(images)
    
    print(f"✓ {len(images)} görsel bulundu: {input_path.resolve()}")
    
    return images


def batch_predict(model, image_paths, batch_size=32):
    """
    Batch halinde tahmin yap.
    
    Args:
        model: YOLO modeli
        image_paths: Görsel yolları listesi
        batch_size: Batch boyutu
        
    Returns:
        list: Her görsel için (image_path, pred_label, prob_positive, prob_negative) tuple listesi
    """
    predictions = []
    
    # Batch'lere böl
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Tahmin yapılıyor", unit="batch"):
        batch_paths = image_paths[i:i + batch_size]
        
        # Batch tahmin
        # verbose=False: logları kapat
        results = model.predict(
            source=batch_paths,
            verbose=False,
            imgsz=224,  # Model'in eğitildiği boyut
            device=None  # Auto-detect
        )
        
        # Her sonucu işle
        for result, image_path in zip(results, batch_paths):
            # Probs object'ini al
            probs = result.probs
            
            # Class probabilities (numpy array)
            # probs.data: [prob_class0, prob_class1, ...]
            # YOLOv8'de sınıflar alfabetik sırada: negative (0), positive (1)
            prob_values = probs.data.cpu().numpy()
            
            # Predicted class
            pred_class = int(probs.top1)  # En yüksek olasılıklı sınıf
            
            # Class names'i kontrol et (negative=0, positive=1)
            # Model eğitilirken alfabetik sıralama kullanılır
            if pred_class == 0:
                # Negative sınıfı tahmin edildi
                prob_negative = float(prob_values[0])
                prob_positive = float(prob_values[1])
                pred_label = 0
            else:
                # Positive sınıfı tahmin edildi
                prob_negative = float(prob_values[0])
                prob_positive = float(prob_values[1])
                pred_label = 1
            
            predictions.append((
                str(image_path.resolve()),
                pred_label,
                prob_positive,
                prob_negative
            ))
    
    return predictions


def save_predictions(predictions, image_paths, output_path):
    """
    Tahminleri CSV'ye kaydet.
    
    Args:
        predictions: Tahmin listesi
        image_paths: Görsel yolları
        output_path: Çıktı CSV dosyası
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['image_path', 'true_label', 'pred_label', 'prob_positive', 'prob_negative'])
        
        # Her satır için
        for (img_path, pred_label, prob_pos, prob_neg), orig_path in zip(predictions, image_paths):
            # True label'ı çıkar
            true_label = infer_true_label(orig_path)
            
            # True label'ı string'e çevir (None ise boş)
            true_label_str = str(true_label) if true_label is not None else ''
            
            # Yaz
            writer.writerow([
                img_path,
                true_label_str,
                pred_label,
                f"{prob_pos:.6f}",
                f"{prob_neg:.6f}"
            ])
    
    print(f"✓ Tahminler kaydedildi: {output_file.resolve()}")


def print_summary(predictions, image_paths):
    """
    Tahmin özetini yazdır.
    
    Args:
        predictions: Tahmin listesi
        image_paths: Görsel yolları
    """
    print("\n" + "=" * 60)
    print("TAHMİN ÖZETİ")
    print("=" * 60)
    
    # Toplam
    total = len(predictions)
    print(f"Toplam görsel: {total}")
    
    # Tahmin dağılımı
    pred_positive = sum(1 for _, pred, _, _ in predictions if pred == 1)
    pred_negative = total - pred_positive
    
    print(f"\nTahmin edilen dağılım:")
    print(f"  Positive (transit var): {pred_positive} ({pred_positive/total*100:.1f}%)")
    print(f"  Negative (transit yok): {pred_negative} ({pred_negative/total*100:.1f}%)")
    
    # True label'ları kontrol et
    true_labels = [infer_true_label(p) for p in image_paths]
    has_true_labels = any(label is not None for label in true_labels)
    
    if has_true_labels:
        true_positive_count = sum(1 for label in true_labels if label == 1)
        true_negative_count = sum(1 for label in true_labels if label == 0)
        unknown_count = sum(1 for label in true_labels if label is None)
        
        print(f"\nGerçek dağılım:")
        print(f"  Positive: {true_positive_count}")
        print(f"  Negative: {true_negative_count}")
        if unknown_count > 0:
            print(f"  Bilinmeyen: {unknown_count}")
        
        # Doğruluk hesapla (sadece bilinen etiketler için)
        correct = 0
        labeled_count = 0
        
        for (_, pred_label, _, _), true_label in zip(predictions, true_labels):
            if true_label is not None:
                labeled_count += 1
                if pred_label == true_label:
                    correct += 1
        
        if labeled_count > 0:
            accuracy = correct / labeled_count * 100
            print(f"\nDoğruluk (labeled images): {correct}/{labeled_count} ({accuracy:.2f}%)")
    else:
        print("\nℹ True label bilgisi bulunamadı (path'te positive/negative klasörü yok)")
    
    # Güven skorları
    all_probs = [(prob_pos, prob_neg) for _, _, prob_pos, prob_neg in predictions]
    max_probs = [max(p, n) for p, n in all_probs]
    avg_confidence = sum(max_probs) / len(max_probs)
    min_confidence = min(max_probs)
    max_confidence = max(max_probs)
    
    print(f"\nGüven skorları:")
    print(f"  Ortalama: {avg_confidence:.4f}")
    print(f"  Minimum: {min_confidence:.4f}")
    print(f"  Maksimum: {max_confidence:.4f}")
    
    # Düşük güvenli tahminler
    low_confidence_threshold = 0.6
    low_confidence_count = sum(1 for prob in max_probs if prob < low_confidence_threshold)
    
    if low_confidence_count > 0:
        print(f"\nDüşük güvenli tahminler (<{low_confidence_threshold}): {low_confidence_count}")
    
    print("=" * 60)


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(
        description='Klasör bazlı toplu tahmin',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--weights', '--model', dest='weights', required=True,
                       help='Model dosyası (örn: models/best.pt)')
    parser.add_argument('--input_dir', '--source', dest='input_dir', required=True,
                       help='Giriş klasörü (görsellerin bulunduğu)')
    parser.add_argument('--out', default='evaluation_results/predictions_detail.csv',
                       help='Çıktı CSV dosyası')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch boyutu (GPU memory\'ye göre ayarlayın)')
    parser.add_argument('--recursive', action='store_true', default=True,
                       help='Alt klasörleri de tara')
    parser.add_argument('--no_recursive', action='store_false', dest='recursive',
                       help='Sadece ana klasörü tara')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("KLASÖR BAZLI TOPLU TAHMİN")
    print("=" * 60)
    print(f"Model: {args.weights}")
    print(f"Giriş klasörü: {args.input_dir}")
    print(f"Çıktı dosyası: {args.out}")
    print(f"Batch boyutu: {args.batch_size}")
    print(f"Recursive: {'Evet' if args.recursive else 'Hayır'}")
    print("=" * 60)
    print()
    
    # 1. Model dosyasını kontrol et
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"HATA: Model dosyası bulunamadı: {args.weights}")
        sys.exit(1)
    
    # 2. Görselleri bul
    print("Görseller taranıyor...")
    image_paths = find_images(args.input_dir, recursive=args.recursive)
    
    if len(image_paths) == 0:
        print("HATA: Hiç görsel bulunamadı!")
        sys.exit(1)
    
    # 3. Modeli yükle
    print(f"\nModel yükleniyor: {weights_path.resolve()}")
    try:
        model = YOLO(str(weights_path))
        print("✓ Model yüklendi")
    except Exception as e:
        print(f"HATA: Model yüklenirken hata oluştu: {e}")
        sys.exit(1)
    
    # 4. Tahmin yap
    print(f"\nTahmin yapılıyor ({len(image_paths)} görsel)...")
    try:
        predictions = batch_predict(model, image_paths, batch_size=args.batch_size)
    except Exception as e:
        print(f"\nHATA: Tahmin sırasında hata oluştu: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 5. Sonuçları kaydet
    print("\nSonuçlar kaydediliyor...")
    save_predictions(predictions, image_paths, args.out)
    
    # 6. Özet yazdır
    print_summary(predictions, image_paths)
    
    # 7. Final mesajı
    print("\n" + "=" * 60)
    print("✓ TAHMİN TAMAMLANDI")
    print("=" * 60)
    print(f"Detaylı sonuçlar: {Path(args.out).resolve()}")
    print("\nMetrik hesaplama için:")
    print(f"  python scripts/evaluate_model.py --predictions {args.out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
