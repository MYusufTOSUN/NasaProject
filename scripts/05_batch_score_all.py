#!/usr/bin/env python3
"""
Tüm grafikleri skorlama scripti.
graphs/images klasöründeki tüm PNG dosyalarını tarar ve tahmin yapar.
Hata veren dosyaları listeler.
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


def find_all_images(images_dir, extensions=None):
    """
    Klasördeki tüm görsel dosyalarını bul.
    
    Args:
        images_dir: Görsel klasörü
        extensions: Desteklenen uzantılar
        
    Returns:
        list: Görsel dosya yolları
    """
    images_path = Path(images_dir)
    
    if not images_path.exists():
        print(f"HATA: Klasör bulunamadı: {images_dir}")
        sys.exit(1)
    
    if extensions is None:
        extensions = ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']
    
    # Tüm görselleri bul (recursive)
    all_images = []
    for ext in extensions:
        all_images.extend(images_path.rglob(ext))
    
    # Sırala
    all_images = sorted(all_images)
    
    print(f"✓ {len(all_images)} görsel bulundu: {images_path.resolve()}")
    
    return all_images


def predict_single_image(model, image_path, imgsz=224):
    """
    Tek bir görsel için tahmin yap.
    
    Args:
        model: YOLO modeli
        image_path: Görsel yolu
        imgsz: Görsel boyutu
        
    Returns:
        tuple: (success, pred_label, prob_positive, prob_negative, error_msg)
    """
    try:
        # Tahmin
        results = model.predict(
            source=str(image_path),
            verbose=False,
            imgsz=imgsz,
            device=None
        )
        
        # İlk (ve tek) sonucu al
        result = results[0]
        probs = result.probs
        
        # Probabilities
        prob_values = probs.data.cpu().numpy()
        
        # Predicted class
        pred_class = int(probs.top1)
        
        # Class probabilities (negative=0, positive=1)
        prob_negative = float(prob_values[0])
        prob_positive = float(prob_values[1])
        
        # Predicted label
        pred_label = pred_class  # 0 veya 1
        
        return True, pred_label, prob_positive, prob_negative, None
        
    except Exception as e:
        return False, None, None, None, str(e)


def batch_predict_with_errors(model, image_paths, batch_size=32, imgsz=224):
    """
    Batch halinde tahmin yap ve hataları yakala.
    
    Args:
        model: YOLO modeli
        image_paths: Görsel yolları
        batch_size: Batch boyutu
        imgsz: Görsel boyutu
        
    Returns:
        tuple: (predictions, errors)
            predictions: [(image_path, pred_label, prob_pos, prob_neg), ...]
            errors: [(image_path, error_msg), ...]
    """
    predictions = []
    errors = []
    
    # Her görseli tek tek işle (hata yakalama için)
    for image_path in tqdm(image_paths, desc="Tahmin yapılıyor", unit="görsel"):
        success, pred_label, prob_pos, prob_neg, error_msg = predict_single_image(
            model, image_path, imgsz
        )
        
        if success:
            predictions.append((
                str(image_path.resolve()),
                pred_label,
                prob_pos,
                prob_neg
            ))
        else:
            errors.append((str(image_path.resolve()), error_msg))
    
    return predictions, errors


def save_predictions_csv(predictions, output_path):
    """
    Tahminleri CSV'ye kaydet.
    
    Args:
        predictions: Tahmin listesi
        output_path: Çıktı dosyası
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['image_path', 'pred_label', 'prob_positive', 'prob_negative', 'confidence'])
        
        # Her satır
        for img_path, pred_label, prob_pos, prob_neg in predictions:
            # Confidence = maximum probability
            confidence = max(prob_pos, prob_neg)
            
            writer.writerow([
                img_path,
                pred_label,
                f"{prob_pos:.6f}",
                f"{prob_neg:.6f}",
                f"{confidence:.6f}"
            ])
    
    print(f"✓ Tahminler kaydedildi: {output_file.resolve()}")


def save_errors_log(errors, output_dir):
    """
    Hataları log dosyasına kaydet.
    
    Args:
        errors: Hata listesi
        output_dir: Çıktı klasörü
    """
    if not errors:
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    error_log = output_dir / 'prediction_errors.log'
    
    with open(error_log, 'w', encoding='utf-8') as f:
        f.write("TAHMIN HATALARI\n")
        f.write("=" * 60 + "\n\n")
        
        for img_path, error_msg in errors:
            f.write(f"Dosya: {img_path}\n")
            f.write(f"Hata: {error_msg}\n")
            f.write("-" * 60 + "\n")
    
    print(f"⚠ Hata logu kaydedildi: {error_log.resolve()}")


def print_summary(predictions, errors, image_paths):
    """
    Özet rapor yazdır.
    
    Args:
        predictions: Tahmin listesi
        errors: Hata listesi
        image_paths: Tüm görsel yolları
    """
    print("\n" + "=" * 60)
    print("TAHMİN ÖZETİ")
    print("=" * 60)
    
    total = len(image_paths)
    success_count = len(predictions)
    error_count = len(errors)
    
    print(f"Toplam görsel: {total}")
    print(f"Başarılı tahmin: {success_count} ({success_count/total*100:.1f}%)")
    
    if error_count > 0:
        print(f"Hatalı: {error_count} ({error_count/total*100:.1f}%)")
    
    if success_count > 0:
        # Tahmin dağılımı
        pred_positive = sum(1 for _, label, _, _ in predictions if label == 1)
        pred_negative = success_count - pred_positive
        
        print(f"\nTahmin dağılımı:")
        print(f"  Positive (transit var): {pred_positive} ({pred_positive/success_count*100:.1f}%)")
        print(f"  Negative (transit yok): {pred_negative} ({pred_negative/success_count*100:.1f}%)")
        
        # Güven skorları
        confidences = [max(prob_pos, prob_neg) for _, _, prob_pos, prob_neg in predictions]
        avg_conf = sum(confidences) / len(confidences)
        min_conf = min(confidences)
        max_conf = max(confidences)
        
        print(f"\nGüven skorları:")
        print(f"  Ortalama: {avg_conf:.4f}")
        print(f"  Minimum: {min_conf:.4f}")
        print(f"  Maksimum: {max_conf:.4f}")
        
        # Düşük güvenli tahminler
        low_conf_threshold = 0.6
        low_conf_count = sum(1 for c in confidences if c < low_conf_threshold)
        
        if low_conf_count > 0:
            print(f"\nDüşük güvenli tahminler (<{low_conf_threshold}): {low_conf_count} ({low_conf_count/success_count*100:.1f}%)")
    
    if error_count > 0:
        print(f"\n⚠ HATA DETAYLARI:")
        print(f"  Toplam {error_count} görsel işlenemedi")
        
        # İlk 5 hatayı göster
        print(f"\n  İlk {min(5, error_count)} hata:")
        for img_path, error_msg in errors[:5]:
            filename = Path(img_path).name
            print(f"    - {filename}: {error_msg[:50]}...")
        
        if error_count > 5:
            print(f"    ... ve {error_count - 5} hata daha (bkz: prediction_errors.log)")
    
    print("=" * 60)


def extract_target_info(image_path):
    """
    Görsel yolundan hedef bilgisini çıkar (analiz için).
    
    Args:
        image_path: Görsel yolu
        
    Returns:
        dict: {target, mission, image_type} veya None
    """
    try:
        filename = Path(image_path).stem  # Uzantısız dosya adı
        
        # Dosya adı formatı: Target_Mission_type.png
        # Örnek: Kepler_10_Kepler_overview.png
        
        parts = filename.split('_')
        
        if len(parts) >= 3:
            # Son kısım image type (overview/phase)
            image_type = parts[-1] if parts[-1] in ['overview', 'phase', 'phasefold'] else 'unknown'
            
            # Mission genellikle ortada
            mission = None
            for part in parts:
                if part.lower() in ['kepler', 'tess', 'k2']:
                    mission = part
                    break
            
            # Target, filename'in başlangıcı
            if mission:
                target_parts = []
                for part in parts:
                    if part == mission:
                        break
                    target_parts.append(part)
                target = '_'.join(target_parts) if target_parts else 'unknown'
            else:
                target = 'unknown'
                mission = 'unknown'
            
            return {
                'target': target,
                'mission': mission,
                'image_type': image_type
            }
    except:
        pass
    
    return None


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(
        description='Tüm grafikleri skorlama - toplu tahmin',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--weights', '--model', dest='weights', required=True,
                       help='Model dosyası (örn: models/best.pt)')
    parser.add_argument('--images_dir', required=True,
                       help='Görsel klasörü (örn: graphs/images)')
    parser.add_argument('--out', default='evaluation_results/all_graphs_predictions.csv',
                       help='Çıktı CSV dosyası')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch boyutu (kullanılmıyor, tek tek işleme yapılıyor)')
    parser.add_argument('--imgsz', type=int, default=224,
                       help='Görsel boyutu')
    parser.add_argument('--extensions', nargs='+', default=['*.png', '*.PNG'],
                       help='Dosya uzantıları')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TÜM GRAFİKLERİ SKORLAMA")
    print("=" * 60)
    print(f"Model: {args.weights}")
    print(f"Görsel klasörü: {args.images_dir}")
    print(f"Çıktı dosyası: {args.out}")
    print(f"Görsel boyutu: {args.imgsz}")
    print("=" * 60)
    print()
    
    # 1. Model dosyasını kontrol et
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"HATA: Model dosyası bulunamadı: {args.weights}")
        sys.exit(1)
    
    # 2. Görselleri bul
    print("Görseller taranıyor...")
    image_paths = find_all_images(args.images_dir, args.extensions)
    
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
    
    # 4. Tahmin yap (hata yakalama ile)
    print(f"\nTahmin yapılıyor ({len(image_paths)} görsel)...")
    print("(Her görsel tek tek işleniyor - hata yakalama için)\n")
    
    try:
        predictions, errors = batch_predict_with_errors(
            model, image_paths, batch_size=args.batch_size, imgsz=args.imgsz
        )
    except KeyboardInterrupt:
        print("\n\n⚠ İşlem kullanıcı tarafından durduruldu!")
        sys.exit(130)
    except Exception as e:
        print(f"\nHATA: Beklenmeyen hata: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 5. Sonuçları kaydet
    if predictions:
        print("\nSonuçlar kaydediliyor...")
        save_predictions_csv(predictions, args.out)
    else:
        print("\n⚠ Hiç başarılı tahmin yok!")
    
    # 6. Hataları kaydet
    if errors:
        output_dir = Path(args.out).parent
        save_errors_log(errors, output_dir)
    
    # 7. Özet yazdır
    print_summary(predictions, errors, image_paths)
    
    # 8. Final mesajı
    print("\n" + "=" * 60)
    if len(predictions) > 0:
        print("✓ SKORLAMA TAMAMLANDI")
    else:
        print("⚠ SKORLAMA BAŞARISIZ")
    print("=" * 60)
    
    if predictions:
        print(f"Tahmin dosyası: {Path(args.out).resolve()}")
    
    if errors:
        print(f"Hata logu: {Path(args.out).parent / 'prediction_errors.log'}")
    
    print(f"\nBaşarı oranı: {len(predictions)}/{len(image_paths)} ({len(predictions)/len(image_paths)*100:.1f}%)")
    print("=" * 60)
    
    # Exit code
    if len(predictions) == 0:
        sys.exit(1)
    elif len(errors) > 0:
        sys.exit(2)  # Partial success
    else:
        sys.exit(0)  # Full success


if __name__ == "__main__":
    main()

