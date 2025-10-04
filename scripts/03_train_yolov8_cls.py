#!/usr/bin/env python3
"""
YOLOv8 sınıflandırma eğitimi sarmalayıcı script.
ultralytics Python API kullanarak model eğitir ve sonuçları kaydeder.
"""

import argparse
import shutil
import sys
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

try:
    from ultralytics import YOLO
except ImportError:
    print("HATA: ultralytics kütüphanesi bulunamadı!")
    print("Lütfen kurun: pip install ultralytics>=8.2.0")
    sys.exit(1)


def validate_data_dir(data_dir):
    """
    Dataset klasörünü doğrula.
    
    Args:
        data_dir: Dataset klasörü yolu
        
    Returns:
        bool: Geçerli mi
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"HATA: Dataset klasörü bulunamadı: {data_dir}")
        return False
    
    # Train/val/test klasörlerini kontrol et
    required_dirs = ['train/positive', 'train/negative', 'val/positive', 'val/negative']
    
    for dir_name in required_dirs:
        dir_path = data_path / dir_name
        if not dir_path.exists():
            print(f"HATA: Gerekli klasör bulunamadı: {dir_path}")
            return False
        
        # Klasörde dosya var mı kontrol et
        images = list(dir_path.glob('*.png')) + list(dir_path.glob('*.jpg'))
        if len(images) == 0:
            print(f"UYARI: {dir_path} klasöründe görsel bulunamadı!")
    
    # Test klasörünü kontrol et (opsiyonel)
    test_dirs = ['test/positive', 'test/negative']
    has_test = all((data_path / d).exists() for d in test_dirs)
    
    if has_test:
        print(f"✓ Dataset doğrulandı: {data_path.resolve()}")
        print(f"  - Train, Val ve Test setleri mevcut")
    else:
        print(f"✓ Dataset doğrulandı: {data_path.resolve()}")
        print(f"  - Train ve Val setleri mevcut (Test yok)")
    
    return True


def count_images(data_dir):
    """
    Dataset'teki görsel sayılarını hesapla.
    
    Args:
        data_dir: Dataset klasörü
        
    Returns:
        dict: Split -> label -> count mapping
    """
    data_path = Path(data_dir)
    counts = {}
    
    for split in ['train', 'val', 'test']:
        split_path = data_path / split
        if not split_path.exists():
            continue
        
        counts[split] = {}
        for label in ['positive', 'negative']:
            label_path = split_path / label
            if label_path.exists():
                images = list(label_path.glob('*.png')) + list(label_path.glob('*.jpg')) + list(label_path.glob('*.jpeg'))
                counts[split][label] = len(images)
            else:
                counts[split][label] = 0
    
    return counts


def print_dataset_info(counts):
    """
    Dataset bilgilerini yazdır.
    
    Args:
        counts: Görsel sayıları
    """
    print("\n" + "=" * 60)
    print("DATASET BİLGİLERİ")
    print("=" * 60)
    
    for split in ['train', 'val', 'test']:
        if split not in counts:
            continue
        
        positive = counts[split].get('positive', 0)
        negative = counts[split].get('negative', 0)
        total = positive + negative
        
        if total > 0:
            ratio = positive / total * 100
            print(f"\n{split.upper()} Set: {total} görsel")
            print(f"  Positive (transit var): {positive} ({ratio:.1f}%)")
            print(f"  Negative (transit yok): {negative} ({100-ratio:.1f}%)")
    
    print("=" * 60)


def train_model(args):
    """
    Model eğitimi.
    
    Args:
        args: Komut satırı argümanları
        
    Returns:
        bool: Başarılı mı
    """
    try:
        print("\n" + "=" * 60)
        print("MODEL EĞİTİMİ BAŞLIYOR")
        print("=" * 60)
        print(f"Model: {args.model}")
        print(f"Epochs: {args.epochs}")
        print(f"Image size: {args.imgsz}")
        print(f"Batch size: {args.batch}")
        print(f"Device: {args.device}")
        print(f"Data directory: {Path(args.data_dir).resolve()}")
        print(f"Project: {args.project}")
        print(f"Name: {args.name}")
        print(f"Pretrained: {args.pretrained}")
        print("=" * 60)
        
        # Modeli yükle
        print(f"\nModel yükleniyor: {args.model}")
        model = YOLO(args.model)
        print("✓ Model yüklendi")
        
        # Eğitim parametreleri
        train_kwargs = {
            'data': args.data_dir,
            'epochs': args.epochs,
            'imgsz': args.imgsz,
            'batch': args.batch,
            'device': args.device,
            'project': args.project,
            'name': args.name,
            'pretrained': args.pretrained,
            'optimizer': args.optimizer,
            'lr0': args.lr0,
            'verbose': True,
            'patience': args.patience,
            'save': True,
            'save_period': args.save_period,
            'workers': args.workers,
            'exist_ok': args.exist_ok,
        }
        
        # Augmentation parametreleri
        if args.augment:
            train_kwargs.update({
                'hsv_h': 0.015,  # HSV-Hue augmentation
                'hsv_s': 0.7,    # HSV-Saturation augmentation
                'hsv_v': 0.4,    # HSV-Value augmentation
                'degrees': 0.0,  # Rotation
                'translate': 0.1, # Translation
                'scale': 0.5,    # Scale
                'shear': 0.0,    # Shear
                'perspective': 0.0, # Perspective
                'flipud': 0.0,   # Flip up-down
                'fliplr': 0.5,   # Flip left-right
                'mosaic': 1.0,   # Mosaic augmentation
                'mixup': 0.0,    # Mixup augmentation
            })
        
        # Cache kullanımı
        if args.cache:
            train_kwargs['cache'] = args.cache
        
        print("\nEğitim başlıyor...")
        print("(Bu işlem uzun sürebilir. Lütfen bekleyin...)\n")
        
        # Eğitimi başlat
        results = model.train(**train_kwargs)
        
        print("\n" + "=" * 60)
        print("✓ EĞİTİM TAMAMLANDI")
        print("=" * 60)
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n⚠ Eğitim kullanıcı tarafından durduruldu!")
        return False
        
    except Exception as e:
        print(f"\n\nHATA: Eğitim sırasında hata oluştu: {e}")
        import traceback
        traceback.print_exc()
        return False


def copy_best_model(project, name, output_dir='models'):
    """
    En iyi modeli models/ klasörüne kopyala.
    
    Args:
        project: Proje klasörü
        name: Eğitim adı
        output_dir: Çıktı klasörü
        
    Returns:
        bool: Başarılı mı
    """
    try:
        # Kaynak dosya
        source_path = Path(project) / name / 'weights' / 'best.pt'
        
        if not source_path.exists():
            print(f"\n⚠ UYARI: En iyi model bulunamadı: {source_path}")
            # last.pt'yi dene
            source_path = Path(project) / name / 'weights' / 'last.pt'
            if not source_path.exists():
                print(f"⚠ UYARI: Son model de bulunamadı: {source_path}")
                return False
            print(f"ℹ Son model (last.pt) kullanılacak")
        
        # Hedef klasör
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Hedef dosya
        dest_path = output_path / 'best.pt'
        
        # Yedek al (varsa)
        if dest_path.exists():
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = output_path / f'best_backup_{timestamp}.pt'
            shutil.copy2(dest_path, backup_path)
            print(f"✓ Mevcut model yedeklendi: {backup_path.name}")
        
        # Kopyala
        shutil.copy2(source_path, dest_path)
        
        print(f"✓ En iyi model kopyalandı: {dest_path.resolve()}")
        print(f"  Kaynak: {source_path.resolve()}")
        
        # Dosya boyutunu göster
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"  Boyut: {size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"\nHATA: Model kopyalama hatası: {e}")
        return False


def print_results_summary(project, name):
    """
    Eğitim sonuçlarını özetle.
    
    Args:
        project: Proje klasörü
        name: Eğitim adı
    """
    results_dir = Path(project) / name
    
    print("\n" + "=" * 60)
    print("EĞİTİM SONUÇLARI")
    print("=" * 60)
    print(f"Sonuç klasörü: {results_dir.resolve()}")
    
    # Weights
    weights_dir = results_dir / 'weights'
    if weights_dir.exists():
        best_pt = weights_dir / 'best.pt'
        last_pt = weights_dir / 'last.pt'
        
        print(f"\nModel dosyaları:")
        if best_pt.exists():
            size = best_pt.stat().st_size / (1024 * 1024)
            print(f"  ✓ best.pt ({size:.2f} MB)")
        if last_pt.exists():
            size = last_pt.stat().st_size / (1024 * 1024)
            print(f"  ✓ last.pt ({size:.2f} MB)")
    
    # Results
    results_files = [
        'results.csv',
        'results.png',
        'confusion_matrix.png',
        'confusion_matrix_normalized.png'
    ]
    
    available_results = []
    for file in results_files:
        file_path = results_dir / file
        if file_path.exists():
            available_results.append(file)
    
    if available_results:
        print(f"\nSonuç dosyaları:")
        for file in available_results:
            print(f"  ✓ {file}")
    
    print("=" * 60)


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(
        description='YOLOv8 sınıflandırma modeli eğitimi',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Temel parametreler
    parser.add_argument('--model', default='yolov8n-cls.pt',
                       help='Model dosyası (örn: yolov8n-cls.pt, yolov8s-cls.pt)')
    parser.add_argument('--data_dir', default='data/plots',
                       help='Dataset klasörü (train/val/test içermeli)')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Eğitim epoch sayısı')
    parser.add_argument('--imgsz', type=int, default=224,
                       help='Görsel boyutu (piksel)')
    parser.add_argument('--batch', type=int, default=64,
                       help='Batch boyutu')
    parser.add_argument('--device', default='0',
                       help='Device (0, 1, 2, ... veya cpu)')
    
    # Proje parametreleri
    parser.add_argument('--project', default='runs/classify',
                       help='Proje klasörü')
    parser.add_argument('--name', default='exp_cls',
                       help='Eğitim adı')
    parser.add_argument('--exist_ok', action='store_true',
                       help='Mevcut proje klasörünün üzerine yaz')
    
    # Eğitim parametreleri
    parser.add_argument('--pretrained', type=bool, default=True,
                       help='Pretrained weights kullan')
    parser.add_argument('--optimizer', default='auto',
                       choices=['SGD', 'Adam', 'AdamW', 'auto'],
                       help='Optimizer seçimi')
    parser.add_argument('--lr0', type=float, default=0.01,
                       help='İlk learning rate')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience (epoch)')
    parser.add_argument('--save_period', type=int, default=-1,
                       help='Model kaydetme periyodu (-1: sadece best)')
    parser.add_argument('--workers', type=int, default=8,
                       help='DataLoader worker sayısı')
    
    # Augmentation
    parser.add_argument('--augment', action='store_true',
                       help='Data augmentation kullan')
    parser.add_argument('--cache', choices=['ram', 'disk', None], default=None,
                       help='Görsel cache stratejisi')
    
    # Model kaydetme
    parser.add_argument('--output_dir', default='models',
                       help='En iyi modelin kaydedileceği klasör')
    parser.add_argument('--skip_copy', action='store_true',
                       help='Model kopyalamayı atla')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("YOLOv8 SINIFLANDIRMA EĞİTİMİ")
    print("=" * 60)
    print(f"Başlangıç zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 1. Dataset'i doğrula
    if not validate_data_dir(args.data_dir):
        sys.exit(1)
    
    # 2. Dataset bilgilerini göster
    counts = count_images(args.data_dir)
    print_dataset_info(counts)
    
    # 3. Kullanıcıdan onay al
    print("\nEğitime başlamak için hazır.")
    print(f"Tahmini süre: {args.epochs} epoch (~{args.epochs * 0.5:.0f}-{args.epochs:.0f} dakika)")
    
    # 4. Eğitimi başlat
    success = train_model(args)
    
    if not success:
        print("\n⚠ Eğitim başarısız oldu veya durduruldu!")
        sys.exit(1)
    
    # 5. Sonuçları özetle
    print_results_summary(args.project, args.name)
    
    # 6. En iyi modeli kopyala
    if not args.skip_copy:
        print("\nEn iyi model kopyalanıyor...")
        copy_success = copy_best_model(args.project, args.name, args.output_dir)
        
        if not copy_success:
            print("⚠ Model kopyalama başarısız!")
    
    # 7. Final mesajı
    print("\n" + "=" * 60)
    print("✓ TÜM İŞLEMLER TAMAMLANDI")
    print("=" * 60)
    print(f"Bitiş zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nSonuç klasörü: {Path(args.project) / args.name}")
    if not args.skip_copy:
        print(f"Model dosyası: {Path(args.output_dir) / 'best.pt'}")
    print("\nModeli test etmek için:")
    print(f"  python scripts/04_predict_folder.py --model {Path(args.output_dir) / 'best.pt'} --source test_images/")
    print("=" * 60)


if __name__ == "__main__":
    main()
