#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8 Sınıflandırma Eğitimi
Pre-trained yolov8n-cls.pt ile transfer learning
"""

import os
import shutil
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("✗ HATA: ultralytics kütüphanesi yüklü değil!")
    print("  pip install ultralytics komutunu çalıştırın.")
    exit(1)


def train_yolov8_classifier():
    """
    YOLOv8 classification modelini eğit
    """
    print("="*60)
    print("🚀 YOLOv8 Sınıflandırma Eğitimi")
    print("="*60)
    
    # data.yaml kontrolü
    data_yaml = 'data/data.yaml'
    if not os.path.exists(data_yaml):
        print(f"✗ HATA: {data_yaml} bulunamadı!")
        print("  Önce scripts/02_split_build_dataset.py çalıştırın.")
        return
    
    print(f"✓ Dataset config: {data_yaml}")
    
    # Pre-trained model yükle
    print("\n📦 YOLOv8n-cls modeli yükleniyor...")
    model = YOLO('yolov8n-cls.pt')
    print("✓ Model yüklendi")
    
    # Eğitim parametreleri
    print("\n⚙️ Eğitim parametreleri:")
    params = {
        'data': data_yaml,
        'epochs': 200,
        'imgsz': 224,
        'batch': 64,
        'patience': 50,  # Early stopping
        'save': True,
        'project': 'runs/classify',
        'name': 'exoplanet_transit',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'verbose': True,
        'device': 'cuda:0',  # GPU varsa kullanılır, yoksa otomatik CPU'ya geçer
    }
    
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Eğitimi başlat
    print("\n🏋️ Eğitim başlıyor...")
    print("-"*60)
    
    try:
        results = model.train(**params)
        print("-"*60)
        print("✓ Eğitim tamamlandı!")
        
        # Best model'i kopyala
        best_model_src = Path('runs/classify/exoplanet_transit/weights/best.pt')
        best_model_dst = Path('models/best.pt')
        best_model_dst.parent.mkdir(parents=True, exist_ok=True)
        
        if best_model_src.exists():
            shutil.copy2(best_model_src, best_model_dst)
            print(f"\n✓ En iyi model kopyalandı: {best_model_dst}")
        else:
            print(f"\n⚠ Best model bulunamadı: {best_model_src}")
        
        # Eğitim sonuçlarını göster
        print("\n📊 Eğitim Sonuçları:")
        print(f"  En iyi epoch: {results.best_epoch if hasattr(results, 'best_epoch') else 'N/A'}")
        print(f"  Sonuç klasörü: runs/classify/exoplanet_transit/")
        
    except Exception as e:
        print(f"\n✗ HATA: Eğitim sırasında hata oluştu:")
        print(f"  {e}")
        return
    
    print("\n" + "="*60)
    print("✓ Model eğitimi başarıyla tamamlandı!")
    print("  Şimdi tahmin ve değerlendirme yapabilirsiniz:")
    print("  - python scripts/04_predict_folder.py")
    print("  - python scripts/05_batch_score_all.py")
    print("="*60)


def main():
    train_yolov8_classifier()


if __name__ == '__main__':
    main()

