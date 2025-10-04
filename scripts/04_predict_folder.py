#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hızlı Klasör Testi
graphs/images/ içindeki ilk 20 görseli test eder
"""

import os
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("✗ HATA: ultralytics kütüphanesi yüklü değil!")
    exit(1)


def predict_folder(model_path='models/best.pt', images_dir='graphs/images', limit=20):
    """
    Klasördeki görselleri test et
    """
    print("="*60)
    print("🔍 Hızlı Klasör Testi")
    print("="*60)
    
    # Model kontrolü
    if not os.path.exists(model_path):
        print(f"✗ HATA: Model bulunamadı: {model_path}")
        print("  Önce model eğitimi yapın: scripts/03_train_yolov8_cls.py")
        return
    
    # Klasör kontrolü
    images_dir = Path(images_dir)
    if not images_dir.exists():
        print(f"✗ HATA: Klasör bulunamadı: {images_dir}")
        return
    
    # Görselleri bul
    image_files = list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg'))
    
    if len(image_files) == 0:
        print(f"✗ HATA: {images_dir} içinde görsel bulunamadı!")
        return
    
    # Limit uygula
    image_files = image_files[:limit]
    
    print(f"✓ Model: {model_path}")
    print(f"✓ {len(image_files)} görsel test edilecek\n")
    
    # Model yükle
    print("📦 Model yükleniyor...")
    model = YOLO(model_path)
    print("✓ Model yüklendi\n")
    
    # Tahminler
    print("🎯 Tahminler:")
    print("-"*60)
    
    for img_path in image_files:
        results = model(str(img_path), verbose=False)
        
        # İlk sonucu al
        result = results[0]
        
        # Sınıf isimleri
        class_names = result.names
        
        # En yüksek skorlu sınıf
        probs = result.probs.data.cpu().numpy()
        top_idx = probs.argmax()
        top_class = class_names[top_idx]
        top_conf = probs[top_idx]
        
        # Çıktı
        print(f"{img_path.name:50s} | {top_class:10s} | Güven: {top_conf:.4f}")
    
    print("-"*60)
    print("\n✓ Test tamamlandı!")


def main():
    predict_folder()


if __name__ == '__main__':
    main()

