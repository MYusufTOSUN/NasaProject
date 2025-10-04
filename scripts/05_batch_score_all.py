#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Toplu Skor Üretimi
graphs/images/ altındaki TÜM görselleri skorlar ve predictions_detail.csv oluşturur
"""

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

try:
    from ultralytics import YOLO
except ImportError:
    print("✗ HATA: ultralytics kütüphanesi yüklü değil!")
    exit(1)


def batch_score_all(model_path='models/best.pt', images_dir='graphs/images', 
                    output_csv='evaluation_results/predictions_detail.csv'):
    """
    Tüm görselleri skorla ve CSV'ye kaydet
    """
    print("="*60)
    print("📊 Toplu Skor Üretimi")
    print("="*60)
    
    # Model kontrolü
    if not os.path.exists(model_path):
        print(f"✗ HATA: Model bulunamadı: {model_path}")
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
    
    print(f"✓ {len(image_files)} görsel bulundu")
    print(f"✓ Model: {model_path}\n")
    
    # Model yükle
    print("📦 Model yükleniyor...")
    model = YOLO(model_path)
    print("✓ Model yüklendi\n")
    
    # Tahminleri topla
    predictions = []
    
    print("🎯 Skorlama başlıyor...")
    for img_path in tqdm(image_files, desc="İşleniyor"):
        try:
            results = model(str(img_path), verbose=False)
            result = results[0]
            
            # Sınıf isimleri
            class_names = result.names
            
            # Olasılıklar
            probs = result.probs.data.cpu().numpy()
            
            # Sınıf indexlerini bul
            # Varsayım: 0=negative, 1=positive
            neg_idx = 0 if class_names[0] == 'negative' else 1
            pos_idx = 1 if class_names[1] == 'positive' else 0
            
            conf_neg = probs[neg_idx]
            conf_pos = probs[pos_idx]
            
            # Tahmin edilen sınıf
            pred_idx = probs.argmax()
            pred_label = class_names[pred_idx]
            
            predictions.append({
                'image_path': str(img_path.relative_to(Path.cwd())),
                'pred_label': pred_label,
                'conf_pos': float(conf_pos),
                'conf_neg': float(conf_neg),
            })
            
        except Exception as e:
            print(f"\n⚠ Hata ({img_path.name}): {e}")
            continue
    
    # DataFrame oluştur
    df = pd.DataFrame(predictions)
    
    # Kaydet
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Tahminler kaydedildi: {output_path}")
    print(f"  Toplam: {len(df)} tahmin")
    print(f"  Positive tahmin: {(df['pred_label'] == 'positive').sum()}")
    print(f"  Negative tahmin: {(df['pred_label'] == 'negative').sum()}")
    
    print("\n" + "="*60)
    print("✓ Skorlama tamamlandı!")
    print("  Şimdi değerlendirme yapabilirsiniz:")
    print("  python scripts/evaluate_model.py")
    print("="*60)


def main():
    batch_score_all()


if __name__ == '__main__':
    main()

