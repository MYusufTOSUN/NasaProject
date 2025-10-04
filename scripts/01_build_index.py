#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Görsel indeksleyici
graphs/images/ altındaki görselleri tarar ve index.csv oluşturur
"""

import os
import pandas as pd
import re
from pathlib import Path


def extract_info_from_filename(filename):
    """
    Dosya adından target ve mission bilgilerini çıkar
    Örnek: Kepler-10_Kepler_phase.png -> target=Kepler-10, mission=Kepler
    """
    # .png veya .jpg uzantısını kaldır
    name_without_ext = filename.rsplit('.', 1)[0]
    
    # Pattern: <target>_<mission>_phase
    parts = name_without_ext.split('_')
    
    if len(parts) >= 3:
        # Son iki part: mission ve "phase"
        mission = parts[-2]
        # İlk kısım target (birden fazla underscore olabilir)
        target = '_'.join(parts[:-2])
        is_phase = parts[-1] == 'phase'
        
        return {
            'target': target,
            'mission': mission,
            'is_phase': is_phase,
            'is_binned': True  # Varsayılan: binned
        }
    
    return None


def build_index(images_dir, output_csv):
    """
    Görsel klasörünü tara ve index oluştur
    """
    images_dir = Path(images_dir)
    
    if not images_dir.exists():
        print(f"✗ HATA: {images_dir} klasörü bulunamadı!")
        return
    
    # Tüm PNG dosyalarını bul
    image_files = list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg'))
    
    print(f"📂 {len(image_files)} görsel dosyası bulundu")
    
    # Index verilerini topla
    index_data = []
    
    for img_path in image_files:
        filename = img_path.name
        info = extract_info_from_filename(filename)
        
        if info is None:
            print(f"⚠ Atlanıyor (format uyumsuz): {filename}")
            continue
        
        # Relative path
        rel_path = str(img_path.relative_to(Path.cwd()))
        
        index_data.append({
            'target': info['target'],
            'mission': info['mission'],
            'label': '',  # Boş - sonraki adımda metadata ile birleştirilecek
            'image_path': rel_path,
            'is_binned': info['is_binned'],
            'is_phase': info['is_phase']
        })
    
    # DataFrame oluştur
    df = pd.DataFrame(index_data)
    
    # Kaydet
    df.to_csv(output_csv, index=False)
    
    print(f"\n✓ Index oluşturuldu: {output_csv}")
    print(f"  Toplam kayıt: {len(df)}")
    print(f"  Unique targets: {df['target'].nunique()}")
    print(f"  Misyonlar: {', '.join(df['mission'].unique())}")
    
    # İlk birkaç satırı göster
    print("\n📋 İlk 5 kayıt:")
    print(df.head())


def main():
    images_dir = 'graphs/images'
    output_csv = 'index.csv'
    
    print("="*60)
    print("🔍 Görsel İndeksleyici")
    print("="*60)
    
    build_index(images_dir, output_csv)
    
    print("\n" + "="*60)
    print("✓ İşlem tamamlandı")
    print("="*60)


if __name__ == '__main__':
    main()

