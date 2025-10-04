#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Veri bölme ve YOLOv8 dataset oluşturma
- index.csv ile metadata'yı birleştirir
- Aynı hedefe ait görselleri aynı split'e atar
- Stratified split: %70 train, %15 val, %15 test
- data.yaml oluşturur
"""

import os
import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml


def merge_index_metadata(index_csv, metadata_csv):
    """
    index.csv ile metadata'yı birleştir, label değerlerini al
    """
    print("📊 Index ve metadata birleştiriliyor...")
    
    df_index = pd.read_csv(index_csv)
    df_metadata = pd.read_csv(metadata_csv)
    
    # Target üzerinden merge
    df_merged = df_index.merge(
        df_metadata[['target', 'label']], 
        on='target', 
        how='left',
        suffixes=('', '_meta')
    )
    
    # Label sütununu güncelle
    df_merged['label'] = df_merged['label_meta'].fillna(df_merged['label'])
    df_merged = df_merged.drop(columns=['label_meta'], errors='ignore')
    
    # Label eksik olanları kontrol et
    missing_labels = df_merged['label'].isna().sum()
    if missing_labels > 0:
        print(f"⚠ {missing_labels} görselin label'ı yok, 'negative' olarak işaretleniyor")
        df_merged['label'] = df_merged['label'].fillna('negative')
    
    # Label'ları normalize et (positive/negative)
    df_merged['label'] = df_merged['label'].str.lower()
    df_merged['label'] = df_merged['label'].apply(
        lambda x: 'positive' if x in ['positive', 'transit', '1', 1, 'yes'] else 'negative'
    )
    
    print(f"✓ Birleştirildi: {len(df_merged)} kayıt")
    print(f"  Positive: {(df_merged['label'] == 'positive').sum()}")
    print(f"  Negative: {(df_merged['label'] == 'negative').sum()}")
    
    return df_merged


def split_dataset_by_target(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Dataset'i target bazlı böl (aynı target aynı split'te)
    Stratified split: pozitif/negatif oranını koru
    """
    print("\n🔀 Dataset bölünüyor...")
    
    # Her target için ana label'ı belirle (çoğunluk oyu)
    target_labels = df.groupby('target')['label'].agg(lambda x: x.mode()[0]).reset_index()
    target_labels.columns = ['target', 'main_label']
    
    # Önce train ve temp (val+test) olarak böl
    train_targets, temp_targets = train_test_split(
        target_labels['target'],
        test_size=(val_ratio + test_ratio),
        stratify=target_labels['main_label'],
        random_state=random_state
    )
    
    # Temp'i val ve test olarak böl
    temp_labels = target_labels[target_labels['target'].isin(temp_targets)]
    val_targets, test_targets = train_test_split(
        temp_labels['target'],
        test_size=(test_ratio / (val_ratio + test_ratio)),
        stratify=temp_labels['main_label'],
        random_state=random_state
    )
    
    # Her split için DataFrame oluştur
    df_train = df[df['target'].isin(train_targets)]
    df_val = df[df['target'].isin(val_targets)]
    df_test = df[df['target'].isin(test_targets)]
    
    print(f"\n✓ Bölme tamamlandı:")
    print(f"  Train: {len(df_train)} görsel, {len(train_targets)} target")
    print(f"    - Positive: {(df_train['label'] == 'positive').sum()}")
    print(f"    - Negative: {(df_train['label'] == 'negative').sum()}")
    print(f"  Val: {len(df_val)} görsel, {len(val_targets)} target")
    print(f"    - Positive: {(df_val['label'] == 'positive').sum()}")
    print(f"    - Negative: {(df_val['label'] == 'negative').sum()}")
    print(f"  Test: {len(df_test)} görsel, {len(test_targets)} target")
    print(f"    - Positive: {(df_test['label'] == 'positive').sum()}")
    print(f"    - Negative: {(df_test['label'] == 'negative').sum()}")
    
    return df_train, df_val, df_test


def copy_images_to_splits(df_train, df_val, df_test, output_dir='data/plots'):
    """
    Görselleri split klasörlerine kopyala
    YOLOv8 classification format: data/plots/<split>/<class>/image.png
    """
    print(f"\n📁 Görseller kopyalanıyor: {output_dir}/")
    
    output_dir = Path(output_dir)
    
    splits = {
        'train': df_train,
        'val': df_val,
        'test': df_test
    }
    
    for split_name, df_split in splits.items():
        for label in ['positive', 'negative']:
            target_dir = output_dir / split_name / label
            target_dir.mkdir(parents=True, exist_ok=True)
        
        # Görselleri kopyala
        for _, row in df_split.iterrows():
            src_path = Path(row['image_path'])
            if not src_path.exists():
                print(f"⚠ Dosya bulunamadı, atlanıyor: {src_path}")
                continue
            
            label = row['label']
            dst_dir = output_dir / split_name / label
            dst_path = dst_dir / src_path.name
            
            shutil.copy2(src_path, dst_path)
        
        print(f"  ✓ {split_name}: {len(df_split)} görsel kopyalandı")
    
    print(f"\n✓ Tüm görseller kopyalandı")


def create_data_yaml(output_dir='data', dataset_name='Exoplanet Transit'):
    """
    YOLOv8 için data.yaml oluştur
    """
    yaml_path = Path(output_dir) / 'data.yaml'
    
    # Absolute path'leri hesapla
    base_path = Path.cwd() / output_dir / 'plots'
    
    data_yaml = {
        'path': str(base_path),
        'train': 'train',
        'val': 'val',
        'test': 'test',
        
        'names': {
            0: 'negative',
            1: 'positive'
        },
        
        'nc': 2,  # number of classes
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✓ data.yaml oluşturuldu: {yaml_path}")
    print(f"  Path: {data_yaml['path']}")
    print(f"  Classes: {data_yaml['names']}")


def main():
    print("="*60)
    print("🔄 Veri Bölme ve Dataset Oluşturma")
    print("="*60)
    
    # Girdi dosyaları
    index_csv = 'index.csv'
    metadata_csv = 'data/metadata/metadata1500.csv'
    
    if not os.path.exists(index_csv):
        print(f"✗ HATA: {index_csv} bulunamadı!")
        print("  Önce scripts/01_build_index.py çalıştırın.")
        return
    
    if not os.path.exists(metadata_csv):
        print(f"✗ HATA: {metadata_csv} bulunamadı!")
        print("  Metadata dosyasını data/metadata/ klasörüne yerleştirin.")
        return
    
    # 1. Merge
    df = merge_index_metadata(index_csv, metadata_csv)
    
    # 2. Split
    df_train, df_val, df_test = split_dataset_by_target(df)
    
    # 3. Görselleri kopyala
    copy_images_to_splits(df_train, df_val, df_test)
    
    # 4. data.yaml oluştur
    create_data_yaml()
    
    print("\n" + "="*60)
    print("✓ Dataset hazır!")
    print("  Şimdi YOLOv8 eğitimini başlatabilirsiniz:")
    print("  python scripts/03_train_yolov8_cls.py")
    print("="*60)


if __name__ == '__main__':
    main()

