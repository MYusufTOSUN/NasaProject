#!/usr/bin/env python3
"""
Stratified split (yıldız bazlı) ve dataset oluşturma scripti.
index.csv'den verileri okuyup train/val/test'e ayırır ve kopyalar.
"""

import argparse
import csv
import shutil
import sys
import yaml
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_index(index_path):
    """
    index.csv dosyasını oku.
    
    Args:
        index_path: Index CSV dosya yolu
        
    Returns:
        list: Her satır için dictionary listesi
    """
    index_data = []
    
    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                index_data.append({
                    'target': row['target'],
                    'mission': row['mission'],
                    'label': int(row['label']),
                    'image_path': row['image_path'],
                    'is_binned': int(row['is_binned']),
                    'is_phase': int(row['is_phase'])
                })
        
        print(f"✓ {len(index_data)} görsel yüklendi: {index_path}")
        return index_data
        
    except FileNotFoundError:
        print(f"HATA: {index_path} dosyası bulunamadı!")
        sys.exit(1)
    except Exception as e:
        print(f"HATA: Index okurken hata: {e}")
        sys.exit(1)


def group_by_target(index_data):
    """
    Görselleri target'a göre grupla.
    
    Args:
        index_data: Index verileri
        
    Returns:
        dict: {target: [image_list]} şeklinde dictionary
    """
    target_groups = defaultdict(list)
    
    for item in index_data:
        target = item['target']
        target_groups[target].append(item)
    
    return dict(target_groups)


def stratified_split_targets(target_groups, train_ratio, val_ratio, test_ratio, seed):
    """
    Hedefleri stratified şekilde train/val/test'e ayır.
    
    Args:
        target_groups: Hedef grupları
        train_ratio: Train oranı
        val_ratio: Validation oranı
        test_ratio: Test oranı
        seed: Random seed
        
    Returns:
        tuple: (train_targets, val_targets, test_targets)
    """
    # Her hedef için label'ı al (bir hedefin tüm görselleri aynı label'a sahip)
    targets = []
    labels = []
    
    for target, images in target_groups.items():
        targets.append(target)
        # İlk görselin label'ını al (hepsi aynı olmalı)
        labels.append(images[0]['label'])
    
    # Önce train ve (val+test) ayır
    train_targets, temp_targets, train_labels, temp_labels = train_test_split(
        targets, labels,
        train_size=train_ratio,
        stratify=labels,
        random_state=seed
    )
    
    # Sonra val ve test ayır
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    
    val_targets, test_targets, val_labels, test_labels = train_test_split(
        temp_targets, temp_labels,
        train_size=val_ratio_adjusted,
        stratify=temp_labels,
        random_state=seed
    )
    
    print(f"✓ Split tamamlandı:")
    print(f"  Train: {len(train_targets)} hedef")
    print(f"  Val: {len(val_targets)} hedef")
    print(f"  Test: {len(test_targets)} hedef")
    
    return train_targets, val_targets, test_targets


def copy_images(target_groups, train_targets, val_targets, test_targets, output_dir):
    """
    Görselleri ilgili klasörlere kopyala.
    
    Args:
        target_groups: Hedef grupları
        train_targets: Train hedefleri
        val_targets: Val hedefleri
        test_targets: Test hedefleri
        output_dir: Çıktı klasörü
        
    Returns:
        dict: İstatistikler
    """
    output_path = Path(output_dir)
    
    # Klasörleri oluştur
    for split in ['train', 'val', 'test']:
        for label_dir in ['positive', 'negative']:
            (output_path / split / label_dir).mkdir(parents=True, exist_ok=True)
    
    # Split -> targets mapping
    split_mapping = {
        'train': train_targets,
        'val': val_targets,
        'test': test_targets
    }
    
    # İstatistikler
    stats = {
        'train': {'positive': 0, 'negative': 0},
        'val': {'positive': 0, 'negative': 0},
        'test': {'positive': 0, 'negative': 0}
    }
    
    # Her split için kopyala
    total_images = sum(len(images) for images in target_groups.values())
    
    with tqdm(total=total_images, desc="Görseller kopyalanıyor", unit="görsel") as pbar:
        for split, targets in split_mapping.items():
            for target in targets:
                images = target_groups[target]
                
                for item in images:
                    # Kaynak dosya
                    src_path = Path(item['image_path'])
                    
                    if not src_path.exists():
                        print(f"\n⚠ UYARI: Dosya bulunamadı: {src_path}")
                        pbar.update(1)
                        continue
                    
                    # Hedef klasör
                    label_dir = 'positive' if item['label'] == 1 else 'negative'
                    dst_dir = output_path / split / label_dir
                    
                    # Hedef dosya adı (orijinal dosya adını koru)
                    dst_path = dst_dir / src_path.name
                    
                    # Dosya zaten varsa, unique isim oluştur
                    counter = 1
                    original_dst = dst_path
                    while dst_path.exists():
                        stem = original_dst.stem
                        suffix = original_dst.suffix
                        dst_path = dst_dir / f"{stem}_{counter}{suffix}"
                        counter += 1
                    
                    # Kopyala
                    shutil.copy2(src_path, dst_path)
                    
                    # İstatistik güncelle
                    stats[split][label_dir] += 1
                    
                    pbar.update(1)
    
    return stats


def print_statistics(stats):
    """
    İstatistikleri ekrana yazdır.
    
    Args:
        stats: İstatistik dictionary
    """
    print("\n" + "=" * 60)
    print("DATASET İSTATİSTİKLERİ")
    print("=" * 60)
    
    total_train = stats['train']['positive'] + stats['train']['negative']
    total_val = stats['val']['positive'] + stats['val']['negative']
    total_test = stats['test']['positive'] + stats['test']['negative']
    total_all = total_train + total_val + total_test
    
    print(f"\nToplam görsel: {total_all}")
    print(f"\nTrain Set: {total_train} ({total_train/total_all*100:.1f}%)")
    print(f"  Positive: {stats['train']['positive']}")
    print(f"  Negative: {stats['train']['negative']}")
    print(f"  Ratio: {stats['train']['positive']/(stats['train']['positive']+stats['train']['negative'])*100:.1f}% positive")
    
    print(f"\nValidation Set: {total_val} ({total_val/total_all*100:.1f}%)")
    print(f"  Positive: {stats['val']['positive']}")
    print(f"  Negative: {stats['val']['negative']}")
    print(f"  Ratio: {stats['val']['positive']/(stats['val']['positive']+stats['val']['negative'])*100:.1f}% positive")
    
    print(f"\nTest Set: {total_test} ({total_test/total_all*100:.1f}%)")
    print(f"  Positive: {stats['test']['positive']}")
    print(f"  Negative: {stats['test']['negative']}")
    print(f"  Ratio: {stats['test']['positive']/(stats['test']['positive']+stats['test']['negative'])*100:.1f}% positive")
    
    print("=" * 60)


def save_summary(stats, output_dir, train_targets, val_targets, test_targets):
    """
    Split özetini evaluation_results/summary.csv'ye kaydet.
    
    Args:
        stats: İstatistikler
        output_dir: Çıktı klasörü
        train_targets: Train hedefleri
        val_targets: Val hedefleri
        test_targets: Test hedefleri
    """
    summary_dir = Path('evaluation_results')
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / 'split_summary.csv'
    
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['split', 'positive', 'negative', 'total', 'positive_ratio', 'num_targets'])
        
        # Train
        train_total = stats['train']['positive'] + stats['train']['negative']
        train_ratio = stats['train']['positive'] / train_total if train_total > 0 else 0
        writer.writerow(['train', stats['train']['positive'], stats['train']['negative'], 
                        train_total, f"{train_ratio:.4f}", len(train_targets)])
        
        # Val
        val_total = stats['val']['positive'] + stats['val']['negative']
        val_ratio = stats['val']['positive'] / val_total if val_total > 0 else 0
        writer.writerow(['val', stats['val']['positive'], stats['val']['negative'], 
                        val_total, f"{val_ratio:.4f}", len(val_targets)])
        
        # Test
        test_total = stats['test']['positive'] + stats['test']['negative']
        test_ratio = stats['test']['positive'] / test_total if test_total > 0 else 0
        writer.writerow(['test', stats['test']['positive'], stats['test']['negative'], 
                        test_total, f"{test_ratio:.4f}", len(test_targets)])
    
    print(f"✓ Özet kaydedildi: {summary_path.resolve()}")


def update_data_yaml(output_dir):
    """
    data/data.yaml dosyasını kontrol et ve güncelle.
    
    Args:
        output_dir: Dataset klasörü
    """
    yaml_path = Path(output_dir).parent / 'data.yaml'
    
    # Beklenen içerik
    expected_content = {
        'path': str(Path(output_dir).resolve()),
        'train': 'train',
        'val': 'val',
        'test': 'test',
        'names': {
            0: 'negative',
            1: 'positive'
        },
        'nc': 2
    }
    
    # Dosya varsa oku
    if yaml_path.exists():
        with open(yaml_path, 'r', encoding='utf-8') as f:
            current_content = yaml.safe_load(f)
        
        # Path'i güncelle (mutlak yol olmalı)
        if current_content:
            current_content['path'] = str(Path(output_dir).resolve())
            
            # Kaydet
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(current_content, f, default_flow_style=False, sort_keys=False)
            
            print(f"✓ data.yaml güncellendi: {yaml_path.resolve()}")
        else:
            # Boşsa yeni içerik yaz
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(expected_content, f, default_flow_style=False, sort_keys=False)
            
            print(f"✓ data.yaml oluşturuldu: {yaml_path.resolve()}")
    else:
        # Yoksa oluştur
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(expected_content, f, default_flow_style=False, sort_keys=False)
        
        print(f"✓ data.yaml oluşturuldu: {yaml_path.resolve()}")


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(
        description='Stratified split (yıldız bazlı) ve dataset oluşturma',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--index', default='index.csv',
                       help='Index CSV dosyası')
    parser.add_argument('--out', default='data/plots',
                       help='Çıktı klasörü')
    parser.add_argument('--train', type=float, default=0.7,
                       help='Train oranı')
    parser.add_argument('--val', type=float, default=0.15,
                       help='Validation oranı')
    parser.add_argument('--test', type=float, default=0.15,
                       help='Test oranı')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Oranları kontrol et
    total_ratio = args.train + args.val + args.test
    if abs(total_ratio - 1.0) > 0.01:
        print(f"HATA: Train + val + test = {total_ratio:.2f} (1.0 olmalı)")
        sys.exit(1)
    
    print("=" * 60)
    print("STRATIFIED SPLIT & DATASET OLUŞTURMA")
    print("=" * 60)
    print(f"Index dosyası: {args.index}")
    print(f"Çıktı klasörü: {args.out}")
    print(f"Split oranları: Train={args.train}, Val={args.val}, Test={args.test}")
    print(f"Random seed: {args.seed}")
    print("=" * 60)
    print()
    
    # 1. Index'i yükle
    index_data = load_index(args.index)
    
    # 2. Target'a göre grupla
    print(f"\nGörseller target'a göre gruplanıyor...")
    target_groups = group_by_target(index_data)
    print(f"✓ {len(target_groups)} unique target bulundu")
    
    # Label dağılımını göster
    positive_targets = sum(1 for images in target_groups.values() if images[0]['label'] == 1)
    negative_targets = len(target_groups) - positive_targets
    print(f"  Positive (transit var): {positive_targets} target")
    print(f"  Negative (transit yok): {negative_targets} target")
    
    # 3. Stratified split
    print(f"\nStratified split yapılıyor (yıldız bazlı)...")
    train_targets, val_targets, test_targets = stratified_split_targets(
        target_groups,
        args.train,
        args.val,
        args.test,
        args.seed
    )
    
    # 4. Görselleri kopyala
    print(f"\nGörseller kopyalanıyor...")
    stats = copy_images(target_groups, train_targets, val_targets, test_targets, args.out)
    
    # 5. İstatistikleri yazdır
    print_statistics(stats)
    
    # 6. Özeti kaydet
    save_summary(stats, args.out, train_targets, val_targets, test_targets)
    
    # 7. data.yaml'ı güncelle
    update_data_yaml(args.out)
    
    # 8. Final mesajı
    print("\n" + "=" * 60)
    print("✓ DATASET OLUŞTURMA TAMAMLANDI")
    print("=" * 60)
    print(f"Dataset klasörü: {Path(args.out).resolve()}")
    print(f"Özet rapor: evaluation_results/split_summary.csv")
    print(f"YAML dosyası: {(Path(args.out).parent / 'data.yaml').resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
