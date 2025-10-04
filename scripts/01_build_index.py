#!/usr/bin/env python3
"""
Ham görsellerden index.csv üreten script.
raw_images klasöründeki PNG/JPG dosyalarını tarayıp targets.csv ile eşleştirir.
"""

import argparse
import csv
import sys
from pathlib import Path
from tqdm import tqdm


def load_targets(csv_path):
    """
    targets.csv dosyasını oku.
    
    Args:
        csv_path: CSV dosya yolu
        
    Returns:
        dict: {target_name: (mission, label)} şeklinde dictionary
    """
    targets_dict = {}
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                target = row['target'].strip()
                mission = row['mission'].strip()
                label = int(row['label'])
                targets_dict[target] = (mission, label)
        
        print(f"✓ {len(targets_dict)} hedef yüklendi: {csv_path}")
        return targets_dict
        
    except FileNotFoundError:
        print(f"HATA: {csv_path} dosyası bulunamadı!")
        sys.exit(1)
    except Exception as e:
        print(f"HATA: targets.csv okurken hata: {e}")
        sys.exit(1)


def normalize_name(name):
    """
    İsmi normalize et (boşlukları ve tireleri kaldır, küçük harfe çevir).
    
    Args:
        name: İsim
        
    Returns:
        str: Normalize edilmiş isim
    """
    return name.lower().replace(' ', '').replace('-', '').replace('_', '')


def find_matching_target(filename, targets_dict):
    """
    Dosya adından hedef ismi bul.
    
    Args:
        filename: Dosya adı
        targets_dict: Hedef dictionary
        
    Returns:
        tuple: (target, mission, label) veya None
    """
    # Dosya adını normalize et
    normalized_filename = normalize_name(filename)
    
    # Her hedef için kontrol et
    for target, (mission, label) in targets_dict.items():
        normalized_target = normalize_name(target)
        
        # Dosya adında hedef ismi geçiyor mu?
        if normalized_target in normalized_filename:
            return target, mission, label
    
    return None


def scan_images(raw_dir, targets_dict):
    """
    raw_dir altındaki PNG/JPG dosyalarını tara ve eşleştir.
    
    Args:
        raw_dir: Ham görsel klasörü
        targets_dict: Hedef dictionary
        
    Returns:
        list: Her görsel için (target, mission, label, image_path, is_binned, is_phase) tuple listesi
    """
    raw_path = Path(raw_dir)
    
    if not raw_path.exists():
        print(f"HATA: {raw_dir} klasörü bulunamadı!")
        sys.exit(1)
    
    # PNG ve JPG dosyalarını bul
    image_extensions = ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(raw_path.rglob(ext))
    
    print(f"✓ {len(all_images)} görsel dosyası bulundu")
    
    # Her görseli işle
    matched_images = []
    unmatched_images = []
    
    for image_path in tqdm(all_images, desc="Görseller eşleştiriliyor", unit="görsel"):
        filename = image_path.name
        
        # Hedef eşleştir
        match_result = find_matching_target(filename, targets_dict)
        
        if match_result:
            target, mission, label = match_result
            
            # Tam yol
            full_path = image_path.resolve()
            
            # is_binned kontrolü
            is_binned = 1 if 'binned' in filename.lower() else 0
            
            # is_phase kontrolü
            is_phase = 1 if 'phase' in filename.lower() else 0
            
            matched_images.append((
                target,
                mission,
                label,
                str(full_path),
                is_binned,
                is_phase
            ))
        else:
            unmatched_images.append(filename)
    
    print(f"✓ {len(matched_images)} görsel eşleştirildi")
    
    if unmatched_images:
        print(f"⚠ {len(unmatched_images)} görsel eşleştirilemedi")
        
        # İlk 10 eşleşmeyen dosyayı göster
        if len(unmatched_images) <= 10:
            for img in unmatched_images:
                print(f"  - {img}")
        else:
            for img in unmatched_images[:10]:
                print(f"  - {img}")
            print(f"  ... ve {len(unmatched_images) - 10} dosya daha")
    
    return matched_images


def save_index(matched_images, output_path):
    """
    Index CSV dosyasını kaydet.
    
    Args:
        matched_images: Eşleştirilmiş görsel listesi
        output_path: Çıktı dosya yolu
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['target', 'mission', 'label', 'image_path', 'is_binned', 'is_phase'])
        
        # Veri satırları
        for row in matched_images:
            writer.writerow(row)
    
    print(f"✓ Index kaydedildi: {output_file.resolve()}")


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(
        description='Ham görsellerden index.csv üret',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--raw_dir', required=True,
                       help='Ham görsel klasörü (örn: scripts/raw_images)')
    parser.add_argument('--targets', default='targets.csv',
                       help='Hedef listesi CSV dosyası')
    parser.add_argument('--out', default='index.csv',
                       help='Çıktı index dosyası')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("HAM GÖRSEL INDEX OLUŞTURUCU")
    print("=" * 60)
    print(f"Ham görsel klasörü: {args.raw_dir}")
    print(f"Hedef dosyası: {args.targets}")
    print(f"Çıktı dosyası: {args.out}")
    print("=" * 60)
    print()
    
    # 1. Hedefleri yükle
    targets_dict = load_targets(args.targets)
    
    # 2. Görselleri tara ve eşleştir
    matched_images = scan_images(args.raw_dir, targets_dict)
    
    if not matched_images:
        print("\nUYARI: Hiç görsel eşleştirilemedi!")
        print("Lütfen dosya adlarının targets.csv'deki hedef isimlerini içerdiğinden emin olun.")
        sys.exit(1)
    
    # 3. Index'i kaydet
    save_index(matched_images, args.out)
    
    # 4. İstatistikler
    print("\n" + "=" * 60)
    print("İSTATİSTİKLER")
    print("=" * 60)
    
    # Label dağılımı
    label_counts = {}
    for _, _, label, _, _, _ in matched_images:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"Toplam görsel: {len(matched_images)}")
    for label, count in sorted(label_counts.items()):
        label_name = "Transit Var" if label == 1 else "Transit Yok"
        print(f"  Label {label} ({label_name}): {count}")
    
    # Phase/overview dağılımı
    phase_count = sum(1 for _, _, _, _, _, is_phase in matched_images if is_phase == 1)
    overview_count = len(matched_images) - phase_count
    print(f"\nGörsel tipi:")
    print(f"  Phase-fold: {phase_count}")
    print(f"  Overview: {overview_count}")
    
    # Binned dağılımı
    binned_count = sum(1 for _, _, _, _, is_binned, _ in matched_images if is_binned == 1)
    print(f"\nBinned: {binned_count}")
    print(f"Non-binned: {len(matched_images) - binned_count}")
    
    # Görev dağılımı
    mission_counts = {}
    for _, mission, _, _, _, _ in matched_images:
        mission_counts[mission] = mission_counts.get(mission, 0) + 1
    
    print(f"\nGörev dağılımı:")
    for mission, count in sorted(mission_counts.items()):
        print(f"  {mission}: {count}")
    
    print("=" * 60)
    print("✓ İŞLEM TAMAMLANDI")
    print("=" * 60)


if __name__ == "__main__":
    main()
