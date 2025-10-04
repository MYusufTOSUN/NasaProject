#!/usr/bin/env python3
"""
Toplu grafik üretimi scripti - YOLO dataset hazırlığı için
targets.csv'den hedefleri okuyup her biri için overview ve phase-fold grafikleri üretir.
Opsiyonel olarak transit bbox etiketleri de oluşturabilir.
"""

import argparse
import csv
import sys
import warnings
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Step 2'deki fonksiyonları import et
from importlib.util import spec_from_file_location, module_from_spec

# 01_download_clean_bls_fast.py modülünü yükle
spec = spec_from_file_location("pipeline", "01_download_clean_bls_fast.py")
pipeline = module_from_spec(spec)
spec.loader.exec_module(pipeline)

warnings.filterwarnings('ignore')


def load_targets(csv_path):
    """
    targets.csv dosyasını oku.
    
    Args:
        csv_path: CSV dosya yolu
        
    Returns:
        list: Her satır için (target, mission, label) tuple'larının listesi
    """
    targets = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            target = row['target'].strip()
            mission = row['mission'].strip()
            label = int(row['label'])
            targets.append((target, mission, label))
    
    return targets


def estimate_transit_bbox(phase, flux, duration_phase):
    """
    Phase-folded grafikte transit bölgesini tahmin et ve YOLO bbox formatında döndür.
    
    Args:
        phase: Faz dizisi (-0.5 ile 0.5 arası)
        flux: Akı dizisi
        duration_phase: Transit süresi (faz birimi)
        
    Returns:
        tuple: (x_center, y_center, width, height) normalize edilmiş 0-1 arası
    """
    # Transit merkezde olduğu için x_center = 0.5
    # Phase aralığı [-0.5, 0.5] olduğu için normalize etmeliyiz
    
    # Transit bölgesindeki verileri bul
    in_transit = np.abs(phase) < duration_phase / 2
    
    if np.sum(in_transit) < 5:
        # Yeterli veri yoksa varsayılan değerler kullan
        x_center = 0.5
        y_center = 0.5
        width = min(duration_phase * 2, 0.2)  # Duration'ın 2 katı, max 0.2
        height = 0.1  # Varsayılan yükseklik
    else:
        # Transit bölgesindeki flux değerlerini analiz et
        transit_flux = flux[in_transit]
        transit_phase = phase[in_transit]
        
        # X merkezi: transit'in merkezi (phase = 0, normalize 0-1'de 0.5)
        x_center = 0.5
        
        # Y merkezi: transit'teki ortalama flux
        # Flux aralığını [0, 1] olarak normalize et
        flux_min = np.min(flux)
        flux_max = np.max(flux)
        flux_range = flux_max - flux_min
        
        if flux_range > 0:
            # Transit'in ortalama flux'unu normalize et
            mean_transit_flux = np.mean(transit_flux)
            y_center = (mean_transit_flux - flux_min) / flux_range
            
            # Yükseklik: transit depth'e göre (yüzde olarak)
            # Transit depth = 1 - mean_transit_flux (çünkü normalize edilmiş)
            depth = abs(1.0 - mean_transit_flux)
            height = min(depth * 3, 0.3)  # Depth'in 3 katı, max 0.3
        else:
            y_center = 0.5
            height = 0.1
        
        # Genişlik: transit duration'a göre
        # Phase range [-0.5, 0.5] -> [0, 1] mapping
        phase_extent = np.max(transit_phase) - np.min(transit_phase)
        width = min(phase_extent * 2, 0.2)  # 2 katı marj ile, max 0.2
    
    # Sınırları kontrol et
    x_center = np.clip(x_center, 0.0, 1.0)
    y_center = np.clip(y_center, 0.0, 1.0)
    width = np.clip(width, 0.05, 0.5)
    height = np.clip(height, 0.05, 0.5)
    
    return x_center, y_center, width, height


def save_yolo_label(label_path, class_id, bbox):
    """
    YOLO formatında etiket dosyası kaydet.
    
    Args:
        label_path: Etiket dosya yolu
        class_id: Sınıf ID (0: transit yok, 1: transit var)
        bbox: (x_center, y_center, width, height) tuple
    """
    label_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(label_path, 'w') as f:
        x_center, y_center, width, height = bbox
        # YOLO formatı: class x_center y_center width height
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def process_target(target, mission, label, output_dir, make_labels=False, 
                   min_period=0.5, max_period=20, oversample=5):
    """
    Tek bir hedef için grafikleri ve etiketleri üret.
    
    Args:
        target: Hedef adı
        mission: Görev adı
        label: Etiket (0 veya 1)
        output_dir: Çıktı klasörü
        make_labels: YOLO etiketleri oluşturulsun mu
        min_period: Minimum period (gün)
        max_period: Maksimum period (gün)
        oversample: Oversampling faktörü
        
    Returns:
        tuple: (success, error_message)
    """
    try:
        # Güvenli dosya adı oluştur
        safe_target = target.replace(' ', '_').replace('-', '_')
        safe_mission = mission.replace(' ', '_')
        
        # 1. Light curve indir
        lc = pipeline.load_lightcurve(target, mission)
        if lc is None:
            return False, "Failed to load light curve"
        
        # 2. BLS analizi
        bls_results = pipeline.run_bls(
            lc.time.value,
            lc.flux.value,
            min_period=min_period,
            max_period=max_period,
            oversample=oversample
        )
        
        if bls_results is None:
            return False, "BLS analysis failed"
        
        # 3. Phase fold
        phase, flux = pipeline.phase_fold(
            lc.time.value,
            lc.flux.value,
            bls_results['period'],
            bls_results['t0']
        )
        
        # 4. Overview grafiği
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        overview_path = images_dir / f"{safe_target}_{safe_mission}_overview.png"
        pipeline.plot_overview(
            lc.time.value,
            lc.flux.value,
            bls_results,
            target,
            mission,
            overview_path
        )
        
        # 5. Phase-fold grafiği
        phasefold_path = images_dir / f"{safe_target}_{safe_mission}_phase.png"
        pipeline.plot_phasefold(
            phase,
            flux,
            bls_results,
            target,
            mission,
            phasefold_path
        )
        
        # 6. YOLO etiketleri (opsiyonel)
        if make_labels:
            labels_dir = output_dir / "labels"
            labels_dir.mkdir(parents=True, exist_ok=True)
            
            # Overview etiketi
            overview_label_path = labels_dir / f"{safe_target}_{safe_mission}_overview.txt"
            # Overview grafiğinde genellikle bbox yok (ham data)
            # Eğer label=1 ise, genel bir bbox koyabiliriz (opsiyonel)
            # Şimdilik boş bırakıyoruz
            
            # Phase-fold etiketi
            phasefold_label_path = labels_dir / f"{safe_target}_{safe_mission}_phase.txt"
            
            if label == 1:
                # Transit var, bbox tahmin et
                duration_phase = bls_results['duration'] / 24 / bls_results['period']
                bbox = estimate_transit_bbox(phase, flux, duration_phase)
                
                # Class ID: 0 = transit (tek sınıf detection için)
                # Eğer çok sınıflı olsaydı, farklı class ID'ler kullanabilirdik
                save_yolo_label(phasefold_label_path, 0, bbox)
            else:
                # Transit yok, boş etiket dosyası
                phasefold_label_path.touch()
        
        return True, None
        
    except Exception as e:
        return False, str(e)


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(
        description='Toplu grafik üretimi - YOLO dataset hazırlığı',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--targets', default='targets.csv',
                       help='Hedef listesi CSV dosyası')
    parser.add_argument('--out', default='graphs',
                       help='Çıktı klasörü')
    parser.add_argument('--make_labels', action='store_true',
                       help='YOLO formatında bbox etiketleri oluştur')
    parser.add_argument('--min_period', type=float, default=0.5,
                       help='Minimum period (gün)')
    parser.add_argument('--max_period', type=float, default=20,
                       help='Maksimum period (gün)')
    parser.add_argument('--oversample', type=int, default=5,
                       help='BLS oversampling faktörü')
    parser.add_argument('--skip_errors', action='store_true',
                       help='Hata olan hedefleri atla ve devam et')
    
    args = parser.parse_args()
    
    # Çıktı klasörünü oluştur
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Hedefleri yükle
    print(f"Hedefler yükleniyor: {args.targets}")
    try:
        targets = load_targets(args.targets)
    except FileNotFoundError:
        print(f"HATA: {args.targets} dosyası bulunamadı!")
        sys.exit(1)
    except Exception as e:
        print(f"HATA: Hedefler yüklenirken hata oluştu: {e}")
        sys.exit(1)
    
    print(f"Toplam {len(targets)} hedef bulundu")
    print(f"Çıktı klasörü: {output_dir}")
    print(f"Etiketler oluşturulacak: {'Evet' if args.make_labels else 'Hayır'}")
    print("=" * 60)
    
    # İstatistikler
    success_count = 0
    error_count = 0
    error_list = []
    
    # Her hedef için işlem yap
    for target, mission, label in tqdm(targets, desc="Grafikler üretiliyor", unit="hedef"):
        success, error_msg = process_target(
            target, mission, label, output_dir,
            make_labels=args.make_labels,
            min_period=args.min_period,
            max_period=args.max_period,
            oversample=args.oversample
        )
        
        if success:
            success_count += 1
        else:
            error_count += 1
            error_list.append((target, mission, error_msg))
            
            if not args.skip_errors:
                print(f"\nHATA: {target} ({mission}) işlenirken hata oluştu: {error_msg}")
                print("Devam etmek için --skip_errors kullanın")
                sys.exit(1)
    
    # Sonuçları özetle
    print("\n" + "=" * 60)
    print("İŞLEM TAMAMLANDI")
    print("=" * 60)
    print(f"Başarılı: {success_count}/{len(targets)}")
    print(f"Hatalı: {error_count}/{len(targets)}")
    
    if error_list:
        print("\nHatalı hedefler:")
        for target, mission, error_msg in error_list:
            print(f"  - {target} ({mission}): {error_msg}")
    
    print(f"\nGrafikler: {output_dir / 'images'}")
    if args.make_labels:
        print(f"Etiketler: {output_dir / 'labels'}")
    print("=" * 60)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
