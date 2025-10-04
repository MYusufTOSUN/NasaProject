#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Metadata-first grafik üretici - MAST fallback ile
Metadata'dan period/t0/duration varsa kullanır, yoksa BLS ile bulur.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI olmadan çalışma
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    import lightkurve as lk
    from astropy.timeseries import BoxLeastSquares
except ImportError as e:
    print(f"HATA: Gerekli kütüphaneler yüklü değil: {e}")
    print("pip install lightkurve astropy komutunu çalıştırın.")
    exit(1)


def download_lightcurve(target, mission='auto'):
    """
    MAST'tan ışık eğrisi indir
    """
    try:
        search_result = lk.search_lightcurve(target, mission=mission if mission != 'auto' else None)
        if len(search_result) == 0:
            return None, None
        
        # İlk sonucu indir
        lc_collection = search_result.download_all()
        if lc_collection is None or len(lc_collection) == 0:
            return None, None
        
        # Stitch (birleştir)
        lc = lc_collection.stitch()
        return lc, search_result[0].mission[0] if hasattr(search_result[0], 'mission') else 'Unknown'
    except Exception as e:
        print(f"  İndirme hatası: {e}")
        return None, None


def clean_lightcurve(lc):
    """
    Işık eğrisini temizle ve detrend yap
    """
    try:
        # NaN'ları kaldır
        lc = lc.remove_nans()
        
        # Outlier'ları temizle (sigma clipping)
        lc = lc.remove_outliers(sigma=5)
        
        # Flatten (detrend)
        lc_flat = lc.flatten(window_length=2001)
        
        return lc_flat
    except Exception as e:
        print(f"  Temizleme hatası: {e}")
        return None


def find_period_bls(lc, period_min=0.5, period_max=20.0):
    """
    Box Least Squares ile period bul
    """
    try:
        # BLS modeli oluştur
        bls = BoxLeastSquares(lc.time.value, lc.flux.value)
        
        # Period gridini oluştur
        periods = np.linspace(period_min, period_max, 5000)
        
        # BLS periodogram hesapla
        results = bls.power(periods, duration=np.linspace(0.01, 0.2, 10))
        
        # En iyi period
        best_period = results.period[np.argmax(results.power)]
        
        # Transit parametrelerini bul
        stats = bls.compute_stats(best_period, duration=np.linspace(0.01, 0.2, 10))
        
        return {
            'period': float(best_period),
            't0': float(stats['transit_time'][np.argmax(stats['depth'])]),
            'duration': float(stats['duration'][np.argmax(stats['depth'])])
        }
    except Exception as e:
        print(f"  BLS hatası: {e}")
        return None


def create_phase_plot(lc, period, t0, duration, output_path, target_name):
    """
    Faz-katlanmış görsel oluştur
    """
    try:
        # Faz katlama
        lc_folded = lc.fold(period=period, epoch_time=t0)
        
        # Binning
        lc_binned = lc_folded.bin(bins=100)
        
        # Görsel oluştur
        fig, ax = plt.subplots(figsize=(8, 6), dpi=96)
        
        # Scatter plot (tüm noktalar)
        ax.scatter(lc_folded.phase.value, lc_folded.flux.value, 
                  s=1, alpha=0.3, c='gray', label='Data')
        
        # Binned line plot
        ax.plot(lc_binned.phase.value, lc_binned.flux.value, 
               'r-', linewidth=2, label='Binned')
        
        # Transit bölgesini vurgula
        transit_phase = duration / period / 2
        ax.axvline(-transit_phase, color='blue', linestyle='--', alpha=0.5)
        ax.axvline(transit_phase, color='blue', linestyle='--', alpha=0.5)
        ax.axvline(0.5 - transit_phase, color='blue', linestyle='--', alpha=0.5)
        ax.axvline(0.5 + transit_phase, color='blue', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Phase', fontsize=12)
        ax.set_ylabel('Normalized Flux', fontsize=12)
        ax.set_title(f'{target_name} - P={period:.4f}d', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=96, bbox_inches='tight')
        plt.close()
        
        return True
    except Exception as e:
        print(f"  Görsel oluşturma hatası: {e}")
        return False


def create_yolo_label(duration, period, label_path):
    """
    YOLO formatında etiket dosyası oluştur
    Transit bölgelerini işaretle
    """
    try:
        # Transit fazı genişliği
        transit_phase_width = (duration / period) * 2  # Güvenlik için 2x
        
        # İki transit bölgesi: faz 0.0 ve 0.5 civarı
        boxes = []
        
        # Merkez transit (phase=0.0)
        x_center = 0.5  # Normalized image coordinates
        y_center = 0.5
        width = min(transit_phase_width, 0.3)  # Max %30
        height = 0.4  # Dikey %40
        boxes.append(f"0 {x_center} {y_center} {width} {height}")
        
        # İkincil transit (phase=0.5) - varsa
        if duration > 0:
            boxes.append(f"0 {x_center} {y_center} {width} {height}")
        
        # Dosyaya yaz
        with open(label_path, 'w') as f:
            for box in boxes[:1]:  # Sadece merkez kutu
                f.write(box + '\n')
        
        return True
    except Exception as e:
        print(f"  Etiket oluşturma hatası: {e}")
        return False


def process_target(row, output_images_dir, output_labels_dir, use_bls_fallback=True):
    """
    Tek bir hedefi işle
    """
    target = row.get('target', '')
    mission = row.get('mission', 'auto')
    
    # Metadata'dan period bilgilerini al
    period = row.get('period', None)
    t0 = row.get('t0', None)
    duration = row.get('duration', None)
    
    print(f"\n🎯 İşleniyor: {target} ({mission})")
    
    # Period bilgileri kontrolü
    has_period_info = pd.notna(period) and pd.notna(t0) and pd.notna(duration)
    
    if has_period_info:
        print(f"  ✓ Metadata'dan period bilgileri alındı: P={period:.4f}d")
    else:
        print(f"  ⚠ Metadata'da period bilgisi yok, MAST + BLS kullanılacak...")
    
    # 1. Işık eğrisini indir
    lc, detected_mission = download_lightcurve(target, mission)
    if lc is None:
        print(f"  ✗ Işık eğrisi indirilemedi, atlanıyor.")
        return False
    
    print(f"  ✓ Işık eğrisi indirildi: {len(lc)} veri noktası")
    
    # 2. Temizle
    lc_clean = clean_lightcurve(lc)
    if lc_clean is None:
        print(f"  ✗ Temizleme başarısız, atlanıyor.")
        return False
    
    print(f"  ✓ Temizlendi: {len(lc_clean)} veri noktası")
    
    # 3. Period bilgisi yoksa BLS ile bul
    if not has_period_info:
        if not use_bls_fallback:
            print(f"  ✗ BLS fallback kapalı, atlanıyor.")
            return False
        
        print(f"  ⏳ BLS ile period bulunuyor (bu işlem uzun sürebilir)...")
        bls_result = find_period_bls(lc_clean)
        if bls_result is None:
            print(f"  ✗ BLS başarısız, atlanıyor.")
            return False
        
        period = bls_result['period']
        t0 = bls_result['t0']
        duration = bls_result['duration']
        print(f"  ✓ BLS ile bulundu: P={period:.4f}d, t0={t0:.2f}, dur={duration:.4f}d")
    
    # 4. Faz-katlanmış görsel oluştur
    image_filename = f"{target.replace(' ', '_')}_{detected_mission}_phase.png"
    image_path = os.path.join(output_images_dir, image_filename)
    
    success = create_phase_plot(lc_clean, period, t0, duration, image_path, target)
    if not success:
        print(f"  ✗ Görsel oluşturulamadı, atlanıyor.")
        return False
    
    print(f"  ✓ Görsel kaydedildi: {image_filename}")
    
    # 5. YOLO etiketi oluştur
    label_filename = f"{target.replace(' ', '_')}_{detected_mission}_phase.txt"
    label_path = os.path.join(output_labels_dir, label_filename)
    
    create_yolo_label(duration, period, label_path)
    print(f"  ✓ Etiket kaydedildi: {label_filename}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Metadata-first grafik üretici')
    parser.add_argument('--metadata', type=str, required=True, 
                       help='Metadata CSV dosyası yolu (örn: data/metadata/metadata1500.csv)')
    parser.add_argument('--limit', type=int, default=None, 
                       help='İşlenecek maksimum hedef sayısı (test için)')
    parser.add_argument('--no-bls', action='store_true',
                       help='BLS fallback\'i devre dışı bırak (sadece metadata\'lı hedefleri işle)')
    
    args = parser.parse_args()
    
    # Çıktı klasörleri
    output_images_dir = 'graphs/images'
    output_labels_dir = 'graphs/labels'
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # Metadata'yı oku
    print(f"📂 Metadata okunuyor: {args.metadata}")
    try:
        df = pd.read_csv(args.metadata)
        print(f"✓ {len(df)} hedef bulundu")
    except Exception as e:
        print(f"✗ HATA: Metadata okunamadı: {e}")
        return
    
    # Limit uygula
    if args.limit:
        df = df.head(args.limit)
        print(f"⚠ İlk {args.limit} hedef işlenecek")
    
    # İstatistikler
    success_count = 0
    fail_count = 0
    
    # Her hedefi işle
    print("\n" + "="*60)
    print("🚀 Grafik üretimi başlıyor...")
    print("="*60)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="İşleniyor"):
        success = process_target(row, output_images_dir, output_labels_dir, 
                                use_bls_fallback=not args.no_bls)
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # Özet rapor
    print("\n" + "="*60)
    print("📊 İŞLEM TAMAMLANDI")
    print("="*60)
    print(f"✓ Başarılı: {success_count}")
    print(f"✗ Başarısız: {fail_count}")
    print(f"📁 Görseller: {output_images_dir}/")
    print(f"🏷️  Etiketler: {output_labels_dir}/")
    print("="*60)


if __name__ == '__main__':
    main()

