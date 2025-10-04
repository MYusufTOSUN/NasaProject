#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tek-yıldız hızlı akış CLI
MAST'tan LC indir → BLS (opsiyonel) → hızlı görsel üret
"""

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    import lightkurve as lk
    from astropy.timeseries import BoxLeastSquares
    import numpy as np
except ImportError as e:
    print(f"HATA: Gerekli kütüphaneler yüklü değil: {e}")
    exit(1)


def download_lightcurve(target, mission='auto'):
    """MAST'tan ışık eğrisi indir"""
    print(f"📡 MAST'tan indiriliyor: {target} ({mission})")
    try:
        search_result = lk.search_lightcurve(target, mission=mission if mission != 'auto' else None)
        if len(search_result) == 0:
            print("  ✗ Sonuç bulunamadı")
            return None
        
        print(f"  ✓ {len(search_result)} sonuç bulundu")
        lc_collection = search_result.download_all()
        if lc_collection is None or len(lc_collection) == 0:
            return None
        
        lc = lc_collection.stitch()
        print(f"  ✓ {len(lc)} veri noktası indirildi")
        return lc
    except Exception as e:
        print(f"  ✗ Hata: {e}")
        return None


def clean_lightcurve(lc):
    """Temizle ve detrend yap"""
    print("🧹 Temizleniyor...")
    try:
        lc = lc.remove_nans().remove_outliers(sigma=5)
        lc_flat = lc.flatten(window_length=2001)
        print(f"  ✓ Temizlendi: {len(lc_flat)} nokta")
        return lc_flat
    except Exception as e:
        print(f"  ✗ Hata: {e}")
        return None


def find_period_bls(lc, period_min=0.5, period_max=20.0):
    """BLS ile period bul"""
    print(f"🔍 BLS ile period bulunuyor ({period_min}-{period_max} gün)...")
    try:
        bls = BoxLeastSquares(lc.time.value, lc.flux.value)
        periods = np.linspace(period_min, period_max, 5000)
        results = bls.power(periods, duration=np.linspace(0.01, 0.2, 10))
        
        best_period = results.period[np.argmax(results.power)]
        stats = bls.compute_stats(best_period, duration=np.linspace(0.01, 0.2, 10))
        
        params = {
            'period': float(best_period),
            't0': float(stats['transit_time'][np.argmax(stats['depth'])]),
            'duration': float(stats['duration'][np.argmax(stats['depth'])])
        }
        
        print(f"  ✓ P={params['period']:.4f}d, t0={params['t0']:.2f}, dur={params['duration']:.4f}d")
        return params
    except Exception as e:
        print(f"  ✗ Hata: {e}")
        return None


def plot_phase_folded(lc, period, t0, output_path, target_name):
    """Faz-katlanmış görsel oluştur"""
    print(f"📊 Faz-katlanmış görsel oluşturuluyor...")
    try:
        lc_folded = lc.fold(period=period, epoch_time=t0)
        lc_binned = lc_folded.bin(bins=100)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(lc_folded.phase.value, lc_folded.flux.value, s=1, alpha=0.3, c='gray')
        ax.plot(lc_binned.phase.value, lc_binned.flux.value, 'r-', linewidth=2)
        ax.set_xlabel('Phase')
        ax.set_ylabel('Normalized Flux')
        ax.set_title(f'{target_name} - Period={period:.4f}d')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Kaydedildi: {output_path}")
        return True
    except Exception as e:
        print(f"  ✗ Hata: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Tek yıldız hızlı analiz')
    parser.add_argument('--target', type=str, required=True, help='Hedef adı (örn: Kepler-10)')
    parser.add_argument('--mission', type=str, default='auto', 
                       choices=['auto', 'Kepler', 'TESS', 'K2'],
                       help='Misyon seçimi')
    parser.add_argument('--period', type=float, default=None, help='Period (gün) - BLS atlanır')
    parser.add_argument('--t0', type=float, default=None, help='Transit zamanı')
    parser.add_argument('--duration', type=float, default=None, help='Transit süresi (gün)')
    parser.add_argument('--output', type=str, default='temp_phase_plot.png', 
                       help='Çıktı dosyası adı')
    
    args = parser.parse_args()
    
    print("="*60)
    print(f"🪐 Tek Yıldız Hızlı Analiz: {args.target}")
    print("="*60)
    
    # 1. İndir
    lc = download_lightcurve(args.target, args.mission)
    if lc is None:
        print("\n✗ İndirme başarısız, sonlandırılıyor.")
        return
    
    # 2. Temizle
    lc_clean = clean_lightcurve(lc)
    if lc_clean is None:
        print("\n✗ Temizleme başarısız, sonlandırılıyor.")
        return
    
    # 3. Period parametrelerini belirle
    if args.period and args.t0 and args.duration:
        print(f"\n✓ Manuel parametreler kullanılıyor")
        period = args.period
        t0 = args.t0
        duration = args.duration
    else:
        print(f"\n⏳ Period bilinmiyor, BLS ile bulunacak...")
        bls_result = find_period_bls(lc_clean)
        if bls_result is None:
            print("\n✗ BLS başarısız, sonlandırılıyor.")
            return
        period = bls_result['period']
        t0 = bls_result['t0']
        duration = bls_result['duration']
    
    # 4. Görsel oluştur
    print()
    success = plot_phase_folded(lc_clean, period, t0, args.output, args.target)
    
    if success:
        print("\n" + "="*60)
        print("✓ İşlem tamamlandı!")
        print(f"  Çıktı: {args.output}")
        print("="*60)
    else:
        print("\n✗ Görsel oluşturulamadı")


if __name__ == '__main__':
    main()

