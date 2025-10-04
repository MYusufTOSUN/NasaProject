#!/usr/bin/env python3
"""
Tek hedef pipeline: indir/temizle/BLS/faz-katla/grafik/metric
"""

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from lightkurve import search_lightcurve
try:
    from astropy.timeseries import BoxLeastSquares
except ImportError:
    from astropy.stats import BoxLeastSquares
from astropy import units as u
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def retry_with_backoff(func, max_retries=3, initial_delay=2):
    """
    Retry fonksiyonu exponential backoff ile.
    
    Args:
        func: Çalıştırılacak fonksiyon
        max_retries: Maksimum deneme sayısı
        initial_delay: İlk bekleme süresi (saniye)
    
    Returns:
        Fonksiyon sonucu veya None
    """
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed: {e}")
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print(f"All {max_retries} attempts failed")
                raise
    return None


def load_lightcurve(target, mission):
    """
    MAST'tan light curve verisini indir, birleştir, NaN/outlier temizle, flatten et.
    
    Args:
        target: Hedef yıldız adı
        mission: Görev adı (Kepler, TESS, K2)
    
    Returns:
        Temizlenmiş light curve objesi veya None
    """
    print(f"[1/6] Downloading light curve for {target} from {mission}...")
    
    def download_and_process():
        # MAST araması
        search_result = search_lightcurve(target, mission=mission)
        
        if len(search_result) == 0:
            raise ValueError(f"No data found for {target}")
        
        print(f"Found {len(search_result)} light curve(s)")
        
        # Tüm sonuçları indir ve birleştir
        lc_collection = search_result.download_all()
        
        if lc_collection is None or len(lc_collection) == 0:
            raise ValueError("Failed to download light curves")
        
        # Birleştir (stitch)
        if len(lc_collection) > 1:
            print(f"Stitching {len(lc_collection)} light curves...")
            lc = lc_collection.stitch()
        else:
            lc = lc_collection[0]
        
        print(f"Downloaded {len(lc)} data points")
        
        # NaN değerleri kaldır
        lc = lc.remove_nans()
        print(f"After removing NaNs: {len(lc)} points")
        
        # Outlier'ları kaldır (5-sigma clipping)
        lc = lc.remove_outliers(sigma=5)
        print(f"After removing outliers: {len(lc)} points")
        
        # Flatten (trend removal)
        lc = lc.flatten(window_length=401)
        print(f"Flattened light curve")
        
        # Normalize
        lc = lc.normalize()
        print(f"Normalized light curve")
        
        return lc
    
    try:
        # Retry mekanizması ile indir
        lc = retry_with_backoff(download_and_process, max_retries=3, initial_delay=2)
        return lc
    except Exception as e:
        print(f"ERROR: Failed to load light curve after retries: {e}")
        return None


def run_bls(time, flux, min_period=0.5, max_period=20, oversample=5):
    """
    Astropy BoxLeastSquares ile en iyi periyodu bul.
    
    Args:
        time: Zaman dizisi (numpy array)
        flux: Akı dizisi (numpy array)
        min_period: Minimum period (gün)
        max_period: Maksimum period (gün)
        oversample: Oversampling faktörü
    
    Returns:
        dict: BLS sonuçları (period, t0, depth, duration, snr, odd_even_depth_ratio)
    """
    print(f"[2/6] Running Box Least Squares (period range: {min_period}-{max_period} days)...")
    
    try:
        # BLS modelini oluştur
        bls = BoxLeastSquares(time * u.day, flux)
        
        # Period grid oluştur
        # Frequency spacing based on oversample factor
        baseline = time.max() - time.min()
        df = oversample / baseline
        fmin = 1.0 / max_period
        fmax = 1.0 / min_period
        nf = int((fmax - fmin) / df)
        
        print(f"Searching {nf} periods with oversample factor {oversample}...")
        
        # Duration grid oluştur (min period'un %80'ine kadar)
        max_duration = min(min_period * 0.8, 0.5)  # En fazla 0.5 gün
        durations = np.linspace(0.02, max_duration, 10) * u.day
        
        # Periodogram hesapla
        periodogram = bls.autopower(
            durations,
            minimum_period=min_period,
            maximum_period=max_period,
            frequency_factor=oversample
        )
        
        # En iyi period
        best_period = periodogram.period[np.argmax(periodogram.power)]
        best_power = np.max(periodogram.power)
        
        # Best-fit parametreleri
        best_params = bls.compute_stats(
            periodogram.period[np.argmax(periodogram.power)],
            periodogram.duration[np.argmax(periodogram.power)],
            periodogram.transit_time[np.argmax(periodogram.power)]
        )
        
        # Transit depth hesapla
        depth = best_params['depth'][0] if hasattr(best_params['depth'], '__len__') else best_params['depth']
        depth_percent = abs(depth * 100)  # Yüzde olarak
        
        # Duration
        duration = best_params['duration'][0] if hasattr(best_params['duration'], '__len__') else best_params['duration']
        duration_hours = duration * 24  # Saat cinsinden
        
        # Transit time (t0)
        t0 = best_params['transit_time'][0] if hasattr(best_params['transit_time'], '__len__') else best_params['transit_time']
        
        # SNR hesapla
        snr = best_params.get('snr', 0)
        if snr == 0:
            # Manuel SNR hesapla
            snr = best_power / np.std(periodogram.power)
        
        # Odd-even depth ratio hesapla
        # Tek ve çift transitlerde depth farkını kontrol et
        phase = ((time - t0) / best_period.value) % 1.0
        in_transit = np.abs(phase - 0.5) < (duration / best_period.value / 2)
        
        # Tek ve çift transitlerde flux ortalaması
        transit_times = time[in_transit]
        transit_flux = flux[in_transit]
        
        if len(transit_times) > 0:
            transit_numbers = np.floor((transit_times - t0) / best_period.value)
            odd_mask = (transit_numbers % 2) == 1
            even_mask = (transit_numbers % 2) == 0
            
            if np.sum(odd_mask) > 0 and np.sum(even_mask) > 0:
                odd_depth = 1 - np.mean(transit_flux[odd_mask])
                even_depth = 1 - np.mean(transit_flux[even_mask])
                odd_even_ratio = odd_depth / even_depth if even_depth > 0 else 1.0
            else:
                odd_even_ratio = 1.0
        else:
            odd_even_ratio = 1.0
        
        results = {
            'period': float(best_period.value),
            't0': float(t0),
            'depth': float(depth_percent),
            'duration': float(duration_hours),
            'snr': float(snr),
            'power': float(best_power),
            'odd_even_depth_ratio': float(odd_even_ratio),
            'periodogram': periodogram  # Grafik için
        }
        
        print(f"Best period: {results['period']:.4f} days")
        print(f"Transit depth: {results['depth']:.3f}%")
        print(f"Transit duration: {results['duration']:.2f} hours")
        print(f"SNR: {results['snr']:.2f}")
        print(f"Odd/Even ratio: {results['odd_even_depth_ratio']:.3f}")
        
        return results
        
    except Exception as e:
        print(f"ERROR: BLS failed: {e}")
        return None


def phase_fold(time, flux, period, t0):
    """
    Faz katlama uygula.
    
    Args:
        time: Zaman dizisi
        flux: Akı dizisi
        period: Period (gün)
        t0: Transit zamanı
    
    Returns:
        tuple: (phase, flux) sıralı
    """
    print(f"[3/6] Phase folding with period {period:.4f} days...")
    
    # Faz hesapla
    phase = ((time - t0) / period) % 1.0
    
    # [-0.5, 0.5] aralığına kaydır (transit 0'da olsun)
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    
    # Sırala
    sort_idx = np.argsort(phase)
    phase_sorted = phase[sort_idx]
    flux_sorted = flux[sort_idx]
    
    print(f"Phase folded {len(phase)} data points")
    
    return phase_sorted, flux_sorted


def plot_overview(time, flux, bls_results, target, mission, output_path):
    """
    Overview grafik: ham data + periodogram.
    
    Args:
        time: Zaman dizisi
        flux: Akı dizisi
        bls_results: BLS sonuçları
        target: Hedef adı
        mission: Görev adı
        output_path: Çıktı dosya yolu
    """
    print(f"[4/6] Creating overview plot...")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Üst panel: Ham light curve
    ax1 = axes[0]
    ax1.plot(time, flux, 'k.', markersize=1, alpha=0.5)
    ax1.set_xlabel('Time [days]', fontsize=12)
    ax1.set_ylabel('Normalized Flux', fontsize=12)
    ax1.set_title(f'{target} ({mission}) - Raw Light Curve', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Alt panel: BLS periodogram
    ax2 = axes[1]
    periodogram = bls_results['periodogram']
    ax2.plot(periodogram.period, periodogram.power, 'b-', linewidth=0.5)
    ax2.axvline(bls_results['period'], color='r', linestyle='--', linewidth=2, 
                label=f"Best Period: {bls_results['period']:.4f} d")
    ax2.set_xlabel('Period [days]', fontsize=12)
    ax2.set_ylabel('BLS Power', fontsize=12)
    ax2.set_title('Box Least Squares Periodogram', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Metrikleri ekle (text box)
    metrics_text = (
        f"Period: {bls_results['period']:.4f} d\n"
        f"Depth: {bls_results['depth']:.3f}%\n"
        f"Duration: {bls_results['duration']:.2f} h\n"
        f"SNR: {bls_results['snr']:.2f}\n"
        f"Odd/Even: {bls_results['odd_even_depth_ratio']:.3f}"
    )
    ax2.text(0.98, 0.97, metrics_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Overview plot saved: {output_path}")


def plot_phasefold(phase, flux, bls_results, target, mission, output_path):
    """
    Phase-folded light curve grafik.
    
    Args:
        phase: Faz dizisi
        flux: Akı dizisi
        bls_results: BLS sonuçları
        target: Hedef adı
        mission: Görev adı
        output_path: Çıktı dosya yolu
    """
    print(f"[5/6] Creating phase-folded plot...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot
    ax.plot(phase, flux, 'b.', markersize=2, alpha=0.6, label='Data')
    
    # Binned data (daha temiz görünüm için)
    try:
        from scipy.stats import binned_statistic
        bin_means, bin_edges, _ = binned_statistic(phase, flux, statistic='mean', bins=100)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.plot(bin_centers, bin_means, 'r-', linewidth=2, alpha=0.8, label='Binned (100 bins)')
    except ImportError:
        pass
    
    # Transit bölgesini vurgula
    duration_phase = bls_results['duration'] / 24 / bls_results['period']
    ax.axvspan(-duration_phase/2, duration_phase/2, alpha=0.2, color='red', 
               label=f'Transit Duration: {bls_results["duration"]:.2f} h')
    
    ax.set_xlabel('Phase', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalized Flux', fontsize=14, fontweight='bold')
    ax.set_title(f'{target} ({mission}) - Phase-Folded Light Curve\nPeriod: {bls_results["period"]:.4f} days', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 0.5)
    
    # Metrikleri ekle
    metrics_text = (
        f"Depth: {bls_results['depth']:.3f}%\n"
        f"Duration: {bls_results['duration']:.2f} h\n"
        f"SNR: {bls_results['snr']:.2f}\n"
        f"Odd/Even: {bls_results['odd_even_depth_ratio']:.3f}\n"
        f"Data Points: {len(flux)}"
    )
    ax.text(0.02, 0.03, metrics_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Phase-folded plot saved: {output_path}")


def save_metrics(metrics, output_path):
    """
    Metrikleri JSON olarak kaydet.
    
    Args:
        metrics: Metrik dictionary
        output_path: Çıktı dosya yolu
    """
    print(f"[6/6] Saving metrics to {output_path}...")
    
    # Periodogram objesini kaldır (JSON'a yazılamaz)
    metrics_to_save = {k: v for k, v in metrics.items() if k != 'periodogram'}
    
    with open(output_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    
    print(f"Metrics saved successfully")


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(
        description='Single target exoplanet detection pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--target', required=True, 
                       help='Target name (e.g., "Kepler-10", "TOI 700")')
    parser.add_argument('--mission', required=True, 
                       help='Mission name (Kepler, TESS, K2)')
    parser.add_argument('--out', default='figs', 
                       help='Output directory for plots')
    parser.add_argument('--min_period', type=float, default=0.5, 
                       help='Minimum period in days')
    parser.add_argument('--max_period', type=float, default=20, 
                       help='Maximum period in days')
    parser.add_argument('--oversample', type=int, default=5, 
                       help='Oversampling factor for BLS')
    
    args = parser.parse_args()
    
    # Pathlib kullanarak klasörleri oluştur
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = Path('data')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Dosya adı için güvenli format
    safe_target = args.target.replace(' ', '_').replace('-', '_')
    safe_mission = args.mission.replace(' ', '_')
    
    print("=" * 60)
    print(f"NASA Exoplanet Detection Pipeline")
    print(f"Target: {args.target}")
    print(f"Mission: {args.mission}")
    print(f"Period range: {args.min_period} - {args.max_period} days")
    print(f"Oversample factor: {args.oversample}")
    print("=" * 60)
    
    try:
        # 1. Light curve indir ve temizle
        lc = load_lightcurve(args.target, args.mission)
        if lc is None:
            print("\nERROR: Failed to load light curve")
            sys.exit(1)
        
        # 2. BLS çalıştır
        bls_results = run_bls(
            lc.time.value, 
            lc.flux.value,
            min_period=args.min_period,
            max_period=args.max_period,
            oversample=args.oversample
        )
        
        if bls_results is None:
            print("\nERROR: BLS analysis failed")
            sys.exit(1)
        
        # 3. Phase fold
        phase, flux = phase_fold(
            lc.time.value,
            lc.flux.value,
            bls_results['period'],
            bls_results['t0']
        )
        
        # 4. Overview grafiği oluştur
        overview_path = output_dir / f"{safe_target}_{safe_mission}_overview.png"
        plot_overview(
            lc.time.value,
            lc.flux.value,
            bls_results,
            args.target,
            args.mission,
            overview_path
        )
        
        # 5. Phase-folded grafiği oluştur
        phasefold_path = output_dir / f"{safe_target}_{safe_mission}_phasefold.png"
        plot_phasefold(
            phase,
            flux,
            bls_results,
            args.target,
            args.mission,
            phasefold_path
        )
        
        # 6. Metrikleri kaydet
        metrics_path = data_dir / f"{safe_target}_{safe_mission}_metrics.json"
        
        # Ek metrikler ekle
        bls_results['target'] = args.target
        bls_results['mission'] = args.mission
        bls_results['data_points'] = len(lc)
        bls_results['time_span_days'] = float(lc.time.value.max() - lc.time.value.min())
        
        save_metrics(bls_results, metrics_path)
        
        # Sonuçları özetle
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Target: {args.target}")
        print(f"Mission: {args.mission}")
        print(f"Period: {bls_results['period']:.4f} days")
        print(f"Transit depth: {bls_results['depth']:.3f}%")
        print(f"Transit duration: {bls_results['duration']:.2f} hours")
        print(f"SNR: {bls_results['snr']:.2f}")
        print(f"Odd/Even depth ratio: {bls_results['odd_even_depth_ratio']:.3f}")
        print(f"Data points: {bls_results['data_points']}")
        print(f"\nOutput files:")
        print(f"  Overview: {overview_path}")
        print(f"  Phase-fold: {phasefold_path}")
        print(f"  Metrics: {metrics_path}")
        print("=" * 60)
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
        
    except Exception as e:
        print(f"\nERROR: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
