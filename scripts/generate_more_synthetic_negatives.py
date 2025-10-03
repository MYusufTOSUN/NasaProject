"""
Sentetik Negatif Veri Üreteci
Her split (train, val, test) için gerçekçi negatif ışık eğrisi görselleri üretir.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# Çıkış klasörleri
PLOTS_ROOT = Path("data/plots")
OUT_DIRS = {
    "train": PLOTS_ROOT / "train" / "negative",
    "val": PLOTS_ROOT / "val" / "negative",
    "test": PLOTS_ROOT / "test" / "negative",
}


def ensure_dirs():
    """Çıkış klasörlerini oluştur"""
    for p in OUT_DIRS.values():
        p.mkdir(parents=True, exist_ok=True)


def find_next_id(output_dir: Path, prefix: str = "synthetic_neg_") -> int:
    """
    Çıkış klasöründe synthetic_neg_* dosyalarının son ID'sini bulup +1 döndürür.
    """
    existing = list(output_dir.glob(f"{prefix}*.png"))
    if not existing:
        return 1
    
    max_id = 0
    for f in existing:
        name = f.stem
        try:
            num_part = name.replace(prefix, "").split("_")[0]
            num = int(num_part)
            if num > max_id:
                max_id = num
        except Exception:
            continue
    return max_id + 1


def generate_synthetic_negative(out_png: Path, title: str, seed: int, scenario: str = None) -> bool:
    """
    Gerçekçi sentetik negatif ışık eğrisi üretir.
    Gezegen sinyali OLMAYAN tipik durumlar:
    - pure_noise: Sadece gürültü
    - stellar_rotation: Yıldız rotasyonu (yavaş varyasyon)
    - stellar_pulsation: Pulsasyon (düzenli ama gezegen değil)
    - eclipsing_binary: Binary yıldız (asimetrik, derin)
    - instrumental_drift: Sistematik trend + gürültü
    - single_artifact: Tek olay (kozmik ışın vs)
    - flare: Yıldız patlaması
    - transit_timing_variation: Transit zamanlama varyasyonu benzeri gürültü
    """
    np.random.seed(seed)
    
    try:
        # Zaman serisi parametreleri
        n_points = np.random.randint(1000, 2500)
        time_span = np.random.uniform(60, 180)
        time = np.linspace(0, time_span, n_points)
        
        # Temel flux seviyesi
        base_flux = 1.0
        
        # Senaryo belirleme
        scenarios = [
            "pure_noise",
            "stellar_rotation",
            "stellar_pulsation",
            "eclipsing_binary",
            "instrumental_drift",
            "single_artifact",
            "flare",
            "transit_timing_variation",
            "multiple_flares",
            "long_term_variability",
        ]
        
        if scenario is None or scenario not in scenarios:
            scenario = np.random.choice(scenarios)
        
        # Senaryo bazlı sinyal üretimi
        if scenario == "pure_noise":
            # Sadece fotonik gürültü
            noise_level = np.random.uniform(0.0005, 0.003)
            flux = base_flux + np.random.normal(0, noise_level, n_points)
        
        elif scenario == "stellar_rotation":
            # Yıldız yüzeyindeki lekeler -> yavaş sinüzoidal
            period = np.random.uniform(5, 40)  # Uzun periyot (gün)
            amplitude = np.random.uniform(0.003, 0.025)
            phase_shift = np.random.uniform(0, 2 * np.pi)
            flux = base_flux + amplitude * np.sin(2 * np.pi * time / period + phase_shift)
            # Gürültü ekle
            noise = np.random.normal(0, 0.0012, n_points)
            flux += noise
        
        elif scenario == "stellar_pulsation":
            # Değişken yıldız (Cepheid, RR Lyrae, Delta Scuti)
            period = np.random.uniform(0.2, 8)  # Kısa periyot
            amplitude = np.random.uniform(0.015, 0.12)
            # Asimetrik pulsasyon (harmonikler)
            flux = base_flux + amplitude * (
                np.sin(2 * np.pi * time / period) + 
                0.3 * np.sin(4 * np.pi * time / period) +
                0.15 * np.sin(6 * np.pi * time / period)
            )
            noise = np.random.normal(0, 0.002, n_points)
            flux += noise
        
        elif scenario == "eclipsing_binary":
            # İki yıldızın birbirini tutması
            period = np.random.uniform(0.8, 15)
            depth_primary = np.random.uniform(0.08, 0.5)  # Çok derin (yıldız boyutu)
            depth_secondary = depth_primary * np.random.uniform(0.2, 0.9)
            width = np.random.uniform(0.04, 0.18)  # Geniş (yıldız boyutu)
            
            flux = np.ones(n_points) * base_flux
            phase = (time % period) / period
            
            # Primary eclipse (daha derin, V veya U şekilli)
            primary_mask = np.abs(phase - 0.0) < width
            if np.any(primary_mask):
                # V şekilli tutulma
                phase_in_eclipse = (phase[primary_mask] - 0.0) / width
                eclipse_shape = 1 - depth_primary * (1 - np.abs(phase_in_eclipse) ** 1.5)
                flux[primary_mask] *= eclipse_shape
            
            # Secondary eclipse (daha sığ)
            secondary_mask = np.abs(phase - 0.5) < width * 0.8
            if np.any(secondary_mask):
                phase_in_eclipse = (phase[secondary_mask] - 0.5) / (width * 0.8)
                eclipse_shape = 1 - depth_secondary * (1 - np.abs(phase_in_eclipse) ** 1.3)
                flux[secondary_mask] *= eclipse_shape
            
            noise = np.random.normal(0, 0.003, n_points)
            flux += noise
        
        elif scenario == "instrumental_drift":
            # Uzay aracı ısınma/soğuma -> polinom trend
            coeffs = [
                base_flux,
                np.random.uniform(-0.08, 0.08),
                np.random.uniform(-0.03, 0.03),
                np.random.uniform(-0.01, 0.01)
            ]
            trend = np.polynomial.polynomial.polyval(time / time[-1], coeffs)
            noise = np.random.normal(0, 0.0025, n_points)
            flux = trend + noise
        
        elif scenario == "single_artifact":
            # Temiz sinyal + tek olay (kozmik ışın, momentum dump)
            flux = base_flux + np.random.normal(0, 0.001, n_points)
            artifact_idx = np.random.randint(100, n_points - 100)
            artifact_width = np.random.randint(1, 8)
            artifact_depth = np.random.uniform(0.03, 0.25)
            artifact_type = np.random.choice(["dip", "spike"])
            if artifact_type == "dip":
                flux[artifact_idx:artifact_idx + artifact_width] -= artifact_depth
            else:
                flux[artifact_idx:artifact_idx + artifact_width] += artifact_depth
        
        elif scenario == "flare":
            # Yıldız patlaması (ani yükseliş, yavaş düşüş)
            flux = base_flux + np.random.normal(0, 0.001, n_points)
            n_flares = np.random.randint(1, 4)
            for _ in range(n_flares):
                flare_idx = np.random.randint(50, n_points - 50)
                flare_amplitude = np.random.uniform(0.02, 0.15)
                flare_rise = np.random.randint(3, 10)
                flare_decay = np.random.randint(10, 40)
                
                # Yükseliş (hızlı)
                rise_end = min(flare_idx + flare_rise, n_points)
                rise_profile = np.linspace(0, 1, rise_end - flare_idx)
                flux[flare_idx:rise_end] += flare_amplitude * rise_profile
                
                # Düşüş (yavaş, eksponansiyel)
                decay_end = min(rise_end + flare_decay, n_points)
                decay_profile = np.exp(-np.linspace(0, 4, decay_end - rise_end))
                flux[rise_end:decay_end] += flare_amplitude * decay_profile
        
        elif scenario == "transit_timing_variation":
            # Düzensiz zaman damgalı mini olaylar (TTV benzeri ama gezegen değil)
            flux = base_flux + np.random.normal(0, 0.0015, n_points)
            n_events = np.random.randint(3, 10)
            event_times = sorted(np.random.uniform(time[0] + 5, time[-1] - 5, n_events))
            for t_event in event_times:
                closest_idx = np.argmin(np.abs(time - t_event))
                event_width = np.random.randint(5, 20)
                event_depth = np.random.uniform(0.01, 0.05)
                start = max(0, closest_idx - event_width // 2)
                end = min(n_points, closest_idx + event_width // 2)
                flux[start:end] -= event_depth * np.random.uniform(0.3, 1.0)
        
        elif scenario == "multiple_flares":
            # Çoklu küçük patlamalar
            flux = base_flux + np.random.normal(0, 0.001, n_points)
            n_flares = np.random.randint(5, 15)
            for _ in range(n_flares):
                flare_idx = np.random.randint(0, n_points - 20)
                flare_amplitude = np.random.uniform(0.005, 0.04)
                flare_width = np.random.randint(3, 15)
                end_idx = min(flare_idx + flare_width, n_points)
                flare_profile = np.exp(-np.linspace(0, 3, end_idx - flare_idx))
                flux[flare_idx:end_idx] += flare_amplitude * flare_profile
        
        else:  # long_term_variability
            # Uzun dönemli düzensiz varyasyon
            n_components = np.random.randint(2, 5)
            flux = base_flux * np.ones(n_points)
            for _ in range(n_components):
                period = np.random.uniform(20, 100)
                amplitude = np.random.uniform(0.005, 0.03)
                phase = np.random.uniform(0, 2 * np.pi)
                flux += amplitude * np.sin(2 * np.pi * time / period + phase)
            noise = np.random.normal(0, 0.002, n_points)
            flux += noise
        
        # Fiziksel olmayan değerleri temizle
        flux = np.clip(flux, 0.4, 1.6)
        
        # Görselleştirme
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(time, flux, 'k.', markersize=1.5, alpha=0.6)
        ax.set_title(f"{title} (Sentetik Negatif - {scenario})", fontsize=11)
        ax.set_xlabel("Zaman (gün)", fontsize=10)
        ax.set_ylabel("Normalize Akı", fontsize=10)
        ax.set_ylim(flux.min() - 0.02, flux.max() + 0.02)
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        
        # Kaydet
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return True
        
    except Exception as e:
        print(f"[HATA] {title} üretilirken hata: {e}")
        return False


def generate_for_split(split_name: str, count: int, start_seed: int = 50000):
    """Belirtilen split için sentetik negatifler üret"""
    output_dir = OUT_DIRS[split_name]
    next_id = find_next_id(output_dir, prefix="synthetic_neg_")
    
    print(f"\n{'='*60}")
    print(f"[{split_name.upper()}] {count} sentetik negatif üretiliyor...")
    print(f"Başlangıç ID: {next_id}")
    print(f"Çıkış klasörü: {output_dir}")
    print(f"{'='*60}\n")
    
    success_count = 0
    failed_count = 0
    
    for i in tqdm(range(count), desc=f"{split_name} üretimi"):
        current_id = next_id + i
        seed = start_seed + current_id
        out_name = f"synthetic_neg_{current_id:05d}.png"
        out_path = output_dir / out_name
        
        title = f"Synthetic_{split_name}_{current_id}"
        
        if generate_synthetic_negative(out_path, title, seed=seed):
            success_count += 1
        else:
            failed_count += 1
    
    print(f"\n[{split_name.upper()}] Tamamlandı!")
    print(f"  ✓ Başarılı: {success_count}")
    if failed_count > 0:
        print(f"  ✗ Başarısız: {failed_count}")
    print(f"  Toplam: {success_count + failed_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Train, Val ve Test setleri için sentetik negatif veri üretir."
    )
    parser.add_argument(
        "--train", 
        type=int, 
        default=50,
        help="Train seti için üretilecek sentetik negatif sayısı (varsayılan: 50)"
    )
    parser.add_argument(
        "--val", 
        type=int, 
        default=40,
        help="Validation seti için üretilecek sentetik negatif sayısı (varsayılan: 40)"
    )
    parser.add_argument(
        "--test", 
        type=int, 
        default=40,
        help="Test seti için üretilecek sentetik negatif sayısı (varsayılan: 40)"
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=50000,
        help="Rastgele seed başlangıç değeri (varsayılan: 50000)"
    )
    
    args = parser.parse_args()
    
    # Klasörleri oluştur
    ensure_dirs()
    
    print("\n" + "="*60)
    print("SENTETİK NEGATİF VERİ ÜRETİCİ")
    print("="*60)
    print(f"Train için: {args.train} adet")
    print(f"Val için  : {args.val} adet")
    print(f"Test için : {args.test} adet")
    print(f"Toplam    : {args.train + args.val + args.test} adet")
    print("="*60)
    
    # Her split için üret
    if args.train > 0:
        generate_for_split("train", args.train, start_seed=args.seed_start)
    
    if args.val > 0:
        generate_for_split("val", args.val, start_seed=args.seed_start + 10000)
    
    if args.test > 0:
        generate_for_split("test", args.test, start_seed=args.seed_start + 20000)
    
    print("\n" + "="*60)
    print("TÜM İŞLEMLER TAMAMLANDI!")
    print("="*60)
    
    # Güncel sayıları göster
    print("\nGüncel veri dağılımı:")
    for split_name, out_dir in OUT_DIRS.items():
        count = len(list(out_dir.glob("*.png")))
        print(f"  {split_name:6s} negative: {count:4d} adet")


if __name__ == "__main__":
    main()

