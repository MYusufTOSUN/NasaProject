# nasa_bls_pipeline.py
# Kepler/TESS -> indir -> temizle -> adaptif bin -> BLS (coarse→fine) -> faz-katla/binle
# -> metrikler (depth/SNR) -> odd/even kontrol -> ephemeris -> görseller/CSV/JSON
# Python 3.9+ | pip: lightkurve astropy numpy matplotlib astroquery

import os, sys, json, math, time, argparse
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares
from astroquery.mast import Conf
from numpy.ma import getdata as _ma_get

Conf.timeout = 60  # ağ zaman aşımı (s)

# --------------------------- yardımcılar ---------------------------

def _np(a, dtype=float):
    """Masked array'leri güvenle düz ndarray'e çevir."""
    return np.asarray(_ma_get(a), dtype=dtype).ravel()

def _colnames(sr):
    try: return list(sr.table.colnames)
    except Exception: return []

def _getcol(sr, name):
    try:
        if name in sr.table.colnames:
            return sr.table[name]
    except Exception:
        pass
    return None

def _sort_sr(sr, key):
    try:
        idx = np.argsort(sr.table[key])
        return sr[idx]
    except Exception:
        return sr

def _ensure_dirs(*dirs):
    for d in dirs: os.makedirs(d, exist_ok=True)

# --------------------------- veri çekme ---------------------------

def fetch_kepler_llc(target: str, max_files: int = 2):
    """Kepler long-cadence (llc) light curve dosyalarını seç ve birleştir."""
    s = lk.search_lightcurve(target, mission="Kepler")
    if len(s) == 0:
        raise RuntimeError("Kepler ışık eğrisi bulunamadı.")

    fn = _getcol(s, "productFilename")
    mask = None
    if fn is not None:
        fn_low = np.array([str(x).lower() for x in fn])
        mask = np.array([("llc" in x) for x in fn_low])  # long-lc

    if mask is None or mask.sum() == 0:
        desc = _getcol(s, "description")
        if desc is not None:
            dlow = np.array([str(x).lower() for x in desc])
            mask = np.array([("long" in x) for x in dlow])

    s_sorted = _sort_sr(s, "year")
    sel = s_sorted[mask] if (mask is not None and mask.sum() > 0) else s_sorted
    sel = sel[:max_files]
    print(f"[i] Kepler seçildi: {len(sel)} dosya")
    lcc = sel.download_all()
    return lcc.stitch()

def fetch_tess_spoc_lc(target: str, max_files: int = 2):
    """TESS SPOC light curve (lc.fits) dosyalarını seç ve birleştir."""
    s = lk.search_lightcurve(target, mission="TESS", author="SPOC")
    if len(s) == 0:
        raise RuntimeError("TESS SPOC ışık eğrisi bulunamadı.")
    fn = _getcol(s, "productFilename")
    if fn is not None:
        fn_low = np.array([str(x).lower() for x in fn])
        mask = np.array([("_lc.fits" in x) and ("ffic" not in x) for x in fn_low])
        if mask.sum() > 0:
            s = s[mask]
    s = _sort_sr(s, "year")[:max_files]
    print(f"[i] TESS seçildi: {len(s)} dosya")
    lcc = s.download_all()
    return lcc.stitch()

def fetch_with_retry(func, retries=2, delay=5, **kwargs):
    """Ağ hatalarında birkaç kez yeniden dene."""
    last = None
    for k in range(retries+1):
        try:
            return func(**kwargs)
        except Exception as e:
            last = e
            if k < retries:
                print(f"[!] İndirme hatası: {e} → {delay}s sonra tekrar...")
                time.sleep(delay)
    raise last

# --------------------------- işleme ---------------------------

def clean_flatten(lc: lk.LightCurve, sigma=5.0, window_length=401) -> lk.LightCurve:
    """NaN temizle, normalize et, outlier kırp, trendi çıkar."""
    return (lc.remove_nans()
              .normalize()
              .remove_outliers(sigma=sigma)
              .flatten(window_length=window_length)
              .remove_nans())

def adaptive_bin(lc_flat: lk.LightCurve, max_points=200_000, min_bin_days=0.02):
    """Nokta sayısını kontrol altında tutmak için zaman bin'leme."""
    t = _np(lc_flat.time.value)
    if len(t) <= max_points:
        return lc_flat, 0.0
    span = float(np.nanmax(t) - np.nanmin(t))
    binsize = max(span / max_points, min_bin_days)  # gün
    print(f"[i] Adaptif bin: ~{binsize:.5f} gün hedef ~{max_points} nokta")
    return lc_flat.bin(time_bin_size=binsize).remove_nans(), binsize

def bls_coarse_to_fine(t, y, pmin=0.5, pmax=20.0, top_peaks=3):
    """Önce kaba periyot taraması, sonra en güçlü birkaç aday etrafında ince tarama."""
    t = _np(t); y = _np(y)
    durations = np.linspace(0.05, 0.3, 25)  # 1.2–7.2 saat
    bls = BoxLeastSquares(t, y)

    p_coarse = np.geomspace(pmin, pmax, 2000)
    res_c = bls.power(p_coarse, durations)
    top_idx = np.argpartition(res_c.power, -top_peaks)[-top_peaks:]
    cand = sorted(res_c.period[top_idx])

    best = None; best_res = None
    for pc in cand:
        p1, p2 = pc * 0.98, pc * 1.02
        p_fine = np.linspace(p1, p2, 6000)
        res_f = bls.power(p_fine, durations)
        k = np.argmax(res_f.power)
        cb = dict(period=float(res_f.period[k]),
                  duration=float(res_f.duration[k]),
                  t0=float(res_f.transit_time[k]),
                  power=float(res_f.power[k]))
        if (best is None) or (cb["power"] > best["power"]):
            best, best_res = cb, res_f
    return best, best_res

def phase_bin_array(phase, flux, bins=180, robust=True):
    edges   = np.linspace(-0.5, 0.5, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    which   = np.digitize(phase, edges) - 1
    yb      = np.full(bins, np.nan)
    for i in range(bins):
        m = which == i
        if not np.any(m):
            continue
        yb[i] = np.nanmedian(flux[m]) if robust else np.nanmean(flux[m])
    # Boşları lineer interpolasyonla doldur:
    ok = np.isfinite(yb)
    if ok.sum() >= 2:
        yb[~ok] = np.interp(centers[~ok], centers[ok], yb[ok])
    else:
        yb[~ok] = 0.0
    return centers, yb


def compute_metrics_from_fold(phase, flux, period, duration):
    """Depth (ppm), SNR, faz genişliği vb. temel metrikler."""
    phi_dur = duration / period
    in_mask  = np.abs(phase) <= 0.6 * phi_dur      # transit çekirdeği
    oot_mask = np.abs(phase) >= 3.0 * phi_dur      # güvenli OOT
    baseline = float(np.nanmedian(flux[oot_mask]))
    in_med   = float(np.nanmedian(flux[in_mask]))
    depth    = baseline - in_med
    depth_ppm = depth * 1e6
    rms_oot  = float(np.nanstd(flux[oot_mask]))
    n_in     = int(np.nansum(in_mask))
    snr      = float(depth / (rms_oot / math.sqrt(max(n_in,1)))) if n_in>0 and rms_oot>0 else float("nan")
    return dict(phi_dur=float(phi_dur), baseline=baseline, depth=depth,
                depth_ppm=depth_ppm, rms_oot=rms_oot, n_in=n_in, snr=snr)

def odd_even_depth(phase, flux, period, duration, k_sigma=0.6, min_pts=5):
    """Tek/çift transit derinliği (boş maske/az nokta için güvenli).
       Gerekirse 2*P ile fold edip tekrar dener."""
    def _depth_at(ph, center, phi_dur):
        m = np.abs(ph - center) <= k_sigma * phi_dur
        if np.count_nonzero(m) < min_pts:
            return np.nan
        return -float(np.nanmedian(flux[m]))  # normalize ~ 0, transit negatif; pozitif derinlik dön
    phi_dur = duration / period

    # 1) Fazı [0,1) aralığına taşı
    ph = ((phase + 0.5) % 1.0)

    d_odd  = _depth_at(ph, 0.0, phi_dur)
    d_even = _depth_at(ph, 0.5, phi_dur)

    # 2) Gerekirse 2P ile tekrar dene (bazı katlamalarda örnekler kayık kalabiliyor)
    if not np.isfinite(d_odd) or not np.isfinite(d_even):
        ph2 = ((phase + 0.5) % 2.0)
        d_odd2  = _depth_at(ph2, 0.0, 2*phi_dur)
        d_even2 = _depth_at(ph2, 1.0, 2*phi_dur)
        d_odd  = d_odd if np.isfinite(d_odd) else d_odd2
        d_even = d_even if np.isfinite(d_even) else d_even2

    diff_ppm = (d_odd - d_even)*1e6 if (np.isfinite(d_odd) and np.isfinite(d_even)) else np.nan
    return dict(depth_odd=d_odd, depth_even=d_even, diff_ppm=diff_ppm)

def ephemeris(t0, period, n=5):
    """Bir sonraki birkaç transit zamanı (BKJD/TBJD)."""
    return [float(t0 + k*period) for k in range(n)]

# --------------------------- çizimler/kayıt ---------------------------

def plot_overview(lc_raw, lc_flat, target, mission, outdir="figs"):
    _ensure_dirs(outdir)
    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(2,1,1)
    lc_raw.normalize().plot(ax=ax1)
    ax1.set_title(f"{target} ({mission}) – Raw (normalized)")
    ax2 = fig.add_subplot(2,1,2)
    lc_flat.plot(ax=ax2)
    ax2.set_title("Flattened")
    fig.tight_layout()
    path = os.path.join(outdir, f"{target.replace(' ','_')}_{mission}_overview.png")
    fig.savefig(path, dpi=140)
    print("[✓] Kaydedildi:", path)

def plot_phasefold(phase, flux, title, filename, outdir="figs", binned=None):
    _ensure_dirs(outdir)
    plt.figure(figsize=(9,5))
    plt.scatter(phase, flux, s=3, alpha=0.35, label="points")
    if binned is not None:
        xb, yb = binned
        plt.plot(xb, yb, lw=2, label="phase-binned")
        plt.legend()
    plt.xlabel("Phase (days)"); plt.ylabel("Flux (norm)")
    plt.title(title)
    plt.tight_layout()
    path = os.path.join(outdir, filename)
    plt.savefig(path, dpi=150)
    print("[✓] Kaydedildi:", path)

def save_csv_work(t, y, target, mission, outdir="data"):
    _ensure_dirs(outdir)
    t_np, y_np = _np(t), _np(y)
    arr = np.column_stack([t_np, y_np])
    path = os.path.join(outdir, f"{target.replace(' ','_')}_{mission}_work.csv")
    np.savetxt(path, arr, delimiter=",", header="time,flux", comments="", fmt="%.10f")
    print("[✓] CSV:", path)

def save_csv_folded_binned(xb, yb, target, mission, outdir="data"):
    _ensure_dirs(outdir)
    path = os.path.join(outdir, f"{target.replace(' ','_')}_{mission}_folded_binned.csv")
    np.savetxt(path, np.column_stack([xb, yb]), delimiter=",",
               header="phase,flux_binned", comments="", fmt="%.8f")
    print("[✓] CSV:", path)

def save_metrics_json(meta, target, mission, outdir="data"):
    _ensure_dirs(outdir)
    path = os.path.join(outdir, f"{target.replace(' ','_')}_{mission}_metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("[✓] Metrics JSON:", path)

# --------------------------- ana akış ---------------------------

def run_pipeline(target="Kepler-10", mission="Kepler", max_files=2,
                 pmin=0.5, pmax=20.0, bins=180):
    print(f"→ Hedef: {target} | Misyon: {mission} | max_files={max_files}")

    if mission.lower() == "kepler":
        lc = fetch_with_retry(fetch_kepler_llc, target=target, max_files=max_files)
    else:
        lc = fetch_with_retry(fetch_tess_spoc_lc, target=target, max_files=max_files)

    print(f"[i] İndirilen nokta: {len(lc.time)}")
    lc_flat = clean_flatten(lc)
    print(f"[i] Flatten sonrası: {len(lc_flat.time)}")

    lc_work, binsize = adaptive_bin(lc_flat)
    t = lc_work.time.value; y = lc_work.flux.value
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    print(f"[i] BLS giriş noktası: {len(t)}")

    best, _ = bls_coarse_to_fine(t, y, pmin=pmin, pmax=pmax)
    print(f"[BLS] P={best['period']:.6f} gün | dur={best['duration']:.4f} gün (~{24*best['duration']:.1f} saat) | power={best['power']:.3e}")
    print(f"[BLS] t0={best['t0']:.6f}")

    # Görseller
    plot_overview(lc, lc_flat, target, mission)

    folded = lk.LightCurve(time=t, flux=y).fold(period=best["period"], epoch_time=best["t0"])
    phase = _np(folded.time.value); flux = _np(folded.flux.value)

    xb, yb = phase_bin_array(phase, flux, bins=bins, robust=True)
    plot_phasefold(phase, flux,
                   f"Phase-folded @ P={best['period']:.5f} d",
                   f"{target.replace(' ','_')}_{mission}_phasefold.png")
    plot_phasefold(phase, flux,
                   f"Phase-folded @ P={best['period']:.5f} d (binned)",
                   f"{target.replace(' ','_')}_{mission}_phasefold_binned.png",
                   binned=(xb, yb))

    # Kayıtlar
    save_csv_work(t, y, target, mission)
    save_csv_folded_binned(xb, yb, target, mission)

    # Metrikler
    metrics = compute_metrics_from_fold(phase, flux, best["period"], best["duration"])
    oe      = odd_even_depth(phase, flux, best["period"], best["duration"])
    eph     = ephemeris(best["t0"], best["period"], n=5)
    meta = {
        "target": target, "mission": mission,
        "period_days": best["period"],
        "duration_days": best["duration"],
        "t0": best["t0"],
        "binsize_days": float(binsize),
        "metrics": metrics,
        "odd_even": oe,
        "ephemeris_next_BKJD_TBJD": eph
    }
    print(f"[METRICS] depth ≈ {metrics['depth_ppm']:.0f} ppm | SNR ≈ {metrics['snr']:.1f} | N_in={metrics['n_in']}")
    print(f"[ODD/EVEN] Δdepth ≈ {oe['diff_ppm']:.0f} ppm (odd={oe['depth_odd']:.3e}, even={oe['depth_even']:.3e})")

    save_metrics_json(meta, target, mission)
    print("[✓] İş akışı tamam.")

# --------------------------- CLI ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Kepler/TESS transit arama (BLS) mini pipeline")
    p.add_argument("target", nargs="?", default="Kepler-10")
    p.add_argument("mission", nargs="?", default="Kepler", choices=["Kepler","TESS","kepler","tess"])
    p.add_argument("--max-files", type=int, default=2, help="İndirilecek maksimum LC dosyası")
    p.add_argument("--pmin", type=float, default=0.5, help="BLS min periyot (gün)")
    p.add_argument("--pmax", type=float, default=20.0, help="BLS max periyot (gün)")
    p.add_argument("--bins", type=int, default=180, help="Phase bin sayısı")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(target=args.target,
                 mission=args.mission,
                 max_files=args.max_files,
                 pmin=args.pmin, pmax=args.pmax,
                 bins=args.bins)
