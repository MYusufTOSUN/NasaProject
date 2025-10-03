# build_dataset.py
# Çok hedefli dataset üretimi (Kişi 1)
# Çıktılar: dataset/X_*.npy, y_*.npy, index_*.csv (+ logs)
import os, sys, csv, json, math, time, argparse
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares
from astroquery.mast import Conf
from numpy.ma import getdata as _ma_get

Conf.timeout = 60

# ----------------- yardımcı -----------------
def _np(a, dtype=float): return np.asarray(_ma_get(a), dtype=dtype).ravel()
def _ensure_dirs(*ds):   [os.makedirs(d, exist_ok=True) for d in ds]

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

# ----------------- veri çekme -----------------
def fetch_kepler_llc(target, max_files=2):
    s = lk.search_lightcurve(target, mission="Kepler")
    if len(s) == 0: raise RuntimeError("Kepler LC bulunamadı")
    fn = _getcol(s, "productFilename"); mask = None
    if fn is not None:
        fn_low = np.array([str(x).lower() for x in fn])
        mask   = np.array([("llc" in x) for x in fn_low])  # long cadence
    if mask is None or mask.sum() == 0:
        desc = _getcol(s, "description")
        if desc is not None:
            dlow = np.array([str(x).lower() for x in desc])
            mask = np.array([("long" in x) for x in dlow])
    s = _sort_sr(s, "year")
    sel = s[mask] if (mask is not None and mask.sum() > 0) else s
    sel = sel[:max_files]
    lcc = sel.download_all()
    return lcc.stitch()

def fetch_tess_spoc_lc(target, max_files=2):
    s = lk.search_lightcurve(target, mission="TESS", author="SPOC")
    if len(s) == 0: raise RuntimeError("TESS SPOC LC bulunamadı")
    fn = _getcol(s, "productFilename")
    if fn is not None:
        fn_low = np.array([str(x).lower() for x in fn])
        mask   = np.array([("_lc.fits" in x) and ("ffic" not in x) for x in fn_low])
        if mask.sum() > 0: s = s[mask]
    s = _sort_sr(s, "year")[:max_files]
    lcc = s.download_all()
    return lcc.stitch()

def fetch_with_retry(kind, target, max_files, retries=2, delay=4):
    last = None
    for k in range(retries+1):
        try:
            return (fetch_kepler_llc if kind=="kepler" else fetch_tess_spoc_lc)(target, max_files)
        except Exception as e:
            last = e
            if k < retries:
                print(f"[!] {target}: indirme hatası: {e} → {delay}s sonra tekrar…")
                time.sleep(delay)
    raise last

# ----------------- işleme -----------------
def clean_flatten(lc, sigma=5.0, window_length=401):
    return (lc.remove_nans().normalize()
              .remove_outliers(sigma=sigma)
              .flatten(window_length=window_length)
              .remove_nans())

def bls_coarse_to_fine(t, y, pmin=0.5, pmax=20.0):
    t = _np(t); y = _np(y)
    bls = BoxLeastSquares(t, y)
    durations = np.linspace(0.05, 0.3, 25)
    p_coarse  = np.geomspace(pmin, pmax, 2000)
    res_c     = bls.power(p_coarse, durations)
    top_idx   = np.argpartition(res_c.power, -3)[-3:]
    cand      = sorted(res_c.period[top_idx])
    best=None
    for pc in cand:
        p1, p2  = pc*0.98, pc*1.02
        p_fine  = np.linspace(p1, p2, 6000)
        res_f   = bls.power(p_fine, durations)
        k       = np.argmax(res_f.power)
        cb      = dict(period=float(res_f.period[k]),
                       duration=float(res_f.duration[k]),
                       t0=float(res_f.transit_time[k]),
                       power=float(res_f.power[k]))
        if (best is None) or (cb["power"] > best["power"]):
            best = cb
    return best

def phase_bin_fixed(phase, flux, bins=200, robust=True):
    edges   = np.linspace(-0.5, 0.5, bins+1)
    centers = 0.5*(edges[:-1]+edges[1:])
    which   = np.digitize(phase, edges)-1
    yb      = np.full(bins, np.nan)
    for i in range(bins):
        m = which==i
        if not np.any(m): continue
        yb[i] = np.nanmedian(flux[m]) if robust else np.nanmean(flux[m])
    return centers, yb

def compute_metrics(phase, flux, period, duration):
    phi_dur  = duration/period
    in_mask  = np.abs(phase) <= 0.6*phi_dur
    oot_mask = np.abs(phase) >= 3.0*phi_dur
    baseline = float(np.nanmedian(flux[oot_mask]))
    in_med   = float(np.nanmedian(flux[in_mask]))
    depth    = baseline - in_med
    depth_ppm= depth*1e6
    rms_oot  = float(np.nanstd(flux[oot_mask]))
    n_in     = int(np.nansum(in_mask))
    snr      = float(depth/(rms_oot/np.sqrt(max(n_in,1)))) if n_in>0 and rms_oot>0 else float("nan")
    return dict(phi_dur=float(phi_dur), depth_ppm=depth_ppm, snr=snr, n_in=n_in)

# ----------------- tek hedef örnek üret -----------------
def build_example(target, mission, max_files=2, pmin=0.5, pmax=20.0, bins=200,
                  min_snr=7.0, min_depth_ppm=50.0):
    kind = mission.lower()
    lc   = fetch_with_retry("kepler" if kind=="kepler" else "tess",
                            target=target, max_files=max_files)
    lc_f = clean_flatten(lc)
    t    = lc_f.time.value; y = lc_f.flux.value
    m    = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]

    best = bls_coarse_to_fine(t, y, pmin=pmin, pmax=pmax)
    folded = lk.LightCurve(time=t, flux=y).fold(period=best["period"], epoch_time=best["t0"])
    phase  = _np(folded.time.value); flux = _np(folded.flux.value)

    # binned vektör ve metrikler
    xb, yb = phase_bin_fixed(phase, flux, bins=bins, robust=True)
    mets   = compute_metrics(phase, flux, best["period"], best["duration"])

    # kalite filtresi
    if not (np.isfinite(mets["snr"]) and mets["snr"] >= min_snr and mets["depth_ppm"] >= min_depth_ppm):
        raise RuntimeError(f"Kalite düşük: SNR={mets['snr']:.1f}, depth={mets['depth_ppm']:.0f} ppm")

    # normalizasyon: out-of-transit median ~0 olacak şekilde
    yb = yb - np.nanmedian(yb)
    # eksikler 0 ile doldur
    yb = np.nan_to_num(yb, nan=0.0)

    meta = dict(target=target, mission=mission,
                period_days=best["period"], duration_days=best["duration"],
                t0=best["t0"], **mets)
    return yb.astype(np.float32), meta

# ----------------- stratified split -----------------
def stratified_split(labels, seed=42, ratios=(0.7, 0.15, 0.15)):
    y = np.asarray(labels)
    idx_pos = np.where(y==1)[0]; idx_neg = np.where(y==0)[0]
    rng = np.random.RandomState(seed)
    rng.shuffle(idx_pos); rng.shuffle(idx_neg)

    def take(idx):
        n = len(idx); n_tr = int(n*ratios[0]); n_va = int(n*ratios[1])
        tr = idx[:n_tr]; va = idx[n_tr:n_tr+n_va]; te = idx[n_tr+n_va:]
        return tr, va, te
    tr_p, va_p, te_p = take(idx_pos)
    tr_n, va_n, te_n = take(idx_neg)

    tr = np.concatenate([tr_p, tr_n]); va = np.concatenate([va_p, va_n]); te = np.concatenate([te_p, te_n])
    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    return tr, va, te

# ----------------- ana -----------------
def main():
    ap = argparse.ArgumentParser("Dataset Builder (Kişi 1)")
    ap.add_argument("--targets", type=str, default="targets.csv")
    ap.add_argument("--out", type=str, default="dataset")
    ap.add_argument("--max-files", type=int, default=2)
    ap.add_argument("--bins", type=int, default=200)
    ap.add_argument("--pmin", type=float, default=0.5)
    ap.add_argument("--pmax", type=float, default=20.0)
    ap.add_argument("--min-snr", type=float, default=7.0)
    ap.add_argument("--min-depth-ppm", type=float, default=50.0)
    args = ap.parse_args()

    _ensure_dirs(args.out, "logs")
    log_path = os.path.join("logs", "dataset_build.log")
    ok, fail = 0, 0
    X, metas, y = [], [], []

    with open(args.targets, newline="", encoding="utf-8") as f, open(log_path, "w", encoding="utf-8") as lf:
        reader = csv.DictReader(f)
        for row in reader:
            target  = row["target"].strip()
            mission = row["mission"].strip()
            label   = int(row["label"])
            try:
                vec, meta = build_example(target, mission,
                                          max_files=args.max_files,
                                          pmin=args.pmin, pmax=args.pmax,
                                          bins=args.bins,
                                          min_snr=args.min_snr,
                                          min_depth_ppm=args.min_depth_ppm)
                X.append(vec); y.append(label); metas.append({**meta, "label": label})
                ok += 1
                print(f"[✓] {target} ({mission}) eklendi | SNR≈{meta['snr']:.1f}, depth≈{meta['depth_ppm']:.0f} ppm")
            except Exception as e:
                fail += 1
                msg = f"[x] {target} ({mission}) SKIP: {e}"
                print(msg); lf.write(msg + "\n")

    if ok == 0:
        print("Hiç örnek üretilmedi! targets.csv / ağ / filtre eşiklerini kontrol et.")
        sys.exit(1)

    X = np.stack(X, axis=0)         # (N, bins)
    y = np.asarray(y, dtype=np.int64)

    # stratified split
    tr, va, te = stratified_split(y, seed=42, ratios=(0.7, 0.15, 0.15))

    # kayıtlar
    def save_split(name, idx):
        np.save(os.path.join(args.out, f"X_{name}.npy"), X[idx])
        np.save(os.path.join(args.out, f"y_{name}.npy"), y[idx])
        # index CSV
        with open(os.path.join(args.out, f"index_{name}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            cols = ["i","target","mission","period_days","duration_days","t0","snr","depth_ppm","label","split"]
            w.writerow(cols)
            for k in idx:
                m = metas[k]
                w.writerow([k, m["target"], m["mission"], m["period_days"], m["duration_days"],
                            m["t0"], m["snr"], m["depth_ppm"], m["label"], name])

    save_split("train", tr)
    save_split("val",   va)
    save_split("test",  te)

    # özet
    print(f"\n[ÖZET] ok={ok}, fail={fail}")
    print(f"[SHAPE] X={X.shape} | train={len(tr)}, val={len(va)}, test={len(te)}")
    print(f"[OUT] {args.out} klasörüne kaydedildi. Log: {log_path}")

if __name__ == "__main__":
    main()
