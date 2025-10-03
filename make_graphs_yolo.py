# make_graphs_yolo.py
# targets.csv'deki her hedef için: indir -> temizle -> BLS -> faz-katla
# Görsel üret: phase-folded (scatter + binned) ve overview (opsiyonel)
# YOLO etiketleri: transit pencereleri için bounding box (sınıf 0: transit)
# Çıktılar: graphs/images/*.png ve graphs/labels/*.txt

import os, sys, csv, math, argparse, time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares
from astroquery.mast import Conf
from numpy.ma import getdata as _ma_get

Conf.timeout = 60


# ----------------- yardımcılar -----------------
def _np(a, dtype=float):
    return np.asarray(_ma_get(a), dtype=dtype).ravel()


def _ensure_dirs(*ds):
    for d in ds:
        os.makedirs(d, exist_ok=True)


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
def fetch_kepler_llc(target: str, max_files: int = 2):
    s = lk.search_lightcurve(target, mission="Kepler")
    if len(s) == 0:
        raise RuntimeError("Kepler ışık eğrisi bulunamadı.")

    fn = _getcol(s, "productFilename")
    mask = None
    if fn is not None:
        fn_low = np.array([str(x).lower() for x in fn])
        mask = np.array([("llc" in x) for x in fn_low])

    if mask is None or mask.sum() == 0:
        desc = _getcol(s, "description")
        if desc is not None:
            dlow = np.array([str(x).lower() for x in desc])
            mask = np.array([("long" in x) for x in dlow])

    s_sorted = _sort_sr(s, "year")
    sel = s_sorted[mask] if (mask is not None and mask.sum() > 0) else s_sorted
    sel = sel[:max_files]
    lcc = sel.download_all()
    return lcc.stitch()


def fetch_tess_spoc_lc(target: str, max_files: int = 2):
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
    lcc = s.download_all()
    return lcc.stitch()


def fetch_with_retry(kind: str, target: str, max_files: int, retries: int = 2, delay: int = 5):
    last = None
    for k in range(retries + 1):
        try:
            if kind.lower() == "kepler":
                return fetch_kepler_llc(target, max_files)
            return fetch_tess_spoc_lc(target, max_files)
        except Exception as e:
            last = e
            if k < retries:
                print(f"[!] İndirme hatası: {e} → {delay}s sonra tekrar...")
                time.sleep(delay)
    raise last


# ----------------- işleme -----------------
def clean_flatten(lc: lk.LightCurve, sigma: float = 5.0, window_length: int = 401) -> lk.LightCurve:
    return (
        lc.remove_nans()
        .normalize()
        .remove_outliers(sigma=sigma)
        .flatten(window_length=window_length)
        .remove_nans()
    )


def bls_coarse_to_fine(t, y, pmin=0.5, pmax=20.0, top_peaks=3):
    t = _np(t)
    y = _np(y)
    durations = np.linspace(0.05, 0.3, 25)
    bls = BoxLeastSquares(t, y)

    p_coarse = np.geomspace(pmin, pmax, 2000)
    res_c = bls.power(p_coarse, durations)
    top_idx = np.argpartition(res_c.power, -top_peaks)[-top_peaks:]
    cand = sorted(res_c.period[top_idx])

    best = None
    for pc in cand:
        p1, p2 = pc * 0.98, pc * 1.02
        p_fine = np.linspace(p1, p2, 6000)
        res_f = bls.power(p_fine, durations)
        k = np.argmax(res_f.power)
        cb = dict(
            period=float(res_f.period[k]),
            duration=float(res_f.duration[k]),
            t0=float(res_f.transit_time[k]),
            power=float(res_f.power[k]),
        )
        if (best is None) or (cb["power"] > best["power"]):
            best = cb
    return best


def phase_bin_fixed(phase, flux, bins=200, robust=True):
    edges = np.linspace(-0.5, 0.5, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    which = np.digitize(phase, edges) - 1
    yb = np.full(bins, np.nan)
    for i in range(bins):
        m = which == i
        if not np.any(m):
            continue
        yb[i] = np.nanmedian(flux[m]) if robust else np.nanmean(flux[m])
    ok = np.isfinite(yb)
    if ok.sum() >= 2:
        yb[~ok] = np.interp(centers[~ok], centers[ok], yb[ok])
    else:
        yb[~ok] = 0.0
    return centers, yb


def compute_metrics_from_fold(phase, flux, period, duration):
    phi_dur = duration / period
    in_mask = np.abs(phase) <= 0.6 * phi_dur
    oot_mask = np.abs(phase) >= 3.0 * phi_dur
    baseline = float(np.nanmedian(flux[oot_mask]))
    in_med = float(np.nanmedian(flux[in_mask]))
    depth = baseline - in_med
    depth_ppm = depth * 1e6
    rms_oot = float(np.nanstd(flux[oot_mask]))
    n_in = int(np.nansum(in_mask))
    snr = float(depth / (rms_oot / math.sqrt(max(n_in, 1)))) if n_in > 0 and rms_oot > 0 else float("nan")
    return dict(phi_dur=float(phi_dur), baseline=baseline, depth=depth, depth_ppm=depth_ppm, rms_oot=rms_oot, n_in=n_in, snr=snr)


# ----------------- çizimler -----------------
def plot_phase_images_and_labels(target, mission, phase, flux, xb, yb, metrics, out_images_dir, out_labels_dir, image_size=(1200, 800), draw_overview=False):
    """Faz-katlanmış iki görsel (points ve binned) ve YOLO label dosyaları üretir."""
    _ensure_dirs(out_images_dir, out_labels_dir)

    # Y ekseni limitlerini robust belirle (görüntü/label uyumu için sabitle)
    y_lo = np.nanpercentile(flux, 1)
    y_hi = np.nanpercentile(flux, 99)
    pad = 0.15 * (y_hi - y_lo)
    y_min = y_lo - pad
    y_max = y_hi + pad

    def _save_fig(x_points, y_points, with_binned=False, fname_suffix="phase", dpi=150):
        plt.figure(figsize=(image_size[0] / 100.0, image_size[1] / 100.0))
        plt.scatter(x_points, y_points, s=3, alpha=0.35, label="points")
        if with_binned:
            plt.plot(xb, yb, lw=2, color="tab:red", label="binned")
        plt.xlim(-0.5, 0.5)
        plt.ylim(y_min, y_max)
        plt.xlabel("Phase (days)")
        plt.ylabel("Flux (norm)")
        plt.title(f"{target} ({mission}) phase-folded")
        plt.tight_layout()
        img_name = f"{target.replace(' ','_')}_{mission}_{fname_suffix}{'_binned' if with_binned else ''}.png"
        img_path = os.path.join(out_images_dir, img_name)
        plt.savefig(img_path, dpi=dpi)
        plt.close()
        return img_name, img_path

    # Görseller
    img_name_raw, _ = _save_fig(phase, flux, with_binned=False, fname_suffix="phase")
    img_name_bin, _ = _save_fig(phase, flux, with_binned=True, fname_suffix="phase")

    # YOLO Label hesapla (hem raw hem binned görsel için aynı bbox'lar)
    phi_dur = float(metrics["phi_dur"]) if "phi_dur" in metrics else float((metrics["depth"] * 0 + 1) * 0.0)  # default 0
    # Eğer phi_dur yoksa duration/period'dan hesaplanmalı; bu fonksiyona gelen metrics compute_metrics_from_fold'tan geliyor, içinde phi_dur var.
    phi_dur = metrics.get("phi_dur", phi_dur)

    # 1) Birincil transit: merkez 0.0
    boxes = []
    def add_box(center_x_phase):
        # Yatay genişlik: 1.5x transit süresi (faz birimi)
        w_phase = max(0.02, 1.5 * 2.0 * phi_dur)  # güvenli genişlik
        x1 = max(-0.5, center_x_phase - w_phase / 2.0)
        x2 = min(0.5, center_x_phase + w_phase / 2.0)
        # Düşey: transit çukurunu kapsayacak şekilde baseline ve depth'e göre
        baseline = metrics.get("baseline", float(np.nanmedian(flux)))
        depth = metrics.get("depth", float(baseline - np.nanmedian(flux)))
        y_low = baseline - 1.2 * depth
        y_high = baseline - 0.2 * depth
        # Eksen limitleri içinde kırp
        y1 = max(y_min, min(y_low, y_high))
        y2 = min(y_max, max(y_low, y_high))
        # YOLO formatına normalize
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = (x2 - x1)
        h = (y2 - y1)
        # Normalize [0,1] (x: [-0.5,0.5] → [0,1], y: [y_min,y_max] → [0,1])
        nx = (cx + 0.5)
        ny = (cy - y_min) / (y_max - y_min)
        nw = w
        nh = h / (y_max - y_min)
        boxes.append((0, nx, ny, nw, nh))  # class 0: transit

    add_box(0.0)
    # 2) İkincil (even) transit adayı: 0.5
    add_box(0.5)

    def _write_label(img_name):
        stem = os.path.splitext(img_name)[0]
        label_path = os.path.join(out_labels_dir, f"{stem}.txt")
        with open(label_path, "w", encoding="utf-8") as f:
            for cls_id, cx, cy, w, h in boxes:
                # YOLO v5/v8: class cx cy w h (hepsi normalize [0,1])
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    _write_label(img_name_raw)
    _write_label(img_name_bin)


# ----------------- tek hedef akışı -----------------
def process_target(target: str, mission: str, max_files: int, pmin: float, pmax: float, bins: int, out_root: str):
    kind = mission.lower()
    lc = fetch_with_retry(kind, target=target, max_files=max_files)
    lc_f = clean_flatten(lc)

    t = lc_f.time.value
    y = lc_f.flux.value
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]

    best = bls_coarse_to_fine(t, y, pmin=pmin, pmax=pmax)
    folded = lk.LightCurve(time=t, flux=y).fold(period=best["period"], epoch_time=best["t0"]) 
    phase = _np(folded.time.value)
    flux = _np(folded.flux.value)

    xb, yb = phase_bin_fixed(phase, flux, bins=bins, robust=True)
    mets = compute_metrics_from_fold(phase, flux, best["period"], best["duration"])

    images_dir = os.path.join(out_root, "images")
    labels_dir = os.path.join(out_root, "labels")
    plot_phase_images_and_labels(target, mission, phase, flux, xb, yb, mets, images_dir, labels_dir)


# ----------------- ana -----------------
def main():
    ap = argparse.ArgumentParser("Graphs + YOLO Labels Üretici")
    ap.add_argument("--targets", type=str, default="targets.csv")
    ap.add_argument("--out", type=str, default="graphs")
    ap.add_argument("--max-files", type=int, default=2)
    ap.add_argument("--bins", type=int, default=200)
    ap.add_argument("--pmin", type=float, default=0.5)
    ap.add_argument("--pmax", type=float, default=20.0)
    ap.add_argument("--limit", type=int, default=0, help="İşlenecek maksimum hedef sayısı (0: sınırsız)")
    args = ap.parse_args()

    _ensure_dirs(args.out, os.path.join(args.out, "images"), os.path.join(args.out, "labels"))

    count = 0
    with open(args.targets, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            target = row["target"].strip()
            mission = row["mission"].strip()
            try:
                print(f"→ {target} ({mission}) işleniyor...")
                process_target(target, mission, max_files=args.max_files, pmin=args.pmin, pmax=args.pmax, bins=args.bins, out_root=args.out)
                count += 1
                if args.limit and count >= args.limit:
                    print(f"[i] Limit ({args.limit}) nedeniyle durduruldu.")
                    break
            except Exception as e:
                print(f"[x] {target} ({mission}) atlandı: {e}")

    print(f"[✓] Tamamlandı. Çıktılar: {args.out}")


if __name__ == "__main__":
    main()


