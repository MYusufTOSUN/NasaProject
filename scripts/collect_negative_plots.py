import os
import re
from pathlib import Path
from typing import List, Optional, Tuple
import shutil
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# FITS opsiyonel; yoksa sadece mevcut görseller kopyalanır
try:
    from astropy.io import fits
    HAS_FITS = True
except Exception:
    HAS_FITS = False

# Çevrimiçi ışık eğrisi indirme (opsiyonel)
try:
    # lightkurve, astroquery ile MAST'tan veri indirir
    from lightkurve import search_lightcurve
    HAS_LIGHTKURVE = True
except Exception:
    HAS_LIGHTKURVE = False


# =========================
# CONFIG
# =========================
LABELS_CSV = Path("data/labels.csv")
IMAGES_ROOT = Path("data/images")
# Projede görsellerin 'scripts/raw_images' altında olması halinde fallback kökü
SCRIPTS_IMAGES_ROOT = Path("scripts/raw_images")
FITS_ROOT = Path("data/fits")

# Üretilen veya normalize edilmiş negatif görsellerin önbellek kökü
CACHE_IMAGES_ROOT = Path("graphs/images")

# İstenirse tüm split negatif klasörlerine de kopyala (kullanıcı isteği)
COPY_TO_ALL_SPLITS = False

PLOTS_ROOT = Path("data/plots")
OUT_DIRS = {
    "train": PLOTS_ROOT / "train" / "negative",
    "val":   PLOTS_ROOT / "val" / "negative",
    "test":  PLOTS_ROOT / "test" / "negative",
}

LOG_PATH = PLOTS_ROOT / "negatives_log.txt"
MISSING_PATH = PLOTS_ROOT / "missing_negatives.txt"


# =========================
# UTILS
# =========================
def ensure_dirs():
    PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
    for p in OUT_DIRS.values():
        p.mkdir(parents=True, exist_ok=True)
    CACHE_IMAGES_ROOT.mkdir(parents=True, exist_ok=True)


def extract_numeric_id(target: str) -> Optional[int]:
    """
    target -> 'kic1026032', 'Kepler-10', 'tic12345' vs.
    İçinden ilk rakam dizisini döndür.
    """
    if not isinstance(target, str):
        return None
    m = re.search(r"(\d+)", target)
    return int(m.group(1)) if m else None


def kic_pad9(num: int) -> str:
    return f"{num:09d}"


def image_search_patterns(kic_int: int) -> List[str]:
    """
    Görsel arama için çoklu desenler (alt klasörler dahil).
    Önce 9 haneli pad'li, sonra padsiz varyasyonlar.
    """
    pad9 = kic_pad9(kic_int)
    patterns = [
        # 9 haneli
        f"**/kplr{pad9}*.png", f"**/kplr{pad9}*.jpg", f"**/kplr{pad9}*.jpeg",
        f"**/KIC_{pad9}*.png", f"**/KIC_{pad9}*.jpg", f"**/KIC_{pad9}*.jpeg",
        f"**/kic{pad9}*.png",  f"**/kic{pad9}*.jpg",  f"**/kic{pad9}*.jpeg",
        f"**/{pad9}*.png",     f"**/{pad9}*.jpg",     f"**/{pad9}*.jpeg",

        # padsiz fallback
        f"**/kplr{kic_int}*.png", f"**/kplr{kic_int}*.jpg",
        f"**/KIC_{kic_int}*.png", f"**/KIC_{kic_int}*.jpg",
        f"**/kic{kic_int}*.png",  f"**/kic{kic_int}*.jpg",
        f"**/{kic_int}*.png",     f"**/{kic_int}*.jpg",
    ]
    return patterns


def find_existing_image(kic_int: int, root: Path) -> Optional[Path]:
    for pat in image_search_patterns(kic_int):
        hits = list(root.glob(pat))
        if hits:
            # Tercihen PNG
            hits_sorted = sorted(hits, key=lambda p: (p.suffix.lower() != ".png", str(p).lower()))
            return hits_sorted[0]
    return None


def find_existing_image_in_roots(kic_int: int, roots: List[Path]) -> Optional[Path]:
    for r in roots:
        if r.exists():
            hit = find_existing_image(kic_int, r)
            if hit is not None:
                return hit
    return None


def find_fits_candidates(kic_int: int, root: Path) -> List[Path]:
    """
    FITS ararken yaygın Kepler/TESS adlandırmalarını dene.
    """
    if not root.exists():
        return []

    pad9 = kic_pad9(kic_int)
    patterns = [
        f"**/kplr{pad9}*.fits",
        f"**/kplr{kic_int}*.fits",
        f"**/KIC_{pad9}*.fits",
        f"**/KIC_{kic_int}*.fits",
        f"**/{pad9}*.fits",
        f"**/{kic_int}*.fits",
        # TESS/TIC muhtemeli:
        f"**/tess*tic*{kic_int}*.fits",
        f"**/tic*{kic_int}*.fits",
    ]

    results = []
    for pat in patterns:
        results.extend(list(root.glob(pat)))
    # benzersiz ve deterministik sırala
    uniq = sorted(set(results), key=lambda p: str(p).lower())
    return uniq


def generate_plot_from_fits(fits_path: Path, out_png: Path, title: str) -> bool:
    """
    TIME ve (PDCSAP_FLUX | SAP_FLUX) sütunlarını okuyup basit çizim yap.
    """
    if not HAS_FITS:
        return False
    try:
        with fits.open(fits_path, memmap=True) as hdul:
            # Kepler/TESS light curve genelde HDU 1'de olur
            # Bazı dosyalarda 2. HDU da olabilir, robust olalım.
            time, flux = None, None
            for hdu in hdul:
                data = getattr(hdu, "data", None)
                if data is None:
                    continue
                cols = data.columns.names if hasattr(data, "columns") else []
                cols_lower = [c.lower() for c in cols] if cols else []

                # TIME kolonu var mi?
                if "time" in cols_lower:
                    time_col = data.field(cols[cols_lower.index("time")])
                else:
                    continue

                # Flux kolon tercihi: PDCSAP_FLUX > SAP_FLUX > FLUX
                flux_colname = None
                for cand in ["pdcsap_flux", "sap_flux", "flux"]:
                    if cand in cols_lower:
                        flux_colname = cols[cols_lower.index(cand)]
                        break
                if flux_colname is None:
                    continue

                flux_col = data.field(flux_colname)
                time = np.array(time_col, dtype=float)
                flux = np.array(flux_col, dtype=float)

                # NaN temizle
                mask = np.isfinite(time) & np.isfinite(flux)
                time = time[mask]
                flux = flux[mask]
                if len(time) > 5:
                    break  # uygun bir HDU bulduk

            if time is None or flux is None or len(time) < 5:
                return False

        # Çiz
        plt.figure()
        plt.plot(time, flux, linewidth=1)
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Flux")
        plt.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=150)
        plt.close()
        return True
    except Exception:
        return False


def generate_plot_from_online(num_id: int, mission: Optional[str], out_png: Path, title: str) -> bool:
    """
    Yerel FITS bulunamazsa Lightkurve üzerinden MAST'tan indirerek üret.
    TESS için TIC, Kepler için KIC sorgularını dener.
    """
    if not HAS_LIGHTKURVE:
        return False
    try:
        query = None
        mission_lower = (mission or "").lower()
        
        # Kepler/K2 için KIC sorgusu
        if "kep" in mission_lower or "k2" in mission_lower:
            try:
                query = search_lightcurve(f"KIC {num_id}", mission="Kepler")
            except Exception:
                pass
        # TESS için TIC sorgusu
        elif "tes" in mission_lower:
            try:
                query = search_lightcurve(f"TIC {num_id}", mission="TESS")
            except Exception:
                pass
        else:
            # Misyon bilinmiyorsa her ikisini sırayla dene
            try:
                query = search_lightcurve(f"KIC {num_id}")
            except Exception:
                pass
            if query is None or len(query) == 0:
                try:
                    query = search_lightcurve(f"TIC {num_id}")
                except Exception:
                    pass

        if query is None or len(query) == 0:
            return False

        # İlk sonucu indir
        lc = None
        try:
            lc = query[0].download()
        except Exception:
            pass
        
        # Stitch denemesi
        if lc is None:
            try:
                lc_collection = query.download_all()
                if lc_collection is not None and len(lc_collection) > 0:
                    lc = lc_collection.stitch()
            except Exception:
                pass
        
        if lc is None:
            return False

        # Zaman/Flux al - lightkurve API
        try:
            time = np.array(lc.time.value, dtype=float)
            flux = np.array(lc.flux.value, dtype=float)
        except Exception:
            try:
                # Alternatif API
                time = np.array(lc.time, dtype=float)
                flux = np.array(lc.flux, dtype=float)
            except Exception:
                return False

        mask = np.isfinite(time) & np.isfinite(flux)
        time = time[mask]
        flux = flux[mask]
        if len(time) < 5:
            return False

        # Çiz ve kaydet
        plt.figure()
        plt.plot(time, flux, linewidth=1, color="steelblue")
        plt.title(title)
        plt.xlabel("Time (days)")
        plt.ylabel("Flux")
        plt.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=150)
        plt.close()
        return True
    except Exception as e:
        # Hata ayıklama (opsiyonel)
        # print(f"[DEBUG] Online generation failed for {num_id}: {e}")
        return False


def generate_synthetic_negative(num_id: int, out_png: Path, title: str, seed: Optional[int] = None) -> bool:
    """
    Gerçekçi sentetik negatif ışık eğrisi üret.
    Gezegen sinyali OLMAYAN tipik durumlar: gürültü, stellar var, binary, artifacts.
    """
    if seed is not None:
        np.random.seed(seed)
    
    try:
        n_points = np.random.randint(800, 2000)
        time = np.linspace(0, np.random.uniform(50, 150), n_points)
        
        # Temel flux seviyesi
        base_flux = 1.0
        
        # Negatif türü seç (gerçek negatif senaryolar)
        neg_type = np.random.choice([
            "pure_noise",           # Sadece gürültü
            "stellar_rotation",     # Yıldız rotasyonu (yavaş varyasyon)
            "stellar_pulsation",    # Pulsasyon (düzenli ama gezegen değil)
            "eclipsing_binary",     # Binary yıldız (asimetrik, derin)
            "instrumental_drift",   # Sistematik trend + gürültü
            "single_artifact",      # Tek olay (kozmik ışın vs)
        ])
        
        if neg_type == "pure_noise":
            # Sadece fotonik gürültü
            noise_level = np.random.uniform(0.0005, 0.002)
            flux = base_flux + np.random.normal(0, noise_level, n_points)
        
        elif neg_type == "stellar_rotation":
            # Yıldız yüzeyindeki lekeler -> yavaş sinüzoidal
            period = np.random.uniform(5, 30)  # Uzun periyot
            amplitude = np.random.uniform(0.005, 0.02)  # Belirgin varyasyon
            flux = base_flux + amplitude * np.sin(2 * np.pi * time / period)
            noise = np.random.normal(0, 0.001, n_points)
            flux += noise
        
        elif neg_type == "stellar_pulsation":
            # Değişken yıldız (Cepheid, RR Lyrae tarzı)
            period = np.random.uniform(0.3, 5)  # Kısa periyot
            amplitude = np.random.uniform(0.02, 0.1)  # Büyük amplitüd
            # Asimetrik pulsasyon (gezegen geçişinden farklı)
            flux = base_flux + amplitude * (np.sin(2 * np.pi * time / period) + 
                                           0.3 * np.sin(4 * np.pi * time / period))
            noise = np.random.normal(0, 0.002, n_points)
            flux += noise
        
        elif neg_type == "eclipsing_binary":
            # İki yıldızın birbirini tutması (gezegenden farklı şekil)
            period = np.random.uniform(1, 10)
            depth_primary = np.random.uniform(0.1, 0.4)  # Çok derin
            depth_secondary = depth_primary * np.random.uniform(0.3, 0.8)
            width = np.random.uniform(0.05, 0.15)  # Geniş
            
            flux = np.ones(n_points) * base_flux
            phase = (time % period) / period
            
            # Primary eclipse (daha derin)
            primary_mask = np.abs(phase - 0.0) < width
            flux[primary_mask] *= (1 - depth_primary)
            
            # Secondary eclipse (daha sığ)
            secondary_mask = np.abs(phase - 0.5) < width
            flux[secondary_mask] *= (1 - depth_secondary)
            
            noise = np.random.normal(0, 0.003, n_points)
            flux += noise
        
        elif neg_type == "instrumental_drift":
            # Uzay aracı ısınma/soğuma -> trend
            trend = np.polynomial.polynomial.polyval(
                time / time[-1], 
                [base_flux, np.random.uniform(-0.05, 0.05), np.random.uniform(-0.02, 0.02)]
            )
            noise = np.random.normal(0, 0.002, n_points)
            flux = trend + noise
        
        else:  # single_artifact
            # Temiz sinyal + tek olay (kozmik ışın vuruşu)
            flux = base_flux + np.random.normal(0, 0.0008, n_points)
            artifact_idx = np.random.randint(100, n_points - 100)
            artifact_width = np.random.randint(1, 5)
            artifact_depth = np.random.uniform(0.05, 0.2)
            flux[artifact_idx:artifact_idx + artifact_width] -= artifact_depth
        
        # Fiziksel olmayan değerleri temizle
        flux = np.clip(flux, 0.5, 1.5)
        
        # Grafik çiz
        plt.figure(figsize=(10, 4))
        plt.plot(time, flux, 'k.', markersize=1, alpha=0.6)
        plt.title(f"{title} (Synthetic Negative - {neg_type})")
        plt.xlabel("Time (days)")
        plt.ylabel("Normalized Flux")
        plt.ylim(flux.min() - 0.02, flux.max() + 0.02)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=150)
        plt.close()
        return True
    except Exception:
        return False


def safe_copy(src: Path, dst: Path) -> bool:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    except Exception:
        return False


def copy_to_outputs(cache_png: Path, out_name: str, primary_split: str, logs: List[str]) -> None:
    """
    Önbellekteki görseli hedef split negatif klasörüne ve istenirse tüm split'lere kopyala.
    """
    if COPY_TO_ALL_SPLITS:
        for split_name, out_dir in OUT_DIRS.items():
            dst = out_dir / out_name
            if safe_copy(cache_png, dst):
                logs.append(f"[COPY_OUT] split={split_name} src={cache_png} -> {dst}")
    else:
        dst = OUT_DIRS[primary_split] / out_name
        if safe_copy(cache_png, dst):
            logs.append(f"[COPY_OUT] split={primary_split} src={cache_png} -> {dst}")


def deterministic_split_for_target(target: str) -> str:
    """
    labels.csv yoksa split'i belirlemek için deterministik ve bağımsız bir yöntem.
    SHA1 tabanlı hash ile 0-1 aralığına indirger, 0.70/0.15/0.15 oranında böler.
    """
    h = hashlib.sha1(target.encode("utf-8")).hexdigest()
    # İlk 12 hex'i alıp 0-1'e indirger (sabit ve yeterince rastgele)
    val = int(h[:12], 16) / float(16 ** 12)
    if val < 0.70:
        return "train"
    elif val < 0.85:
        return "val"
    else:
        return "test"


def load_labels_df() -> pd.DataFrame:
    """
    Öncelik: data/labels.csv
    Yoksa: targets.csv -> (target, mission, label) ve deterministik split
    O da yoksa: index.csv -> target bazlı unique + label ile deterministik split
    Dönüş: kolonlar ['target','mission','label','split']
    """
    # 1) data/labels.csv
    if LABELS_CSV.exists():
        df = pd.read_csv(LABELS_CSV)
        # Beklenen kolonlar kontrolü
        missing = [c for c in ["target", "label", "split"] if c not in df.columns]
        if missing:
            raise ValueError(f"labels.csv eksik kolonlar: {missing}")
        # mission olmayabilir, None ile doldur
        if "mission" not in df.columns:
            df["mission"] = None
        return df[["target", "mission", "label", "split"]]

    # 2) targets.csv
    tgt_csv = Path("targets.csv")
    if tgt_csv.exists():
        df = pd.read_csv(tgt_csv)
        # Beklenen asgari kolonlar
        if not {"target", "label"}.issubset(df.columns):
            raise ValueError("targets.csv içinde en az 'target' ve 'label' kolonları olmalı")
        if "mission" not in df.columns:
            df["mission"] = None
        # target bazında tekilleştir (varsa tekrarlar)
        g = df.groupby("target").agg({"mission": "first", "label": "first"}).reset_index()
        g["split"] = g["target"].astype(str).apply(deterministic_split_for_target)
        return g[["target", "mission", "label", "split"]]

    # 3) index.csv
    idx_csv = Path("index.csv")
    if idx_csv.exists():
        df = pd.read_csv(idx_csv)
        if not {"target", "label"}.issubset(df.columns):
            raise ValueError("index.csv içinde en az 'target' ve 'label' kolonları olmalı")
        if "mission" not in df.columns:
            df["mission"] = None
        g = df.groupby("target").agg({"mission": "first", "label": "first"}).reset_index()
        g["split"] = g["target"].astype(str).apply(deterministic_split_for_target)
        return g[["target", "mission", "label", "split"]]

    # Hiçbiri yoksa
    raise FileNotFoundError(
        "Ne data/labels.csv ne de targets.csv/index.csv bulundu. Etiket kaynağı gerekli."
    )


def main():
    ensure_dirs()

    # Etiketleri yükle (labels.csv yoksa targets.csv/index.csv fallback)
    df = load_labels_df()

    # Negatifler
    neg = df[df["label"] == 0].copy()

    # Hedef ID çıkar
    neg["num_id"] = neg["target"].apply(extract_numeric_id)
    neg = neg.dropna(subset=["num_id"])
    neg["num_id"] = neg["num_id"].astype(int)

    total = len(neg)
    copied = 0
    generated = 0
    failed = 0

    missing_ids = []
    logs = []

    for _, row in tqdm(neg.iterrows(), total=total, desc="Processing negatives"):
        target = str(row["target"])
        split = str(row["split"]).lower()
        if split not in OUT_DIRS:
            # bilinmeyen split -> val'e düş
            split = "val"

        out_dir = OUT_DIRS[split]
        num_id = int(row["num_id"])

        # Çıkış dosya adı: öncelik pad'li
        out_name = f"{kic_pad9(num_id)}.png"
        out_path = out_dir / out_name

        # 1) Varolan görsel var mı? Önce data/images, scripts/raw_images ve graphs/images (önbellek)
        img_roots: List[Path] = [IMAGES_ROOT]
        if SCRIPTS_IMAGES_ROOT.exists():
            img_roots.append(SCRIPTS_IMAGES_ROOT)
        if CACHE_IMAGES_ROOT.exists():
            img_roots.append(CACHE_IMAGES_ROOT)
        img = find_existing_image_in_roots(num_id, img_roots)
        if img is not None:
            # Önbellekte normalleştir -> graphs/images/{KIC9}.png olarak da saklayalım
            cache_name = f"{kic_pad9(num_id)}{img.suffix.lower()}"
            cache_png = CACHE_IMAGES_ROOT / cache_name
            if not cache_png.exists():
                safe_copy(img, cache_png)
            # Çıkış(lar)a kopyala
            copy_to_outputs(cache_png, f"{kic_pad9(num_id)}.png", split, logs)
            copied += 1
            logs.append(f"[COPY] split={split} target={target} src={img} -> cache={cache_png}")
            continue  # sonraki kayda geç

        # 2) FITS'ten üret
        produced = False
        if FITS_ROOT.exists():
            fits_list = find_fits_candidates(num_id, FITS_ROOT)
            for fp in fits_list:
                title = f"{target}"
                # Önce önbelleğe üret, sonra çıkışlara kopyala
                cache_png = CACHE_IMAGES_ROOT / f"{kic_pad9(num_id)}.png"
                if generate_plot_from_fits(fp, cache_png, title=title):
                    copy_to_outputs(cache_png, f"{kic_pad9(num_id)}.png", split, logs)
                    generated += 1
                    logs.append(f"[GENERATE] split={split} target={target} fits={fp} -> cache={cache_png}")
                    produced = True
                    break

        # 3) Çevrimiçi (Lightkurve) ile indirip üret (FITS yerelde yoksa)
        if not produced:
            cache_png = CACHE_IMAGES_ROOT / f"{kic_pad9(num_id)}.png"
            mission = str(row.get("mission", "")) if "mission" in row else None
            if generate_plot_from_online(num_id, mission, cache_png, title=str(target)):
                copy_to_outputs(cache_png, f"{kic_pad9(num_id)}.png", split, logs)
                generated += 1
                logs.append(f"[GENERATE_ONLINE] split={split} target={target} -> cache={cache_png}")
                produced = True

        # 4) Sentetik negatif üret (gerçekçi, gezegen olmayan senaryolar)
        if not produced:
            cache_png = CACHE_IMAGES_ROOT / f"{kic_pad9(num_id)}.png"
            # Deterministik seed = hedef ID (tekrarlanabilir)
            if generate_synthetic_negative(num_id, cache_png, title=str(target), seed=num_id):
                copy_to_outputs(cache_png, f"{kic_pad9(num_id)}.png", split, logs)
                generated += 1
                logs.append(f"[GENERATE_SYNTHETIC] split={split} target={target} -> cache={cache_png}")
                produced = True

        if not produced:
            failed += 1
            missing_ids.append(target)
            logs.append(f"[MISS] split={split} target={target} (no image, no fits-derived plot, synthetic failed)")

    # Özet
    summary = [
        "==== NEGATIVE PLOTS SUMMARY ====",
        f"Total negatives: {total}",
        f"Copied (existing images): {copied}",
        f"Generated from FITS: {generated}",
        f"Failed (missing): {failed}",
        "",
        "Output dirs:",
        *(f"- {k}: {v}" for k, v in OUT_DIRS.items()),
        "",
    ]

    # Log yaz
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(summary))
        f.write("\n==== DETAILED LOGS ====\n")
        for line in logs:
            f.write(line + "\n")

    # Missing yaz
    if missing_ids:
        with open(MISSING_PATH, "w", encoding="utf-8") as f:
            for t in missing_ids:
                f.write(str(t) + "\n")

    # Konsola özet bas
    print("\n".join(summary))
    if failed:
        print(f"Missing list saved to: {MISSING_PATH}")
    print(f"Log saved to: {LOG_PATH}")


if __name__ == "__main__":
    main()


