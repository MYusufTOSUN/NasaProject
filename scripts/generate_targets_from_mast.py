#!/usr/bin/env python3
"""
MAST Doğrulamalı Hedef Listesi Üretici
NASA Exoplanet Archive ve MAST'tan doğrulanmış Kepler/TESS hedefleri çeker.
Her hedefin gerçekten indirilebilir ışık eğrisi olup olmadığını kontrol eder.
"""

import argparse
import csv
import random
import time
from pathlib import Path
from typing import List, Tuple, Set

import numpy as np
from tqdm import tqdm

from astroquery.mast import Observations
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
from astroquery.exceptions import InvalidQueryError
import lightkurve as lk


# ============================================================
# Genel yardımcılar
# ============================================================
def retry(func, tries=3, delay=1.5, backoff=2.0, exceptions=(Exception,), *args, **kwargs):
    """Retry mekanizması (örnek: MAST geçici hataları için)."""
    attempt = 0
    wait = delay
    while True:
        try:
            return func(*args, **kwargs)
        except exceptions:
            attempt += 1
            if attempt >= tries:
                raise
            time.sleep(wait)
            wait *= backoff


def target_exists_in_mast(target_name: str, mission: str) -> bool:
    """
    Hedefin gerçekten MAST'ta olup olmadığını hızlıca kontrol eder.
    Işık eğrisi aramasında 0 sonuç dönüyorsa False döner.
    """
    try:
        res = lk.search_lightcurve(target_name, mission=mission)
        return len(res) > 0
    except Exception:
        return False


def has_lightcurve(target_name: str, mission: str, min_rows: int = 50) -> bool:
    """
    Hedefin ışık eğrisinin indirilebilir olup olmadığını test eder.
    En az min_rows uzunluğunda bir eğri varsa True döner.
    """
    if not target_exists_in_mast(target_name, mission):
        return False

    def _search():
        return lk.search_lightcurve(target_name, mission=mission)
    
    try:
        sr = retry(_search, tries=3, delay=1.0, backoff=1.8)
    except Exception:
        return False
    
    if len(sr) == 0:
        return False

    try:
        obj = sr[0].download()
        if obj is None:
            return False
        length = len(getattr(obj, "time", [])) if hasattr(obj, "time") else len(obj)
        return length >= min_rows
    except Exception:
        return False


# ============================================================
# Pozitif hedefler (transit sistemler)
# ============================================================
def fetch_confirmed_transiting_hosts() -> List[Tuple[str, str]]:
    """
    NASA Exoplanet Archive üzerinden doğrulanmış, transiting gezegenli
    Kepler ve TESS yıldızlarını çeker.
    """
    try:
        table = retry(NasaExoplanetArchive.query_criteria, 
                     tries=3, delay=1.0, backoff=2.0,
                     table="pscomppars", select="*")
    except (InvalidQueryError, Exception) as e:
        print(f"⚠ Exoplanet Archive query failed: {e}")
        print("Alternatif method denenecek...")
        try:
            table = NasaExoplanetArchive.query_criteria(table="ps", select="*")
        except Exception as e2:
            raise RuntimeError(f"Exoplanet Archive query failed: {e2}")

    # Transit flag kontrolü
    if "tran_flag" in table.colnames:
        mask_transit = np.array(table["tran_flag"]) == 1
        table = table[mask_transit]
    elif "pl_tranflag" in table.colnames:
        mask_transit = np.array(table["pl_tranflag"]) == 1
        table = table[mask_transit]

    out = []
    for row in table:
        # Host name'i çeşitli kolonlardan dene
        host = None
        for col in ["hostname", "pl_hostname", "pl_name"]:
            if col in row.colnames:
                host = str(row.get(col) or "").strip()
                if host:
                    break
        
        if not host:
            continue
        
        pl_name = str(row.get("pl_name") or "").strip() if "pl_name" in row.colnames else ""
        
        # Facility bilgisi
        facility = ""
        if "disc_facility" in row.colnames:
            facility = str(row.get("disc_facility") or "").lower()

        # Mission belirleme
        mission = None
        if host.startswith("Kepler-") or pl_name.startswith("Kepler-") or "kepler" in facility:
            mission = "Kepler"
        elif "tess" in facility or host.startswith("TOI") or pl_name.startswith("TOI") or "tic" in host.lower():
            mission = "TESS"

        if mission:
            target = host or pl_name
            if target:
                out.append((target, mission))

    # Benzersiz liste (host+mission)
    seen, uniq = set(), []
    for pair in out:
        if pair not in seen:
            seen.add(pair)
            uniq.append(pair)
    
    return uniq


# ============================================================
# Negatif örnekler (KIC/TIC ID)
# ============================================================
def sample_negatives(mission: str, n_wanted: int, positive_names: Set[str], seed: int) -> List[Tuple[str, str]]:
    """KIC/TIC ID'lerinden indirilebilir ışık eğrisi bulunan negatif örnekler toplar."""
    random.seed(seed)
    results = []
    seen = set()

    if mission == "Kepler":
        candidates = list(range(1_000_000, 13_999_999, 2345))
        prefix = "KIC "
    else:
        candidates = list(range(10_000_000, 999_999_999, 9_999_999))
        prefix = "TIC "

    random.shuffle(candidates)
    pbar = tqdm(total=n_wanted, desc=f"Sampling negatives ({mission})")
    
    for cid in candidates:
        if len(results) >= n_wanted:
            break
        
        name = f"{prefix}{cid}"
        if name in seen or name in positive_names:
            continue
        
        seen.add(name)

        # Önce hedef MAST'ta var mı kontrol et
        if not target_exists_in_mast(name, mission):
            continue
        
        if has_lightcurve(name, mission):
            results.append((name, mission))
            pbar.update(1)
    
    pbar.close()
    return results


# ============================================================
# Ana fonksiyon
# ============================================================
def build_targets(total: int, pos_fraction: float, seed: int, out_csv: Path, min_rows: int):
    """
    Ana hedef listesi oluşturma fonksiyonu.
    
    Args:
        total: Toplam hedef sayısı
        pos_fraction: Pozitif örnek oranı (0-1)
        seed: Random seed
        out_csv: Çıktı CSV yolu
        min_rows: Minimum ışık eğrisi uzunluğu
    """
    random.seed(seed)
    np.random.seed(seed)

    print("="*60)
    print("MAST DOĞRULAMALI HEDEF LİSTESİ ÜRETİCİ")
    print("="*60)
    print(f"Toplam hedef: {total}")
    print(f"Pozitif oran: {pos_fraction}")
    print(f"Seed: {seed}")
    print(f"Min. ışık eğrisi uzunluğu: {min_rows}")
    print("="*60)

    print("\n[1/4] Pozitif (transit) hedefler çekiliyor...")
    pos_candidates = fetch_confirmed_transiting_hosts()
    print(f"  {len(pos_candidates)} aday bulundu")
    random.shuffle(pos_candidates)

    desired_pos = int(total * pos_fraction)
    desired_neg = total - desired_pos
    neg_kepler_target = desired_neg // 2
    neg_tess_target = desired_neg - neg_kepler_target

    print(f"\n[2/4] Pozitif hedefler doğrulanıyor (indirilebilirlik test)...")
    print(f"  Hedef: {desired_pos} pozitif")
    
    positives, pos_names_set = [], set()
    for target, mission in tqdm(pos_candidates, desc="Verifying positives"):
        if target in pos_names_set:
            continue
        
        if not target_exists_in_mast(target, mission):
            continue
        
        if has_lightcurve(target, mission, min_rows=min_rows):
            positives.append((target, mission))
            pos_names_set.add(target)
        
        if len(positives) >= desired_pos * 2:  # Extra topla
            break
    
    positives = positives[:desired_pos]
    print(f"  ✓ {len(positives)} pozitif doğrulandı")

    print(f"\n[3/4] Negatif örnekler toplanıyor...")
    print(f"  Kepler hedefi: {neg_kepler_target}")
    print(f"  TESS hedefi: {neg_tess_target}")
    
    neg_kepler = sample_negatives("Kepler", neg_kepler_target, pos_names_set, seed=seed + 1)
    neg_tess = sample_negatives("TESS", neg_tess_target, pos_names_set, seed=seed + 2)
    
    print(f"  ✓ Kepler negatif: {len(neg_kepler)}")
    print(f"  ✓ TESS negatif: {len(neg_tess)}")

    # Tüm satırları birleştir
    rows = [(t, m, 1) for (t, m) in positives] + [(t, m, 0) for (t, m) in (neg_kepler + neg_tess)]
    random.shuffle(rows)
    rows = rows[:total]

    print(f"\n[4/4] {len(rows)} satır doğrulandı, CSV yazılıyor...")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["target", "mission", "label"])
        for row in rows:
            writer.writerow(row)

    n_pos = sum(1 for _, _, y in rows if y == 1)
    n_neg = len(rows) - n_pos
    
    print("\n" + "="*60)
    print("✅ TAMAMLANDI!")
    print("="*60)
    print(f"Dosya: {out_csv}")
    print(f"Toplam: {len(rows)}")
    print(f"Pozitif: {n_pos} ({n_pos/len(rows)*100:.1f}%)")
    print(f"Negatif: {n_neg} ({n_neg/len(rows)*100:.1f}%)")
    print("\nİlk 10 örnek:")
    for i, r in enumerate(rows[:10], 1):
        label_str = "Pozitif" if r[2] == 1 else "Negatif"
        print(f"  {i}. {r[0]:<20} | {r[1]:<10} | {label_str}")
    print("="*60)


def parse_args():
    """Komut satırı argümanlarını parse et"""
    ap = argparse.ArgumentParser(
        description="MAST doğrulamalı Kepler/TESS hedef listesi üretir.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--total", type=int, default=1500, 
                   help="Toplam hedef sayısı")
    ap.add_argument("--pos_fraction", type=float, default=0.5, 
                   help="Pozitif oranı (0-1)")
    ap.add_argument("--seed", type=int, default=42, 
                   help="Rastgele tohum")
    ap.add_argument("--out_csv", type=str, default="targets.csv", 
                   help="Çıktı dosyası yolu")
    ap.add_argument("--min_rows", type=int, default=50, 
                   help="Minimum ışık eğrisi uzunluğu")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_targets(
        total=args.total,
        pos_fraction=args.pos_fraction,
        seed=args.seed,
        out_csv=Path(args.out_csv),
        min_rows=args.min_rows,
    )

