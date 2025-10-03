# scripts/check_negatives.py
import os
import re
from pathlib import Path
from glob import glob

# Negatif hedeflerin olduğu metin/CSV'den KIC listesi okunduğunu varsayıyorum.
# Eğer CSV ise burayı kendi okuma şekline göre düzenle.
NEGATIVE_IDS_TXT = r"C:\Users\tosun\PycharmProjects\NasaProject\data\negatives_ids.txt"
# Görsellerin (veya üretilmiş PNG/JPG'lerin) kök klasörü:
IMAGES_ROOT = r"C:\Users\tosun\PycharmProjects\NasaProject\data\images"

def load_kic_ids(path):
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Satırda "kic12345" gibi bir şey varsa rakamları çek
            m = re.search(r'(\d+)', line.lower())
            if m:
                ids.append(int(m.group(1)))
    return ids

def search_image_for_kic(kic_int, root):
    """
    Bu fonksiyon farklı isim kalıplarını dener:
    - kic123, KIC_00000123, kplr00000123, vs.
    - .png/.jpg
    - Tüm alt klasörler
    """
    pad9 = f"{kic_int:09d}"

    patterns = [
        # kplr + 9 haneli
        f"**/kplr{pad9}*.png",
        f"**/kplr{pad9}*.jpg",
        f"**/kplr{pad9}*.jpeg",

        # KIC_ + 9 haneli
        f"**/KIC_{pad9}*.png",
        f"**/KIC_{pad9}*.jpg",
        f"**/KIC_{pad9}*.jpeg",

        # kic + 9 haneli
        f"**/kic{pad9}*.png",
        f"**/kic{pad9}*.jpg",
        f"**/kic{pad9}*.jpeg",

        # sadece sayıyla başlayan
        f"**/{pad9}*.png",
        f"**/{pad9}*.jpg",
        f"**/{pad9}*.jpeg",

        # 0 dolgusuz varyasyon (diskte yanlış kaydedilmişse)
        f"**/kplr{kic_int}*.png",
        f"**/kplr{kic_int}*.jpg",
        f"**/KIC_{kic_int}*.png",
        f"**/KIC_{kic_int}*.jpg",
        f"**/kic{kic_int}*.png",
        f"**/kic{kic_int}*.jpg",
        f"**/{kic_int}*.png",
        f"**/{kic_int}*.jpg",
    ]

    root = Path(root)
    for pat in patterns:
        hits = list(root.glob(pat))
        if hits:
            return hits  # bir veya birden fazla eşleşme
    return []

def main():
    ids = load_kic_ids(NEGATIVE_IDS_TXT)
    total = len(ids)

    found_count = 0
    missing = []

    for kic in ids:
        hits = search_image_for_kic(kic, IMAGES_ROOT)
        if hits:
            found_count += 1
        else:
            missing.append(kic)

    print(f"Toplam negatif hedef: {total}")
    print(f"Bulunan görseller: {found_count}")
    print(f"Kayıp görseller: {len(missing)}")

    if missing:
        out = Path(IMAGES_ROOT) / "missing_negatives.txt"
        with open(out, "w", encoding="utf-8") as f:
            for k in missing:
                f.write(f"{k:09d}\n")
        print(f"Kayıp liste yazıldı: {out}")

    # İlk 20 kayıp ID'yi önizleme
    if missing:
        preview = ", ".join(f"{m:09d}" for m in missing[:20])
        print("Örnek kayıplar:", preview)

if __name__ == "__main__":
    main()
