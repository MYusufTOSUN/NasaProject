import pandas as pd

# Farkli delimiter'lari dene
print("=== Delimiter testi ===")

# 1. Virgul
try:
    df = pd.read_csv('data/metadata/metadata1500.csv', sep=',')
    print(f"Virgul: {len(df.columns)} kolon - {list(df.columns)[:3]}")
except Exception as e:
    print(f"Virgul HATA: {e}")

# 2. Tab
try:
    df = pd.read_csv('data/metadata/metadata1500.csv', sep='\t')
    print(f"Tab: {len(df.columns)} kolon - {list(df.columns)[:3]}")
except Exception as e:
    print(f"Tab HATA: {e}")

# 3. Dosyayi satirlara ayir ve ilk satiri incele
with open('data/metadata/metadata1500.csv', 'r', encoding='utf-8') as f:
    first_line = f.readline().strip()
    print(f"\nIlk satir uzunlugu: {len(first_line)}")
    print(f"Ilk 100 karakter: {repr(first_line[:100])}")
    print(f"Virgul sayisi: {first_line.count(',')}")

