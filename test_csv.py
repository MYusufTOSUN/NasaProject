import pandas as pd

# Dosyayi oku
try:
    df = pd.read_csv('data/metadata/metadata1500.csv', quotechar='"')
    print(f"BASARILI! Toplam satir: {len(df)}")
    print(f"Kolonlar: {list(df.columns)}")
    print("\nIlk 3 satir:")
    print(df.head(3))
    print("\nSon 3 satir:")
    print(df.tail(3))
except Exception as e:
    print(f"HATA: {e}")
