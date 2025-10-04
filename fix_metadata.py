"""
CSV dosyasindaki tirnakları kaldir ve duzgun formata cevir
"""
import csv

input_file = 'data/metadata/metadata1500.csv'
output_file = 'data/metadata/metadata1500_fixed.csv'
backup_file = 'data/metadata/metadata1500_backup.csv'

print("CSV duzeltme basladi...")

# Backup olustur
import shutil
shutil.copy(input_file, backup_file)
print(f"Yedek olusturuldu: {backup_file}")

# Dosyayi oku ve tirnaklari kaldir
with open(input_file, 'r', encoding='utf-8') as f_in:
    with open(output_file, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.writer(f_out, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        
        for line_num, line in enumerate(f_in, 1):
            # Bas ve sondaki tirnaklari kaldir
            line = line.strip()
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            
            # Virgule gore ayir
            values = line.split(',')
            writer.writerow(values)

print(f"Duzeltilmis dosya: {output_file}")

# Kontrol
import pandas as pd
df = pd.read_csv(output_file)
print(f"\nKontrol:")
print(f"  Satir sayisi: {len(df)}")
print(f"  Kolon sayisi: {len(df.columns)}")
print(f"  Kolonlar: {list(df.columns)}")
print(f"\nIlk 3 satir:")
print(df.head(3))

# Eger basarili ise, orijinal dosyayi degistir
if len(df.columns) == 11:
    shutil.copy(output_file, input_file)
    print(f"\nBASARILI! Orijinal dosya guncellendi: {input_file}")
    print(f"Yedek: {backup_file}")
else:
    print(f"\nUYARI: Kolon sayisi yanlis ({len(df.columns)}), orijinal dosya korundu")

