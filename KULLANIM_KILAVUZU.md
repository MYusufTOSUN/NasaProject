# NASA Exoplanet Detection Projesi - Kullanım Kılavuzu

## Kurulum

### 1. Sanal Ortam Kurulumu
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Proje Yapısı Kontrolü
```bash
python qc_check.py
```

## Veri Akışı

### 1. Hedef Listesi Hazırlama
- `targets.csv` dosyasını düzenleyin
- Her satır: hedef_adı, görev (Kepler/TESS), etiket (1=exoplanet, 0=yok)
- Toplam 602 satır olmalı

### 2. Ham Veri İndirme ve İşleme
```bash
# Tek hedef için
python 01_download_clean_bls_fast.py --target "Kepler-10" --mission Kepler

# Toplu grafik üretimi
python make_graphs_yolo.py --targets targets.csv --out graphs
```

### 3. Veri Seti Oluşturma
```bash
# İndeks dosyası oluştur
python scripts/01_build_index.py --raw_dir scripts/raw_images --targets targets.csv --out index.csv

# Veri setini böl
python scripts/02_split_build_dataset.py --index index.csv --out data/plots
```

## Veri Bölme Kuralları (Yıldız Bazlı)

- **Yıldız bazlı stratified split**: Aynı yıldız farklı setlere düşmez
- **Train**: %70 (pozitif ve negatif örnekler dengeli)
- **Validation**: %15
- **Test**: %15
- **Sınıf dengesi**: Her sette pozitif/negatif oranı korunur

## Model Eğitimi

### YOLOv8 Sınıflandırma Eğitimi
```bash
python scripts/03_train_yolov8_cls.py --model yolov8n-cls.pt --epochs 200 --imgsz 224 --batch 64 --device 0
```

### Parametreler:
- `--epochs`: Eğitim dönem sayısı (varsayılan: 200)
- `--imgsz`: Görüntü boyutu (varsayılan: 224)
- `--batch`: Batch boyutu (varsayılan: 64)
- `--device`: GPU/CPU (0=GPU, cpu=CPU)

## Model Değerlendirme

### Tahmin Yapma
```bash
python scripts/04_predict_folder.py --weights models/best.pt --input_dir data/plots/test
```

### Model Performansı Değerlendirme
```bash
python scripts/evaluate_model.py --pred_csv evaluation_results/predictions_detail.csv
```

### Çıktılar:
- `summary.csv`: Genel metrikler
- `predictions_detail.csv`: Detaylı tahminler
- `errors_analysis.csv`: Hatalı tahminlerin analizi
- `confusion_matrix.png`: Karışıklık matrisi
- `roc_curve.png`: ROC eğrisi

## Web Arayüzü Kullanımı

### Başlatma
```bash
START_WEB_UI.bat
```
veya
```bash
cd app
python exoplanet_detector.py
```

### Kullanım:
1. Tarayıcıda `http://localhost:5000` açılır
2. Light curve verilerini yükleyin
3. Analiz sonuçlarını görüntüleyin

## Sık Karşılaşılan Hatalar

### 1. Ultralytics Argüman Hatası
**Hata**: `unrecognized arguments`
**Çözüm**: Ultralytics versiyonunu kontrol edin:
```bash
pip install --upgrade ultralytics
```

### 2. MAST Rate-Limit
**Hata**: `HTTP 429 Too Many Requests`
**Çözüm**: İstekler arasında bekleme süresi ekleyin veya daha küçük batch'ler kullanın

### 3. Eksik Model Dosyası
**Hata**: `models/best.pt not found`
**Çözüm**: Önce model eğitimi yapın veya model dosyasını doğru konuma kopyalayın

### 4. Boş Klasör Hatası
**Hata**: `No images found in directory`
**Çözüm**: Veri seti oluşturma adımlarını tekrar çalıştırın

### 5. GPU Bellek Hatası
**Hata**: `CUDA out of memory`
**Çözüm**: Batch boyutunu küçültün veya CPU kullanın:
```bash
--batch 32 --device cpu
```

### 6. Port Kullanımda Hatası
**Hata**: `Address already in use`
**Çözüm**: Farklı port kullanın:
```bash
python -m flask run --port 5001
```

## Performans İpuçları

1. **GPU Kullanımı**: NVIDIA GPU varsa `--device 0` kullanın
2. **Batch Boyutu**: GPU belleğine göre ayarlayın (16, 32, 64)
3. **Görüntü Boyutu**: Daha büyük görüntüler daha iyi sonuç verir ama daha yavaştır
4. **Epoch Sayısı**: Overfitting'i önlemek için early stopping kullanın

## Sorun Giderme

### Log Dosyaları Kontrol
- Eğitim logları: `runs/classify/` klasöründe
- Flask logları: Terminal çıktısında
- Hata logları: `evaluation_results/` klasöründe

### Veri Kalitesi Kontrol
```bash
python qc_check.py
```

### Model Performansı Kontrol
`MODEL_PERFORMANCE_REPORT.md` dosyasını inceleyin.
