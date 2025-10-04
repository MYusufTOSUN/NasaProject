# 🪐 Ekzoplanet Tespit Sistemi

YOLOv8 tabanlı derin öğrenme ile ışık eğrilerinden ekzoplanet transit sinyallerini tespit eden tam otomatik sistem.

## 📋 Özellikler

- **Metadata-First Yaklaşım**: Period/t0/duration bilgileri varsa doğrudan kullanılır, eksikse MAST + BLS ile otomatik hesaplanır
- **YOLOv8 Sınıflandırma**: Faz-katlanmış ışık eğrisi görsellerinden transit tespiti
- **Web Arayüzü**: Flask backend + dinamik frontend ile gerçek zamanlı analiz
- **Kapsamlı Değerlendirme**: Confusion matrix, ROC eğrisi, detaylı metrikler
- **Kepler & TESS Desteği**: Her iki misyon için MAST entegrasyonu

## 🚀 Hızlı Başlangıç

### 1. Kurulum

```bash
# Sanal ortam oluştur
python -m venv venv

# Aktif et (Windows)
venv\Scripts\activate

# Bağımlılıkları yükle
pip install -r requirements.txt
```

### 2. Metadata Hazırlığı

`data/metadata/metadata1500.csv` dosyasını manuel olarak yerleştirin. Beklenen kolonlar:
- `target`, `mission`, `period`, `t0`, `duration`, `depth_ppm`, `snr`, `ra`, `dec`, `mag`, `label`, `archive_url`

### 3. Grafik Üretimi

```bash
# Deneme için ilk 30 hedef
python make_graphs_from_metadata.py --metadata data/metadata/metadata1500.csv --limit 30

# Tüm hedefleri işle
python make_graphs_from_metadata.py --metadata data/metadata/metadata1500.csv
```

### 4. Dataset Oluşturma

```bash
python scripts/01_build_index.py
python scripts/02_split_build_dataset.py
```

### 5. Model Eğitimi

```bash
python scripts/03_train_yolov8_cls.py
```

### 6. Değerlendirme

```bash
python scripts/05_batch_score_all.py
python scripts/evaluate_model.py
```

## 🌐 Web UI Kullanımı

### Manuel Başlatma

```bash
python app/exoplanet_detector.py
```

### Otomatik Başlatma (Windows)

```bash
START_WEB_UI.bat
```

Tarayıcıda `http://localhost:5000` adresini açın.

### Kullanım Seçenekleri

1. **Hedef Adı ile Analiz**: 
   - Target Name: `Kepler-10`
   - Mission: `auto` (otomatik tespit)
   - Analyze butonuna tıklayın

2. **Görsel Yükleme**:
   - PNG/JPEG formatında faz-katlanmış ışık eğrisi görseli yükleyin
   - "Run on Image" ile tahmin alın
   - "Explain the Model" ile saliency haritası görün

## 📊 Özellikler

- **Light Curve Playback**: Ham ve faz-katlanmış görsellerin animasyonlu geçişi
- **Confidence Meter**: 0-100 arası güven skorunu gauge ile görselleştirme
- **Mission Badge**: Kepler (🔵) ve TESS (🔴) rozetleri
- **Discovery Card**: NASA onaylı gezegenler için detaylı bilgi kartı
- **Similar Planets**: Depth_ppm bazlı benzer gezegen önerileri
- **Dark/Light Theme**: Dinamik tema değiştirme

## 🔧 Ek Araçlar

### Tek Yıldız Hızlı Analiz

```bash
python 01_download_clean_bls_fast.py --target "Kepler-10" --mission Kepler
```

### Hızlı Klasör Testi

```bash
python scripts/04_predict_folder.py
```

## 📁 Proje Yapısı

```
KapsülProje/
├── app/                          # Flask uygulaması
│   ├── exoplanet_detector.py
│   ├── templates/
│   │   └── index.html
│   └── static/temp/
├── scripts/                      # Yardımcı scriptler
│   ├── 01_build_index.py
│   ├── 02_split_build_dataset.py
│   ├── 03_train_yolov8_cls.py
│   ├── 04_predict_folder.py
│   ├── 05_batch_score_all.py
│   └── evaluate_model.py
├── data/
│   ├── metadata/                 # metadata1500.csv buraya
│   └── plots/                    # Train/val/test görselleri
├── graphs/
│   ├── images/                   # Üretilen faz-katlanmış görseller
│   └── labels/                   # YOLO etiket dosyaları
├── models/                       # Eğitilmiş model ağırlıkları
├── evaluation_results/           # Metrikler ve grafikler
├── make_graphs_from_metadata.py  # Ana grafik üretici
└── 01_download_clean_bls_fast.py # Tek hedef CLI aracı
```

## 📝 Notlar

- `targets.csv` kullanılmıyor, tüm akış `metadata1500.csv` tabanlı
- Metadata'da period/t0/duration varsa BLS atlanır (hızlı mod)
- MAST bağlantısı yavaş olabilir, sabırlı olun
- GPU varsa YOLO eğitimi çok daha hızlı olacaktır

## 📚 Dokümantasyon

- [KULLANIM_KILAVUZU.md](KULLANIM_KILAVUZU.md) - Detaylı Türkçe kılavuz
- [MODEL_PERFORMANCE_REPORT.md](MODEL_PERFORMANCE_REPORT.md) - Model performans raporu

## 🤝 Katkıda Bulunma

Bu proje eğitim ve araştırma amaçlıdır. İyileştirme önerileri için issue açabilirsiniz.

## 📄 Lisans

MIT License

