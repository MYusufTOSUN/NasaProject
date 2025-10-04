# 🪐 Ekzoplanet Tespit Sistemi - Proje Özeti

## 📦 Proje İçeriği

Bu proje, YOLOv8 derin öğrenme modeli ile Kepler ve TESS misyon verilerinden ekzoplanet transit sinyallerini tespit eden tam otomatik bir sistemdir.

## 🎯 Temel Özellikler

### 1. Metadata-First Yaklaşım
- Period/t0/duration bilgileri metadata'dan alınır (hızlı)
- Eksik bilgiler için BLS algoritması devreye girer (yavaş ama otomatik)
- MAST API ile Kepler ve TESS ışık eğrilerini otomatik indirir

### 2. YOLOv8 Sınıflandırma
- Pre-trained `yolov8n-cls.pt` ile transfer learning
- 224x224 faz-katlanmış ışık eğrisi görselleri
- İkili sınıflandırma: positive (transit var) / negative (transit yok)

### 3. Premium Web Arayüzü
- Flask backend + dinamik JavaScript frontend
- **Light Curve Playback**: Ham ve faz-katlanmış görsellerin animasyonlu geçişi
- **Confidence Meter**: 0-100 arası gauge göstergesi
- **Mission Badge**: Kepler (🔵) ve TESS (🔴) rozetleri
- **Discovery Card**: NASA onaylı gezegenler için detaylı bilgi kartı
- **Similar Planets**: Depth_ppm bazlı benzer gezegen önerileri
- **Saliency Map**: Model açıklaması (occlusion-based)
- Dark/Light tema desteği

### 4. Kapsamlı Pipeline
- Otomatik grafik üretimi
- Stratified dataset bölme (aynı target aynı split'te)
- Model eğitimi ve early stopping
- Confusion matrix, ROC curve, detaylı metrikler

## 📁 Dosya Yapısı

```
KapsülProje/
│
├── 📄 README.md                          # Genel bilgilendirme
├── 📄 KULLANIM_KILAVUZU.md              # Detaylı Türkçe kılavuz
├── 📄 MODEL_PERFORMANCE_REPORT.md       # Model performans raporu
├── 📄 requirements.txt                   # Python bağımlılıkları
├── 📄 .gitignore                         # Git ignore kuralları
├── 📄 START_WEB_UI.bat                   # Windows için otomatik başlatıcı
│
├── 📄 make_graphs_from_metadata.py       # Ana grafik üretici (metadata-first)
├── 📄 01_download_clean_bls_fast.py      # Tek hedef CLI aracı
│
├── 📂 app/
│   ├── 📄 exoplanet_detector.py          # Flask backend API
│   ├── 📂 templates/
│   │   └── 📄 index.html                 # Premium frontend
│   └── 📂 static/temp/                   # Geçici dosyalar
│
├── 📂 scripts/
│   ├── 📄 01_build_index.py              # Görsel indeksleyici
│   ├── 📄 02_split_build_dataset.py      # Dataset bölme + data.yaml
│   ├── 📄 03_train_yolov8_cls.py         # YOLOv8 eğitim scripti
│   ├── 📄 04_predict_folder.py           # Hızlı klasör testi
│   ├── 📄 05_batch_score_all.py          # Toplu skor üretimi
│   └── 📄 evaluate_model.py              # Model değerlendirme
│
├── 📂 data/
│   ├── 📂 metadata/
│   │   └── 📄 metadata1500.csv           # MANUEL EKLE (kullanıcı)
│   ├── 📂 plots/                         # Train/val/test görselleri
│   └── 📄 data.yaml                      # YOLOv8 config
│
├── 📂 graphs/
│   ├── 📂 images/                        # Üretilen faz-katlanmış görseller
│   └── 📂 labels/                        # YOLO label dosyaları
│
├── 📂 models/
│   └── 📄 best.pt                        # En iyi eğitilmiş model
│
└── 📂 evaluation_results/
    ├── 📄 predictions_detail.csv         # Tüm tahminler
    ├── 📄 summary.csv                    # Metrikler özeti
    ├── 📄 confusion_matrix.png           # Confusion matrix grafiği
    └── 📄 roc_curve.png                  # ROC eğrisi grafiği
```

## 🚀 Hızlı Başlangıç

### Adım 1: Kurulum
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Adım 2: Metadata Hazırlığı
`data/metadata/metadata1500.csv` dosyasını yerleştirin.

### Adım 3: Grafik Üretimi (Test)
```bash
python make_graphs_from_metadata.py --metadata data/metadata/metadata1500.csv --limit 30
```

### Adım 4: Dataset Oluşturma
```bash
python scripts/01_build_index.py
python scripts/02_split_build_dataset.py
```

### Adım 5: Model Eğitimi
```bash
python scripts/03_train_yolov8_cls.py
```

### Adım 6: Değerlendirme
```bash
python scripts/05_batch_score_all.py
python scripts/evaluate_model.py
```

### Adım 7: Web UI
```bash
START_WEB_UI.bat
# veya
python app/exoplanet_detector.py
```

Tarayıcıda: `http://localhost:5000`

## 🎨 Web UI Özellikleri

### Hedef Adı ile Analiz
1. Target Name: `Kepler-10`
2. Mission: `auto`
3. **Analyze** butonuna tıklayın
4. Sonuçları görüntüleyin:
   - Tahmin (positive/negative)
   - Güven skoru (gauge göstergesi)
   - Light curve animasyonu (hover ile başlar)
   - Discovery Card (NASA onaylıysa)
   - Benzer gezegenler

### Görsel Yükleme ile Analiz
1. PNG/JPEG formatında faz-katlanmış görsel yükleyin
2. **Görseli Analiz Et** butonuna tıklayın
3. **Modeli Açıkla** ile saliency map görüntüleyin

## 📊 Beklenen Metrikler

| Metrik      | Hedef Değer |
|-------------|-------------|
| Accuracy    | >90%        |
| Precision   | >90%        |
| Recall      | >90%        |
| ROC AUC     | >0.95       |

## ⚙️ Sistem Gereksinimleri

### Minimum
- CPU: 4 core
- RAM: 8GB
- Disk: 5GB boş alan
- Python: 3.8+

### Önerilen
- CPU: 8+ core
- RAM: 16GB
- GPU: NVIDIA (CUDA destekli)
- Disk: 10GB+ boş alan
- Python: 3.10+

## 🔧 Önemli Notlar

### ⚠️ Dikkat Edilecekler

1. **`targets.csv` KULLANILMIYOR**: Tüm akış `metadata1500.csv` tabanlıdır.

2. **BLS Yavaştır**: Period bilgisi eksikse tek hedef ~1-3 dakika sürer.

3. **MAST Bağlantısı**: İyi internet gerekir, bazen yavaş olabilir.

4. **GPU Önemli**: Eğitim CPU'da çok yavaş (saatler sürebilir).

5. **Metadata Eksikliği**: Label eksik olan görseller "negative" olarak işaretlenir.

### ✅ İyi Pratikler

✅ İlk defa kullanıyorsanız `--limit 10` ile test edin

✅ Metadata'da period bilgilerini mümkün olduğunca doldurun

✅ Eğitim öncesi `data.yaml` dosyasını kontrol edin

✅ Eğitim sırasında tensorboard ile metrikleri izleyin

✅ Web UI'ı production'da kullanmayın (debug=True)

## 🎯 Hız Profilleri

### Hızlı Deneme (1-2 saat)
```bash
# 30 hedef + kısa eğitim
python make_graphs_from_metadata.py --metadata data/metadata/metadata1500.csv --limit 30
# Dataset + Eğitim (epochs=50)
```

### Dengeli (3-8 saat)
```bash
# Tüm hedefler, BLS sadece gerektiğinde
python make_graphs_from_metadata.py --metadata data/metadata/metadata1500.csv
# Normal eğitim (epochs=200)
```

### Tam Veri (10-20 saat)
```bash
# Tüm hedefler + BLS her yerde
python make_graphs_from_metadata.py --metadata data/metadata/metadata1500.csv
# Uzun eğitim + augmentation
```

## 📚 Dokümantasyon

- **[README.md](README.md)**: Genel bakış ve hızlı başlangıç
- **[KULLANIM_KILAVUZU.md](KULLANIM_KILAVUZU.md)**: Detaylı adım adım kılavuz
- **[MODEL_PERFORMANCE_REPORT.md](MODEL_PERFORMANCE_REPORT.md)**: Model performans raporu

## 🐛 Sorun Giderme

### MAST Bağlantı Hatası
- İnternet bağlantınızı kontrol edin
- VPN kapatmayı deneyin
- `--limit` ile küçük subset test edin

### BLS Çok Yavaş
- Metadata'da period bilgilerini ekleyin
- `--no-bls` ile sadece metadata'lı hedefleri işleyin

### Model Yüklenemiyor
- Önce eğitimi tamamlayın: `scripts/03_train_yolov8_cls.py`
- `models/best.pt` dosyasının varlığını kontrol edin

### GPU Tanınmıyor
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
False ise CUDA kurulumunu kontrol edin.

## 🔬 İleri Düzey Özellikler

### Grad-CAM Entegrasyonu
`app/exoplanet_detector.py` içinde `create_saliency_map()` fonksiyonunu Grad-CAM ile değiştirin.

### Multi-GPU Eğitim
```python
# scripts/03_train_yolov8_cls.py içinde
params['device'] = '0,1,2,3'  # 4 GPU
```

### Hyperparameter Tuning
YOLOv8 `.tune()` metodunu kullanın:
```python
model.tune(data='data/data.yaml', iterations=100)
```

### API Rate Limiting
Flask'a `flask-limiter` ekleyin:
```python
from flask_limiter import Limiter
limiter = Limiter(app, key_func=lambda: request.remote_addr)
```

## 📝 Kontrol Listesi

### Kurulum
- [ ] Python 3.8+ yüklü
- [ ] Sanal ortam oluşturuldu
- [ ] Bağımlılıklar yüklendi

### Veri
- [ ] `metadata1500.csv` yerleştirildi
- [ ] Kolonlar kontrol edildi
- [ ] Label değerleri doğru

### Pipeline
- [ ] Grafik üretimi tamamlandı
- [ ] Dataset oluşturuldu
- [ ] Model eğitildi
- [ ] Değerlendirme yapıldı

### Web UI
- [ ] Flask başlatıldı
- [ ] Hedef analizi test edildi
- [ ] Görsel yükleme test edildi
- [ ] Tüm özellikler çalışıyor

## 🤝 Katkıda Bulunma

Bu proje eğitim ve araştırma amaçlıdır. İyileştirme önerileri için issue açabilirsiniz.

## 📄 Lisans

MIT License

---

**Proje Versiyonu**: 1.0  
**Son Güncelleme**: 2025-10-04  
**Yazar**: Ekzoplanet Tespit Sistemi Ekibi

