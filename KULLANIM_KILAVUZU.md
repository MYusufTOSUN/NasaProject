# 📖 Ekzoplanet Tespit Sistemi - Detaylı Kullanım Kılavuzu

## İçindekiler

1. [Kurulum](#1-kurulum)
2. [Veri Hazırlığı](#2-veri-hazırlığı)
3. [Grafik Üretimi](#3-grafik-üretimi)
4. [Dataset Oluşturma](#4-dataset-oluşturma)
5. [Model Eğitimi](#5-model-eğitimi)
6. [Model Değerlendirme](#6-model-değerlendirme)
7. [Web UI Kullanımı](#7-web-ui-kullanımı)
8. [Sorun Giderme](#8-sorun-giderme)
9. [Kontrol Listesi](#9-kontrol-listesi)

---

## 1. Kurulum

### 1.1 Gereksinimler

- Python 3.8 veya üzeri
- 8GB+ RAM (önerilen: 16GB)
- GPU (opsiyonel, ancak eğitim için şiddetle önerilir)
- İnternet bağlantısı (MAST API erişimi için)

### 1.2 Sanal Ortam Oluşturma

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 1.3 Bağımlılıkları Yükleme

```bash
pip install -r requirements.txt
```

**Not**: İlk yükleme 5-10 dakika sürebilir.

### 1.4 Kurulum Doğrulama

```bash
python -c "import torch, ultralytics, lightkurve; print('✓ Kurulum başarılı!')"
```

---

## 2. Veri Hazırlığı

### 2.1 Metadata Dosyası

`data/metadata/metadata1500.csv` dosyasını manuel olarak yerleştirin.

**Beklenen Kolonlar:**

| Kolon        | Açıklama                           | Gerekli mi? |
|--------------|------------------------------------|-------------|
| target       | Hedef yıldız adı (örn: Kepler-10)  | ✓ Evet      |
| mission      | Misyon adı (Kepler/TESS)           | ✓ Evet      |
| period       | Orbital period (gün)               | ✗ Hayır*    |
| t0           | Transit zamanı (BKJD)              | ✗ Hayır*    |
| duration     | Transit süresi (gün)               | ✗ Hayır*    |
| depth_ppm    | Transit derinliği (ppm)            | ✗ Hayır     |
| snr          | Sinyal-gürültü oranı               | ✗ Hayır     |
| ra           | Sağ açıklık                        | ✗ Hayır     |
| dec          | Sapma                              | ✗ Hayır     |
| mag          | Magnitude                          | ✗ Hayır     |
| label        | Etiket (positive/negative)         | ✓ Evet      |
| archive_url  | NASA Archive linki                 | ✗ Hayır     |

**\*Not**: `period`, `t0`, `duration` yoksa BLS ile otomatik hesaplanır (yavaş).

### 2.2 Örnek Satır

```csv
target,mission,period,t0,duration,depth_ppm,snr,ra,dec,mag,label,archive_url
Kepler-10,Kepler,0.8374907,131.51217,0.0581,461.2,89.3,285.679,50.241,10.96,positive,https://exoplanetarchive.ipac.caltech.edu/...
```

---

## 3. Grafik Üretimi

### 3.1 Hızlı Deneme (İlk 30 Hedef)

```bash
 ```

**Beklenen Süre**: Period bilgileri varsa ~5-10 dakika, BLS gerektiriyorsa ~30-60 dakika.

### 3.2 Tüm Dataset

```bash
python make_graphs_from_metadata.py --metadata data/metadata/metadata1500.csv
```

**Beklenen Süre**: Period bilgileri varsa ~30-60 dakika, BLS gerektiriyorsa ~5-10 saat.

### 3.3 BLS Olmadan (Sadece Metadata'lı Hedefler)

```bash
python make_graphs_from_metadata.py --metadata data/metadata/metadata1500.csv --no-bls
```

Bu mod yalnızca metadata'da `period`, `t0`, `duration` bilgisi olan hedefleri işler.

### 3.4 Çıktılar

- **Görseller**: `graphs/images/`
  - Format: `<target>_<mission>_phase.png`
  - Örnek: `Kepler-10_Kepler_phase.png`

- **Etiketler**: `graphs/labels/`
  - Format: `<target>_<mission>_phase.txt`
  - YOLO format (şu an kullanılmıyor, ileride object detection için)

### 3.5 İpuçları

✅ **En Hızlı Yol**: Metadata'da tüm period bilgileri varsa BLS atlanır.

⚠️ **BLS Yavaş**: Tek bir hedef için 1-3 dakika sürebilir.

🌐 **MAST Yavaş**: İnternet hızınıza bağlı, bazı hedefler zaman aşımına uğrayabilir.

---

## 4. Dataset Oluşturma

### 4.1 İndeks Oluşturma

```bash
python scripts/01_build_index.py
```

**Çıktı**: `index.csv` (tüm görsellerin listesi)

### 4.2 Dataset Bölme ve YOLOv8 Yapısı Oluşturma

```bash
python scripts/02_split_build_dataset.py
```

**İşlemler**:
1. `index.csv` ile `metadata1500.csv` birleştirilir
2. Aynı target aynı split'te kalacak şekilde bölme
3. Stratified split: %70 train, %15 val, %15 test
4. Görseller `data/plots/<split>/<label>/` altına kopyalanır
5. `data/data.yaml` oluşturulur

**Beklenen Süre**: ~1-2 dakika

**Çıktılar**:
```
data/
├── plots/
│   ├── train/
│   │   ├── positive/
│   │   └── negative/
│   ├── val/
│   │   ├── positive/
│   │   └── negative/
│   └── test/
│       ├── positive/
│       └── negative/
└── data.yaml
```

---

## 5. Model Eğitimi

### 5.1 YOLOv8 Eğitimi

```bash
python scripts/03_train_yolov8_cls.py
```

**Parametreler**:
- Model: `yolov8n-cls.pt` (pre-trained)
- Image size: 224x224
- Batch size: 64
- Epochs: 200
- Early stopping: Patience 50

**Beklenen Süre**:
- GPU (CUDA): ~30-60 dakika
- CPU: ~5-10 saat

**Çıktılar**:
- `runs/classify/exoplanet_transit/` (eğitim logları, grafikler)
- `models/best.pt` (en iyi model)

### 5.2 Eğitim Sırasında İzleme

Eğitim sırasında şu metrikler terminalde görünür:
- Loss (train/val)
- Accuracy (top1)
- Learning rate

### 5.3 İpuçları

🚀 **GPU Yoksa**: Batch size'ı azaltın (32 veya 16).

📊 **Tensorboard**: `tensorboard --logdir runs/classify/` ile detaylı grafikleri görüntüleyin.

---

## 6. Model Değerlendirme

### 6.1 Hızlı Klasör Testi

```bash
python scripts/04_predict_folder.py
```

İlk 20 görseli test eder ve sonuçları terminalde gösterir.

### 6.2 Toplu Skor Üretimi

```bash
python scripts/05_batch_score_all.py
```

**Çıktı**: `evaluation_results/predictions_detail.csv`

### 6.3 Değerlendirme (Metrikler + Grafikler)

```bash
python scripts/evaluate_model.py
```

**Çıktılar**:
- `evaluation_results/summary.csv` (metrikler)
- `evaluation_results/confusion_matrix.png`
- `evaluation_results/roc_curve.png`

**Metrikler**:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC AUC

---

## 7. Web UI Kullanımı

### 7.1 Otomatik Başlatma (Windows)

```bash
START_WEB_UI.bat
```

Bu script:
1. Sanal ortam oluşturur (yoksa)
2. Bağımlılıkları yükler (yoksa)
3. Flask'ı başlatır

### 7.2 Manuel Başlatma

```bash
python app/exoplanet_detector.py
```

Tarayıcıda açın: `http://localhost:5000`

### 7.3 Hedef Adı ile Analiz

1. **Target Name** alanına hedef adını girin (örn: `Kepler-10`)
2. **Mission** seçin (auto önerilir)
3. **Analyze** butonuna tıklayın
4. Sonuçlar sağ panelde görünür:
   - Tahmin sonucu (positive/negative)
   - Güven skoru (0-100%)
   - Confidence Meter (gauge)
   - Light Curve Playback (animasyonlu)
   - Discovery Card (NASA onaylıysa)
   - Similar Planets (benzer gezegenler)

### 7.4 Görsel Yükleme ile Analiz

1. **Görsel Yükle** bölümüne tıklayın
2. PNG/JPEG formatında faz-katlanmış ışık eğrisi görseli seçin
3. **Görseli Analiz Et** butonuna tıklayın
4. **Modeli Açıkla** butonu ile saliency map görebilirsiniz

### 7.5 UI Özellikleri

#### Light Curve Playback
- Hover ile animasyon başlar
- Ham ve faz-katlanmış görseller arası 2 saniyelik geçiş

#### Confidence Meter
- Yarım daire gauge
- Renkler:
  - 🔴 Kırmızı: <60%
  - 🟠 Turuncu: 60-90%
  - 🟢 Yeşil: >90%

#### Mission Badge
- 🔵 Mavi: Kepler
- 🔴 Kırmızı: TESS

#### Dark/Light Theme
- Sağ üst köşeden tema değiştirin

---

## 8. Sorun Giderme

### 8.1 MAST Bağlantı Hataları

**Semptom**: `Timeout` veya `Connection error`

**Çözüm**:
- İnternet bağlantınızı kontrol edin
- VPN kullanıyorsanız kapatmayı deneyin
- `--limit` ile küçük bir subset test edin

### 8.2 BLS Çok Yavaş

**Semptom**: BLS adımı saatlerce sürüyor

**Çözüm**:
- Metadata'da period bilgileri eksiksiz olmalı
- `--no-bls` bayrağı ile sadece metadata'lı hedefleri işleyin
- Period aralığını daraltın (kod içinde `period_min`, `period_max`)

### 8.3 GPU Tanınmıyor

**Semptom**: `device: cpu` görünüyor

**Çözüm**:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
False ise CUDA kurulumunu kontrol edin.

### 8.4 Model Yüklenemiyor

**Semptom**: `Model bulunamadı: models/best.pt`

**Çözüm**:
- Önce eğitimi tamamlayın: `scripts/03_train_yolov8_cls.py`
- `runs/classify/exoplanet_transit/weights/best.pt` varsa manuel kopyalayın

### 8.5 Metadata Okunamıyor

**Semptom**: `Metadata okunamadı`

**Çözüm**:
- `data/metadata/metadata1500.csv` dosyasının varlığını kontrol edin
- Dosya encoding'i UTF-8 olmalı
- Gerekli kolonlar (`target`, `mission`, `label`) mevcut mu?

---

## 9. Kontrol Listesi

### Kurulum
- [ ] Python 3.8+ yüklü
- [ ] Sanal ortam oluşturuldu ve aktif edildi
- [ ] `requirements.txt` bağımlılıkları yüklendi
- [ ] Kurulum doğrulandı

### Veri Hazırlığı
- [ ] `data/metadata/metadata1500.csv` yerleştirildi
- [ ] Metadata kolonları kontrol edildi
- [ ] Label değerleri (positive/negative) doğru

### Grafik Üretimi
- [ ] Deneme için `--limit 30` ile test edildi
- [ ] Tüm hedefler için grafik üretimi tamamlandı
- [ ] `graphs/images/` klasöründe görseller var

### Dataset Oluşturma
- [ ] `index.csv` oluşturuldu
- [ ] Dataset bölme tamamlandı
- [ ] `data/plots/` klasöründe train/val/test görselleri var
- [ ] `data/data.yaml` oluşturuldu

### Model Eğitimi
- [ ] YOLOv8 eğitimi tamamlandı
- [ ] `models/best.pt` mevcut
- [ ] Eğitim grafikleri incelendi

### Değerlendirme
- [ ] Toplu skor üretimi yapıldı
- [ ] `evaluation_results/predictions_detail.csv` oluşturuldu
- [ ] Confusion matrix ve ROC curve grafikleri oluşturuldu
- [ ] Metrikler tatmin edici (Accuracy >90%?)

### Web UI
- [ ] Flask başlatıldı
- [ ] `http://localhost:5000` açıldı
- [ ] Hedef adı ile analiz test edildi
- [ ] Görsel yükleme ile analiz test edildi
- [ ] Tüm UI özellikleri (playback, gauge, card) çalışıyor

---

## 10. Hız Profilleri

### 10.1 Hızlı Deneme Profili
**Amaç**: Sistemi hızlıca test etmek

```bash
# 1. İlk 30 hedef (metadata'da period varsa)
python make_graphs_from_metadata.py --metadata data/metadata/metadata1500.csv --limit 30

# 2. Dataset oluştur
python scripts/01_build_index.py
python scripts/02_split_build_dataset.py

# 3. Kısa eğitim (epochs=50)
# scripts/03_train_yolov8_cls.py içinde epochs=50 yapın
python scripts/03_train_yolov8_cls.py

# Toplam süre: ~1-2 saat (GPU)
```

### 10.2 Dengeli Profil
**Amaç**: Tüm veriyi kullan, BLS sadece gerektiğinde

```bash
# Metadata'da period bilgisi eksik olanlar için BLS kullan
python make_graphs_from_metadata.py --metadata data/metadata/metadata1500.csv

# Dataset + Eğitim normal
# Toplam süre: ~3-8 saat (GPU + iyi internet)
```

### 10.3 Zor Ağ Profili
**Amaç**: İnternet yavaş/sorunlu

```bash
# Tek hedef CLI aracı ile test
python 01_download_clean_bls_fast.py --target "Kepler-10" --mission Kepler

# BLS'i atla, sadece metadata'lı hedefleri işle
python make_graphs_from_metadata.py --metadata data/metadata/metadata1500.csv --no-bls
```

---

## 11. Önemli Notlar

### ⚠️ Dikkat Edilecekler

1. **`targets.csv` Kullanılmıyor**: Tüm akış `metadata1500.csv` tabanlıdır.

2. **BLS Süresi**: Tek bir hedef için 1-3 dakika. 1000 hedef için ~20-50 saat.

3. **MAST Rate Limiting**: Çok hızlı istek atarsanız geçici olarak engellenebilirsiniz.

4. **GPU Önemli**: Eğitim CPU'da çok yavaş olacaktır.

5. **Disk Alanı**: Görseller ~500MB-2GB arası yer kaplayabilir.

### 🎯 En İyi Pratikler

✅ İlk defa kullanıyorsanız `--limit 10` ile başlayın

✅ Metadata'da period bilgilerini eksiksiz doldurun

✅ GPU varsa mutlaka kullanın

✅ Eğitim sırasında logları kaydedin

✅ Web UI'ı production'da kullanmayın (debug=True)

### 🔧 İleri Düzey

**Grad-CAM Entegrasyonu**: `create_saliency_map()` fonksiyonunu Grad-CAM ile değiştirin.

**Multi-GPU Eğitim**: YOLOv8 `device='0,1,2,3'` ile multi-GPU destekler.

**API Rate Limiting**: Flask'a rate limiting ekleyin.

**Model Versiyonlama**: Her eğitimde timestamp ile model kaydedin.

---

## 12. İletişim ve Destek

### Hata Bildirimi
Sorun yaşarsanız şu bilgileri toplayın:
- Python versiyonu
- İşletim sistemi
- Hata mesajı (tam çıktı)
- Adımları tekrarlama yöntemi

### Performans Raporlama
Model performansını `MODEL_PERFORMANCE_REPORT.md` dosyasında güncelleyin.

---

**Son Güncelleme**: 2025-10-04  
**Versiyon**: 1.0  
**Proje**: Ekzoplanet Tespit Sistemi
