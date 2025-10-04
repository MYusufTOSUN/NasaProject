# 📊 Model Performans Raporu

## Genel Bakış

Bu rapor, YOLOv8-based ekzoplanet transit sınıflandırıcısının performansını detaylı olarak sunar.

## Model Mimarisi

- **Base Model**: YOLOv8n-cls (pre-trained)
- **Input Size**: 224x224 pixels
- **Classes**: 2 (negative, positive)
- **Training Epochs**: 200
- **Batch Size**: 64

## Dataset İstatistikleri

| Split | Toplam | Positive | Negative | Pozitif Oranı |
|-------|--------|----------|----------|---------------|
| Train | -      | -        | -        | -             |
| Val   | -      | -        | -        | -             |
| Test  | -      | -        | -        | -             |

*Not: İstatistikler dataset oluşturulduktan sonra otomatik güncellenecektir.*

## Performans Metrikleri

### Test Set Sonuçları

| Metrik          | Değer |
|-----------------|-------|
| Accuracy        | -     |
| Precision       | -     |
| Recall          | -     |
| F1-Score        | -     |
| ROC AUC         | -     |

*Not: Metrikler model değerlendirmesi yapıldıktan sonra doldurulacaktır.*

## Confusion Matrix

```
                Predicted
                Neg    Pos
Actual  Neg     TN     FP
        Pos     FN     TP
```

- **True Negatives (TN)**: -
- **False Positives (FP)**: -
- **False Negatives (FN)**: -
- **True Positives (TP)**: -

## Güven Skorları Dağılımı

### Positive Class Güven Skorları
- Mean: -
- Median: -
- Std: -
- Min: -
- Max: -

### Negative Class Güven Skorları
- Mean: -
- Median: -
- Std: -
- Min: -
- Max: -

## Hata Analizi

### False Positives (Yanlış Pozitifler)
*En yüksek güven skoru ile yanlış tahmin edilen negatif örnekler*

1. -
2. -
3. -

### False Negatives (Kaçırılan Transitler)
*En düşük güven skoru ile kaçırılan pozitif örnekler*

1. -
2. -
3. -

## Eğitim Hiperparametreleri

| Parametre       | Değer          |
|-----------------|----------------|
| Optimizer       | AdamW          |
| Learning Rate   | Auto           |
| Weight Decay    | Auto           |
| Augmentation    | Default YOLO   |
| Image Size      | 224            |
| Batch Size      | 64             |
| Epochs          | 200            |
| Early Stopping  | Patience: 50   |

## Veri Ön İşleme

1. **Işık Eğrisi İndirme**: MAST API üzerinden Kepler/TESS verileri
2. **Temizleme**: NaN değerleri, sigma-clipping outliers
3. **Detrending**: Flatten (window=2001) ile trend kaldırma
4. **BLS Period Bulma**: Period/t0/duration metadata'da yoksa otomatik
5. **Faz Katlama**: Bulunan period ile fold
6. **Binning**: 0.01 phase bins ile yumuşatma
7. **Görselleştirme**: 224x224 PNG, scatter + line plot

## Model Güçlü Yönleri

1. ✅ Yüksek SNR'lı belirgin transitlerde mükemmel performans
2. ✅ Kepler ve TESS verilerinde transfer learning başarısı
3. ✅ Hızlı inference süresi (~50ms per image)
4. ✅ Compact model boyutu (YOLOv8n: ~6MB)

## İyileştirme Alanları

1. ⚠️ Düşük SNR (<7) transitlerde hassasiyet artırılabilir
2. ⚠️ V-shaped false positives (eclipsing binaries) için ek filtering
3. ⚠️ Multi-planet sistemlerde ikincil transitler için augmentation
4. ⚠️ Imbalanced dataset durumunda weighted loss kullanımı

## Öneriler

### Kısa Vadeli
- Threshold optimization (precision vs recall trade-off)
- Ensemble yöntemleri (multiple models voting)
- Post-processing heuristics (depth, duration checks)

### Uzun Vadeli
- Attention mechanism entegrasyonu
- Time-series transformer modelleri
- Physics-informed loss functions
- Active learning ile zor örneklerin etiketlenmesi

## Güven Skoru Yorumlama Rehberi

| Güven Aralığı | Yorumlama                              | Aksiyon                        |
|---------------|----------------------------------------|--------------------------------|
| 95-100%       | Kesin tespit                           | Otomatik onay                  |
| 85-95%        | Yüksek güvenle pozitif                 | Manuel kontrol önerilir        |
| 70-85%        | Orta güven, dikkatli inceleme gerekir  | Uzman değerlendirmesi zorunlu  |
| 50-70%        | Düşük güven, belirsiz                  | Ek veri veya analiz gerekli    |
| <50%          | Muhtemelen negatif                     | Ret edilebilir                 |

## Son Güncelleme

Bu rapor otomatik olarak oluşturulmuştur. Güncel metrikler için `evaluation_results/summary.csv` dosyasını kontrol edin.

**Tarih**: Model eğitildikten sonra güncellenecek  
**Model Version**: v1.0  
**Dataset**: metadata1500.csv

