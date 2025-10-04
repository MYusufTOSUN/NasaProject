#!/usr/bin/env python3
"""
Metrik değerlendirme scripti - confusion matrix, ROC curve, performans raporları
predictions_detail.csv veya all_graphs_predictions.csv dosyasından metrikleri hesaplar.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def load_predictions(pred_csv):
    """
    Tahmin sonuçlarını yükle.
    
    Args:
        pred_csv: Predictions CSV dosyası
        
    Returns:
        DataFrame veya None
    """
    print(f"✓ Tahminler yükleniyor: {pred_csv}")
    
    pred_path = Path(pred_csv)
    if not pred_path.exists():
        print(f"HATA: Dosya bulunamadı: {pred_csv}")
        return None
    
    try:
        df = pd.read_csv(pred_csv)
        print(f"  {len(df)} satır yüklendi")
        
        # Gerekli kolonları kontrol et
        required_cols = ['image_path', 'pred_label', 'prob_positive', 'prob_negative']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"HATA: Eksik kolonlar: {missing_cols}")
            return None
        
        return df
        
    except Exception as e:
        print(f"HATA: Dosya okuma hatası: {e}")
        return None


def check_true_labels(df):
    """
    True label kolonunu kontrol et ve hazırla.
    
    Args:
        df: DataFrame
        
    Returns:
        tuple: (has_labels, df)
    """
    print("\n✓ True label kontrolü...")
    
    # true_label kolonu var mı?
    if 'true_label' in df.columns:
        # Boş değerler var mı kontrol et
        non_empty = df['true_label'].notna() & (df['true_label'] != '')
        
        if non_empty.sum() > 0:
            # String'leri int'e çevir
            df.loc[non_empty, 'true_label'] = df.loc[non_empty, 'true_label'].astype(int)
            
            labeled_count = non_empty.sum()
            print(f"  {labeled_count}/{len(df)} görsel için true label mevcut")
            
            # Dağılımı göster
            label_dist = df.loc[non_empty, 'true_label'].value_counts().to_dict()
            print(f"  Dağılım: {label_dist}")
            
            return True, df
        else:
            print("  ⚠ true_label kolonu var ama hepsi boş")
            return False, df
    else:
        print("  ℹ true_label kolonu yok (sadece tahminler)")
        return False, df


def calculate_metrics(y_true, y_pred, y_proba_pos):
    """
    Temel metrikleri hesapla.
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        y_proba_pos: Positive sınıf olasılıkları
        
    Returns:
        dict: Metrikler
    """
    print("\n✓ Metrikler hesaplanıyor...")
    
    # Temel metrikler
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    # ROC-AUC
    try:
        fpr, tpr, _ = roc_curve(y_true, y_proba_pos)
        roc_auc = auc(fpr, tpr)
        metrics['roc_auc'] = roc_auc
        
        # PR-AUC
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba_pos)
        pr_auc = average_precision_score(y_true, y_proba_pos)
        metrics['pr_auc'] = pr_auc
        
    except Exception as e:
        print(f"  ⚠ AUC metrikleri hesaplanamadı: {e}")
        metrics['roc_auc'] = 0.0
        metrics['pr_auc'] = 0.0
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    if 'roc_auc' in metrics:
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return metrics


def create_confusion_matrix(y_true, y_pred, output_dir):
    """
    Confusion matrix oluştur ve kaydet.
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        output_dir: Çıktı klasörü
        
    Returns:
        ndarray: Confusion matrix
    """
    print("\n✓ Confusion matrix oluşturuluyor...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Grafik oluştur
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative (0)', 'Positive (1)'],
                yticklabels=['Negative (0)', 'Positive (1)'],
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix\nExoplanet Transit Detection', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    
    # Yüzdeleri ekle
    cm_sum = cm.sum()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            percentage = cm[i, j] / cm_sum * 100
            plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='gray')
    
    # Kaydet
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cm_path = output_path / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Kaydedildi: {cm_path}")
    
    return cm


def create_roc_curve(y_true, y_proba_pos, output_dir):
    """
    ROC curve oluştur ve kaydet.
    
    Args:
        y_true: Gerçek etiketler
        y_proba_pos: Positive sınıf olasılıkları
        output_dir: Çıktı klasörü
        
    Returns:
        float: ROC-AUC değeri
    """
    print("\n✓ ROC curve oluşturuluyor...")
    
    try:
        # ROC curve hesapla
        fpr, tpr, thresholds = roc_curve(y_true, y_proba_pos)
        roc_auc = auc(fpr, tpr)
        
        # Grafik
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier (AUC = 0.5)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curve - Exoplanet Transit Detection', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Kaydet
        output_path = Path(output_dir)
        roc_path = output_path / 'roc_curve.png'
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Kaydedildi: {roc_path}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        
        return roc_auc
        
    except Exception as e:
        print(f"  ⚠ ROC curve oluşturulamadı: {e}")
        return None


def create_pr_curve(y_true, y_proba_pos, output_dir):
    """
    Precision-Recall curve oluştur ve kaydet.
    
    Args:
        y_true: Gerçek etiketler
        y_proba_pos: Positive sınıf olasılıkları
        output_dir: Çıktı klasörü
        
    Returns:
        float: PR-AUC değeri
    """
    print("\n✓ Precision-Recall curve oluşturuluyor...")
    
    try:
        # PR curve hesapla
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba_pos)
        pr_auc = average_precision_score(y_true, y_proba_pos)
        
        # Grafik
        plt.figure(figsize=(10, 8))
        plt.plot(recall_vals, precision_vals, color='blue', lw=2, 
                label=f'PR curve (AUC = {pr_auc:.4f})')
        
        # Baseline (pozitif sınıf oranı)
        baseline = sum(y_true) / len(y_true)
        plt.axhline(y=baseline, color='red', linestyle='--', lw=2,
                   label=f'Baseline (ratio = {baseline:.4f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12, fontweight='bold')
        plt.ylabel('Precision', fontsize=12, fontweight='bold')
        plt.title('Precision-Recall Curve - Exoplanet Transit Detection', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Kaydet
        output_path = Path(output_dir)
        pr_path = output_path / 'precision_recall_curve.png'
        plt.savefig(pr_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Kaydedildi: {pr_path}")
        print(f"  PR-AUC: {pr_auc:.4f}")
        
        return pr_auc
        
    except Exception as e:
        print(f"  ⚠ PR curve oluşturulamadı: {e}")
        return None


def analyze_errors(df_labeled, output_dir):
    """
    Hatalı tahminleri analiz et.
    
    Args:
        df_labeled: Labeled DataFrame
        output_dir: Çıktı klasörü
        
    Returns:
        DataFrame: Hata analizi
    """
    print("\n✓ Hata analizi yapılıyor...")
    
    # Hataları bul
    df_labeled['correct'] = df_labeled['true_label'] == df_labeled['pred_label']
    errors_df = df_labeled[~df_labeled['correct']].copy()
    
    if len(errors_df) == 0:
        print("  ℹ Hiç hata yok!")
        return pd.DataFrame()
    
    print(f"  {len(errors_df)} hatalı tahmin bulundu")
    
    # Hata tipleri
    errors_df['error_type'] = errors_df.apply(
        lambda row: 'False Positive' if row['true_label'] == 0 else 'False Negative',
        axis=1
    )
    
    # Güven skoru
    errors_df['confidence'] = errors_df.apply(
        lambda row: max(row['prob_positive'], row['prob_negative']),
        axis=1
    )
    
    # Sırala
    errors_df = errors_df.sort_values('confidence', ascending=False)
    
    # İstatistikler
    error_types = errors_df['error_type'].value_counts()
    print(f"  False Positive: {error_types.get('False Positive', 0)}")
    print(f"  False Negative: {error_types.get('False Negative', 0)}")
    
    # Kaydet
    output_path = Path(output_dir)
    errors_path = output_path / 'error_analysis.csv'
    errors_df[['image_path', 'true_label', 'pred_label', 'prob_positive', 'prob_negative', 
               'confidence', 'error_type']].to_csv(errors_path, index=False)
    
    print(f"  Kaydedildi: {errors_path}")
    
    return errors_df


def save_metrics_summary(metrics, cm, output_dir):
    """
    Metrik özetini CSV'ye kaydet.
    
    Args:
        metrics: Metrik dictionary
        cm: Confusion matrix
        output_dir: Çıktı klasörü
    """
    print("\n✓ Metrik özeti kaydediliyor...")
    
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1_score'],
        'roc_auc': metrics.get('roc_auc', 0.0),
        'pr_auc': metrics.get('pr_auc', 0.0),
        'true_negative': int(cm[0, 0]) if cm is not None else 0,
        'false_positive': int(cm[0, 1]) if cm is not None else 0,
        'false_negative': int(cm[1, 0]) if cm is not None else 0,
        'true_positive': int(cm[1, 1]) if cm is not None else 0
    }
    
    output_path = Path(output_dir)
    summary_path = output_path / 'summary.csv'
    
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    
    print(f"  Kaydedildi: {summary_path}")


def load_split_summary():
    """
    Dataset split özetini yükle (varsa).
    
    Returns:
        dict veya None
    """
    split_summary_path = Path('evaluation_results/split_summary.csv')
    
    if not split_summary_path.exists():
        return None
    
    try:
        df = pd.read_csv(split_summary_path)
        
        summary = {}
        for _, row in df.iterrows():
            split = row['split']
            summary[split] = {
                'positive': int(row['positive']),
                'negative': int(row['negative']),
                'total': int(row['total']),
                'num_targets': int(row['num_targets'])
            }
        
        return summary
        
    except Exception as e:
        print(f"  ⚠ Split summary okunamadı: {e}")
        return None


def update_performance_report(metrics, cm, errors_df, output_dir):
    """
    MODEL_PERFORMANCE_REPORT.md dosyasını güncelle.
    
    Args:
        metrics: Metrikler
        cm: Confusion matrix
        errors_df: Hata DataFrame
        output_dir: Çıktı klasörü
    """
    print("\n✓ Performans raporu güncelleniyor...")
    
    # Rapor içeriğini hazırla
    current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Dataset split bilgilerini yükle
    split_summary = load_split_summary()
    
    # Yanlış sınıflanan örnekler (en fazla 15)
    error_list_str = ""
    if len(errors_df) > 0:
        top_errors = errors_df.head(15)
        for idx, row in top_errors.iterrows():
            filename = Path(row['image_path']).name
            error_list_str += f"- **{filename}**\n"
            error_list_str += f"  - True: {row['true_label']} | Pred: {row['pred_label']} | "
            error_list_str += f"Confidence: {row['confidence']:.4f} | Type: {row['error_type']}\n"
    else:
        error_list_str = "- Hiç hata yok! Mükemmel performans! 🎉\n"
    
    # Rapor içeriği
    report_content = f"""# Model Performance Report

## 1. Overview
**Version**: 1.0  
**Date**: {current_date}  
**Dataset**: NASA Exoplanet Transit Detection

## 2. Dataset Split
"""
    
    if split_summary:
        for split_name in ['train', 'val', 'test']:
            if split_name in split_summary:
                split_data = split_summary[split_name]
                report_content += f"""
**{split_name.capitalize()} Set**:
- Positive: {split_data['positive']} samples
- Negative: {split_data['negative']} samples
- Total: {split_data['total']} samples
- Unique targets: {split_data['num_targets']}
"""
    else:
        report_content += """
**Training Set**: 
- Positive: [Veri yok] samples
- Negative: [Veri yok] samples
- Total: [Veri yok] samples

**Validation Set**:
- Positive: [Veri yok] samples  
- Negative: [Veri yok] samples
- Total: [Veri yok] samples

**Test Set**:
- Positive: [Veri yok] samples
- Negative: [Veri yok] samples  
- Total: [Veri yok] samples
"""
    
    report_content += f"""
## 3. Classification Metrics

| Metric | Value | Percentage |
|--------|-------|------------|
| **Accuracy** | {metrics['accuracy']:.4f} | {metrics['accuracy']*100:.2f}% |
| **Precision** | {metrics['precision']:.4f} | {metrics['precision']*100:.2f}% |
| **Recall** | {metrics['recall']:.4f} | {metrics['recall']*100:.2f}% |
| **F1-Score** | {metrics['f1_score']:.4f} | {metrics['f1_score']*100:.2f}% |
| **ROC-AUC** | {metrics.get('roc_auc', 0.0):.4f} | {metrics.get('roc_auc', 0.0)*100:.2f}% |
| **PR-AUC** | {metrics.get('pr_auc', 0.0):.4f} | {metrics.get('pr_auc', 0.0)*100:.2f}% |

### Metric Açıklamaları
- **Accuracy**: Tüm tahminlerin ne kadarının doğru olduğu
- **Precision**: Positive tahminlerin ne kadarının gerçekten positive olduğu
- **Recall**: Gerçek positive'lerin ne kadarının bulunduğu
- **F1-Score**: Precision ve Recall'ın harmonik ortalaması
- **ROC-AUC**: ROC eğrisi altındaki alan (model ayrıştırma gücü)
- **PR-AUC**: Precision-Recall eğrisi altındaki alan (dengesiz veri için)

## 4. Confusion Matrix

"""
    
    if cm is not None:
        report_content += f"""
| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | {cm[0,0]} (TN) | {cm[0,1]} (FP) |
| **Actual Positive** | {cm[1,0]} (FN) | {cm[1,1]} (TP) |

- **True Negative (TN)**: {cm[0,0]} - Doğru negatif tahminler
- **False Positive (FP)**: {cm[0,1]} - Yanlışlıkla positive denilen negatifler
- **False Negative (FN)**: {cm[1,0]} - Kaçırılan positive'ler
- **True Positive (TP)**: {cm[1,1]} - Doğru positive tahminler

"""
    
    # Göreli path'ler (output_dir'e göre)
    report_content += f"""
![Confusion Matrix](evaluation_results/confusion_matrix.png)

## 5. ROC Curve
ROC (Receiver Operating Characteristic) eğrisi, modelin farklı eşik değerlerindeki performansını gösterir.
AUC (Area Under Curve) değeri ne kadar yüksekse model o kadar iyidir (maksimum 1.0).

![ROC Curve](evaluation_results/roc_curve.png)

## 6. Precision-Recall Curve
Precision-Recall eğrisi, özellikle dengesiz veri setlerinde modelin performansını değerlendirmek için kullanılır.

![PR Curve](evaluation_results/precision_recall_curve.png)

## 7. Error Analysis

### Toplam Hatalar
- **Toplam yanlış tahmin**: {len(errors_df)}
"""
    
    if len(errors_df) > 0:
        fp_count = len(errors_df[errors_df['error_type'] == 'False Positive'])
        fn_count = len(errors_df[errors_df['error_type'] == 'False Negative'])
        report_content += f"""- **False Positive**: {fp_count} (negatif ama positive dendi)
- **False Negative**: {fn_count} (positive ama negatif dendi)
"""
    
    report_content += f"""
### En Hatalı {min(15, len(errors_df))} Tahmin
(Güvene göre sıralı - yüksek güvenle yapılan hatalar)

{error_list_str}

Detaylı hata analizi için: `evaluation_results/error_analysis.csv`

## 8. Performance Summary

### Güçlü Yönler
"""
    
    # Güçlü yönleri otomatik belirle
    if metrics['accuracy'] >= 0.95:
        report_content += "- ✅ Çok yüksek doğruluk oranı (>95%)\n"
    elif metrics['accuracy'] >= 0.90:
        report_content += "- ✅ Yüksek doğruluk oranı (>90%)\n"
    
    if metrics['precision'] >= 0.95:
        report_content += "- ✅ Çok düşük False Positive oranı\n"
    
    if metrics['recall'] >= 0.95:
        report_content += "- ✅ Çok düşük False Negative oranı (neredeyse tüm transitler bulunuyor)\n"
    
    if metrics.get('roc_auc', 0) >= 0.95:
        report_content += "- ✅ Mükemmel sınıf ayrıştırma gücü (ROC-AUC >0.95)\n"
    
    report_content += """
### İyileştirme Alanları
"""
    
    # İyileştirme alanları
    if metrics['recall'] < 0.90:
        report_content += "- ⚠️ Recall düşük - bazı transitler kaçırılıyor\n"
    
    if metrics['precision'] < 0.90:
        report_content += "- ⚠️ Precision düşük - fazla False Positive var\n"
    
    if len(errors_df) > 0 and errors_df['confidence'].mean() > 0.7:
        report_content += "- ⚠️ Hatalar yüksek güvenle yapılıyor - model aşırı güvenli\n"
    
    if metrics['accuracy'] < 0.90:
        report_content += "- ⚠️ Genel doğruluk artırılabilir\n"
    
    report_content += """
## 9. Next Steps & Recommendations

### Model İyileştirme
- [ ] Daha fazla epoch ile eğitim dene
- [ ] Farklı augmentation stratejileri uygula
- [ ] Hatalı örnekleri manuel incele
- [ ] Ensemble modeller dene (birden fazla model kombinasyonu)

### Veri İyileştirme
- [ ] Daha fazla örnek topla (özellikle hata yapılan sınıflar için)
- [ ] Veri kalitesini artır (daha iyi grafik üretimi)
- [ ] Farklı grafik tiplerini dene (farklı normalizasyon, binning)

### Değerlendirme
- [ ] Farklı mission'lar için ayrı performans analizi yap
- [ ] Farklı yıldız türleri için performansı incele
- [ ] Production ortamında gerçek verilerle test et

---

**Rapor Oluşturma Tarihi**: {current_date}  
**Script**: `scripts/evaluate_model.py`  
**Otomatik üretildi** ✨
"""
    
    # Güncellenen raporu kaydet
    output_report_path = Path(output_dir) / 'MODEL_PERFORMANCE_REPORT.md'
    with open(output_report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"  Kaydedildi: {output_report_path}")
    
    # Ana klasöre de kopyala
    main_report_path = Path('MODEL_PERFORMANCE_REPORT.md')
    with open(main_report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"  Ana rapor güncellendi: {main_report_path}")


def print_final_summary(metrics, cm):
    """
    Final özeti yazdır.
    
    Args:
        metrics: Metrikler
        cm: Confusion matrix
    """
    print("\n" + "=" * 60)
    print("DEĞERLENDİRME ÖZETİ")
    print("=" * 60)
    
    print(f"\nSınıflandırma Metrikleri:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    
    if 'roc_auc' in metrics and metrics['roc_auc'] > 0:
        print(f"\nAUC Metrikleri:")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:  {metrics['pr_auc']:.4f}")
    
    if cm is not None:
        print(f"\nConfusion Matrix:")
        print(f"  True Negative:  {cm[0,0]:4d}")
        print(f"  False Positive: {cm[0,1]:4d}")
        print(f"  False Negative: {cm[1,0]:4d}")
        print(f"  True Positive:  {cm[1,1]:4d}")
        
        total = cm.sum()
        print(f"\n  Toplam: {total}")
        print(f"  Doğru tahmin: {cm[0,0] + cm[1,1]} ({(cm[0,0] + cm[1,1])/total*100:.2f}%)")
        print(f"  Yanlış tahmin: {cm[0,1] + cm[1,0]} ({(cm[0,1] + cm[1,0])/total*100:.2f}%)")
    
    print("=" * 60)


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(
        description='Model performans değerlendirmesi',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--predictions', '--pred_csv', dest='predictions', required=True,
                       help='Tahmin CSV dosyası (örn: predictions_detail.csv)')
    parser.add_argument('--output_dir', '--out_dir', dest='output_dir', default='evaluation_results',
                       help='Çıktı klasörü')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MODEL PERFORMANS DEĞERLENDİRMESİ")
    print("=" * 60)
    print(f"Tahmin dosyası: {args.predictions}")
    print(f"Çıktı klasörü: {args.output_dir}")
    print("=" * 60)
    
    # 1. Tahminleri yükle
    df = load_predictions(args.predictions)
    if df is None:
        sys.exit(1)
    
    # 2. True label kontrolü
    has_labels, df = check_true_labels(df)
    
    if not has_labels:
        print("\n⚠ True label bulunamadı - metrik hesaplaması yapılamaz!")
        print("Sadece tahmin istatistikleri:")
        pred_dist = df['pred_label'].value_counts()
        print(f"  Positive tahmin: {pred_dist.get(1, 0)}")
        print(f"  Negative tahmin: {pred_dist.get(0, 0)}")
        sys.exit(0)
    
    # 3. Labeled verileri filtrele
    df_labeled = df[df['true_label'].notna()].copy()
    df_labeled['true_label'] = df_labeled['true_label'].astype(int)
    
    print(f"\n✓ {len(df_labeled)} etiketli görsel ile değerlendirme yapılacak")
    
    # 4. Arrays hazırla
    y_true = df_labeled['true_label'].values
    y_pred = df_labeled['pred_label'].values
    y_proba_pos = df_labeled['prob_positive'].values
    
    # 5. Metrikleri hesapla
    metrics = calculate_metrics(y_true, y_pred, y_proba_pos)
    
    # 6. Confusion matrix
    cm = create_confusion_matrix(y_true, y_pred, args.output_dir)
    
    # 7. ROC curve
    create_roc_curve(y_true, y_proba_pos, args.output_dir)
    
    # 8. PR curve
    create_pr_curve(y_true, y_proba_pos, args.output_dir)
    
    # 9. Hata analizi
    errors_df = analyze_errors(df_labeled, args.output_dir)
    
    # 10. Metrik özetini kaydet
    save_metrics_summary(metrics, cm, args.output_dir)
    
    # 11. Performans raporunu güncelle
    update_performance_report(metrics, cm, errors_df, args.output_dir)
    
    # 12. Final özet
    print_final_summary(metrics, cm)
    
    print("\n" + "=" * 60)
    print("✓ DEĞERLENDİRME TAMAMLANDI")
    print("=" * 60)
    print(f"Sonuçlar: {Path(args.output_dir).resolve()}")
    print(f"Performans raporu: MODEL_PERFORMANCE_REPORT.md")
    print("=" * 60)


if __name__ == "__main__":
    main()

