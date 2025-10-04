#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Değerlendirme
predictions_detail.csv ile metadata'yı birleştirir ve metrikleri hesaplar
Confusion Matrix, ROC Curve, Summary CSV üretir
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, roc_curve
)
import seaborn as sns


def load_predictions_and_labels(predictions_csv, metadata_csv):
    """
    Tahminleri ve gerçek label'ları yükle
    """
    print("📊 Veriler yükleniyor...")
    
    df_pred = pd.read_csv(predictions_csv)
    df_meta = pd.read_csv(metadata_csv)
    
    # image_path'ten target bilgisini çıkar
    def extract_target(path_str):
        filename = Path(path_str).name
        # Örnek: Kepler-10_Kepler_phase.png
        parts = filename.rsplit('_', 2)
        if len(parts) >= 2:
            return parts[0]
        return None
    
    df_pred['target'] = df_pred['image_path'].apply(extract_target)
    
    # Metadata ile birleştir
    df_merged = df_pred.merge(
        df_meta[['target', 'label']],
        on='target',
        how='left'
    )
    
    # Label'ı normalize et
    df_merged['label'] = df_merged['label'].str.lower()
    df_merged['label'] = df_merged['label'].apply(
        lambda x: 'positive' if x in ['positive', 'transit', '1', 1, 'yes'] else 'negative'
    )
    
    # Eksik label'ları kontrol et
    missing = df_merged['label'].isna().sum()
    if missing > 0:
        print(f"⚠ {missing} görselin ground truth label'ı yok, kaldırılıyor...")
        df_merged = df_merged.dropna(subset=['label'])
    
    print(f"✓ {len(df_merged)} görsel değerlendirilecek")
    print(f"  Ground Truth Positive: {(df_merged['label'] == 'positive').sum()}")
    print(f"  Ground Truth Negative: {(df_merged['label'] == 'negative').sum()}")
    
    return df_merged


def calculate_metrics(df):
    """
    Performans metriklerini hesapla
    """
    print("\n📈 Metrikler hesaplanıyor...")
    
    y_true = (df['label'] == 'positive').astype(int)
    y_pred = (df['pred_label'] == 'positive').astype(int)
    y_score = df['conf_pos']
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.0,
    }
    
    print("✓ Metrikler:")
    for key, value in metrics.items():
        print(f"  {key.upper()}: {value:.4f}")
    
    return metrics, y_true, y_pred, y_score


def plot_confusion_matrix(y_true, y_pred, output_path):
    """
    Confusion matrix görselleştir
    """
    print("\n📊 Confusion Matrix oluşturuluyor...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion Matrix kaydedildi: {output_path}")


def plot_roc_curve(y_true, y_score, output_path, auc_score):
    """
    ROC eğrisi çiz
    """
    print("\n📈 ROC Curve oluşturuluyor...")
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ ROC Curve kaydedildi: {output_path}")


def save_summary(metrics, output_csv):
    """
    Metrikleri CSV'ye kaydet
    """
    print(f"\n💾 Özet rapor kaydediliyor...")
    
    df_summary = pd.DataFrame([metrics])
    df_summary.to_csv(output_csv, index=False)
    
    print(f"✓ Summary kaydedildi: {output_csv}")


def main():
    print("="*60)
    print("📊 Model Değerlendirme")
    print("="*60)
    
    # Dosya yolları
    predictions_csv = 'evaluation_results/predictions_detail.csv'
    metadata_csv = 'data/metadata/metadata1500.csv'
    
    output_dir = Path('evaluation_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cm_output = output_dir / 'confusion_matrix.png'
    roc_output = output_dir / 'roc_curve.png'
    summary_output = output_dir / 'summary.csv'
    
    # Kontroller
    if not os.path.exists(predictions_csv):
        print(f"✗ HATA: {predictions_csv} bulunamadı!")
        print("  Önce scripts/05_batch_score_all.py çalıştırın.")
        return
    
    if not os.path.exists(metadata_csv):
        print(f"✗ HATA: {metadata_csv} bulunamadı!")
        return
    
    # 1. Verileri yükle
    df = load_predictions_and_labels(predictions_csv, metadata_csv)
    
    # 2. Metrikleri hesapla
    metrics, y_true, y_pred, y_score = calculate_metrics(df)
    
    # 3. Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, cm_output)
    
    # 4. ROC Curve
    if len(np.unique(y_true)) > 1:
        plot_roc_curve(y_true, y_score, roc_output, metrics['roc_auc'])
    else:
        print("\n⚠ Tek sınıf var, ROC curve atlanıyor")
    
    # 5. Summary kaydet
    save_summary(metrics, summary_output)
    
    print("\n" + "="*60)
    print("✓ Değerlendirme tamamlandı!")
    print(f"  Confusion Matrix: {cm_output}")
    print(f"  ROC Curve: {roc_output}")
    print(f"  Summary: {summary_output}")
    print("="*60)


if __name__ == '__main__':
    main()

