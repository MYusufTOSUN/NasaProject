"""
Model Değerlendirme Script'i
Eğitilmiş modeli test seti üzerinde değerlendirir ve detaylı raporlar oluşturur.
"""

import os
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image


def evaluate_on_test_set(model_path, test_dir, output_dir="evaluation_results"):
    """
    Modeli test seti üzerinde değerlendir ve detaylı raporlar üret
    """
    print("="*60)
    print("MODEL DEĞERLENDİRME BAŞLIYOR")
    print("="*60)
    
    # Çıkış klasörü
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Model yükle
    print(f"\n[1/6] Model yükleniyor: {model_path}")
    model = YOLO(model_path)
    
    # Test görselleri topla
    print(f"\n[2/6] Test verileri toplanıyor: {test_dir}")
    test_path = Path(test_dir)
    
    # Negatif ve pozitif görselleri bul
    neg_images = list((test_path / "negative").glob("*.png"))
    pos_images = list((test_path / "positive").glob("*.png"))
    
    all_images = [(img, 0) for img in neg_images] + [(img, 1) for img in pos_images]
    
    print(f"  • Negatif örnekler: {len(neg_images)}")
    print(f"  • Pozitif örnekler: {len(pos_images)}")
    print(f"  • Toplam: {len(all_images)}")
    
    # Tahminleri topla
    print(f"\n[3/6] Model tahminleri yapılıyor...")
    y_true = []
    y_pred = []
    y_probs = []
    predictions_detail = []
    
    for img_path, true_label in tqdm(all_images, desc="Tahmin"):
        try:
            # Model tahmini
            results = model(str(img_path), verbose=False)
            
            # Sonuçları al
            probs = results[0].probs
            predicted_class = int(probs.top1)
            confidence = float(probs.top1conf)
            
            # Sınıf olasılıklarını al (0: negative, 1: positive)
            class_probs = probs.data.cpu().numpy()
            prob_positive = class_probs[1] if len(class_probs) > 1 else (1.0 if predicted_class == 1 else 0.0)
            
            y_true.append(true_label)
            y_pred.append(predicted_class)
            y_probs.append(prob_positive)
            
            predictions_detail.append({
                "image": img_path.name,
                "true_label": "positive" if true_label == 1 else "negative",
                "predicted_label": "positive" if predicted_class == 1 else "negative",
                "confidence": confidence,
                "prob_negative": class_probs[0] if len(class_probs) > 0 else (1.0 - prob_positive),
                "prob_positive": prob_positive,
                "correct": true_label == predicted_class
            })
            
        except Exception as e:
            print(f"\n[HATA] {img_path.name}: {e}")
            continue
    
    # Numpy arraylere çevir
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    
    # Sonuçları hesapla
    print(f"\n[4/6] Metrikler hesaplanıyor...")
    
    # Temel metrikler
    accuracy = (y_true == y_pred).mean()
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Metrikler
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # ROC AUC
    try:
        roc_auc = roc_auc_score(y_true, y_probs)
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    except Exception:
        roc_auc = None
        fpr, tpr, thresholds = None, None, None
    
    # Classification report
    class_report = classification_report(
        y_true, 
        y_pred, 
        target_names=["negative", "positive"],
        digits=4
    )
    
    # Sonuçları yazdır
    print("\n" + "="*60)
    print("TEST SETİ SONUÇLARI")
    print("="*60)
    print(f"\n📊 Genel Metrikler:")
    print(f"  • Accuracy (Doğruluk)    : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  • Precision (Kesinlik)   : {precision:.4f} ({precision*100:.2f}%)")
    print(f"  • Recall (Duyarlılık)    : {recall:.4f} ({recall*100:.2f}%)")
    print(f"  • F1-Score               : {f1:.4f} ({f1*100:.2f}%)")
    print(f"  • Specificity (Özgüllük) : {specificity:.4f} ({specificity*100:.2f}%)")
    if roc_auc:
        print(f"  • ROC AUC                : {roc_auc:.4f}")
    
    print(f"\n📉 Confusion Matrix:")
    print(f"  • True Negatives  (TN): {tn:4d}")
    print(f"  • False Positives (FP): {fp:4d}")
    print(f"  • False Negatives (FN): {fn:4d}")
    print(f"  • True Positives  (TP): {tp:4d}")
    
    print(f"\n📈 Hata Analizi:")
    print(f"  • False Positive Rate : {fp/(tn+fp)*100:.2f}% (Yanlış alarm)")
    print(f"  • False Negative Rate : {fn/(tp+fn)*100:.2f}% (Kaçırılan gezegen)")
    
    print("\n" + "="*60)
    print("Detaylı Sınıflandırma Raporu:")
    print("="*60)
    print(class_report)
    
    # Görselleştirmeler
    print(f"\n[5/6] Görselleştirmeler oluşturuluyor...")
    
    # 1. Confusion Matrix (Normalized)
    plt.figure(figsize=(10, 8))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                cbar_kws={'label': 'Oran'})
    plt.title('Normalize Confusion Matrix (Test Seti)', fontsize=14, fontweight='bold')
    plt.ylabel('Gerçek Sınıf', fontsize=12)
    plt.xlabel('Tahmin Edilen Sınıf', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path / "confusion_matrix_normalized.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confusion Matrix (Counts)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                cbar_kws={'label': 'Sayı'})
    plt.title('Confusion Matrix - Sayılar (Test Seti)', fontsize=14, fontweight='bold')
    plt.ylabel('Gerçek Sınıf', fontsize=12)
    plt.xlabel('Tahmin Edilen Sınıf', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path / "confusion_matrix_counts.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. ROC Curve
    if roc_auc and fpr is not None:
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve (Test Seti)', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / "roc_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Confidence Distribution
    df_pred = pd.DataFrame(predictions_detail)
    
    plt.figure(figsize=(12, 6))
    
    # Doğru ve yanlış tahminlerin confidence dağılımı
    correct_conf = df_pred[df_pred['correct'] == True]['confidence']
    incorrect_conf = df_pred[df_pred['correct'] == False]['confidence']
    
    plt.subplot(1, 2, 1)
    plt.hist(correct_conf, bins=30, alpha=0.7, color='green', label=f'Doğru (n={len(correct_conf)})')
    plt.hist(incorrect_conf, bins=30, alpha=0.7, color='red', label=f'Yanlış (n={len(incorrect_conf)})')
    plt.xlabel('Confidence', fontsize=11)
    plt.ylabel('Frekans', fontsize=11)
    plt.title('Confidence Dağılımı (Doğru vs Yanlış)', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Sınıf bazlı confidence
    plt.subplot(1, 2, 2)
    pos_conf = df_pred[df_pred['true_label'] == 'positive']['confidence']
    neg_conf = df_pred[df_pred['true_label'] == 'negative']['confidence']
    plt.hist(pos_conf, bins=30, alpha=0.7, color='blue', label=f'Positive (n={len(pos_conf)})')
    plt.hist(neg_conf, bins=30, alpha=0.7, color='orange', label=f'Negative (n={len(neg_conf)})')
    plt.xlabel('Confidence', fontsize=11)
    plt.ylabel('Frekans', fontsize=11)
    plt.title('Confidence Dağılımı (Sınıf Bazlı)', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "confidence_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # CSV'lere kaydet
    print(f"\n[6/6] Sonuçlar kaydediliyor...")
    
    # Tüm tahminler
    df_pred.to_csv(output_path / "predictions_detail.csv", index=False)
    
    # Yanlış tahminler
    df_errors = df_pred[df_pred['correct'] == False].sort_values('confidence', ascending=False)
    df_errors.to_csv(output_path / "errors_analysis.csv", index=False)
    
    # Özet rapor
    summary = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "Specificity", "ROC AUC",
                   "True Negatives", "False Positives", "False Negatives", "True Positives",
                   "Total Samples", "Negative Samples", "Positive Samples"],
        "Value": [accuracy, precision, recall, f1, specificity, roc_auc if roc_auc else 0,
                  tn, fp, fn, tp,
                  len(all_images), len(neg_images), len(pos_images)]
    }
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(output_path / "summary.csv", index=False)
    
    print(f"\n✅ Tüm sonuçlar '{output_path}' klasörüne kaydedildi:")
    print(f"  • summary.csv - Özet metrikler")
    print(f"  • predictions_detail.csv - Tüm tahmin detayları")
    print(f"  • errors_analysis.csv - Hatalı tahminler")
    print(f"  • confusion_matrix_*.png - Confusion matrix görselleri")
    print(f"  • roc_curve.png - ROC eğrisi")
    print(f"  • confidence_distribution.png - Confidence dağılımları")
    
    print("\n" + "="*60)
    print("DEĞERLENDİRME TAMAMLANDI!")
    print("="*60)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "roc_auc": roc_auc,
        "confusion_matrix": cm
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Model test seti değerlendirmesi")
    parser.add_argument(
        "--model",
        type=str,
        default="models/best.pt",
        help="Model dosyası yolu (varsayılan: models/best.pt)"
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="data/plots/test",
        help="Test klasörü yolu (varsayılan: data/plots/test)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results",
        help="Çıkış klasörü (varsayılan: evaluation_results)"
    )
    
    args = parser.parse_args()
    
    # Değerlendirmeyi çalıştır
    results = evaluate_on_test_set(
        model_path=args.model,
        test_dir=args.test_dir,
        output_dir=args.output
    )

