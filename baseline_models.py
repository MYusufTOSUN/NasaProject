#!/usr/bin/env python3
"""
Baseline Klasik ML Modelleri
BLS metriklerinden (period, depth, snr, vb.) LogisticRegression ve RandomForest
eğitir ve YOLO modeline karşı referans (baseline) sağlar.
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')


def load_targets(csv_path):
    """
    targets.csv dosyasını yükle.
    
    Args:
        csv_path: CSV dosya yolu
        
    Returns:
        DataFrame
    """
    print(f"✓ Hedefler yükleniyor: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"  {len(df)} hedef yüklendi")
        
        # Label dağılımı
        label_dist = df['label'].value_counts().to_dict()
        print(f"  Label dağılımı: {label_dist}")
        
        return df
        
    except Exception as e:
        print(f"HATA: targets.csv okunamadı: {e}")
        sys.exit(1)


def find_metric_files(data_dir='data'):
    """
    data/ klasöründeki *_metrics.json dosyalarını bul.
    
    Args:
        data_dir: Data klasörü
        
    Returns:
        list: Metric dosya yolları
    """
    print(f"\n✓ Metrik dosyaları taranıyor: {data_dir}")
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"HATA: {data_dir} klasörü bulunamadı!")
        sys.exit(1)
    
    # *_metrics.json dosyalarını bul
    metric_files = list(data_path.glob('*_metrics.json'))
    
    print(f"  {len(metric_files)} metrik dosyası bulundu")
    
    return metric_files


def extract_target_mission_from_filename(filename):
    """
    Dosya adından target ve mission bilgisini çıkar.
    
    Args:
        filename: Dosya adı (örn: Kepler_10_Kepler_metrics.json)
        
    Returns:
        tuple: (target, mission) veya (None, None)
    """
    # Dosya adı formatı: Target_Mission_metrics.json
    # Örnek: Kepler_10_Kepler_metrics.json -> target=Kepler-10, mission=Kepler
    
    stem = filename.stem  # .json'sız
    parts = stem.split('_')
    
    # Son kısmı kaldır (_metrics)
    if parts[-1] == 'metrics':
        parts = parts[:-1]
    
    # Mission genellikle son kısım
    mission = None
    for part in reversed(parts):
        if part in ['Kepler', 'TESS', 'K2']:
            mission = part
            break
    
    if mission:
        # Mission'dan önceki kısımlar target
        mission_idx = len(parts) - 1 - list(reversed(parts)).index(mission)
        target_parts = parts[:mission_idx]
        target = '-'.join(target_parts)  # Kepler_10 -> Kepler-10
        
        return target, mission
    
    return None, None


def load_metrics(metric_files):
    """
    Metrik dosyalarını yükle ve DataFrame'e dönüştür.
    
    Args:
        metric_files: Metrik dosya yolları
        
    Returns:
        DataFrame
    """
    print(f"\n✓ Metrikler yükleniyor...")
    
    metrics_list = []
    failed_files = []
    
    for metric_file in metric_files:
        try:
            with open(metric_file, 'r') as f:
                metrics = json.load(f)
            
            # Dosya adından target ve mission çıkar
            target, mission = extract_target_mission_from_filename(metric_file)
            
            if target is None or mission is None:
                failed_files.append(str(metric_file))
                continue
            
            # Özellikleri çıkar
            features = {
                'target': target,
                'mission': mission,
                'period': metrics.get('period', np.nan),
                'duration': metrics.get('duration', np.nan),
                'depth': metrics.get('depth', np.nan),
                'snr': metrics.get('snr', np.nan),
                'odd_even_depth_ratio': metrics.get('odd_even_depth_ratio', np.nan),
                'power': metrics.get('power', np.nan),
                't0': metrics.get('t0', np.nan),
                'data_points': metrics.get('data_points', np.nan),
                'time_span_days': metrics.get('time_span_days', np.nan)
            }
            
            metrics_list.append(features)
            
        except Exception as e:
            failed_files.append(str(metric_file))
            continue
    
    print(f"  {len(metrics_list)} metrik başarıyla yüklendi")
    
    if failed_files:
        print(f"  ⚠ {len(failed_files)} dosya yüklenemedi")
    
    if not metrics_list:
        print(f"HATA: Hiç metrik yüklenemedi!")
        sys.exit(1)
    
    df = pd.DataFrame(metrics_list)
    
    return df


def merge_with_targets(metrics_df, targets_df):
    """
    Metrikleri targets ile birleştir.
    
    Args:
        metrics_df: Metrik DataFrame
        targets_df: Target DataFrame
        
    Returns:
        DataFrame: Birleştirilmiş
    """
    print(f"\n✓ Metrikler ve hedefler birleştiriliyor...")
    
    # targets.csv'de target isimleri normalize et (boşluk/tire)
    # Kepler-10 veya Kepler 10 olabilir
    
    # Merge
    merged_df = pd.merge(
        metrics_df,
        targets_df,
        on=['target', 'mission'],
        how='inner'
    )
    
    print(f"  {len(merged_df)} örnek eşleştirildi")
    
    if len(merged_df) == 0:
        print(f"HATA: Hiç örnek eşleşmedi!")
        print(f"Metrik targets: {metrics_df['target'].unique()[:5]}")
        print(f"CSV targets: {targets_df['target'].unique()[:5]}")
        sys.exit(1)
    
    # Label dağılımı
    label_dist = merged_df['label'].value_counts().to_dict()
    print(f"  Label dağılımı: {label_dist}")
    
    return merged_df


def prepare_features(df, feature_cols):
    """
    Feature'ları hazırla ve temizle.
    
    Args:
        df: DataFrame
        feature_cols: Feature kolonları
        
    Returns:
        DataFrame: Temizlenmiş
    """
    print(f"\n✓ Feature'lar hazırlanıyor...")
    
    # NaN kontrolü
    print(f"  Eksik değer kontrolü:")
    for col in feature_cols:
        missing = df[col].isna().sum()
        if missing > 0:
            print(f"    {col}: {missing} eksik değer")
    
    # NaN içeren satırları kaldır
    df_clean = df.dropna(subset=feature_cols)
    
    removed = len(df) - len(df_clean)
    if removed > 0:
        print(f"  ⚠ {removed} satır kaldırıldı (eksik değer)")
    
    print(f"  {len(df_clean)} temiz örnek kaldı")
    
    # Inf kontrolü
    for col in feature_cols:
        inf_count = np.isinf(df_clean[col]).sum()
        if inf_count > 0:
            print(f"  ⚠ {col}: {inf_count} sonsuz değer -> medyan ile doldurulacak")
            median_val = df_clean[col][~np.isinf(df_clean[col])].median()
            df_clean.loc[np.isinf(df_clean[col]), col] = median_val
    
    return df_clean


def stratified_split_by_target(df, train_size=0.7, val_size=0.15, random_state=42):
    """
    Yıldız bazlı stratified split.
    
    Args:
        df: DataFrame
        train_size: Train oranı
        val_size: Val oranı
        random_state: Random seed
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    print(f"\n✓ Yıldız bazlı split yapılıyor...")
    
    # Her target için bir label (bir target'ın tüm metrikleri aynı label'a sahip)
    target_labels = df.groupby('target')['label'].first()
    
    # Unique target'lar
    targets = target_labels.index.values
    labels = target_labels.values
    
    print(f"  {len(targets)} unique target")
    
    # Train ve (val+test) ayır
    train_targets, temp_targets = train_test_split(
        targets,
        train_size=train_size,
        stratify=labels,
        random_state=random_state
    )
    
    # Val ve test ayır
    temp_labels = target_labels[temp_targets].values
    val_ratio = val_size / (val_size + (1 - train_size - val_size))
    
    val_targets, test_targets = train_test_split(
        temp_targets,
        train_size=val_ratio,
        stratify=temp_labels,
        random_state=random_state
    )
    
    # DataFrame'leri oluştur
    train_df = df[df['target'].isin(train_targets)]
    val_df = df[df['target'].isin(val_targets)]
    test_df = df[df['target'].isin(test_targets)]
    
    print(f"  Train: {len(train_df)} örnek ({len(train_targets)} target)")
    print(f"  Val:   {len(val_df)} örnek ({len(val_targets)} target)")
    print(f"  Test:  {len(test_df)} örnek ({len(test_targets)} target)")
    
    return train_df, val_df, test_df


def train_evaluate_model(model, model_name, X_train, y_train, X_test, y_test, scaler=None):
    """
    Model eğit ve değerlendir.
    
    Args:
        model: Model instance
        model_name: Model adı
        X_train, y_train: Train verisi
        X_test, y_test: Test verisi
        scaler: Scaler (varsa)
        
    Returns:
        dict: Metrikler
    """
    print(f"\n✓ {model_name} eğitiliyor...")
    
    # Scaling
    if scaler is not None:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    # Eğitim
    model.fit(X_train_scaled, y_train)
    print(f"  Eğitim tamamlandı")
    
    # Tahmin
    y_pred = model.predict(X_test_scaled)
    
    # Olasılıklar (ROC-AUC için)
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_proba = model.decision_function(X_test_scaled)
    
    # Metrikler
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['true_negative'] = int(cm[0, 0])
    metrics['false_positive'] = int(cm[0, 1])
    metrics['false_negative'] = int(cm[1, 0])
    metrics['true_positive'] = int(cm[1, 1])
    
    # Yazdır
    print(f"\n  Metrikler:")
    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    F1-Score:  {metrics['f1_score']:.4f}")
    print(f"    ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}, TP={cm[1,1]}")
    
    return metrics


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(
        description='Baseline klasik ML modelleri (BLS metriklerinden)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--targets', default='targets.csv',
                       help='Hedef listesi CSV')
    parser.add_argument('--data_dir', default='data',
                       help='Metrik dosyalarının bulunduğu klasör')
    parser.add_argument('--output', default='evaluation_results/baseline_metrics.csv',
                       help='Çıktı metrik dosyası')
    parser.add_argument('--train', type=float, default=0.7,
                       help='Train oranı')
    parser.add_argument('--val', type=float, default=0.15,
                       help='Val oranı')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    print("="*60)
    print("BASELINE KLASİK ML MODELLERİ")
    print("="*60)
    print(f"Targets: {args.targets}")
    print(f"Data dir: {args.data_dir}")
    print(f"Output: {args.output}")
    print(f"Split: {args.train}/{args.val}/{1-args.train-args.val}")
    print("="*60)
    
    # 1. Hedefleri yükle
    targets_df = load_targets(args.targets)
    
    # 2. Metrik dosyalarını bul
    metric_files = find_metric_files(args.data_dir)
    
    if not metric_files:
        print("\n⚠ Hiç metrik dosyası bulunamadı!")
        print(f"Lütfen önce grafik üretimi yapın:")
        print(f"  python make_graphs_yolo.py --targets {args.targets} --out graphs")
        sys.exit(1)
    
    # 3. Metrikleri yükle
    metrics_df = load_metrics(metric_files)
    
    # 4. Targets ile birleştir
    merged_df = merge_with_targets(metrics_df, targets_df)
    
    # 5. Feature'ları hazırla
    feature_cols = ['period', 'duration', 'depth', 'snr', 'odd_even_depth_ratio']
    df_clean = prepare_features(merged_df, feature_cols)
    
    if len(df_clean) < 10:
        print(f"\nHATA: Yeterli veri yok ({len(df_clean)} örnek)")
        sys.exit(1)
    
    # 6. Split
    train_df, val_df, test_df = stratified_split_by_target(
        df_clean,
        train_size=args.train,
        val_size=args.val,
        random_state=args.seed
    )
    
    # 7. X, y hazırla
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df['label'].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values
    
    print(f"\n✓ Feature'lar: {feature_cols}")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    
    # 8. Modelleri eğit
    print("\n" + "="*60)
    print("MODEL EĞİTİMİ")
    print("="*60)
    
    all_metrics = []
    
    # Logistic Regression
    scaler_lr = StandardScaler()
    lr_model = LogisticRegression(random_state=args.seed, max_iter=1000)
    lr_metrics = train_evaluate_model(
        lr_model, 'LogisticRegression',
        X_train, y_train, X_test, y_test,
        scaler=scaler_lr
    )
    all_metrics.append(lr_metrics)
    
    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=args.seed,
        max_depth=10
    )
    rf_metrics = train_evaluate_model(
        rf_model, 'RandomForest',
        X_train, y_train, X_test, y_test,
        scaler=None
    )
    all_metrics.append(rf_metrics)
    
    # 9. Sonuçları kaydet
    print("\n" + "="*60)
    print("SONUÇLAR")
    print("="*60)
    
    results_df = pd.DataFrame(all_metrics)
    
    # Timestamp ekle
    from datetime import datetime
    results_df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    results_df['num_features'] = len(feature_cols)
    results_df['train_samples'] = len(train_df)
    results_df['test_samples'] = len(test_df)
    
    # Kaydet
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Baseline metrikler kaydedildi: {output_path}")
    
    # Karşılaştırma tablosu
    print("\n" + "="*60)
    print("BASELINE MODEL KARŞILAŞTIRMASI")
    print("="*60)
    print("\n| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |")
    print("|-------|----------|-----------|--------|----------|---------|")
    
    for _, row in results_df.iterrows():
        print(f"| {row['model']:<20} | {row['accuracy']:.4f} | {row['precision']:.4f} | "
              f"{row['recall']:.4f} | {row['f1_score']:.4f} | {row['roc_auc']:.4f} |")
    
    print("\n" + "="*60)
    print("ℹ NOT: Bu baseline modeller sadece REFERANS içindir.")
    print("Asıl model: YOLOv8 Classification (görüntü tabanlı)")
    print("="*60)
    
    print("\nYOLOv8 ile karşılaştırmak için:")
    print(f"  cat {args.output}")
    print(f"  cat evaluation_results/summary.csv")


if __name__ == "__main__":
    main()

