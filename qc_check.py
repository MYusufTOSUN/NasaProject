#!/usr/bin/env python3
"""
Pipeline Kalite Kontrol Scripti
Proje dosyalarının ve klasör yapısının tutarlılığını kontrol eder.
"""

import sys
from pathlib import Path
import pandas as pd


# ASCII ikonları
CHECK = "✓"  # Başarılı
CROSS = "✗"  # Başarısız
WARN = "⚠"   # Uyarı
INFO = "ℹ"   # Bilgi


def print_section(title):
    """Bölüm başlığı yazdır"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print('='*60)


def check_item(condition, item_name, critical=True):
    """
    Tek bir item'ı kontrol et ve sonucu yazdır.
    
    Args:
        condition: Kontrol koşulu (True/False)
        item_name: Item adı
        critical: Kritik mi (False ise uyarı, True ise hata)
        
    Returns:
        tuple: (passed, is_critical_failure)
    """
    if condition:
        print(f"  {CHECK} {item_name}")
        return True, False
    else:
        if critical:
            print(f"  {CROSS} {item_name} [KRİTİK]")
            return False, True
        else:
            print(f"  {WARN} {item_name} [UYARI]")
            return False, False


def check_targets_csv():
    """
    targets.csv dosyasını kontrol et.
    
    Returns:
        tuple: (passed, critical_failures)
    """
    print_section("1. TARGETS.CSV KONTROLÜ")
    
    critical_failures = []
    
    # Dosya varlığı
    targets_path = Path('targets.csv')
    if not targets_path.exists():
        print(f"  {CROSS} targets.csv bulunamadı [KRİTİK]")
        return False, 1
    
    print(f"  {CHECK} targets.csv mevcut")
    
    # Dosya içeriği
    try:
        df = pd.read_csv(targets_path)
        
        # Başlık kontrolü
        required_cols = ['target', 'mission', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"  {CROSS} Eksik kolonlar: {missing_cols} [KRİTİK]")
            critical_failures.append('targets_csv_columns')
        else:
            print(f"  {CHECK} Tüm gerekli kolonlar mevcut: {required_cols}")
        
        # Satır sayısı
        if len(df) == 0:
            print(f"  {CROSS} Dosya boş! [KRİTİK]")
            critical_failures.append('targets_csv_empty')
        else:
            print(f"  {CHECK} {len(df)} hedef bulundu")
        
        # Label dağılımı
        if 'label' in df.columns:
            label_dist = df['label'].value_counts().to_dict()
            positive = label_dist.get(1, 0)
            negative = label_dist.get(0, 0)
            print(f"  {INFO} Label dağılımı: Positive={positive}, Negative={negative}")
        
        # Mission dağılımı
        if 'mission' in df.columns:
            missions = df['mission'].unique()
            print(f"  {INFO} Mission'lar: {', '.join(missions)}")
        
        return len(critical_failures) == 0, len(critical_failures)
        
    except Exception as e:
        print(f"  {CROSS} Dosya okuma hatası: {e} [KRİTİK]")
        return False, 1


def check_data_plots():
    """
    data/plots/ klasör yapısını ve dosya sayılarını kontrol et.
    
    Returns:
        tuple: (passed, critical_failures)
    """
    print_section("2. DATA/PLOTS/ KLASÖR YAPISI")
    
    critical_failures = []
    
    # Ana klasör
    data_plots = Path('data/plots')
    if not data_plots.exists():
        print(f"  {CROSS} data/plots/ klasörü bulunamadı [KRİTİK]")
        return False, 1
    
    print(f"  {CHECK} data/plots/ mevcut")
    
    # Alt klasörler
    splits = ['train', 'val', 'test']
    labels = ['positive', 'negative']
    
    total_images = 0
    empty_folders = []
    
    for split in splits:
        split_total = 0
        print(f"\n  {split.upper()} Set:")
        
        for label in labels:
            folder_path = data_plots / split / label
            
            if not folder_path.exists():
                print(f"    {CROSS} {split}/{label}/ klasörü yok [KRİTİK]")
                critical_failures.append(f'{split}_{label}_missing')
                continue
            
            # PNG/JPG dosyalarını say
            images = list(folder_path.glob('*.png')) + list(folder_path.glob('*.jpg'))
            count = len(images)
            split_total += count
            total_images += count
            
            if count == 0:
                print(f"    {WARN} {split}/{label}/: BOŞ (0 görsel)")
                empty_folders.append(f"{split}/{label}")
            else:
                print(f"    {CHECK} {split}/{label}/: {count} görsel")
        
        print(f"    {INFO} {split} toplam: {split_total} görsel")
    
    print(f"\n  {INFO} Toplam görsel sayısı: {total_images}")
    
    # Boş klasör uyarısı
    if empty_folders:
        print(f"  {WARN} Boş klasörler: {', '.join(empty_folders)}")
    else:
        print(f"  {CHECK} Boş klasör yok")
    
    # Kritik: Hiç görsel yoksa
    if total_images == 0:
        print(f"  {CROSS} Hiç görsel bulunamadı! [KRİTİK]")
        critical_failures.append('no_images')
    
    return len(critical_failures) == 0, len(critical_failures)


def check_model():
    """
    models/best.pt dosyasını kontrol et.
    
    Returns:
        tuple: (passed, critical_failures)
    """
    print_section("3. MODEL DOSYASI")
    
    model_path = Path('models/best.pt')
    
    if not model_path.exists():
        print(f"  {CROSS} models/best.pt bulunamadı [KRİTİK]")
        print(f"  {INFO} Model henüz eğitilmemiş olabilir")
        return False, 1
    
    print(f"  {CHECK} models/best.pt mevcut")
    
    # Dosya boyutu
    size_mb = model_path.stat().st_size / (1024 * 1024)
    
    if size_mb < 1:
        print(f"  {WARN} Model boyutu çok küçük: {size_mb:.2f} MB")
        print(f"  {INFO} Model bozuk olabilir")
        return False, 0
    else:
        print(f"  {CHECK} Model boyutu: {size_mb:.2f} MB")
    
    return True, 0


def check_evaluation_results():
    """
    evaluation_results/summary.csv dosyasını kontrol et.
    
    Returns:
        tuple: (passed, critical_failures)
    """
    print_section("4. DEĞERLENDİRME SONUÇLARI")
    
    summary_path = Path('evaluation_results/summary.csv')
    
    if not summary_path.exists():
        print(f"  {CROSS} evaluation_results/summary.csv bulunamadı [KRİTİK]")
        print(f"  {INFO} Model henüz değerlendirilmemiş olabilir")
        return False, 1
    
    print(f"  {CHECK} evaluation_results/summary.csv mevcut")
    
    # Dosya içeriği
    try:
        df = pd.read_csv(summary_path)
        
        # Metrik kolonları
        expected_cols = ['timestamp', 'accuracy', 'precision', 'recall', 'f1_score']
        present_cols = [col for col in expected_cols if col in df.columns]
        
        if len(present_cols) < len(expected_cols):
            missing = set(expected_cols) - set(present_cols)
            print(f"  {WARN} Bazı metrikler eksik: {missing}")
        else:
            print(f"  {CHECK} Tüm temel metrikler mevcut")
        
        # Son değerlendirme tarihi
        if 'timestamp' in df.columns and len(df) > 0:
            last_eval = df.iloc[-1]['timestamp']
            print(f"  {INFO} Son değerlendirme: {last_eval}")
        
        # Metrik değerleri
        if 'accuracy' in df.columns and len(df) > 0:
            accuracy = df.iloc[-1]['accuracy']
            print(f"  {INFO} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return True, 0
        
    except Exception as e:
        print(f"  {WARN} Dosya okuma hatası: {e}")
        return False, 0


def check_additional_files():
    """
    Ek dosyaları kontrol et (kritik değil).
    
    Returns:
        tuple: (passed, critical_failures)
    """
    print_section("5. EK DOSYALAR (Opsiyonel)")
    
    optional_files = {
        'index.csv': 'Ham görsel index dosyası',
        'evaluation_results/predictions_detail.csv': 'Test seti tahminleri',
        'evaluation_results/confusion_matrix.png': 'Confusion matrix grafiği',
        'evaluation_results/roc_curve.png': 'ROC curve grafiği',
        'MODEL_PERFORMANCE_REPORT.md': 'Performans raporu',
        'graphs/images': 'Üretilen grafikler klasörü'
    }
    
    for file_path, description in optional_files.items():
        path = Path(file_path)
        if path.exists():
            if path.is_dir():
                # Klasör ise dosya sayısını göster
                files = list(path.glob('**/*.*'))
                print(f"  {CHECK} {file_path}: {len(files)} dosya - {description}")
            else:
                print(f"  {CHECK} {file_path}: {description}")
        else:
            print(f"  {WARN} {file_path}: Yok - {description}")
    
    return True, 0  # Opsiyonel olduğu için her zaman True


def check_scripts():
    """
    Script dosyalarını kontrol et.
    
    Returns:
        tuple: (passed, critical_failures)
    """
    print_section("6. SCRIPT DOSYALARI")
    
    critical_failures = []
    
    required_scripts = [
        ('01_download_clean_bls_fast.py', 'Tek hedef pipeline', True),
        ('make_graphs_yolo.py', 'Toplu grafik üretimi', True),
        ('scripts/01_build_index.py', 'Index oluşturucu', True),
        ('scripts/02_split_build_dataset.py', 'Dataset split', True),
        ('scripts/03_train_yolov8_cls.py', 'Model eğitimi', True),
        ('scripts/04_predict_folder.py', 'Tahmin scripti', True),
        ('scripts/05_batch_score_all.py', 'Toplu skorlama', False),
        ('scripts/evaluate_model.py', 'Değerlendirme', True),
    ]
    
    for script_path, description, critical in required_scripts:
        path = Path(script_path)
        passed, is_critical = check_item(path.exists(), f"{script_path}: {description}", critical)
        
        if is_critical:
            critical_failures.append(script_path)
    
    return len(critical_failures) == 0, len(critical_failures)


def generate_summary(results):
    """
    Özet rapor oluştur.
    
    Args:
        results: Kontrol sonuçları dictionary
    """
    print_section("ÖZET RAPOR")
    
    total_checks = len(results)
    passed_checks = sum(1 for r in results.values() if r['passed'])
    total_critical = sum(r['critical_failures'] for r in results.values())
    
    print(f"\nToplam Kontrol: {total_checks}")
    print(f"Başarılı: {passed_checks}")
    print(f"Başarısız: {total_checks - passed_checks}")
    print(f"Kritik Hata: {total_critical}")
    
    print(f"\nDetaylı Sonuçlar:")
    for check_name, result in results.items():
        icon = CHECK if result['passed'] else CROSS
        status = "BAŞARILI" if result['passed'] else "BAŞARISIZ"
        critical_info = f" ({result['critical_failures']} kritik hata)" if result['critical_failures'] > 0 else ""
        print(f"  {icon} {check_name}: {status}{critical_info}")
    
    # CSV raporu
    report_data = {
        'check': [],
        'passed': [],
        'critical_failures': []
    }
    
    for check_name, result in results.items():
        report_data['check'].append(check_name)
        report_data['passed'].append(result['passed'])
        report_data['critical_failures'].append(result['critical_failures'])
    
    report_df = pd.DataFrame(report_data)
    report_path = Path('qc_report.csv')
    report_df.to_csv(report_path, index=False)
    
    print(f"\n{INFO} Detaylı rapor kaydedildi: {report_path}")
    
    return total_critical


def main():
    """Ana fonksiyon"""
    print("="*60)
    print("NASA EXOPLANET DETECTION - KALİTE KONTROL")
    print("="*60)
    print("\nPipeline dosyalarının ve klasör yapısının tutarlılığı kontrol ediliyor...\n")
    
    # Tüm kontrolleri çalıştır
    results = {}
    
    results['targets.csv'] = {
        'passed': False,
        'critical_failures': 0
    }
    results['targets.csv']['passed'], results['targets.csv']['critical_failures'] = check_targets_csv()
    
    results['data/plots/'] = {
        'passed': False,
        'critical_failures': 0
    }
    results['data/plots/']['passed'], results['data/plots/']['critical_failures'] = check_data_plots()
    
    results['models/best.pt'] = {
        'passed': False,
        'critical_failures': 0
    }
    results['models/best.pt']['passed'], results['models/best.pt']['critical_failures'] = check_model()
    
    results['evaluation_results/'] = {
        'passed': False,
        'critical_failures': 0
    }
    results['evaluation_results/']['passed'], results['evaluation_results/']['critical_failures'] = check_evaluation_results()
    
    results['scripts'] = {
        'passed': False,
        'critical_failures': 0
    }
    results['scripts']['passed'], results['scripts']['critical_failures'] = check_scripts()
    
    results['ek_dosyalar'] = {
        'passed': False,
        'critical_failures': 0
    }
    results['ek_dosyalar']['passed'], results['ek_dosyalar']['critical_failures'] = check_additional_files()
    
    # Özet rapor
    total_critical = generate_summary(results)
    
    # Final durum
    print("\n" + "="*60)
    if total_critical == 0:
        print(f"{CHECK} TÜM KRİTİK KONTROLLER BAŞARILI!")
        print("="*60)
        print(f"\n{INFO} Proje hazır! Pipeline çalıştırılabilir.")
        sys.exit(0)
    else:
        print(f"{CROSS} {total_critical} KRİTİK HATA BULUNDU!")
        print("="*60)
        print(f"\n{WARN} Lütfen yukarıdaki hataları düzeltin.")
        print(f"{INFO} Kritik hatalar düzeltilmeden pipeline çalıştırılamaz.")
        sys.exit(1)


if __name__ == "__main__":
    main()
