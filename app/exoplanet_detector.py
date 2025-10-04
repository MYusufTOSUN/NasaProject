#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ekzoplanet Tespit Sistemi - Flask Backend
Tüm UI özellikleri için API endpoint'leri sağlar
"""

import os
import io
import base64
import random
from pathlib import Path

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

try:
    from ultralytics import YOLO
    import lightkurve as lk
    from astropy.timeseries import BoxLeastSquares
except ImportError as e:
    print(f"⚠ UYARI: Bazı kütüphaneler yüklü değil: {e}")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Global değişkenler
MODEL_PATH = 'models/best.pt'
METADATA_PATH = 'data/metadata/metadata1500.csv'
model = None
metadata_df = None


def load_model():
    """Model yükle"""
    global model
    if model is None and os.path.exists(MODEL_PATH):
        try:
            model = YOLO(MODEL_PATH)
            print(f"✓ Model yüklendi: {MODEL_PATH}")
        except Exception as e:
            print(f"⚠ Model yüklenemedi: {e}")
            model = None
    return model


def load_metadata():
    """Metadata yükle"""
    global metadata_df
    if metadata_df is None and os.path.exists(METADATA_PATH):
        try:
            metadata_df = pd.read_csv(METADATA_PATH)
            print(f"✓ Metadata yüklendi: {len(metadata_df)} kayıt")
        except Exception as e:
            print(f"⚠ Metadata yüklenemedi: {e}")
            metadata_df = None
    return metadata_df


def get_metadata_for_target(target_name):
    """Metadata'dan hedef bilgilerini al"""
    df = load_metadata()
    if df is None:
        return None
    
    # Hedefi bul (case-insensitive)
    matches = df[df['target'].str.lower() == target_name.lower()]
    if len(matches) > 0:
        return matches.iloc[0].to_dict()
    
    return None


def download_and_clean_lc(target, mission='auto'):
    """Işık eğrisi indir ve temizle"""
    try:
        search_result = lk.search_lightcurve(target, mission=mission if mission != 'auto' else None)
        if len(search_result) == 0:
            return None, None
        
        lc_collection = search_result.download_all()
        if lc_collection is None or len(lc_collection) == 0:
            return None, None
        
        lc = lc_collection.stitch()
        detected_mission = search_result[0].mission[0] if hasattr(search_result[0], 'mission') else 'Unknown'
        
        # Temizle
        lc = lc.remove_nans().remove_outliers(sigma=5)
        lc_flat = lc.flatten(window_length=2001)
        
        return lc_flat, detected_mission
    except Exception as e:
        print(f"LC indirme hatası: {e}")
        return None, None


def find_period_bls(lc):
    """BLS ile period bul"""
    try:
        bls = BoxLeastSquares(lc.time.value, lc.flux.value)
        periods = np.linspace(0.5, 20.0, 3000)
        results = bls.power(periods, duration=np.linspace(0.01, 0.2, 10))
        
        best_period = results.period[np.argmax(results.power)]
        stats = bls.compute_stats(best_period, duration=np.linspace(0.01, 0.2, 10))
        
        return {
            'period': float(best_period),
            't0': float(stats['transit_time'][np.argmax(stats['depth'])]),
            'duration': float(stats['duration'][np.argmax(stats['depth'])])
        }
    except Exception as e:
        print(f"BLS hatası: {e}")
        return None


def create_raw_plot_base64(lc, target_name):
    """Ham ışık eğrisi görseli (base64)"""
    try:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=96)
        ax.plot(lc.time.value, lc.flux.value, 'k.', markersize=1)
        ax.set_xlabel('Time (BKJD)')
        ax.set_ylabel('Normalized Flux')
        ax.set_title(f'{target_name} - Raw Light Curve')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=96, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"Raw plot hatası: {e}")
        return None


def create_phase_plot_base64(lc, period, t0, target_name):
    """Faz-katlanmış görsel (base64)"""
    try:
        lc_folded = lc.fold(period=period, epoch_time=t0)
        lc_binned = lc_folded.bin(bins=100)
        
        fig, ax = plt.subplots(figsize=(8, 6), dpi=96)
        ax.scatter(lc_folded.phase.value, lc_folded.flux.value, s=1, alpha=0.3, c='gray', label='Data')
        ax.plot(lc_binned.phase.value, lc_binned.flux.value, 'r-', linewidth=2, label='Binned')
        ax.set_xlabel('Phase')
        ax.set_ylabel('Normalized Flux')
        ax.set_title(f'{target_name} - P={period:.4f}d')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=96, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"Phase plot hatası: {e}")
        return None


def predict_image(image_pil):
    """Görsel üzerinde model tahmini yap"""
    mdl = load_model()
    if mdl is None:
        return None, None
    
    try:
        # Geçici kaydet
        temp_path = 'app/static/temp/temp_input.png'
        os.makedirs('app/static/temp', exist_ok=True)
        image_pil.save(temp_path)
        
        # Tahmin
        results = mdl(temp_path, verbose=False)
        result = results[0]
        
        probs = result.probs.data.cpu().numpy()
        class_names = result.names
        
        # Indexleri bul
        neg_idx = 0 if class_names[0] == 'negative' else 1
        pos_idx = 1 if class_names[1] == 'positive' else 0
        
        pred_label = 'positive' if probs[pos_idx] > probs[neg_idx] else 'negative'
        confidence = float(probs[pos_idx] if pred_label == 'positive' else probs[neg_idx])
        
        return pred_label, {
            'positive': float(probs[pos_idx]),
            'negative': float(probs[neg_idx])
        }
    except Exception as e:
        print(f"Tahmin hatası: {e}")
        return None, None


def create_saliency_map(image_pil):
    """Basit occlusion saliency map (opsiyonel)"""
    try:
        # Placeholder - gerçek saliency map için daha karmaşık kod gerekir
        # Şimdilik orijinal görseli döndür
        buf = io.BytesIO()
        image_pil.save(buf, format='PNG')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"Saliency map hatası: {e}")
        return None


def get_similar_planets(depth_ppm, count=3):
    """Benzer gezegenleri bul (depth_ppm yakınlığına göre)"""
    df = load_metadata()
    if df is None or pd.isna(depth_ppm):
        return []
    
    # Depth_ppm olanları filtrele
    df_valid = df[df['depth_ppm'].notna()].copy()
    if len(df_valid) == 0:
        return []
    
    # Farkı hesapla
    df_valid['depth_diff'] = abs(df_valid['depth_ppm'] - depth_ppm)
    df_sorted = df_valid.sort_values('depth_diff').head(count)
    
    similar = []
    for _, row in df_sorted.iterrows():
        similar.append({
            'name': row['target'],
            'depth_ppm': row['depth_ppm'],
            'archive_url': row.get('archive_url', '#')
        })
    
    return similar


def generate_discovery_name(user_input):
    """Kullanıcı girişinden discovery adı üret"""
    # Basit: ilk harfler + sayı
    clean = ''.join(e for e in user_input if e.isalnum())
    if len(clean) < 2:
        clean = 'User'
    
    discovery_num = random.randint(1, 9999)
    return f"{clean[:5]}-{discovery_num}b"


@app.route('/')
def index():
    """Ana sayfa"""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Ana analiz endpoint'i
    - Hedef adı ile analiz
    - Görsel yükleme ile analiz
    """
    try:
        # Veri tipini kontrol et
        if request.content_type and 'multipart/form-data' in request.content_type:
            # Form-data (görsel upload)
            return handle_image_upload(request)
        else:
            # JSON (hedef adı)
            return handle_target_analysis(request)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def handle_target_analysis(req):
    """Hedef adı ile analiz"""
    data = req.get_json()
    target_name = data.get('target_name', '')
    mission = data.get('mission', 'auto')
    
    if not target_name:
        return jsonify({'error': 'Target name gerekli'}), 400
    
    print(f"🎯 Analiz başlatılıyor: {target_name}")
    
    # 1. Metadata'dan bilgi al
    meta_info = get_metadata_for_target(target_name)
    period = None
    t0 = None
    duration = None
    
    if meta_info:
        period = meta_info.get('period')
        t0 = meta_info.get('t0')
        duration = meta_info.get('duration')
        if pd.notna(period) and pd.notna(t0) and pd.notna(duration):
            print(f"  ✓ Metadata'dan period alındı: P={period:.4f}d")
    
    # 2. Işık eğrisi indir
    lc, detected_mission = download_and_clean_lc(target_name, mission)
    if lc is None:
        return jsonify({'error': 'Işık eğrisi indirilemedi'}), 500
    
    print(f"  ✓ LC indirildi: {detected_mission}")
    
    # 3. Period yoksa BLS ile bul
    if period is None or pd.isna(period):
        print("  ⏳ BLS ile period bulunuyor...")
        bls_result = find_period_bls(lc)
        if bls_result is None:
            return jsonify({'error': 'Period bulunamadı'}), 500
        period = bls_result['period']
        t0 = bls_result['t0']
        duration = bls_result['duration']
        print(f"  ✓ BLS: P={period:.4f}d")
    
    # 4. Görseller oluştur
    raw_plot_b64 = create_raw_plot_base64(lc, target_name)
    phase_plot_b64 = create_phase_plot_base64(lc, period, t0, target_name)
    
    if phase_plot_b64 is None:
        return jsonify({'error': 'Görsel oluşturulamadı'}), 500
    
    # 5. Phase görselini geçici kaydet ve model ile tahmin yap
    # Base64'ten PIL'e çevir
    phase_img_data = base64.b64decode(phase_plot_b64.split(',')[1])
    phase_img = Image.open(io.BytesIO(phase_img_data))
    
    pred_label, probs = predict_image(phase_img)
    
    if pred_label is None:
        return jsonify({'error': 'Model tahmini başarısız'}), 500
    
    confidence = probs['positive'] if pred_label == 'positive' else probs['negative']
    
    # 6. Discovery Card bilgileri
    star_info = {}
    if meta_info:
        star_info = {
            'archive_url': meta_info.get('archive_url', '#'),
            'nasa_confirmed': meta_info.get('label', '').lower() == 'positive',
            'discovery_method': 'Transit',
            'discovery_year': 'Unknown',
            'discovery_facility': detected_mission
        }
    else:
        star_info = {
            'archive_url': '#',
            'nasa_confirmed': False,
            'discovery_method': 'Transit',
            'discovery_year': 'Unknown',
            'discovery_facility': detected_mission
        }
    
    # 7. Benzer gezegenler
    depth_ppm = meta_info.get('depth_ppm') if meta_info else None
    similar = get_similar_planets(depth_ppm, count=3)
    
    # 8. Mission badge
    mission_badge = '🔵 Kepler' if 'kepler' in detected_mission.lower() else '🔴 TESS'
    
    # Yanıt
    response = {
        'success': True,
        'prediction': pred_label,
        'confidence': float(confidence * 100),  # Yüzde
        'probs': {
            'positive': float(probs['positive'] * 100),
            'negative': float(probs['negative'] * 100)
        },
        'plots': {
            'raw': raw_plot_b64,
            'phase': phase_plot_b64
        },
        'star_info': star_info,
        'mission_badge': mission_badge,
        'similar': similar,
        'target_name': target_name
    }
    
    return jsonify(response)


def handle_image_upload(req):
    """Görsel yükleme ile analiz"""
    if 'file' not in req.files:
        return jsonify({'error': 'Dosya bulunamadı'}), 400
    
    file = req.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400
    
    # Görseli aç
    try:
        img = Image.open(file.stream)
        img = img.convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Görsel açılamadı: {e}'}), 400
    
    # Tahmin
    pred_label, probs = predict_image(img)
    
    if pred_label is None:
        return jsonify({'error': 'Model tahmini başarısız'}), 500
    
    confidence = probs['positive'] if pred_label == 'positive' else probs['negative']
    
    # Saliency map (opsiyonel)
    saliency_b64 = create_saliency_map(img)
    
    # Discovery name üret
    user_name = req.form.get('user_name', 'User')
    discovery_name = generate_discovery_name(user_name)
    
    # Yanıt
    response = {
        'success': True,
        'prediction': pred_label,
        'confidence': float(confidence * 100),
        'probs': {
            'positive': float(probs['positive'] * 100),
            'negative': float(probs['negative'] * 100)
        },
        'plots': {
            'saliency': saliency_b64
        },
        'discovery_name': discovery_name,
        'star_info': {
            'nasa_confirmed': False,
            'archive_url': '#'
        }
    }
    
    return jsonify(response)


if __name__ == '__main__':
    print("="*60)
    print("🪐 Ekzoplanet Tespit Sistemi - Flask Backend")
    print("="*60)
    
    # Model ve metadata yükle
    load_model()
    load_metadata()
    
    print("\n🌐 Server başlatılıyor...")
    print("   http://localhost:5000")
    print("\n⌨️  Durdurmak için Ctrl+C\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

