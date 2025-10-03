"""
Exoplanet Detection Web UI
Premium Flask application for detecting exoplanets from light curve data
"""
import os
import io
import base64
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

# Lightkurve için ışık eğrisi indirme
try:
    from lightkurve import search_lightcurve
    HAS_LIGHTKURVE = True
except ImportError:
    HAS_LIGHTKURVE = False

app = Flask(__name__)
app.config['SECRET_KEY'] = 'exoplanet-detection-2025'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Model yolu
MODEL_PATH = Path("models/best.pt")
UPLOAD_FOLDER = Path("app/static/temp")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# Global model
model = None


def load_model():
    """Model'i yükle"""
    global model
    if model is None and MODEL_PATH.exists():
        model = YOLO(str(MODEL_PATH))
        print(f"[✓] Model yüklendi: {MODEL_PATH}")
    return model


def check_local_data(target_name):
    """
    Yerel data/plots klasöründen görsel bul
    """
    # KIC/TIC ID çıkar
    import re
    match = re.search(r'(\d+)', target_name)
    if not match:
        return None
    
    num_id = int(match.group(1))
    pad_id = f"{num_id:09d}"
    
    # Pozitif ve negatif klasörlerde ara
    search_paths = [
        Path("data/plots/train/positive"),
        Path("data/plots/val/positive"),
        Path("data/plots/test/positive"),
        Path("data/plots/train/negative"),
        Path("data/plots/val/negative"),
        Path("data/plots/test/negative"),
        Path("scripts/raw_images"),
        Path("graphs/images")
    ]
    
    for search_path in search_paths:
        if not search_path.exists():
            continue
        
        # Dosya ara
        for pattern in [f"{pad_id}*.png", f"*{target_name.replace(' ', '_')}*.png", f"*{target_name.replace(' ', '-')}*.png"]:
            matches = list(search_path.glob(pattern))
            if matches:
                return matches[0]
    
    return None


# SENTETİK VERİ FONKSİYONU KALDIRILDI
# Web UI SADECE GERÇEK VERİ KULLANIR:
# 1. Yerel dataset (data/plots, graphs/images)
# 2. MAST'tan indirilen gerçek FITS verileri


def download_and_save_lightcurve(target_name, mission=None):
    """
    MAST'tan gerçek FITS indir, grafik üret, dataset'e kaydet
    Sentetik veri YOK - sadece gerçek gözlemler
    """
    if not HAS_LIGHTKURVE:
        return None, "Lightkurve yüklü değil. Gerçek veri indirme devre dışı."
    
    try:
        print(f"[i] MAST sorgusu: {target_name} (mission={mission or 'auto'})")
        
        # Misyon belirtilmemişse otomatik
        if mission and mission.lower() != "auto":
            query = search_lightcurve(target_name, mission=mission, radius=300)
        else:
            # Tüm misyonları dene
            query = search_lightcurve(target_name, radius=300)
        
        if query is None or len(query) == 0:
            return None, f"MAST'ta '{target_name}' için veri bulunamadı"
        
        print(f"[i] {len(query)} sonuç bulundu, indiriliyor...")
        
        # İlk sonucu indir
        lc = None
        try:
            lc = query[0].download()
            print(f"[✓] Işık eğrisi indirildi")
        except Exception as e_download:
            print(f"[!] İndirme hatası: {e_download}")
            # Alternatif: Tüm sonuçları birleştir
            try:
                lc_collection = query.download_all()
                if lc_collection and len(lc_collection) > 0:
                    lc = lc_collection.stitch()
                    print(f"[✓] {len(lc_collection)} sonuç birleştirildi")
            except Exception:
                pass
        
        if lc is None:
            return None, "Işık eğrisi indirilemedi"
        
        # Time ve Flux çıkar
        try:
            time = np.array(lc.time.value, dtype=float)
            flux = np.array(lc.flux.value, dtype=float)
        except Exception:
            try:
                time = np.array(lc.time, dtype=float)
                flux = np.array(lc.flux, dtype=float)
            except Exception as e_extract:
                return None, f"Veri çıkarma hatası: {str(e_extract)}"
        
        # NaN temizle
        mask = np.isfinite(time) & np.isfinite(flux)
        time = time[mask]
        flux = flux[mask]
        
        if len(time) < 10:
            return None, f"Yetersiz veri noktası: {len(time)}"
        
        # Normalize et
        flux = flux / np.median(flux)
        mission_name = query[0].mission[0] if hasattr(query[0], 'mission') else 'Unknown'
        
        print(f"[✓] Gerçek veri: {len(time)} nokta, {mission_name}")
        return {"time": time, "flux": flux, "mission": mission_name, "target": target_name}, None
    
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg:
            return None, "MAST 404: Hedef bulunamadı"
        elif "ConnectionError" in error_msg:
            return None, "Bağlantı hatası: MAST sunucusuna erişilemiyor"
        else:
            return None, f"MAST hatası: {error_msg}"


def save_to_dataset(target_name, time, flux, prediction_result, mission):
    """
    Yeni hedefi dataset'e kaydet (sonraki eğitimde kullanılmak üzere)
    graphs/images/ altına PNG kaydet
    index.csv'ye satır ekle
    """
    try:
        import re
        from datetime import datetime
        
        # Numeric ID çıkar
        match = re.search(r'(\d+)', target_name)
        num_id = int(match.group(1)) if match else hash(target_name) % (10**9)
        pad_id = f"{num_id:09d}"
        
        # Grafik kaydet (graphs/images altına)
        output_dir = Path("graphs/images")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{pad_id}.png"
        
        # Grafik çiz ve kaydet
        fig = plt.figure(figsize=(10, 4))
        plt.plot(time, flux, 'k.', markersize=1, alpha=0.6)
        plt.xlabel("Time (days)")
        plt.ylabel("Normalized Flux")
        plt.title(f"{target_name} ({mission})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        
        print(f"[✓] Grafik kaydedildi: {output_path}")
        
        # index.csv'ye ekle
        index_csv = Path("index.csv")
        if index_csv.exists():
            # Mevcut kayıt var mı kontrol et
            df = pd.read_csv(index_csv)
            if target_name not in df['target'].values:
                # Tahmini label belirle
                predicted_label = 1 if prediction_result['prediction'] == "Exoplanet Candidate" else 0
                
                # Yeni satır ekle
                new_row = {
                    'target': target_name,
                    'mission': mission,
                    'label': predicted_label,
                    'image_path': str(output_path.absolute()),
                    'is_binned': 0,
                    'is_phase': 1
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                df.to_csv(index_csv, index=False)
                print(f"[✓] index.csv güncellendi (label={predicted_label})")
            else:
                print(f"[i] Hedef zaten index.csv'de mevcut")
        
        return True
    
    except Exception as e:
        print(f"[!] Dataset kaydetme hatası: {e}")
        return False


def search_and_download_lightcurve(target_name, mission=None):
    """
    Işık eğrisi verisi al - Önce yerel, sonra MAST'tan gerçek veri indir
    SENTETİK VERİ YOK!
    """
    # 1. Yerel dosyalarda ara
    local_file = check_local_data(target_name)
    if local_file:
        print(f"[✓] Yerel dosya bulundu: {local_file}")
        # ÖNEMLI: Yerel dosyayı direkt kullan, sentetik üretme!
        # Dosya yolunu döndür (analyze fonksiyonu bunu kullanacak)
        return {
            "local_file": str(local_file),
            "mission": "Local Dataset",
            "target": target_name
        }, None
    
    # 2. MAST'tan gerçek veri indir (sentetik yok!)
    print(f"[i] Yerel veri yok, MAST'tan indiriliyor...")
    return download_and_save_lightcurve(target_name, mission)


def plot_lightcurve(time, flux, title="Light Curve"):
    """
    Işık eğrisini çiz ve base64 string döndür
    """
    fig, ax = plt.subplots(figsize=(12, 5), facecolor='#1a1a2e')
    ax.set_facecolor('#16213e')
    
    # Scatter plot
    ax.plot(time, flux, 'o', markersize=1.5, alpha=0.6, color='#00d9ff')
    
    # Stil
    ax.set_xlabel('Time (days)', fontsize=12, color='white', fontweight='bold')
    ax.set_ylabel('Normalized Flux', fontsize=12, color='white', fontweight='bold')
    ax.set_title(title, fontsize=14, color='white', fontweight='bold', pad=15)
    ax.grid(True, alpha=0.2, color='white', linestyle='--')
    ax.tick_params(colors='white', labelsize=10)
    
    # Spine renkleri
    for spine in ax.spines.values():
        spine.set_edgecolor('#00d9ff')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    # Base64'e çevir
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, facecolor='#1a1a2e', edgecolor='none')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_base64


def save_temp_image(time, flux, filename):
    """
    Geçici PNG dosyası kaydet (model tahmini için)
    """
    filepath = UPLOAD_FOLDER / filename
    
    fig = plt.figure(figsize=(10, 4))
    plt.plot(time, flux, 'k.', markersize=1, alpha=0.6)
    plt.xlabel("Time (days)")
    plt.ylabel("Normalized Flux")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    
    return filepath


def predict_exoplanet(image_path):
    """
    Model ile tahmin yap
    """
    global model
    if model is None:
        model = load_model()
    
    if model is None:
        return None, "Model yüklenemedi"
    
    try:
        results = model.predict(source=str(image_path), verbose=False)
        
        # Probability
        probs = results[0].probs.data.cpu().numpy()
        positive_prob = float(probs[1]) if len(probs) > 1 else 0.0
        
        # Prediction
        predicted_class = "Exoplanet Candidate" if positive_prob > 0.5 else "No Exoplanet"
        confidence = positive_prob if positive_prob > 0.5 else (1 - positive_prob)
        
        return {
            "prediction": predicted_class,
            "confidence": confidence * 100,
            "positive_probability": positive_prob * 100,
            "negative_probability": (1 - positive_prob) * 100
        }, None
    
    except Exception as e:
        return None, f"Tahmin hatası: {str(e)}"


@app.route('/')
def index():
    """Ana sayfa"""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """Yıldız analizi endpoint"""
    data = request.get_json()
    
    target_name = data.get('target_name', '').strip()
    mission = data.get('mission', 'auto')
    
    if not target_name:
        return jsonify({"error": "Hedef ismi gerekli"}), 400
    
    # 1. Işık eğrisi indir veya yerel dosya bul
    lc_data, error = search_and_download_lightcurve(target_name, mission)
    if error:
        return jsonify({"error": error}), 404
    
    mission_name = lc_data.get('mission', 'Unknown')
    
    # Yerel dosya mı yoksa MAST'tan indirilen mi?
    if 'local_file' in lc_data:
        # YEREL DOSYA - direkt kullan
        local_file = Path(lc_data['local_file'])
        print(f"[i] Yerel dosya kullanılıyor: {local_file.name}")
        
        # Görseli base64'e çevir (web görüntüsü için)
        with open(local_file, 'rb') as f:
            img_bytes = f.read()
            plot_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # Model tahmini için yerel dosyayı kullan
        prediction_result, pred_error = predict_exoplanet(str(local_file))
        if pred_error:
            return jsonify({"error": pred_error}), 500
        
        # Sonuç
        result = {
            "success": True,
            "target": target_name,
            "mission": mission_name,
            "data_points": "N/A (local file)",
            "time_span": "N/A (local file)",
            "plot": plot_base64,
            "prediction": prediction_result,
            "info": "✓ Yerel dataset'ten yüklendi"
        }
        
        return jsonify(result)
    
    else:
        # MAST'TAN İNDİRİLEN - işle ve kaydet
        time = lc_data['time']
        flux = lc_data['flux']
        
        # 2. Grafik oluştur (web görüntüsü için)
        plot_base64 = plot_lightcurve(time, flux, f"{target_name} Light Curve ({mission_name})")
        
        # 3. Model tahmini için geçici dosya kaydet
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"{target_name.replace(' ', '_')}_{timestamp}.png"
        temp_image_path = save_temp_image(time, flux, temp_filename)
        
        # 4. Model tahmini
        prediction_result, pred_error = predict_exoplanet(temp_image_path)
        if pred_error:
            return jsonify({"error": pred_error}), 500
        
        # 5. Dataset'e kaydet (sonraki eğitim için)
        saved_to_dataset = save_to_dataset(target_name, time, flux, prediction_result, mission_name)
        if saved_to_dataset:
            print(f"[✓] '{target_name}' dataset'e eklendi")
        
        # 6. Geçici dosyayı temizle
        try:
            temp_image_path.unlink()
        except Exception:
            pass
        
        # 7. Sonuç döndür
        result = {
            "success": True,
            "target": target_name,
            "mission": mission_name,
            "data_points": len(time),
            "time_span": f"{time.max() - time.min():.2f} days",
            "plot": plot_base64,
            "prediction": prediction_result
        }
        
        if saved_to_dataset:
            result["info"] = "✓ Yeni hedef dataset'e kaydedildi (sonraki eğitimde kullanılacak)"
        
        return jsonify(result)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    model_loaded = model is not None or MODEL_PATH.exists()
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded,
        "lightkurve_available": HAS_LIGHTKURVE
    })


if __name__ == '__main__':
    print("=" * 60)
    print("🌟 EXOPLANET DETECTION WEB UI")
    print("=" * 60)
    
    # Model yükle
    load_model()
    
    print("\n[i] Server başlatılıyor...")
    print("[i] Tarayıcınızda açın: http://localhost:5000")
    print("[i] Durdurmak için: Ctrl+C")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)


