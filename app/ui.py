#!/usr/bin/env python3
"""
Streamlit UI - NASA Exoplanet Detection
Alternatif arayüz: Flask'tan bağımsız, doğrudan Python ile çalışır
Komut: streamlit run app/ui.py
"""

import streamlit as st
import sys
import subprocess
import tempfile
from pathlib import Path
from PIL import Image
import pandas as pd
import time

# Ultralytics import
try:
    from ultralytics import YOLO
except ImportError:
    st.error("HATA: ultralytics kütüphanesi bulunamadı!")
    st.code("pip install ultralytics>=8.2.0")
    st.stop()


# Sayfa yapılandırması
st.set_page_config(
    page_title="NASA Exoplanet Detection",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    h1 {
        color: #667eea;
        text-align: center;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #a0a0a0;
        margin-bottom: 2rem;
    }
    .success-box {
        background: rgba(74, 222, 128, 0.1);
        border: 2px solid #4ade80;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background: rgba(248, 113, 113, 0.1);
        border: 2px solid #f87171;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """
    YOLOv8 modelini yükle (cached).
    
    Returns:
        YOLO model veya None
    """
    model_path = Path('models/best.pt')
    
    if not model_path.exists():
        return None
    
    try:
        model = YOLO(str(model_path))
        return model
    except Exception as e:
        st.error(f"Model yükleme hatası: {e}")
        return None


def find_existing_phase_image(target_name, mission):
    """
    graphs/images klasöründe mevcut phase görselini ara.
    
    Args:
        target_name: Hedef adı
        mission: Mission adı
        
    Returns:
        Path veya None
    """
    graphs_dir = Path('graphs/images')
    
    if not graphs_dir.exists():
        return None
    
    # Normalize
    normalized_target = target_name.replace(' ', '_').replace('-', '_')
    normalized_mission = mission.replace(' ', '_')
    
    # Aramalar
    search_patterns = [
        f"{normalized_target}_{normalized_mission}_phase.png",
        f"{normalized_target}_{normalized_mission}_phasefold.png",
        f"{normalized_target}_*_phase*.png"
    ]
    
    for pattern in search_patterns:
        matches = list(graphs_dir.glob(pattern))
        if matches:
            return matches[0]
    
    return None


def generate_phase_image(target_name, mission, progress_callback=None):
    """
    Step-2 pipeline çağırıp phase görseli üret.
    
    Args:
        target_name: Hedef adı
        mission: Mission adı
        progress_callback: İlerleme callback fonksiyonu
        
    Returns:
        Path veya None
    """
    if progress_callback:
        progress_callback("Downloading light curve data...")
    
    # Geçici klasör
    temp_dir = Path(tempfile.gettempdir()) / 'exoplanet_temp'
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Script çağrısı
        cmd = [
            sys.executable,
            '01_download_clean_bls_fast.py',
            '--target', target_name,
            '--mission', mission,
            '--out', str(temp_dir),
            '--min_period', '0.5',
            '--max_period', '20'
        ]
        
        if progress_callback:
            progress_callback("Running BLS analysis...")
        
        # Subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 dakika
        )
        
        if result.returncode != 0:
            st.error(f"Pipeline hatası: {result.stderr[:500]}")
            return None
        
        if progress_callback:
            progress_callback("Generating plots...")
        
        # Üretilen dosyayı bul
        safe_target = target_name.replace(' ', '_').replace('-', '_')
        safe_mission = mission.replace(' ', '_')
        
        # Olası dosya adları
        possible_names = [
            f"{safe_target}_{safe_mission}_phasefold.png",
            f"{safe_target}_{safe_mission}_phase.png"
        ]
        
        for name in possible_names:
            phase_file = temp_dir / name
            if phase_file.exists():
                return phase_file
        
        # Joker arama
        for phase_pattern in ['*phase*.png', '*phasefold*.png']:
            matches = list(temp_dir.glob(phase_pattern))
            if matches:
                return matches[0]
        
        return None
        
    except subprocess.TimeoutExpired:
        st.error("Timeout: İşlem çok uzun sürdü (>5 dakika)")
        return None
    except Exception as e:
        st.error(f"Hata: {e}")
        return None


def predict_image(model, image_path):
    """
    Görseli YOLO modeline ver ve tahmin al.
    
    Args:
        model: YOLO modeli
        image_path: Görsel yolu
        
    Returns:
        dict: Tahmin sonuçları
    """
    try:
        # Tahmin
        results = model.predict(
            source=str(image_path),
            verbose=False,
            imgsz=224
        )
        
        if not results or len(results) == 0:
            return None
        
        result = results[0]
        
        if not hasattr(result, 'probs') or result.probs is None:
            return None
        
        probs = result.probs
        prob_values = probs.data.cpu().numpy()
        
        # Predicted class
        pred_class = int(probs.top1)
        
        # Probabilities (negative=0, positive=1)
        prob_negative = float(prob_values[0])
        prob_positive = float(prob_values[1])
        
        # Prediction
        is_exoplanet = (pred_class == 1)
        prediction = "Exoplanet Candidate ✓" if is_exoplanet else "No Exoplanet ✗"
        confidence = prob_positive if is_exoplanet else prob_negative
        
        return {
            'prediction': prediction,
            'is_exoplanet': is_exoplanet,
            'confidence': confidence,
            'prob_positive': prob_positive,
            'prob_negative': prob_negative,
            'predicted_class': pred_class
        }
        
    except Exception as e:
        st.error(f"Tahmin hatası: {e}")
        return None


def main():
    """Ana fonksiyon"""
    
    # Header
    st.markdown("<h1>🌌 NASA Exoplanet Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>AI-Powered Transit Detection System</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Bilgi")
        st.markdown("""
        Bu sistem, yıldızların ışık eğrilerini analiz ederek 
        **exoplanet (öte gezegen)** geçişlerini tespit eder.
        
        **Nasıl Çalışır?**
        1. Hedef yıldızı seçin
        2. NASA verilerinden ışık eğrisi indirilir
        3. BLS algoritması ile analiz edilir
        4. YOLOv8 CNN modeli tahmin yapar
        
        **Desteklenen Misyonlar:**
        - Kepler
        - TESS
        - K2
        """)
        
        st.divider()
        
        st.header("📊 Model Bilgisi")
        model = load_model()
        if model:
            st.success("✓ Model yüklü")
            model_path = Path('models/best.pt')
            size_mb = model_path.stat().st_size / (1024 * 1024)
            st.metric("Model Boyutu", f"{size_mb:.2f} MB")
        else:
            st.error("✗ Model yüklenemedi")
            st.info("Lütfen önce modeli eğitin:")
            st.code("python scripts/03_train_yolov8_cls.py --model yolov8n-cls.pt --data_dir data/plots")
    
    # Ana içerik
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎯 Hedef Seçimi")
        
        # Target input
        target_name = st.text_input(
            "Target Name",
            placeholder="örn: Kepler-10, TOI 700, Kepler-22",
            help="NASA veri tabanındaki hedef yıldız adı"
        )
        
        # Mission selection
        mission = st.selectbox(
            "Mission",
            options=["Kepler", "TESS", "K2"],
            index=0,
            help="Hangi görevden veri indirilecek"
        )
        
        # Hızlı örnekler
        st.markdown("**Hızlı Örnekler:**")
        col_ex1, col_ex2, col_ex3 = st.columns(3)
        
        with col_ex1:
            if st.button("Kepler-10"):
                st.session_state.target_name = "Kepler-10"
                st.session_state.mission = "Kepler"
                st.rerun()
        
        with col_ex2:
            if st.button("TOI 700"):
                st.session_state.target_name = "TOI 700"
                st.session_state.mission = "TESS"
                st.rerun()
        
        with col_ex3:
            if st.button("Kepler-22"):
                st.session_state.target_name = "Kepler-22"
                st.session_state.mission = "Kepler"
                st.rerun()
        
        # Session state'ten değerleri al
        if 'target_name' in st.session_state:
            target_name = st.session_state.target_name
        if 'mission' in st.session_state:
            mission = st.session_state.mission
        
        st.divider()
        
        # Analyze butonu
        analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("📊 Sonuçlar")
        
        if analyze_clicked:
            if not target_name:
                st.error("⚠️ Lütfen bir hedef adı girin!")
            elif not model:
                st.error("⚠️ Model yüklü değil!")
            else:
                # Progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 1. Mevcut görseli ara
                status_text.text("1/4: Mevcut görseller aranıyor...")
                progress_bar.progress(25)
                
                image_path = find_existing_phase_image(target_name, mission)
                
                if image_path:
                    st.info(f"✓ Mevcut görsel bulundu: {image_path.name}")
                else:
                    # 2. Yeni görsel üret
                    status_text.text("2/4: Grafik üretiliyor (30-60 saniye)...")
                    progress_bar.progress(40)
                    
                    def update_status(msg):
                        status_text.text(f"2/4: {msg}")
                    
                    image_path = generate_phase_image(target_name, mission, update_status)
                    
                    if not image_path:
                        progress_bar.empty()
                        status_text.empty()
                        st.error("⚠️ Grafik üretilemedi! Hedef adını kontrol edin.")
                        st.stop()
                    
                    st.success(f"✓ Grafik üretildi: {image_path.name}")
                
                # 3. Tahmin
                status_text.text("3/4: Tahmin yapılıyor...")
                progress_bar.progress(75)
                
                prediction_result = predict_image(model, image_path)
                
                if not prediction_result:
                    progress_bar.empty()
                    status_text.empty()
                    st.error("⚠️ Tahmin başarısız!")
                    st.stop()
                
                # 4. Sonuçları göster
                status_text.text("4/4: Sonuçlar hazırlanıyor...")
                progress_bar.progress(100)
                time.sleep(0.5)
                
                progress_bar.empty()
                status_text.empty()
                
                # Sonuç kutusu
                if prediction_result['is_exoplanet']:
                    st.markdown("<div class='success-box'>", unsafe_allow_html=True)
                    st.markdown(f"### {prediction_result['prediction']}")
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='error-box'>", unsafe_allow_html=True)
                    st.markdown(f"### {prediction_result['prediction']}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Görseli göster
                st.image(str(image_path), caption="Phase-Folded Light Curve", use_container_width=True)
                
                # Confidence bar
                st.metric(
                    "Confidence",
                    f"{prediction_result['confidence']*100:.2f}%"
                )
                st.progress(prediction_result['confidence'])
                
                # Detaylar
                st.divider()
                st.markdown("**Detaylar:**")
                
                col_d1, col_d2 = st.columns(2)
                
                with col_d1:
                    st.metric("Target", target_name)
                    st.metric("Positive Prob.", f"{prediction_result['prob_positive']*100:.2f}%")
                
                with col_d2:
                    st.metric("Mission", mission)
                    st.metric("Negative Prob.", f"{prediction_result['prob_negative']*100:.2f}%")
                
                # DataFrame
                st.divider()
                st.markdown("**Sonuç Tablosu:**")
                
                result_df = pd.DataFrame([{
                    'Target': target_name,
                    'Mission': mission,
                    'Prediction': prediction_result['prediction'],
                    'Confidence': f"{prediction_result['confidence']*100:.2f}%",
                    'Positive Prob': f"{prediction_result['prob_positive']*100:.2f}%",
                    'Negative Prob': f"{prediction_result['prob_negative']*100:.2f}%"
                }])
                
                st.dataframe(result_df, use_container_width=True)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #a0a0a0; font-size: 0.9rem;'>
        NASA Exoplanet Detection System | YOLOv8 Classification | 
        <a href='https://github.com' style='color: #667eea;'>GitHub</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

