import streamlit as st
import pandas as pd
from ultralytics import YOLO
from pathlib import Path

st.set_page_config(page_title="Exoplanet Transit Classifier", layout="wide")
st.title("Exoplanet Transit Classifier (YOLOv8-CLS, DB-less)")

# load
@st.cache_data
def load_index():
    return pd.read_csv("index.csv")

@st.cache_resource
def load_model():
    path = Path("models/best.pt")
    if not path.exists():
        st.warning("Model not found at models/best.pt. Train first.")
        return None
    return YOLO(str(path))

df = load_index()
model = load_model()

stars = sorted(df["target"].unique().tolist())
col1, col2 = st.columns([2,1])
with col1:
    star = st.selectbox("Select a star", stars)
with col2:
    show_binned_only = st.checkbox("Show binned only", value=False)

subset = df[df["target"] == star].copy()
if show_binned_only:
    subset = subset[subset["image_path"].str.contains("binned", case=False)]

if subset.empty:
    st.info("No images found for this star.")
else:
    cols = st.columns(3)
    for i, (_, r) in enumerate(subset.iterrows()):
        with cols[i % 3]:
            st.image(r["image_path"], caption=Path(r["image_path"]).name, use_column_width=True)
            if model:
                out = model.predict(source=r["image_path"], verbose=False)
                probs = out[0].probs.data.cpu().tolist()
                pos_prob = probs[1] if len(probs) > 1 else 0.0
                st.metric("Candidate probability", f"{pos_prob:.1%}")


