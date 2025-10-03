# Exoplanet Transit Classifier (YOLOv8-CLS, DB-less)

**What it does**
- Takes phase-folded light-curve PNGs/JPGs per star (e.g., `Kepler-40_Kepler_phase.png`).
- Uses `targets.csv` (`target,mission,label`) to map stars to labels (0=negative, 1=positive).
- Builds a file-based dataset, trains a YOLOv8 classification model, and serves a Streamlit UI.

**Quickstart**
```bash
# 1) env
pip install -r requirements.txt

# 2) tell the builder where your raw images live
# (edit the RAW_DIR variable when first run)
python scripts/01_build_index.py

# 3) create stratified splits and copy images into data/plots/
python scripts/02_split_build_dataset.py --train 0.7 --val 0.15 --test 0.15

# 4) train
bash scripts/03_train_yolov8_cls.sh

# 5) batch score (optional)
python scripts/05_batch_score_all.py

# 6) run UI
streamlit run app/ui.py
```


Data expectations

`targets.csv` must contain: `target,mission,label`

Image filenames must contain the target substring (case-insensitive).

Put all your phase-folded images under `data/raw_images/` or set a custom `RAW_DIR`.

Outputs

Trained weights under `runs/classify/.../weights/best.pt` (auto-copied into `models/`).

`batch_scores.csv` with per-image probabilities.

No database is used. Everything is CSV + folders.
