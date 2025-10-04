# NASA Exoplanet Detection Project

## Overview
This project uses YOLOv8 classification to detect exoplanets from light curve data. The system processes Kepler and TESS mission data to classify stellar light curves as positive (exoplanet detected) or negative (no exoplanet).

## Quickstart

### Setup Environment
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Single Target Analysis
```bash
python 01_download_clean_bls_fast.py --target "Kepler-10" --mission Kepler
```

### Batch Graph Generation
```bash
python make_graphs_yolo.py --targets targets.csv --out graphs
```

### Build Dataset Index
```bash
python scripts/01_build_index.py --raw_dir scripts/raw_images --targets targets.csv --out index.csv
```

### Split Dataset
```bash
python scripts/02_split_build_dataset.py --index index.csv --out data/plots
```

### Train Model
```bash
python scripts/03_train_yolov8_cls.py --model yolov8n-cls.pt --epochs 200 --imgsz 224 --batch 64 --device 0
```

### Predict
```bash
python scripts/04_predict_folder.py --weights models/best.pt --input_dir data/plots/test
```

### Evaluate Model
```bash
python scripts/evaluate_model.py --pred_csv evaluation_results/predictions_detail.csv
```

### Launch Web UI
```bash
START_WEB_UI.bat
```

## Project Flow
1. **Data Collection**: Download light curve data from MAST
2. **Preprocessing**: Clean and apply BLS (Box Least Squares) analysis
3. **Visualization**: Generate phase-folded light curve plots
4. **Dataset Creation**: Build stratified train/val/test splits
5. **Training**: Train YOLOv8 classification model
6. **Evaluation**: Generate metrics and performance reports
7. **Web Interface**: Deploy Flask-based analysis tool

## Train & Evaluate
The training pipeline uses YOLOv8's classification capabilities with custom preprocessing for astronomical data. Evaluation includes confusion matrix, ROC curves, and detailed error analysis.

## Web UI
The Flask application provides a web interface for uploading and analyzing light curve data in real-time.

## Folder Layout
- `data/plots/`: Organized dataset splits (train/val/test with positive/negative)
- `models/`: Trained model weights
- `graphs/`: Generated visualization outputs
- `scripts/`: Core processing and training scripts
- `app/`: Flask web application
- `evaluation_results/`: Model performance metrics and visualizations
