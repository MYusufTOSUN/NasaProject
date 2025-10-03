# 🌟 Exoplanet Detector - Web UI

Premium, AI-powered web application for detecting exoplanets from stellar light curves.

## ✨ Features

- 🎯 **Real-time Analysis**: Instant light curve download from MAST archive
- 🤖 **AI Detection**: YOLOv8-based classification model
- 📊 **Interactive Visualization**: Beautiful, responsive light curve plots
- 🚀 **Multi-Mission Support**: Kepler, K2, TESS data
- 💎 **Premium UI**: Modern, space-themed interface
- ⚡ **Fast Predictions**: Sub-second inference time

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd app
pip install -r requirements.txt
```

### 2. Start Server

```bash
python exoplanet_detector.py
```

### 3. Open Browser

Navigate to: `http://localhost:5000`

## 📖 Usage

1. **Enter Target Name**: 
   - Known planets: `Kepler-10`, `TOI 700`, `HAT-P-7`
   - Star IDs: `KIC 8462852`, `TIC 307210830`

2. **Select Mission** (optional):
   - Auto: Search all missions
   - Kepler/K2/TESS: Specific mission

3. **Analyze**: Click "ANALYZE NOW" button

4. **Results**:
   - Interactive light curve plot
   - AI prediction with confidence
   - Detailed probability breakdown

## 🎨 Example Targets

### Confirmed Exoplanets
- `Kepler-10` - First rocky exoplanet (Kepler)
- `Kepler-90` - 8-planet system
- `TOI 700` - TESS Earth-sized planet
- `HAT-P-7` - Hot Jupiter
- `K2-18` - Habitable zone planet

### Interesting Non-Planets
- `KIC 8462852` - Tabby's Star (mysterious dips)
- `KIC 10666592` - Eclipsing binary
- Random KIC/TIC IDs for testing

## 🏗️ Architecture

```
app/
├── exoplanet_detector.py  # Flask backend
├── templates/
│   └── index.html         # Premium frontend
├── static/
│   └── temp/              # Temporary image storage
└── requirements.txt       # Dependencies
```

## 🔧 API Endpoints

### POST `/analyze`
Analyze a target star.

**Request:**
```json
{
  "target_name": "Kepler-10",
  "mission": "auto"
}
```

**Response:**
```json
{
  "success": true,
  "target": "Kepler-10",
  "mission": "Kepler",
  "data_points": 1234,
  "time_span": "120.5 days",
  "plot": "base64_image_data",
  "prediction": {
    "prediction": "Exoplanet Candidate",
    "confidence": 95.67,
    "positive_probability": 95.67,
    "negative_probability": 4.33
  }
}
```

### GET `/health`
Health check endpoint.

## 🎯 Model Performance

- **Accuracy**: ~93% on validation set
- **Precision**: 95%+ on confirmed planets
- **Recall**: 100% (no missed planets in test)
- **Inference**: <0.5s per prediction

## 🔬 Technical Details

### Data Pipeline
1. **Download**: Lightkurve queries MAST archive
2. **Preprocessing**: Time/flux normalization
3. **Visualization**: Matplotlib custom plots
4. **Classification**: YOLOv8n-cls model
5. **Results**: JSON response with base64 plots

### Model
- **Architecture**: YOLOv8 Nano Classification
- **Training**: 100 epochs, 552 train samples
- **Dataset**: Balanced positive/negative (sentetik negatifler)
- **Input**: 224x224 RGB light curve images

## 🛠️ Troubleshooting

### Model not found
```bash
# Copy trained model to app directory
cp models/best.pt ../models/
```

### Lightkurve errors
```bash
# Ensure proper installation
pip install lightkurve --upgrade
```

### Port already in use
```bash
# Change port in exoplanet_detector.py
app.run(port=5001)
```

## 📊 Performance Tips

- First query may be slow (MAST download)
- Subsequent queries cached locally
- Use specific mission for faster results
- Model runs on CPU/GPU automatically

## 🌐 Production Deployment

### Docker (Optional)
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "exoplanet_detector.py"]
```

### Nginx Reverse Proxy
```nginx
location / {
    proxy_pass http://localhost:5000;
    proxy_set_header Host $host;
}
```

## 📝 License

MIT License - Feel free to use for research and education.

## 🙏 Credits

- **Data**: NASA Exoplanet Archive, MAST
- **Model**: Ultralytics YOLOv8
- **Framework**: Flask, Lightkurve
- **UI**: Custom space-themed design

## 📧 Support

For issues or questions, please open a GitHub issue.

---

**Made with 💫 for exoplanet hunters**

