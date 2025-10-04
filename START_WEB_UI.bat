@echo off
setlocal
if not exist .venv (
  python -m venv .venv
)
call .venv\Scripts\activate
pip show flask >nul 2>&1 || pip install -r requirements.txt
set FLASK_APP=app\exoplanet_detector.py
set FLASK_ENV=production
start "" http://localhost:5000
python -m flask run --port 5000
