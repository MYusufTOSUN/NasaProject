@echo off
echo =========================================
echo   EXOPLANET DETECTOR - WEB UI
echo =========================================
echo.

cd /d %~dp0

echo [1/3] Activating virtual environment...
call .venv\Scripts\activate.bat

echo [2/3] Checking Flask installation...
python -m pip show flask >nul 2>&1
if errorlevel 1 (
    echo Flask not found, installing...
    python -m pip install flask -q
)

echo [3/3] Starting web server...
echo.
echo =========================================
echo  Server starting at: http://localhost:5000
echo  Press Ctrl+C to stop
echo =========================================
echo.

python app\exoplanet_detector.py

pause

