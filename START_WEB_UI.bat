@echo off
REM Ekzoplanet Tespit Sistemi - Web UI Başlatıcı

echo ==========================================
echo   Ekzoplanet Tespit Sistemi
echo   Web UI Baslatiyor...
echo ==========================================
echo.

REM Sanal ortam var mı kontrol et
if not exist "venv\" (
    echo [1/3] Sanal ortam olusturuluyor...
    python -m venv venv
    if errorlevel 1 (
        echo HATA: Sanal ortam olusturulamadi!
        echo Python'un PATH'de oldugunu kontrol edin.
        pause
        exit /b 1
    )
    echo Sanal ortam basariyla olusturuldu.
    echo.
) else (
    echo [1/3] Sanal ortam mevcut.
    echo.
)

REM Sanal ortamı aktif et
echo [2/3] Sanal ortam aktif ediliyor...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo HATA: Sanal ortam aktif edilemedi!
    pause
    exit /b 1
)
echo.

REM Bağımlılıkları kontrol et ve yükle
echo [3/3] Bagimliliklar kontrol ediliyor...
pip show flask >nul 2>&1
if errorlevel 1 (
    echo Flask bulunamadi, bagimliliklar yukleniyor...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo HATA: Bagimliliklar yuklenemedi!
        pause
        exit /b 1
    )
    echo Bagimliliklar basariyla yuklendi.
) else (
    echo Bagimliliklar mevcut.
)
echo.

echo ==========================================
echo   Flask Server Baslatiyor...
echo   Tarayicinizda http://localhost:5000
echo   adresini acin.
echo.
echo   Kapatmak icin Ctrl+C basin.
echo ==========================================
echo.

REM Flask uygulamasını başlat
python app\exoplanet_detector.py

REM Hata durumunda bekleme
if errorlevel 1 (
    echo.
    echo HATA: Flask uygulamasi baslamadi!
    echo app\exoplanet_detector.py dosyasini kontrol edin.
    pause
)

