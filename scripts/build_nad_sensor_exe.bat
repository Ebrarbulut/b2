@echo off
REM Canli NIDS script'ini tek .exe haline getirir.
REM Gereksinim: pip install pyinstaller
REM Cikti: dist\nad_sensor.exe

cd /d "%~dp0\.."
if not exist "venv\Scripts\activate.bat" (
  echo [HATA] venv bulunamadi. Once: python -m venv venv
  exit /b 1
)
call venv\Scripts\activate.bat
pip install pyinstaller -q
pyinstaller --onefile --noconsole --name nad_sensor --hidden-import=scapy realtime_nids_scapy.py
echo.
echo Tamamlandi. Calistirilabilir: dist\nad_sensor.exe
echo Cift tiklayinca terminal acilmaz; log: dist\nad_sensor.log
echo (Yonetici olarak calistirmaniz gerekebilir.)
