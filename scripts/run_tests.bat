@echo off
REM Proje kokunden testleri calistirir. Once: pip install -r requirements.txt
cd /d "%~dp0\.."
python -m pytest tests/ -v --tb=short 2>&1
if errorlevel 1 exit /b 1
exit /b 0
