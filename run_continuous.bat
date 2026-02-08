@echo off
chcp 65001 >nul
cd /d "%~dp0"
for %%d in (Python314 Python312) do (
  if exist "C:\Users\sck32\AppData\Local\Programs\Python\%%d\python.exe" (
    "C:\Users\sck32\AppData\Local\Programs\Python\%%d\python.exe" run_continuous.py
    exit /b
  )
)
python run_continuous.py
exit /b %errorlevel%
