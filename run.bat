@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "PY="
for %%d in (Python314 Python312) do (
  if exist "C:\Users\sck32\AppData\Local\Programs\Python\%%d\python.exe" set "PY=C:\Users\sck32\AppData\Local\Programs\Python\%%d\python.exe"
  if defined PY goto :found
)
where python >nul 2>&1 && set "PY=python" || goto :nopy
:found

echo [1/2] 패키지 설치 중...
"%PY%" -m pip install -r requirements.txt -q
if errorlevel 1 (
  echo pip 설치 실패. python 경로 확인 후 다시 실행하세요.
  pause
  exit /b 1
)

echo.
echo [2/2] 파이프라인 실행 중...
"%PY%" run_all.py

echo.
pause

:nopy
echo Python을 찾을 수 없습니다.
echo 1) VSCode: Ctrl+Shift+P -^> "Python: Select Interpreter" 선택 후 Ctrl+Shift+B
echo 2) 또는 set PY=실제python경로 후 이 배치 파일 다시 실행
pause
