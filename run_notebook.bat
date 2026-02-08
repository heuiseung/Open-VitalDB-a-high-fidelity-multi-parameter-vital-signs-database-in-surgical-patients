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

echo [1/2] 패키지 설치...
"%PY%" -m pip install -q -r requirements.txt

echo [2/2] 노트북 자동 실행 (hypotension_pipeline.ipynb)...
"%PY%" -m jupyter nbconvert --to notebook --execute --inplace hypotension_pipeline.ipynb

echo.
echo 완료. 출력: hypotension_pipeline.ipynb
pause
exit /b 0

:nopy
echo Python 없음. VSCode/Cursor에서 노트북을 열고 Run All 실행하세요.
pause
exit /b 1
