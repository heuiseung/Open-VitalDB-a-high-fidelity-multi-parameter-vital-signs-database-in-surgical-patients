@echo off
chcp 65001 >nul
cd /d "%~dp0.."

where git >nul 2>&1 || (echo Git이 설치되어 있지 않거나 PATH에 없습니다. & pause & exit /b 1)

echo [진행] 변경 사항 스테이징...
git add -A

echo [진행] 커밋 중...
for /f "tokens=1-3 delims=/ " %%a in ('date /t') do set D=%%a-%%b-%%c
for /f "tokens=1-3 delims=: " %%a in ('time /t') do set T=%%a:%%b:%%c
git commit -m "자동 저장: %D% %T%" 2>nul || (echo 변경 사항이 없거나 이미 커밋됨. & goto push)

:push
echo [진행] GitHub에 푸시 중...
git push

if %errorlevel% equ 0 (echo [완료] GitHub에 저장되었습니다.) else (echo [안내] 원격이 없으면: git remote add origin 주소)
pause
