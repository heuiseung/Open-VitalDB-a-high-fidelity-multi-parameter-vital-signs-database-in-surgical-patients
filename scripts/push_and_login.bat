@echo off
chcp 65001 >nul
cd /d "%~dp0.."

echo.
echo [GitHub 푸시] 최초 실행 시 로그인 창 또는 브라우저가 열립니다.
echo             GitHub 계정으로 로그인하면 이후에는 자동으로 인증됩니다.
echo.

git push origin main

if %errorlevel% equ 0 (
    echo.
    echo [완료] 푸시 성공.
) else (
    echo.
    echo [안내] 로그인 창이 안 뜨면:
    echo   1. GitHub - Settings - Developer settings - Personal access tokens
    echo   2. 토큰 생성 후, 비밀번호 묻는 곳에 토큰 붙여넣기
)

echo.
pause
