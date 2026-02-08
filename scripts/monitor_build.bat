@echo off
REM 빌드 진행 상황 자동 모니터링 (매 5분마다)
setlocal enabledelayedexpansion
cd /d "%~dp0.."

:monitor_loop
cls
echo.
echo ====================================================
echo.  빌드 진행 상황 모니터링
echo ====================================================
echo.
echo [%date% %time%] 체크...

if exist hypotension_dataset.csv (
    for %%A in (hypotension_dataset.csv) do (
        set size_mb=%%~zA
        set size_mb=!size_mb:~0,-6!
        echo ✅ CSV 발견: %%~zA bytes
    )
    REM 라인 수 계산 (Windows 느림)
    echo 행 수 계산 중...
) else (
    echo ⏳ CSV 아직 생성 안됨...
)

echo.
echo [다음 체크: 5분 후]
echo.

timeout /t 300 /nobreak

goto monitor_loop
