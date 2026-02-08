# 노트북 자동 실행
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location (Split-Path $PSScriptRoot -Parent)

$py = $null
@(
    "C:\Users\sck32\AppData\Local\Programs\Python\Python314\python.exe",
    "C:\Users\sck32\AppData\Local\Programs\Python\Python312\python.exe"
) | ForEach-Object { if (Test-Path $_) { $py = $_; return } }
if (-not $py) { $py = (Get-Command python -ErrorAction SilentlyContinue).Source }

if (-not $py) {
    Write-Host "Python 없음. Cursor/VSCode에서 노트북 열고 Run All 실행하세요."
    exit 1
}

Write-Host "[1/2] 패키지 설치..."
& $py -m pip install -q -r requirements.txt
Write-Host "[2/2] 노트북 자동 실행..."
& $py -m jupyter nbconvert --to notebook --execute --inplace notebooks/hypotension_pipeline.ipynb
Write-Host "완료. notebooks/hypotension_pipeline.ipynb"
exit $LASTEXITCODE
