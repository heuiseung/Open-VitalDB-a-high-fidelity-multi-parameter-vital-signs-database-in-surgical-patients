# UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location $PSScriptRoot

$py = $null
$candidates = @(
    "C:\Users\sck32\AppData\Local\Programs\Python\Python314\python.exe",
    "C:\Users\sck32\AppData\Local\Programs\Python\Python312\python.exe"
)
foreach ($p in $candidates) {
    if (Test-Path $p) { $py = $p; break }
}
if (-not $py) {
    try { $py = (Get-Command python -ErrorAction Stop).Source } catch {}
}
if (-not $py) {
    Write-Host "Python을 찾을 수 없습니다. VSCode에서 Ctrl+Shift+B 로 실행하세요."
    exit 1
}

Write-Host "[1/2] 패키지 설치 중..."
& $py -m pip install -r requirements.txt -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "pip 설치 실패."
    exit 1
}
Write-Host ""
Write-Host "[2/2] 파이프라인 실행 중..."
& $py run_all.py
exit $LASTEXITCODE
