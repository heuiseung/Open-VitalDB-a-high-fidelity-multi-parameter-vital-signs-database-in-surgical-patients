# GitHub에 자동 저장 (스테이징 + 커밋 + 푸시)
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Git이 설치되어 있지 않거나 PATH에 없습니다."
    exit 1
}

Write-Host "[진행] 변경 사항 스테이징..."
git add -A

$msg = "자동 저장: $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
Write-Host "[진행] 커밋 중..."
git commit -m $msg 2>$null
if ($LASTEXITCODE -ne 0) { Write-Host "변경 사항이 없거나 이미 커밋됨." }

Write-Host "[진행] GitHub에 푸시 중..."
git push
if ($LASTEXITCODE -eq 0) { Write-Host "[완료] GitHub에 저장되었습니다." } else { Write-Host "[안내] 원격이 없으면: git remote add origin <저장소주소>" }
exit $LASTEXITCODE
