# Cursor · VS Code · GitHub 동기화

이 저장소는 **Cursor**와 **VS Code**에서 같은 GitHub 원격과 연동해 사용할 수 있습니다.

## GitHub에 자동 저장 (권장)

### 방법 1: 스크립트 한 번에 실행
- **push_to_github.bat** 더블클릭 또는 터미널에서 `push_to_github.bat`  
  → 변경 사항 스테이징 → 커밋(메시지: 자동 저장) → **GitHub에 푸시**
- PowerShell: `.\push_to_github.ps1`

### 방법 2: Cursor / VS Code 태스크
- `Ctrl+Shift+P` → **Tasks: Run Task** → **Git: GitHub에 자동 저장 (커밋+푸시)**  
  → 위와 동일하게 커밋 후 푸시

### 방법 3: 커밋할 때마다 자동 푸시
한 번만 설정하면, 이후 **커밋할 때마다 자동으로 GitHub에 푸시**됩니다.

```bash
git config core.hooksPath .githooks
```

(이미 `.githooks/post-commit`에 `git push`가 들어 있어, 커밋 직후 자동 실행됩니다.)

## 수동으로 커밋 후 푸시

1. **Cursor** 또는 **VS Code**에서 `C:\Users\sck32\hypo_vitaldb` 폴더 열기
2. **소스 제어** (`Ctrl+Shift+G`) 열기
3. 변경된 파일 확인 → 스테이징(+) → 커밋 메시지 입력 → ✓ **커밋**
4. **⋯** 메뉴 → **Push** (GitHub에 업로드)

## 다른 PC/에디터에서 받기

1. 같은 폴더를 **다른 에디터**에서 열기
2. **⋯** → **Pull** (GitHub에서 최신 받기)

## 커밋에서 제외되는 것 (`.gitignore`)

- `hypotension_dataset.csv`, `checkpoints/` (생성 파일·용량 큼)
- `__pycache__/`, `.ipynb_checkpoints/`
- `.venv/`, `venv/`

코드와 설정만 GitHub에 올라가고, 데이터·체크포인트는 로컬에만 유지됩니다.
