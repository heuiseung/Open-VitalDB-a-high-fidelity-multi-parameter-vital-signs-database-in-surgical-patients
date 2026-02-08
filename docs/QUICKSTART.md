# 빠른 시작 (자동 진행)

## 주피터 노트북 (권장)

- **hypotension_pipeline.ipynb** 열기 → 상단 메뉴 **Run** → **Run All** (또는 `Shift+Enter`로 셀 단위 실행)
- **자동 실행**: `run_notebook.bat` 더블클릭 또는 터미널에서 `.\run_notebook.ps1`  
  → 패키지 설치 후 노트북 전체 자동 실행, 결과가 같은 노트북에 저장됨
- VSCode/Cursor: `Ctrl+Shift+B` → **"3. 주피터 노트북 자동 실행"** 선택

---

## 1. VSCode에서 스크립트 한 번에 실행

1. **폴더 열기**  
   `파일` → `폴더 열기` → `C:\Users\sck32\hypo_vitaldb` 선택

2. **Python 선택**  
   `Ctrl+Shift+P` → **Python: Select Interpreter** →  
   `Python 3.14` 또는 `Python 3.12` (또는 `C:\Users\sck32\AppData\Local\Programs\Python\Python314\python.exe`) 선택

3. **실행**  
   `Ctrl+Shift+B` → **"2. 전체 파이프라인 실행 (run_all)"** 선택  
   → 패키지 설치 후 데이터셋 구축(100건) → 학습이 자동으로 진행됩니다.

## 2. 탐색기에서 한 번에 실행

- `C:\Users\sck32\hypo_vitaldb\run.bat` **더블클릭**  
  → 패키지 설치 후 파이프라인 자동 실행

## 3. 터미널에서 한 번에 실행

```cmd
cd C:\Users\sck32\hypo_vitaldb
run.bat
```

또는 (PowerShell):

```powershell
cd C:\Users\sck32\hypo_vitaldb
.\run.ps1
```

---

## 실행 전 설정 검증 (선택)

- **python check_setup.py** 또는 **Tasks: Run Task** → **설정 검증 (check_setup)**  
  → 데이터 경로·케이스 수·vital 파일 샘플·기존 데이터셋 여부를 한글로 출력. 오류 시 메시지로 원인 확인 가능.

---

## 다음 단계

- **전체 케이스**: `build_dataset.py`에서 `MAX_CASES = None`으로 변경 후 다시 실행.
- **모델 재학습**: `hypotension_dataset.csv` 삭제 후 `run_all.py` 실행 시 데이터셋부터 다시 구축.
- **GitHub 저장**: `push_to_github.bat` 또는 **Tasks** → **Git: GitHub에 자동 저장**.

---

- **데이터**: VitalDB 폴더(`clinical_data.csv`, `vital_files`)는 `config.py`에서 자동 참조됩니다.  
- **첫 실행**: 100건만 처리(약 1~2분).
