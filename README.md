# 수술 중 저혈압(Hypotension) 조기 예측 프로젝트

VitalDB 데이터로 MAP < 65 mmHg 저혈압을 **5분 후** 발생 여부를 예측합니다.  
CUDA 사용, 진행률 표시, **과금 방지**(최대 실행 시간/스텝 도달 시 자동 저장 후 중단).

**처음이면 [시작하기.md](시작하기.md) 를 열어 한 번에 진행하세요.**

## Cursor / VS Code / GitHub 연동

- **Cursor**와 **VS Code** 모두 이 폴더를 열고 **GitHub**와 연동되어 있으면, 한쪽에서 수정한 내용이 **소스 제어**에서 동일하게 보입니다.
- **GitHub에 자동 저장**: **push_to_github.bat** 더블클릭 또는 `Ctrl+Shift+P` → **Tasks: Run Task** → **Git: GitHub에 자동 저장 (커밋+푸시)** → 변경 사항 커밋 후 푸시.
- **커밋할 때마다 자동 푸시**: 터미널에서 `git config core.hooksPath .githooks` 한 번 실행 후, 커밋 시 자동으로 GitHub에 푸시됩니다. (자세한 내용: `.github/SYNC.md`)
- **풀**: 소스 제어 **⋯** → **Pull**.
- `hypotension_dataset.csv`, `checkpoints/` 등 큰/생성 파일은 `.gitignore`에 있어 자동 제외됩니다.

## 프로젝트 위치

- **코드/실행**: `C:\Users\sck32\hypo_vitaldb\`
- **데이터**: 기존 VitalDB 폴더  
  `C:\Users\sck32\Documents\Python_Scripts\Open VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients\`  
  (vital_files, clinical_data.csv 자동 참조)

## 실행 방법 (VSCode)

1. **VSCode에서 폴더 열기**: `파일` → `폴더 열기` → `C:\Users\sck32\hypo_vitaldb` 선택  
   또는 터미널에서: `code C:\Users\sck32\hypo_vitaldb`

2. **Python 인터프리터 선택**: `Ctrl+Shift+P` → "Python: Select Interpreter" → Python 3.12 또는 3.14 선택

3. **실행**
   - **방법 A** `Ctrl+Shift+B` (빌드/실행) → **"2. 전체 파이프라인 실행 (run_all)"** 선택  
     → 자동으로 패키지 설치 후 `run_all.py` 실행
   - **방법 B** `Ctrl+Shift+P` → "Tasks: Run Task" → **"2. 전체 파이프라인 실행 (run_all)"**
   - **방법 C** F5 (디버그) → **"전체 파이프라인 (run_all)"** 선택

## 실행 방법 (터미널)

1. 프로젝트 폴더로 이동: `cd C:\Users\sck32\hypo_vitaldb`
2. 패키지 설치: `pip install -r requirements.txt`
3. 전체 실행: `python run_all.py`

## 설정 (config.py)

- `MAX_RUNTIME_MINUTES = 30`: 데이터셋 구축 최대 30분 (과금 방지)
- `MAX_TRAIN_STEPS = 500`: 학습 최대 500 스텝 (과금 방지)
- `None`으로 두면 제한 없이 실행

## 생성 파일

- `hypotension_dataset.csv`: 구축된 학습 데이터
- `checkpoints/hypo_model.pt`: 저장된 모델
- `checkpoints/train_state.pt`: 학습 step 등 상태

## 한 번에 실행 (배치/스크립트)

- **run.bat** 더블클릭 또는 `cmd`에서 `run.bat` → 패키지 설치 후 파이프라인 실행
- **run.ps1** PowerShell에서 `.\run.ps1` → 동일

(실행 전 Python이 `C:\Users\sck32\AppData\Local\Programs\Python\Python314` 또는 `Python312`에 있거나, PATH에 `python`이 있어야 합니다.)

## 개별 실행

- 데이터셋만 구축: `python build_dataset.py`
- 학습만 실행: `python train_model.py`

## 설정 검증

- **python check_setup.py** 또는 **Tasks: Run Task** → **설정 검증 (check_setup)**  
  → 데이터 경로·clinical_data 건수·vital_files 샘플·기존 데이터셋 여부를 한글로 출력.

## 참고

- 첫 실행 시 `build_dataset.py`는 **100건**만 처리(빠른 테스트). 전체는 `build_dataset.py`에서 `MAX_CASES = None`으로 변경.
