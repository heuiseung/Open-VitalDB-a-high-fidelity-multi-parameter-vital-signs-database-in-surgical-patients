# scripts / notebooks 루트 모듈 import 확인 방법

## 1. scripts/ 내 파이썬 파일 확인

**방법 A: 검증 스크립트 실행 (가장 간단)**

프로젝트 루트에서:

```bash
python scripts/check_imports.py
```

또는 `scripts` 폴더로 이동한 뒤:

```bash
cd scripts
python check_imports.py
```

- `[OK] import config`, `data_loader`, `model`, `utils` 가 모두 나오면 **정상**입니다.
- `[FAIL]` 이 나오면 해당 모듈 경로 설정을 다시 확인하세요.

**방법 B: 실제 스크립트로 확인**

프로젝트 루트에서:

```bash
python scripts/check_setup.py
```

- 설정 검증 메시지가 한글로 출력되면 `config` 등 루트 모듈이 정상적으로 불려진 것입니다.

`scripts` 폴더에서 직접 실행해도 됩니다:

```bash
cd scripts
python check_setup.py
```

---

## 2. notebooks/ 확인

1. Jupyter를 **프로젝트 루트**에서 실행한 뒤 노트북을 엽니다.

   ```bash
   cd C:\Users\sck32\hypo_vitaldb
   jupyter notebook
   ```

   브라우저에서 `notebooks/hypotension_pipeline.ipynb` 를 연 다음, **셀 2(경로 설정 및 import)** 부터 순서대로 실행(Run)합니다.

2. 또는 **notebooks 폴더**를 Jupyter에서 연 뒤 `hypotension_pipeline.ipynb` 를 실행해도 됩니다.  
   셀 2에서 프로젝트 루트를 찾아 `sys.path`에 넣고 `os.chdir(ROOT)` 하므로, 두 경우 모두 루트 모듈을 불러올 수 있습니다.

**정상일 때:** 셀 2 실행 후 `[진행상황] 프로젝트: ...`, `[진행상황] ...` 메시지가 나오고, `config`, `data_loader` import 오류가 없어야 합니다.

**오류가 날 때:** `ModuleNotFoundError: No module named 'config'` 등이 나오면, 셀 2의 `ROOT` 경로가 실제 프로젝트 루트(예: `hypo_vitaldb`)를 가리키는지 확인하세요.

---

## 요약

| 실행 위치              | 확인 명령 / 방법                          |
|------------------------|-------------------------------------------|
| 프로젝트 루트          | `python scripts/check_imports.py`         |
| scripts/ 폴더          | `python check_imports.py`                 |
| 노트북 (루트에서 실행) | `notebooks/hypotension_pipeline.ipynb` 셀 2 실행 |
| 노트북 (notebooks/에서) | 동일 노트북 셀 2 실행 (자동으로 루트 찾음) |
