# 수술 중 저혈압(Hypotension) 조기 예측

**GitHub**: [heuiseung/VitalDB-Hypotension-Prediction](https://github.com/heuiseung/VitalDB-Hypotension-Prediction)  
(서울대병원 VitalDB 데이터 기반)

이 프로젝트는 VitalDB 원시 신호로부터 **5분 후** 발생할 저혈압(MAP 기준)을 예측하는 파이프라인입니다.

요약
- 입력: VitalDB vital files (`vital_files/`) 및 `clinical_data.csv`
- 출력: `hypotension_dataset.csv` (특징 + `label`), 학습 체크포인트(`checkpoints/`)
- 주요 스크립트: `build_dataset.py`, `train_model.py`, `run_all.py`

빠른 시작
1. 레포를 클론하거나 이 폴더를 연다.
2. Python 가상환경을 만들고 활성화:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
3. 전체 파이프라인 실행 (패키지 설치 포함):
```powershell
python run_all.py
```

개별 실행
- 데이터셋만 구축: `python build_dataset.py`
- 모델 학습만: `python train_model.py`

설정
- 핵심 설정은 `config.py`에 있습니다. 주요 항목:
  - `MAP_THRESHOLD_MMHG` : 저혈압 임계값 (mmHg)
  - `HYPOTENSION_DURATION_SEC` : 연속 저혈압 판정 지속 시간 (초)
  - `MAX_RUNTIME_MINUTES`, `MAX_TRAIN_STEPS` : 실행/학습 시간 제한(과금 방지)

데이터 경로
- VitalDB 원본 데이터 폴더(예시):
  `C:\Users\sck32\Documents\Python_Scripts\Open VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients\`
  해당 경로 아래의 `vital_files/`와 `clinical_data.csv`를 사용합니다.

주의 및 권장
- `hypotension_dataset.csv`와 `checkpoints/`는 크기가 클 수 있으므로 Git에 직접 커밋하지 마세요. `.gitignore`에 기본으로 포함되어 있습니다.
- 처음 실행 시 빠른 확인을 위해 `build_dataset.py`는 기본적으로 작은 샘플만 처리할 수 있으니 `config.py`의 `MAX_CASES`를 확인하세요.

문서 및 추가 자료
- 파이프라인 실행 스크립트: `run_all.py`
- 라벨 생성, 데이터 처리 논리: `data_loader.py`
- 개선 로그 및 보고서: `IMPROVEMENT_REPORT.md`, `IMPROVEMENT_SUMMARY.md`

지원
- 질문이나 병합(merge) 관련 요청은 GitHub 이슈로 남겨 주세요.

---
최신: 파이프라인 완료 (다중 에폭+검증, 특성 확장, CUDA 학습). 최종 성능은 `RESULTS.md` 참조 (AUC-ROC 0.925).
