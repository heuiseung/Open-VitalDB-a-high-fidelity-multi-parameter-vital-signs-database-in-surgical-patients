# 프로젝트 디렉토리 구조

```
C:\Users\sck32\hypo_vitaldb\
│
├── 📄 README.md                          ← 프로젝트 개요
├── 📄 QUICKSTART.md                      ← 빠른 시작 가이드
├── 📄 PROGRESS_REPORT.md                 ← 작업 완료 보고서
├── 📄 FINAL_SAVE_STATUS.md               ← 최종 저장 현황
├── 📄 DIRECTORY_STRUCTURE.md             ← 이 파일
├── 📄 requirements.txt                   ← Python 패키지 의존성
│
├── 🔧 설정 & 데이터 로더
│   ├── config.py                         ← 중앙 설정 (경로, 파라미터)
│   └── data_loader.py                    ← VitalDB 데이터 로더
│
├── 📊 데이터 & 모델
│   ├── hypotension_dataset.csv           ← 생성된 데이터셋 (19,432행)
│   ├── build_dataset.py                  ← 데이터셋 구축 스크립트
│   ├── train_model.py                    ← 기본 모델 학습
│   ├── train_model_advanced.py           ← 고급 모델 (Batch Norm+Early Stopping)
│   ├── run_all.py                        ← 전체 파이프라인
│   └── analyze_results.py                ← 결과 분석 & 시각화
│
├── 📁 checkpoints/                       ← 모델 체크포인트
│   ├── hypo_model.pt                     ← 학습된 모델 가중치
│   ├── train_state.pt                    ← 훈련 상태
│   ├── training_loss.png                 ← 손실 곡선 그래프
│   ├── analysis_feature_distribution.png ← 특성 분포 히스토그램
│   └── analysis_label_distribution.png   ← 라벨 분포 차트
│
├── 📓 노트북
│   └── hypotension_pipeline.ipynb        ← 대화형 Jupyter 노트북
│
└── 📁 .github/                           ← GitHub 설정
    └── copilot-instructions.md           ← AI 에이전트 가이드
```

---

## 📂 **주요 디렉토리 설명**

### 🏠 루트 디렉토리
**용도:** 프로젝트 진입점 및 설정 관리

| 파일 | 설명 | 크기 |
|------|------|------|
| config.py | 모든 설정의 중추 (경로, 파라미터, 임계값) | 1 KB |
| requirements.txt | pip 패키지 의존성 | <1 KB |

### 🗂️ checkpoints/
**용도:** 모델 및 결과물 저장

| 파일 | 설명 | 크기 |
|------|------|------|
| hypo_model.pt | PyTorch 신경망 가중치 | 38 KB |
| training_loss.png | 훈련 손실 곡선 | 32 KB |
| analysis_*.png | 데이터 분석 그래프 | 55+28 KB |

### 📄 문서
**용도:** 프로젝트 이해 및 가이드

```
README.md              → 처음 읽기
QUICKSTART.md          → 빠른 실행
PROGRESS_REPORT.md     → 상세 내용
FINAL_SAVE_STATUS.md   → 저장 현황
```

---

## 🚀 **실행 흐름**

```
시작
  ↓
config.py (설정 로드)
  ├─ VITAL_DIR (데이터 경로)
  ├─ MAX_CASES (처리량)
  └─ DEVICE (GPU/CPU)
  ↓
build_dataset.py (데이터 추출)
  ├─ load_vital_case() [data_loader.py]
  ├─ extract_features()
  ├─ build_labels_for_case() [data_loader.py]
  └─ → hypotension_dataset.csv
  ↓
train_model.py (모델 학습)
  ├─ HypoNet (신경망 정의)
  ├─ 데이터 로드 & 정규화
  ├─ 80/20 분할
  ├─ 훈련 루프
  └─ → hypo_model.pt
  ↓
analyze_results.py (결과 분석)
  ├─ 모델 로드
  ├─ 예측 수행
  ├─ 성능 평가
  └─ → PNG 그래프
  ↓
완료
```

---

## 💾 **파일 의존성 그래프**

```
hypotension_dataset.csv
  ↑
  └─ build_dataset.py
     ├─ data_loader.py
     └─ config.py

hypo_model.pt
  ↑
  └─ train_model.py
     ├─ hypotension_dataset.csv
     └─ config.py

분석 결과 (PNG)
  ↑
  └─ analyze_results.py
     ├─ hypotension_dataset.csv
     ├─ hypo_model.pt
     ├─ train_model.py
     └─ config.py

run_all.py
  ├─ build_dataset.py
  └─ train_model.py
```

---

## 🔄 **데이터 흐름**

```
VitalDB (vital_files)
  ↓ (load_vital_case)
  └─ Time-series vital signals (MAP, HR, ...)
     ↓ (extract_features)
     └─ 5-min lookback window features
        ↓ (build_labels_for_case)
        └─ 5-min prediction horizon labels
           ↓
           └─ hypotension_dataset.csv
              ├─ Features: MAP_mean, MAP_std, MAP_min, HR_mean, HR_std, HR_min
              ├─ Label: 0 (No Hypotension) or 1 (Hypotension)
              └─ CaseID: 1-6388
```

---

## 🎯 **중요 상수 (config.py)**

```python
# 비용 제어
MAX_RUNTIME_MINUTES = 30       # 데이터셋 구축 최대 30분
MAX_TRAIN_STEPS = 500          # 모델 학습 최대 500스텝

# 의료 기준
MAP_THRESHOLD_MMHG = 65        # 저혈압 기준값
HYPOTENSION_DURATION_SEC = 60  # 지속 시간

# 예측 설정
PREDICTION_HORIZON_MIN = 5     # 5분 후 예측
LOOKBACK_MIN = 5               # 5분 과거 데이터 사용

# 데이터셋 설정
TEST_SIZE = 0.2                # 80% train, 20% test
RANDOM_STATE = 42              # 재현성

# 모델 설정
DEVICE = "cuda"                # GPU 사용
```

---

## 📦 **패키지 의존성 (requirements.txt)**

```
vitaldb>=1.6.0          # VitalDB 데이터 로드
pandas>=1.5.0           # 데이터프레임 처리
numpy>=1.23.0           # 수치 계산
scikit-learn>=1.2.0     # 머신러닝 유틸
matplotlib>=3.6.0       # 시각화
seaborn>=0.12.0         # 통계 시각화
tqdm>=4.65.0            # 진행률 표시
jupyter>=1.0.0          # 노트북
nbconvert>=7.0.0        # 노트북 변환
torch>=2.0.0            # PyTorch (CUDA 12.1)
```

---

## 🔑 **핵심 함수 위치**

| 함수 | 위치 | 역할 |
|------|------|------|
| `load_vital_case()` | data_loader.py | VitalDB vital 파일 로드 |
| `extract_features()` | build_dataset.py | 5분 window 특성 추출 |
| `build_labels_for_case()` | data_loader.py | 저혈압 라벨 생성 |
| `HypoNet` | train_model.py | 기본 신경망 모델 |
| `HypoNetAdvanced` | train_model_advanced.py | 고급 신경망 (Batch Norm) |
| `main()` | 각 스크립트 | 각 단계 메인 함수 |

---

## 📊 **성능 지표 저장 위치**

```
Training Performance:
  └─ training_loss.png              (checkpoints/)
     └─ 손실 값 변화 시각화

Data Analysis:
  ├─ analysis_feature_distribution.png    (checkpoints/)
  │  └─ 6개 특성의 분포 히스토그램
  └─ analysis_label_distribution.png      (checkpoints/)
     └─ 라벨 0/1 분포

Model Evaluation (콘솔 출력):
  ├─ Accuracy
  ├─ Confusion Matrix
  └─ Classification Report
```

---

## 🔗 **상호 참조**

```
설정 변경:
  config.py → 수정 → 모든 스크립트에 자동 반영

새로운 특성 추가:
  data_loader.py (line 25)
  → extract_features() 수정
  → build_dataset.py 자동 반영

모델 개선:
  train_model_advanced.py
  → HypoNetAdvanced 클래스 수정
  → run_all.py 실행

결과 분석:
  analyze_results.py
  → hypo_model.pt + hypotension_dataset.csv 로드
  → 시각화 생성
```

---

## 🎓 **학습 경로**

### 1️⃣ 초보자 (기본 이해)
```
1. README.md 읽기
2. config.py 검토
3. QUICKSTART.md 실행
4. 결과 확인
```

### 2️⃣ 중급자 (커스터마이징)
```
1. data_loader.py 분석
2. build_dataset.py 이해
3. 특성 추가/수정
4. 라벨 기준 변경
```

### 3️⃣ 고급자 (확장)
```
1. train_model_advanced.py 수정
2. 새로운 모델 아키텍처 구현
3. 하이퍼파라미터 튜닝
4. 배포 준비
```

---

**이 구조는 재현 가능성, 유지보수성, 확장성을 모두 고려하여 설계되었습니다.** ✨
