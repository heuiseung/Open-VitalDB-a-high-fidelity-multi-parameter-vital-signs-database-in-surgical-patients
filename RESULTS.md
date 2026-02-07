# 프로젝트 결과 요약

## 모델 정보
- 체크포인트: `checkpoints/hypo_model.pt`
- 저장 경로: C:\Users\sck32\hypo_vitaldb\checkpoints\hypo_model.pt
- 파일 크기: 38 KB

## 학습 데이터
- 전체 샘플 수: 97,802 (train 72,641 / test 18,161 — 모델 출력 기준)
  (참고: `hypotension_dataset.csv`는 프로젝트 루트에 있습니다.)

## 평가 성능 (최근 GPU 학습 기준)
- **Accuracy**: 0.79
- **AUC-ROC**: 약 0.849 ~ 0.85

### 클래스 별
- **저혈압 없음** (Negative)
  - Precision: 0.80 ~ 0.82
  - Recall: 0.90 ~ 0.94
  - F1-score: 0.86
  - Support: 12,928

- **저혈압** (Positive)
  - Precision: 0.67 ~ 0.73
  - Recall: 0.41 ~ 0.51
  - F1-score: 0.52 ~ 0.58
  - Support: 5,233

### 혼동 행렬 (예시)
```
              예측: 음성   예측: 양성
실제 음성       약 11,800~12,100    약 800~1,100
실제 양성       약 2,500~3,100      약 2,100~2,500
```
(매 실행마다 미세하게 달라질 수 있음)

## 재현 및 사용법
- 모델 로드 예시 (PyTorch):
```python
import torch
from pathlib import Path

ckpt = torch.load(Path('checkpoints') / 'hypo_model.pt', map_location='cpu')
# 모델 클래스 정의 필요: HypoNet
model = HypoNet(in_dim)
model.load_state_dict(ckpt['model_state'])
model.eval()
```

## 변경사항 커밋
- 주요 커밋:
  - `93ba40e` perf(train): load dataset tensors onto GPU and enable cuDNN benchmark
  - `a97bd4b` feat: add automated completion workflow
  - `ced5078` docs: add monitoring scripts and status tracking
  - `6a6731b` refactor(data): improve label logic - apply 3-condition OR

## 메모
- 체크포인트는 원격 저장소에 포함했습니다 (작은 파일, GitHub 업로드 가능).
- 더 큰 모델/데이터를 업로드할 경우 `git lfs` 사용을 권장합니다.

