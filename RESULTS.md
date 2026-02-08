# 프로젝트 결과 요약

**최종 업데이트**: GPU/CUDA 전체 재학습 완료 (특성 확장 데이터 500케이스, 데이터 GPU 상주)

## 모델 정보
- 체크포인트: `checkpoints/hypo_model.pt`
- 저장 경로: C:\Users\sck32\hypo_vitaldb\checkpoints\hypo_model.pt
- 학습 장치: **NVIDIA GeForce RTX 4070** (약 12 GB), CUDA

## 학습 데이터 (이번 실행)
- **특성 확장 데이터**: 500케이스 → train 62,150건 / val 분리 / test 19,318건
- 케이스 단위 분할: train 340, val 60, test 100케이스
- 학습/검증/테스트 텐서 **전부 GPU 메모리 상주** (non_blocking, BATCH_SIZE=512, TF32)

## 평가 성능 (GPU 전체 재학습 기준)

- **Accuracy**: 0.84
- **AUC-ROC (테스트)**: **0.922**
- **검증 best AUC**: 0.959 (에폭 10에서 저장)

### 클래스 별
- **저혈압 없음** (Negative)
  - Precision: 0.92, Recall: 0.84, F1-score: 0.88
  - Support: 12,817

- **저혈압** (Positive)
  - Precision: 0.73, Recall: 0.85, F1-score: 0.79
  - Support: 6,501

### 혼동 행렬 (테스트 19,318건)
```
              예측: 음성   예측: 양성
실제 음성        10,802      2,015
실제 양성           996      5,505
```

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

