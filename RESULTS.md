# 프로젝트 결과 요약

**최종 업데이트**: 파이프라인 완료 (다중 에폭 + 검증 best 모델, CUDA/GPU 학습)

## 모델 정보
- 체크포인트: `checkpoints/hypo_model.pt`
- 저장 경로: C:\Users\sck32\hypo_vitaldb\checkpoints\hypo_model.pt
- 파일 크기: 38 KB

## 학습 데이터
- 테스트 샘플: 19,318건 (케이스 단위 분할, 검증 15% 분리)
- 참고: `hypotension_dataset.csv`는 프로젝트 루트에 있습니다.

## 평가 성능 (최종 실행 기준 — 다중 에폭 + 검증 best)

- **Accuracy**: 0.85
- **AUC-ROC**: **0.925** (검증 best AUC: 0.960)

### 클래스 별
- **저혈압 없음** (Negative)
  - Precision: 0.91, Recall: 0.86, F1-score: 0.88
  - Support: 12,817

- **저혈압** (Positive)
  - Precision: 0.75, Recall: 0.84, F1-score: 0.79
  - Support: 6,501

### 혼동 행렬 (테스트 19,318건)
```
              예측: 음성   예측: 양성
실제 음성        10,966      1,851
실제 양성         1,043      5,458
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

