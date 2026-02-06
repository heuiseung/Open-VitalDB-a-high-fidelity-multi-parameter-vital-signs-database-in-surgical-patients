"""PyTorch CUDA 학습 - 진행률 표시, 최대 스텝 도달 시 저장 후 중단 (과금 방지)"""
import sys
import io
if getattr(sys.stdout, "buffer", None) and (sys.stdout.encoding or "").lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from tqdm import tqdm

from config import (
    TEST_SIZE,
    RANDOM_STATE,
    DEVICE,
    DATASET_PATH,
    CHECKPOINT_DIR,
    MODEL_PATH,
    TRAIN_STATE_PATH,
    MAX_TRAIN_STEPS,
)


class HypoNet(nn.Module):
    """저혈압 이진 분류용 소형 MLP."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def save_checkpoint(model, optimizer, step, reason: str = ""):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
        },
        MODEL_PATH,
    )
    torch.save({"step": step}, TRAIN_STATE_PATH)
    print(f"\n[저장 완료] 모델 -> {MODEL_PATH}")
    if reason:
        print(f"[중단] {reason}")
    raise SystemExit(0)


def main() -> None:
    if not DATASET_PATH.exists():
        print("[안내] 먼저 데이터셋 구축을 실행해 주세요. (run_all.py 또는 build_dataset.py)")
        return
    df = pd.read_csv(DATASET_PATH)
    target = "label"
    feature_cols = [c for c in df.columns if c not in ("caseid", target)]
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    y = df[target].values.astype(np.int64)
    # 한쪽 클래스만 있으면 stratify 불가
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        print("[안내] 라벨이 한쪽만 있어 stratify 생략")
    # CUDA 설정 — GPU 활용, 데이터를 GPU에 올려 학습
    use_cuda = torch.cuda.is_available() and DEVICE.startswith("cuda")
    device = torch.device(DEVICE if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[CUDA] GPU 사용: {gpu_name} (약 {gpu_mem:.1f} GB)")
        print("[CUDA] 학습 데이터를 GPU 메모리에 올려 진행합니다.")
    print(f"[진행상황] 학습 장치: {device} | train {len(X_train)}건, test {len(X_test)}건")

    # 데이터를 GPU 메모리에 올려서 학습 (GPU 사용 시)
    X_train_t = torch.from_numpy(X_train).to(device=device, dtype=torch.float32)
    y_train_t = torch.from_numpy(y_train).to(device=device, dtype=torch.float32)
    X_test_t = torch.from_numpy(X_test).to(device=device, dtype=torch.float32)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    model = HypoNet(len(feature_cols)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    step = 0
    try:
        pbar = tqdm(train_loader, desc="[2/2] 모델 학습 중", unit="배치")
        for batch_x, batch_y in pbar:
            if MAX_TRAIN_STEPS is not None and step >= MAX_TRAIN_STEPS:
                save_checkpoint(
                    model, opt, step,
                    f"최대 학습 스텝 {MAX_TRAIN_STEPS} 도달 (과금 방지)",
                )
            # 입력 배치는 이미 적절한 디바이스에 있음
            batch_y = batch_y.float().unsqueeze(1)
            opt.zero_grad()
            logits = model(batch_x).unsqueeze(1)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            step += 1
    except torch.cuda.OutOfMemoryError:
        save_checkpoint(model, opt, step, "CUDA OOM")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "step": step,
        },
        MODEL_PATH,
    )
    print(f"\n[진행상황] 최종 모델 저장 완료 -> {MODEL_PATH}")
    print("[진행상황] 평가 중...")
    model.eval()
    with torch.no_grad():
        X_t = X_test_t
        logits = model(X_t).detach().cpu().numpy()
    y_prob = 1 / (1 + np.exp(-logits))
    y_pred = (y_prob >= 0.5).astype(int)
    print("\n[결과] 분류 성능 (한글)")
    print(classification_report(y_test, y_pred, target_names=["저혈압 없음", "저혈압"]))
    try:
        auc = roc_auc_score(y_test, y_prob)
        print("AUC-ROC:", auc)
    except ValueError:
        print("AUC-ROC: (단일 클래스라 계산 생략)")
    print("혼동 행렬:\n", confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
