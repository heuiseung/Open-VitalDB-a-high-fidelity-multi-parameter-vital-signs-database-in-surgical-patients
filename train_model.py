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
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config import (
    TEST_SIZE,
    VAL_RATIO,
    N_EPOCHS,
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
    caseids = df["caseid"].values if "caseid" in df.columns else None

    # 케이스(caseid) 단위 분할: train / 검증 / 테스트
    val_cases = np.array([])
    X_val = np.zeros((0, 0))
    y_val = np.array([], dtype=np.int64)
    if caseids is not None and len(np.unique(caseids)) > 1:
        unique_cases = np.unique(caseids)
        try:
            train_cases, test_cases = train_test_split(
                unique_cases, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )
            if len(train_cases) > 1 and VAL_RATIO > 0:
                train_cases, val_cases = train_test_split(
                    train_cases, test_size=VAL_RATIO, random_state=RANDOM_STATE
                )
                val_mask = np.isin(caseids, val_cases)
                X_val = X[val_mask].astype(np.float32)
                y_val = y[val_mask]
            else:
                X_val = np.zeros((0, X.shape[1]), dtype=np.float32)
                y_val = np.array([], dtype=np.int64)
        except Exception:
            train_cases = unique_cases[: int(len(unique_cases) * (1 - TEST_SIZE))]
            test_cases = unique_cases[int(len(unique_cases) * (1 - TEST_SIZE)) :]
            X_val = np.zeros((0, X.shape[1]), dtype=np.float32)
            y_val = np.array([], dtype=np.int64)
        train_mask = np.isin(caseids, train_cases)
        test_mask = np.isin(caseids, test_cases)
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        print(f"[개선] 케이스 단위 분할: train {len(train_cases)}케이스, val {len(val_cases)}케이스, test {len(test_cases)}케이스")
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )
            print("[안내] 라벨이 한쪽만 있어 stratify 생략")
        if VAL_RATIO > 0 and len(X_train) > 10:
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=VAL_RATIO, random_state=RANDOM_STATE, stratify=y_train
                )
            except ValueError:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=VAL_RATIO, random_state=RANDOM_STATE
                )
            X_val = X_val.astype(np.float32)
        else:
            X_val = np.zeros((0, X_train.shape[1]), dtype=np.float32)
            y_val = np.array([], dtype=np.int64)

    # 특성 표준화 (학습 데이터 기준 fit, train/val/test 동일 스케일)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    if len(X_val) > 0:
        X_val = scaler.transform(X_val).astype(np.float32)
    print("[개선] 특성 표준화(StandardScaler) 적용")
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
    has_val = len(X_val) > 0
    if has_val:
        X_val_t = torch.from_numpy(X_val).to(device=device, dtype=torch.float32)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    model = HypoNet(len(feature_cols)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if n_pos > 0:
        print(f"[개선] 클래스 가중치 적용 (양성 비율 반영, pos_weight≈{pos_weight.item():.2f})")
    print(f"[개선] 다중 에폭: {N_EPOCHS}회, 검증 세트: {'사용' if has_val else '없음'}")

    best_auc = -1.0
    step = 0
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batch = 0
        pbar = tqdm(train_loader, desc=f"[에폭 {epoch+1}/{N_EPOCHS}] 학습", unit="배치")
        try:
            for batch_x, batch_y in pbar:
                if MAX_TRAIN_STEPS is not None and step >= MAX_TRAIN_STEPS:
                    break
                batch_y = batch_y.float().unsqueeze(1)
                opt.zero_grad()
                logits = model(batch_x).unsqueeze(1)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                n_batch += 1
                step += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        except torch.cuda.OutOfMemoryError:
            torch.save({"model_state": model.state_dict(), "optimizer_state": opt.state_dict(), "step": step}, MODEL_PATH)
            print("\n[중단] CUDA OOM")
            break
        if MAX_TRAIN_STEPS is not None and step >= MAX_TRAIN_STEPS:
            break

        # 검증 AUC 계산 및 best 모델 저장
        if has_val and len(y_val) > 0:
            model.eval()
            with torch.no_grad():
                logits_val = model(X_val_t).detach().cpu().numpy()
            y_val_prob = 1 / (1 + np.exp(-logits_val))
            try:
                val_auc = roc_auc_score(y_val, y_val_prob)
                if val_auc > best_auc:
                    best_auc = val_auc
                    torch.save(
                        {"model_state": model.state_dict(), "optimizer_state": opt.state_dict(), "step": step, "epoch": epoch, "val_auc": val_auc},
                        MODEL_PATH,
                    )
                tqdm.write(f"  검증 AUC: {val_auc:.4f} (best: {best_auc:.4f})")
            except ValueError:
                pass
        else:
            # 검증 없으면 매 에폭 저장
            torch.save(
                {"model_state": model.state_dict(), "optimizer_state": opt.state_dict(), "step": step, "epoch": epoch},
                MODEL_PATH,
            )

    # best 모델 로드 (검증 사용 시; 아니면 마지막 모델이 이미 MODEL_PATH에 있음)
    if has_val and best_auc >= 0:
        ckpt = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state"])
    print(f"\n[진행상황] 최종 모델 저장 완료 -> {MODEL_PATH}" + (f" (검증 best AUC: {best_auc:.4f})" if best_auc >= 0 else ""))
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
