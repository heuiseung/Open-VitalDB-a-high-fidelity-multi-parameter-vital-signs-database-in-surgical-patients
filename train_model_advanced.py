"""
PyTorch CUDA 학습 - 고급 버전 (Batch Norm, Early Stopping, 더 깊은 아키텍처)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from tqdm import tqdm
import matplotlib.pyplot as plt

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


class HypoNetAdvanced(nn.Module):
    """더 깊고 강력한 신경망"""
    def __init__(self, in_dim, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def save_checkpoint(model, optimizer, step, reason=""):
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
    print(f"\n[저장] 모델 -> {MODEL_PATH}")
    if reason:
        print(f"[중단] {reason}")


def main():
    print("=" * 70)
    print("[고급 모델] 저혈압 조기 예측 (Batch Norm + Early Stopping)")
    print("=" * 70)
    
    if not DATASET_PATH.exists():
        print("먼저 build_dataset.py 를 실행해 주세요.")
        return

    df = pd.read_csv(DATASET_PATH)
    target = "label"
    feature_cols = [c for c in df.columns if c not in ("caseid", target)]
    
    print(f"\n[데이터] {len(df)}행, {len(feature_cols)}개 특성")
    print(f"[분포] Label 0: {(df[target]==0).sum()}, Label 1: {(df[target]==1).sum()}")
    
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    y = df[target].values.astype(np.int64)
    
    # 정규화
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1
    X = (X - X_mean) / X_std
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"\n[장치] {device}")
    print(f"[학습] train {len(X_train)}건, test {len(X_test)}건")
    
    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=0)
    
    model = HypoNetAdvanced(len(feature_cols), dropout_rate=0.3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=3, verbose=False
    )
    loss_fn = nn.BCEWithLogitsLoss()
    
    model.train()
    step = 0
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 5
    
    losses = []
    
    try:
        pbar = tqdm(train_loader, desc="[학습] 진행 중", unit="batch")
        for batch_x, batch_y in pbar:
            if MAX_TRAIN_STEPS is not None and step >= MAX_TRAIN_STEPS:
                save_checkpoint(
                    model, opt, step,
                    f"최대 스텝 {MAX_TRAIN_STEPS} 도달",
                )
                raise SystemExit(0)
            
            batch_x = batch_x.to(device)
            batch_y = batch_y.float().unsqueeze(1).to(device)
            
            opt.zero_grad()
            logits = model(batch_x).unsqueeze(1)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            
            loss_val = loss.item()
            losses.append(loss_val)
            pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{opt.param_groups[0]['lr']:.2e}")
            
            # Early Stopping
            if loss_val < best_loss:
                best_loss = loss_val
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"\n[조기 종료] {max_patience}배치 개선 없음")
                    break
            
            scheduler.step(loss_val)
            step += 1
    
    except torch.cuda.OutOfMemoryError:
        save_checkpoint(model, opt, step, "CUDA OOM")
        raise SystemExit(0)
    
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "step": step,
            "feature_mean": X_mean,
            "feature_std": X_std,
        },
        MODEL_PATH,
    )
    print(f"\n[저장] 최종 모델 -> {MODEL_PATH}")
    
    # 평가
    print("\n" + "=" * 70)
    print("[평가] 테스트 성능")
    print("=" * 70)
    
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X_test).to(device)
        logits = model(X_t).cpu().numpy()
    
    y_prob = 1 / (1 + np.exp(-logits))
    y_pred = (y_prob >= 0.5).astype(int)
    
    try:
        print(classification_report(y_test, y_pred, target_names=["저혈압 없음", "저혈압"], zero_division=0))
        try:
            auc_score = roc_auc_score(y_test, y_prob)
            print(f"[AUC-ROC] {auc_score:.4f}")
        except:
            print("[AUC-ROC] 계산 불가 (한 클래스만 존재)")
    except ValueError:
        print(classification_report(y_test, y_pred, zero_division=0))
    
    print("[혼동 행렬]")
    print(confusion_matrix(y_test, y_pred))
    
    # 손실 그래프
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(CHECKPOINT_DIR / 'training_loss.png', dpi=100, bbox_inches='tight')
    print(f"\n[저장] 손실 그래프 -> {CHECKPOINT_DIR / 'training_loss.png'}")
    
    # ROC 커브 (가능한 경우)
    if len(np.unique(y_test)) > 1:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(CHECKPOINT_DIR / 'roc_curve.png', dpi=100, bbox_inches='tight')
        print(f"[저장] ROC 커브 -> {CHECKPOINT_DIR / 'roc_curve.png'}")
    
    print("\n" + "=" * 70)
    print("[완료] 고급 모델 학습 및 평가 완료")
    print("=" * 70)


if __name__ == "__main__":
    main()
