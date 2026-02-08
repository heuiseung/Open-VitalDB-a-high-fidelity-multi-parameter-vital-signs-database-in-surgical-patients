"""
학습 루프 및 실행 코드.

데이터 로드 → GPU 배치 학습 → 검증 AUC 기준 best 모델 저장 → 테스트 평가 출력.
config, data_loader, model, utils에 의존하며, 단독 실행 시 `python train.py`로 호출.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from tqdm import tqdm

from config import (
    DATASET_PATH,
    CHECKPOINT_DIR,
    MODEL_PATH,
    BATCH_SIZE,
    N_EPOCHS,
    MAX_TRAIN_STEPS,
)
from data_loader import load_csv_and_preprocess
from model import HypotensionModel
from utils import set_utf8_stdout, get_device, save_checkpoint


def main() -> None:
    """전체 학습 파이프라인: 전처리 → 학습 → 검증 best 저장 → 테스트 평가.

    데이터셋이 없으면 안내만 하고 종료. 있으면 케이스 단위 분할·표준화 후
    GPU에 텐서를 올리고, BCEWithLogitsLoss + pos_weight로 불균형을 보정하며 학습.
    검증 세트가 있으면 에폭마다 AUC를 계산해 best 모델만 저장하고, 마지막에 해당
    체크포인트를 로드해 테스트 평가를 출력합니다.

    Returns:
        None. 모든 결과는 콘솔 출력 및 MODEL_PATH 체크포인트로 남김.

    Raises:
        torch.cuda.OutOfMemoryError: GPU 메모리 부족 시 예외 후 모델 저장하고 루프 종료.
    """
    set_utf8_stdout()

    if not DATASET_PATH.exists():
        print("[안내] 먼저 데이터셋 구축을 실행해 주세요. (python main.py 또는 build_dataset.py)")
        return

    data = load_csv_and_preprocess(DATASET_PATH)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    feature_cols = data["feature_cols"]
    has_val = data["has_val"]

    device, use_cuda = get_device()
    if use_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[CUDA] GPU: {gpu_name} (약 {gpu_mem:.1f} GB)")
        print("[CUDA] 학습/검증/테스트 데이터를 GPU 메모리에 올립니다.")
    print(f"[진행] 학습 장치: {device} | train {len(X_train)}건, test {len(X_test)}건")

    # 전체 학습/테스트(및 검증) 데이터를 한 번에 GPU로 올려 배치 시 추가 복사 없이 사용
    X_train_t = torch.from_numpy(X_train).to(device=device, dtype=torch.float32, non_blocking=True)
    y_train_t = torch.from_numpy(y_train).to(device=device, dtype=torch.float32, non_blocking=True)
    X_test_t = torch.from_numpy(X_test).to(device=device, dtype=torch.float32, non_blocking=True)
    if has_val:
        X_val_t = torch.from_numpy(X_val).to(device=device, dtype=torch.float32, non_blocking=True)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)

    in_dim = len(feature_cols)
    model = HypotensionModel(in_dim=in_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # 양성(저혈압)이 적을 때 손실에서 양성 오분류 패널티를 키우기 위함
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if n_pos > 0:
        print(f"[개선] 클래스 가중치 적용 (pos_weight≈{pos_weight.item():.2f})")
    print(f"[개선] 에폭: {N_EPOCHS}, 배치: {BATCH_SIZE}, 검증: {'사용' if has_val else '없음'}")

    best_auc = -1.0
    step = 0
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(N_EPOCHS):
        model.train()
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
                step += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        except torch.cuda.OutOfMemoryError:
            # OOM 시 현재 상태라도 저장 후 종료 (과금/장시간 실행 방지)
            torch.save({"model_state": model.state_dict(), "optimizer_state": opt.state_dict(), "step": step}, MODEL_PATH)
            print("\n[중단] CUDA OOM")
            break
        if MAX_TRAIN_STEPS is not None and step >= MAX_TRAIN_STEPS:
            break

        # 검증이 있으면 에폭마다 AUC 계산 후 best만 저장 (과적합 완화)
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
                        {
                            "model_state": model.state_dict(),
                            "optimizer_state": opt.state_dict(),
                            "step": step,
                            "epoch": epoch,
                            "val_auc": val_auc,
                        },
                        MODEL_PATH,
                    )
                tqdm.write(f"  검증 AUC: {val_auc:.4f} (best: {best_auc:.4f})")
            except ValueError:
                pass
        else:
            # 검증 없으면 매 에폭 마지막 가중치 저장
            torch.save(
                {"model_state": model.state_dict(), "optimizer_state": opt.state_dict(), "step": step, "epoch": epoch},
                MODEL_PATH,
            )

    # 검증으로 best를 저장했으면 그 가중치로 테스트 평가 (최종 보고용)
    if has_val and best_auc >= 0:
        ckpt = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    print(f"\n[진행] 최종 모델 저장 -> {MODEL_PATH}" + (f" (검증 best AUC: {best_auc:.4f})" if best_auc >= 0 else ""))
    print("[진행] 평가 중...")
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t).detach().cpu().numpy()
    y_prob = 1 / (1 + np.exp(-logits))
    y_pred = (y_prob >= 0.5).astype(int)
    print("\n[결과] 분류 성능")
    print(classification_report(y_test, y_pred, target_names=["No hypotension", "Hypotension"]))
    try:
        auc = roc_auc_score(y_test, y_prob)
        print("AUC-ROC:", auc)
    except ValueError:
        print("AUC-ROC: (skip)")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
