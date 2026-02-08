"""
설정 및 헬퍼 함수.
"""
import sys
import io
import torch
from pathlib import Path

from config import (
    DEVICE,
    CHECKPOINT_DIR,
    MODEL_PATH,
    TRAIN_STATE_PATH,
    N_EPOCHS,
    BATCH_SIZE,
    MAX_TRAIN_STEPS,
    TEST_SIZE,
    VAL_RATIO,
    RANDOM_STATE,
    DATASET_PATH,
)


def set_utf8_stdout() -> None:
    """Windows 콘솔 한글 출력을 위해 stdout/stderr를 UTF-8로 래핑."""
    if getattr(sys.stdout, "buffer", None) and (sys.stdout.encoding or "").lower() != "utf-8":
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
        except Exception:
            pass


def get_device():
    """CUDA 사용 가능 시 device 및 설정 적용. (device, use_cuda) 반환."""
    use_cuda = torch.cuda.is_available() and DEVICE.startswith("cuda")
    device = torch.device(DEVICE if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device, use_cuda


def save_checkpoint(model, optimizer, step, reason: str = "") -> None:
    """모델·옵티마이저·스텝 저장 후 SystemExit(0)."""
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
