"""
데이터 로딩 및 전처리.
- VitalDB .vital 로드 및 라벨 생성 (build_dataset에서 사용)
- CSV 기반 Dataset 클래스 및 train/val/test 전처리 (학습용)
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import vitaldb
except ImportError:
    vitaldb = None

from config import (
    VITAL_DIR,
    MAP_THRESHOLD_MMHG,
    HYPOTENSION_DURATION_SEC,
    LOOKBACK_MIN,
    PREDICTION_HORIZON_MIN,
    SAMPLE_INTERVAL_SEC,
    TRACK_MAP,
    TRACKS_VITAL,
    TEST_SIZE,
    VAL_RATIO,
    RANDOM_STATE,
)


# ---------------------------------------------------------------------------
# VitalDB .vital 로드 및 라벨 생성 (데이터셋 구축용)
# ---------------------------------------------------------------------------

def load_vital_case(caseid: int, max_retries: int = 3) -> pd.DataFrame | None:
    """VitalDB 케이스 로드 (재시도 최대 3회)."""
    if vitaldb is None:
        raise ImportError("pip install vitaldb 필요")
    path = VITAL_DIR / f"{caseid:04d}.vital"
    if not path.exists():
        return None
    for attempt in range(max_retries):
        try:
            vf = vitaldb.VitalFile(str(path))
            return vf.to_pandas(TRACKS_VITAL, SAMPLE_INTERVAL_SEC)
        except Exception:
            if attempt < max_retries - 1:
                continue
            return None


def build_labels_for_case(df: pd.DataFrame) -> np.ndarray:
    """케이스 시계열에서 구간별 저혈압 발생 라벨(0/1) 생성 (3-조건 OR)."""
    if df is None or TRACK_MAP not in df.columns or df.empty:
        return np.array([])
    map_vals = np.asarray(df[TRACK_MAP], dtype=float)
    n = len(map_vals)
    lookback_s = LOOKBACK_MIN * 60
    horizon_s = PREDICTION_HORIZON_MIN * 60
    step_s = 60
    labels = []
    for t in range(0, n - lookback_s - horizon_s, step_s):
        future = map_vals[t + lookback_s : t + lookback_s + horizon_s]
        future_clean = future[(future >= 0) & (future < 200)]
        if len(future_clean) == 0:
            labels.append(0)
            continue
        below = (future >= 0) & (future < MAP_THRESHOLD_MMHG)
        run = np.convolve(below.astype(int), np.ones(HYPOTENSION_DURATION_SEC), mode="valid")
        condition1 = np.any(run >= HYPOTENSION_DURATION_SEC) if len(run) > 0 else False
        condition2 = (below.sum() / len(below)) >= 0.2
        condition3 = future_clean.min() < (MAP_THRESHOLD_MMHG - 10)
        label = 1 if (condition1 or condition2 or condition3) else 0
        labels.append(label)
    return np.array(labels)


# ---------------------------------------------------------------------------
# 학습용: CSV 로드, 케이스 단위 분할, 표준화, Dataset 클래스
# ---------------------------------------------------------------------------

def load_csv_and_preprocess(dataset_path):
    """
    CSV 데이터셋 로드 후 케이스 단위 train/val/test 분할 및 StandardScaler 적용.
    Returns:
        dict with: X_train, y_train, X_val, y_val, X_test, y_test (numpy),
                   scaler, feature_cols, has_val (bool)
    """
    df = pd.read_csv(dataset_path)
    target = "label"
    feature_cols = [c for c in df.columns if c not in ("caseid", target)]
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    y = df[target].values.astype(np.int64)
    caseids = df["caseid"].values if "caseid" in df.columns else None

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
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )
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

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    if len(X_val) > 0:
        X_val = scaler.transform(X_val).astype(np.float32)

    has_val = len(X_val) > 0
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "has_val": has_val,
    }


class HypotensionDataset(Dataset):
    """저혈압 예측용 PyTorch Dataset. (X, y) 배열을 (feature, label) 쌍으로 반환."""

    def __init__(self, X: np.ndarray, y: np.ndarray, device=None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.device = device

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.device is not None:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
        return x, y
