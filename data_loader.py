"""
데이터 로딩 및 전처리 모듈.

- VitalDB .vital 파일 로드 및 라벨 생성 (build_dataset에서 사용).
- CSV 기반 학습용 전처리: 케이스 단위 분할, 표준화, PyTorch Dataset 제공.
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
    """VitalDB 단일 케이스 .vital 파일을 DataFrame으로 로드 (재시도 포함).

    Args:
        caseid: VitalDB 케이스 ID (예: 1 → 0001.vital).
        max_retries: 로드 실패 시 재시도 횟수. 기본 3회.

    Returns:
        TRACKS_VITAL 트랙을 SAMPLE_INTERVAL_SEC 간격으로 샘플링한 DataFrame.
        파일 없음/실패 시 None.

    Raises:
        ImportError: vitaldb 패키지가 없을 때 (pip install vitaldb 필요).
    """
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
    """케이스 시계열에서 구간별 저혈압 발생 여부(0/1) 라벨 생성 (3-조건 OR).

    lookback 구간 이후 prediction_horizon 구간 내에 저혈압이 발생하면 1, 아니면 0.
    조건: (1) 연속 N초 이상 MAP < 임계값 (2) 구간 내 20% 이상 저혈압 (3) 최소 MAP가 임계값-10 미만.

    Args:
        df: load_vital_case로 얻은 DataFrame. TRACK_MAP(MAP) 컬럼 필요.

    Returns:
        각 시간 스텝(step_s 간격)에 대한 이진 라벨 배열. 길이 0 가능 (데이터 부족 시).
    """
    if df is None or TRACK_MAP not in df.columns or df.empty:
        return np.array([])
    map_vals = np.asarray(df[TRACK_MAP], dtype=float)
    n = len(map_vals)
    lookback_s = LOOKBACK_MIN * 60
    horizon_s = PREDICTION_HORIZON_MIN * 60
    step_s = 60
    labels = []
    for t in range(0, n - lookback_s - horizon_s, step_s):
        # 미래 구간만 사용: lookback 끝 ~ horizon 끝
        future = map_vals[t + lookback_s : t + lookback_s + horizon_s]
        future_clean = future[(future >= 0) & (future < 200)]
        if len(future_clean) == 0:
            labels.append(0)
            continue
        below = (future >= 0) & (future < MAP_THRESHOLD_MMHG)
        # 연속 N초 이상 저혈압: 1D convolution으로 “연속 1” 길이 계산
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

def load_csv_and_preprocess(dataset_path) -> dict:
    """CSV 데이터셋 로드 후 케이스 단위 train/val/test 분할 및 StandardScaler 적용.

    caseid가 있으면 같은 케이스가 train/val/test에 섞이지 않도록 분할하여
    데이터 누수(data leakage)를 막습니다. 없으면 행 단위 무작위 분할.

    Args:
        dataset_path: hypotension_dataset.csv 등 CSV 파일 경로 (Path 또는 str).

    Returns:
        다음 키를 가진 dict:
            - X_train, y_train, X_val, y_val, X_test, y_test: numpy 배열 (float32 / int64).
            - scaler: fit된 StandardScaler (학습 데이터 기준).
            - feature_cols: 특성 컬럼 이름 리스트.
            - has_val: 검증 세트가 비어 있지 않으면 True.

    Raises:
        FileNotFoundError: dataset_path에 해당 파일이 없을 때.
        KeyError: CSV에 'label' 또는 필요한 컬럼이 없을 때.
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

    # 케이스 단위 분할: 동일 caseid가 train/val/test에 나뉘지 않도록
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
        # caseid 없으면 행 단위 분할 (stratify로 클래스 비율 유지 시도)
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

    # 학습 데이터로만 fit하여 val/test에 동일 스케일 적용 (정보 누수 방지)
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
    """저혈압 예측용 PyTorch Dataset. (X, y) numpy 배열을 (feature, label) 텐서 쌍으로 반환.

    Attributes:
        X: 특성 텐서 (float).
        y: 라벨 텐서 (float).
        device: None이면 CPU 텐서 반환; 지정 시 __getitem__에서 해당 device로 이동.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, device=None) -> None:
        """HypotensionDataset 초기화.

        Args:
            X: 특성 배열 (n_samples, n_features). float32 권장.
            y: 라벨 배열 (n_samples,). 0/1 이진.
            device: torch device 또는 None. None이면 CPU; 지정 시 배치를 해당 device로 반환.
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.device = device

    def __len__(self) -> int:
        """데이터셋 샘플 수."""
        return len(self.X)

    def __getitem__(self, idx: int):
        """idx번째 (특성, 라벨) 쌍 반환. device가 설정되어 있으면 해당 device로 이동.

        Args:
            idx: 샘플 인덱스 (0 ~ len-1).

        Returns:
            (x, y): 각각 (n_features,), 스칼라. device 지정 시 해당 device의 텐서.
        """
        x = self.X[idx]
        y = self.y[idx]
        if self.device is not None:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
        return x, y
