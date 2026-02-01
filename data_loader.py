"""Vital 파일 로드 + 저혈압 라벨 (진행률 표시)"""
import pandas as pd
import numpy as np
from pathlib import Path

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
)


def load_vital_case(caseid: int) -> pd.DataFrame | None:
    if vitaldb is None:
        raise ImportError("pip install vitaldb 필요")
    path = VITAL_DIR / f"{caseid:04d}.vital"
    if not path.exists():
        return None
    try:
        vf = vitaldb.VitalFile(str(path))
        return vf.to_pandas(TRACKS_VITAL, SAMPLE_INTERVAL_SEC)
    except Exception:
        return None


def build_labels_for_case(df: pd.DataFrame) -> np.ndarray:
    """케이스 시계열에서 구간별 저혈압 발생 라벨(0/1) 생성."""
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
        below = (future >= 0) & (future < MAP_THRESHOLD_MMHG)
        run = np.convolve(
            below.astype(int), np.ones(HYPOTENSION_DURATION_SEC), mode="valid"
        )
        labels.append(1 if np.any(run >= HYPOTENSION_DURATION_SEC) else 0)
    return np.array(labels)
