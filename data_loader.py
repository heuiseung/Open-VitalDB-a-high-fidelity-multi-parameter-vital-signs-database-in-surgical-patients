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


def load_vital_case(caseid: int, max_retries: int = 3) -> pd.DataFrame | None:
    """VitalDB 케이스 로드 (재시도 최대 3회)"""
    if vitaldb is None:
        raise ImportError("pip install vitaldb 필요")
    path = VITAL_DIR / f"{caseid:04d}.vital"
    if not path.exists():
        return None
    
    for attempt in range(max_retries):
        try:
            vf = vitaldb.VitalFile(str(path))
            return vf.to_pandas(TRACKS_VITAL, SAMPLE_INTERVAL_SEC)
        except Exception as e:
            if attempt < max_retries - 1:
                continue  # 재시도
            return None


def build_labels_for_case(df: pd.DataFrame) -> np.ndarray:
    """케이스 시계열에서 구간별 저혈압 발생 라벨(0/1) 생성 (3-조건 OR 로직)."""
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
        # 유효한 범위 필터링 (0-200 mmHg)
        future_clean = future[(future >= 0) & (future < 200)]
        if len(future_clean) == 0:
            labels.append(0)
            continue
        
        below = (future >= 0) & (future < MAP_THRESHOLD_MMHG)
        
        # 3가지 조건 중 하나라도 만족하면 저혈압 판정
        # 조건1: HYPOTENSION_DURATION_SEC 초 이상 연속 저혈압
        run = np.convolve(
            below.astype(int), np.ones(HYPOTENSION_DURATION_SEC), mode="valid"
        )
        condition1 = np.any(run >= HYPOTENSION_DURATION_SEC) if len(run) > 0 else False
        
        # 조건2: 20% 이상의 샘플이 threshold 이하
        condition2 = (below.sum() / len(below)) >= 0.2
        
        # 조건3: 최소 MAP이 매우 낮음 (threshold - 10)
        condition3 = future_clean.min() < (MAP_THRESHOLD_MMHG - 10)
        
        label = 1 if (condition1 or condition2 or condition3) else 0
        labels.append(label)
    return np.array(labels)
