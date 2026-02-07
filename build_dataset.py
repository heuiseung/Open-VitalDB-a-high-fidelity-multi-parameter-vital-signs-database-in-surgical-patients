"""데이터셋 구축 - 진행률 표시, 최대 실행 시간 도달 시 저장 후 중단 (과금 방지)"""
import pandas as pd
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm

from config import (
    VITAL_DIR,
    CLINICAL_CSV,
    LOOKBACK_MIN,
    TRACK_MAP,
    TRACK_HR,
    TRACK_SBP,
    TRACK_DBP,
    DATASET_PATH,
    MAX_RUNTIME_MINUTES,
    check_data_paths,
)
from data_loader import load_vital_case, build_labels_for_case

LOOKBACK_SEC = LOOKBACK_MIN * 60
MAX_CASES = 500  # 테스트: 500개로 제한하여 빠르게 검증


def extract_features(df: pd.DataFrame, start_idx: int) -> dict | None:
    """LOOKBACK 구간에서 MAP/HR/SBP/DBP 요약 통계 + 추세 추출 (특성 확장)."""
    end_idx = min(start_idx + LOOKBACK_SEC, len(df))
    if end_idx - start_idx < LOOKBACK_SEC // 2:
        return None
    seg = df.iloc[start_idx:end_idx]
    feats = {}
    tracks = [TRACK_MAP, TRACK_HR, TRACK_SBP, TRACK_DBP]
    for col in tracks:
        if col not in seg.columns:
            continue
        s = pd.to_numeric(seg[col], errors="coerce").dropna()
        if len(s) < 10:
            continue
        key = col.split("/")[-1]
        feats[f"{key}_mean"] = s.mean()
        feats[f"{key}_std"] = s.std() if len(s) > 1 else 0.0
        feats[f"{key}_min"] = s.min()
        feats[f"{key}_max"] = s.max()
        # 추세: (끝 - 처음) / 길이 (구간 내 변화율)
        feats[f"{key}_trend"] = (s.iloc[-1] - s.iloc[0]) / len(s) if len(s) > 0 else 0.0
    return feats if feats else None


def save_and_exit(rows: list, reason: str):
    out = pd.DataFrame(rows)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(DATASET_PATH, index=False)
    print(f"\n[저장 완료] {len(out)}행 -> {DATASET_PATH}")
    print(f"[중단] {reason}")
    raise SystemExit(0)


def main() -> None:
    ok, msg = check_data_paths()
    if not ok:
        print(f"[오류] {msg}")
        return
    clinical = pd.read_csv(CLINICAL_CSV)
    caseids = clinical["caseid"].dropna().astype(int).unique()
    if MAX_CASES:
        caseids = caseids[:MAX_CASES]
    total = len(caseids)
    print(f"[진행상황] 데이터셋 구축 시작 (총 {total}건 케이스 예정)")
    rows = []
    start_time = time.perf_counter()
    limit_sec = (MAX_RUNTIME_MINUTES * 60) if MAX_RUNTIME_MINUTES else None
    try:
        for caseid in tqdm(caseids, desc="[1/2] 케이스 처리 중", unit="건"):
            if limit_sec is not None:
                elapsed = time.perf_counter() - start_time
                if elapsed >= limit_sec:
                    save_and_exit(
                        rows,
                        f"최대 실행 시간 {MAX_RUNTIME_MINUTES}분 도달 (과금 방지)",
                    )
            path = VITAL_DIR / f"{caseid:04d}.vital"
            if not path.exists():
                continue
            try:
                df = load_vital_case(caseid)
            except Exception as e:
                print(f"⚠️ 케이스 {caseid:04d} 로드 실패: {e}")
                continue
            if df is None or df.empty:
                continue
            labels = build_labels_for_case(df)
            if len(labels) == 0:
                continue
            for i, label in enumerate(labels):
                feats = extract_features(df, i * 60)
                if feats is None:
                    continue
                feats["caseid"] = caseid
                feats["label"] = int(label)
                rows.append(feats)
    except MemoryError:
        save_and_exit(rows, "시스템 메모리 부족")
    out = pd.DataFrame(rows)
    out.to_csv(DATASET_PATH, index=False)
    print(f"\n[진행상황] 데이터셋 구축 완료 — {len(out)}행 저장: {DATASET_PATH}")


if __name__ == "__main__":
    main()
