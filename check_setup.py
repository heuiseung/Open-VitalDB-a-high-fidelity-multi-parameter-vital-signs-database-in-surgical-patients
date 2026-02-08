"""설정·경로 검증 (한글 진행상황). 실행: python check_setup.py"""
from __future__ import annotations

import sys
import io
from pathlib import Path

# Windows cp949에서 한글/기호 출력
try:
    if sys.stdout.encoding and "cp" in (sys.stdout.encoding or "").lower():
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

# 프로젝트 루트 기준 실행
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    BASE_DIR,
    VITAL_DIR,
    CLINICAL_CSV,
    DATASET_PATH,
    check_data_paths,
)


def main() -> None:
    print("=" * 50)
    print("저혈압 조기 예측 프로젝트 - 설정 검증")
    print("=" * 50)
    print(f"[경로] 프로젝트: {BASE_DIR}")
    print(f"[경로] VitalDB 루트: {CLINICAL_CSV.parent}")
    ok, msg = check_data_paths()
    print(f"[검증] {msg}")
    if not ok:
        print("[종료] 데이터 경로를 확인한 뒤 다시 실행하세요.")
        return
    try:
        import pandas as pd
        clinical = pd.read_csv(CLINICAL_CSV)
        caseids = clinical["caseid"].dropna().astype(int)
        n_cases = len(caseids)
        sample = caseids.head(10).tolist()
        n_vital = sum(1 for c in sample if (VITAL_DIR / f"{c:04d}.vital").exists())
        print(f"[데이터] clinical_data.csv: {n_cases}건 케이스")
        print(f"[데이터] vital_files 샘플(상위 10건): {n_vital}/10건 .vital 존재")
    except Exception as e:
        print(f"[경고] 데이터 읽기: {e}")
    if DATASET_PATH.exists():
        import pandas as pd
        df = pd.read_csv(DATASET_PATH)
        print(f"[기존] hypotension_dataset.csv: {len(df)}행")
    else:
        print("[기존] hypotension_dataset.csv 없음 → run_all.py 또는 노트북 실행 시 생성")
    print("=" * 50)
    print("[완료] 설정 검증 끝. run_all.py 또는 hypotension_pipeline.ipynb 실행 가능.")


if __name__ == "__main__":
    main()
