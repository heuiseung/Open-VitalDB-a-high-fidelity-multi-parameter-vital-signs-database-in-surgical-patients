"""
라벨 생성 로직 디버깅 스크립트
- MAP < 70 mmHg 기준값 검증
- 저혈압 이벤트 실제 발생 여부 확인
"""
import os
import sys
os.environ['PYTHONIOENCODING'] = 'utf-8'

from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import numpy as np
from tqdm import tqdm

from config import (
    VITAL_DIR, CLINICAL_CSV, MAP_THRESHOLD_MMHG,
    HYPOTENSION_DURATION_SEC, LOOKBACK_MIN, PREDICTION_HORIZON_MIN
)
from data_loader import load_vital_case, build_labels_for_case

print("=" * 70)
print("[진단] 저혈압 라벨 생성 로직 검증")
print("=" * 70)

# 1. 기준값 확인
print(f"\n[설정값]")
print(f"- MAP 임계값: {MAP_THRESHOLD_MMHG} mmHg")
print(f"- 저혈압 지속 시간: {HYPOTENSION_DURATION_SEC}초")
print(f"- 룩백 윈도우: {LOOKBACK_MIN}분")
print(f"- 예측 호라이즌: {PREDICTION_HORIZON_MIN}분")

# 2. 데이터 로드
clinical = pd.read_csv(CLINICAL_CSV)
caseids = clinical["caseid"].dropna().astype(int).unique()[:20]  # 처음 20건만

print(f"\n[데이터] 검사할 케이스: {len(caseids)}건")

statistics = {
    'total_cases': 0,
    'cases_with_data': 0,
    'cases_with_map': 0,
    'cases_with_hypotension': 0,
    'map_values_below_threshold': 0,
    'total_map_samples': 0,
    'min_map_found': 999,
    'max_map_found': -999,
}

for caseid in tqdm(caseids, desc="케이스 분석 중"):
    statistics['total_cases'] += 1
    
    path = VITAL_DIR / f"{caseid:04d}.vital"
    if not path.exists():
        continue
    
    df = load_vital_case(caseid)
    if df is None or df.empty:
        continue
    
    statistics['cases_with_data'] += 1
    
    # MAP 컬럼 확인
    from config import TRACK_MAP
    if TRACK_MAP not in df.columns:
        continue
    
    statistics['cases_with_map'] += 1
    
    # MAP 값 통계
    map_vals = pd.to_numeric(df[TRACK_MAP], errors='coerce').dropna()
    if len(map_vals) == 0:
        continue
    
    statistics['total_map_samples'] += len(map_vals)
    statistics['min_map_found'] = min(statistics['min_map_found'], map_vals.min())
    statistics['max_map_found'] = max(statistics['max_map_found'], map_vals.max())
    
    # 임계값 이하인 샘플 확인
    below_threshold = (map_vals < MAP_THRESHOLD_MMHG).sum()
    statistics['map_values_below_threshold'] += below_threshold
    
    # 라벨 생성 시뮬레이션
    labels = build_labels_for_case(df)
    if len(labels) > 0 and labels.sum() > 0:
        statistics['cases_with_hypotension'] += 1
        print(f"\n[HYPOTENSION DETECTED] Case {caseid:04d}")
        print(f"   - Label count: {len(labels)}")
        print(f"   - Hypotension events: {labels.sum()}")
        print(f"   - Hypotension rate: {labels.mean()*100:.1f}%")
        print(f"   - MAP minimum: {map_vals.min():.1f} mmHg")
        print(f"   - MAP average: {map_vals.mean():.1f} mmHg")

# 3. 결과 요약
print("\n" + "=" * 70)
print("[진단 결과]")
print("=" * 70)
print(f"- 총 케이스: {statistics['total_cases']}")
print(f"- 데이터 있는 케이스: {statistics['cases_with_data']}")
print(f"- MAP 데이터 있는 케이스: {statistics['cases_with_map']}")
print(f"- 저혈압 감지된 케이스: {statistics['cases_with_hypotension']}")
print(f"\n- 총 MAP 샘플: {statistics['total_map_samples']:,}")
print(f"- 임계값({MAP_THRESHOLD_MMHG} mmHg) 이하: {statistics['map_values_below_threshold']:,} ({statistics['map_values_below_threshold']/max(1, statistics['total_map_samples'])*100:.1f}%)")
print(f"- MAP 범위: {statistics['min_map_found']:.1f} ~ {statistics['max_map_found']:.1f} mmHg")

# 4. 진단
print("\n" + "=" * 70)
print("[진단 해석]")
print("=" * 70)

if statistics['cases_with_hypotension'] == 0:
    print("[PROBLEM] No hypotension events found!")
    print("\nRoot Cause Analysis:")
    
    if statistics['map_values_below_threshold'] == 0:
        print("  1. MAP never falls below threshold")
        print(f"     → Lowest MAP: {statistics['min_map_found']:.1f} mmHg")
        print("     → Patients in stable condition")
    else:
        print(f"  1. Found MAP < {MAP_THRESHOLD_MMHG} mmHg samples: {statistics['map_values_below_threshold']:,}")
        print(f"  2. But 'sustained duration' condition not met")
        print(f"     → No {HYPOTENSION_DURATION_SEC}-second continuous hypotension")
        print(f"     → Sharp fluctuations but too brief")
    
    print("\n[SOLUTION]")
    print(f"  1. Lower threshold: 65 → 70 mmHg (DONE)")
    print(f"  2. Shorten duration: 60s → 30s (DONE)")
    print(f"  3. Check if improvement works")
else:
    print(f"[SUCCESS] Hypotension events detected!")
    print(f"   {statistics['cases_with_hypotension']} cases with hypotension")

# 5. 권장사항
print("\n" + "=" * 70)
print("[권장 개선 방안]")
print("=" * 70)
print("""
Option 1: 기준값 조정 (즉시 시행 가능)
  - config.py 수정:
    MAP_THRESHOLD_MMHG = 70        # 65 → 70
    HYPOTENSION_DURATION_SEC = 30  # 60 → 30

Option 2: 다중 기준 적용
  - 다양한 기준 동시 사용:
    MAP < 65 AND 수축기 < 90
    또는
    MAP < 70 AND 심박 > 100

Option 3: 동적 임계값
  - 환자별 기준값 다르게 설정
    (정상 상태 대비 20% 감소 등)
""")
