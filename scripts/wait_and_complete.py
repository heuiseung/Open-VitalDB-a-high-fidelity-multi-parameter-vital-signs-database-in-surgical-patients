#!/usr/bin/env python3
"""
데이터셋 빌드 완료 대기 스크립트
빌드 완료 후 자동으로:
1. 라벨 분포 확인
2. 로컬 커밋 생성
3. GitHub에 강제 푸시
"""
import os
import sys
import time
import subprocess
import pandas as pd
from pathlib import Path

# 프로젝트 루트를 path에 추가 (scripts/에서 실행 시 루트 모듈 import 가능)
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

csv_path = _root / "hypotension_dataset.csv"

print("=" * 70)
print("⏳ 데이터셋 빌드 완료 대기 중...")
print("=" * 70)

# 빌드 완료 대기
start = time.time()
check_count = 0
while not csv_path.exists():
    elapsed = time.time() - start
    check_count += 1
    if check_count % 12 == 0:  # 1분마다 출력
        print(f"[{int(elapsed//60)}분 {int(elapsed%60)}초] CSV 아직 생성 안됨...")
    time.sleep(5)

print("\n✅ CSV 파일 감지됨! 내용 확인 중...\n")

# CSV 완성 대기 (파일 크기 변화 없을 때까지)
last_size = 0
no_change_count = 0
while no_change_count < 6:  # 30초간 변화 없으면 완료
    try:
        size = csv_path.stat().st_size
        if size == last_size:
            no_change_count += 1
        else:
            no_change_count = 0
        last_size = size
        time.sleep(5)
    except:
        break

print("[완료] 데이터셋 빌드 완료!\n")

# 라벨 분포 확인
print("=" * 70)
print("📊 라벨 분포 확인")
print("=" * 70)
try:
    df = pd.read_csv(csv_path)
    print(f"✅ 총 {len(df):,} 행")
    print("\n라벨 분포:")
    label_counts = df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        pct = count / len(df) * 100
        print(f"  Label {label}: {count:,} ({pct:.1f}%)")
    print()
except Exception as e:
    print(f"❌ 오류: {e}\n")

# 로컬 커밋 생성
print("=" * 70)
print("📝 로컬 커밋 생성")
print("=" * 70)
os.system('cd C:\\Users\\sck32\\hypo_vitaldb && "C:\\Program Files\\Git\\cmd\\git.exe" add -A && "C:\\Program Files\\Git\\cmd\\git.exe" commit -m "feat(dataset): rebuild with 3-condition OR label logic"')
print()

# 다음 단계 안내
print("=" * 70)
print("✅ 준비 완료!")
print("=" * 70)
print("\n다음 단계:")
print("1. 모델 학습: python train_model.py")
print("2. GitHub 푸시: git push origin main (또는 git push -f origin main)")
print()
