#!/usr/bin/env python3
"""데이터셋 빌드 진행 상황 모니터링 (3초마다 갱신)"""
import os
import sys
import time
import subprocess
from pathlib import Path

# 프로젝트 루트를 path에 추가 (scripts/에서 실행 시 루트 모듈 import 가능)
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

csv_path = _root / "hypotension_dataset.csv"

def get_csv_stats():
    """현재 CSV 파일 통계 반환"""
    if not csv_path.exists():
        return None, None
    try:
        size = csv_path.stat().st_size / (1024 * 1024)  # MB
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = sum(1 for _ in f)
        return lines, size
    except:
        return None, None

def clear_screen():
    """화면 지우기"""
    os.system('cls' if os.name == 'nt' else 'clear')

print("=" * 60)
print("📊 데이터셋 빌드 모니터링 (build_dataset.py)")
print("=" * 60)
print("명령어:")
print("  Ctrl+C: 모니터링 중지")
print("-" * 60)

try:
    while True:
        lines, size = get_csv_stats()
        if lines is not None:
            progress = (lines - 1) / 6388 * 100  # 헤더 제외
            print(f"[{time.strftime('%H:%M:%S')}] 진행률: {progress:.1f}% | 행: {lines-1:,} | 크기: {size:.1f} MB", flush=True)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] CSV 아직 생성 안됨...", flush=True)
        
        time.sleep(3)
except KeyboardInterrupt:
    print("\n[중지] 모니터링 종료. build_dataset.py는 계속 실행 중입니다.")
