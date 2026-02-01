#!/usr/bin/env python3
"""
데이터셋 빌드 완료 후 자동 처리
1. 라벨 분포 확인 (Label 0/1 비율)
2. 만약 혼합 라벨 발견 시 전체 빌드 진행
3. 모델 재학습
4. GitHub 강제 푸시
"""
import os
import sys
import time
import pandas as pd
import subprocess
from pathlib import Path

csv_path = Path("hypotension_dataset.csv")

def wait_for_csv(timeout_sec=1800):  # 30분 대기
    """CSV 파일 완성 대기"""
    print("⏳ 데이터셋 빌드 완료 대기 중...")
    start = time.time()
    last_size = 0
    no_change_count = 0
    
    while (time.time() - start) < timeout_sec:
        if csv_path.exists():
            try:
                size = csv_path.stat().st_size
                if size == last_size:
                    no_change_count += 1
                    if no_change_count >= 6:  # 30초 변화 없음
                        return True
                else:
                    no_change_count = 0
                last_size = size
            except:
                pass
        time.sleep(5)
    
    return False

def check_labels():
    """라벨 분포 확인"""
    try:
        df = pd.read_csv(csv_path)
        print(f"\n✅ CSV 완성: {len(df):,} 행\n")
        
        label_counts = df['label'].value_counts().sort_index()
        print("📊 라벨 분포:")
        for label, count in label_counts.items():
            pct = count / len(df) * 100
            print(f"  Label {label}: {count:,} ({pct:.1f}%)")
        
        # 혼합 라벨 확인
        has_label_1 = 1 in label_counts.index
        return has_label_1, df
    except Exception as e:
        print(f"❌ 오류: {e}")
        return False, None

def build_full_dataset():
    """전체 데이터셋 빌드"""
    print("\n" + "=" * 70)
    print("🚀 전체 데이터셋 빌드 시작")
    print("=" * 70)
    
    # MAX_CASES = None으로 변경
    config_file = Path("build_dataset.py")
    content = config_file.read_text(encoding='utf-8')
    content = content.replace(
        "MAX_CASES = 500  # 테스트: 500개로 제한",
        "MAX_CASES = None  # 전체 데이터 처리"
    )
    config_file.write_text(content, encoding='utf-8')
    
    # CSV 삭제
    csv_path.unlink(missing_ok=True)
    
    # 빌드 시작
    subprocess.run([
        sys.executable, "build_dataset.py"
    ], check=False)

def train_model():
    """모델 재학습"""
    print("\n" + "=" * 70)
    print("🤖 모델 재학습 시작")
    print("=" * 70 + "\n")
    
    subprocess.run([
        sys.executable, "train_model.py"
    ], check=False)

def commit_and_push():
    """로컬 커밋 생성 및 GitHub 강제 푸시"""
    print("\n" + "=" * 70)
    print("💾 GitHub에 강제 저장")
    print("=" * 70 + "\n")
    
    os.system('cd C:\\Users\\sck32\\hypo_vitaldb && "C:\\Program Files\\Git\\cmd\\git.exe" add -A')
    os.system('cd C:\\Users\\sck32\\hypo_vitaldb && "C:\\Program Files\\Git\\cmd\\git.exe" commit -m "feat: complete improved dataset rebuild with 3-condition OR logic"')
    os.system('cd C:\\Users\\sck32\\hypo_vitaldb && "C:\\Program Files\\Git\\cmd\\git.exe" push -f origin main')

# 메인
if __name__ == "__main__":
    print("=" * 70)
    print("✨ 자동 처리 시작")
    print("=" * 70 + "\n")
    
    # 1. CSV 완성 대기
    if not wait_for_csv():
        print("❌ 시간 초과: CSV 빌드 미완료")
        sys.exit(1)
    
    # 2. 라벨 분포 확인
    has_label_1, df = check_labels()
    
    if not has_label_1:
        print("\n⚠️  Label 1 미발견! (모든 Label 0)")
        print("→ 라벨 로직 재검토 필요")
        sys.exit(1)
    
    # 3. 혼합 라벨 발견 시 전체 빌드
    if len(df) < 100:  # 500개 미만이면 전체 빌드
        build_full_dataset()
        wait_for_csv(timeout_sec=14400)  # 4시간 대기
        has_label_1, df = check_labels()
    
    if not has_label_1:
        print("\n❌ 전체 빌드에서도 Label 1 미발견")
        sys.exit(1)
    
    # 4. 모델 재학습
    train_model()
    
    # 5. GitHub 강제 푸시
    commit_and_push()
    
    print("\n" + "=" * 70)
    print("✅ 모든 작업 완료!")
    print("=" * 70)
