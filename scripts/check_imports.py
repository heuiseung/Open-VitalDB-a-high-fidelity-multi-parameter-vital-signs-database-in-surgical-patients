#!/usr/bin/env python3
"""
scripts/ 또는 루트에서 실행해도 루트 모듈(config, data_loader, model 등)이 import 되는지 확인.
사용법: python scripts/check_imports.py  또는  cd scripts && python check_imports.py
"""
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

def main():
    errors = []
    try:
        import config
        print("[OK] import config")
    except Exception as e:
        errors.append(f"config: {e}")
        print(f"[FAIL] import config: {e}")

    try:
        import data_loader
        print("[OK] import data_loader")
    except Exception as e:
        errors.append(f"data_loader: {e}")
        print(f"[FAIL] import data_loader: {e}")

    try:
        import model
        print("[OK] import model")
    except Exception as e:
        errors.append(f"model: {e}")
        print(f"[FAIL] import model: {e}")

    try:
        import utils
        print("[OK] import utils")
    except Exception as e:
        errors.append(f"utils: {e}")
        print(f"[FAIL] import utils: {e}")

    if errors:
        print("\n[결과] 일부 모듈 import 실패")
        sys.exit(1)
    print("\n[결과] 모든 루트 모듈 import 성공 (scripts/ 경로 설정 정상)")


if __name__ == "__main__":
    main()
