"""
VitalDB 저혈압 조기 예측 — 실행 진입점.

사용법:
  python main.py

데이터셋이 없으면 자동으로 구축한 뒤, GPU(CUDA)로 학습합니다.
"""
from utils import set_utf8_stdout
from config import DATASET_PATH, check_data_paths

set_utf8_stdout()


def main() -> None:
    print("=" * 60)
    print("VitalDB-Hypotension-Prediction")
    print("수술 중 저혈압 조기 예측 (CUDA)")
    print("=" * 60)
    ok, msg = check_data_paths()
    if not ok:
        print(f"[오류] {msg}")
        return
    if not DATASET_PATH.exists():
        print("\n[1/2] 데이터셋 구축 중...")
        import build_dataset
        build_dataset.main()
    else:
        print("\n[1/2] 데이터셋 있음 → 구축 생략")
    print("\n[2/2] 모델 학습 중... (PyTorch CUDA)")
    import train
    train.main()
    print("\n[완료] 파이프라인 실행이 끝났습니다.")


if __name__ == "__main__":
    main()
