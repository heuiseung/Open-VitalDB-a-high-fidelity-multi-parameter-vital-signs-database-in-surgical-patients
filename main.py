"""
VitalDB 저혈압 조기 예측 파이프라인 — 실행 진입점.

전체 워크플로: (1) 데이터 경로 검증 (2) 데이터셋 미존재 시 구축 (3) GPU 학습 실행.
대학원 제출/재현용 단일 진입점으로, 사용자는 `python main.py` 한 번만 실행하면 됨.

Usage:
    python main.py

Note:
    데이터셋이 없으면 자동으로 build_dataset을 호출한 뒤, train 모듈로 학습합니다.
"""
from utils import set_utf8_stdout
from config import DATASET_PATH, check_data_paths

set_utf8_stdout()


def main() -> None:
    """파이프라인 진입점: 경로 검증 → 데이터셋 구축(필요 시) → 학습 실행.

    Returns:
        None. 콘솔에 진행 메시지를 출력하고, 내부에서 build_dataset.main(),
        train.main()을 순차 호출합니다.

    Raises:
        FileNotFoundError: config에서 지정한 VitalDB/clinical 경로가 없을 때
            (check_data_paths에서 검증).
    """
    print("=" * 60)
    print("VitalDB-Hypotension-Prediction")
    print("수술 중 저혈압 조기 예측 (CUDA)")
    print("=" * 60)

    # 경로 검증 실패 시 조기 종료 (VitalDB 데이터 누락 등)
    ok, msg = check_data_paths()
    if not ok:
        print(f"[오류] {msg}")
        return

    # 데이터셋 CSV 없으면 먼저 구축 (Vital → CSV 변환)
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
