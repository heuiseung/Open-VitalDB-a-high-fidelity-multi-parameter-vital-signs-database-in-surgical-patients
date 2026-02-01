"""전체 파이프라인 (진행상황 한글 표시, 과금 방지)"""
from pathlib import Path
from config import DATASET_PATH, check_data_paths


def main() -> None:
    print("=" * 60)
    print("수술 중 저혈압 조기 예측 (CUDA)")
    print("진행상황 한글 표시, 과금 방지(시간/스텝 제한 시 자동 저장 후 중단)")
    print("=" * 60)
    ok, msg = check_data_paths()
    if not ok:
        print(f"[오류] {msg}")
        return
    if not DATASET_PATH.exists():
        print("\n[진행 1/2] 데이터셋 구축 중... (Vital 파일 → 특성·라벨 추출)")
        import build_dataset
        build_dataset.main()
    else:
        print("\n[진행 1/2] 데이터셋 이미 있음 → 구축 생략")
    print("\n[진행 2/2] 모델 학습 중... (PyTorch CUDA)")
    import train_model
    train_model.main()
    print("\n[완료] 전체 파이프라인 실행이 끝났습니다.")


if __name__ == "__main__":
    main()
