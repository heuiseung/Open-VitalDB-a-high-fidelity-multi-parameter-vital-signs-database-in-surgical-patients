"""
PyTorch CUDA 학습 — 기존 실행 호환용 래퍼.
실제 학습 로직은 train.py에 있으며, 여기서는 train.main()을 호출합니다.
"""
from utils import set_utf8_stdout
from train import main as train_main


def main() -> None:
    set_utf8_stdout()
    train_main()


if __name__ == "__main__":
    main()
