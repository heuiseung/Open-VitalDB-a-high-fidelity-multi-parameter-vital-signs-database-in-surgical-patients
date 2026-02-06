"""지속 실행: 설정 검증 → 파이프라인 → GitHub 동기화 (보고 없이 자동 진행)"""
import sys
import io
import subprocess
from pathlib import Path

# 한글 출력
if getattr(sys.stdout, "buffer", None) and (sys.stdout.encoding or "").lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str], cwd: Path = ROOT) -> bool:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.stdout:
        print(r.stdout, end="")
    if r.stderr:
        print(r.stderr, end="", file=sys.stderr)
    return r.returncode == 0


def main() -> None:
    print("[자동] 설정 검증 → 파이프라인 → GitHub 동기화")
    print("=" * 50)
    # 1) 설정 검증
    r = subprocess.run([sys.executable, "check_setup.py"], cwd=ROOT)
    if r.returncode != 0:
        print("[중단] 설정 검증 실패")
        return
    # 2) 파이프라인
    r = subprocess.run([sys.executable, "run_all.py"], cwd=ROOT)
    if r.returncode != 0:
        print("[중단] 파이프라인 실패")
        return
    # 3) GitHub (소스만, checkpoints/ 데이터 제외)
    run(["git", "add", "run_all.py", "train_model.py", "build_dataset.py", "config.py", "data_loader.py"])
    run(["git", "add", "run_continuous.py", "README.md", "QUICKSTART.md", ".github/SYNC.md"])
    run(["git", "status", "--short"])
    run(["git", "commit", "-m", "자동: 파이프라인 실행 후 동기화"])
    if run(["git", "push", "origin", "main"]):
        print("[자동] GitHub 푸시 완료")
    print("=" * 50)
    print("[자동] 1회 사이클 완료. 다시 실행하려면 run_continuous.py 또는 run_continuous.bat 실행.")


if __name__ == "__main__":
    main()
