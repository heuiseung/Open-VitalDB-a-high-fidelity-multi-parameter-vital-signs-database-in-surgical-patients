"""수술 중 저혈압 조기 예측 - 설정 (과금 방지: 최대 시간/스텝 도달 시 자동 중단·저장)"""
from __future__ import annotations

from pathlib import Path

# 프로젝트 폴더 (hypo_vitaldb)
BASE_DIR = Path(__file__).resolve().parent

# VitalDB 데이터 폴더 (쉼표 있는 경로)
VITALDB_ROOT = Path(
    r"C:\Users\sck32\Documents\Python_Scripts\Open VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients"
)
VITAL_DIR = VITALDB_ROOT / "vital_files"
CLINICAL_CSV = VITALDB_ROOT / "clinical_data.csv"


def check_data_paths() -> tuple[bool, str]:
    """VitalDB 데이터 경로 존재 여부 확인. (성공 여부, 메시지) 반환."""
    if not CLINICAL_CSV.exists():
        return False, f"clinical_data.csv 없음: {CLINICAL_CSV}"
    if not VITAL_DIR.exists():
        return False, f"vital_files 폴더 없음: {VITAL_DIR}"
    return True, "데이터 경로 확인됨"

MAP_THRESHOLD_MMHG = 75
HYPOTENSION_DURATION_SEC = 10
PREDICTION_HORIZON_MIN = 5
LOOKBACK_MIN = 5
SAMPLE_INTERVAL_SEC = 1.0

TRACK_MAP = "Solar8000/ART_MBP"
TRACK_SBP = "Solar8000/ART_SBP"
TRACK_DBP = "Solar8000/ART_DBP"
TRACK_HR = "Solar8000/HR"
TRACKS_VITAL = [TRACK_MAP, TRACK_SBP, TRACK_DBP, TRACK_HR]

TEST_SIZE = 0.2
VAL_RATIO = 0.15  # 학습 데이터 중 검증 비율 (케이스 단위)
N_EPOCHS = 10     # 다중 에폭
RANDOM_STATE = 42
DEVICE = "cuda"
BATCH_SIZE = 512  # GPU 최대 활용 (데이터 전부 GPU 상주, OOM 시 256으로 조정)

MAX_RUNTIME_MINUTES = None  # 시간 제한 없음
MAX_TRAIN_STEPS = None      # 스텝 제한 없음 (에폭당이 아닌 전체)

DATASET_PATH = BASE_DIR / "hypotension_dataset.csv"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
MODEL_PATH = CHECKPOINT_DIR / "hypo_model.pt"
TRAIN_STATE_PATH = CHECKPOINT_DIR / "train_state.pt"
