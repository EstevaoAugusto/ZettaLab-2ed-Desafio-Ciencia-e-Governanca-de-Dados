from pathlib import Path

ROOT_DIR = Path(__file__).parent.absolute()
DATA_DIRECTORY_PATH = ROOT_DIR / "data"
RAW_DATA_DIRECTORY_PATH = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIRECTORY_PATH = ROOT_DIR / "data" / "processed"
METRICS_DIRECTORY_PATH = ROOT_DIR / "metrics"
MODEL_DIRECTORY_PATH = ROOT_DIR / "model"
REPORTS_DIRECTORY_PATH = ROOT_DIR / "reports"