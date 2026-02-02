from pathlib import Path

ROOT_DIR = Path(__file__).parent.absolute()
DATA_DIRECTORY_PATH = ROOT_DIR / "data"
RAW_DATA_DIRECTORY_PATH = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIRECTORY_PATH = ROOT_DIR / "data" / "processed"
METRICS_DIRECTORY_PATH = ROOT_DIR / "metrics"
MODELS_DIRECTORY_PATH = ROOT_DIR / "models"
REPORTS_DIRECTORY_PATH = ROOT_DIR / "reports"
FEATURES_DIRECTORY_PATH = ROOT_DIR / "features"
DASHBOARD_DIRECTORY_PATH = ROOT_DIR / "dashboard"
INTERACTIVE_REPORTS_PATH = ROOT_DIR / "interactive_reports"