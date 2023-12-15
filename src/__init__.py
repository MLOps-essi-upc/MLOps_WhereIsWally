"""
Src init
"""
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(Path(__file__).resolve().parent.parent)

RAW_DATA_DIR = ROOT_DIR / "data/raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data/processed"

METRICS_DIR = ROOT_DIR / "metrics"
MODELS_DIR = ROOT_DIR / "models/best"
REPORTS_DIR = ROOT_DIR / "reports"
DATA_YAML_DIR = ROOT_DIR / "data/processed/data.yaml"
ARTIFACTS_DIR = ROOT_DIR / "runs/detect"
API_DIR = ROOT_DIR / "app"
DRIFT_DETECTOR_DIR= ROOT_DIR / "models/drift_detector"