from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw_data"
PROCESSED_DIR = DATA_DIR / "processed"
INTERP_DIR = DATA_DIR / "interpolated"
CLEAN_DIR = DATA_DIR / "clean"
MODEL_DIR = DATA_DIR / "models"

WEATHER_PARQUET = CLEAN_DIR / "valais_clean.parquet"
STATIONS_CSV = DATA_DIR / "valais_stations.csv"