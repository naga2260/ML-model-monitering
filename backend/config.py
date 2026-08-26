from pathlib import Path


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model directory
MODEL_DIR = BASE_DIR / "models"

# Model file
MODEL_PATH = MODEL_DIR / "student_performance_pipeline.joblib"