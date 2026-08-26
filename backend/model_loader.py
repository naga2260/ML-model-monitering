import joblib
from pathlib import Path


def load_model(model_path: Path):
    """
    Load the trained ML pipeline from disk.
    """

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    model = joblib.load(model_path)

    return model