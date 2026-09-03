import logging
from pathlib import Path


def setup_logging():
    """Configure application logging."""

    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "inference.log"),
            logging.StreamHandler()  # Also show logs in terminal
        ]
    )