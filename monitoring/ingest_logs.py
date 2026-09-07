import json
import logging
from pathlib import Path

import pandas as pd


# ============================================================
# Project paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

LOG_FILE = BASE_DIR / "logs" / "inference.log"

OUTPUT_FILE = (
    BASE_DIR / "data" / "production_data.csv"
)

STATE_FILE = (
    BASE_DIR / "monitoring" / "ingest_state.json"
)


# ============================================================
# Logging
# ============================================================

LOG_DIR = BASE_DIR / "logs"

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / "ingest.log",
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# ============================================================
# State management
# ============================================================

def load_last_position() -> int:
    """
    Load the last processed byte position from the
    state file.

    If no state file exists, start from the beginning
    of the log.
    """

    if not STATE_FILE.exists():

        logger.info(
            "No ingestion state found. "
            "Starting from beginning of log."
        )

        return 0

    try:

        with STATE_FILE.open(
            "r",
            encoding="utf-8"
        ) as file:

            state = json.load(file)

        position = state.get(
            "last_position",
            0
        )

        return int(position)

    except (
        json.JSONDecodeError,
        ValueError,
        TypeError
    ):

        logger.warning(
            "Invalid state file. "
            "Starting from beginning of log."
        )

        return 0


def save_last_position(
    position: int
) -> None:
    """
    Save the current log position.
    """

    STATE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    state = {
        "last_position": position
    }

    with STATE_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            state,
            file,
            indent=4
        )


# ============================================================
# Read NEW inference logs
# ============================================================

def load_new_inference_logs(
    log_file: Path,
    start_position: int
) -> tuple[list[dict], int]:
    """
    Read only the new portion of inference.log.

    Returns:

        records
        new file position
    """

    records = []

    if not log_file.exists():

        raise FileNotFoundError(
            f"Log file not found: {log_file}"
        )

    # --------------------------------------------------------
    # Check whether the log was reset/truncated.
    # --------------------------------------------------------

    file_size = log_file.stat().st_size

    if start_position > file_size:

        logger.warning(
            "Log file is smaller than the saved "
            "position. The log may have been reset "
            "or rotated."
        )

        start_position = 0

    # --------------------------------------------------------
    # Open log and jump to last processed position.
    # --------------------------------------------------------

    with log_file.open(
        "r",
        encoding="utf-8"
    ) as file:

        file.seek(start_position)

        for line_number, line in enumerate(
            file,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            try:

                # ------------------------------------------------
                # Log format:
                #
                # timestamp | level | logger | JSON payload
                # ------------------------------------------------

                json_start = line.find("{")

                if json_start == -1:
                    continue

                json_data = line[json_start:]

                payload = json.loads(
                    json_data
                )

                # ------------------------------------------------
                # Only successful prediction events
                # ------------------------------------------------

                if payload.get("event") == "prediction":

                    records.append(
                        payload
                    )

            except json.JSONDecodeError:

                logger.warning(
                    "Skipping invalid JSON line."
                )

        # --------------------------------------------------------
        # Save new position after reading.
        # --------------------------------------------------------

        new_position = file.tell()

    return records, new_position


# ============================================================
# Convert records to DataFrame
# ============================================================

def create_production_dataframe(
    records: list[dict]
) -> pd.DataFrame:
    """
    Convert inference records into a production
    DataFrame.
    """

    rows = []

    for record in records:

        input_features = record.get(
            "input_features",
            {}
        )

        row = {
            **input_features,
            "prediction": record.get(
                "prediction"
            ),
            "latency_ms": record.get(
                "latency_ms"
            ),
            "timestamp": record.get(
                "timestamp"
            ),
        }

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Append production data
# ============================================================

def append_production_data(
    df: pd.DataFrame,
    output_file: Path
) -> None:
    """
    Append new production records to the existing
    production_data.csv.

    If the file doesn't exist, create it with headers.
    """

    if df.empty:

        logger.info(
            "No new production records to append."
        )

        return

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    file_exists = output_file.exists()

    df.to_csv(
        output_file,
        mode="a",
        header=not file_exists,
        index=False
    )

    logger.info(
        "Appended %d new records to %s",
        len(df),
        output_file
    )


# ============================================================
# Main
# ============================================================

def main():

    logger.info("=" * 60)
    logger.info("Starting incremental log ingestion")
    logger.info("=" * 60)

    # --------------------------------------------------------
    # 1. Load previous position
    # --------------------------------------------------------

    last_position = load_last_position()

    logger.info(
        "Last processed log position: %d",
        last_position
    )

    # --------------------------------------------------------
    # 2. Read only NEW records
    # --------------------------------------------------------

    records, new_position = (
        load_new_inference_logs(
            LOG_FILE,
            last_position
        )
    )

    logger.info(
        "New prediction records found: %d",
        len(records)
    )

    # --------------------------------------------------------
    # 3. Convert to DataFrame
    # --------------------------------------------------------

    if records:

        df_production = (
            create_production_dataframe(
                records
            )
        )

        logger.info(
            "New production data shape: %s",
            df_production.shape
        )

        # ----------------------------------------------------
        # 4. Append to production dataset
        # ----------------------------------------------------

        append_production_data(
            df_production,
            OUTPUT_FILE
        )

    else:

        logger.info(
            "No new prediction records found."
        )

    # --------------------------------------------------------
    # 5. Update state
    # --------------------------------------------------------

    save_last_position(
        new_position
    )

    logger.info(
        "Updated log position: %d",
        new_position
    )

    logger.info(
        "Incremental ingestion completed successfully."
    )

    logger.info("=" * 60)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()