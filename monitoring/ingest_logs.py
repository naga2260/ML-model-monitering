import json
from pathlib import Path

import pandas as pd


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

LOG_FILE = BASE_DIR / "logs" / "inference.log"
OUTPUT_FILE = BASE_DIR / "data" / "production_data.csv"


def load_inference_logs(log_file: Path) -> list[dict]:
    """
    Read inference.log and convert each JSON log entry
    into a Python dictionary.
    """

    records = []

    if not log_file.exists():
        raise FileNotFoundError(
            f"Log file not found: {log_file}"
        )

    with log_file.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                # Log format:
                # timestamp | level | logger | JSON payload

                json_start = line.find("{")

                if json_start == -1:
                    continue

                json_data = line[json_start:]

                payload = json.loads(json_data)

                # Only collect successful predictions
                if payload.get("event") == "prediction":
                    records.append(payload)

            except json.JSONDecodeError:
                print(
                    f"Skipping invalid JSON at line {line_number}"
                )

    return records


def create_production_dataframe(
    records: list[dict],
) -> pd.DataFrame:
    """
    Convert inference records into a production DataFrame.
    """

    rows = []

    for record in records:

        input_features = record.get(
            "input_features",
            {}
        )

        row = {
            **input_features,
            "prediction": record.get("prediction"),
            "latency_ms": record.get("latency_ms"),
            "timestamp": record.get("timestamp"),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def save_production_data(
    df: pd.DataFrame,
    output_file: Path,
) -> None:
    """
    Save production data to CSV.
    """

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_file,
        index=False
    )


def main():

    print("Reading inference logs...")

    records = load_inference_logs(LOG_FILE)

    print(f"Found {len(records)} prediction records.")

    if not records:
        print("No prediction records found.")
        return

    df_production = create_production_dataframe(
        records
    )

    save_production_data(
        df_production,
        OUTPUT_FILE
    )

    print(
        f"Production data saved to: {OUTPUT_FILE}"
    )

    print(
        f"Production dataset shape: "
        f"{df_production.shape}"
    )

    print("\nProduction Data:")
    print(df_production.head())


if __name__ == "__main__":
    main()