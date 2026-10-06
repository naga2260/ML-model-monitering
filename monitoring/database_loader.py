import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import text

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.database import engine


def load_production_data(
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> pd.DataFrame:
    """
    Load successful production inference records
    from PostgreSQL.

    If start_time and end_time are provided,
    only records inside that monitoring window
    are returned.
    """

    query = text(
        """
        SELECT
            id,
            timestamp,
            input_features,
            prediction,
            latency_ms,
            status
        FROM inference_logs
        WHERE status = 'success'
          AND (
                :start_time IS NULL
                OR timestamp >= :start_time
              )
          AND (
                :end_time IS NULL
                OR timestamp < :end_time
              )
        ORDER BY id ASC
        """
    )

    with engine.connect() as connection:

        result = connection.execute(
            query,
            {
                "start_time": start_time,
                "end_time": end_time,
            },
        )

        rows = result.mappings().all()

    if not rows:
        print("No inference records found for the monitoring window.")
        return pd.DataFrame()

    records = []

    for row in rows:

        input_features = row["input_features"]

        record = {
            **input_features,
            "prediction": row["prediction"],
            "latency_ms": row["latency_ms"],
            "timestamp": row["timestamp"],
            "id": row["id"],
            "status": row["status"],
        }

        records.append(record)

    return pd.DataFrame(records)


if __name__ == "__main__":

    print("=" * 60)
    print("Loading production data from PostgreSQL")
    print("=" * 60)

    df = load_production_data()

    print(f"\nProduction dataset shape: {df.shape}")

    if not df.empty:

        print("\nColumns:")
        print(df.columns.tolist())

        print("\nLatest records:")
        print(df.tail())

