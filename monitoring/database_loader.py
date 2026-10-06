import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text


# ============================================================
# Add project root to Python path
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# ============================================================
# Import database engine
# ============================================================

from backend.database import engine


# ============================================================
# Load production data from PostgreSQL
# ============================================================

def load_production_data() -> pd.DataFrame:
    """
    Load inference records from PostgreSQL and convert
    them into a DataFrame suitable for monitoring.
    """

    query = text("""
        SELECT
            id,
            timestamp,
            input_features,
            prediction,
            latency_ms,
            status
        FROM inference_logs
        WHERE status = 'success'
        ORDER BY id ASC
    """)

    with engine.connect() as connection:

        result = connection.execute(query)

        rows = result.mappings().all()

    if not rows:

        print(
            "No inference records found in PostgreSQL."
        )

        return pd.DataFrame()

    # --------------------------------------------------------
    # Convert database rows to DataFrame
    # --------------------------------------------------------

    records = []

    for row in rows:

        input_features = row["input_features"]

        record = {
            **input_features,

            "prediction": row["prediction"],

            "latency_ms": row["latency_ms"],

            "timestamp": row["timestamp"],

            "id": row["id"],
        }

        records.append(record)

    df = pd.DataFrame(records)

    return df


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Loading production data from PostgreSQL")
    print("=" * 60)

    df = load_production_data()

    print(
        f"\nProduction dataset shape: {df.shape}"
    )

    if not df.empty:

        print("\nColumns:")
        print(df.columns.tolist())

        print("\nLatest records:")
        print(df.tail())

