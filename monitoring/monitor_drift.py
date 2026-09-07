
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


# ==================================================
# Project paths
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# --------------------------------------------------
# Data paths
# --------------------------------------------------

REFERENCE_FILE = (
    BASE_DIR / "data" / "reference_data.csv"
)

PRODUCTION_FILE = (
    BASE_DIR / "data" / "production_data.csv"
)

REFERENCE_PREDICTIONS_FILE = (
    BASE_DIR / "data" / "reference_predictions.csv"
)


# --------------------------------------------------
# Report paths
# --------------------------------------------------

REPORT_DIR = BASE_DIR / "reports"

DATA_DRIFT_REPORT_FILE = (
    REPORT_DIR / "data_drift_report.html"
)

PRODUCTION_DRIFT_REPORT_FILE = (
    REPORT_DIR / "production_drift_report.html"
)


# ==================================================
# Load data
# ==================================================

def load_data():
    """
    Load reference and production datasets.
    """

    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(
            f"Reference data not found:\n{REFERENCE_FILE}"
        )

    if not PRODUCTION_FILE.exists():
        raise FileNotFoundError(
            f"Production data not found:\n{PRODUCTION_FILE}"
        )

    reference_df = pd.read_csv(
        REFERENCE_FILE
    )

    production_df = pd.read_csv(
        PRODUCTION_FILE
    )

    return reference_df, production_df


# ==================================================
# Prepare data for DATA DRIFT
# ==================================================

def prepare_data(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame
):
    """
    Prepare datasets for input feature drift.

    prediction, latency_ms and timestamp are
    monitoring/output columns and are excluded
    from input feature drift.
    """

    monitoring_columns = [
        "prediction",
        "latency_ms",
        "timestamp",
    ]

    # ------------------------------------------------
    # Remove monitoring columns from production data
    # ------------------------------------------------

    columns_to_remove = [
        column
        for column in monitoring_columns
        if column in production_df.columns
    ]

    production_df = production_df.drop(
        columns=columns_to_remove
    )

    # ------------------------------------------------
    # Find common feature columns
    # ------------------------------------------------

    common_columns = [
        column
        for column in reference_df.columns
        if column in production_df.columns
    ]

    if not common_columns:
        raise ValueError(
            "No common feature columns found between "
            "reference and production datasets."
        )

    reference_df = reference_df[
        common_columns
    ]

    production_df = production_df[
        common_columns
    ]

    return reference_df, production_df


# ==================================================
# Generate DATA DRIFT report
# ==================================================

def generate_data_drift_report(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame
):
    """
    Generate Evidently data drift report.

    Compares input features between:
    
        reference_data
              VS
        production_data
    """

    print("\n------------------------------------------")
    print("Generating DATA DRIFT report")
    print("------------------------------------------")

    report = Report(
        [
            DataDriftPreset()
        ]
    )

    result = report.run(
        reference_data=reference_df,
        current_data=production_df
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    result.save_html(
        str(DATA_DRIFT_REPORT_FILE)
    )

    if not DATA_DRIFT_REPORT_FILE.exists():
        raise RuntimeError(
            "Data drift report was not created."
        )

    print(
        f"Data drift report created:\n"
        f"{DATA_DRIFT_REPORT_FILE}"
    )


# ==================================================
# Generate PRODUCTION / PREDICTION DRIFT report
# ==================================================

def generate_production_drift_report():
    """
    Generate Evidently prediction drift report.

    Compares:

        reference_predictions.csv
                    VS
        production_data.csv -> prediction

    The reference prediction file currently contains
    the prediction column under the name 'Exam_Score'.

    We rename it internally to 'prediction' so that
    both datasets have the same column name.
    """

    print("\n------------------------------------------")
    print("Generating PRODUCTION DRIFT report")
    print("------------------------------------------")

    # ------------------------------------------------
    # Check reference prediction file
    # ------------------------------------------------

    if not REFERENCE_PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            "Reference predictions file not found:\n"
            f"{REFERENCE_PREDICTIONS_FILE}"
        )

    # ------------------------------------------------
    # Load files
    # ------------------------------------------------

    reference_predictions = pd.read_csv(
        REFERENCE_PREDICTIONS_FILE
    )

    production_df = pd.read_csv(
        PRODUCTION_FILE
    )

    # ------------------------------------------------
    # Check reference column
    # ------------------------------------------------

    if "Exam_Score" not in reference_predictions.columns:
        raise ValueError(
            "reference_predictions.csv must contain "
            "an 'Exam_Score' column."
        )

    # ------------------------------------------------
    # Check production column
    # ------------------------------------------------

    if "prediction" not in production_df.columns:
        raise ValueError(
            "production_data.csv must contain "
            "a 'prediction' column."
        )

    # ------------------------------------------------
    # Rename reference prediction column
    # ------------------------------------------------

    reference_predictions = (
        reference_predictions
        .rename(
            columns={
                "Exam_Score": "prediction"
            }
        )
    )

    # ------------------------------------------------
    # Select prediction only
    # ------------------------------------------------

    reference_predictions = (
        reference_predictions[
            ["prediction"]
        ]
    )

    production_predictions = (
        production_df[
            ["prediction"]
        ]
    )

    # ------------------------------------------------
    # Check for empty datasets
    # ------------------------------------------------

    if reference_predictions.empty:
        raise ValueError(
            "Reference predictions dataset is empty."
        )

    if production_predictions.empty:
        raise ValueError(
            "Production predictions dataset is empty."
        )

    # ------------------------------------------------
    # Display information
    # ------------------------------------------------

    print(
        f"Reference predictions: "
        f"{reference_predictions.shape}"
    )

    print(
        f"Production predictions: "
        f"{production_predictions.shape}"
    )

    print("\nReference prediction statistics:")

    print(
        reference_predictions[
            "prediction"
        ].describe()
    )

    print("\nProduction prediction statistics:")

    print(
        production_predictions[
            "prediction"
        ].describe()
    )

    # ------------------------------------------------
    # Generate Evidently report
    # ------------------------------------------------

    report = Report(
        [
            DataDriftPreset()
        ]
    )

    result = report.run(
        reference_data=reference_predictions,
        current_data=production_predictions
    )

    # ------------------------------------------------
    # Save report
    # ------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    result.save_html(
        str(PRODUCTION_DRIFT_REPORT_FILE)
    )

    # ------------------------------------------------
    # Verify report
    # ------------------------------------------------

    if not PRODUCTION_DRIFT_REPORT_FILE.exists():
        raise RuntimeError(
            "Production drift report was not created."
        )

    print(
        f"\nProduction drift report created:\n"
        f"{PRODUCTION_DRIFT_REPORT_FILE}"
    )


# ==================================================
# Main monitoring pipeline
# ==================================================

def main():

    print("\n==========================================")
    print("          ML MONITORING PIPELINE")
    print("==========================================")

    # ==================================================
    # 1. Load datasets
    # ==================================================

    print("\nLoading datasets...")

    reference_df, production_df = load_data()

    print(
        f"Reference dataset: "
        f"{reference_df.shape}"
    )

    print(
        f"Production dataset: "
        f"{production_df.shape}"
    )

    # ==================================================
    # 2. DATA DRIFT
    # ==================================================

    reference_features, production_features = (
        prepare_data(
            reference_df,
            production_df
        )
    )

    print(
        f"\nComparing "
        f"{len(reference_features.columns)} "
        f"input features..."
    )

    generate_data_drift_report(
        reference_features,
        production_features
    )

    # ==================================================
    # 3. PRODUCTION / PREDICTION DRIFT
    # ==================================================

    generate_production_drift_report()

    # ==================================================
    # 4. Complete
    # ==================================================

    print("\n==========================================")
    print("       MONITORING COMPLETED")
    print("==========================================")

    print("\nReports generated:")

    print(
        f"\n1. Data drift:"
        f"\n   {DATA_DRIFT_REPORT_FILE}"
    )

    print(
        f"\n2. Production drift:"
        f"\n   {PRODUCTION_DRIFT_REPORT_FILE}"
    )


# ==================================================
# Entry point
# ==================================================

if __name__ == "__main__":
    main()

