from datetime import datetime, timedelta
from pathlib import Path
import sys

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from database_loader import load_production_data


# ==================================================
# Project paths
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# --------------------------------------------------
# Backend and Alerting Imports
# --------------------------------------------------

from backend.database import SessionLocal
from backend.models import MonitoringMetric
from alerting import (
    build_alert_message,
    build_recovery_message,
    send_telegram_alert,
)


# --------------------------------------------------
# Data paths
# --------------------------------------------------

REFERENCE_FILE = (
    BASE_DIR / "data" / "reference_data.csv"
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

def load_data(
    start_time: datetime,
    end_time: datetime,
):
    """
    Load reference data from CSV and production
    inference data from PostgreSQL for a specific time window.

    PostgreSQL is the single source of production
    data for both monitoring reports.
    """

    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(
            f"Reference data not found:\n{REFERENCE_FILE}"
        )

    reference_df = pd.read_csv(
        REFERENCE_FILE
    )

    # Production data comes from PostgreSQL within the window
    production_df = load_production_data(
        start_time=start_time,
        end_time=end_time,
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

    prediction, latency_ms, timestamp, id and status
    are monitoring/output columns and are excluded
    from input feature drift.
    """

    monitoring_columns = [
        "id",
        "prediction",
        "latency_ms",
        "timestamp",
        "status",
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
    Generate Evidently data drift report and return
    drift metrics.
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

    # ------------------------------------------------
    # Extract drift metrics from Evidently
    # ------------------------------------------------

    result_dict = result.dict()

    metrics = result_dict.get("metrics", [])

    drifted_features = 0
    total_features = len(reference_df.columns)
    data_drift_detected = False

    for metric in metrics:

        metric_id = metric.get("metric_id", "")

        if "DriftedColumnsCount" in metric_id:

            value = metric.get("value", {})

            if isinstance(value, dict):

                drifted_features = int(
                    value.get("count", 0)
                )

                data_drift_detected = (
                    drifted_features > 0
                )

            break

    print("\nData Drift Metrics:")
    print(
        f"Drifted features : "
        f"{drifted_features}"
    )
    print(
        f"Total features   : "
        f"{total_features}"
    )
    print(
        f"Data drift detected: "
        f"{data_drift_detected}"
    )

    return {
        "data_drift_detected": data_drift_detected,
        "drifted_features": drifted_features,
        "total_features": total_features,
    }


# ==================================================
# Generate PRODUCTION / PREDICTION DRIFT report
# ==================================================

def generate_production_drift_report(
    production_df: pd.DataFrame
):
    """
    Generate Evidently prediction drift report.

    Compares:

        reference_predictions.csv
                    VS
        PostgreSQL production predictions
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
    # Load reference predictions
    # ------------------------------------------------

    reference_predictions = pd.read_csv(
        REFERENCE_PREDICTIONS_FILE
    )

    # ------------------------------------------------
    # Check reference prediction column
    # ------------------------------------------------

    if "prediction" in reference_predictions.columns:

        # Preferred format
        reference_predictions = (
            reference_predictions[
                ["prediction"]
            ]
        )

    elif "Exam_Score" in reference_predictions.columns:

        # Temporary compatibility with your
        # current reference_predictions.csv
        reference_predictions = (
            reference_predictions[
                ["Exam_Score"]
            ]
            .rename(
                columns={
                    "Exam_Score": "prediction"
                }
            )
        )

    else:

        raise ValueError(
            "reference_predictions.csv must contain "
            "either 'prediction' or 'Exam_Score'."
        )

    # ------------------------------------------------
    # Check PostgreSQL production prediction column
    # ------------------------------------------------

    if "prediction" not in production_df.columns:
        raise ValueError(
            "PostgreSQL production data must contain "
            "a 'prediction' column."
        )

    # ------------------------------------------------
    # Get production predictions from PostgreSQL
    # ------------------------------------------------

    production_predictions = (
        production_df[
            ["prediction"]
        ]
        .copy()
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
            "Production predictions dataset from "
            "PostgreSQL is empty."
        )

    # ------------------------------------------------
    # Display information
    # ------------------------------------------------

    print(
        f"Reference predictions: "
        f"{reference_predictions.shape}"
    )

    print(
        f"Production predictions from PostgreSQL: "
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

    # ------------------------------------------------
    # Extract prediction drift result
    # ------------------------------------------------

    result_dict = result.dict()

    metrics = result_dict.get("metrics", [])

    prediction_drift_detected = False

    for metric in metrics:

        metric_id = metric.get("metric_id", "")

        if "DriftedColumnsCount" in metric_id:

            value = metric.get("value", {})

            if isinstance(value, dict):

                drifted_columns = int(
                    value.get("count", 0)
                )

                prediction_drift_detected = (
                    drifted_columns > 0
                )

            break

    print("\nPrediction Drift Metrics:")
    print(
        f"Prediction drift detected: "
        f"{prediction_drift_detected}"
    )

    return {
        "prediction_drift_detected":
            prediction_drift_detected
    }


# ==================================================
# Calculate Monitoring Metrics
# ==================================================

def calculate_monitoring_metrics(production_df):
    """
    Calculate operational metrics from PostgreSQL production data.
    """

    if production_df.empty:
        raise ValueError("Production dataset is empty.")

    request_count = len(production_df)

    avg_latency_ms = float(
        production_df["latency_ms"].mean()
    )

    p95_latency_ms = float(
        production_df["latency_ms"].quantile(0.95)
    )

    print("\n------------------------------------------")
    print("Production Metrics")
    print("------------------------------------------")
    print(f"Request count : {request_count}")
    print(f"Average latency: {avg_latency_ms:.2f} ms")
    print(f"P95 latency   : {p95_latency_ms:.2f} ms")

    return {
        "request_count": request_count,
        "avg_latency_ms": avg_latency_ms,
        "p95_latency_ms": p95_latency_ms,
    }


# ==================================================
# Evaluate Monitoring Status
# ==================================================

def evaluate_monitoring_status(
    monitoring_metrics,
    data_drift_metrics,
    prediction_drift_metrics
):
    """
    Evaluate monitoring metrics and determine
    NORMAL, WARNING, or ALERT status.
    """

    warnings = []
    alerts = []

    avg_latency = monitoring_metrics[
        "avg_latency_ms"
    ]

    p95_latency = monitoring_metrics[
        "p95_latency_ms"
    ]

    # ------------------------------------------------
    # Average latency
    # ------------------------------------------------

    if avg_latency > 500:
        alerts.append(
            f"Average latency is {avg_latency:.2f} ms"
        )

    elif avg_latency >= 200:
        warnings.append(
            f"Average latency is {avg_latency:.2f} ms"
        )

    # ------------------------------------------------
    # P95 latency
    # ------------------------------------------------

    if p95_latency > 500:
        alerts.append(
            f"P95 latency is {p95_latency:.2f} ms"
        )

    elif p95_latency >= 300:
        warnings.append(
            f"P95 latency is {p95_latency:.2f} ms"
        )

    # ------------------------------------------------
    # Data drift
    # ------------------------------------------------

    if data_drift_metrics[
        "data_drift_detected"
    ]:

        alerts.append(
            "Data drift detected in "
            f"{data_drift_metrics['drifted_features']} "
            "features"
        )

    # ------------------------------------------------
    # Prediction drift
    # ------------------------------------------------

    if prediction_drift_metrics[
        "prediction_drift_detected"
    ]:

        alerts.append(
            "Prediction drift detected"
        )

    # ------------------------------------------------
    # Determine overall status
    # ------------------------------------------------

    if alerts:
        status = "ALERT"

    elif warnings:
        status = "WARNING"

    else:
        status = "NORMAL"

    print("\n------------------------------------------")
    print("Monitoring Status")
    print("------------------------------------------")
    print(f"Status: {status}")

    if warnings:
        print("\nWarnings:")

        for warning in warnings:
            print(f"  ⚠️ {warning}")

    if alerts:
        print("\nALERTS:")

        for alert in alerts:
            print(f"  🚨 {alert}")

    return {
        "status": status,
        "warnings": warnings,
        "alerts": alerts,
    }


# ==================================================
# Alert Handling & Previous Status Functions
# ==================================================

def get_previous_monitoring_status():
    """
    Get the most recent monitoring status from PostgreSQL.
    """

    db = SessionLocal()

    try:

        previous_record = (
            db.query(MonitoringMetric)
            .order_by(
                MonitoringMetric.id.desc()
            )
            .first()
        )

        if previous_record is None:
            return None

        return previous_record.status

    finally:
        db.close()


def handle_alerts(
    monitoring_metrics,
    data_drift_metrics,
    prediction_drift_metrics,
    status_metrics,
):
    """
    Send Telegram notifications when the monitoring
    status changes or an alert condition occurs.
    """

    current_status = status_metrics["status"]

    previous_status = (
        get_previous_monitoring_status()
    )

    reasons = (
        status_metrics["alerts"]
        + status_metrics["warnings"]
    )

    # ----------------------------------------------
    # ALERT
    # ----------------------------------------------

    if current_status == "ALERT":

        # Don't repeatedly notify for the same
        # continuing alert.
        if previous_status != "ALERT":

            message = build_alert_message(
                status=current_status,
                monitoring_metrics=monitoring_metrics,
                data_drift_metrics=data_drift_metrics,
                prediction_drift_metrics=
                    prediction_drift_metrics,
                reasons=reasons,
            )

            send_telegram_alert(message)

            print(
                "\n🚨 Telegram alert sent."
            )

        else:

            print(
                "\nALERT still active. "
                "Telegram notification skipped."
            )

    # ----------------------------------------------
    # WARNING
    # ----------------------------------------------

    elif current_status == "WARNING":

        if previous_status != "WARNING":

            message = build_alert_message(
                status=current_status,
                monitoring_metrics=monitoring_metrics,
                data_drift_metrics=data_drift_metrics,
                prediction_drift_metrics=
                    prediction_drift_metrics,
                reasons=reasons,
            )

            send_telegram_alert(message)

            print(
                "\n⚠️ Telegram warning sent."
            )

    # ----------------------------------------------
    # RECOVERY
    # ----------------------------------------------

    elif current_status == "NORMAL":

        if previous_status in {
            "WARNING",
            "ALERT",
        }:

            send_telegram_alert(
                build_recovery_message()
            )

            print(
                "\n✅ Telegram recovery "
                "notification sent."
            )


# ==================================================
# Save Monitoring Metrics
# ==================================================

def save_monitoring_metrics(
    monitoring_metrics,
    data_drift_metrics,
    prediction_drift_metrics,
    status_metrics
):
    """
    Save one complete monitoring run to PostgreSQL.
    """

    db = SessionLocal()

    try:

        monitoring_record = MonitoringMetric(
            request_count=monitoring_metrics[
                "request_count"
            ],

            avg_latency_ms=monitoring_metrics[
                "avg_latency_ms"
            ],

            p95_latency_ms=monitoring_metrics[
                "p95_latency_ms"
            ],

            data_drift_detected=data_drift_metrics[
                "data_drift_detected"
            ],

            drifted_features=data_drift_metrics[
                "drifted_features"
            ],

            total_features=data_drift_metrics[
                "total_features"
            ],

            prediction_drift_detected=
                prediction_drift_metrics[
                    "prediction_drift_detected"
                ],

            status=status_metrics[
                "status"
            ],
        )

        db.add(monitoring_record)
        db.commit()
        db.refresh(monitoring_record)

        print("\n------------------------------------------")
        print("Monitoring result saved")
        print("------------------------------------------")
        print(
            f"Monitoring record ID: "
            f"{monitoring_record.id}"
        )

    except Exception:

        db.rollback()
        raise

    finally:

        db.close()


# ==================================================
# Main monitoring pipeline
# ==================================================

def main():

    print("\n==========================================")
    print("          ML MONITORING PIPELINE")
    print("==========================================")

    # ==================================================
    # 0. Define Hourly Monitoring Window
    # ==================================================

    now = datetime.now()
    end_time = now.replace(
        minute=0,
        second=0,
        microsecond=0,
    )
    start_time = end_time - timedelta(hours=1)

    print(
        f"\nMonitoring window:"
        f"\n  Start: {start_time}"
        f"\n  End:   {end_time}"
    )

    # ==================================================
    # 1. Load datasets
    # ==================================================

    print("\nLoading datasets...")

    reference_df, production_df = load_data(
        start_time,
        end_time,
    )

    if production_df.empty:
        print(
            "\nNo successful inference requests "
            "during this monitoring window."
        )
        return

    print(
        f"Reference dataset: "
        f"{reference_df.shape}"
    )

    print(
        f"Production dataset from PostgreSQL: "
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

    data_drift_metrics = (
        generate_data_drift_report(
            reference_features,
            production_features
        )
    )

    # ==================================================
    # 3. PRODUCTION / PREDICTION DRIFT
    # ==================================================

    prediction_drift_metrics = (
        generate_production_drift_report(
            production_df
        )
    )

    # ==================================================
    # 4. OPERATIONAL METRICS
    # ==================================================

    monitoring_metrics = (
        calculate_monitoring_metrics(
            production_df
        )
    )

    # ==================================================
    # 5. ALERT / STATUS EVALUATION
    # ==================================================

    status_metrics = evaluate_monitoring_status(
        monitoring_metrics,
        data_drift_metrics,
        prediction_drift_metrics
    )

    # ==================================================
    # 6. HANDLE ALERTS (TELEGRAM)
    # ==================================================

    handle_alerts(
        monitoring_metrics,
        data_drift_metrics,
        prediction_drift_metrics,
        status_metrics,
    )

    # ==================================================
    # 7. SAVE MONITORING RESULT
    # ==================================================

    save_monitoring_metrics(
        monitoring_metrics,
        data_drift_metrics,
        prediction_drift_metrics,
        status_metrics
    )

    # ==================================================
    # 8. Complete
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

    print(
        f"\nOverall status: "
        f"{status_metrics['status']}"
    )


# ==================================================
# Entry point
# ==================================================

if __name__ == "__main__":
    main()