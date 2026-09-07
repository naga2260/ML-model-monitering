import logging
import subprocess
import sys
from pathlib import Path
from datetime import datetime


# ============================================================
# Project paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MONITORING_DIR = BASE_DIR / "monitoring"
LOG_DIR = BASE_DIR / "logs"

INGEST_SCRIPT = MONITORING_DIR / "ingest_logs.py"
DRIFT_SCRIPT = MONITORING_DIR / "monitor_drift.py"

LOG_FILE = LOG_DIR / "monitoring.log"


# ============================================================
# Logging configuration
# ============================================================

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_FILE,
            encoding="utf-8"
        ),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


# ============================================================
# Run a Python script
# ============================================================

def run_script(
    script_path: Path,
    script_name: str
) -> None:

    logger.info("=" * 60)
    logger.info("Starting %s", script_name)
    logger.info("=" * 60)

    if not script_path.exists():

        raise FileNotFoundError(
            f"{script_name} not found: {script_path}"
        )

    try:

        result = subprocess.run(
            [
                sys.executable,
                str(script_path)
            ],
            cwd=BASE_DIR,
            capture_output=True,
            text=True
        )

        # ----------------------------------------------------
        # Log stdout
        # ----------------------------------------------------

        if result.stdout:

            logger.info(
                "%s output:\n%s",
                script_name,
                result.stdout.strip()
            )

        # ----------------------------------------------------
        # Log stderr
        # ----------------------------------------------------

        if result.stderr:

            logger.warning(
                "%s stderr:\n%s",
                script_name,
                result.stderr.strip()
            )

        # ----------------------------------------------------
        # Check exit status
        # ----------------------------------------------------

        if result.returncode != 0:

            raise RuntimeError(
                f"{script_name} failed "
                f"with exit code {result.returncode}"
            )

        logger.info(
            "%s completed successfully.",
            script_name
        )

    except Exception:

        logger.exception(
            "%s failed.",
            script_name
        )

        raise


# ============================================================
# Main monitoring pipeline
# ============================================================

def main():

    start_time = datetime.now()

    logger.info("")
    logger.info("=" * 70)
    logger.info("              ML MONITORING PIPELINE")
    logger.info("=" * 70)
    logger.info(
        "Pipeline started at: %s",
        start_time.strftime("%Y-%m-%d %H:%M:%S")
    )
    logger.info(
        "Project directory: %s",
        BASE_DIR
    )

    try:

        # ----------------------------------------------------
        # STEP 1: Ingest inference logs
        # ----------------------------------------------------

        logger.info("")
        logger.info("STEP 1/2: Ingesting inference logs")

        run_script(
            INGEST_SCRIPT,
            "ingest_logs.py"
        )

        # ----------------------------------------------------
        # STEP 2: Generate drift reports
        # ----------------------------------------------------

        logger.info("")
        logger.info("STEP 2/2: Generating drift reports")

        run_script(
            DRIFT_SCRIPT,
            "monitor_drift.py"
        )

        # ----------------------------------------------------
        # Pipeline completed
        # ----------------------------------------------------

        end_time = datetime.now()

        duration = end_time - start_time

        logger.info("")
        logger.info("=" * 70)
        logger.info("          MONITORING PIPELINE COMPLETED")
        logger.info("=" * 70)

        logger.info(
            "Completed at: %s",
            end_time.strftime("%Y-%m-%d %H:%M:%S")
        )

        logger.info(
            "Total duration: %s",
            duration
        )

        logger.info(
            "Data drift report: %s",
            BASE_DIR / "reports" / "data_drift_report.html"
        )

        logger.info(
            "Prediction drift report: %s",
            BASE_DIR / "reports" / "production_drift_report.html"
        )

    except Exception:

        # ----------------------------------------------------
        # Pipeline failed
        # ----------------------------------------------------

        logger.exception("")
        logger.exception("=" * 70)
        logger.exception("       MONITORING PIPELINE FAILED")
        logger.exception("=" * 70)

        # Important for cron:
        # exit with non-zero status so cron knows the job failed.
        sys.exit(1)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()