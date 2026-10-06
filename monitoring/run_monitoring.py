import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

MONITOR_SCRIPT = (
    BASE_DIR / "monitoring" / "monitor_drift.py"
)


def main():

    print("=" * 60)
    print("HOURLY ML MONITORING")
    print("=" * 60)

    result = subprocess.run(
        [
            sys.executable,
            str(MONITOR_SCRIPT),
        ],
        cwd=BASE_DIR,
        check=False,
    )

    if result.returncode != 0:

        print(
            "\nMonitoring job failed."
        )

        sys.exit(result.returncode)

    print(
        "\nMonitoring job completed successfully."
    )


if __name__ == "__main__":
    main()

