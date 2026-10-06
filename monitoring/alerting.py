import os
import urllib.parse
import urllib.request

from dotenv import load_dotenv

load_dotenv()


TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN"
)

TELEGRAM_CHAT_ID = os.getenv(
    "TELEGRAM_CHAT_ID"
)


def send_telegram_alert(message: str) -> None:
    """
    Send a message through a Telegram bot.
    """

    if not TELEGRAM_BOT_TOKEN:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN is not configured."
        )

    if not TELEGRAM_CHAT_ID:
        raise ValueError(
            "TELEGRAM_CHAT_ID is not configured."
        )

    url = (
        f"https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    data = urllib.parse.urlencode(
        {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
    )

    with urllib.request.urlopen(
        request,
        timeout=10,
    ) as response:

        if response.status != 200:
            raise RuntimeError(
                "Telegram alert failed with "
                f"HTTP status {response.status}."
            )


def build_alert_message(
    status: str,
    monitoring_metrics: dict,
    data_drift_metrics: dict,
    prediction_drift_metrics: dict,
    reasons: list[str],
) -> str:

    lines = [
        "🚨 ML MONITORING ALERT",
        "",
        f"Status: {status}",
        "",
        "Metrics:",
        (
            f"Requests: "
            f"{monitoring_metrics['request_count']}"
        ),
        (
            f"Average latency: "
            f"{monitoring_metrics['avg_latency_ms']:.2f} ms"
        ),
        (
            f"P95 latency: "
            f"{monitoring_metrics['p95_latency_ms']:.2f} ms"
        ),
        (
            f"Data drift: "
            f"{data_drift_metrics['data_drift_detected']}"
        ),
        (
            f"Drifted features: "
            f"{data_drift_metrics['drifted_features']}/"
            f"{data_drift_metrics['total_features']}"
        ),
        (
            f"Prediction drift: "
            f"{prediction_drift_metrics['prediction_drift_detected']}"
        ),
        "",
        "Reasons:",
    ]

    for reason in reasons:
        lines.append(f"• {reason}")

    return "\n".join(lines)


def build_recovery_message() -> str:

    return (
        "✅ ML MONITORING RECOVERY\n\n"
        "The monitoring system has returned "
        "to NORMAL status."
    )

