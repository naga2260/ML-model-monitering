from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class InferenceLog(Base):
    """
    Stores one record for every successful model inference.
    """

    __tablename__ = "inference_logs"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    timestamp: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    input_features: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False
    )

    prediction: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )

    latency_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )

    status: Mapped[str] = mapped_column(
        String(20),
        default="success",
        nullable=False
    )
