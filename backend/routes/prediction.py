import json
import logging
import time
from datetime import datetime

import pandas as pd
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from backend.schemas import StudentInput, PredictionResponse
from backend.model_loader import load_model
from backend.config import MODEL_PATH
from backend.database import get_db
from backend.models import InferenceLog


router = APIRouter(
    prefix="/api/v1",
    tags=["Prediction"]
)


# ============================================================
# Logger
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# Load model once when this module is initialized
# ============================================================

model = load_model(MODEL_PATH)


# ============================================================
# Prediction endpoint
# ============================================================

@router.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_score(
    data: StudentInput,
    db: Session = Depends(get_db)
):

    # --------------------------------------------------------
    # Start timer
    # --------------------------------------------------------

    start_time = time.perf_counter()

    try:

        # ----------------------------------------------------
        # Convert Pydantic model → dictionary
        # ----------------------------------------------------

        input_data = data.model_dump()

        # ----------------------------------------------------
        # Dictionary → DataFrame
        # ----------------------------------------------------

        input_df = pd.DataFrame(
            [input_data]
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            input_df
        )

        pred_value = float(
            prediction[0]
        )

        # ----------------------------------------------------
        # Calculate latency
        # ----------------------------------------------------

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        latency_ms = round(
            latency_ms,
            2
        )

        # ====================================================
        # 1. Write application log
        # ====================================================

        log_payload = {

            "event": "prediction",

            "timestamp": datetime.now().isoformat(),

            "input_features": input_data,

            "prediction": pred_value,

            "latency_ms": latency_ms

        }

        logger.info(
            json.dumps(log_payload)
        )

        # ====================================================
        # 2. Write inference to PostgreSQL
        # ====================================================

        inference_record = InferenceLog(

            timestamp=datetime.now(),

            input_features=input_data,

            prediction=pred_value,

            latency_ms=latency_ms,

            status="success"

        )

        db.add(
            inference_record
        )

        db.commit()

        # ====================================================
        # Return API response
        # ====================================================

        return PredictionResponse(

            status="success",

            predicted_exam_score=pred_value

        )

    except Exception as exc:

        # ----------------------------------------------------
        # Calculate latency for failed request
        # ----------------------------------------------------

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        latency_ms = round(
            latency_ms,
            2
        )

        # ----------------------------------------------------
        # Rollback database transaction
        # ----------------------------------------------------

        db.rollback()

        # ----------------------------------------------------
        # Write error log
        # ----------------------------------------------------

        error_payload = {

            "event": "prediction_failed",

            "input_features": data.model_dump(),

            "latency_ms": latency_ms,

            "error": str(exc)

        }

        logger.error(
            json.dumps(error_payload)
        )

        raise HTTPException(

            status_code=500,

            detail="Prediction failed."

        ) from exc

