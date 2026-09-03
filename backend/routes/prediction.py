
import json
import logging
import time
from datetime import datetime
import pandas as pd
from fastapi import APIRouter, HTTPException

from backend.schemas import StudentInput, PredictionResponse
from backend.model_loader import load_model
from backend.config import MODEL_PATH


router = APIRouter(
    prefix="/api/v1",
    tags=["Prediction"]
)


# Create logger for this module
logger = logging.getLogger(__name__)


# Load model once when this module is initialized
model = load_model(MODEL_PATH)


@router.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_score(data: StudentInput):

    # Start timer
    start_time = time.perf_counter()

    try:
        # Convert Pydantic model → dictionary
        input_data = data.model_dump()

        # Dictionary → DataFrame
        input_df = pd.DataFrame([input_data])

        # Prediction
        prediction = model.predict(input_df)

        pred_value = float(prediction[0])

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Create logging payload
        log_payload = {
            "event": "prediction",
            "timestamp": datetime.now().isoformat(),
            "input_features": input_data,
            "prediction": pred_value,
            "latency_ms": round(latency_ms, 2)
        }

        # Write prediction log
        logger.info(json.dumps(log_payload))

        return PredictionResponse(
            status="success",
            predicted_exam_score=pred_value
        )

    except Exception as exc:

        # Calculate latency even when prediction fails
        latency_ms = (time.perf_counter() - start_time) * 1000

        error_payload = {
            "event": "prediction_failed",
            "input_features": data.model_dump(),
            "latency_ms": round(latency_ms, 2),
            "error": str(exc)
        }

        # Write error log
        logger.error(json.dumps(error_payload))

        raise HTTPException(
            status_code=500,
            detail="Prediction failed."
        ) from exc
        