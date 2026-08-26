import pandas as pd

from fastapi import APIRouter, HTTPException

from backend.schemas import StudentInput, PredictionResponse
from backend.model_loader import load_model
from backend.config import MODEL_PATH


router = APIRouter(
    prefix="/api/v1",
    tags=["Prediction"]
)


# Load model once when this module is initialized
model = load_model(MODEL_PATH)


@router.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_score(data: StudentInput):

    try:
        # Convert Pydantic model → dictionary
        input_data = data.model_dump()

        # Dictionary → DataFrame
        input_df = pd.DataFrame([input_data])

        # Prediction
        prediction = model.predict(input_df)

        return PredictionResponse(
            status="success",
            predicted_exam_score=float(prediction[0])
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed."
        ) from exc