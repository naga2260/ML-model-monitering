from fastapi import FastAPI

from backend.routes.prediction import router as prediction_router


app = FastAPI(
    title="Student Exam Score Prediction API",
    description="API for predicting student exam scores using a trained ML pipeline.",
    version="1.0.0",
)


@app.get("/")
def home():
    return {
        "message": "Student Exam Score Prediction API is running!"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


app.include_router(prediction_router)