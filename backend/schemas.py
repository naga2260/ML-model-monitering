from pydantic import BaseModel, Field, ConfigDict


class StudentInput(BaseModel):
    """
    Input schema for student exam score prediction.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Hours_Studied": 25,
                "Attendance": 85,
                "Parental_Involvement": "Medium",
                "Access_to_Resources": "High",
                "Extracurricular_Activities": "Yes",
                "Sleep_Hours": 7,
                "Previous_Scores": 80,
                "Motivation_Level": "Medium",
                "Internet_Access": "Yes",
                "Tutoring_Sessions": 1,
                "Family_Income": "Medium",
                "Teacher_Quality": "Medium",
                "School_Type": "Public",
                "Peer_Influence": "Neutral",
                "Physical_Activity": 3,
                "Learning_Disabilities": "No",
                "Parental_Education_Level": "College",
                "Distance_from_Home": "Near",
                "Gender": "Female",
            }
        }
    )

    Hours_Studied: int = Field(..., ge=0, description="Hours studied")
    Attendance: int = Field(..., ge=0, le=100)
    
    Parental_Involvement: str
    Access_to_Resources: str
    Extracurricular_Activities: str

    Sleep_Hours: int = Field(..., ge=0, le=24)
    Previous_Scores: int = Field(..., ge=0, le=100)

    Motivation_Level: str
    Internet_Access: str

    Tutoring_Sessions: int = Field(..., ge=0)

    Family_Income: str
    Teacher_Quality: str
    School_Type: str
    Peer_Influence: str

    Physical_Activity: int = Field(..., ge=0)

    Learning_Disabilities: str
    Parental_Education_Level: str
    Distance_from_Home: str
    Gender: str


class PredictionResponse(BaseModel):
    """
    Response schema for prediction endpoint.
    """

    status: str
    predicted_exam_score: float