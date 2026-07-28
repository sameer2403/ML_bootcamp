# pyrefly: ignore [missing-import]
from fastapi import FastAPI
from pydantic import BaseModel


from predictor import predict

app = FastAPI(
    title = "Diabetes Prediction API",
    description = "API for diabetes prediction",
    version = "1.0.0"
)


#define input schema
class predictionInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int


#ml prediction endpoint
@app.post("/predict")    
def predict_diabetes(input_data: predictionInput):
    prediction = predict(input_data.model_dump())    
    return{
        "Prediction":int(prediction)
    }




# API - menu(list of items) - collection of endpoints
# Endpoints - items - individual services : http://localhost:8000/user
# Get/Post -  methods - how are we going to interact with api