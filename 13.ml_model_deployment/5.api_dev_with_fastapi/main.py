from fastapi import FastAPI
from pydantic import BaseModel


# create fastapi application
app = FastAPI(title="FastAPI Basics Demo")

# GET endpoint
@app.get("/greet")
def home():
    return {"message": "Welcome to FastAPI"}

# request body schema
class User(BaseModel):
    name: str
    age: int

# POST
@app.post("/user")
def create_user(user: User):
    return {
        "status": "success",
        "message": f"User {user.name} created",
        "age": user.age
    }

# mock ml prediction endpoint

class PredictionInput(BaseModel):
    age: int
    bmi: float
    glucose: float

@app.post("/predict")
def predict(input_data: PredictionInput):
    # simple fake rule
    if input_data.glucose > 140 or input_data.bmi > 35:
        prediction = "High Risk"
    else:
        prediction = "Low Risk"
    
    return {
        "prediction": prediction,
        "model": "mock-diabetes-model",
        "input": input_data
    }
