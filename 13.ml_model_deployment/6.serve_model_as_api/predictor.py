import os
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from joblib import load

# load .env content ro env vars
load_dotenv()

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))
MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR") / os.getenv("MODEL_NAME")
LOG_PATH = PROJECT_ROOT / os.getenv("LOG_DIR") / os.getenv("LOG_NAME")

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH)
    ]
)

# loading the trained model
# NOTE: model is loaded only once and not for every prediction
logging.info("Loading trained model...")
model = load(MODEL_PATH)
logging.info("Model loaded successfully.")

def predict(input_data: dict):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    logging.info("Model provided a prediction")
    return prediction


# example usage
# input_data = {
#     "Pregnancies": 2,
#     "Glucose": 120,
#     "BloodPressure": 150,
#     "SkinThickness": 25,
#     "Insulin": 50,
#     "BMI": 28.5,
#     "DiabetesPedigreeFunction": 0.583,
#     "Age": 30
# }

# model_prediction = predict(input_data=input_data)
# print(model_prediction)
