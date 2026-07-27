import os
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from joblib import load


def predict(model, input_data: dict):
    df = pd.DataFrame([input_data])
    predicted_value = model.predict(df)[0]
    return predicted_value


def main():
    try:
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

        loaded_model = load(MODEL_PATH)
        logging.info("Model loaded")

        input_data = {
            "Pregnancies": 2,
            "Glucose": 120,
            "BloodPressure": 70,
            "SkinThickness": 25,
            "Insulin": 80,
            "BMI": 28.5,
            "DiabetesPedigreeFunction": 0.45,
            "Age": 35
        }
        prediction = predict(model=loaded_model, input_data=input_data)

        if prediction == 1:
            print("⚠️ Model predicts: Diabetes")
            logging.info("Prediction result: Diabetes")
        else:
            print("✅ Model predicts: No Diabetes")
            logging.info("Prediction result: No Diabetes")

        logging.info("Prediction script finished")


    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        logging.exception(f"Prediction script failed: {e}")
        raise


if __name__ == "__main__":
    main()
