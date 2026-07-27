import os
import logging
import pandas as pd
from dotenv import load_dotenv

# load .env content to env variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),            # terminal
        logging.FileHandler("app.log")      # file
    ]
)

DATASET_PATH = os.getenv("DATASET_PATH")

logging.info("Loading dataset")
df = pd.read_csv(DATASET_PATH)
logging.info("Dataset loaded successfully")

print(df.head())
