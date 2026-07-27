import os
from dotenv import load_dotenv

# load values from .env file to environment variables
load_dotenv()

dataset_path = os.getenv("DATASET_PATH")
model_path = os.getenv("MODEL_PATH")
environment = os.getenv("ENVIRONMENT")

print("Dataset path:",dataset_path)
print("Model path:", model_path)
print("Environment:",environment)
