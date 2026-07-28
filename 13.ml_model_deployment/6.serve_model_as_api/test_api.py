import requests

# fastapi endpoint url
API_URL = "http://127.0.0.1:8000/predict"

payload = {
    "Pregnancies": 2,
    "Glucose": 80,
    "BloodPressure": 101,
    "SkinThickness": 20,
    "Insulin": 90,
    "BMI": 29,
    "DiabetesPedigreeFunction": 0.483,
    "Age": 27
}

#send post request
response = requests.post(API_URL, json=payload)

print("Status code", response.status_code)
print("Response", response.json())
