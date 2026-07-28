import requests

# fastapi endpoint url
API_URL = "http://127.0.0.1:8000/predict"

payload = {
    "Pregnancies": 2,
    "Glucose": 140,
    "BloodPressure": 100,
    "SkinThickness": 30,
    "Insulin": 80,
    "BMI": 25,
    "DiabetesPedigreeFunction": 0.355,
    "Age": 28
}

# send POST request
response = requests.post(API_URL, json=payload)
response_dictionary = response.json()
prediction = response_dictionary["Prediction"]


print("Status Code:", response.status_code)
print("Response JSON:", response)
print("Response dictionary:", response_dictionary)
print("Prediction:", prediction)
