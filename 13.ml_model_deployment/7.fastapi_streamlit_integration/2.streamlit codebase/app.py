import os
import requests
import streamlit as st
from dotenv import load_dotenv



# load .env values to env vars
load_dotenv()

# get API_URL saved in env variable
API_URL = os.getenv("API_URL")

# set page configurations
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details and click **Predict**")

st.subheader("Patient Details")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 10, 2)
    glucose = st.number_input("Glucose", 0, 400, 120)
    blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
    skin_thickness = st.number_input("Skin Thickness", 0, 100, 25)

with col2:
    insulin = st.number_input("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 28.5)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.45)
    age = st.number_input("Age", 1, 100, 35)


if st.button("🔍 Predict"):
    input_data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }

    response = requests.post(API_URL, json=input_data)
    response_dictionary = response.json()
    prediction = response_dictionary["Prediction"]

    st.divider()

    if prediction == 1:
        st.error("The patient has Diabetes **(+ve)**")
    else:
        st.success("The patinet doen't have Diabetes **(-ve)**")
        