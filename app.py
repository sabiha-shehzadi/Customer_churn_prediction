import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("customer_churn_model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

st.title("Customer Churn Prediction App")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0)

if st.button("Predict"):
    # Encoding categorical inputs
    data = {
        "gender": gender,
        "Partner": Partner,
        "Dependents": Dependents,
        "PhoneService": PhoneService,
        "InternetService": InternetService
    }

    for col in data:
        data[col] = encoders[col].transform([data[col]])[0]

    features = np.array([[data["gender"], SeniorCitizen, data["Partner"], data["Dependents"],
                          tenure, data["PhoneService"], data["InternetService"],
                          MonthlyCharges, TotalCharges]])

    prediction = model.predict(features)[0]

    if prediction == 1:
        st.error("⚠️ Customer is likely to CHURN!")
    else:
        st.success("✨ Customer is NOT likely to churn.")
