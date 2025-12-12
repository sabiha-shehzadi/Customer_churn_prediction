import gradio as gr
import pickle
import numpy as np

# Load model and encoders
model = pickle.load(open("customer_churn_model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

def predict_churn(gender, senior, partner, dependents, tenure, phone, internet, monthly, total):
    
    # Apply encoders
    gender = encoders["gender"].transform([gender])[0]
    partner = encoders["Partner"].transform([partner])[0]
    dependents = encoders["Dependents"].transform([dependents])[0]
    phone = encoders["PhoneService"].transform([phone])[0]
    internet = encoders["InternetService"].transform([internet])[0]

    features = np.array([[gender, senior, partner, dependents,
                          tenure, phone, internet, monthly, total]])

    pred = model.predict(features)[0]

    return "⚠️ Customer will CHURN" if pred == 1 else "✅ Customer will NOT churn"


# Gradio UI
demo = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Gender"),
        gr.Dropdown([0, 1], label="Senior Citizen"),
        gr.Dropdown(["Yes", "No"], label="Partner"),
        gr.Dropdown(["Yes", "No"], label="Dependents"),
        gr.Number(label="Tenure (Months)"),
        gr.Dropdown(["Yes", "No"], label="Phone Service"),
        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service"),
        gr.Number(label="Monthly Charges"),
        gr.Number(label="Total Charges"),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Customer Churn Prediction App"
)

demo.launch()
