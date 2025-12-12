# 🛡️ ChurnGuard AI: Customer Retention Intelligence

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)
![Gradio](https://img.shields.io/badge/Frontend-Gradio-orange?style=for-the-badge&logo=gradio&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/AI-Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Data-Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

### 🚀 **Live Demo:** [Click Here to Open App](https://sabihakhan-customer-churn-khan.hf.space)

---

## 📋 Project Overview

**ChurnGuard AI** is an advanced Machine Learning application designed to help telecom companies predict customer churn and take proactive action. 

Unlike standard classification models that simply output "Yes/No," this system serves as a complete **Decision Support Dashboard**. It analyzes customer demographics, services, and billing patterns to calculate a specific **Churn Probability Score**.

If a customer is identified as "High Risk," the system automatically provides **AI-driven recommendations** and drafts a **retention email** to help the support team save the customer.

## ✨ Key Features

### 1. 📊 Real-Time Predictive Dashboard
- User-friendly interface built with **Gradio** using a modern "Glassmorphism" design.
- Instantly predicts churn risk based on 9 key inputs (Gender, Tenure, Monthly Charges, etc.).
- Visualizes confidence scores with color-coded Risk Cards (Green for Safe, Red for Risk).

### 2. 🧠 "What-If" Analysis (AI Optimization)
- The system doesn't just predict problems; it suggests solutions.
- **Simulation Engine:** If a customer is high-risk, the AI simulates how offering a **15% discount** or **Free Tech Support** would impact their probability of staying.

### 3. ✉️ Automated Retention Emailer
- Bridges the gap between Data Science and Marketing.
- Automatically generates a personalized, professional email draft tailored to the customer's specific churn reasons, ready for the support agent to send.

### 4. 📂 Batch Processing System
- Enterprise-grade feature allowing users to upload a **CSV file** containing thousands of customers.
- The system processes the file in bulk and generates a downloadable report with predictions for every customer.

---

## 🛠️ Tech Stack

*   **Language:** Python 3.10
*   **Machine Learning:** Scikit-Learn (Random Forest Classifier)
*   **Interface:** Gradio (SaaS-style UI)
*   **Data Processing:** Pandas, NumPy
*   **Deployment:** Hugging Face Spaces

---

## ⚙️ How to Run Locally

If you want to run this application on your own machine:

1.  **Clone the repository**
    ```bash
    git clone https://huggingface.co/spaces/sabihakhan/customer-churn-khan
    cd customer-churn-khan
    ```

2.  **Install requirements**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**
    ```bash
    python app.py
    ```
    The app will launch in your browser at `http://127.0.0.1:7860`.

---

## 🧠 Model Details

*   **Algorithm:** Random Forest Classifier (Optimized for tabular data).
*   **Training Data:** Telco Customer Churn Dataset (7,000+ records).
*   **Preprocessing:** 
    *   Label Encoding for categorical variables.
    *   Imputation logic to handle missing inputs in the UI.
*   **Input Features:** 
    *   Demographics (Gender, Senior Citizen, Partner, Dependents)
    *   Services (Phone, Internet, Online Security, Tech Support, etc.)
    *   Account Info (Contract, Payment Method, Tenure, Monthly Charges)

---

## 📸 Screenshots

<img width="1917" height="1078" alt="image" src="https://github.com/user-attachments/assets/429287bd-b5a5-4261-93d2-c4340ac5b8bd" />
<img width="1918" height="1076" alt="image" src="https://github.com/user-attachments/assets/702230e7-48cf-405d-9fae-3ac00255b152" />

## 🤝 Contact & Credits

Developed by **Sabiha Khan**. 
This project demonstrates the application of AI in Business Intelligence and Customer Relationship Management (CRM).

[Visit Live App](https://sabihakhan-customer-churn-khan.hf.space)
