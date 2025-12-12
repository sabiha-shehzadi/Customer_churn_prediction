import gradio as gr
import pickle
import numpy as np
import pandas as pd
import warnings
import os

# Suppress warnings
warnings.filterwarnings("ignore")

# --- 1. LOAD SYSTEM ASSETS ---
try:
    with open("customer_churn_model.pkl", "rb") as f:
        loaded_object = pickle.load(f)

    # Robust model extraction
    if isinstance(loaded_object, dict):
        if "model" in loaded_object: model = loaded_object["model"]
        elif "classifier" in loaded_object: model = loaded_object["classifier"]
        else: model = list(loaded_object.values())[0]
    else:
        model = loaded_object

    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
        
    print("✅ System: ChurnGuard AI loaded successfully.")
    
except Exception as e:
    print(f"❌ System Error: {e}")
    model = None
    encoders = {}

# --- 2. CORE UTILITIES ---

def safe_encode(col_name, value):
    """Safely encodes a value, returning 0 if encoder or key is missing."""
    try:
        if col_name in encoders:
            return encoders[col_name].transform([value])[0]
        for key in encoders.keys():
            if key.lower() == col_name.lower():
                return encoders[key].transform([value])[0]
        return 0 # Default fallback
    except:
        return 0

def create_feature_array(gender, senior, partner, dependents, tenure, phone, internet, monthly, total, contract="Month-to-month", tech_support="No", online_security="No"):
    """Creates the 19-column numpy array expected by the model."""
    senior_val = 1 if senior == "Yes" else 0
    
    # Defaults for hidden fields to ensure shape match
    # We allow Contract, TechSupport, OnlineSecurity to be overridden for "What-If" analysis
    return np.array([[
        safe_encode("gender", gender),      
        senior_val,          
        safe_encode("Partner", partner),     
        safe_encode("Dependents", dependents),  
        float(tenure),          
        safe_encode("PhoneService", phone),       
        safe_encode("MultipleLines", "No"),       
        safe_encode("InternetService", internet),    
        safe_encode("OnlineSecurity", online_security),      
        safe_encode("OnlineBackup", "No"),        
        safe_encode("DeviceProtection", "No"),    
        safe_encode("TechSupport", tech_support),         
        safe_encode("StreamingTV", "No"),         
        safe_encode("StreamingMovies", "No"),     
        safe_encode("Contract", contract),
        safe_encode("PaperlessBilling", "Yes"),   
        safe_encode("PaymentMethod", "Electronic check"), 
        float(monthly),         
        float(total)            
    ]])

def generate_email(name, risk_drivers, offer_details):
    """Generates a personalized retention email."""
    return f"""Subject: Exclusive Offer for our Valued Customer

Dear Valued Customer,

We noticed you've been with us for a while, and we want to ensure you're getting the best experience possible.
{f"We understand that {risk_drivers} might be a concern." if risk_drivers else ""}

To show our appreciation, we'd like to offer you:
👉 {offer_details}

Please reply to this email to activate this offer immediately.

Warm regards,
Customer Success Team"""

# --- 3. MAIN PREDICTION LOGIC (SINGLE USER) ---

def analyze_single_customer(gender, senior, partner, dependents, tenure, phone, internet, monthly, total):
    if model is None: return "Error: Model not loaded.", "", ""

    # A. Base Prediction
    features = create_feature_array(gender, senior, partner, dependents, tenure, phone, internet, monthly, total)
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1] if hasattr(model, "predict_proba") else 0
    
    # B. Explainability Logic (Heuristic based on common churn drivers)
    reasons = []
    if float(monthly) > 80: reasons.append("High Monthly Charges")
    if float(tenure) < 12: reasons.append("New Customer (Low Tenure)")
    if internet == "Fiber optic": reasons.append("Fiber Optic Issues (Common)")
    
    explanation_text = " • ".join(reasons) if reasons else "Standard Usage Pattern"

    # C. "What-If" AI Optimization (The Magic Feature)
    optimization_msg = "✅ No intervention needed."
    email_draft = "Customer is safe. No email required."
    
    if prob > 0.40: # If risk is significant
        # Simulate: Give 15% discount and Tech Support
        new_monthly = float(monthly) * 0.85
        new_features = create_feature_array(gender, senior, partner, dependents, tenure, phone, internet, new_monthly, total, contract="One year", tech_support="Yes")
        new_prob = model.predict_proba(new_features)[0][1]
        
        drop = prob - new_prob
        if drop > 0:
            optimization_msg = f"""
            <div style="background: #ecfdf5; padding: 15px; border-radius: 8px; border-left: 5px solid #10b981;">
                <b>🚀 AI Recommendation:</b><br>
                If we offer a <b>15% Discount</b> & free <b>Tech Support</b>:<br>
                Churn Probability drops from <b>{prob:.1%}</b> ➝ <b>{new_prob:.1%}</b>.
            </div>
            """
            email_draft = generate_email("Customer", explanation_text, "A 15% Discount on your monthly bill + Free Tech Support upgrade")
        else:
            optimization_msg = "⚠️ Risk is structural. Discounts may not help."
            email_draft = generate_email("Customer", "recent service issues", "A Priority Support Plan")

    # D. Result Card HTML
    if pred == 1:
        card_html = f"""
        <div class="result-card churn-card">
            <div class="icon-box">⚠️</div>
            <h2>High Churn Risk</h2>
            <div class="score" style="color: #991b1b;">{prob:.1%}</div>
            <p>Risk Drivers: <b>{explanation_text}</b></p>
        </div>
        """
    else:
        card_html = f"""
        <div class="result-card safe-card">
            <div class="icon-box">🛡️</div>
            <h2>Safe Customer</h2>
            <div class="score" style="color: #065f46;">{prob:.1%}</div>
            <p>Status: Healthy</p>
        </div>
        """
        
    return card_html, optimization_msg, email_draft

# --- 4. BATCH PROCESSING LOGIC ---

def process_batch_file(file_obj):
    if file_obj is None: return None
    
    try:
        # Read CSV
        df = pd.read_csv(file_obj.name)
        
        # Check for required columns (Basic check)
        required = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'InternetService', 'MonthlyCharges', 'TotalCharges']
        
        # Process each row
        results = []
        probs = []
        
        for index, row in df.iterrows():
            # Map CSV columns to inputs (using .get for safety)
            f = create_feature_array(
                str(row.get('gender', 'Male')), 
                "Yes" if row.get('SeniorCitizen', 0)==1 else "No",
                str(row.get('Partner', 'No')),
                str(row.get('Dependents', 'No')),
                float(row.get('tenure', 0)),
                str(row.get('PhoneService', 'No')),
                str(row.get('InternetService', 'No')),
                float(row.get('MonthlyCharges', 0)),
                float(row.get('TotalCharges', 0))
            )
            pred = model.predict(f)[0]
            prob = model.predict_proba(f)[0][1]
            results.append("Churn" if pred == 1 else "No Churn")
            probs.append(round(prob, 4))
            
        df['Prediction'] = results
        df['Churn_Probability'] = probs
        
        # Save to new file
        output_path = "processed_churn_results.csv"
        df.to_csv(output_path, index=False)
        return output_path
        
    except Exception as e:
        # Return a dummy text file with error if pandas fails
        with open("error_log.txt", "w") as f:
            f.write(f"Error processing file: {str(e)}")
        return "error_log.txt"

# --- 5. UI CONFIGURATION ---

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
body { font-family: 'Plus Jakarta Sans', sans-serif; }
.result-card { background: white; border-radius: 16px; padding: 25px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
.churn-card { border: 1px solid #fda4af; background: #fff1f2; }
.safe-card { border: 1px solid #6ee7b7; background: #ecfdf5; }
.score { font-size: 3rem; font-weight: 800; margin: 10px 0; }
.section-header { font-weight: 700; color: #334155; margin-bottom: 10px; border-left: 4px solid #6366f1; padding-left: 10px; }
"""

theme = gr.themes.Soft(primary_hue="indigo", neutral_hue="zinc")

# --- 6. GRADIO APP STRUCTURE ---

with gr.Blocks(theme=theme, css=custom_css, title="ChurnGuard AI") as demo:
    
    gr.Markdown("# 🛡️ ChurnGuard AI: Retention Intelligence System")
    
    with gr.Tabs():
        
        # TAB 1: DASHBOARD
        with gr.TabItem("📊 Live Analysis Dashboard"):
            with gr.Row():
                # LEFT: INPUTS
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("<div class='section-header'>👤 Profile</div>")
                        with gr.Row():
                            t1_gender = gr.Radio(["Male", "Female"], label="Gender", value="Male")
                            t1_senior = gr.Radio(["No", "Yes"], label="Senior Citizen", value="No")
                        with gr.Row():
                            t1_partner = gr.Radio(["Yes", "No"], label="Has Partner?", value="No")
                            t1_dep = gr.Radio(["Yes", "No"], label="Has Dependents?", value="No")
                            
                    with gr.Group():
                        gr.Markdown("<div class='section-header'>📡 Services</div>")
                        t1_phone = gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes")
                        t1_net = gr.Dropdown(["Fiber optic", "DSL", "No"], label="Internet Type", value="Fiber optic")

                    with gr.Group():
                        gr.Markdown("<div class='section-header'>💳 Financials</div>")
                        t1_tenure = gr.Number(label="Tenure (Months)", value=12)
                        t1_monthly = gr.Number(label="Monthly Bill ($)", value=85.0)
                        t1_total = gr.Number(label="Total Charges ($)", value=1020.0, interactive=True)
                    
                    btn_analyze = gr.Button("🚀 Analyze Risk", variant="primary", size="lg")

                # RIGHT: RESULTS
                with gr.Column(scale=1):
                    gr.Markdown("### Analysis Results")
                    out_card = gr.HTML(label="Risk Assessment")
                    
                    gr.Markdown("### 💡 AI Recommendations")
                    out_opt = gr.HTML(label="What-If Analysis")
                    
                    gr.Markdown("### ✉️ Retention Action")
                    # FIXED LINE HERE: Removed 'show_copy_button=True'
                    out_email = gr.Textbox(label="Draft Email for Customer", lines=6, interactive=True)

        # TAB 2: BATCH PROCESSING
        with gr.TabItem("📂 Batch Processing (CSV)"):
            gr.Markdown("### Upload a CSV file with customer data to process thousands of records at once.")
            gr.Markdown("*Required Columns: gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, InternetService, MonthlyCharges, TotalCharges*")
            
            with gr.Row():
                file_upload = gr.File(label="Upload CSV", file_types=[".csv"])
                btn_process = gr.Button("⚙️ Process Batch", variant="primary")
            
            out_file = gr.File(label="Download Results")

    # --- LOGIC WIRING ---
    
    # 1. Auto-calc Total
    def auto_calc(t, m): return round(float(t or 0)*float(m or 0), 2)
    t1_tenure.change(auto_calc, [t1_tenure, t1_monthly], t1_total)
    t1_monthly.change(auto_calc, [t1_tenure, t1_monthly], t1_total)

    # 2. Main Analysis
    btn_analyze.click(
        analyze_single_customer,
        inputs=[t1_gender, t1_senior, t1_partner, t1_dep, t1_tenure, t1_phone, t1_net, t1_monthly, t1_total],
        outputs=[out_card, out_opt, out_email]
    )
    
    # 3. Batch Processing
    btn_process.click(process_batch_file, inputs=file_upload, outputs=out_file)

# Launch
demo.launch(share=True, ssr_mode=False)
