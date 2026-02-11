import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page config
st.set_page_config(layout="wide", page_title="Customer Churn Prediction", page_icon="⚡")

@st.cache_resource
def load_artifacts():
    if not os.path.exists('models/churn_model.pkl'):
        st.error("Model artifacts not found. Please run 'python src/model.py' first.")
        return None, None, None
        
    model = joblib.load('models/churn_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    encoders = joblib.load('models/encoders.pkl')
    return model, scaler, encoders

model, scaler, encoders = load_artifacts()

def main():
    st.title("⚡ Customer Churn Prediction App")
    st.markdown("""
    This application predicts the likelihood of a customer churning based on their subscription details.
    
    **How it works:**
    1. Enter customer details in the sidebar.
    2. Click 'Predict' to see the result.
    """)
    
    if model is None:
        return

    st.sidebar.header("Customer Details")
    
    # Input fields
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.sidebar.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
    total_charges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, value=500.0)

    # Preprocessing
    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })

    # Encode categorical variables
    for col, le in encoders.items():
        if col in input_data.columns:
             # Handle unseen labels carefully (though here inputs are controlled)
            input_data[col] = le.transform(input_data[col])
            
    # Scale numerical features
    # Note: We need to ensure the columns are in the same order as training
    # Loading feature names if possible or ensuring order manually
    # For robust production code, we'd save feature names. Here assuming standard order from generator.
    # The scaler expects all features, including encoded ones.
    
    # Reorder columns to match training logic if needed. 
    # train_test_split in model.py was on X which was df.drop('Churn'). 
    # df columns were: gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, 
    # InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, 
    # StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges.
    # We should ensure input_data has same column order.
    
    cols_order = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 
                  'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                  'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
                  
    input_data = input_data[cols_order]

    input_scaled = scaler.transform(input_data)

    if st.button("Predict Churn"):
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"High Risk of Churn! (Probability: {probability:.2%})")
            st.warning("Customer is likely to leave.")
        else:
            st.success(f"Low Risk of Churn. (Probability: {probability:.2%})")
            st.info("Customer is likely to stay.")
            
        # Feature Importance (Optional)
        st.subheader("Model Insights")
        importances = model.feature_importances_
        feature_df = pd.DataFrame({'Feature': cols_order, 'Importance': importances})
        feature_df = feature_df.sort_values(by='Importance', ascending=False).head(5)
        st.bar_chart(feature_df.set_index('Feature'))

if __name__ == "__main__":
    main()

# TODO: Add sidebar branding
