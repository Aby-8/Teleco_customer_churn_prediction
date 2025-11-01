import streamlit as st
import pandas as pd
import joblib
st.title('Customer Churn Prediction App')
st.write("Enter customer details to predict whether they will churn or not.")
# Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])  # 0=No, 1=Yes
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, step=1.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, step=1.0)
tenure_grp = st.selectbox("Tenure Group", ["0-12", "12-24", "24-48", "48-60", "60+"])

# Prediction button
if st.button("Predict"):
    # Convert input to DataFrame
    input_data = pd.DataFrame([[
        gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines,
        InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
        StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
        MonthlyCharges, TotalCharges, tenure_grp
    ]], columns=[
        "gender","SeniorCitizen","Partner","Dependents","PhoneService","MultipleLines",
        "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport",
        "StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod",
        "MonthlyCharges","TotalCharges","tenure_grp"
    ])
    input_data=pd.get_dummies(input_data)
    model_colnms=joblib.load("model columns.pkl")
    input_data=input_data.reindex(columns=model_colnms,fill_value=0)
    # Run prediction
    model=joblib.load('model redt')
    prediction = model.predict(input_data)[0]
    
    # Show result
    if prediction == 1:
        st.error("⚠️ The customer is likely to Churn.")
    else:
        st.success("✅ The customer is not likely to Churn.")
#py -3.10 -m streamlit run app.py
