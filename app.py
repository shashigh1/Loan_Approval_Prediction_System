import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load preprocessor and model
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open('best_xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit app layout
st.set_page_config(page_title="Loan Approval Prediction", layout="centered")
st.title("🏦 Loan Approval Prediction App")

# User Inputs
def user_input_features():
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10)
    education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
    self_employed = st.selectbox("Self Employed", ['Yes', 'No'])
    income_annum = st.number_input("Annual Income (₹)", min_value=0)
    loan_amount = st.number_input("Loan Amount (₹)", min_value=0)
    loan_term = st.selectbox("Loan Term (in months)", [6, 12, 24, 36, 48, 60])
    cibil_score = st.slider("CIBIL Score", 300, 900, 700)
    residential_assets_value = st.number_input("Residential Asset Value (₹)", min_value=0)
    commercial_assets_value = st.number_input("Commercial Asset Value (₹)", min_value=0)
    luxury_assets_value = st.number_input("Luxury Asset Value (₹)", min_value=0)
    bank_asset_value = st.number_input("Bank Asset Value (₹)", min_value=0)

    data = {
        'no_of_dependents': [no_of_dependents],
        'education': [education],
        'self_employed': [self_employed],
        'income_annum': [income_annum],
        'loan_amount': [loan_amount],
        'loan_term': [loan_term],
        'cibil_score': [cibil_score],
        'residential_assets_value': [residential_assets_value],
        'commercial_assets_value': [commercial_assets_value],
        'luxury_assets_value': [luxury_assets_value],
        'bank_asset_value': [bank_asset_value]
    }

    return pd.DataFrame(data)

# Get user input
input_df = user_input_features()

# Make prediction
if st.button("Predict"):
    try:
        input_processed = preprocessor.transform(input_df)
        prediction = model.predict(input_processed)
        result = " Loan request will be Approved" if prediction[0] == 1 else " Loan request will be Rejected"
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
