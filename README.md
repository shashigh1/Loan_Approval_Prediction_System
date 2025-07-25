# Loan_Approval_Prediction_System
🏦 Loan Approval Prediction System
A machine learning-powered Streamlit web application that predicts whether a loan application will be approved or rejected, based on user input such as income, loan amount, credit score (CIBIL), and other financial details.


 Demo
 Try the App: https://loan-approval-prediction-system-shashi.streamlit.app/

Features
Interactive UI built with Streamlit

Preprocessing with StandardScaler and OneHotEncoder using scikit-learn's ColumnTransformer

Predicts loan approval using the best tuned XGBoost model

Real-time prediction output based on user-provided inputs

Clean and responsive design

Project Structure
bash
Copy
Edit
Loan_Approval_Prediction_System/
│
├── app.py                  # Streamlit frontend script
├── preprocessor.pkl        # Saved preprocessing pipeline
├── best_xgb_model.pkl      # Trained XGBoost model
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
ML Model Details
Model Used: XGBoost Classifier

Target Variable: Loan Status (Approved / Rejected)

Training Metrics:

Accuracy: ~99%

Precision, Recall, F1-score analyzed during training phase

Features Used:

no_of_dependents

education

self_employed

income_annum

loan_amount

loan_term

cibil_score

residential_assets_value

commercial_assets_value

luxury_assets_value

bank_asset_value
