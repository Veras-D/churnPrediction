import streamlit as st
import joblib
import numpy as np
import pickle


# Load the model and scaler
with open("random_forest_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit Interface
st.title("📊 Bank Churn Prediction")
st.write("Fill in the details below to predict whether a customer will leave the bank.")

# User Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
account_balance = st.number_input("Account Balance", min_value=0, value=5000)
has_credit_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
is_active = st.selectbox("Is the Customer Active?", ["Yes", "No"])
salary = st.number_input("Salary", min_value=0, value=3000)

# Convert inputs to the expected format
has_credit_card = 1 if has_credit_card == "Yes" else 0
is_active = 1 if is_active == "Yes" else 0

# Prediction Button
if st.button("Predict Churn"):
    input_data = np.array([[age, account_balance, has_credit_card, is_active, salary]])
    input_data_scaled = scaler.transform(input_data)
    prediction = rf_model.predict(input_data_scaled)

    if prediction[0] == 1:
        st.error("❌ The customer has a HIGH CHANCE of leaving the bank!")
    else:
        st.success("✅ The customer is likely to STAY with the bank.")
