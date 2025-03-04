import streamlit as st
import pickle
import numpy as np

# Load model and scaler
with open("random_forest_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit UI
st.title("🔮 Customer Churn Prediction")
st.write("Enter customer details to predict churn.")

# User Inputs
account_length = st.number_input("Account Length", min_value=0, max_value=400, value=100)
international_plan = st.selectbox("International Plan", ["No", "Yes"])
voice_mail_plan = st.selectbox("Voice Mail Plan", ["No", "Yes"])
total_day_minutes = st.number_input("Total Day Minutes", min_value=0.0, max_value=400.0, value=180.0)
total_day_calls = st.number_input("Total Day Calls", min_value=0, max_value=400, value=100)
total_eve_minutes = st.number_input("Total Evening Minutes", min_value=0.0, max_value=400.0, value=150.0)
total_eve_calls = st.number_input("Total Evening Calls", min_value=0, max_value=400, value=100)
total_night_minutes = st.number_input("Total Night Minutes", min_value=0.0, max_value=400.0, value=200.0)
total_night_calls = st.number_input("Total Night Calls", min_value=0, max_value=400, value=100)
total_intl_minutes = st.number_input("Total Intl Minutes", min_value=0.0, max_value=400.0, value=20.0)
total_intl_calls = st.number_input("Total Intl Calls", min_value=0, max_value=20, value=5)
customer_service_calls = st.number_input("Customer Service Calls", min_value=0, max_value=10, value=1)

# Ensure input matches the 19 features expected by the model
input_data = np.array([[
    account_length,
    int(international_plan == "Yes"),
    int(voice_mail_plan == "Yes"),
    total_day_minutes,
    total_day_calls,
    total_eve_minutes,
    total_eve_calls,
    total_night_minutes,
    total_night_calls,
    total_intl_minutes,
    total_intl_calls,
    customer_service_calls,
    0,  # Placeholder for missing feature
    0,  # Add correct default values based on training data
    0,
    0,
    0,
    0,
    0
]])

# Debugging: Check input shape before scaling
st.write(f"🛠️ Input Shape Before Scaling: {input_data.shape}")
st.write(f"🔍 Scaler Expects: {scaler.n_features_in_} features")

# Apply scaling
input_scaled = scaler.transform(input_data)

# Predict button
if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)
    result = "Yes" if prediction[0] == 1 else "No"
    st.write(f"**Churn Prediction: {result}**")
