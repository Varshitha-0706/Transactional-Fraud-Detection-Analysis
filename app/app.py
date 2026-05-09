import streamlit as st
import numpy as np
import joblib
import pandas as pd

# LOAD MODEL
model = joblib.load("fraud_detection_model.pkl")

# TRY LOADING SCALER (SAFE MODE)
try:
    scaler = joblib.load("scaler.pkl")
    scaler_available = True
except:
    scaler_available = False

# PAGE SETTINGS
st.set_page_config(page_title="Fraud Detection App", layout="wide")

# TITLE
st.title("💳 Credit Card Fraud Detection System")
st.markdown("### 🔍 Detect fraudulent transactions using Machine Learning")


# SIDEBAR INPUTS
st.sidebar.header("🧾 Enter Transaction Details")

inputs = []

for i in range(1, 29):
    val = st.sidebar.number_input(f"V{i}", value=0.0, format="%.5f")
    inputs.append(val)

amount = st.sidebar.number_input("💰 Transaction Amount", value=0.0)

# SCALE AMOUNT IF SCALER EXISTS
if scaler_available:
    amount = scaler.transform([[amount]])[0][0]

inputs.append(amount)

# THRESHOLD SLIDER
threshold = st.sidebar.slider(
    "⚙️ Fraud Detection Threshold",
    0.0, 1.0, 0.5
)

# PREDICTION
input_array = np.array(inputs).reshape(1, -1)

if st.button("🚀 Predict Transaction"):

    probability = model.predict_proba(input_array)[0][1]
    prediction = 1 if probability > threshold else 0

    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("⚠️ Fraudulent Transaction Detected")
        else:
            st.success("✅ Legitimate Transaction")

    with col2:
        st.metric("Fraud Probability", f"{probability:.4f}")

# RANDOM SAMPLE TEST
st.markdown("---")
st.subheader("🎲 Test with Random Transaction")

if st.button("Generate Random Transaction"):

    random_data = np.random.normal(size=(1, 29))

    prob = model.predict_proba(random_data)[0][1]
    pred = 1 if prob > threshold else 0

    st.write("### Generated Data")
    st.dataframe(pd.DataFrame(random_data, columns=[f"Feature_{i}" for i in range(29)]))

    if pred == 1:
        st.error("⚠️ Fraud Detected")
    else:
        st.success("✅ Legitimate Transaction")

    st.write(f"**Fraud Probability:** {prob:.4f}")

# FOOTER
st.markdown("---")
st.warning("⚠️ Note: This model uses anonymized PCA features (V1–V28). Inputs are for demonstration purposes.")
st.caption("Built with ❤️ using Streamlit | ML Fraud Detection Project")