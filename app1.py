import streamlit as st
import pickle
import numpy as np

# Page Config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

st.markdown("""
<style>

label {
    color: #00E5FF !important;
    font-weight: bold !important;
    font-size: 18px !important;
}

.stApp {
    background: linear-gradient(to right, #000000, #8B0000);
}

h1 {
    color: white !important;
    text-align: center;
}

.stButton > button {
    width: 100%;
    background-color: green;
    color: #FFFFFF;
    border-radius: 10px;
    height: 50px;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# Load Model
with open("attrition_model.pkl", "rb") as f:
    model = pickle.load(f)

# Title
st.title("💳 AI Fraud Detection System")
st.markdown(
    "<h3 style='color:#f1c40f;'>Detect suspicious financial transactions using Machine Learning 🐎</h3>",
    unsafe_allow_html=True
)

st.markdown("---")

# Input Fields (ONLY 6 FEATURES)
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h3 style='color:#00E5FF;'>👤 Sender Details</h3>", unsafe_allow_html=True)

    step = st.number_input("⚡ Step", min_value=0)
    type_ = st.number_input("🔄 Transaction Type (Encoded)", min_value=0)
    amount = st.number_input("💰 Amount", min_value=0.0)

with col2:
    st.markdown("<h3 style='color:#FFD700;'>🏦 Balance Details</h3>", unsafe_allow_html=True)

    oldbalanceOrg = st.number_input("🏦 Old Balance Sender", min_value=0.0)
    oldbalanceDest = st.number_input("🏦 Old Balance Receiver", min_value=0.0)
    newbalanceDest = st.number_input("🏦 New Balance Receiver", min_value=0.0)

st.markdown("---")

# Prediction
if st.button("🔍 Detect Fraud", use_container_width=True):

    features = np.array([[ 
        step,
        type_,
        amount,
        oldbalanceOrg,
        oldbalanceDest,
        newbalanceDest
    ]])

    prediction = model.predict(features)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("🚨 FRAUD TRANSACTION DETECTED")
        st.warning("This transaction appears suspicious.")
    else:
        st.success("✅ LEGITIMATE TRANSACTION")
        st.info("No fraud detected.")

# Footer
st.markdown("""
<div style='text-align:center;
            color:#FFD700;
            font-size:18px;
            font-weight:bold;'>
    🚀 Developed by Karan | Machine Learning Fraud Detection Project
</div>
""", unsafe_allow_html=True)
