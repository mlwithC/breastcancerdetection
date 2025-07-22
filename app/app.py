import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("svm_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🔬 Breast Cancer Prediction App")
st.markdown("Enter the 30 features to predict if the tumor is **Benign** or **Malignant**.")

# Define default benign and malignant samples
benign_sample = [
    12.45, 15.7, 82.57, 477.1, 0.1278, 0.17, 0.1578, 0.08089, 0.2087, 0.07613,
    0.3345, 1.916, 2.261, 27.19, 0.00551, 0.03058, 0.04422, 0.01654, 0.01872, 0.005217,
    14.5, 20.49, 95.29, 641.2, 0.1713, 0.3912, 0.4336, 0.1823, 0.3215, 0.1032
]

malignant_sample = [
    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
]

# Initialize session state for inputs
if "input_data" not in st.session_state:
    st.session_state.input_data = [0.0] * 30

# Fill sample buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Use Benign Sample"):
        st.session_state.input_data = benign_sample
with col2:
    if st.button("Use Malignant Sample"):
        st.session_state.input_data = malignant_sample

# Input fields
input_data = []
for i in range(30):
    val = st.number_input(f"Feature {i+1}", value=float(st.session_state.input_data[i]), format="%.5f")
    input_data.append(val)

# Predict button
if st.button("🔍 Predict"):
    prediction = model.predict([input_data])[0]
    if prediction == 1:
        st.success("🟢 The tumor is **Benign**.")
    else:
        st.error("🔴 The tumor is **Malignant**.")
