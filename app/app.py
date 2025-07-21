import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer

# Load trained model
model = joblib.load('svm_model.pkl')

# Load original dataset to get sample test cases
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

# Get example cases from dataset
malignant_example = X[np.where(y == 0)[0][0]]  # y=0 is malignant
benign_example = X[np.where(y == 1)[0][0]]     # y=1 is benign

# Title
st.title("🔬 Breast Cancer Prediction App")
st.markdown("Select a test case or enter custom values to predict **Benign** or **Malignant**.")

# Dropdown to select mode
case_option = st.selectbox("Choose Test Mode:", ["Benign Test Case", "Malignant Test Case", "Custom Input"])

if case_option == "Benign Test Case":
    st.subheader("🟢 Benign Case Input")
    input_data = benign_example
    for i, val in enumerate(input_data):
        st.write(f"**{feature_names[i]}:** {val:.2f}")

elif case_option == "Malignant Test Case":
    st.subheader("🔴 Malignant Case Input")
    input_data = malignant_example
    for i, val in enumerate(input_data):
        st.write(f"**{feature_names[i]}:** {val:.2f}")

else:
    st.subheader("✍️ Custom Input Features")
    input_data = []
    for i in range(len(feature_names)):
        val = st.slider(
            f"{feature_names[i]}",
            float(np.min(X[:, i])),
            float(np.max(X[:, i])),
            float(np.mean(X[:, i])),
            key=f"custom_{i}"
        )
        input_data.append(val)
    input_data = np.array(input_data)

# Prediction
input_reshaped = input_data.reshape(1, -1)
prediction = model.predict(input_reshaped)[0]

# Result Display
st.subheader("📊 Prediction Result")
if prediction == 0:
    st.error("🔴 Prediction: **Malignant** (Cancer Present)")
else:
    st.success("🟢 Prediction: **Benign** (No Cancer)")

# Optional Debug: Show raw input values
with st.expander("📄 Show Raw Input Values"):
    for i, val in enumerate(input_data):
        st.write(f"{feature_names[i]}: {val:.2f}")
