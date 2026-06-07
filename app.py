import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import joblib
import json

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Predictive Maintenance AI System",
    layout="wide"
)

# ----------------------------
# MODEL DEFINITION
# ----------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

# ----------------------------
# LOAD ARTIFACTS
# ----------------------------
model = Net()
model.load_state_dict(torch.load("models/model.pt", map_location="cpu"))
model.eval()

scaler = joblib.load("models/scaler.pkl")

with open("models/threshold.json", "r") as f:
    threshold = json.load(f)["threshold"]

# ----------------------------
# HEADER
# ----------------------------
st.title("🔧 Predictive Maintenance AI System")
st.markdown("Predict machine failure using sensor data + Machine Learning + Deep Learning")

st.divider()

# ----------------------------
# LAYOUT
# ----------------------------
col1, col2 = st.columns(2)

# ----------------------------
# INPUT SECTION
# ----------------------------
with col1:
    st.subheader("📥 Sensor Inputs")

    air_temp = st.slider("Air Temperature (K)", 250.0, 320.0, 298.0)
    process_temp = st.slider("Process Temperature (K)", 250.0, 320.0, 308.0)
    rpm = st.slider("Rotational Speed (RPM)", 0, 3000, 1500)
    torque = st.slider("Torque (Nm)", 0.0, 100.0, 40.0)
    tool_wear = st.slider("Tool Wear (min)", 0, 300, 100)

    predict_btn = st.button("🔍 Predict Machine Status")

# ----------------------------
# PREDICTION SECTION
# ----------------------------
with col2:
    st.subheader("📊 Prediction Result")

    if predict_btn:

        # Prepare input
        x = np.array([[air_temp, process_temp, rpm, torque, tool_wear]])
        x = scaler.transform(x)
        x = torch.tensor(x, dtype=torch.float32)

        # Model inference
        with torch.no_grad():
            logit = model(x)
            prob = torch.sigmoid(logit).item()

        pred = int(prob > threshold)

        # Display metrics
        st.metric(label="Failure Probability", value=f"{prob:.3f}")

        st.progress(float(prob))

        # Decision
        if pred == 1:
            st.error("⚠ HIGH RISK: Machine Failure Likely")
        else:
            st.success("✅ LOW RISK: Machine Healthy")

# ----------------------------
# PROJECT INFO
# ----------------------------
st.divider()

st.markdown("""
### 📌 Project Overview
This system predicts machine failure using sensor data and compares classical ML models with a deep learning model.

### 🧠 Models Used
- Logistic Regression
- Random Forest
- PyTorch Neural Network

### 📊 Output
- Failure probability
- Binary risk classification
""")