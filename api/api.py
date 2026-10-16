from pydantic import BaseModel, Field
import pandas as pd
from fastapi import FastAPI
from pathlib import Path
import torch
import joblib
import numpy as np
import json

app = FastAPI()

# -----------------------
# MODEL ARCHITECTURE (MUST MATCH TRAINING)
# -----------------------
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(5, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

# -----------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# LOAD ARTIFACTS
# -----------------------
model = Net()
model.load_state_dict(
    torch.load(MODEL_DIR / "model.pt", map_location=torch.device("cpu"))
)
model.eval()

scaler = joblib.load(MODEL_DIR /"scaler.pkl")

with open(MODEL_DIR /"threshold.json", "r") as f:
    threshold = json.load(f)["threshold"]

# -----------------------
class PredictionInput(BaseModel):
    air_temp: float = Field(..., gt=0)
    process_temp: float = Field(..., gt=0)
    rpm: float = Field(..., ge=0)
    torque: float = Field(..., ge=0)
    tool_wear: float = Field(..., ge=0)

# API ENDPOINT
# -----------------------
@app.post("/predict")
def predict(data: PredictionInput):

    features = pd.DataFrame([{
        "air_temp": data.air_temp,
        "process_temp": data.process_temp,
        "rpm": data.rpm,
        "torque": data.torque,
        "tool_wear": data.tool_wear
    }])

    # Ensure column names match the fitted scaler
    if hasattr(scaler, "feature_names_in_"):
        features.columns = scaler.feature_names_in_

    # Preprocessing
    features_scaled = scaler.transform(features)

    # Tensor conversion
    x = torch.tensor(features_scaled, dtype=torch.float32)

    # Model inference
    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit).item()

    pred = int(prob > threshold)

    return {
        "failure_probability": float(prob),
        "failure": pred,
        "threshold": threshold
    }