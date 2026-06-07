from fastapi import FastAPI
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
# LOAD ARTIFACTS
# -----------------------
model = Net()
model.load_state_dict(torch.load("model.pt", map_location=torch.device('cpu')))
model.eval()

scaler = joblib.load("scaler.pkl")

with open("threshold.json", "r") as f:
    threshold = json.load(f)["threshold"]

# -----------------------
# API ENDPOINT
# -----------------------
@app.post("/predict")
def predict(data: dict):

    features = np.array([[
        data["air_temp"],
        data["process_temp"],
        data["rpm"],
        data["torque"],
        data["tool_wear"]
    ]])

    # scale
    features = scaler.transform(features)

    # tensor
    x = torch.tensor(features, dtype=torch.float32)

    # inference
    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit).item()

    pred = int(prob > threshold)

    return {
        "failure_probability": float(prob),
        "failure": pred,
        "threshold": threshold
    }