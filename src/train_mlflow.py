from pathlib import Path
import json
import random

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
TRACKING_DIR = ROOT / "mlruns"

mlflow.set_tracking_uri(TRACKING_DIR.as_uri())
mlflow.set_experiment("Predictive-Maintenance-Comparison")

# Load dataset
df = pd.read_csv(DATA_PATH)

y = df["Machine failure"]
X = df.drop(columns=[
    "UDI", "Product ID", "Type", "Machine failure",
    "TWF", "HDF", "PWF", "OSF", "RNF"
])

# 60% train, 20% validation, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=SEED, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5,
    random_state=SEED, stratify=y_temp
)

def evaluate(y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities))
    }

def log_results(model_name, parameters, metrics, threshold):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(parameters)
        mlflow.log_param("dataset", "AI4I_2020")
        mlflow.log_param("split_seed", SEED)
        mlflow.log_param("train_fraction", 0.6)
        mlflow.log_param("validation_fraction", 0.2)
        mlflow.log_param("test_fraction", 0.2)
        mlflow.log_metric("decision_threshold", threshold)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("model_family", model_name)

    print(f"{model_name}: {metrics}")

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=SEED)
lr.fit(X_train, y_train)

lr_val_prob = lr.predict_proba(X_val)[:, 1]
lr_test_prob = lr.predict_proba(X_test)[:, 1]

# Classical models use a fixed 0.5 threshold
log_results(
    "Logistic_Regression",
    {"max_iter": 1000},
    evaluate(y_test, lr_test_prob),
    0.5
)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=100, random_state=SEED
)
rf.fit(X_train, y_train)

rf_test_prob = rf.predict_proba(X_test)[:, 1]

log_results(
    "Random_Forest",
    {"n_estimators": 100},
    evaluate(y_test, rf_test_prob),
    0.5
)

# PyTorch neural network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

x_train = torch.tensor(X_train_scaled, dtype=torch.float32)
x_val = torch.tensor(X_val_scaled, dtype=torch.float32)
x_test = torch.tensor(X_test_scaled, dtype=torch.float32)
t_train = torch.tensor(
    y_train.to_numpy(), dtype=torch.float32
).reshape(-1, 1)

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

model = Net()

positive = float(t_train.sum().item())
negative = float(t_train.numel() - positive)

criterion = torch.nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([negative / positive])
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100

for epoch in range(epochs):
    model.train()
    logits = model(x_train)
    loss = criterion(logits, t_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    val_prob = torch.sigmoid(model(x_val)).numpy().ravel()
    test_prob = torch.sigmoid(model(x_test)).numpy().ravel()

# Select threshold using validation data only
thresholds = np.arange(0.1, 0.91, 0.05)
best_threshold = max(
    thresholds,
    key=lambda t: f1_score(
        y_val, val_prob >= t, zero_division=0
    )
)

test_metrics = evaluate(
    y_test, test_prob, threshold=float(best_threshold)
)

log_results(
    "PyTorch_Neural_Network",
    {
        "epochs": epochs,
        "learning_rate": 0.001,
        "hidden_layers": "32,16",
        "optimizer": "Adam"
    },
    test_metrics,
    float(best_threshold)
)

# Save a compact experiment summary
summary = {
    "logistic_regression": evaluate(y_test, lr_test_prob),
    "random_forest": evaluate(y_test, rf_test_prob),
    "pytorch": test_metrics,
    "pytorch_validation_threshold": float(best_threshold)
}

output = ROOT / "experiment_summary.json"
output.write_text(json.dumps(summary, indent=2))

print("\nMLflow experiment tracking completed.")
print(json.dumps(summary, indent=2))
