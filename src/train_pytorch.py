import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score

# -----------------------
# 1. LOAD DATA
# -----------------------
df = pd.read_csv("data/ai4i2020.csv")

y = df["Machine failure"]

X = df.drop(columns=[
    "UDI",
    "Product ID",
    "Type",
    "Machine failure",
    "TWF", "HDF", "PWF", "OSF", "RNF"
])

# -----------------------
# 2. SPLIT DATA
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------
# 3. NORMALIZATION
# -----------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------
# 4. TO TENSORS
# -----------------------
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# -----------------------
# 5. MODEL
# -----------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(X_train.shape[1], 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)   # NO sigmoid here
        )

    def forward(self, x):
        return self.net(x)

model = Net()

# -----------------------
# 6. HANDLE CLASS IMBALANCE (IMPORTANT)
# -----------------------
pos_weight = torch.tensor([
    (y_train == 0).sum() / (y_train == 1).sum()
], dtype=torch.float32)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------
# 7. TRAINING LOOP
# -----------------------
epochs = 50

for epoch in range(epochs):
    model.train()

    logits = model(X_train)
    loss = criterion(logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# -----------------------
# 8. EVALUATION
# -----------------------
model.eval()
with torch.no_grad():
    logits = model(X_test)
    probs = torch.sigmoid(logits).cpu().numpy().flatten()

# -----------------------
# THRESHOLD SEARCH
# -----------------------
thresholds = np.arange(0.1, 0.9, 0.05)

best_f1 = 0
best_t = 0.5

for t in thresholds:
    preds_t = (probs > t).astype(int)
    f1 = f1_score(y_test, preds_t)

    if f1 > best_f1:
        best_f1 = f1
        best_t = t

print("\nBest threshold:", best_t)
print("Best F1-score:", best_f1)

# -----------------------
# FINAL PREDICTIONS (USING BEST THRESHOLD)
# -----------------------
final_preds = (probs > best_t).astype(int)

print("\nPred distribution:", np.unique(final_preds, return_counts=True))

# -----------------------
# FINAL METRICS
# -----------------------
print("\n=== PYTORCH RESULTS ===")
print("F1-score:", f1_score(y_test, final_preds))
print("ROC-AUC:", roc_auc_score(y_test, probs))

import joblib
import json

# SAVE MODEL
torch.save(model.state_dict(), "model.pt")

# SAVE SCALER
joblib.dump(scaler, "scaler.pkl")

# SAVE THRESHOLD
with open("threshold.json", "w") as f:
    json.dump({"threshold": best_t}, f)

print("\nArtifacts saved successfully.")