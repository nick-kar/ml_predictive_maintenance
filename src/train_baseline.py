import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# CONFIG: SELECT DATASET
# -------------------------
DATASET = "kaggle"   # options: "kaggle" or "synthetic"


# -------------------------
# DATA LOADER
# -------------------------
def load_data(dataset_name):

    if dataset_name == "synthetic":
        df = pd.read_csv("data/sensor_data.csv")

        X = df.drop("failure", axis=1)
        y = df["failure"]

    elif dataset_name == "kaggle":
        df = pd.read_csv("data/ai4i2020.csv")

        y = df["Machine failure"]

        X = df.drop(columns=[
            "UDI",
            "Product ID",
            "Type",
            "Machine failure",
            "TWF", "HDF", "PWF", "OSF", "RNF"
        ])

    else:
        raise ValueError("Unknown dataset. Choose 'kaggle' or 'synthetic'.")

    return X, y


# -------------------------
# LOAD DATA
# -------------------------
X, y = load_data(DATASET)

print(f"\nUsing dataset: {DATASET}")
print("Shape:", X.shape)


# -------------------------
# TRAIN / TEST SPLIT
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if DATASET == "kaggle" else None
)


# -------------------------
# EVALUATION FUNCTION
# -------------------------
def evaluate(name, y_true, pred, prob):
    print(f"\n=== {name} ===")
    print("Accuracy:", round(accuracy_score(y_true, pred), 4))
    print("F1-score:", round(f1_score(y_true, pred), 4))
    print("ROC-AUC:", round(roc_auc_score(y_true, prob), 4))


# -------------------------
# LOGISTIC REGRESSION
# -------------------------
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

pred_lr = log_reg.predict(X_test)
prob_lr = log_reg.predict_proba(X_test)[:, 1]

evaluate("Logistic Regression", y_test, pred_lr, prob_lr)


# -------------------------
# RANDOM FOREST
# -------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)
prob_rf = rf.predict_proba(X_test)[:, 1]

evaluate("Random Forest", y_test, pred_rf, prob_rf)