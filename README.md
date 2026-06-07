# 🔧 Predictive Maintenance AI System

A full-stack Machine Learning system that predicts machine failure from industrial sensor data.

It combines classical ML, deep learning, model deployment (FastAPI), and an interactive dashboard (Streamlit).

---

## 🎯 Problem Statement

Industrial machines generate sensor data continuously.  
The goal is to predict machine failure before it happens to reduce downtime and maintenance costs.

---

## ⚙️ Tech Stack

- Python
- PyTorch
- Scikit-learn
- FastAPI
- Streamlit
- Pandas / NumPy

---

## 🚀 System Overview

- Data preprocessing (Kaggle AI4I 2020 + synthetic data)
- Classical ML models (baseline comparison)
- Deep Learning model (PyTorch)
- REST API for inference (FastAPI)
- Interactive dashboard (Streamlit)

---

## 🧠 Models

- Logistic Regression  
- Random Forest  
- Neural Network (PyTorch)

---

## 📊 Evaluation Metrics

- Accuracy  
- F1-score  
- ROC-AUC  

---

## 📁 Project Structure

ml_predictive_maintenance/
│
├── api/
├── data/
├── experiments/
├── models/
├── src/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

---

## ▶ How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Train models
python src/train_baseline.py  
python src/train_pytorch.py  

### 3. Run API
uvicorn api.api:app --reload

### 4. Run Streamlit dashboard
streamlit run app.py

---

## 📈 Output Example

- Failure probability score  
- Binary classification (Safe / Risk)  
- Interactive dashboard visualization  

---

## 🧠 Key Learnings

- End-to-end ML system design  
- Imbalanced classification handling  
- PyTorch model training  
- FastAPI deployment  
- Streamlit dashboard creation  

This project demonstrates an end-to-end production-style ML workflow from data to deployment.
---

## 👨‍💻 Author

Physics + Electronic Physics + PhD in Electrical & Computer Engineering  
Focus: Machine Learning, Scientific AI, Predictive Systems

