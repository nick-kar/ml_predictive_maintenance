# 🔧 Predictive Maintenance AI System

[![ML API CI](https://github.com/nick-kar/ml_predictive_maintenance/actions/workflows/ci.yml/badge.svg)](https://github.com/nick-kar/ml_predictive_maintenance/actions/workflows/ci.yml)

An end-to-end machine learning project for predicting industrial equipment failure from sensor data, combining classical ML, PyTorch neural networks, a FastAPI inference service, a Streamlit dashboard, Docker containerization, and automated CI testing.

## 🎯 Problem Statement

Unexpected industrial equipment failures can lead to costly downtime and maintenance operations.

This project explores predictive maintenance using machine operating conditions and sensor measurements to estimate the probability of machine failure.

The system is developed as a reproducible ML engineering portfolio project using the AI4I 2020 Predictive Maintenance dataset and synthetic data.

## ⚙️ Technology Stack

| Component | Technologies |
|---|---|
| Machine learning | Scikit-learn, PyTorch |
| Data processing | Python, Pandas, NumPy |
| Backend API | FastAPI, Pydantic |
| Dashboard | Streamlit |
| Containerization | Docker |
| Automated testing | pytest, HTTPX |
| Continuous integration | GitHub Actions |

## 🚀 System Overview

The project includes:

1. Data preprocessing and feature preparation.
2. Classical ML baselines for failure classification.
3. A PyTorch neural network for failure probability estimation.
4. A FastAPI REST endpoint for model inference.
5. Pydantic validation of incoming prediction requests.
6. An interactive Streamlit dashboard.
7. Docker packaging for reproducible API execution.
8. Automated testing and container verification through GitHub Actions.

## 🧠 Machine Learning Models

Three model families were explored:

- **Logistic Regression:** interpretable baseline classifier.
- **Random Forest:** ensemble model for nonlinear relationships.
- **PyTorch Neural Network:** feedforward classifier used by the inference API.

### Model Evaluation

Performance is evaluated using accuracy, F1-score, and ROC-AUC, with particular attention to class imbalance.

| Model | Accuracy | F1-score | ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.9725 | 0.3678 | 0.8949 |
| Random Forest | 0.9830 | 0.6667 | 0.9651 |
| PyTorch Neural Network | — | ≈ 0.454 | ≈ 0.926 |

*Reported results are from experiments on the AI4I 2020 dataset. The neural network's F1-score reflects threshold tuning; its accuracy is omitted here rather than estimated.*

## 🌐 FastAPI Inference

The REST API exposes a prediction endpoint:

`POST /predict`

### Example Request

```json
{
  "air_temp": 298.1,
  "process_temp": 308.6,
  "rpm": 1500,
  "torque": 40,
  "tool_wear": 100
}
```

### Example Response

```json
{
  "failure_probability": 0.4483,
  "failure": 0,
  "threshold": 0.6
}
```

The response provides the estimated failure probability, a binary classification, and the configured decision threshold.

Input validation is implemented using Pydantic. Missing required fields and invalid negative measurements produce HTTP 422 validation responses.

## 🐳 Docker Containerization

The FastAPI inference service is packaged as a Docker image using Python 3.11 and CPU-only PyTorch.

### Build the Image

```bash
docker build -t predictive-maintenance-api .
```

### Start the Container

```bash
docker run --rm -p 8000:8000 predictive-maintenance-api
```

The API is then accessible at:

`http://localhost:8000/docs`

The interactive Swagger documentation can be used to submit prediction requests.

## 🧪 Automated Testing

The project uses pytest and FastAPI's TestClient to verify the API.

The test suite covers:

- Successful prediction requests.
- Missing required input features.
- Invalid negative input values.
- Consistency of repeated predictions.

**8 automated tests pass successfully.**

To run the tests in an environment with the project and development dependencies installed:

```bash
python -m pytest tests/ -v
```

## 🔄 GitHub Actions Continuous Integration

The GitHub Actions workflow automatically runs on pushes and pull requests.

It performs the following operations:

1. Checks out the repository.
2. Configures Python.
3. Installs CPU-only PyTorch and project dependencies.
4. Executes automated pytest tests.
5. Builds the Docker image.
6. Starts the API inside a Docker container.
7. Sends a prediction request to the running API and validates the response.

**The complete CI pipeline has been successfully executed on a GitHub-hosted Linux runner.**

This verifies that the API can be built, started, and tested in a containerized environment independently of the local development machine.

[View CI workflow](https://github.com/nick-kar/ml_predictive_maintenance/actions/workflows/ci.yml)

## 📁 Project Structure

```text
ml_predictive_maintenance/
├── .github/
│   └── workflows/
│       └── ci.yml
├── api/
│   └── api.py
├── data/
├── experiments/
├── models/
│   ├── model.pt
│   ├── scaler.pkl
│   └── threshold.json
├── src/
├── tests/
│   └── test_api.py
├── app.py
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

## ▶️ Local Development

### 1. Install Dependencies

Install the dependencies listed in `requirements.txt` and a compatible CPU-only PyTorch installation.

```bash
python -m pip install -r requirements.txt
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

### 2. Run the API

```bash
uvicorn api.api:app --reload
```

Open `http://127.0.0.1:8000/docs` to explore the API.

### 3. Run Automated Tests

Install the testing dependencies:

```bash
python -m pip install pytest httpx
python -m pytest tests/ -v
```

### 4. Run the Streamlit Dashboard

With Streamlit installed:

```bash
streamlit run app.py
```

## 📈 Key Engineering Outcomes

- Implemented and evaluated multiple ML classifiers on industrial failure data.
- Developed a PyTorch inference pipeline using saved model artifacts.
- Exposed predictions through a FastAPI REST service.
- Added structured input validation and HTTP error handling.
- Implemented automated API tests.
- Containerized the inference service using Docker.
- Established a working GitHub Actions CI pipeline with container-level API verification.

## 👨‍💻 Author

**Nikolaos Karkanis**

Background in Physics, Electronic Physics and Telecommunications, and Electrical and Computer Engineering.

Focus: Machine Learning, Scientific AI, Predictive Systems, and ML Software Engineering.

