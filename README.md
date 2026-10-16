
# 🔧 Predictive Maintenance AI System

[![ML API CI](https://github.com/nick-kar/ml_predictive_maintenance/actions/workflows/ci.yml/badge.svg)](https://github.com/nick-kar/ml_predictive_maintenance/actions/workflows/ci.yml)
[![MLflow Experiments](https://github.com/nick-kar/ml_predictive_maintenance/actions/workflows/mlflow.yml/badge.svg)](https://github.com/nick-kar/ml_predictive_maintenance/actions/workflows/mlflow.yml)

An end-to-end machine learning project for predicting industrial equipment failure from sensor data.

The system combines classical machine learning, PyTorch neural networks, MLflow experiment tracking, a FastAPI inference service, a Streamlit dashboard, Docker containerization, automated testing, and GitHub Actions continuous integration.

## 🎯 Problem Statement

Unexpected industrial equipment failures can lead to costly downtime, production interruptions, and maintenance expenses.

Predictive maintenance aims to identify potential failures before they occur by analyzing machine operating conditions and sensor measurements.

This project develops and evaluates machine learning models that estimate the probability of equipment failure.

The implementation emphasizes reproducible experimentation, model evaluation, API development, containerization, and automated software testing.

**Dataset:** AI4I 2020 Predictive Maintenance Dataset, with additional synthetic data used for experimentation.

## ⚙️ Technology Stack

| Component | Technologies |
|---|---|
| Programming language | Python |
| Machine learning | Scikit-learn, PyTorch |
| Data processing | Pandas, NumPy |
| Experiment tracking | MLflow |
| Backend API | FastAPI, Pydantic |
| Dashboard | Streamlit |
| Containerization | Docker |
| Automated testing | pytest, HTTPX |
| Continuous integration | GitHub Actions |
| Version control | Git, GitHub |

## 🚀 System Overview

The project implements the following components:

1. Data loading, preprocessing, and feature preparation.
2. Classical machine learning models for failure classification.
3. A PyTorch neural network for failure probability estimation.
4. MLflow experiment tracking and model benchmarking.
5. A FastAPI REST endpoint for model inference.
6. Pydantic validation of incoming prediction requests.
7. An interactive Streamlit dashboard.
8. Docker packaging for reproducible API execution.
9. Automated API testing with pytest.
10. GitHub Actions workflows for CI and ML experimentation.

The training, experimentation, and inference components are organized separately to support reproducibility and maintainability.

## 📊 Dataset and Features

The primary dataset is the AI4I 2020 Predictive Maintenance Dataset.

The classification target is:

**Machine failure:** Binary indicator of whether equipment failure occurred.

The models use five numerical input features:

| Feature | Description |
|---|---|
| Air temperature | Ambient air temperature |
| Process temperature | Machine process temperature |
| Rotational speed | Rotational speed of the equipment |
| Torque | Mechanical torque |
| Tool wear | Accumulated tool wear |

Identifier columns, categorical product information, and individual failure-mode indicators are excluded from the model inputs.

The dataset is imbalanced, making F1-score and ROC-AUC important evaluation metrics alongside accuracy.

## 🧠 Machine Learning Models

Three classification approaches are implemented.

### Logistic Regression

A linear baseline classifier used to establish reference performance.

### Random Forest

An ensemble learning model capable of capturing nonlinear relationships between sensor measurements and equipment failure.

### PyTorch Neural Network

A feedforward neural network with the following architecture:

- Input layer: 5 features
- First hidden layer: 32 neurons with ReLU activation
- Second hidden layer: 16 neurons with ReLU activation
- Output layer: 1 neuron producing a classification logit

The neural network uses:

- StandardScaler for feature normalization.
- Binary cross-entropy with logits.
- Class weighting to address imbalanced labels.
- Adam optimization.
- Validation-based classification threshold selection.

The trained neural network is also used by the FastAPI inference service through separately saved model artifacts.

## 📈 Model Evaluation

The reproducible MLflow experiments use a stratified dataset split:

- Training: 60%
- Validation: 20%
- Testing: 20%

A fixed random seed of 42 supports reproducibility.

For the PyTorch model, the classification threshold is selected using the validation set. The test set is reserved for final performance evaluation.

### Test Results

| Model | Accuracy | F1-score | ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 96.85% | 0.2222 | 0.8972 |
| **Random Forest** | **98.20%** | **0.6786** | **0.9750** |
| PyTorch Neural Network | 96.45% | 0.4580 | 0.9229 |

**Best-performing model: Random Forest**

Random Forest achieved the highest performance across the reported test metrics, with an F1-score of 0.6786 and ROC-AUC of 0.9750.

The PyTorch neural network achieved an F1-score of 0.4580 and ROC-AUC of 0.9229 using a validation-selected classification threshold of approximately 0.80.

These results correspond to the reproducible MLflow experiment pipeline. Earlier experiments using different dataset splits or threshold-selection procedures are not directly comparable.

The results demonstrate model performance on the AI4I 2020 dataset and do not establish reliability in real industrial deployments.

## 🔬 Reproducible ML Experiments with MLflow

MLflow is integrated to track and compare model experiments.

Each model is executed as a separate MLflow run.

The experiment pipeline records:

- Model identification.
- Training hyperparameters.
- Dataset information.
- Dataset split configuration.
- Classification threshold.
- Accuracy.
- F1-score.
- ROC-AUC.

The experiment pipeline compares:

1. Logistic Regression.
2. Random Forest.
3. PyTorch Neural Network.

The PyTorch classification threshold is optimized on validation data rather than test data to avoid evaluation bias.

### Experiment Script

`src/train_mlflow.py`

### GitHub Actions Automation

The dedicated MLflow workflow:

1. Checks out the repository.
2. Configures Python 3.11.
3. Installs CPU-only PyTorch and required dependencies.
4. Executes the experiment comparison pipeline.
5. Logs model parameters and evaluation metrics using MLflow.
6. Generates a JSON experiment summary.
7. Uploads MLflow tracking data and the experiment summary as GitHub Actions artifacts.

The experiment pipeline runs on GitHub-hosted infrastructure, without requiring MLflow to be installed locally.

**Workflow:** `.github/workflows/mlflow.yml`

[View MLflow workflow](https://github.com/nick-kar/ml_predictive_maintenance/actions/workflows/mlflow.yml)

### Experiment Output

The workflow generates:

- `mlruns/` — MLflow tracking data.
- `experiment_summary.json` — Evaluation metrics and selected neural-network threshold.

These outputs are generated during execution and made available as downloadable workflow artifacts.

## 🌐 FastAPI Inference Service

The project exposes a REST API for equipment failure prediction.

**Endpoint:**

`POST /predict`

The API accepts five numerical measurements and returns the estimated failure probability, binary prediction, and configured classification threshold.

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

The response above is illustrative. Actual values depend on the saved model artifacts and input measurements.

### Input Validation

Pydantic is used to validate incoming prediction requests.

Validation includes:

- Required input features.
- Numerical input types.
- Nonnegative measurement constraints.

Invalid requests return HTTP 422 responses.

The API loads previously trained model artifacts rather than retraining the neural network for every request.

### Interactive API Documentation

FastAPI automatically generates Swagger documentation at:

http://localhost:8000/docs

## 🐳 Docker Containerization

The FastAPI inference service is packaged as a Docker image.

The container uses:

- Python 3.11.
- CPU-only PyTorch.
- FastAPI.
- Uvicorn.
- Required preprocessing and inference dependencies.

### Build the Docker Image

```bash
docker build -t predictive-maintenance-api .
```

### Start the Container

```bash
docker run --rm -p 8000:8000 predictive-maintenance-api
```

Once the container starts, the API is available at:

http://localhost:8000/docs

Docker provides a reproducible execution environment independent of the local Python installation.

The container build and API startup have been successfully verified using GitHub Actions.

## 🧪 Automated API Testing

The project uses pytest and FastAPI's TestClient to validate the inference service.

The test suite covers:

- Successful prediction requests.
- Missing required input features.
- Invalid negative measurements.
- Consistency of repeated predictions.
- API response validation.

**8 automated tests pass successfully.**

### Run Tests Locally

With the required dependencies installed:

```bash
python -m pytest tests/ -v
```

Automated tests are also executed by the GitHub Actions CI workflow.

## 🔄 GitHub Actions Continuous Integration

The repository includes two automated workflows.

### 1. ML API CI

**File:** `.github/workflows/ci.yml`

This workflow:

1. Checks out the repository.
2. Configures Python.
3. Installs CPU-only PyTorch and project dependencies.
4. Executes the pytest test suite.
5. Builds the Docker image.
6. Starts the FastAPI service inside the container.
7. Sends a prediction request to the running API.
8. Verifies the API response.

The workflow validates both the Python application and its containerized execution.

[View ML API CI workflow](https://github.com/nick-kar/ml_predictive_maintenance/actions/workflows/ci.yml)

### 2. MLflow Experiments

**File:** `.github/workflows/mlflow.yml`

This workflow automatically executes the model comparison pipeline and records experimental results.

It supports reproducible model benchmarking on GitHub-hosted infrastructure.

[View MLflow Experiments workflow](https://github.com/nick-kar/ml_predictive_maintenance/actions/workflows/mlflow.yml)

Both workflows have been successfully executed.

## 📁 Project Structure

```text
ml_predictive_maintenance/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── mlflow.yml
├── api/
│   └── api.py
├── data/
│   ├── ai4i2020.csv
│   └── sensor_data.csv
├── experiments/
├── models/
│   ├── model.pt
│   ├── scaler.pkl
│   └── threshold.json
├── src/
│   ├── train_baseline.py
│   ├── train_pytorch.py
│   └── train_mlflow.py
├── tests/
│   └── test_api.py
├── app.py
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

The MLflow workflow additionally generates experiment tracking data and a JSON summary as runtime artifacts.

## ▶️ Local Development

### 1. Clone the Repository

```bash
git clone https://github.com/nick-kar/ml_predictive_maintenance.git
cd ml_predictive_maintenance
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the virtual environment.

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Install CPU-only PyTorch:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

For automated testing:

```bash
python -m pip install pytest httpx
```

MLflow is installed separately by the experiment workflow and is not required to run the existing inference API.

### 4. Run the FastAPI Service

```bash
uvicorn api.api:app --reload
```

Open:

http://127.0.0.1:8000/docs

### 5. Run Automated Tests

```bash
python -m pytest tests/ -v
```

### 6. Run the Streamlit Dashboard

With Streamlit installed:

```bash
streamlit run app.py
```

### 7. Run MLflow Experiments

To execute the MLflow experiment script locally, install a compatible MLflow version and its dependencies.

Alternatively, use the existing GitHub Actions workflow to run experiments without a local MLflow installation.

The workflow can be started from the GitHub Actions interface using **Run workflow**.

## 📌 Key Engineering Outcomes

- Developed an end-to-end predictive maintenance machine learning project.
- Implemented Logistic Regression, Random Forest, and PyTorch classification models.
- Evaluated models using accuracy, F1-score, and ROC-AUC.
- Achieved 0.9750 ROC-AUC and 0.6786 F1-score using Random Forest on the AI4I 2020 dataset.
- Implemented reproducible experiment tracking with MLflow.
- Improved evaluation methodology using validation-based threshold selection.
- Developed a FastAPI inference service with Pydantic input validation.
- Integrated a Streamlit dashboard.
- Implemented automated API testing with pytest.
- Containerized the inference service using Docker.
- Established GitHub Actions workflows for automated testing, container verification, and ML experimentation.

## 🔮 Future Improvements

Potential extensions include:

- Model monitoring and data drift detection.
- Automated retraining pipelines.
- Hyperparameter optimization.
- Model registry and version management.
- Cloud deployment of the inference API.
- Evaluation on additional industrial datasets.

## 👨‍💻 Author

**Nikolaos Karkanis**

Background in Physics, Electronic Physics and Telecommunications, and Electrical and Computer Engineering.

Focus areas: Machine Learning, Scientific AI, Predictive Systems, and ML Software Engineering.

GitHub: [nick-kar](https://github.com/nick-kar)
