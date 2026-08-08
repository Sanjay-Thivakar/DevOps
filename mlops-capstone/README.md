# Breast Cancer Classification MLOps Pipeline

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0%2B-green.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-3.15.1-orange.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-3.67.1-red.svg)](https://dvc.org/)
[![Docker](https://img.shields.io/badge/Docker-enabled-blue.svg)](https://www.docker.com/)
[![CI Pipeline](https://img.shields.io/badge/CI--Pipeline-passing-brightgreen.svg)](https://github.com/)

An end-to-end Machine Learning Operations (MLOps) capstone project for Breast Cancer Classification. This repository demonstrates modern best practices in modular machine learning pipelines, version control for data, experiment tracking, model registry, local containerized deployment, and continuous integration.

---

## 1. Project Overview
This project implements a reproducible binary classification model to predict whether a breast cancer tumor is malignant or benign. The lifecycle stages are managed using:
*   **Data Versioning (DVC)**: Tracks datasets without committing raw CSV binaries directly to Git.
*   **Modular Pipeline**: Separate scripts handle dataset preparation, preprocessing, and training/evaluation.
*   **Experiment Tracking & Registry (MLflow)**: Logs parameters, metrics, model artifacts, signatures, and handles model registry under a `"champion"` alias.
*   **Production API (FastAPI)**: Serves prediction requests with data validation (Pydantic) and automatic OpenAPI documentation.
*   **Containerization (Docker)**: Packages the FastAPI app, preprocessing scaling artifacts, and MLflow registry into a self-contained runtime.
*   **Continuous Integration (GitHub Actions)**: Validates code linting/formatting, executes tests, and verifies the Docker build on pull requests and pushes.

---

## 2. Technology Stack
*   **Language**: Python 3.11
*   **Machine Learning**: Scikit-Learn, Pandas, NumPy, Joblib
*   **Data Versioning**: DVC (Data Version Control)
*   **Tracking & Registry**: MLflow (using local SQLite database backend)
*   **API Framework**: FastAPI, Pydantic, Uvicorn
*   **Containerization**: Docker
*   **CI/CD**: GitHub Actions
*   **Testing**: Pytest, Httpx2

---

## 3. End-to-End Pipeline Architecture
The execution workflow is structured as follows:

```mermaid
graph TD
    A[Dataset: sklearn Breast Cancer] -->|DVC Tracking| B[data/raw/breast_cancer.csv]
    B -->|data_preprocessing.py| C[data/processed/ splits]
    B -->|data_preprocessing.py| D[models/scaler.joblib]
    C -->|train.py| E[Model 1: Logistic Regression]
    C -->|train.py| F[Model 2: Random Forest]
    C -->|train.py| G[Model 3: SVC]
    E & F & G -->|Log runs, parameters, metrics & artifacts| H[(MLflow sqlite DB)]
    E & F & G -->|Evaluate Metrics| I{Selection Metric: F1}
    I -->|Best Model: Logistic Regression| J[Register in MLflow Registry]
    J -->|Assign Alias: champion| K[models:/BreastCancerClassifier@champion]
    K -->|Startup Loading| L[FastAPI API: app.py]
    D -->|Inference Scaling| L
    L -->|Containerize| M[Docker Image]
    M -->|Automation| N[GitHub Actions CI]
```

---

## 4. Dataset Details
*   **Source**: Wisconsin Breast Cancer Dataset (standard `sklearn` dataset).
*   **Dimensions**: 569 samples, 30 numerical features.
*   **Target Column**: `target`
    *   `0`: Malignant (212 samples)
    *   `1`: Benign (357 samples)

---

## 5. Data Version Control (DVC)
Data preprocessing stages are versioned using DVC to ensure reproducibility.
*   **Parameters (`params.yaml`)**: Stores configuration params:
    ```yaml
    preprocessing:
      test_size: 0.2
      random_state: 42
    ```
*   **Pipeline Stages (`dvc.yaml`)**: Defines dependencies and outputs:
    ```yaml
    stages:
      preprocess:
        cmd: python src/data_preprocessing.py
        deps:
          - data/raw/breast_cancer.csv
          - src/data_preprocessing.py
          - src/utils.py
        params:
          - preprocessing.test_size
          - preprocessing.random_state
        outs:
          - data/processed/X_train.csv
          - data/processed/X_test.csv
          - data/processed/y_train.csv
          - data/processed/y_test.csv
    ```

### DVC Operations
*   To reproduce pipeline stages:
    ```bash
    dvc repro
    ```
*   To check DVC status:
    ```bash
    dvc status
    ```
*   To view pipeline dependency graph:
    ```bash
    dvc dag
    ```

---

## 6. Model Training & Comparison
The project evaluates three models on the scaled features: Logistic Regression, Random Forest, and Support Vector Classifier (SVC). 

### Evaluation Metrics
Each model is evaluated on the test set across: Accuracy, Precision, Recall, F1-score, and ROC-AUC. 
*   **Primary Selection Metric**: F1-score (harmonic mean of precision and recall).
*   **Tie-Breaker**: ROC-AUC.

### Performance Summary
The comparison output saved to `models/model_comparison.csv` is:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LogisticRegression** | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9954 |
| **SVC** | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9950 |
| **RandomForestClassifier** | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9939 |

**Selected Best Model**: `LogisticRegression` (Tied with SVC on F1, but achieved higher ROC-AUC).

---

## 7. MLflow Tracking & Model Registry
Every training run is tracked inside a local SQL backend to support Model Registry functions:
*   **Experiment Name**: `Breast Cancer Classification`
*   **Tracking URI**: `sqlite:///mlflow.db`
*   **Registered Model**: `BreastCancerClassifier`
*   **Assigned Version**: `1`
*   **Assigned Alias**: `champion`

All runs log:
*   Parameters: estimator configurations (e.g. `max_iter`, `penalty`, `random_state`).
*   Metrics: Evaluation metric values (Accuracy, Precision, Recall, F1, ROC-AUC).
*   Artifacts: `model/` directory containing joblib files, input schemas, signatures, and input examples.

---

## 8. FastAPI Inference Service
The REST API is implemented in FastAPI (`src/app.py`). It loads the `champion` model and pre-fitted `scaler.joblib` once during application startup.

### Endpoints
*   `GET /`: Returns API name, version, and endpoints.
*   `GET /health`: Checks service status and validates model/scaler loading.
*   `POST /predict`: Validates incoming payload schema, applies standard scaling, and returns predictions.

### Request Body (`POST /predict`)
Must contain exactly 30 float values matching raw breast cancer characteristics:
```json
{
  "features": [
    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
  ]
}
```

### Response Body
```json
{
  "predicted_class": "malignant",
  "predicted_label": 0,
  "confidence": 1.0,
  "probabilities": {
    "malignant": 1.0,
    "benign": 0.0
  }
}
```

---

## 9. Docker Containerization
The FastAPI application is packaged inside a lightweight Debian Linux container (`python:3.11-slim`).

### Container Commands
*   To build the Docker image:
    ```bash
    docker build -t breast-cancer-mlops:latest .
    ```
*   To run the API container:
    ```bash
    docker run --rm --name breast-cancer-api -p 8000:8000 breast-cancer-mlops:latest
    ```
Once running, the Swagger interactive API docs are available at: [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 10. Verification & Tests
The project features **21 automated unit and integration tests** verifying:
1.  Existence, shape compatibility, valid target values, and scaling stats (no data leakage) in preprocessing.
2.  Proper serialization and load/predict functions of models and metadata.
3.  Successful schema mapping, aliases, and metrics inside local MLflow registries.
4.  Standard scaling, Pydantic input length checking, health reporting, and responses in FastAPI.

### To execute the tests locally:
```bash
pytest
```
*   **Result**: 21 passed.

---

## 11. GitHub Actions Continuous Integration (CI)
The workflow located at [.github/workflows/ci.yml](file:///.github/workflows/ci.yml) triggers automatically on all pushes and pull requests to `main`/`master`.
The pipeline executes:
1.  **Checkout repository**: Clones the workspace codebase.
2.  **Set up Python**: Standardizes environment execution using Python 3.11.
3.  **Install dependencies**: Installs packages listed in `requirements.txt`.
4.  **Download raw dataset**: Inline python script pulls the raw dataset from `sklearn`.
5.  **Run preprocessing & training**: Executes `data_preprocessing.py` and `train.py` to regenerate artifacts.
6.  **Run test suite**: Runs `pytest` to verify the pipeline.
7.  **Build Docker image**: Executes `docker build` with the `ci` tag to verify container builds.

---

## 12. Project Structure
```text
mlops-capstone/
├── .dvc/                   # DVC local storage configuration
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions workflow
├── data/
│   ├── raw/
│   │   ├── breast_cancer.csv.dvc # DVC pointer for raw dataset
│   │   └── .gitignore      # Keeps raw CSV out of Git
│   └── processed/
│       └── .gitignore      # Keeps split CSVs out of Git
├── models/
│   └── .gitkeep            # Tracks directory structure
├── src/
│   ├── app.py              # FastAPI application server
│   ├── data_preprocessing.py # Dataset split and scaling pipeline
│   ├── mlflow_tracking.py  # MLflow tracking logging utils
│   ├── train.py            # Model training & comparison orchestrator
│   └── utils.py            # YAML configuration loader & DB path re-mapper
├── tests/
│   ├── test_api.py         # Test FastAPI validation/health
│   ├── test_mlflow.py      # Test SQLite DB experiment registry
│   ├── test_preprocessing.py # Test data splits/leakage
│   └── test_training.py    # Test predictions & metadata integrity
├── .dockerignore           # Excludes development artifacts from Docker
├── .dvcignore
├── .gitignore              # Ignores venv, cache, mlflow.db, and model joblibs
├── Dockerfile              # Container building instruction recipe
├── dvc.yaml                # Preprocessing pipeline structure
├── dvc.lock                # Locked states for outputs
├── params.yaml             # Shared hyperparameters
└── requirements.txt        # Production-grade package lists
```

---

## 13. Setup & Execution Instructions

### A. Environment Initialization
1.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    # Windows activation
    .\venv\Scripts\activate
    # Linux/Mac activation
    source venv/bin/activate
    ```
2.  Install dependencies:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

### B. Reproduce Pipeline Stages
1.  Generate raw dataset:
    Run the inline dataset preparation snippet:
    ```bash
    python -c "import os, pandas as pd; from sklearn.datasets import load_breast_cancer; os.makedirs('data/raw', exist_ok=True); load_breast_cancer(as_frame=True).frame.to_csv('data/raw/breast_cancer.csv', index=False)"
    ```
2.  Run DVC preprocess step:
    ```bash
    dvc repro
    ```
3.  Train and register models:
    ```bash
    python src/train.py
    ```

### C. Inspection and Deployment
1.  Launch MLflow UI:
    ```bash
    mlflow ui --backend-store-uri sqlite:///mlflow.db
    ```
    Access UI at: [http://localhost:5000](http://localhost:5000)
2.  Launch FastAPI API server:
    ```bash
    uvicorn src.app:app --reload
    ```
    Access interactive documentation at: [http://localhost:8000/docs](http://localhost:8000/docs)
3.  Run entire test suite:
    ```bash
    pytest
    ```
