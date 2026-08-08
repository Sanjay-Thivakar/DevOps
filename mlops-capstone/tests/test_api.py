import os
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from src.app import app
import src.app as app_module

@pytest.fixture(scope="module")
def client():
    # Utilizing TestClient as a context manager to trigger lifespan events (startup/shutdown)
    with TestClient(app) as c:
        yield c

def test_root_endpoint(client):
    """
    Verify GET / returns 200 and describes application metadata.
    """
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Breast Cancer Classification API" in data["message"]
    assert data["version"] == "1.0"
    assert "health" in data["endpoints"]
    assert "predict" in data["endpoints"]

def test_health_endpoint(client):
    """
    Verify GET /health returns 200 and reports model/scaler are loaded.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["scaler_loaded"] is True

def test_predict_endpoint_valid_sample(client):
    """
    Verify POST /predict with a valid 30-feature sample returns 200 and a correct prediction response structure.
    """
    # Load a raw sample from raw dataset
    raw_data_path = "data/raw/breast_cancer.csv"
    assert os.path.exists(raw_data_path), f"Raw dataset not found at: {raw_data_path}"
    df = pd.read_csv(raw_data_path)
    
    # Extract features only for the first sample
    features = df.drop(columns=["target"]).iloc[0].tolist()
    assert len(features) == 30
    
    payload = {"features": features}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200, f"Predict failed: {response.text}"
    data = response.json()
    
    # Validate structure
    assert "predicted_class" in data
    assert "predicted_label" in data
    assert "confidence" in data
    assert "probabilities" in data
    
    # Validate values
    assert data["predicted_class"] in ["malignant", "benign"]
    assert data["predicted_label"] in [0, 1]
    assert 0.0 <= data["confidence"] <= 1.0
    
    probs = data["probabilities"]
    assert "malignant" in probs
    assert "benign" in probs
    assert 0.0 <= probs["malignant"] <= 1.0
    assert 0.0 <= probs["benign"] <= 1.0
    assert abs(probs["malignant"] + probs["benign"] - 1.0) < 1e-4

def test_predict_endpoint_invalid_count(client):
    """
    Verify POST /predict with invalid feature lengths (e.g. 29 or 31) returns 422 validation error.
    """
    # Case 1: 29 features
    payload_29 = {"features": [1.0] * 29}
    response_29 = client.post("/predict", json=payload_29)
    assert response_29.status_code == 422
    assert "exactly 30 numerical values" in response_29.text
    
    # Case 2: 31 features
    payload_31 = {"features": [1.0] * 31}
    response_31 = client.post("/predict", json=payload_31)
    assert response_31.status_code == 422
    assert "exactly 30 numerical values" in response_31.text

def test_api_uses_registered_mlflow_model(client):
    """
    Verify the API is loaded with the registered MLflow model and not a mock/default.
    """
    assert app_module.model is not None, "Model is not loaded."
    # The loaded model should be a LogisticRegression model (our current champion)
    from sklearn.linear_model import LogisticRegression
    assert isinstance(app_module.model, LogisticRegression), "Loaded model is not a LogisticRegression instance."

def test_scaler_used_correctly(client):
    """
    Verify the scaler is successfully loaded and capable of transforming.
    """
    assert app_module.scaler is not None, "Scaler is not loaded."
    assert hasattr(app_module.scaler, "transform"), "Loaded scaler does not support transform()."
    assert hasattr(app_module.scaler, "mean_"), "Scaler has not been fitted."
