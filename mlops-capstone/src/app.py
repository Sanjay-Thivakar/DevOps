import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from contextlib import asynccontextmanager
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import logger, read_params


# Global variables to store the loaded model and scaler
model = None
scaler = None

# Configuration variables (load defaults or read from env)
params = {}
if os.path.exists("params.yaml"):
    try:
        params = read_params("params.yaml")
    except Exception as e:
        logger.warning(f"Could not load params.yaml: {e}")

mlflow_params = params.get("mlflow", {})
default_tracking_uri = mlflow_params.get("tracking_uri", "sqlite:///mlflow.db")
default_model_name = mlflow_params.get("registered_model_name", "BreastCancerClassifier")
default_model_alias = "champion"
default_scaler_path = "models/scaler.joblib"

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", default_tracking_uri)
MODEL_NAME = os.getenv("MODEL_NAME", default_model_name)
MODEL_ALIAS = os.getenv("MODEL_ALIAS", default_model_alias)
SCALER_PATH = os.getenv("SCALER_PATH", default_scaler_path)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan manager to handle model and scaler loading at startup.
    """
    global model, scaler
    
    # Repair MLflow DB paths dynamically to match container/local filesystem paths
    from utils import fix_mlflow_db
    try:
        fix_mlflow_db()
    except Exception as e:
        logger.error(f"Lifespan startup aborted due to DB repair failure: {e}")
        raise e
    
    # 1. Load Scaler

    logger.info(f"Loading scaler from: {SCALER_PATH}")
    if not os.path.exists(SCALER_PATH):
        logger.error(f"Scaler file not found at: {SCALER_PATH}")
        raise FileNotFoundError(f"Scaler file not found at: {SCALER_PATH}")
    try:
        scaler = joblib.load(SCALER_PATH)
        logger.info("Scaler loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load scaler: {e}")
        raise RuntimeError(f"Failed to load scaler: {e}")
        
    # 2. Load MLflow Model
    logger.info(f"Loading MLflow model: {MODEL_NAME}@{MODEL_ALIAS} using tracking URI: {TRACKING_URI}")
    try:
        mlflow.set_tracking_uri(TRACKING_URI)
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        # Using mlflow.sklearn to load the actual scikit-learn model object (supports predict_proba)
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"MLflow model '{MODEL_NAME}' (alias '{MODEL_ALIAS}') loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model from MLflow Model Registry: {e}")
        raise RuntimeError(f"Failed to load model from MLflow Model Registry: {e}")
        
    yield
    
    # Clean up on shutdown
    logger.info("Shutting down API service.")

# Initialize FastAPI app with metadata and lifespan
app = FastAPI(
    title="Breast Cancer Classification API",
    description="Production-style API serving a Breast Cancer Classification model registered in MLflow.",
    version="1.0",
    lifespan=lifespan
)

# Pydantic request model
class PredictRequest(BaseModel):
    features: list[float] = Field(
        ...,
        description="A list containing exactly 30 numerical breast cancer characteristics (measurements)."
    )

    @field_validator("features")
    @classmethod
    def check_features_count(cls, v):
        if len(v) != 30:
            raise ValueError(f"Features list must contain exactly 30 numerical values. Got {len(v)} values.")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": [
                    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
                    1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
                ]
            }
        }
    }

# Pydantic response models
class PredictResponse(BaseModel):
    predicted_class: str
    predicted_label: int
    confidence: float
    probabilities: dict[str, float]

@app.get("/", tags=["General"])
async def root():
    """
    Root endpoint containing API name, version, and endpoints list.
    """
    return {
        "message": "Breast Cancer Classification API",
        "version": "1.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict"
        }
    }

@app.get("/health", tags=["General"])
async def health():
    """
    Health check endpoint reporting API, model, and scaler status.
    """
    model_status = model is not None
    scaler_status = scaler is not None
    
    if not model_status or not scaler_status:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "model_loaded": model_status,
                "scaler_loaded": scaler_status
            }
        )
        
    return {
        "status": "ok",
        "model_loaded": model_status,
        "scaler_loaded": scaler_status
    }

@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest):
    """
    Validates input features, standardizes them using scaler, and predicts class/probabilities.
    """
    global model, scaler
    
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model or scaler not initialized. The service is currently unavailable."
        )
        
    try:
        # 1. Convert to 2D numpy array
        features_array = np.array([request.features])
        
        # 2. Re-create DataFrame with feature names to match StandardScaler fit step and suppress warnings
        feature_names = getattr(scaler, "feature_names_in_", None)
        if feature_names is not None:
            df_features = pd.DataFrame(features_array, columns=feature_names)
        else:
            df_features = features_array
            
        # 3. Transform features (do NOT fit)
        scaled_features = scaler.transform(df_features)
        
        # Wrap back into DataFrame to preserve feature names for the classifier
        if feature_names is not None:
            df_scaled = pd.DataFrame(scaled_features, columns=feature_names)
        else:
            df_scaled = scaled_features
            
        # 4. Generate prediction
        prediction = int(model.predict(df_scaled)[0])
        
        # Map labels (0: malignant, 1: benign)
        class_mapping = {0: "malignant", 1: "benign"}
        predicted_class = class_mapping.get(prediction, "unknown")
        
        # 5. Extract probabilities with safe fallback if predict_proba is not supported
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df_scaled)[0]

            confidence = float(probs[prediction])
            probabilities_dict = {
                "malignant": round(float(probs[0]), 4),
                "benign": round(float(probs[1]), 4)
            }
        else:
            # Fallback if model doesn't support predict_proba
            logger.warning("predict_proba is not supported by the loaded model. Using prediction-based confidence.")
            confidence = 1.0
            probabilities_dict = {
                "malignant": 1.0 if prediction == 0 else 0.0,
                "benign": 1.0 if prediction == 1 else 0.0
            }
            
        return PredictResponse(
            predicted_class=predicted_class,
            predicted_label=prediction,
            confidence=round(confidence, 4),
            probabilities=probabilities_dict
        )
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during prediction: {str(e)}"
        )
