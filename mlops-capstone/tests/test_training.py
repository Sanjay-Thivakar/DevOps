import os
import json
import joblib
import pandas as pd
import pytest

MODELS_DIR = "models"
COMPARISON_FILE = os.path.join(MODELS_DIR, "model_comparison.csv")
METADATA_FILE = os.path.join(MODELS_DIR, "model_metadata.json")
BEST_MODEL_FILE = os.path.join(MODELS_DIR, "best_model.joblib")
PROCESSED_X_TEST = "data/processed/X_test.csv"

def test_comparison_file_exists():
    """
    Verify that models/model_comparison.csv exists.
    """
    assert os.path.exists(COMPARISON_FILE), f"Comparison file {COMPARISON_FILE} does not exist."

def test_comparison_contains_three_models():
    """
    Verify that three expected models are trained and listed.
    """
    df = pd.read_csv(COMPARISON_FILE)
    expected_models = {"LogisticRegression", "RandomForestClassifier", "SVC"}
    actual_models = set(df["Model"].unique())
    assert actual_models == expected_models, f"Expected models {expected_models}, got {actual_models}."

def test_required_metrics_exist():
    """
    Verify that the comparison file contains all required metrics as columns.
    """
    df = pd.read_csv(COMPARISON_FILE)
    required_cols = {"Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"}
    assert required_cols.issubset(df.columns), f"Missing columns. Expected {required_cols}, got {df.columns}."

def test_metadata_best_model_selection():
    """
    Verify that models/model_metadata.json exists and selected best model is valid.
    """
    assert os.path.exists(METADATA_FILE), f"Metadata file {METADATA_FILE} does not exist."
    
    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)
        
    required_keys = {"best_model", "selection_metric", "best_score", "feature_count", "class_names", "training_random_state"}
    assert required_keys.issubset(metadata.keys()), f"Missing metadata keys. Got {metadata.keys()}"
    
    expected_models = ["LogisticRegression", "RandomForestClassifier", "SVC"]
    assert metadata["best_model"] in expected_models, f"Best model {metadata['best_model']} is not one of {expected_models}."

def test_best_model_file_exists():
    """
    Verify that models/best_model.joblib exists.
    """
    assert os.path.exists(BEST_MODEL_FILE), f"Best model file {BEST_MODEL_FILE} does not exist."

def test_best_model_load_and_predict():
    """
    Verify that the best model can be loaded and used to make predictions on test data.
    """
    # Load model
    model = joblib.load(BEST_MODEL_FILE)
    assert model is not None, "Failed to load best model artifact."
    
    # Load sample test data
    assert os.path.exists(PROCESSED_X_TEST), f"Test features {PROCESSED_X_TEST} not found."
    X_test = pd.read_csv(PROCESSED_X_TEST)
    
    # Check shape / feature count
    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)
    assert X_test.shape[1] == metadata["feature_count"], "Test set feature count mismatch with metadata."
    
    # Make a prediction with one sample
    sample = X_test.iloc[[0]]
    prediction = model.predict(sample)
    
    # Verify shape and type
    assert len(prediction) == 1, "Expected single prediction output."
    
    # Prediction must belong to the valid target classes (0 or 1)
    assert prediction[0] in [0, 1], f"Invalid prediction class {prediction[0]}."
