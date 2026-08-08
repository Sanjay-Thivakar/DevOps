import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from utils import read_params, logger
from mlflow_tracking import configure_mlflow, log_model_run, register_best_model


def load_processed_data(processed_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads preprocessed training and test CSV files.
    """
    logger.info(f"Loading preprocessed datasets from {processed_dir}...")
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv"))["target"]
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv"))["target"]
    return X_train, X_test, y_train, y_test

def init_models(random_state: int) -> dict:
    """
    Initializes the three required models with a reproducible random state.
    """
    logger.info("Initializing classification models...")
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=random_state),
        "RandomForestClassifier": RandomForestClassifier(random_state=random_state),
        "SVC": SVC(probability=True, random_state=random_state)
    }

def train_and_evaluate(models: dict, X_train: pd.DataFrame, y_train: pd.Series, 
                       X_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict, dict]:
    """
    Trains each model and computes metrics on the test dataset.
    """
    metrics_results = {}
    trained_models = {}
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        # Train model
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate probabilities/decision scores for ROC-AUC
        # All three models support predict_proba because we set probability=True for SVC
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        metrics_results[name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC-AUC": roc_auc
        }
        logger.info(f"{name} Metrics - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
    return trained_models, metrics_results

def compare_and_save_results(metrics_results: dict, models_dir: str) -> tuple[pd.DataFrame, str, float]:
    """
    Compares model metrics, prints a table, saves it, and identifies the best model.
    """
    # Create comparison dataframe
    records = []
    for model_name, metrics in metrics_results.items():
        record = {"Model": model_name}
        record.update(metrics)
        records.append(record)
        
    df_comparison = pd.DataFrame(records)
    
    # Sort by F1-score descending
    df_comparison = df_comparison.sort_values(by="F1", ascending=False).reset_index(drop=True)
    
    # Print comparison table to terminal
    print("\n" + "="*50)
    print("                MODEL COMPARISON")
    print("="*50)
    print(df_comparison.to_string(index=False))
    print("="*50 + "\n")
    
    # Ensure models directory exists
    os.makedirs(models_dir, exist_ok=True)
    
    # Save comparison to CSV
    csv_path = os.path.join(models_dir, "model_comparison.csv")
    df_comparison.to_csv(csv_path, index=False)
    logger.info(f"Saved model comparison table to {csv_path}")
    
    # Best model is the first row (since sorted by F1 desc)
    best_model_name = df_comparison.iloc[0]["Model"]
    best_f1_score = df_comparison.iloc[0]["F1"]
    
    print(f"Best Model: {best_model_name} (F1-score: {best_f1_score:.4f})\n")
    return df_comparison, best_model_name, best_f1_score

def save_artifacts(trained_models: dict, best_model_name: str, best_f1_score: float, 
                   feature_count: int, random_state: int, models_dir: str):
    """
    Saves the best model and corresponding metadata.
    """
    # Save best model joblib
    best_model = trained_models[best_model_name]
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(best_model, best_model_path)
    logger.info(f"Saved best model artifact to {best_model_path}")
    
    # Save metadata json
    metadata = {
        "best_model": best_model_name,
        "selection_metric": "F1-score",
        "best_score": float(best_f1_score),
        "feature_count": int(feature_count),
        "class_names": ["malignant", "benign"],
        "training_random_state": int(random_state)
    }
    
    metadata_path = os.path.join(models_dir, "model_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Saved model metadata to {metadata_path}")

def main():
    # Load configuration
    params = read_params("params.yaml")
    train_params = params.get("training", {})
    random_state = train_params.get("random_state", 42)
    
    # MLflow Configuration
    mlflow_config = params.get("mlflow", {})
    experiment_name = mlflow_config.get("experiment_name", "Breast Cancer Classification")
    registered_model_name = mlflow_config.get("registered_model_name", "BreastCancerClassifier")
    tracking_uri = mlflow_config.get("tracking_uri", "sqlite:///mlflow.db")
    
    configure_mlflow(tracking_uri, experiment_name)
    
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data("data/processed")
    
    # Initialize models
    models = init_models(random_state)
    
    # Train and evaluate
    trained_models, metrics_results = train_and_evaluate(
        models, X_train, y_train, X_test, y_test
    )
    
    # Log runs to MLflow
    run_ids = {}
    for name, model in trained_models.items():
        model_params = model.get_params()
        # Ensure model name is tracked
        model_params["model_name"] = name
        if "random_state" in model_params:
            model_params["random_state"] = random_state
            
        run_id = log_model_run(
            run_name=name,
            model=model,
            params=model_params,
            metrics=metrics_results[name],
            X_train=X_train
        )
        run_ids[name] = run_id
    
    # Compare models
    _, best_model_name, best_f1_score = compare_and_save_results(metrics_results, "models")
    
    # Save best model & metadata locally
    save_artifacts(
        trained_models=trained_models,
        best_model_name=best_model_name,
        best_f1_score=best_f1_score,
        feature_count=X_train.shape[1],
        random_state=random_state,
        models_dir="models"
    )
    
    # Register the best model in MLflow Model Registry
    best_run_id = run_ids[best_model_name]
    register_best_model(best_run_id, registered_model_name)


if __name__ == "__main__":
    main()
