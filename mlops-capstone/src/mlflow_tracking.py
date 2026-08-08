import os
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from utils import logger

def configure_mlflow(tracking_uri: str, experiment_name: str):
    """
    Sets the tracking URI and configures the MLflow experiment.
    """
    logger.info(f"Configuring MLflow: tracking_uri={tracking_uri}, experiment={experiment_name}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

def log_model_run(run_name: str, model, params: dict, metrics: dict, X_train: pd.DataFrame) -> str:
    """
    Runs a single MLflow experiment run, logging parameters, metrics, model artifacts,
    model signatures, and input examples.
    
    Returns:
        str: The run ID of the completed MLflow run.
    """
    logger.info(f"Starting MLflow run for: {run_name}")
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        
        # 1. Log parameters
        # Clean parameter keys and log
        clean_params = {str(k): str(v) for k, v in params.items()}
        mlflow.log_params(clean_params)
        
        # 2. Log metrics
        clean_metrics = {str(k): float(v) for k, v in metrics.items()}
        mlflow.log_metrics(clean_metrics)
        
        # 3. Infer signature and define input example
        predictions = model.predict(X_train.head(5))
        signature = infer_signature(X_train, predictions)
        input_example = X_train.head(1)
        
        # 4. Log model artifact
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )
        
        logger.info(f"MLflow run {run_name} completed. Run ID: {run_id}")
        return run_id

def register_best_model(run_id: str, registered_model_name: str) -> tuple[str, str]:
    """
    Registers the best-performing model from an MLflow run in the Model Registry.
    Assigns the 'champion' alias to the registered version.
    
    Returns:
        tuple[str, str]: Registered model version number and alias.
    """
    model_uri = f"runs:/{run_id}/model"
    logger.info(f"Registering model from {model_uri} as '{registered_model_name}'...")
    
    # Register the model
    model_version_details = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
    version = str(model_version_details.version)
    logger.info(f"Successfully registered model version {version}")
    
    # Assign alias "champion"
    client = MlflowClient()
    alias = "champion"
    try:
        # set_registered_model_alias is the standard API in recent MLflow versions
        client.set_registered_model_alias(
            name=registered_model_name,
            alias=alias,
            version=version
        )
        logger.info(f"Assigned alias '{alias}' to model version {version}")
    except Exception as e:
        logger.warning(f"Could not assign alias using client.set_registered_model_alias: {e}. Trying alternative methods.")
        try:
            # Fallback for alternative or older MLflow client APIs
            client.set_model_version_tag(
                name=registered_model_name,
                version=version,
                key="alias",
                value=alias
            )
            logger.info(f"Tagged model version {version} with alias='{alias}' as fallback.")
        except Exception as tag_err:
            logger.error(f"Failed to assign alias or tag to model version: {tag_err}")
            raise tag_err
            
    return version, alias
