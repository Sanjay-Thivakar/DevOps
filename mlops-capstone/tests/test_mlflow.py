import os
import pytest
import mlflow
from mlflow.tracking import MlflowClient

@pytest.fixture(scope="module")
def mlflow_client():
    from src.utils import fix_mlflow_db
    fix_mlflow_db()
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    return MlflowClient()


def test_experiment_exists(mlflow_client):
    """
    Verify that the MLflow experiment 'Breast Cancer Classification' exists.
    """
    experiment = mlflow_client.get_experiment_by_name("Breast Cancer Classification")
    assert experiment is not None, "Experiment 'Breast Cancer Classification' was not found."
    assert experiment.name == "Breast Cancer Classification"

def test_three_model_runs_created(mlflow_client):
    """
    Verify that the three model runs (LogisticRegression, RandomForestClassifier, SVC)
    exist in the experiment.
    """
    experiment = mlflow_client.get_experiment_by_name("Breast Cancer Classification")
    runs = mlflow_client.search_runs(experiment_ids=[experiment.experiment_id])
    
    assert len(runs) >= 3, f"Expected at least 3 runs, but found {len(runs)}."
    
    run_names = [run.data.tags.get("mlflow.runName") for run in runs]
    assert "LogisticRegression" in run_names, "LogisticRegression run not logged."
    assert "RandomForestClassifier" in run_names, "RandomForestClassifier run not logged."
    assert "SVC" in run_names, "SVC run not logged."

def test_runs_contain_metrics_and_parameters(mlflow_client):
    """
    Verify that each run contains the expected metrics, parameters, and model artifact logging.
    """
    experiment = mlflow_client.get_experiment_by_name("Breast Cancer Classification")
    runs = mlflow_client.search_runs(experiment_ids=[experiment.experiment_id])
    
    expected_metrics = {"Accuracy", "Precision", "Recall", "F1", "ROC-AUC"}
    
    for run in runs:
        run_name = run.data.tags.get("mlflow.runName")
        if run_name not in ["LogisticRegression", "RandomForestClassifier", "SVC"]:
            continue
            
        metrics = run.data.metrics
        params = run.data.params
        
        # Check metrics
        for metric in expected_metrics:
            assert metric in metrics, f"Metric '{metric}' missing in run '{run_name}'."
            
        # Check parameters
        assert "model_name" in params, f"Parameter 'model_name' missing in run '{run_name}'."
        assert params["model_name"] == run_name
        
        # Verify model artifacts were logged (artifact_uri exists)
        assert run.info.artifact_uri is not None, f"Artifact URI is missing in run '{run_name}'."

def test_registered_model_exists(mlflow_client):
    """
    Verify that the best model is registered under the correct name
    and has at least one version.
    """
    reg_models = mlflow_client.search_registered_models(filter_string="name = 'BreastCancerClassifier'")
    assert len(reg_models) > 0, "Registered model 'BreastCancerClassifier' does not exist."
    
    reg_model = reg_models[0]
    assert reg_model.name == "BreastCancerClassifier"
    assert len(reg_model.latest_versions) >= 1, "Registered model has no versions."

def test_registered_model_corresponds_to_best_run(mlflow_client):
    """
    Verify that the registered model corresponds to the best run (LogisticRegression)
    and has the 'champion' alias or tag.
    """
    reg_model = mlflow_client.get_registered_model("BreastCancerClassifier")
    latest_version = reg_model.latest_versions[0]
    
    # Get run details for this version
    run = mlflow_client.get_run(latest_version.run_id)
    run_name = run.data.tags.get("mlflow.runName")
    
    # Best model must be LogisticRegression
    assert run_name == "LogisticRegression", f"Registered model version run was '{run_name}', expected 'LogisticRegression'."
    
    # Check for champion alias or fallback tag
    has_champion_alias = "champion" in reg_model.aliases or latest_version.tags.get("alias") == "champion"
    assert has_champion_alias, "Alias 'champion' was not assigned to the registered model version."
