import os
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads breast cancer dataset from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        pd.errors.EmptyDataError: If the file is empty.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found at: {file_path}")
        raise FileNotFoundError(f"File not found at: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded dataset from {file_path} with shape {df.shape}")
        return df
    except pd.errors.EmptyDataError as e:
        logger.error(f"The file at {file_path} is empty.")
        raise e
    except Exception as e:
        logger.error(f"An error occurred while loading data from {file_path}: {e}")
        raise e

def read_params(params_path: str = "params.yaml") -> dict:
    """
    Reads parameters from a YAML file.

    Parameters:
        params_path (str): Path to the YAML file.

    Returns:
        dict: Parsed parameters as a dictionary.
    """
    import yaml
    if not os.path.exists(params_path):
        logger.error(f"Parameter file not found at: {params_path}")
        raise FileNotFoundError(f"Parameter file not found at: {params_path}")
    
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        logger.info(f"Successfully read parameters from {params_path}")
        return params
    except Exception as e:
        logger.error(f"Error reading parameter file {params_path}: {e}")
        raise e

def fix_mlflow_db(db_path: str = "mlflow.db"):
    """
    Scans the SQLite MLflow DB and updates absolute artifact paths
    to match the current working directory. This makes the database
    fully portable across host machines, Docker containers, and CI.
    """
    import sqlite3
    if not os.path.exists(db_path):
        logger.warning(f"MLflow database not found at {db_path}, skipping path fix.")
        return

    # Use forward slashes for paths to match MLflow's file URI format
    current_dir = os.path.abspath(".").replace("\\", "/")
    current_uri_prefix = f"file:{current_dir}"

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Find the currently stored URI prefix by querying the experiments table
        cursor.execute("SELECT artifact_location FROM experiments LIMIT 1")
        row = cursor.fetchone()
        if not row or not row[0]:
            conn.close()
            return

        stored_uri = row[0]
        # Locate the '/mlruns' substring to split out the root directory path prefix
        idx = stored_uri.find("/mlruns")
        if idx == -1:
            conn.close()
            return

        stored_prefix = stored_uri[:idx]

        # If they already match, nothing to do
        if stored_prefix == current_uri_prefix:
            conn.close()
            return

        logger.info(f"Re-mapping MLflow DB paths. Replacing '{stored_prefix}' with '{current_uri_prefix}'")

        # Update absolute path entries in all relevant tables
        updates = [
            ("runs", "artifact_uri"),
            ("logged_models", "artifact_location"),
            ("experiments", "artifact_location"),
            ("model_versions", "storage_location")
        ]

        for table, column in updates:
            cursor.execute(
                f"UPDATE {table} SET {column} = REPLACE({column}, ?, ?)",
                (stored_prefix, current_uri_prefix)
            )

        conn.commit()
        conn.close()
        logger.info("MLflow DB paths re-mapped successfully.")
    except Exception as e:
        logger.error(f"Error while repairing MLflow database paths: {e}")
        # Raise here rather than silent pass so that build/startup failure is obvious
        raise e


