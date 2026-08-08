import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from utils import load_data, read_params, logger

def preprocess_data(raw_data_path: str, processed_dir: str, models_dir: str, config_path: str):
    """
    Main function to preprocess the breast cancer dataset.
    """
    # Load parameters
    params = read_params(config_path)
    prep_params = params.get("preprocessing", {})
    test_size = prep_params.get("test_size", 0.2)
    random_state = prep_params.get("random_state", 42)
    
    logger.info(f"Using parameters - test_size: {test_size}, random_state: {random_state}")
    
    # Load dataset
    df = load_data(raw_data_path)
    
    # Separate features and target
    target_col = "target"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split into train and test sets (stratified to maintain class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Split completed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Scale numerical features (avoiding data leakage by fitting only on train)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    logger.info("Features scaled successfully using StandardScaler.")
    
    # Ensure directories exist
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Save processed datasets
    X_train_scaled.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)
    
    # Save scaler artifact
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved scaler to {scaler_path}")
    logger.info("Preprocessing step completed successfully.")

if __name__ == "__main__":
    # Resolve absolute paths relative to project root
    # Note: executing directly with `python src/data_preprocessing.py`
    # works since current working directory is the project root.
    preprocess_data(
        raw_data_path="data/raw/breast_cancer.csv",
        processed_dir="data/processed",
        models_dir="models",
        config_path="params.yaml"
    )
