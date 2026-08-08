import os
import pandas as pd
import pytest

def test_processed_files_exist():
    """
    Verify that the processed datasets exist.
    """
    processed_dir = "data/processed"
    files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
    for f in files:
        path = os.path.join(processed_dir, f)
        assert os.path.exists(path), f"Processed file {path} does not exist."

def test_shape_compatibility():
    """
    Verify that train and test feature/target shapes are compatible.
    """
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")
    
    # 1. Feature counts must match
    assert X_train.shape[1] == X_test.shape[1], "Train and test feature counts do not match."
    assert X_train.shape[1] == 30, "Feature count should be 30 for Breast Cancer dataset."
    
    # 2. Number of samples must match between X and y
    assert X_train.shape[0] == y_train.shape[0], "Train feature and target length mismatch."
    assert X_test.shape[0] == y_test.shape[0], "Test feature and target length mismatch."
    
    # 3. Total split size matches original (569)
    total_samples = X_train.shape[0] + X_test.shape[0]
    assert total_samples == 569, f"Expected 569 total samples, got {total_samples}."

def test_target_values_are_valid():
    """
    Verify that target values contain only binary classes {0, 1}.
    """
    y_train = pd.read_csv("data/processed/y_train.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")
    
    # Assert values are only 0 or 1
    assert set(y_train["target"].unique()).issubset({0, 1}), "y_train contains invalid targets."
    assert set(y_test["target"].unique()).issubset({0, 1}), "y_test contains invalid targets."
    
    # Stratification check (both classes present in both splits)
    assert len(y_train["target"].unique()) == 2
    assert len(y_test["target"].unique()) == 2

def test_no_data_leakage():
    """
    Verify there is no data leakage between train and test datasets.
    """
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    
    # 1. Row-wise check: No identical feature rows in train and test
    # Merge on all features and check for overlap
    merged = pd.merge(X_train, X_test, how="inner")
    # Due to float precision, check if duplicate rows are extremely low or 0
    assert len(merged) == 0, f"Detected {len(merged)} duplicate rows across train and test sets!"
    
    # 2. Scaling check: StandardScaler must be fitted on train, not test.
    # X_train scaled should have mean ~ 0 and std ~ 1.
    # X_test scaled mean and std will not be exactly 0 and 1.
    for col in X_train.columns:
        mean_train = X_train[col].mean()
        std_train = X_train[col].std(ddof=0)
        
        mean_test = X_test[col].mean()
        std_test = X_test[col].std(ddof=0)
        
        # Train should be perfectly standardized
        assert abs(mean_train) < 1e-7, f"Train feature '{col}' mean is not standardized."
        assert abs(std_train - 1.0) < 1e-7, f"Train feature '{col}' std is not standardized."
        
        # Test should be transformed but not yield exact 0/1 stats (since it didn't fit on test)
        # Verify they are not identical to train stats
        assert mean_train != mean_test or std_train != std_test, "Test set scaling properties are identical to train."
