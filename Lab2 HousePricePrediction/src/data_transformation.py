import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.config import read_yaml

class DataTransformation:

    def __init__(self):

        config = read_yaml("config/config.yaml")

        self.train_path = config["data_ingestion"]["train_path"]
        self.test_path = config["data_ingestion"]["test_path"]

        self.preprocessor_path = config["data_transformation"]["preprocessor_path"]

    def initiate_data_transformation(self):

        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        X_train = train_df.drop("Price", axis=1)
        y_train = train_df["Price"]

        X_test = test_df.drop("Price", axis=1)
        y_test = test_df["Price"]

        preprocessor = Pipeline([
            ("scaler", StandardScaler())
        ])

        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

        joblib.dump(preprocessor, self.preprocessor_path)

        return (
            X_train,
            X_test,
            y_train,
            y_test,
            self.preprocessor_path
        )