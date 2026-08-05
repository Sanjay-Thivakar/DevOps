import os

import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from src.config import read_yaml

class DataIngestion:

    def __init__(self):

        config = read_yaml("config/config.yaml")

        self.train_path = config["data_ingestion"]["train_path"]
        self.test_path = config["data_ingestion"]["test_path"]

    def initiate_data_ingestion(self):
        california = fetch_california_housing()
        df = pd.DataFrame(
            california.data,
            columns=california.feature_names
        )

        df["Price"] = california.target
        os.makedirs("artifacts", exist_ok=True)

        train_set, test_set = train_test_split(
                            df,
                            test_size=0.2,
                            random_state=42
                        )

        train_set.to_csv(
            self.train_path,
            index=False
        )

        test_set.to_csv(
            self.test_path,
            index=False
        )

        return self.train_path, self.test_path