import json
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.config import read_yaml

class ModelEvaluation:

    def __init__(self):

        config = read_yaml("config/config.yaml")

        self.metrics_path = config["evaluation"]["metrics_path"]

    def initiate_model_evaluation(
            self,
            model,
            X_test,
            y_test):

        prediction = model.predict(X_test)

        mae = mean_absolute_error(y_test, prediction)

        mse = mean_squared_error(y_test, prediction)

        rmse = mse ** 0.5

        r2 = r2_score(y_test, prediction)

        metrics = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        }

        with open(self.metrics_path, "w") as file:
            json.dump(metrics, file, indent=4)

        return metrics