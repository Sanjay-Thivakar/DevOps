import joblib

from sklearn.linear_model import LinearRegression

from src.config import read_yaml

class ModelTrainer:

    def __init__(self):

        config = read_yaml("config/config.yaml")

        self.model_path = config["model_trainer"]["model_path"]

    def initiate_model_training(
            self,
            X_train,
            X_test,
            y_train,
            y_test):

        model = LinearRegression()

        model.fit(X_train, y_train)

        joblib.dump(model, self.model_path)

        return model