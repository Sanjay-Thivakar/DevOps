"""Stage 2: train the three regression models used in the notebook."""
import yaml
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

MODEL_DIR = Path("model")


def main():
    params = yaml.safe_load(open("params.yaml"))["train"]
    MODEL_DIR.mkdir(exist_ok=True)

    train_df = pd.read_csv("data/train.csv")
    X_train = train_df.drop(columns=["MEDV"])
    y_train = train_df["MEDV"]

    models = {
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(random_state=params["random_state"]),
        "random_forest": RandomForestRegressor(
            n_estimators=params["n_estimators"], random_state=params["random_state"]
        ),
    }
    for model in models.values():
        model.fit(X_train, y_train)

    joblib.dump(models["random_forest"], MODEL_DIR / "model.joblib")
    for name, model in models.items():
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")

    ##Save feature names alongside the model, needed for HF model card / inference
    with open(MODEL_DIR / "features.json", "w") as f:
        json.dump(list(X_train.columns), f)

    print("Trained Linear Regression, Decision Tree, and Random Forest models")


if __name__ == "__main__":
    main()
