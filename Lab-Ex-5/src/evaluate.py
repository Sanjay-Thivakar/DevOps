"""Stage 3: evaluate the notebook's regressors and apply an R-squared gate."""
import sys
import json
import yaml
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    params = yaml.safe_load(open("params.yaml"))["evaluate"]

    test_df = pd.read_csv("data/test.csv")
    X_test = test_df.drop(columns=["MEDV"])
    y_test = test_df["MEDV"]
    metrics = {}
    for name in ("linear_regression", "decision_tree", "random_forest"):
        predictions = joblib.load(f"model/{name}.joblib").predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        metrics[name] = {"mae": mean_absolute_error(y_test, predictions), "mse": mse, "rmse": mse ** 0.5, "r2": r2_score(y_test, predictions)}

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))

    deployed_r2 = metrics["random_forest"]["r2"]
    if deployed_r2 < params["min_r2"]:
        print(
            f"FAIL: Random Forest R2 {deployed_r2:.4f} "
            f"is below gate {params['min_r2']}"
        )
        sys.exit(1)

    print("PASS: Random Forest cleared the R2 quality gate")


if __name__ == "__main__":
    main()
