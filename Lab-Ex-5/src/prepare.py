"""Stage 1: clean the Boston Housing data and create a scaled train/test split."""
import yaml
import pandas as pd
from pathlib import Path
import joblib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

OUT_DIR = Path("data")
RAW_DATA_PATH = OUT_DIR / "raw" / "HousingData.csv"
TARGET_COLUMN = "MEDV"


def load_dataset() -> pd.DataFrame:
    """Load the notebook's Boston Housing CSV from the repository."""
    df = pd.read_csv(RAW_DATA_PATH)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in {RAW_DATA_PATH}")
    return df


def main():
    params = yaml.safe_load(open("params.yaml"))["prepare"]
    OUT_DIR.mkdir(exist_ok=True)

    df = load_dataset()
    # Match the notebook: fill missing numeric values with medians, then remove
    # duplicate rows before splitting the data.
    imputer = SimpleImputer(strategy="median")
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    df = df.drop_duplicates().reset_index(drop=True)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    train_df, test_df = train_test_split(
        pd.concat([X, y], axis=1),
        test_size=params["test_size"],
        random_state=params["random_state"],
    )

    scaler = StandardScaler()
    feature_columns = list(X.columns)
    X_train = scaler.fit_transform(train_df[feature_columns])
    X_test = scaler.transform(test_df[feature_columns])
    pd.DataFrame(X_train, columns=feature_columns).assign(**{TARGET_COLUMN: train_df[TARGET_COLUMN].to_numpy()}).to_csv(OUT_DIR / "train.csv", index=False)
    pd.DataFrame(X_test, columns=feature_columns).assign(**{TARGET_COLUMN: test_df[TARGET_COLUMN].to_numpy()}).to_csv(OUT_DIR / "test.csv", index=False)
    joblib.dump(scaler, OUT_DIR / "scaler.joblib")
    print(f"Wrote {len(train_df)} train rows, {len(test_df)} test rows")


if __name__ == "__main__":
    main()
