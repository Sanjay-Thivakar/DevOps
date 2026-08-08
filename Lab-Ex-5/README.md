# Boston Housing MLOps CI/CD

This project adapts the accompanying Boston Housing notebook into a reproducible CI/CD pipeline. Every push to `main` cleans the supplied `HousingData.csv`, trains three regressors, evaluates them, and deploys the notebook's final Random Forest model only when it passes an R-squared quality gate.

## Pipeline

1. `prepare.py` reads `data/raw/HousingData.csv`, median-imputes missing values, removes duplicates, creates an 80/20 split with seed 42, and standardizes the predictors.
2. `train.py` fits Linear Regression, Decision Tree Regressor, and a 200-tree Random Forest Regressor—the same three models used in the notebook.
3. `evaluate.py` records MAE, MSE, RMSE, and R-squared for all three models. The Random Forest is preserved as `model/model.joblib`, matching the notebook's final prediction and feature-importance model.
4. The GitHub Actions quality gate checks `evaluate.min_r2` in `params.yaml`; only a passing run can deploy the model to Hugging Face Hub.

## Local use

```bash
pip install -r requirements.txt
python src/prepare.py
python src/train.py
python src/evaluate.py
cat metrics.json
```

## GitHub Actions deployment

Set these repository values before pushing to `main`:

- Secret: `HF_TOKEN` — a Hugging Face token with write permission.
- Variable: `HF_REPO_ID` — target model repository, for example `your-user/boston-housing-rf`.

The workflow runs tests on pull requests and pushes. On a successful push to `main`, it trains and evaluates again, then uploads the Random Forest, scaler, feature list, and model card to Hugging Face.

## Quality threshold

Edit `evaluate.min_r2` in `params.yaml` to change the minimum acceptable Random Forest R-squared score.
