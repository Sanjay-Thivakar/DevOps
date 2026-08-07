# 🏠 Boston Housing Regression with MLflow & DagsHub

This project demonstrates how to train, evaluate, and track multiple machine learning regression models on the **Boston Housing Dataset** using **MLflow** integrated with **DagsHub**.

## 📌 Objective

The objective of this project is to:

- Train multiple regression models on the Boston Housing dataset.
- Compare model performance using standard regression metrics.
- Track experiments, parameters, metrics, and models using **MLflow**.
- Visualize and manage experiment runs through **DagsHub**.

---

## 📂 Dataset

**Dataset:** `BostonHousing.csv`

The dataset contains housing-related features such as crime rate, number of rooms, property tax rate, etc., which are used to predict the median value of owner-occupied homes.

---

## 🤖 Models Implemented

The following regression models were trained and evaluated:

- **Linear Regression**
- **Random Forest Regressor**
- **XGBoost Regressor**

---

## 📊 Evaluation Metrics

Each model was evaluated using the following regression metrics:

- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

These metrics were automatically logged using **MLflow** for every experiment run.

---

## 🚀 Experiment Tracking

This project uses **MLflow** integrated with **DagsHub** to:

- Log model parameters
- Track evaluation metrics
- Store trained models
- Compare experiment runs
- Visualize experiment history

---

## 📈 Results

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| Linear Regression | ✅ Logged | ✅ Logged | ✅ Logged |
| Random Forest Regressor | ✅ Logged | ✅ Logged | ✅ Logged |
| XGBoost Regressor | ✅ Logged | ✅ Logged | ✅ Logged |

---

## 📁 Project Structure

```text
.
├── BostonHousing.csv
├── train.py
├── requirements.txt
├── README.md
└── mlruns/
```

---

## 🛠️ Technologies Used

- Python
- Scikit-learn
- XGBoost
- MLflow
- DagsHub
- Pandas
- NumPy

---

## 🔗 Repository Links

### GitHub Repository
https://github.com/Sanjay-Thivakar/DevOps

### DagsHub Experiments
https://dagshub.com/sanjaythivakar7/DevOps.mlflow/#/experiments

---

## 📌 Conclusion

This project demonstrates a complete machine learning experimentation workflow by combining **Scikit-learn**, **XGBoost**, **MLflow**, and **DagsHub**. It highlights how experiment tracking improves reproducibility, model comparison, and overall machine learning lifecycle management.