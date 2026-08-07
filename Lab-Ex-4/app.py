from fastapi import FastAPI
from pydantic import BaseModel
import joblib

##Load model
model = joblib.load("model.joblib")

CLASS_NAMES = ["setosa", "versicolor", "virginica"]

##Create FastAPI app
app = FastAPI(
    title="Iris Classification API",
    description="Predict Iris flower species using a trained ML model",
    version="1.0"
)


##Request schema
class IrisFeatures(BaseModel):
    features: list[float]


@app.get("/")
def home():
    return {
        "message": "Welcome to the Iris Classification API!",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: IrisFeatures):

    prediction = model.predict([data.features])[0]
    probabilities = model.predict_proba([data.features])[0]

    return {
        "predicted_class": CLASS_NAMES[prediction],
        "confidence": round(float(max(probabilities)), 4),
        "probabilities": {
            CLASS_NAMES[i]: round(float(probabilities[i]), 4)
            for i in range(len(CLASS_NAMES))
        }
    }