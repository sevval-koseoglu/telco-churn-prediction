import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
import pickle
from src.preprocess import preprocess_input

app = FastAPI(title="Telco Churn Prediction API", version="1.0.0")

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)


@app.get("/")
def home():
    return {"message": "Telco Churn Prediction API is running"}


@app.post("/predict")
def predict(data: dict):
    df = preprocess_input(data, feature_columns)

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "churn_probability": round(float(probability), 4)
    }