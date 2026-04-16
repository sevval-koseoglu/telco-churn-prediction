from fastapi import FastAPI
import pandas as pd
import pickle

app = FastAPI()

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Telco Churn Prediction API is running"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.drop("customerID", axis=1)

    df = pd.get_dummies(df, drop_first=True)

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "churn_probability": float(probability)
    }