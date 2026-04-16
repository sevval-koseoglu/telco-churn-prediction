from fastapi import FastAPI
import pandas as pd
import pickle

app = FastAPI()

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Telco Churn Prediction API is running"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"prediction": int(prediction[0])}