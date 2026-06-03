from fastapi import FastAPI
import pandas as pd

from api.model_loader import load_model
from ingestion.stock_data import fetch_stock_data
from processing.feature_engineering import add_features

app = FastAPI()

model = load_model()


@app.get("/")
def home():
    return {"message": "AI Market Intelligence API Running"}


@app.get("/predict/{ticker}")
def predict(ticker: str):

    df = fetch_stock_data(ticker)
    df = add_features(df)
    df = df.dropna()

    X = df.drop(columns=["label"], errors="ignore")

    proba = model.predict_proba(X)[:, 1]

    latest = proba[-1]

    if latest > 0.70:
        signal = "BUY"
    elif latest < 0.30:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "ticker": ticker,
        "signal": signal,
        "confidence": float(latest)
    }