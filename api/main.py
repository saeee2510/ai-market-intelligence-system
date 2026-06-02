from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from api.model_loader import load_model

app = FastAPI(title="AI Market Intelligence API")

model = load_model()


# =====================================================
# INPUT SCHEMA
# =====================================================
class Features(BaseModel):
    features: list  # list of numeric features


# =====================================================
# 1. PREDICT SIGNAL
# =====================================================
@app.post("/predict")
def predict(data: Features):

    X = np.array(data.features).reshape(1, -1)

    pred = model.predict(X)[0]

    return {
        "signal": int(pred)
    }


# =====================================================
# 2. PROBABILITY (CONFIDENCE)
# =====================================================
@app.post("/proba")
def proba(data: Features):

    X = np.array(data.features).reshape(1, -1)

    prob = model.predict_proba(X)[0][1]

    return {
        "prob_up": float(prob)
    }


# =====================================================
# 3. ANALYSIS (SIMPLE EXPLANATION LAYER)
# =====================================================
@app.post("/analysis")
def analysis(data: Features):

    X = np.array(data.features).reshape(1, -1)

    prob = model.predict_proba(X)[0][1]

    if prob > 0.65:
        decision = "BUY"
    elif prob < 0.35:
        decision = "SELL"
    else:
        decision = "HOLD"

    return {
        "probability": float(prob),
        "decision": decision,
        "reason": "Based on model confidence thresholding"
    }