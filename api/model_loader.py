import joblib
import os

MODEL_PATH = "ml/models/xgboost_model.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Train model first.")

    model = joblib.load(MODEL_PATH)
    return model