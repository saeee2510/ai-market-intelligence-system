import joblib

MODEL_PATH = "ml/models/xgboost_model.pkl"

def load_model():
    return joblib.load(MODEL_PATH)