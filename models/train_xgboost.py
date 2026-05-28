import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def train_model(df):

    df = df.copy()

    # --------------------------------------------------
    # STEP 1 — CLEAN DATA
    # --------------------------------------------------
    df = df.dropna().reset_index(drop=True)

    # --------------------------------------------------
    # STEP 2 — REGIME FILTER (DO THIS FIRST)
    # --------------------------------------------------
    if "vol_regime" in df.columns:
        df = df[df["vol_regime"] > df["vol_regime"].quantile(0.3)]
        df = df.dropna().reset_index(drop=True)

    # --------------------------------------------------
    # STEP 3 — LABEL DISTRIBUTION CHECK
    # --------------------------------------------------
    print("\n📊 Label distribution (AFTER filtering):")
    print(df["label"].value_counts(normalize=True))

    # --------------------------------------------------
    # STEP 4 — SPLIT FEATURES / TARGET
    # --------------------------------------------------
    drop_cols = ["label", "future_return", "timestamp"]

    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df["label"]

    # --------------------------------------------------
    # STEP 5 — TRAIN / TEST SPLIT (TIME SERIES SAFE)
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    # --------------------------------------------------
    # STEP 6 — MODEL
    # --------------------------------------------------
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    # --------------------------------------------------
    # STEP 7 — TRAIN
    # --------------------------------------------------
    print("\n🤖 Training XGBoost model...")
    model.fit(X_train, y_train)

    # --------------------------------------------------
    # STEP 8 — PREDICT
    # --------------------------------------------------
    y_pred = model.predict(X_test)

    # --------------------------------------------------
    # STEP 9 — EVALUATION
    # --------------------------------------------------
    acc = accuracy_score(y_test, y_pred)

    print("\n📊 MODEL ACCURACY:", acc)

    print("\n📋 CLASSIFICATION REPORT:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    return model, X_test, y_test