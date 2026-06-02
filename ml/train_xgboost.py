import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score
)


# =========================================================
#  TRAIN MODEL + SIGNAL ENGINE + BACKTEST
# =========================================================
def train_model(df):

    df = df.copy()
    df = df.dropna().reset_index(drop=True)

    # =========================================================
    # 📊 LABEL DISTRIBUTION
    # =========================================================
    print("\n Label distribution:")
    print(df["label"].value_counts(normalize=True))

    # =========================================================
    # FEATURES / TARGET
    # =========================================================
    drop_cols = ["label", "future_return", "timestamp"]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["label"]

    # =========================================================
    # TRAIN / TEST SPLIT (TIME SERIES SAFE)
    # =========================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    # =========================================================
    # CLASS IMBALANCE HANDLING
    # =========================================================
    scale_pos_weight = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)

    # =========================================================
    # 🤖 MODEL
    # =========================================================
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )

    print("\n Training XGBoost model...")
    model.fit(X_train, y_train)

    # =========================================================
    # PREDICTIONS
    # =========================================================
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    # =========================================================
    # 📊 MODEL EVALUATION
    # =========================================================
    print("\n MODEL RESULTS")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, zero_division=0))

    print("\n CLASSIFICATION REPORT:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    # =========================================================
    # 🚦 SIGNAL GENERATION
    # =========================================================
    signals = []

    for p in proba:
        if p > 0.65:
            signals.append(1)     # BUY
        elif p < 0.35:
            signals.append(-1)    # SELL
        else:
            signals.append(0)     # HOLD

    signals = np.array(signals)

    # =========================================================
    #  SIGNAL DISTRIBUTION
    # =========================================================
    print("\n Signal distribution:")
    print(pd.Series(signals).value_counts(normalize=True))

    # =========================================================
    # 💰 BACKTEST (CORRECT VERSION)
    # =========================================================
    df_test = df.iloc[-len(y_test):].copy().reset_index(drop=True)

    df_test["signal"] = signals

    # forward returns (correct alignment)
    df_test["returns"] = df_test["Close"].pct_change().shift(-1)

    # strategy returns
    df_test["strategy_returns"] = df_test["signal"] * df_test["returns"]

    df_test = df_test.dropna()

    # performance metrics
    total_return = df_test["strategy_returns"].sum()

    sharpe = (
        df_test["strategy_returns"].mean()
        / (df_test["strategy_returns"].std() + 1e-9)
    ) * np.sqrt(252)

    win_rate = (df_test["strategy_returns"] > 0).mean()

    trade_count = (df_test["signal"] != 0).sum()

    # =========================================================
    #  BACKTEST RESULTS
    # =========================================================
    print("\n BACKTEST RESULTS")
    print("Total Return:", round(total_return, 4))
    print("Sharpe Ratio:", round(sharpe, 4))
    print("Win Rate:", round(win_rate, 4))
    print("Trades Executed:", trade_count)

    return model, X_test, y_test