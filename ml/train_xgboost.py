import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score


# =========================================================
# MAIN TRAIN FUNCTION
# =========================================================
def train_model(df):

    df = df.copy()
    df = df.dropna().reset_index(drop=True)

    print("\n📊 Label distribution:")
    print(df["label"].value_counts(normalize=True))

    # =====================================================
    # FEATURES / TARGET
    # =====================================================
    drop_cols = ["label", "future_return", "timestamp"]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["label"]

    # =====================================================
    # TRAIN / TEST SPLIT
    # =====================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    # =====================================================
    # CLASS BALANCE
    # =====================================================
    scale_pos_weight = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)

    # =====================================================
    # MODEL
    # =====================================================
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

    print("\n🤖 Training XGBoost model...")
    model.fit(X_train, y_train)

    # =====================================================
    # PREDICTIONS
    # =====================================================
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    print("\n📊 MODEL RESULTS")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, zero_division=0))

    print("\n📋 CLASSIFICATION REPORT:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    # =====================================================
# STRATEGY SIGNALS (IMPROVED VERSION)
# =====================================================

    signals = []

    # align test dataframe
    df_test = df.iloc[-len(y_test):].copy()

    # indicators (MUST be outside loop)
    df_test["trend"] = df_test["Close"].rolling(10).mean()
    volatility = df_test["Close"].pct_change().rolling(10).std().fillna(0)

    for i, p in enumerate(proba):

        # skip unstable market conditions
        if volatility.iloc[i] < 0.008:
            signals.append(0)
            continue

        trend_up = df_test["Close"].iloc[i] > df_test["trend"].iloc[i]

        # STRONG BUY
        if p > 0.75 and trend_up:
            signals.append(1)

        # STRONG SELL
        elif p < 0.25 and not trend_up:
            signals.append(-1)

        # NO TRADE
        else:
            signals.append(0)

    # =====================================================
    # BACKTEST (REALISTIC)
    # =====================================================
    transaction_cost = 0.001  # 0.1%

    df_test["signal"] = signals
    df_test["returns"] = df_test["Close"].pct_change()

    df_test["strategy_returns"] = (
        df_test["signal"].shift(1) * df_test["returns"]
    ) - (transaction_cost * df_test["signal"].diff().abs())

    total_return = df_test["strategy_returns"].sum()
    sharpe = df_test["strategy_returns"].mean() / (df_test["strategy_returns"].std() + 1e-9)
    max_drawdown = (df_test["strategy_returns"].cumsum().min())
    win_rate = (df_test["strategy_returns"] > 0).mean()
    trades = df_test["signal"].diff().abs().sum()

    print("\n📊 BACKTEST RESULTS")
    print("Total Return:", round(total_return, 4))
    print("Sharpe Ratio:", round(sharpe, 4))
    print("Max Drawdown:", round(max_drawdown, 4))
    print("Win Rate:", round(win_rate, 4))
    print("Trades Executed:", int(trades))

    # =====================================================
    # RETURN
    # =====================================================
    return model, X_test, y_test