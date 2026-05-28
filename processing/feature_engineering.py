import pandas as pd


def add_features(df):

    df = df.copy()

    # --------------------------------------------------
    # BASIC RETURN
    # --------------------------------------------------
    df["return"] = df["Close"].pct_change()

    # --------------------------------------------------
    # STEP 1 — REGIME FILTER FEATURES
    # --------------------------------------------------
    df["trend"] = df["Close"].rolling(10).mean() - df["Close"].rolling(30).mean()
    df["trend_strength"] = df["trend"] / df["Close"]

    df["vol_regime"] = df["return"].rolling(10).std()

    # --------------------------------------------------
    # STEP 4 — MOMENTUM FEATURES
    # --------------------------------------------------
    df["momentum_3"] = df["Close"] / df["Close"].shift(3) - 1
    df["momentum_7"] = df["Close"] / df["Close"].shift(7) - 1

    df["acceleration"] = df["momentum_3"] - df["momentum_7"]

    # --------------------------------------------------
    # LAG FEATURES
    # --------------------------------------------------
    df["return_lag1"] = df["return"].shift(1)
    df["return_lag2"] = df["return"].shift(2)

    df["volatility_lag1"] = df["return"].rolling(5).std().shift(1)

    # --------------------------------------------------
    # ROLLING FEATURES
    # --------------------------------------------------
    df["return_roll_mean_3"] = df["return"].rolling(3).mean()
    df["return_roll_std_3"] = df["return"].rolling(3).std()

    # --------------------------------------------------
    # PRICE MOMENTUM
    # --------------------------------------------------
    df["price_change_1"] = df["Close"].pct_change(1)
    df["price_change_3"] = df["Close"].pct_change(3)

    # --------------------------------------------------
    # CLEAN NaNs (IMPORTANT)
    # --------------------------------------------------
    df = df.replace([float("inf"), float("-inf")], pd.NA)
    df = df.dropna().reset_index(drop=True)

    return df


# --------------------------------------------------
# OPTIONAL WRAPPER (for backward compatibility)
# --------------------------------------------------
def add_time_series_features(df):
    return add_features(df)