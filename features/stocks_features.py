import os
import pandas as pd
import numpy as np

DATA_DIR = "data/stocks"
OUT_DIR = "data/features"


def load_stock_data(ticker):
    path = f"{DATA_DIR}/{ticker}.csv"

    if not os.path.exists(path):
        return None

    return pd.read_csv(path, index_col=0, parse_dates=True)


def add_features(df):
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    df["sma_7"] = df["close"].rolling(7).mean()
    df["sma_30"] = df["close"].rolling(30).mean()

    df["volatility_7"] = df["returns"].rolling(7).std()

    df["momentum"] = df["close"] - df["close"].shift(5)

    return df


def save(df, ticker):
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(f"{OUT_DIR}/{ticker}_features.csv")
    print(f"[SAVED FEATURES] {ticker}")


def run():
    for t in ["AAPL", "MSFT", "GOOG"]:
        df = load_stock_data(t)

        if df is None:
            print(f"[SKIP] {t}")
            continue

        df = add_features(df)
        save(df, t)


if __name__ == "__main__":
    run()