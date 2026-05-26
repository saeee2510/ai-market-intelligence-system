import numpy as np
import pandas as pd
import os

DATA_DIR = "data/stocks"
OUTPUT_DIR = "data/features"


def load_latest_file(ticker):
    files = sorted([
        f for f in os.listdir(DATA_DIR)
        if f.startswith(ticker)
    ])

    if not files:
        return None

    latest_file = os.path.join(DATA_DIR, files[-1])
    return pd.read_csv(latest_file, index_col=0, parse_dates=True)


def add_features(df):
    # Daily returns
    df["returns"] = df["close"].pct_change()

    # Log returns
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    # Moving averages
    df["sma_7"] = df["close"].rolling(window=7).mean()
    df["sma_30"] = df["close"].rolling(window=30).mean()

    # Volatility (rolling std dev)
    df["volatility_7"] = df["returns"].rolling(window=7).std()

    # Momentum signal
    df["momentum"] = df["close"] - df["close"].shift(5)

    return df


def save_features(df, ticker):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = f"{OUTPUT_DIR}/{ticker}_features.csv"
    df.to_csv(output_path)

    print(f"[SAVED FEATURES] {ticker} → {output_path}")


def run_feature_pipeline(tickers):
    for ticker in tickers:
        print(f"\n[FEATURE ENGINEERING] {ticker}")

        df = load_latest_file(ticker)

        if df is None:
            print(f"[SKIP] No data for {ticker}")
            continue

        df = add_features(df)

        print(df.tail())

        save_features(df, ticker)


if __name__ == "__main__":
    run_feature_pipeline(["AAPL", "MSFT", "GOOG"])