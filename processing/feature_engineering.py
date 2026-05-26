import pandas as pd

def add_stock_features(df):
    df = df.copy()

    # Returns
    df["return"] = df["Close"].pct_change()

    # Moving averages
    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_10"] = df["Close"].rolling(10).mean()

    # Volatility
    df["volatility"] = df["return"].rolling(10).std()

    return df