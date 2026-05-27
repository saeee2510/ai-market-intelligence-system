import pandas as pd


def add_time_series_features(df):
    df = df.copy()

    # ---------------------------
    # LAG FEATURES
    # ---------------------------
    df["return_lag1"] = df["return"].shift(1)
    df["return_lag2"] = df["return"].shift(2)

    df["volatility_lag1"] = df["volatility"].shift(1)

    df["reddit_lag1"] = df["reddit_sentiment"].shift(1)
    df["news_lag1"] = df["news_sentiment"].shift(1)

    # ---------------------------
    # ROLLING FEATURES
    # ---------------------------
    df["return_roll_mean_3"] = df["return"].rolling(3).mean()
    df["return_roll_std_3"] = df["return"].rolling(3).std()

    df["sentiment_roll_mean_3"] = (
        df[["news_sentiment", "reddit_sentiment"]]
        .mean(axis=1)
        .rolling(3)
        .mean()
    )

    # ---------------------------
    # PRICE MOMENTUM
    # ---------------------------
    df["price_change_1"] = df["Close"].pct_change(1)
    df["price_change_3"] = df["Close"].pct_change(3)

    # ---------------------------
    # CLEAN
    # ---------------------------
    df = df.dropna().reset_index(drop=True)

    return df


if __name__ == "__main__":
    print("Feature engineering module ready.")