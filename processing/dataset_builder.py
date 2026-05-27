import pandas as pd


def _normalize_time(df):
    df = df.copy()

    if "timestamp" not in df.columns:
        raise ValueError(f"Missing timestamp column. Columns: {df.columns}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # force UTC consistency
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

    df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    return df


def build_dataset(stock_df, news_df=None, reddit_df=None):

    # -----------------------
    # STOCK
    # -----------------------
    stock_df = _normalize_time(stock_df)

    required_stock_cols = ["timestamp", "Close", "return", "volatility"]
    stock_df = stock_df[[c for c in required_stock_cols if c in stock_df.columns]]

    # -----------------------
    # NEWS
    # -----------------------
    if news_df is not None:
        news_df = _normalize_time(news_df)

        if "sentiment" not in news_df.columns:
            news_df["sentiment"] = 0.0

        news_df = news_df[["timestamp", "sentiment"]]
        news_df = news_df.groupby("timestamp").mean().reset_index()
        news_df.rename(columns={"sentiment": "news_sentiment"}, inplace=True)

    # -----------------------
    # REDDIT
    # -----------------------
    if reddit_df is not None:
        reddit_df = _normalize_time(reddit_df)

        if "reddit_sentiment" not in reddit_df.columns:
            reddit_df["reddit_sentiment"] = 0.0

        reddit_df = reddit_df[["timestamp", "reddit_sentiment"]]
        reddit_df = reddit_df.groupby("timestamp").mean().reset_index()

    # -----------------------
    # MERGE
    # -----------------------
    df = stock_df.copy()

    if news_df is not None:
        df = df.merge(news_df, on="timestamp", how="left")

    if reddit_df is not None:
        df = df.merge(reddit_df, on="timestamp", how="left")

    # -----------------------
    # FILL MISSING
    # -----------------------
    if "news_sentiment" in df.columns:
        df["news_sentiment"] = df["news_sentiment"].fillna(0.0)

    if "reddit_sentiment" in df.columns:
        df["reddit_sentiment"] = df["reddit_sentiment"].fillna(0.0)

    # -----------------------
    # SORT
    # -----------------------
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


if __name__ == "__main__":
    print("Dataset builder ready.")