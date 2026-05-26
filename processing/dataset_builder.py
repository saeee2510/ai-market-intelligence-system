import pandas as pd


def _normalize_timestamp_stock(df):
    """
    Stock timestamps → force UTC
    """
    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["index"])

    # if timezone missing → assume NYSE time
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("America/New_York")

    # convert to UTC (standard for ML pipelines)
    df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

    return df


def _normalize_timestamp_news(df):
    """
    News timestamps → UTC
    """
    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["publishedAt"],
        errors="coerce",
        utc=True
    )

    df = df.dropna(subset=["timestamp"])
    return df


def _aggregate_news(df):
    """
    Hourly sentiment aggregation
    """
    df = df.copy()

    df = df.groupby(df["timestamp"].dt.floor("h"))["sentiment"].mean()
    df = df.reset_index()
    df.columns = ["timestamp", "news_sentiment"]

    return df


def _aggregate_reddit(df):
    """
    Hourly sentiment aggregation
    (currently simplified using current timestamp fallback)
    """
    df = df.copy()

    df["timestamp"] = pd.to_datetime("now", utc=True)

    df = df.groupby(df["timestamp"].dt.floor("h"))["sentiment"].mean()
    df = df.reset_index()
    df.columns = ["timestamp", "reddit_sentiment"]

    return df


def build_dataset(stock_df, news_df=None, reddit_df=None):
    """
    FINAL ML-READY DATASET BUILDER
    """

    # =========================
    # STOCK BASE
    # =========================
    stock_df = _normalize_timestamp_stock(stock_df)

    base = stock_df[[
        "timestamp",
        "Close",
        "return",
        "volatility"
    ]].copy()

    # =========================
    # NEWS FUSION
    # =========================
    if news_df is not None and "sentiment" in news_df.columns:
        news_df = _normalize_timestamp_news(news_df)
        news_agg = _aggregate_news(news_df)

        base = base.merge(news_agg, on="timestamp", how="left")

    # =========================
    # REDDIT FUSION
    # =========================
    if reddit_df is not None and "sentiment" in reddit_df.columns:
        reddit_df = reddit_df.copy()

        reddit_df["timestamp"] = pd.to_datetime("now", utc=True)

        reddit_agg = reddit_df.groupby(
            reddit_df["timestamp"].dt.floor("h")
        )["sentiment"].mean().reset_index()

        reddit_agg.columns = ["timestamp", "reddit_sentiment"]

        base = base.merge(reddit_agg, on="timestamp", how="left")

    # =========================
    # CLEAN FINAL DATASET
    # =========================
    base = base.sort_values("timestamp")
    base = base.fillna(0)

    return base


if __name__ == "__main__":
    print("Dataset builder ready ✔")