import pandas as pd

def merge_datasets(stock_df, news_df, reddit_df):
    stock_df["source"] = "stock"
    news_df["source"] = "news"
    reddit_df["source"] = "reddit"

    # standardize columns
    news_df = news_df.rename(columns={"description": "text"})
    reddit_df = reddit_df.rename(columns={"title": "text"})

    combined = pd.concat([
        stock_df[["Close", "return", "volatility"]].copy(),
        news_df[["text"]],
        reddit_df[["text"]]
    ], ignore_index=True)

    return combined