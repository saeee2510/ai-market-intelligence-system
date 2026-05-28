import os
import praw
import pandas as pd
import numpy as np

from dotenv import load_dotenv
from processing.sentiment import get_sentiment

load_dotenv()


def fetch_reddit(
    subreddit_name="wallstreetbets",
    limit=200   # increased for better signal
):

    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT"),
    )

    posts = []

    for post in reddit.subreddit(subreddit_name).hot(limit=limit):

        if not post.title:
            continue

        sentiment = get_sentiment(post.title)

        # ------------------------------------
        # STRONGER ENGAGEMENT WEIGHTING
        # ------------------------------------
        engagement = np.log1p(post.score) + np.log1p(post.num_comments)

        weighted_sentiment = sentiment * engagement

        posts.append({
            "text": post.title,
            "timestamp": post.created_utc,
            "score": post.score,
            "comments": post.num_comments,
            "sentiment": sentiment,
            "reddit_sentiment": weighted_sentiment
        })

    df = pd.DataFrame(posts)

    if df.empty:
        return pd.DataFrame(columns=["timestamp", "reddit_sentiment"])

    # ------------------------------------
    # TIMESTAMP CLEANING
    # ------------------------------------
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)

    # ------------------------------------
    # ADD TIME FEATURES BEFORE AGGREGATION
    # ------------------------------------
    df["hour"] = df["timestamp"].dt.floor("H")

    # ------------------------------------
    # AGGREGATE PROPERLY (NOT SIMPLE MEAN ONLY)
    # ------------------------------------
    df = df.groupby("hour").agg({
        "reddit_sentiment": ["mean", "sum", "std"],
        "sentiment": "mean",
        "score": "sum",
        "comments": "sum"
    })

    # flatten columns
    df.columns = [
        "reddit_sent_mean",
        "reddit_sent_sum",
        "reddit_sent_std",
        "sentiment_mean",
        "score_sum",
        "comments_sum"
    ]

    df = df.reset_index().rename(columns={"hour": "timestamp"})

    # ------------------------------------
    # FILL MISSING VALUES
    # ------------------------------------
    df = df.fillna(0)

    return df


if __name__ == "__main__":

    df = fetch_reddit()

    print("\n REDDIT DATA PREVIEW:\n")
    print(df.head())

    print("\n Columns:")
    print(df.columns)

    print("\n Shape:")
    print(df.shape)