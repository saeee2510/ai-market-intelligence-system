import os
import praw
import pandas as pd
from dotenv import load_dotenv
from processing.sentiment import get_sentiment

load_dotenv()


def fetch_reddit(subreddit_name="wallstreetbets", limit=50):

    # ----------------------------
    # Reddit client
    # ----------------------------
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT"),
    )

    posts = []

    # ----------------------------
    # Fetch posts
    # ----------------------------
    for post in reddit.subreddit(subreddit_name).hot(limit=limit):
        if post.title:
            posts.append({
                "text": post.title,
                "timestamp": post.created_utc
            })

    df = pd.DataFrame(posts)

    # ----------------------------
    # Convert timestamp
    # ----------------------------
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    # ----------------------------
    # Sentiment
    # ----------------------------
    df["sentiment"] = df["text"].apply(get_sentiment)

    # ----------------------------
    # Keep only ML-relevant columns
    # ----------------------------
    df = df[["timestamp", "sentiment"]]

    # ----------------------------
    # Hourly aggregation
    # ----------------------------
    df = (
        df.groupby(pd.Grouper(key="timestamp", freq="1H"))
        .mean()
        .reset_index()
    )

    # ----------------------------
    # Handle missing hours (VERY IMPORTANT)
    # ----------------------------
    df["sentiment"] = df["sentiment"].fillna(0.0)

    # ----------------------------
    # Rename for pipeline
    # ----------------------------
    df.rename(columns={"sentiment": "reddit_sentiment"}, inplace=True)

    return df


# ----------------------------
# Local test
# ----------------------------
if __name__ == "__main__":

    df = fetch_reddit()

    print("\n📊 REDDIT DATA PREVIEW:\n")
    print(df.head())

    print("\n📌 Columns:")
    print(df.columns)

    print("\n📊 Shape:", df.shape)