import feedparser
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

RSS_URL = "https://finance.yahoo.com/rss/topstories"
analyzer = SentimentIntensityAnalyzer()

OUT_DIR = "data/sentiment"


def fetch_news():
    feed = feedparser.parse(RSS_URL)

    return pd.DataFrame([{
        "title": e.title,
        "link": e.link
    } for e in feed.entries])


def analyze(df):
    scores = []

    for t in df["title"]:
        s = analyzer.polarity_scores(t)
        scores.append(s)

    return pd.concat([df, pd.DataFrame(scores)], axis=1)


def run():
    df = fetch_news()
    df = analyze(df)

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(f"{OUT_DIR}/news.csv", index=False)

    print(df.head())


if __name__ == "__main__":
    run()