import feedparser
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import os

OUTPUT_DIR = "data/sentiment"

analyzer = SentimentIntensityAnalyzer()


def fetch_financial_news():
    """
    Fetch Yahoo Finance RSS news
    """
    rss_url = "https://finance.yahoo.com/rss/topstories"

    feed = feedparser.parse(rss_url)

    articles = []

    for entry in feed.entries:
        articles.append({
            "title": entry.title,
            "published": entry.published,
            "link": entry.link
        })

    return pd.DataFrame(articles)


def analyze_sentiment(df):
    sentiments = []

    for title in df["title"]:
        score = analyzer.polarity_scores(title)

        sentiments.append({
            "negative": score["neg"],
            "neutral": score["neu"],
            "positive": score["pos"],
            "compound": score["compound"]
        })

    sentiment_df = pd.DataFrame(sentiments)

    return pd.concat([df, sentiment_df], axis=1)


def save_results(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")

    file_path = f"{OUTPUT_DIR}/news_sentiment_{date_str}.csv"

    df.to_csv(file_path, index=False)

    print(f"[SAVED] Sentiment results → {file_path}")


def run_pipeline():
    print("[FETCHING NEWS]")

    news_df = fetch_financial_news()

    print(news_df.head())

    print("\n[ANALYZING SENTIMENT]")

    sentiment_df = analyze_sentiment(news_df)

    print(sentiment_df.head())

    save_results(sentiment_df)


if __name__ == "__main__":
    run_pipeline()