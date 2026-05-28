import os
import requests
import pandas as pd
from dotenv import load_dotenv
from processing.sentiment import get_sentiment

load_dotenv()


def fetch_news(query="Apple", page_size=50):

    url = "https://newsapi.org/v2/everything"

    params = {
        "q": query,
        "pageSize": page_size,
        "apiKey": os.getenv("NEWS_API_KEY"),
        "language": "en",
        "sortBy": "publishedAt"
    }

    response = requests.get(url, params=params)
    data = response.json()

    articles = data.get("articles", [])

    rows = []

    for a in articles:
        if a.get("title"):

            rows.append({
                "text": a["title"],
                "timestamp": a["publishedAt"]
            })

    df = pd.DataFrame(rows)

    # -----------------------------
    # FIX TIMESTAMP
    # -----------------------------
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)

    # -----------------------------
    # SENTIMENT
    # -----------------------------
    df["sentiment"] = df["text"].apply(get_sentiment)

    # -----------------------------
    # FINAL CLEAN
    # -----------------------------
    df = df.dropna(subset=["timestamp"])

    return df


if __name__ == "__main__":
    df = fetch_news()
    print(df.head())
    print(df.columns)