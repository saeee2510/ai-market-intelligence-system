import requests
import pandas as pd
from config import NEWS_API_KEY

def fetch_news(query="Apple"):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}"
    response = requests.get(url).json()

    articles = response.get("articles", [])

    cleaned = []
    for a in articles:
        cleaned.append({
            "title": a.get("title"),
            "source": a.get("source", {}).get("name"),
            "publishedAt": a.get("publishedAt"),
            "description": a.get("description")
        })

    return pd.DataFrame(cleaned)

if __name__ == "__main__":
    df = fetch_news("Apple")
    print(df.head())
    df.to_csv("data/news.csv", index=False)