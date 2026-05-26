import requests
from config import NEWS_API_KEY

def fetch_news(query="Apple"):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}"
    return requests.get(url).json()

if __name__ == "__main__":
    news = fetch_news("Apple")
    print(news)