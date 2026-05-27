from ingestion.stock_data import fetch_stock_data
from ingestion.news_data import fetch_news
from ingestion.reddit_data import fetch_reddit

from processing.dataset_builder import build_dataset
from processing.create_labels import create_labels

stock_df = fetch_stock_data("MSFT")
news_df = fetch_news()
reddit_df = fetch_reddit()

df = build_dataset(stock_df, news_df, reddit_df)

df = create_labels(df)

print(df.head())

print("\n📌 Columns:")
print(df.columns)

print("\n📊 Label Distribution:")
print(df["label"].value_counts())