import pandas as pd

# -----------------------------
# INGESTION
# -----------------------------
from ingestion.stock_data import fetch_stock_data
from ingestion.reddit_data import fetch_reddit
from ingestion.news_data import fetch_news

# -----------------------------
# PROCESSING
# -----------------------------
from processing.feature_engineering import add_features
from processing.labeling import create_labels

# -----------------------------
# MODEL
# -----------------------------
from models.train_xgboost import train_model


def run_pipeline():

    print("\n📦 Loading data...")

    # -----------------------------
    # STEP 1: STOCK DATA
    # -----------------------------
    stock_df = fetch_stock_data("MSFT")

    # -----------------------------
    # STEP 2: NEWS + REDDIT (OPTIONAL SAFE LOAD)
    # -----------------------------
    try:
        reddit_df = fetch_reddit()
    except:
        print("⚠️ Reddit failed, using empty df")
        reddit_df = pd.DataFrame(columns=["timestamp", "reddit_sentiment"])

    try:
        news_df = fetch_news()
    except:
        print("⚠️ News failed, using empty df")
        news_df = pd.DataFrame(columns=["timestamp", "news_sentiment"])

    # -----------------------------
    # STEP 3: MERGE DATA
    # -----------------------------
    print("\n🔗 Building dataset...")

    df = stock_df.copy()

    if "timestamp" in reddit_df.columns:
        df = df.merge(reddit_df, on="timestamp", how="left")

    if "timestamp" in news_df.columns:
        df = df.merge(news_df, on="timestamp", how="left")

    df = df.fillna(0)

    # -----------------------------
    # STEP 4: FEATURE ENGINEERING
    # -----------------------------
    print("\n⚙️ Feature engineering...")
    df = add_features(df)

    # -----------------------------
    # STEP 5: LABEL CREATION
    # -----------------------------
    print("\n🏷️ Creating labels...")
    df = create_labels(df)

    print("\n📊 Final dataset shape:", df.shape)
    print(df.head())

    # -----------------------------
    # STEP 6: TRAIN MODEL
    # -----------------------------
    print("\n🤖 Training model...")

    model, X_test, y_test = train_model(df)

    print("\n✅ PIPELINE COMPLETE")

    return model


if __name__ == "__main__":
    run_pipeline()