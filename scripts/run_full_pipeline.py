import pandas as pd
import sys
import os

# --------------------------------------------------
# Make imports stable (VERY IMPORTANT)
# --------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --------------------------------------------------
# Imports
# --------------------------------------------------
from ingestion.stock_data import fetch_stock_data
from ingestion.reddit_data import fetch_reddit

from processing.feature_engineering import add_features
from processing.sentiment import get_finbert_sentiment
from processing.create_labels import create_labels

from ml.train_xgboost import train_model


# ==================================================
# STEP 1 — LOAD DATA
# ==================================================
def load_data():
    print("\n Loading data...")

    stock_df = fetch_stock_data("AAPL")
    reddit_posts = fetch_reddit()

    # -----------------------------
    # SENTIMENT (FinBERT)
    # -----------------------------
    reddit_sentiment = get_finbert_sentiment(reddit_posts)

    # attach sentiment (scalar → broadcast)
    stock_df["reddit_sentiment"] = reddit_sentiment

    return stock_df


# ==================================================
# STEP 2 — BUILD FEATURES
# ==================================================
def build_dataset(df):
    print("\n🔗 Building dataset...")

    df = add_features(df)
    return df


# ==================================================
# STEP 3 — CLEAN DATA
# ==================================================
def feature_engineering(df):
    print("\n⚙️ Feature engineering...")

    df = df.dropna().reset_index(drop=True)
    return df


# ==================================================
# PIPELINE
# ==================================================
def run_pipeline():

    # -----------------------------
    # LOAD
    # -----------------------------
    df = load_data()

    # -----------------------------
    # FEATURES
    # -----------------------------
    df = build_dataset(df)

    # -----------------------------
    # CLEAN
    # -----------------------------
    df = feature_engineering(df)

    # ==================================================
    # 🔥 LABEL CREATION (CRITICAL STEP)
    # ==================================================
    print("\n🏷️ Creating labels...")
    df = create_labels(df)

    # sanity check
    print("\n📊 Label distribution:")
    print(df["label"].value_counts(normalize=True))

    # ==================================================
    # SAVE DATASET
    # ==================================================
    df.to_csv("data/final_dataset.csv", index=False)
    print("\n✅ Saved final_dataset.csv")

    # ==================================================
    # TRAIN MODEL
    # ==================================================
    print("\n🤖 Training model...")

    model, X_test, y_test = train_model(df)

    print("\n✅ PIPELINE COMPLETE")


# ==================================================
# RUN
# ==================================================
if __name__ == "__main__":
    run_pipeline()