import pandas as pd
import sys
import os

# make imports stable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.stock_data import fetch_stock_data
from ingestion.reddit_data import fetch_reddit

from processing.feature_engineering import add_features
from processing.sentiment import get_sentiment
from processing.create_labels import create_labels

from ml.train_xgboost import train_model


# -----------------------------
# STEP 1 — LOAD DATA
# -----------------------------
def load_data():
    print("\n📦 Loading data...")

    stock_df = fetch_stock_data("AAPL")
    reddit_posts = fetch_reddit()

    # sentiment
    reddit_sentiment = get_sentiment(reddit_posts)
    stock_df["reddit_sentiment"] = reddit_sentiment

    return stock_df


# -----------------------------
# STEP 2 — BUILD DATASET
# -----------------------------
def build_dataset(df):
    print("\n🔗 Building dataset...")

    df = add_features(df)
    return df


# -----------------------------
# STEP 3 — FEATURE CLEANING
# -----------------------------
def feature_engineering(df):
    print("\n Feature engineering...")

    df = df.dropna().reset_index(drop=True)
    return df


# -----------------------------
# PIPELINE
# -----------------------------
def run_pipeline():

    # 1. load data
    df = load_data()

    # 2. features
    df = build_dataset(df)

    # 3. clean
    df = feature_engineering(df)

    # --------------------------------------------------
    # 🔥 CRITICAL FIX — CREATE LABELS (YOU WERE MISSING THIS)
    # --------------------------------------------------
    print("\n Creating labels...")
    df = create_labels(df)

    # sanity check (VERY IMPORTANT)
    print("\n Label distribution:")
    print(df["label"].value_counts(normalize=True))

    # --------------------------------------------------
    # save dataset
    # --------------------------------------------------
    df.to_csv("data/final_dataset.csv", index=False)
    print("\n Saved final_dataset.csv")

    # --------------------------------------------------
    # train model
    # --------------------------------------------------
    print("\n Training model...")

    model, X_test, y_test = train_model(df)

    print("\n PIPELINE COMPLETE")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    run_pipeline()