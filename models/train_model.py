import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from ingestion.stock_data import fetch_stock_data
from ingestion.news_data import fetch_news
from ingestion.reddit_data import fetch_reddit

from processing.dataset_builder import build_dataset
from processing.labeling import create_labels


# -----------------------------------
# LOAD DATA
# -----------------------------------

stock_df = fetch_stock_data("MSFT")
news_df = fetch_news()
reddit_df = fetch_reddit()

# -----------------------------------
# BUILD DATASET
# -----------------------------------

df = build_dataset(stock_df, news_df, reddit_df)

# -----------------------------------
# CREATE LABELS
# -----------------------------------

df = create_labels(df)

# -----------------------------------
# FEATURES + TARGET
# -----------------------------------

FEATURES = [
    "return",
    "volatility",
    "news_sentiment",
    "reddit_sentiment"
]

TARGET = "label"

X = df[FEATURES]
y = df[TARGET]

# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=False
)

# -----------------------------------
# TRAIN MODEL
# -----------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------------
# PREDICTIONS
# -----------------------------------

predictions = model.predict(X_test)

# -----------------------------------
# EVALUATION
# -----------------------------------

accuracy = accuracy_score(y_test, predictions)

print("\n📊 MODEL ACCURACY:")
print(accuracy)

print("\n📋 CLASSIFICATION REPORT:\n")
print(classification_report(y_test, predictions))