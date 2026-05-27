import joblib
import matplotlib.pyplot as plt
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from sklearn.model_selection import train_test_split

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
# FEATURES
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
# XGBOOST MODEL
# -----------------------------------

model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# -----------------------------------
# PREDICTIONS
# -----------------------------------

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)

# -----------------------------------
# EVALUATION
# -----------------------------------

accuracy = accuracy_score(y_test, predictions)

print("\n📊 XGBOOST ACCURACY:")
print(round(accuracy, 4))

print("\n📋 CLASSIFICATION REPORT:\n")
print(classification_report(y_test, predictions))

print("\n🧠 CONFUSION MATRIX:\n")
print(confusion_matrix(y_test, predictions))

# -----------------------------------
# FEATURE IMPORTANCE
# -----------------------------------

importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print("\n🔥 FEATURE IMPORTANCE:\n")
print(importance_df)

# -----------------------------------
# PLOT FEATURE IMPORTANCE
# -----------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    importance_df["feature"],
    importance_df["importance"]
)

plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("XGBoost Feature Importance")

plt.tight_layout()

plt.savefig("feature_importance.png")

print("\n✅ Saved feature importance chart")

# -----------------------------------
# SAVE MODEL
# -----------------------------------

joblib.dump(model, "models/xgboost_model.pkl")

print("\n✅ Model saved to models/xgboost_model.pkl")