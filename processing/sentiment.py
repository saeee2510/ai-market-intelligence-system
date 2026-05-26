from transformers import pipeline

# ⚠️ load once (important for performance)
sentiment_model = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)


def get_sentiment(text):
    if not text:
        return 0.0

    result = sentiment_model(str(text))[0]

    label = result["label"]
    score = result["score"]

    if label == "positive":
        return score
    elif label == "negative":
        return -score
    else:
        return 0.0