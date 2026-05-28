from transformers import pipeline

sentiment_model = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)


def get_sentiment(text):
    result = sentiment_model(str(text))[0]

    label = result["label"].lower()
    score = result["score"]

    if "positive" in label:
        return score
    elif "negative" in label:
        return -score
    else:
        return 0.0