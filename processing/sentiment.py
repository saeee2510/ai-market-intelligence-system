from transformers import pipeline
import numpy as np

# --------------------------------------------------
# Load FinBERT once
# --------------------------------------------------
finbert = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)


# --------------------------------------------------
# FinBERT Sentiment Function
# --------------------------------------------------
def get_finbert_sentiment(texts):
    """
    Robust FinBERT sentiment scoring.

    Supports:
    - list[str]
    - pandas Series
    - numpy arrays
    - single string

    Returns:
        float sentiment score
        positive -> bullish
        negative -> bearish
        0 -> neutral
    """

    if texts is None:
        return 0.0

    # pandas Series / numpy array
    if hasattr(texts, "tolist"):
        texts = texts.tolist()

    # single string
    if isinstance(texts, str):
        texts = [texts]

    # fallback
    if not isinstance(texts, list):
        texts = [str(texts)]

    # remove empty values
    texts = [
        t for t in texts
        if isinstance(t, str) and len(t.strip()) > 0
    ]

    if len(texts) == 0:
        return 0.0

    results = finbert(texts)

    scores = []

    for r in results:
        label = r["label"].lower()
        score = r["score"]

        if label == "positive":
            scores.append(score)

        elif label == "negative":
            scores.append(-score)

        else:
            scores.append(0.0)

    return float(np.mean(scores))


# --------------------------------------------------
# Backward Compatibility
# --------------------------------------------------
def get_sentiment(texts):
    """
    Legacy wrapper so older imports continue working.
    """
    return get_finbert_sentiment(texts)