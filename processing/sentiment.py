from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    if not text:
        return 0

    score = analyzer.polarity_scores(str(text))
    return score["compound"]


def add_sentiment(df, text_col):
    df = df.copy()
    df["sentiment"] = df[text_col].apply(get_sentiment)
    return df