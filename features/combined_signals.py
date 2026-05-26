import pandas as pd


def compute(stock_df, sentiment_df):
    stock_signal = 1 if stock_df["close"].iloc[-1] > stock_df["sma_7"].iloc[-1] else -1

    avg_sent = sentiment_df["compound"].mean()

    if avg_sent > 0.2:
        sent_signal = 1
    elif avg_sent < -0.2:
        sent_signal = -1
    else:
        sent_signal = 0

    score = stock_signal + sent_signal

    if score >= 2:
        final = "STRONG BULLISH"
    elif score == 1:
        final = "BULLISH"
    elif score == 0:
        final = "NEUTRAL"
    else:
        final = "BEARISH"

    return final