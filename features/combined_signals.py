import pandas as pd
import os


# -----------------------------
# LOAD DATA
# -----------------------------
def load_stock_features(ticker):
    path = f"data/features/{ticker}_features.csv"

    if not os.path.exists(path):
        print(f"[MISSING STOCK DATA] {ticker}")
        return None

    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_sentiment():
    path = "data/sentiment"

    files = sorted([f for f in os.listdir(path) if f.endswith(".csv")])

    if not files:
        print("[MISSING SENTIMENT DATA]")
        return None

    latest = os.path.join(path, files[-1])
    return pd.read_csv(latest)


# -----------------------------
# SIMPLE SIGNAL LOGIC
# -----------------------------
def compute_signal(stock_df, sentiment_df):
    # STOCK SIGNAL
    last_row = stock_df.iloc[-1]

    price_trend = last_row["close"] - last_row["sma_7"]

    volatility = last_row["volatility_7"]

    if price_trend > 0:
        stock_signal = 1
    else:
        stock_signal = -1


    # SENTIMENT SIGNAL
    avg_sentiment = sentiment_df["compound"].mean()

    if avg_sentiment > 0.2:
        sentiment_signal = 1
    elif avg_sentiment < -0.2:
        sentiment_signal = -1
    else:
        sentiment_signal = 0


    # FINAL COMBINED SIGNAL
    score = stock_signal + sentiment_signal


    if score >= 2:
        final = "STRONG BULLISH 🟢"
    elif score == 1:
        final = "BULLISH 🟢"
    elif score == 0:
        final = "NEUTRAL 🟡"
    elif score == -1:
        final = "BEARISH 🔴"
    else:
        final = "STRONG BEARISH 🔴"


    return {
        "stock_signal": stock_signal,
        "sentiment_signal": sentiment_signal,
        "final_score": score,
        "final_signal": final
    }


# -----------------------------
# RUN PIPELINE
# -----------------------------
def run(ticker="AAPL"):
    print(f"\n[COMBINING SIGNALS] {ticker}")

    stock_df = load_stock_features(ticker)
    sentiment_df = load_sentiment()

    if stock_df is None or sentiment_df is None:
        print("[FAILED] Missing inputs")
        return

    result = compute_signal(stock_df, sentiment_df)

    print("\n📊 SIGNAL OUTPUT")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    run("AAPL")