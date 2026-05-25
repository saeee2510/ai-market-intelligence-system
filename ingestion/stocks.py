import os
import time
import requests
import pandas as pd
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------
API_KEY = "01FBFSSSQEMNI1CA"  # <-- put your key here
DATA_DIR = "data/stocks"

# Alpha Vantage allows ~5 requests/minute safely
REQUEST_DELAY = 12  # seconds


# -----------------------------
# FETCH DATA
# -----------------------------
def fetch_stock_data(ticker="AAPL"):
    """
    Fetch daily stock data from Alpha Vantage
    """
    try:
        url = "https://www.alphavantage.co/query"

        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "outputsize": "compact",
            "apikey": API_KEY
        }

        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        # API error handling
        if "Time Series (Daily)" not in data:
            print(f"[ERROR] {ticker}: {data}")
            return None

        time_series = data["Time Series (Daily)"]

        df = pd.DataFrame.from_dict(time_series, orient="index")

        df = df.rename(columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume"
        })

        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        df = df.astype(float)
        df["ticker"] = ticker

        return df

    except Exception as e:
        print(f"[EXCEPTION] {ticker}: {e}")
        return None


# -----------------------------
# SAVE DATA
# -----------------------------
def save_to_csv(df, ticker):
    os.makedirs(DATA_DIR, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    file_path = f"{DATA_DIR}/{ticker}_{date_str}.csv"

    df.to_csv(file_path)
    print(f"[SAVED] {ticker} → {file_path}")


# -----------------------------
# PIPELINE RUNNER
# -----------------------------
def run_pipeline(tickers):
    for i, ticker in enumerate(tickers):
        print(f"\n[FETCHING] {ticker}")

        df = fetch_stock_data(ticker)

        if df is not None:
            print(f"\n{ticker} sample:")
            print(df.head())
            save_to_csv(df, ticker)
        else:
            print(f"[FAILED] {ticker}")

        # RATE LIMIT CONTROL (CRITICAL FIX)
        if i < len(tickers) - 1:
            print(f"[SLEEP] Waiting {REQUEST_DELAY}s to avoid rate limit...")
            time.sleep(REQUEST_DELAY)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOG"]
    run_pipeline(tickers)