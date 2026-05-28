import yfinance as yf
import pandas as pd


def fetch_stock_data(ticker="MSFT"):

    data = yf.download(
        tickers=ticker,
        period="2y",
        interval="1d",
        progress=False,
        auto_adjust=False,
        group_by="column",
        threads=False
    )

    # ------------------------------
    # STEP 1: flatten columns safely
    # ------------------------------
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ["_".join(col).strip() if isinstance(col, tuple) else col
                        for col in data.columns]

    # ------------------------------
    # STEP 2: reset index
    # ------------------------------
    data = data.reset_index()

    # ------------------------------
    # STEP 3: fix timestamp
    # ------------------------------
    if "Datetime" in data.columns:
        data.rename(columns={"Datetime": "timestamp"}, inplace=True)
    elif "Date" in data.columns:
        data.rename(columns={"Date": "timestamp"}, inplace=True)
    else:
        data.rename(columns={data.columns[0]: "timestamp"}, inplace=True)

    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")

    # ------------------------------
    # STEP 4: normalize OHLC columns
    # ------------------------------
    def find_col(keyword):
        for c in data.columns:
            if keyword.lower() in c.lower():
                return c
        return None

    open_col = find_col("open")
    high_col = find_col("high")
    low_col = find_col("low")
    close_col = find_col("close")
    volume_col = find_col("volume")

    if close_col is None:
        raise ValueError(f"Close column not found. Columns: {data.columns}")

    data = data.rename(columns={
        open_col: "Open",
        high_col: "High",
        low_col: "Low",
        close_col: "Close",
        volume_col: "Volume"
    })

    # ------------------------------
    # STEP 5: feature engineering
    # ------------------------------
    data["return"] = data["Close"].pct_change()
    data["volatility"] = data["return"].rolling(10).std()

    data = data.dropna().reset_index(drop=True)

    return data


if __name__ == "__main__":
    df = fetch_stock_data("MSFT")
    print(df.head())
    print(df.columns)