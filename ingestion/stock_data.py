import yfinance as yf
import pandas as pd


def fetch_stock_data(ticker="MSFT"):
    # -----------------------------
    # DOWNLOAD DATA
    # -----------------------------
    data = yf.download(
        ticker,
        period="1mo",
        interval="1h",
        progress=False,
        threads=False
    )

    if data is None or data.empty:
        raise ValueError("No data returned from yfinance")

    # -----------------------------
    # FLATTEN STRUCTURE SAFELY
    # -----------------------------
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # -----------------------------
    # RESET INDEX → brings datetime out
    # -----------------------------
    data = data.reset_index()

    # -----------------------------
    # HANDLE TIMESTAMP COLUMN (ROBUST)
    # -----------------------------
    if "Datetime" in data.columns:
        data.rename(columns={"Datetime": "timestamp"}, inplace=True)
    elif "Date" in data.columns:
        data.rename(columns={"Date": "timestamp"}, inplace=True)
    elif "index" in data.columns:
        data.rename(columns={"index": "timestamp"}, inplace=True)
    else:
        # last fallback: first column
        data.rename(columns={data.columns[0]: "timestamp"}, inplace=True)

    # -----------------------------
    # VALIDATE REQUIRED COLUMNS
    # -----------------------------
    required_cols = ["timestamp", "Close", "High", "Low", "Open", "Volume"]
    for col in ["timestamp", "Close"]:
        if col not in data.columns:
            raise ValueError(f"Missing {col}. Columns: {data.columns}")

    # -----------------------------
    # CLEAN TIMESTAMP
    # -----------------------------
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data.dropna(subset=["timestamp", "Close"])

    # -----------------------------
    # FEATURES
    # -----------------------------
    data["return"] = data["Close"].pct_change()
    data["ma_5"] = data["Close"].rolling(5).mean()
    data["ma_10"] = data["Close"].rolling(10).mean()
    data["volatility"] = data["return"].rolling(10).std()

    # -----------------------------
    # CLEAN FINAL DATASET
    # -----------------------------
    data = data.dropna().reset_index(drop=True)

    return data


if __name__ == "__main__":
    df = fetch_stock_data("MSFT")
    print("\n📊 STOCK DATA PREVIEW:\n")
    print(df.head())
    print("\n📌 COLUMNS:\n", df.columns)