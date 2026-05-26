import yfinance as yf
import pandas as pd
from processing.feature_engineering import add_stock_features


def fetch_stock_data(ticker="MSFT"):
    data = yf.download(
        ticker,
        period="1mo",
        interval="1h",
        progress=False,
        threads=False
    )

    # flatten MultiIndex columns (yfinance issue)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.reset_index(inplace=True)

    # clean missing values
    data = data.dropna()

    # =========================
    # 🔥 PHASE 2 INTEGRATION
    # =========================
    data = add_stock_features(data)
    data = data.dropna()

    return data


if __name__ == "__main__":
    df = fetch_stock_data("MSFT")

    print("\n📊 FINAL DATA PREVIEW:\n")
    print(df.head())

    print("\n📌 COLUMNS:\n", df.columns)

    # save ML-ready dataset
    df.to_csv("data/msft_stock.csv", index=False)

    print("\n✅ Saved to data/msft_stock.csv")