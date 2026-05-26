import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker="MSFT"):
    data = yf.download(
        ticker,
        period="1mo",
        interval="1h",
        progress=False,
        threads=False
    )

    #  flatten multi-index columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.reset_index(inplace=True)

    # optional cleanup
    data = data.dropna()

    return data

if __name__ == "__main__":
    df = fetch_stock_data("MSFT")
    print(df.head())
    df.to_csv("data/msft_stock.csv", index=False)