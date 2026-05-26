import yfinance as yf

def load_stock_data(ticker):
    df = yf.download(ticker, period="6mo", auto_adjust=True)
    df.reset_index(inplace=True)
    return df