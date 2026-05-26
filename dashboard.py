import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.data_loader import load_stock_data

from sentiment.news_sentiment import fetch_news, analyze


st.title("AI Market Intelligence System")

ticker = st.sidebar.selectbox("Stock", ["AAPL", "MSFT", "GOOG"])


# LOAD FEATURES (IMPORTANT FIX)
# stock_df = pd.read_csv(f"data/features/{ticker}_features.csv", index_col=0, parse_dates=True)
ticker = "AAPL"  # or Streamlit input later
stock_df = load_stock_data(ticker)

st.subheader("Price Chart")

fig = go.Figure()
fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df["Close"], name="Close"))
fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df["sma_7"], name="SMA 7"))
fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df["sma_30"], name="SMA 30"))

st.plotly_chart(fig, use_container_width=True)


# SENTIMENT
news = fetch_news()
sent = analyze(news)

st.subheader("Sentiment Score")
st.write(sent["compound"].mean())

st.subheader("Raw News")
st.dataframe(sent.head())