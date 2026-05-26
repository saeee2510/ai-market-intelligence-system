import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.data_loader import load_stock_data
from sentiment.news_sentiment import fetch_news, analyze

st.title("AI Market Intelligence System")

# Sidebar ticker
ticker = st.sidebar.selectbox("Stock", ["AAPL", "MSFT", "GOOG"])

# ----------------------------
# LOAD DATA
# ----------------------------
stock_df = load_stock_data(ticker)

# Ensure correct sorting
stock_df = stock_df.sort_values("Date")

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
stock_df["SMA_7"] = stock_df["Close"].rolling(window=7).mean()
stock_df["SMA_30"] = stock_df["Close"].rolling(window=30).mean()

# Fill NaNs (optional but avoids Plotly gaps)
stock_df["SMA_7"] = stock_df["SMA_7"].fillna(method="bfill")
stock_df["SMA_30"] = stock_df["SMA_30"].fillna(method="bfill")

# ----------------------------
# CHART
# ----------------------------
st.subheader("Price Chart")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=stock_df["Date"],
    y=stock_df["Close"],
    name="Close"
))

fig.add_trace(go.Scatter(
    x=stock_df["Date"],
    y=stock_df["SMA_7"],
    name="SMA 7"
))

fig.add_trace(go.Scatter(
    x=stock_df["Date"],
    y=stock_df["SMA_30"],
    name="SMA 30"
))

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# SENTIMENT
# ----------------------------
st.subheader("News Sentiment")

news = fetch_news(ticker)   # better: pass ticker so news matches stock
sent = analyze(news)

st.write("Average Sentiment Score:")
st.write(sent["compound"].mean())

st.subheader("Raw News Sentiment")
st.dataframe(sent)