import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from ingestion.stock_data import fetch_stock_data
from processing.feature_engineering import add_features
from processing.sentiment import get_finbert_sentiment
from processing.create_labels import create_labels
from ml.train_xgboost import train_model

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="AI Market Intelligence", layout="wide")

st.title("📊 AI Market Intelligence Platform")


# -------------------------
# SIDEBAR INPUT
# -------------------------
ticker = st.sidebar.text_input("Enter Ticker", "AAPL")


# -------------------------
# LOAD DATA (SAFE PIPELINE)
# -------------------------
@st.cache_data
def load_data(ticker):
    df = fetch_stock_data(ticker)
    df = add_features(df)
    df = df.dropna()

    # IMPORTANT FIX: create labels BEFORE training
    df = create_labels(df)

    return df


df = load_data(ticker)


# -------------------------
# CHART
# -------------------------
st.subheader("📈 Stock Price")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df.index,
    y=df["Close"],
    name="Close Price"
))

st.plotly_chart(fig, use_container_width=True)


# -------------------------
# MODEL TRAINING
# -------------------------
st.subheader("🤖 Model Prediction")

# IMPORTANT FIX: train only if label exists
if "label" not in df.columns:
    st.error("Label column missing. Check create_labels()")
else:
    model, X_test, y_test = train_model(df)

    latest_features = X_test.iloc[-1].values.reshape(1, -1)

    prediction = model.predict(latest_features)[0]
    proba = model.predict_proba(latest_features)[0][1]

    # SIGNAL LOGIC
    if proba > 0.65:
        signal = "🟢 BUY"
    elif proba < 0.35:
        signal = "🔴 SELL"
    else:
        signal = "🟡 HOLD"

    st.write("Prediction:", signal)
    st.write("Confidence:", round(proba, 3))


# -------------------------
# SENTIMENT
# -------------------------
st.subheader("🧠 Sentiment Score")

sentiment = get_finbert_sentiment([ticker + " stock news"])
st.write("Sentiment Score:", sentiment)


# -------------------------
# INFO
# -------------------------
st.subheader("📊 Model Insights")

st.write("Run training logs in terminal for:")
st.write("- Sharpe Ratio")
st.write("- Total Return")
st.write("- Drawdown")