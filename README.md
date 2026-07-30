# AI Market Intelligence System

An end-to-end system for collecting market data and alternative data (news + social sentiment), engineering features, training ML models to generate market signals, and serving predictions through an API and an interactive dashboard.

## Overview

The system pulls live/historical price data (via `yfinance`) alongside news and Reddit discussion, processes it into model-ready features, and trains machine learning models (scikit-learn / XGBoost, with support for transformer-based NLP) to produce market intelligence — e.g. trend/sentiment signals that feed into a trading layer. Results are exposed through a FastAPI backend and visualized in a Streamlit dashboard. The whole stack is containerized with Docker.

## Architecture

```
ingestion/    → pulls raw data: market prices (yfinance), news (News API), Reddit posts/comments
processing/   → cleans and transforms raw data into engineered features
ml/           → model training / experimentation pipeline
models/       → serialized, trained model artifacts (e.g. via joblib)
api/          → FastAPI service exposing predictions/insights as REST endpoints
dashboard/    → Streamlit app for visualizing signals, features, and predictions
trading/      → strategy/signal logic built on top of model outputs
scripts/      → helper/automation scripts (data refresh, training runs, etc.)
docker/       → containerization assets
config.py     → environment-based configuration (API keys, credentials)
```

## Tech Stack

- **Backend API:** FastAPI + Uvicorn
- **Dashboard:** Streamlit + Plotly
- **Data:** yfinance (market data), News API, Reddit API (PRAW-style credentials)
- **ML/NLP:** scikit-learn, XGBoost, PyTorch, Hugging Face Transformers
- **Data handling:** pandas, NumPy, joblib
- **Deployment:** Docker

## Prerequisites

- Python 3.11+
- API credentials for:
  - [News API](https://newsapi.org/) (`NEWS_API_KEY`)
  - A Reddit app for API access (`REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`)

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/saeee2510/ai-market-intelligence-system.git
   cd ai-market-intelligence-system
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   Create a `.env` file in the project root:
   ```env
   NEWS_API_KEY=your_news_api_key
   REDDIT_CLIENT_ID=your_reddit_client_id
   REDDIT_CLIENT_SECRET=your_reddit_client_secret
   REDDIT_USER_AGENT=your_reddit_user_agent
   ```

## Running the Project

### Run the API locally

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000` (interactive docs at `/docs`).

### Run the dashboard

```bash
streamlit run dashboard/app.py
```

> Adjust the path above if your Streamlit entry point file has a different name.

### Run with Docker

```bash
docker build -t ai-market-intelligence-system .
docker run -p 8000:8000 --env-file .env ai-market-intelligence-system
```

## Project Workflow

1. **Ingest** — `ingestion/` scripts pull market prices and alternative data (news articles, Reddit posts).
2. **Process** — `processing/` cleans the raw data and builds features (e.g. technical indicators, sentiment scores).
3. **Model** — `ml/` trains and evaluates models; artifacts are saved to `models/`.
4. **Serve** — `api/` exposes model outputs as endpoints; `trading/` consumes signals for strategy logic.
5. **Visualize** — `dashboard/` renders charts and insights for interactive exploration.

## Roadmap Ideas

- [ ] Add automated data refresh scheduling
- [ ] Expand model evaluation/backtesting metrics
- [ ] Add authentication to the API
- [ ] CI/CD pipeline for automated testing and deployment

## Contributing

Contributions are welcome! Please open an issue to discuss proposed changes, or submit a pull request.

