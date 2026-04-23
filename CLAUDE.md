# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

A quantitative trading pipeline for Indonesian stocks (IDX). It fetches 5 years of OHLCV data for 30 tickers, engineers 21 technical/macro features, trains an XGBoost classifier to predict whether a stock will gain ≥2% within 3 trading days, backtests a daily top-3 portfolio strategy, and exposes results via a Telegram bot and Streamlit dashboard.

## Common Commands

```bash
# Run the full pipeline (fetch data → train → backtest → predict → optionally start bot)
python main.py

# Run Telegram bot (set token first)
export TELEGRAM_BOT_TOKEN="your_bot_token"
python main.py

# Launch Streamlit dashboard
streamlit run dashboard.py
```

## Architecture

### Data Flow

```
main.py
  → data_loader.fetch_data()          # yfinance: 30 .JK tickers + USD/IDR + S&P500
  → feature_engineering.add_features_and_labels()  # 21 features + binary label
  → StockPredictor.train_and_evaluate()  # XGBoost + TimeSeriesSplit(n_splits=5)
  → run_backtest()                    # Simulated top-3 daily strategy
  → StockPredictor.predict_today()   # Writes data/latest_picks.csv + data/predictions.db
  → run_bot() [optional]             # Async Telegram bot
```

### Modules

| Module | File | Responsibility |
|---|---|---|
| Data | `data/data_loader.py` | Yahoo Finance ingestion, macro data, timezone normalization |
| Features | `features/feature_engineering.py` | RSI, MACD, Bollinger Bands, SMA, rolling stats, returns, macro returns |
| Model | `model/xgboost_model.py` | XGBClassifier training, joblib serialization to `model/xgboost_model.pkl` |
| Backtest | `backtest/backtester.py` | 3-tranche daily portfolio sim, 0.4% round-trip fees, drawdown |
| Bot | `bot/telegram_bot.py` | Async handlers for `/start` and `/top picks`, reads `latest_picks.csv` |
| Dashboard | `dashboard.py` | Streamlit UI reading `predictions.db` for history |

## Key Invariants to Preserve

**No data leakage:** The label is `Future_Return_3d = (Close.shift(-3) / Close) - 1`. Training data drops the last 3 rows per ticker so no future prices appear in features.

**Temporal ordering:** All cross-validation uses `TimeSeriesSplit` — never shuffle time-series data.

**Model path consistency:** `model/xgboost_model.pkl` is the single artifact referenced by both backtester and `predict_today()`. Don't move or rename it without updating all references.

## Environment & Dependencies

No `.env` file — configure via environment variables:
- `TELEGRAM_BOT_TOKEN` — required only to start the Telegram bot

Install dependencies:
```bash
pip install -r requirements.txt
```

Key packages: `yfinance`, `ta`, `xgboost`, `scikit-learn`, `joblib`, `python-telegram-bot`, `streamlit`, `sqlalchemy`

## Outputs

- `data/latest_picks.csv` — today's top 5 predictions (overwritten each run)
- `data/predictions.db` — SQLite historical log of all predictions
- `model/xgboost_model.pkl` — trained model artifact (overwritten each run)
