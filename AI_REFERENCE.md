# AI Reference Documentation: Indonesian Stock Market Prediction System

This document serves as a technical reference for AI assistants, agents, or developers interacting with this codebase.

## 📌 Project Overview
This is a quantitative trading and machine learning pipeline designed to predict Indonesian stock movements. The system fetches daily stock data, performs technical feature engineering, trains a time-series cross-validated XGBoost classifier, runs backtests, and outputs the top daily stock picks. It also includes an optional Telegram Bot interface.

**Objective:** Predict the probability that a stock will increase by $\ge 2\%$ within the next 3 trading days and rank stocks by this probability.

## 📁 Repository Structure

```plaintext
C:\Ahmad\Bot\
├── backtest/
│   └── backtester.py          # Portfolio backtesting simulation (holding top 3 stocks for 3 days)
├── bot/
│   └── telegram_bot.py        # Telegram bot integration handling the '/top picks' command
├── data/
│   ├── data_loader.py         # Yahoo Finance (yfinance) data ingestion module
│   └── latest_picks.csv       # Automatically generated CSV containing today's top 5 predictions (ignored by git)
├── features/
│   └── feature_engineering.py # Technical indicators (RSI, MACD, SMA, BB, etc.) & label generation
├── model/
│   ├── xgboost_model.py       # XGBoost model training, TimeSeriesSplit validation, & prediction logic
│   └── xgboost_model.pkl      # Serialized trained model artifact
├── main.py                    # The central orchestrator script running the end-to-end pipeline
└── requirements.txt           # Python dependencies
```

## 🛠️ Technical Stack & Methodologies

1. **Data Ingestion:** 
   * Source: `yfinance`
   * Target: Top 30 highly liquid Indonesian stocks (LQ45/Popular like BBCA.JK, GOTO.JK, etc.)
   * Horizon: 5 years of daily OHLCV data.

2. **Feature Engineering (`ta` library):**
   * Trend & Momentum: `RSI(14)`, `MACD(12,26,9)`, `SMA(5, 10, 20, 50)`, `Price Momentum (Close/SMA_20)`
   * Volatility: `Bollinger Bands(20)`, Rolling Std Dev of Returns (7d, 14d)
   * Price/Volume: Daily returns (1d, 3d, 7d), Volume change % (1d)

3. **Target Labeling:**
   * Future Return: Calculated using `Close.shift(-3)`. 
   * Label Mapping: `1` if Future Return $\ge$ 0.02 (2%), `0` otherwise. Data leakage is prevented by dropping the last 3 days of historical data during training.

4. **Modeling (XGBoost):**
   * Algorithm: `XGBClassifier` optimized with `logloss`.
   * Cross-Validation: `TimeSeriesSplit(n_splits=5)` to strictly prevent look-ahead bias during evaluation.
   * Metrics Captured: Accuracy, Precision, Recall, ROC-AUC.

5. **Backtesting Strategy:**
   * **Rule:** Buy the top 3 highest-probability stocks daily. Hold each for 3 days.
   * **Execution Engine:** Uses 3 overlapping "tranches" to simulate daily capital allocation. 
   * **Metrics Computed:** Win Rate (is the average 3-day basket return > 0), Total Return, Max Drawdown.

6. **Telegram Bot Interface:**
   * Framework: `python-telegram-bot` (`async`/`await`).
   * Command: `/top picks` reads `data/latest_picks.csv` and returns the formatted top 5 stock probabilities.

## 🚀 Execution Instructions for AI / User

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Running the Pipeline
Run the orchestrator script to train the model, run the backtest, and generate predictions:
```bash
python main.py
```

### 3. Starting the Telegram Bot
To attach the Telegram interface, set the environment variable before running `main.py`:
**Windows (PowerShell):**
```powershell
$env:TELEGRAM_BOT_TOKEN="your_token"
python main.py
```
**Linux/macOS:**
```bash
export TELEGRAM_BOT_TOKEN="your_token"
python main.py
```

## ⚠️ Important AI Context Limits & Guidelines
* **No Data Leakage:** Ensure any modifications to `feature_engineering.py` maintain strict temporal boundaries. Do not use future data (via `.shift(-N)`) in any predictive features.
* **Model Serialization:** The `StockPredictor` class manages state via `joblib`. Ensure the path bindings between `main.py`, `xgboost_model.py`, and `backtester.py` regarding the `.pkl` artifact remain synchronized.
* **Timezones:** Indonesian stocks operate on WIB (UTC+7). Be mindful of this if intraday parsing is ever added. 
