import pandas as pd
import numpy as np
import ta


FEATURE_COLS = [
    'return_1d', 'return_3d', 'return_7d',
    'volatility_7d', 'volatility_14d',
    'high_low_range', 'close_open',
    'volume_spike', 'volume_trend',
    'momentum_10', 'trend_strength',
    'rsi_14', 'macd_diff', 'bb_width',
    'close_zscore_20',
    'usd_idr_return', 'sp500_return',
    'foreign_flow_ratio', 'foreign_trend_7d',
    'sentiment_score',
    'sector_relative_return'
]


def add_features_and_labels(df: pd.DataFrame, sentiment_score: float = 0.0) -> pd.DataFrame:
    """
    Generate technical indicators, macro features, foreign flow, and target labels.

    Labels:
      - future_return_3d: continuous 3-day forward return
      - target_binary:    1 if future_return_3d > 0
      - target_strong:    1 if future_return_3d > 0.02
    """
    df = df.copy()
    df['sentiment_score'] = sentiment_score
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)

    # ── Sector Rotation logic ────────────────────────────────────────────────
    if 'Sektor' in df.columns:
        # Calculate daily return properly for each stock
        df['return_1d'] = df.groupby('Ticker')['Close'].pct_change()
        # Calculate sector average return per day
        df['sector_avg_return'] = df.groupby(['Sektor', 'Date'])['return_1d'].transform('mean')
        # Outperformance vs peers
        df['sector_relative_return'] = df['return_1d'] - df['sector_avg_return']
    else:
        df['sector_relative_return'] = 0.0

    processed = []

    for ticker, group in df.groupby('Ticker', sort=False):
        g = group.copy()
        close, high, low = g['Close'], g['High'], g['Low']
        open_, volume = g['Open'], g['Volume']

        # Foreign Flow Features
        if 'F_Net' in g.columns:
            # Ratio of net foreign flow vs total volume
            g['foreign_flow_ratio'] = g['F_Net'] / (volume + 1e-9)
            # 7-day cumulative foreign flow trend
            g['foreign_trend_7d'] = g['foreign_flow_ratio'].rolling(7).sum()
        else:
            g['foreign_flow_ratio'] = 0.0
            g['foreign_trend_7d'] = 0.0

        # Returns
        g['return_1d'] = close.pct_change(1, fill_method=None)
        g['return_3d'] = close.pct_change(3, fill_method=None)
        g['return_7d'] = close.pct_change(7, fill_method=None)

        # Rolling volatility of daily returns
        g['volatility_7d'] = g['return_1d'].rolling(7).std()
        g['volatility_14d'] = g['return_1d'].rolling(14).std()

        # Intraday range and body
        g['high_low_range'] = (high - low) / (close + 1e-9)
        g['close_open'] = (close - open_) / (open_ + 1e-9)

        # Volume features
        vol_ma20 = volume.rolling(20).mean()
        vol_ma5 = volume.rolling(5).mean()
        g['volume_spike'] = volume / (vol_ma20 + 1e-9)
        g['volume_trend'] = vol_ma5 / (vol_ma20 + 1e-9)

        # Momentum & trend
        g['momentum_10'] = close / (close.shift(10) + 1e-9)
        g['trend_strength'] = close / (close.rolling(50).mean() + 1e-9)

        # Technical indicators
        g['rsi_14'] = ta.momentum.RSIIndicator(close, window=14).rsi()
        macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        g['macd_diff'] = macd.macd_diff()
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        g['bb_width'] = bb.bollinger_wband()

        # Z-score of close
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        g['close_zscore_20'] = (close - sma20) / (std20 + 1e-9)

        # Macro & Sentiment
        g['usd_idr_return'] = g['USD_IDR'].pct_change(fill_method=None) if 'USD_IDR' in g.columns else 0.0
        g['sp500_return'] = g['SP500'].pct_change(fill_method=None) if 'SP500' in g.columns else 0.0
        g['sentiment_score'] = sentiment_score

        # Targets
        future_return = (close.shift(-3) - close) / close
        g['future_return_3d'] = future_return
        g['target_binary'] = (future_return > 0).astype(int)
        g['target_strong'] = (future_return > 0.02).astype(int)

        processed.append(g)

    out = pd.concat(processed, ignore_index=True)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out
