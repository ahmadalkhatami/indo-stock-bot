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
]


def add_features_and_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate technical indicators, macro features, and target labels.

    Labels:
      - future_return_3d: continuous 3-day forward return
      - target_binary:    1 if future_return_3d > 0
      - target_strong:    1 if future_return_3d > 0.02
    """
    df = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)
    processed = []

    for ticker, group in df.groupby('Ticker', sort=False):
        g = group.copy()
        close, high, low = g['Close'], g['High'], g['Low']
        open_, volume = g['Open'], g['Volume']

        # Returns
        g['return_1d'] = close.pct_change(1)
        g['return_3d'] = close.pct_change(3)
        g['return_7d'] = close.pct_change(7)

        # Rolling volatility of daily returns
        g['volatility_7d'] = g['return_1d'].rolling(7).std()
        g['volatility_14d'] = g['return_1d'].rolling(14).std()

        # Intraday range and body
        g['high_low_range'] = (high - low) / close
        g['close_open'] = (close - open_) / open_

        # Volume features
        vol_ma20 = volume.rolling(20).mean()
        vol_ma5 = volume.rolling(5).mean()
        g['volume_spike'] = volume / vol_ma20
        g['volume_trend'] = vol_ma5 / vol_ma20

        # Momentum & trend
        g['momentum_10'] = close / close.shift(10)
        g['trend_strength'] = close / close.rolling(50).mean()

        # Technical indicators
        g['rsi_14'] = ta.momentum.RSIIndicator(close, window=14).rsi()
        macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        g['macd_diff'] = macd.macd_diff()
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        g['bb_width'] = bb.bollinger_wband()

        # Z-score of close
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        g['close_zscore_20'] = (close - sma20) / std20

        # Macro returns
        g['usd_idr_return'] = g['USD_IDR'].pct_change() if 'USD_IDR' in g.columns else 0.0
        g['sp500_return'] = g['SP500'].pct_change() if 'SP500' in g.columns else 0.0

        # Targets
        future_return = (close.shift(-3) - close) / close
        g['future_return_3d'] = future_return
        g['target_binary'] = (future_return > 0).astype(int)
        g['target_strong'] = (future_return > 0.02).astype(int)

        processed.append(g)

    out = pd.concat(processed, ignore_index=True)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out
