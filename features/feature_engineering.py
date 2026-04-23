import pandas as pd
import numpy as np
import ta

def add_features_and_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate technical indicators, macro features, and target labels.
    """
    # Ensure data is sorted by Ticker and Date to avoid leakage
    df = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)
    
    processed_dfs = []
    
    for ticker, group in df.groupby('Ticker'):
        group = group.copy()
        
        # --- Feature Engineering ---
        close = group['Close']
        high = group['High']
        low = group['Low']
        volume = group['Volume']
        
        # RSI (14)
        group['RSI_14'] = ta.momentum.RSIIndicator(close, window=14).rsi()
        
        # MACD (12,26,9)
        macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        group['MACD'] = macd.macd()
        group['MACD_Signal'] = macd.macd_signal()
        group['MACD_Diff'] = macd.macd_diff()
        
        # SMA (5, 10, 20, 50)
        for window in [5, 10, 20, 50]:
            group[f'SMA_{window}'] = ta.trend.SMAIndicator(close, window=window).sma_indicator()
            
        # Bollinger Bands (20)
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        group['BB_High'] = bb.bollinger_hband()
        group['BB_Low'] = bb.bollinger_lband()
        group['BB_Width'] = bb.bollinger_wband()
        
        # Daily return (1d, 3d, 7d)
        group['Return_1d'] = close.pct_change(1)
        group['Return_3d'] = close.pct_change(3)
        group['Return_7d'] = close.pct_change(7)
        
        # Volatility (rolling std 7d, 14d)
        group['Vol_7d'] = group['Return_1d'].rolling(window=7).std()
        group['Vol_14d'] = group['Return_1d'].rolling(window=14).std()
        
        # Volume change (%)
        group['Volume_Change_1d'] = volume.pct_change(1)
        
        # Price momentum (Close / SMA_20)
        group['Momentum_20'] = close / group['SMA_20'] - 1
        
        # Statistical Feature: Z-Score of Close Price
        group['Close_ZScore_20'] = (close - group['SMA_20']) / close.rolling(window=20).std()
        
        # Macro Features
        if 'USD_IDR' in group.columns and 'SP500' in group.columns:
            group['USD_IDR_Return'] = group['USD_IDR'].pct_change()
            group['SP500_Return'] = group['SP500'].pct_change()
        else:
            group['USD_IDR_Return'] = 0.0
            group['SP500_Return'] = 0.0
        
        # --- Target Variable ---
        # Predict probability that stock will increase by at least 2% within next 3 trading days
        group['Future_Return_3d'] = close.shift(-3) / close - 1
        
        # Label: 1 if future return >= +2% (0.02), 0 otherwise
        group['Target'] = (group['Future_Return_3d'] >= 0.02).astype(int)
        
        processed_dfs.append(group)
        
    final_df = pd.concat(processed_dfs, ignore_index=True)
    return final_df
