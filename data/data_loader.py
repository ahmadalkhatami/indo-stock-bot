import yfinance as yf
import pandas as pd
from typing import List

def fetch_data(tickers: List[str], period: str = "5y") -> pd.DataFrame:
    """
    Fetch OHLCV data for given tickers from Yahoo Finance.
    """
    print(f"Fetching data for {len(tickers)} tickers for the past {period}...")
    all_data = []
    
    for ticker in tickers:
        try:
            # yf.download can be noisy, but it's the most reliable way to get history
            df = yf.download(ticker, period=period, progress=False)
            if not df.empty:
                # Handle potential multi-index columns returned by newer yfinance versions
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                    
                df = df.reset_index()
                df['Ticker'] = ticker
                all_data.append(df)
        except Exception as e:
            print(f"Failed to fetch {ticker}: {e}")
            
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()
