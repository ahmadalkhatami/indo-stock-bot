import yfinance as yf
import pandas as pd
from typing import List


def fetch_data(tickers: List[str], period: str = "5y") -> pd.DataFrame:
    """Fetch OHLCV data + macro features (USD/IDR, S&P 500) for given tickers."""
    print(f"Fetching data for {len(tickers)} tickers for the past {period}...")
    all_data = []

    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                df = df.reset_index()
                df['Ticker'] = ticker
                all_data.append(df)
        except Exception as e:
            print(f"Failed to fetch {ticker}: {e}")

    if not all_data:
        return pd.DataFrame()

    df_stocks = pd.concat(all_data, ignore_index=True)

    print("Fetching Macro Data (USD/IDR, S&P 500)...")
    macro_tickers = {"IDR=X": "USD_IDR", "^GSPC": "SP500"}
    macro_dfs = []
    for ticker, name in macro_tickers.items():
        try:
            mdf = yf.download(ticker, period=period, progress=False)
            if not mdf.empty:
                if isinstance(mdf.columns, pd.MultiIndex):
                    mdf.columns = mdf.columns.droplevel(1)
                mdf = mdf.reset_index()[['Date', 'Close']]
                mdf = mdf.rename(columns={'Close': name})
                macro_dfs.append(mdf)
        except Exception as e:
            print(f"Failed to fetch macro {ticker}: {e}")

    if macro_dfs:
        macro_merged = macro_dfs[0]
        for mdf in macro_dfs[1:]:
            macro_merged = pd.merge(macro_merged, mdf, on='Date', how='outer')
        macro_merged = macro_merged.sort_values('Date').ffill()

        df_stocks['Date'] = pd.to_datetime(df_stocks['Date']).dt.tz_localize(None)
        macro_merged['Date'] = pd.to_datetime(macro_merged['Date']).dt.tz_localize(None)
        df_stocks = pd.merge(df_stocks, macro_merged, on='Date', how='left')

        df_stocks = df_stocks.sort_values(['Ticker', 'Date'])
        macro_cols = [c for c in macro_merged.columns if c != 'Date']
        df_stocks[macro_cols] = df_stocks.groupby('Ticker', group_keys=False)[macro_cols].ffill()
        df_stocks[macro_cols] = df_stocks[macro_cols].bfill()

    return df_stocks


def fetch_benchmark(period: str = "5y", benchmark_ticker: str = "^JKSE") -> pd.DataFrame:
    """Fetch the IHSG (Jakarta Composite Index) as a buy-and-hold benchmark."""
    print(f"Fetching benchmark {benchmark_ticker}...")
    try:
        bdf = yf.download(benchmark_ticker, period=period, progress=False)
        if bdf.empty:
            return pd.DataFrame()
        if isinstance(bdf.columns, pd.MultiIndex):
            bdf.columns = bdf.columns.droplevel(1)
        bdf = bdf.reset_index()[['Date', 'Close']]
        bdf['Date'] = pd.to_datetime(bdf['Date']).dt.tz_localize(None)
        return bdf.rename(columns={'Close': 'Benchmark_Close'})
    except Exception as e:
        print(f"Failed to fetch benchmark: {e}")
        return pd.DataFrame()
