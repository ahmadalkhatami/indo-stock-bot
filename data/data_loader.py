import time
import requests
import yfinance as yf
import pandas as pd
from typing import List

# Hardcoded IDX80 fallback (updated periodically by IDX; used if API is unreachable)
_FALLBACK_TICKERS = [
    'AALI.JK', 'ACES.JK', 'ADRO.JK', 'AGII.JK', 'AKRA.JK',
    'AMRT.JK', 'ANTM.JK', 'ARTO.JK', 'ASII.JK', 'BBCA.JK',
    'BBNI.JK', 'BBRI.JK', 'BBTN.JK', 'BFIN.JK', 'BJBR.JK',
    'BJTM.JK', 'BMRI.JK', 'BMTR.JK', 'BNGA.JK', 'BRPT.JK',
    'BSDE.JK', 'CPIN.JK', 'CTRA.JK', 'DMAS.JK', 'DSNG.JK',
    'EMTK.JK', 'ERAA.JK', 'ESSA.JK', 'EXCL.JK', 'GGRM.JK',
    'GOTO.JK', 'HEAL.JK', 'HMSP.JK', 'HRUM.JK', 'ICBP.JK',
    'INCO.JK', 'INDF.JK', 'INKP.JK', 'INTP.JK', 'ISAT.JK',
    'ITMG.JK', 'JPFA.JK', 'JSMR.JK', 'KLBF.JK', 'MAPI.JK',
    'MBMA.JK', 'MDKA.JK', 'MEDC.JK', 'MIKA.JK', 'MNCN.JK',
    'MPPA.JK', 'MTEL.JK', 'PGAS.JK', 'PGEO.JK', 'PNBN.JK',
    'PTBA.JK', 'PTPP.JK', 'PWON.JK', 'SCMA.JK', 'SIDO.JK',
    'SMGR.JK', 'SMRA.JK', 'SRTG.JK', 'TINS.JK', 'TKIM.JK',
    'TLKM.JK', 'TOWR.JK', 'TPIA.JK', 'UNTR.JK', 'UNVR.JK',
    'WIFI.JK', 'WIKA.JK', 'WSKT.JK', 'ITMG.JK', 'BBNI.JK',
    'INCO.JK', 'PTBA.JK', 'ADMR.JK', 'MAPI.JK', 'BUMI.JK',
]


def fetch_idx_tickers(index: str = "IDX80") -> List[str]:
    """
    Fetch IDX80 constituent tickers from Bursa Efek Indonesia.
    Returns Yahoo Finance-compatible symbols (e.g. 'BBCA.JK').
    Falls back to a hardcoded IDX80 list if the API is unreachable.
    """
    url = "https://www.idx.co.id/umbraco/Surface/StockData/GetSecuritiesData"
    params = {"start": 0, "length": 100, "indexCode": index}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json",
        "Referer": "https://www.idx.co.id/id/data-pasar/data-saham/daftar-saham/",
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        tickers = [
            f"{item['StockCode']}.JK"
            for item in data.get('data', [])
            if item.get('StockCode')
        ]
        if tickers:
            print(f"Fetched {len(tickers)} {index} tickers from IDX.")
            return tickers
        raise ValueError("Empty ticker list from IDX API.")
    except Exception as e:
        print(f"IDX API unavailable ({e}). Using fallback IDX80 list ({len(_FALLBACK_TICKERS)} tickers).")
        return list(dict.fromkeys(_FALLBACK_TICKERS))  # deduplicate, preserve order


def fetch_data(
    tickers: List[str],
    period: str = "5y",
    batch_size: int = 50,
    min_rows: int = 252,
) -> pd.DataFrame:
    """
    Fetch OHLCV data + macro features (USD/IDR, S&P 500) for all tickers.

    Downloads in batches of `batch_size` for speed (one HTTP request per batch
    instead of one per ticker). Tickers with fewer than `min_rows` trading days
    are dropped (likely illiquid or recently listed).
    """
    print(f"Fetching data for {len(tickers)} tickers in batches of {batch_size}...")
    all_data = []
    n_batches = (len(tickers) + batch_size - 1) // batch_size

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  Batch {batch_num}/{n_batches} ({len(batch)} tickers)...", end=" ", flush=True)

        try:
            raw = yf.download(batch, period=period, progress=False)
            if raw.empty:
                print("no data.")
                continue

            # yfinance always returns MultiIndex (Price, Ticker) for multi-ticker calls
            if not isinstance(raw.columns, pd.MultiIndex):
                # Single-ticker edge case
                raw.columns = pd.MultiIndex.from_tuples(
                    [(c, batch[0]) for c in raw.columns], names=['Price', 'Ticker']
                )

            available = raw.columns.get_level_values(1).unique().tolist()
            ok = 0
            for ticker in batch:
                if ticker not in available:
                    continue
                try:
                    df_t = raw.xs(ticker, level=1, axis=1).copy()
                    df_t = df_t.reset_index()
                    df_t['Date'] = pd.to_datetime(df_t['Date']).dt.tz_localize(None)
                    df_t['Ticker'] = ticker
                    # Drop illiquid / newly listed tickers
                    if len(df_t.dropna(subset=['Close'])) < min_rows:
                        continue
                    all_data.append(df_t)
                    ok += 1
                except Exception:
                    continue
            print(f"{ok} ok.")
        except Exception as e:
            print(f"batch failed: {e}")

        # Small pause to avoid hitting Yahoo Finance rate limits
        if batch_num < n_batches:
            time.sleep(0.5)

    if not all_data:
        return pd.DataFrame()

    df_stocks = pd.concat(all_data, ignore_index=True)
    print(f"Total: {df_stocks['Ticker'].nunique()} tickers with sufficient data.")

    # ── Macro data ────────────────────────────────────────────────────────────
    print("Fetching Macro Data (USD/IDR, S&P 500)...")
    macro_tickers = {"IDR=X": "USD_IDR", "^GSPC": "SP500"}
    macro_dfs = []
    for sym, name in macro_tickers.items():
        try:
            mdf = yf.download(sym, period=period, progress=False)
            if not mdf.empty:
                if isinstance(mdf.columns, pd.MultiIndex):
                    mdf.columns = mdf.columns.droplevel(1)
                mdf = mdf.reset_index()[['Date', 'Close']]
                mdf['Date'] = pd.to_datetime(mdf['Date']).dt.tz_localize(None)
                mdf = mdf.rename(columns={'Close': name})
                macro_dfs.append(mdf)
        except Exception as e:
            print(f"Failed to fetch macro {sym}: {e}")

    if macro_dfs:
        macro_merged = macro_dfs[0]
        for mdf in macro_dfs[1:]:
            macro_merged = pd.merge(macro_merged, mdf, on='Date', how='outer')
        macro_merged = macro_merged.sort_values('Date').ffill()

        df_stocks['Date'] = pd.to_datetime(df_stocks['Date']).dt.tz_localize(None)
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
