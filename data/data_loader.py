import time
import time
try:
    from curl_cffi import requests
    IMPERSONATE = True
except ImportError:
    import requests
    IMPERSONATE = False
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
    Fetch LIVE tickers from Wikipedia (100% Stable & Anti-Block).
    """
    import json
    import os
    import re
    
    cache_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "ticker_cache.json")
    
    # Wikipedia is very stable for LQ45
    url = "https://id.wikipedia.org/wiki/LQ45"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        raw_html = resp.text
        
        # Mencari ticker 4 huruf di dalam tabel (Pola: ADRO, BBCA, dsb)
        # Wikipedia biasanya menggunakan tag <td><a ...>BBCA</a></td>
        symbols = re.findall(r'<td>([A-Z]{4})</td>', raw_html)
        
        # Jika tidak ketemu, cari di dalam link <a>BBCA</a>
        if not symbols:
            symbols = re.findall(r'title="[^"]+">([A-Z]{4})</a>', raw_html)

        full_tickers = [f"{s}.JK" for s in symbols if len(s) == 4]
        
        # Unique list
        full_tickers = list(dict.fromkeys(full_tickers))

        if len(full_tickers) >= 40: # LQ45 harus ada minimal 45 (atau sekitarnya)
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump(full_tickers, f)
            print(f"Success! Fetched {len(full_tickers)} LIVE tickers from Wikipedia.")
            return full_tickers
            
    except Exception as e:
        print(f"Wikipedia fetch failed ({e}). Checking local cache...")
        
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cached_tickers = json.load(f)
            return cached_tickers
        except:
            pass

    return list(dict.fromkeys(_FALLBACK_TICKERS))


def fetch_foreign_flow() -> pd.DataFrame:
    """
    Fetch daily net foreign transaction data from Kontan Data Center.
    Returns a DataFrame with columns: ['Ticker', 'F_Net']
    """
    print("Fetching Foreign Flow data from Kontan...")
    url = "https://pusatdata.kontan.co.id/market/rekap_data/saham/daily/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
    }
    
    try:
        if IMPERSONATE:
            resp = requests.get(url, headers=headers, impersonate="chrome110", timeout=20)
        else:
            resp = requests.get(url, headers=headers, timeout=20)
        
        resp.raise_for_status()
        
        # Parse all tables
        tables = pd.read_html(resp.text)
        
        # Cari tabel yang memiliki kolom cukup banyak (biasanya > 11 kolom)
        df_ff = None
        for t in tables:
            if t.shape[1] >= 11:
                df_ff = t
                break
        
        if df_ff is None:
            return pd.DataFrame()
        
        # Berdasarkan inspeksi: Indeks 1: Kode, Indeks 9: Buy, Indeks 10: Sell
        # Gunakan iloc untuk keamanan jika nama kolom berubah
        df_ff = df_ff.iloc[:, [1, 9, 10]] 
        df_ff.columns = ['Ticker', 'F_Buy', 'F_Sell']
        
        # Bersihkan data
        df_ff['Ticker'] = df_ff['Ticker'].astype(str).str.strip().str.upper() + ".JK"
        # Hilangkan titik (.) ribuan jika ada agar bisa dikonversi ke angka
        df_ff['F_Buy'] = df_ff['F_Buy'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        df_ff['F_Sell'] = df_ff['F_Sell'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        
        df_ff['F_Net'] = pd.to_numeric(df_ff['F_Buy'], errors='coerce').fillna(0) - \
                         pd.to_numeric(df_ff['F_Sell'], errors='coerce').fillna(0)
        
        print(f"  Successfully fetched foreign flow for {len(df_ff)} stocks.")
        return df_ff[['Ticker', 'F_Net']]
    except Exception as e:
        print(f"  Failed to fetch foreign flow: {e}")
        return pd.DataFrame()

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

    # ── Foreign Flow ──────────────────────────────────────────────────────────
    df_ff = fetch_foreign_flow()
    if not df_ff.empty:
        df_stocks = pd.merge(df_stocks, df_ff, on='Ticker', how='left')
        df_stocks['F_Net'] = df_stocks['F_Net'].fillna(0)
        print("  Foreign Flow integrated into datasets.")

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
