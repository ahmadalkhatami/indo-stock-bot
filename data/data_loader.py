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


def fetch_ticker_metadata(index: str = "LQ45") -> pd.DataFrame:
    """
    Fetch LIVE tickers AND Sectors from Wikipedia (LQ45).
    """
    import pandas as pd
    url = "https://id.wikipedia.org/wiki/LQ45"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        tables = pd.read_html(url)
        df_wiki = None
        for t in tables:
            if 'Kode' in t.columns:
                df_wiki = t
                break
        if df_wiki is not None:
            sector_col = None
            for col in ['Sektor', 'Industri', 'Klasifikasi']:
                if col in df_wiki.columns:
                    sector_col = col
                    break
            if sector_col:
                df_wiki = df_wiki[['Kode', sector_col]].copy()
                df_wiki = df_wiki.rename(columns={'Kode': 'Ticker', sector_col: 'Sektor'})
                df_wiki['Ticker'] = df_wiki['Ticker'].str.strip().str.upper() + ".JK"
                return df_wiki
            else:
                df_wiki = df_wiki[['Kode']].copy()
                df_wiki = df_wiki.rename(columns={'Kode': 'Ticker'})
                df_wiki['Ticker'] = df_wiki['Ticker'].str.strip().str.upper() + ".JK"
                df_wiki['Sektor'] = 'Lainnya'
                return df_wiki
    except Exception as e:
        print(f"Wikipedia sector fetch failed ({e}). Use fallback.")
    fallback_map = {
        'BBCA.JK': 'Finance', 'BBRI.JK': 'Finance', 'BMRI.JK': 'Finance', 'BBNI.JK': 'Finance',
        'TLKM.JK': 'Communication', 'ASII.JK': 'Industrial', 'UNVR.JK': 'Consumer', 
        'ADRO.JK': 'Energy', 'PTBA.JK': 'Energy', 'ITMG.JK': 'Energy', 
        'ANTM.JK': 'Basic Materials', 'MDKA.JK': 'Basic Materials'
    }
    data = [{'Ticker': t, 'Sektor': fallback_map.get(t, 'Lainnya')} for t in _FALLBACK_TICKERS]
    return pd.DataFrame(data)

def fetch_idx_tickers(index: str = "IDX80") -> List[str]:
    df = fetch_ticker_metadata(index)
    return df['Ticker'].tolist()

def fetch_foreign_flow() -> pd.DataFrame:
    print("Fetching Foreign Flow data from Kontan...")
    url = "https://pusatdata.kontan.co.id/market/rekap_data/saham/daily/"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        if IMPERSONATE:
            resp = requests.get(url, headers=headers, impersonate="chrome110", timeout=20)
        else:
            resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
        df_ff = None
        for t in tables:
            if t.shape[1] >= 11:
                df_ff = t
                break
        if df_ff is None: return pd.DataFrame()
        df_ff = df_ff.iloc[:, [1, 9, 10]] 
        df_ff.columns = ['Ticker', 'F_Buy', 'F_Sell']
        df_ff['Ticker'] = df_ff['Ticker'].astype(str).str.strip().str.upper() + ".JK"
        df_ff['F_Buy'] = df_ff['F_Buy'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        df_ff['F_Sell'] = df_ff['F_Sell'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        df_ff['F_Net'] = pd.to_numeric(df_ff['F_Buy'], errors='coerce').fillna(0) - pd.to_numeric(df_ff['F_Sell'], errors='coerce').fillna(0)
        return df_ff[['Ticker', 'F_Net']]
    except Exception as e:
        print(f"Failed to fetch foreign flow: {e}"); return pd.DataFrame()

def fetch_data(tickers: List[str], period: str = "5y", batch_size: int = 50, min_rows: int = 252) -> pd.DataFrame:
    print(f"Fetching data for {len(tickers)} tickers...")
    all_data = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            raw = yf.download(batch, period=period, progress=False)
            if raw.empty: continue
            if not isinstance(raw.columns, pd.MultiIndex):
                raw.columns = pd.MultiIndex.from_tuples([(c, batch[0]) for c in raw.columns], names=['Price', 'Ticker'])
            available = raw.columns.get_level_values(1).unique().tolist()
            for ticker in batch:
                if ticker not in available: continue
                try:
                    df_t = raw.xs(ticker, level=1, axis=1).copy()
                    df_t = df_t.reset_index()
                    df_t['Date'] = pd.to_datetime(df_t['Date']).dt.tz_localize(None)
                    df_t['Ticker'] = ticker
                    if len(df_t.dropna(subset=['Close'])) < min_rows: continue
                    all_data.append(df_t)
                except: continue
        except: continue
        time.sleep(0.5)
    if not all_data: return pd.DataFrame()
    df_stocks = pd.concat(all_data, ignore_index=True)
    
    # Macro Data
    macro_tickers = {"IDR=X": "USD_IDR", "^GSPC": "SP500"}
    for sym, name in macro_tickers.items():
        try:
            mdf = yf.download(sym, period=period, progress=False)
            if not mdf.empty:
                if isinstance(mdf.columns, pd.MultiIndex): mdf.columns = mdf.columns.droplevel(1)
                mdf = mdf.reset_index()[['Date', 'Close']]
                mdf['Date'] = pd.to_datetime(mdf['Date']).dt.tz_localize(None)
                mdf = mdf.rename(columns={'Close': name})
                df_stocks = pd.merge(df_stocks, mdf, on='Date', how='left')
        except: pass
    
    # Foreign Flow
    df_ff = fetch_foreign_flow()
    if not df_ff.empty:
        df_stocks = pd.merge(df_stocks, df_ff, on='Ticker', how='left')
        df_stocks['F_Net'] = df_stocks['F_Net'].fillna(0)
    
    # Sector Metadata
    df_meta = fetch_ticker_metadata()
    if not df_meta.empty:
        df_stocks = pd.merge(df_stocks, df_meta, on='Ticker', how='left')
        df_stocks['Sektor'] = df_stocks['Sektor'].fillna('Lainnya')
        print(f"  Sector information integrated for {df_stocks['Sektor'].nunique()} sectors.")

    return df_stocks

def fetch_benchmark(period: str = "5y", benchmark_ticker: str = "^JKSE") -> pd.DataFrame:
    try:
        bdf = yf.download(benchmark_ticker, period=period, progress=False)
        if bdf.empty: return pd.DataFrame()
        if isinstance(bdf.columns, pd.MultiIndex): bdf.columns = bdf.columns.droplevel(1)
        bdf = bdf.reset_index()[['Date', 'Close']]
        bdf['Date'] = pd.to_datetime(bdf['Date']).dt.tz_localize(None)
        return bdf.rename(columns={'Close': 'Benchmark_Close'})
    except: return pd.DataFrame()
