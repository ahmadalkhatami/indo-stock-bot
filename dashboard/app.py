import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import subprocess
import sys
import time
import json
import requests
from streamlit_lottie import st_lottie

ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(ROOT, 'data', 'artifacts')
PICKS_FILE   = os.path.join(ROOT, 'data', 'latest_picks.csv')
LOGS_DIR     = os.path.join(ROOT, 'logs')

st.set_page_config(page_title="Indo Stock Bot v2.0 - Premium", layout="wide", initial_sidebar_state="expanded")

# --- UTILS ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# --- PREMIUM STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    .stApp {
        background: radial-gradient(circle at top left, #1a1f2c, #0d1117);
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }
    
    /* Glassmorphism Containers */
    div[data-testid="stMetric"] {
        background: rgba(22, 27, 34, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(48, 54, 61, 0.5);
        border-radius: 12px;
        padding: 15px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    [data-testid="stMetricValue"] {
        color: #58a6ff !important;
        font-weight: 800 !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(13, 17, 23, 0.9) !important;
        border-right: 1px solid #30363d;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #238636 0%, #2ea043 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(46, 160, 67, 0.4);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 4px;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_artifacts():
    paths = {
        'equity':   os.path.join(ARTIFACT_DIR, 'equity_curve.csv'),
        'trades':   os.path.join(ARTIFACT_DIR, 'trades.csv'),
        'benchmark':os.path.join(ARTIFACT_DIR, 'benchmark_curve.csv'),
        'metrics':  os.path.join(ARTIFACT_DIR, 'metrics.csv'),
        'roc':      os.path.join(ARTIFACT_DIR, 'roc.csv'),
        'feat_imp': os.path.join(ARTIFACT_DIR, 'feature_importances.csv'),
    }
    return {k: (pd.read_csv(p) if os.path.exists(p) else None) for k, p in paths.items()}

@st.cache_data
def load_picks():
    return pd.read_csv(PICKS_FILE) if os.path.exists(PICKS_FILE) else pd.DataFrame()

def _pct(x, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "N/A"
    return f"{x*100:+.2f}%" if sign else f"{x*100:.2f}%"

def _inf_fmt(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "N/A"
    return "∞" if x == float('inf') else f"{x:.2f}"

# --- SIDEBAR ---
st.sidebar.title("💎 IndoStockBot v2.0")
lottie_trading = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_m6o6tpuk.json")
if lottie_trading: st_lottie(lottie_trading, height=150, key="sidebar_lottie")

page = st.sidebar.radio("Navigation", [
    "🚀 Market Overview",
    "📈 Equity & Analytics",
    "🎯 Top Predictions",
    "📊 Stock Chart Explorer",
    "📖 Trading Logbook",
    "🛡️ Backtest Report",
    "🤖 Model Health",
    "📋 System Logs"
])

st.sidebar.divider()
st.sidebar.caption("System Status: Operational")

arts = load_artifacts()
picks = load_picks()

# ─── NAVIGATION LOGIC ────────────────────────────────────────────────────────

if page == "🚀 Market Overview":
    st.title("Market Overview")
    if arts['metrics'] is None or arts['metrics'].empty:
        st.warning("No metrics found or metrics file is empty. Run AI Prediction first.")
        st.stop()

    m = arts['metrics'].iloc[0].to_dict()
    
    with st.sidebar:
        st.subheader("Control Center")
        
        # --- SCHEDULER SECTION ---
        st.write("⏰ **Auto Scheduler**")
        config_path = os.path.join(ROOT, "data", "scheduler_config.json")
        
        # Load existing config
        current_sched = {"time": "16:30", "active": True}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                current_sched = json.load(f)
        
        col_s1, col_s2 = st.columns([2, 1])
        with col_s1:
            new_time = st.text_input("Run Time (WIB)", value=current_sched.get("time", "16:30"), label_visibility="collapsed")
        with col_s2:
            new_active = st.toggle("On", value=current_sched.get("active", True), label_visibility="collapsed")
        
        # Save on change
        if new_time != current_sched.get("time") or new_active != current_sched.get("active"):
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                json.dump({"time": new_time, "active": new_active}, f)
            st.toast("Scheduler updated!", icon="⏰")

        st.divider()

        # --- MANUAL RUN ---
        if st.button("🚀 EXECUTE AI PIPELINE", use_container_width=True):
            with st.status("Initializing AI engine...", expanded=True) as status:
                process = subprocess.Popen(
                    [sys.executable, os.path.join(ROOT, "main.py")],
                    cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )
                for line in process.stdout: st.text(line.strip())
                process.wait()
                status.update(label="Analysis Finished!", state="complete")
                st.success("Strategy refreshed.")
                time.sleep(1.5)
                st.rerun()

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Return", _pct(m.get('total_return'), True), help="Total keuntungan/kerugian (Profit/Loss). Angka positif (+) berarti strategi ini menghasilkan uang.")
    c2.metric("Win Rate", _pct(m.get('win_rate')), help="Tingkat akurasi. 50% berarti dari 10 kali beli, 5 di antaranya profit.")
    c3.metric("Sharpe Ratio", f"{m.get('sharpe_ratio', 0):.2f}", help="Tingkat 'Kesehatan' profit. Di atas 1.0 sangat bagus, di bawah 0 berarti terlalu berisiko.")
    c4.metric("Max Drawdown", _pct(m.get('max_drawdown')), help="Risiko Terburuk. Persentase penurunan modal terbesar yang pernah dialami AI selama pengetesan.")

    # Benchmark Comparison
    st.markdown("---")
    st.subheader("Benchmark Comparison (IHSG)")
    bench_ret = m.get('benchmark_total_return', 0)
    alpha = m.get('total_return', 0) - float(bench_ret)
    b1, b2, b3 = st.columns(3)
    b1.metric("Alpha (Nilai Tambah AI)", _pct(alpha, True), help="Seberapa jauh AI mengalahkan IHSG (pasar umum). Angka positif berarti AI bekerja lebih baik daripada beli diam (Buy & Hold).")
    b2.metric("IHSG Return", _pct(float(bench_ret)), help="Kenaikan/Penurunan IHSG sebagai pembanding standar.")
    b3.metric("Total Trades", int(m.get('num_trades', 0)), help="Berapa kali AI melakukan transaksi beli/jual selama ini.")

elif page == "📈 Equity & Analytics":
    st.title("Equity Analytics")
    eq = arts['equity']
    if eq is not None:
        eq['Date'] = pd.to_datetime(eq['Date'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq['Date'], y=eq['equity'], name='Strategy', line=dict(color='#00d4ff', width=3)))
        if arts['benchmark'] is not None:
            bench = arts['benchmark']
            bench['Date'] = pd.to_datetime(bench['Date'])
            fig.add_trace(go.Scatter(x=bench['Date'], y=bench['benchmark_equity'], name='IHSG', line=dict(color='#ff9f43', dash='dash')))
        fig.update_layout(template='plotly_dark', height=500, title="Growth of 100M IDR")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run backtest to see results.")

elif page == "🎯 Top Predictions":
    st.title("Top AI Picks")
    if picks.empty:
        st.info("No picks available. Please run the AI pipeline from the Overview page.")
    else:
        with st.expander("💡 Cara Membaca Rekomendasi (Panduan Pemula)", expanded=True):
            st.markdown("""
            * **Ticker**: Kode saham (contoh: *BBCA.JK* adalah BCA).
            * **Probability**: Tingkat keyakinan AI. Angka **0.75** berarti AI yakin **75%** harga akan naik.
            * **Signal**: Fokus HANYA pada saham dengan status **BUY**.
            * **Sektor**: Melihat tren. Jika banyak rekomendasi BUY dari sektor *Energy*, berarti sektor energi sedang jadi favorit pasar (arus uang masuk ke sana).
            * **Saran untuk Pemula**: Jangan gunakan seluruh modal untuk 1 saham. Bagi rata (diversifikasi) ke beberapa rekomendasi teratas untuk keamanan.
            """)
            
        # Pre-process picks to ensure numerical columns for styling
        if 'probability' in picks.columns:
            picks['probability'] = pd.to_numeric(picks['probability'], errors='coerce')
        
        try:
            st.dataframe(
                picks.style.background_gradient(subset=['probability'], cmap='Greens'),
                use_container_width=True
            )
        except:
            # Silent fallback for production: clean UI even if styling encounters edge cases
            st.dataframe(picks, use_container_width=True)

elif page == "🛡️ Backtest Report":
    st.title("Strategy Forensics")
    trades = arts['trades']
    if trades is not None and not trades.empty:
        st.subheader(f"Trade History ({len(trades)} Positions)")
        st.dataframe(trades, use_container_width=True)
    else:
        st.info("No trades to display.")

elif page == "🤖 Model Health":
    st.title("AI Model Diagnostic")
    
    with st.expander("📚 Apa Arti Skor Ini?"):
        st.markdown("""
        *Halaman ini khusus untuk mengecek 'seberapa pintar' AI saat ini berdasarkan tes ujian sejarah.*
        * **AUC-ROC**: Skor ujian keseluruhan AI (Maksimal 1.0). Di atas 0.55 berarti AI lebih pintar dari sekadar menebak koin (50:50).
        * **Precision**: Akurasi Sinyal Beli. Jika Precision 0.60, berarti dari 10 kali AI menyuruh beli, 6 di antaranya benar-benar naik.
        * **Recall**: Kemampuan AI menemukan momentum. Semakin tinggi, semakin AI pandai 'tidak kelewatan' saham bagus.
        * **Feature Importance**: Tabel yang menunjukkan data apa yang paling diandalkan AI saat ini (misal: Volume, Sentimen, atau Tren Asing).
        """)

    if arts['metrics'] is not None:
        m = arts['metrics'].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("AUC-ROC", f"{m['roc_auc']:.4f}")
        c2.metric("Precision", f"{m['precision']:.4f}")
        c3.metric("Recall", f"{m['recall']:.4f}")
        
    if arts['feat_imp'] is not None:
        st.subheader("Feature Importance")
        st.bar_chart(arts['feat_imp'].set_index('feature')['importance'])

elif page == "📋 System Logs":
    st.title("System Diagnostics")
    log_files = []
    if os.path.exists(LOGS_DIR):
        log_files = [f for f in os.listdir(LOGS_DIR) if f.endswith('.log')]
    
    if not log_files:
        st.info("No log files found.")
    else:
        selected_log = st.selectbox("Select Log File", sorted(log_files, reverse=True))
        log_path = os.path.join(LOGS_DIR, selected_log)
        with open(log_path, 'r') as f:
            lines = f.readlines()
            st.text_area("Last 100 entries", "".join(lines[-100:]), height=600)
            st.button("Refresh Logs")

elif page == "📊 Stock Chart Explorer":
    st.title("Interactive Stock Chart")
    st.markdown("Pilih saham yang masuk dalam rekomendasi AI atau ketik kode manual untuk melihat pergerakan harganya. Sangat berguna untuk mengkonfirmasi sinyal AI.")
    
    import yfinance as yf
    
    # Get available tickers from picks to suggest
    suggested_tickers = ['BBCA.JK', 'BRPT.JK', 'TLKM.JK', 'ADRO.JK'] # Fallbacks
    if not picks.empty and 'Ticker' in picks.columns:
        suggested_tickers = picks['Ticker'].tolist()
        
    # Allows entering custom like 'GOTO.JK'
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_ticker = st.selectbox("Pilih / Ketik Ticker Saham (Wajib akhiran .JK)", suggested_tickers)
    with col2:
        period_label = st.selectbox("Rentang Waktu", ["1 Bulan", "3 Bulan", "6 Bulan", "1 Tahun", "5 Tahun"], index=2)
        
    period_map = {"1 Bulan": "1mo", "3 Bulan": "3mo", "6 Bulan": "6mo", "1 Tahun": "1y", "5 Tahun": "5y"}
    period = period_map[period_label]
    
    if selected_ticker:
        with st.spinner(f"Mengambil data {period_label} untuk {selected_ticker}..."):
            try:
                df_chart = yf.download(selected_ticker, period=period, progress=False)
                
                if not df_chart.empty:
                    # Clean up yfinance MultiIndex output if present
                    if isinstance(df_chart.columns, pd.MultiIndex):
                        df_chart.columns = df_chart.columns.droplevel(1)
                    df_chart = df_chart.reset_index()
                    
                    # Ensure timezone-naive
                    if df_chart['Date'].dt.tz is not None:
                        df_chart['Date'] = df_chart['Date'].dt.tz_localize(None)
                    
                    # Plotly Candlestick
                    fig = go.Figure(data=[go.Candlestick(x=df_chart['Date'],
                                    open=df_chart['Open'],
                                    high=df_chart['High'],
                                    low=df_chart['Low'],
                                    close=df_chart['Close'],
                                    name=selected_ticker)])
                    
                    # Add EMA 20 (Fast Trend)
                    df_chart['EMA20'] = df_chart['Close'].ewm(span=20, adjust=False).mean()
                    fig.add_trace(go.Scatter(x=df_chart['Date'], y=df_chart['EMA20'], 
                                             line=dict(color='orange', width=2), 
                                             name='Trend Cepat (EMA 20)'))
                                             
                    # Add EMA 50 (Mid Trend)
                    df_chart['EMA50'] = df_chart['Close'].ewm(span=50, adjust=False).mean()
                    fig.add_trace(go.Scatter(x=df_chart['Date'], y=df_chart['EMA50'], 
                                             line=dict(color='cyan', width=2), 
                                             name='Trend Menengah (EMA 50)'))
                    
                    fig.update_layout(
                        title=f"Grafik Candlestick {selected_ticker} ({period_label} Terakhir)",
                        yaxis_title='Harga (IDR)',
                        template='plotly_dark',
                        xaxis_rangeslider_visible=False,
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("💡 Cara Membaca Grafik", expanded=True):
                        st.markdown('''
                        * **Batang Hijau (Naik)**: Harga tutup hari itu lebih tinggi dari harga bukanya.
                        * **Batang Merah (Turun)**: Harga tutup hari itu lebih rendah dari harga bukanya.
                        * **Garis EMA (Rata-rata)**: Jika harga saham (batang merah/hijau) berada di ATAS garis Oranye & Biru Muda, itu artinya saham sedang Uptrend (Layak Beli). Jika di bawahnya, berarti sedang Downtrend (Risiko Tinggi).
                        ''')
                else:
                    st.error(f"Data untuk {selected_ticker} kosong. Pastikan pasar tidak tutup terlalu lama atau kodenya benar.")
            except Exception as e:
                st.error(f"Gagal mengambil data dari Yahoo Finance: {e}")

elif page == "📖 Trading Logbook":
    st.title("Trading Logbook (Jurnal Trading)")
    st.markdown("Catat setiap transaksi beli/jual kamu di sini agar rapi seperti investor profesional.")
    
    logbook_path = os.path.join(ROOT, "data", "trading_logbook.csv")
    
    # Initialize Logbook if not exist
    if not os.path.exists(logbook_path):
        initial_data = pd.DataFrame({
            "Tanggal": [pd.Timestamp.now().strftime("%Y-%m-%d")],
            "Ticker": ["BBCA.JK"],
            "Aksi": ["BUY"],
            "Harga": [10000],
            "Lot": [10],
            "Total_Nilai": [10000000],
            "Catatan": ["Coba-coba mengikuti AI 🚀"]
        })
        os.makedirs(os.path.dirname(logbook_path), exist_ok=True)
        initial_data.to_csv(logbook_path, index=False)
        
    df_log = pd.read_csv(logbook_path)
    # FIX: Convert Tanggal string to datetime.date object for Streamlit DateColumn
    if 'Tanggal' in df_log.columns:
        df_log['Tanggal'] = pd.to_datetime(df_log['Tanggal'], errors='coerce').dt.date
    
    st.info("💡 **Tips:** Klik tombol '+' di bagian bawah tabel untuk menambah baris kosong baru. Tekan tombol `Delete` di keyboard untuk menghapus baris.")
    
    # Use st.data_editor to easily edit records
    edited_df = st.data_editor(
        df_log,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Tanggal": st.column_config.DateColumn("Tanggal", format="YYYY-MM-DD"),
            "Aksi": st.column_config.SelectboxColumn("Aksi", options=["BUY", "SELL", "HOLD"]),
            "Harga": st.column_config.NumberColumn("Harga (IDR)", min_value=0, format="%d"),
            "Lot": st.column_config.NumberColumn("Jumlah Lot", min_value=1, format="%d"),
            "Total_Nilai": st.column_config.NumberColumn("Total Transaksi", min_value=0, format="%d"),
        }
    )
    
    if st.button("💾 Simpan Perubahan Jurnal", use_container_width=True):
        edited_df.to_csv(logbook_path, index=False)
        st.success("Jurnal berhasil disimpan permanen! 📒✅")
