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
    c1.metric("Total Return", _pct(m.get('total_return'), True))
    c2.metric("Win Rate", _pct(m.get('win_rate')))
    c3.metric("Sharpe Ratio", f"{m.get('sharpe_ratio', 0):.2f}")
    c4.metric("Max Drawdown", _pct(m.get('max_drawdown')))

    # Benchmark Comparison
    st.markdown("---")
    st.subheader("Benchmark Comparison (IHSG)")
    bench_ret = m.get('benchmark_total_return', 0)
    alpha = m.get('total_return', 0) - float(bench_ret)
    b1, b2, b3 = st.columns(3)
    b1.metric("Alpha", _pct(alpha, True))
    b2.metric("IHSG Return", _pct(float(bench_ret)))
    b3.metric("Total Trades", int(m.get('num_trades', 0)))

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
