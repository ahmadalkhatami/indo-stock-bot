import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import subprocess
import sys
import time

ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(ROOT, 'data', 'artifacts')
PICKS_FILE   = os.path.join(ROOT, 'data', 'latest_picks.csv')

st.set_page_config(page_title="Indo Stock Prediction", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS PREMIUM DESIGN ---
st.markdown("""
<style>
    /* Global Background and Fonts */
    .stApp {
        background-color: #0d1117;
        font-family: 'Inter', sans-serif;
    }
    
    /* Metrics Box Customization (Glassmorphism & Gradients) */
    [data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 800 !important;
        color: #58a6ff !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 14px !important;
        font-weight: 600 !important;
        color: #8b949e !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Headings and Titles */
    h1, h2, h3 {
        color: #c9d1d9 !important;
        margin-bottom: 20px;
    }
    
    /* Dataframes and Tables */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #30363d;
        box-shadow: 0 4px 6px rgba(0,0,0,0.5);
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
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{x*100:+.2f}%" if sign else f"{x*100:.2f}%"


def _inf_fmt(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    if x == float('inf'):
        return "∞"
    return f"{x:.2f}"


st.sidebar.title("Indo Stock Predictor")
page = st.sidebar.radio("Pages", [
    "Overview",
    "Equity Curve",
    "Top Picks Today",
    "Backtest Analysis",
    "Model Performance",
])
st.sidebar.markdown("---")
st.sidebar.caption("Run `python main.py` to refresh data.")

arts  = load_artifacts()
picks = load_picks()


# ─── Overview ────────────────────────────────────────────────────────────────
if page == "Overview":
    st.title("Strategy Overview")
    if arts['metrics'] is None:
        st.warning("No metrics found. Run `python main.py` first.")
        st.stop()

    m = arts['metrics'].iloc[0].to_dict()

    # ── SIDEBAR CONTROL ─────────────────────────────────────────────
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1693/1693746.png", width=80)
        st.title("Control Panel")
        
        # --- SCHEDULER SECTION ---
        st.subheader("⏰ Auto Scheduler")
        sched_time = st.text_input("Run Time (WIB)", value="16:30", help="Format HH:MM")
        is_active = st.toggle("Enable Daily Scheduler", value=True)
        
        # Simpan ke file untuk dibaca oleh desktop_app.py
        import json
        config_path = os.path.join(ROOT, "data", "scheduler_config.json")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump({"time": sched_time, "active": is_active}, f)
        
        if is_active:
            st.success(f"Scheduler active for {sched_time}")
            
        st.divider()

        # --- FOREIGN FLOW INSIGHTS ---
        st.subheader("🌐 Foreign Intelligence")
        try:
            # Load foreign flow data if exists in latest df
            if 'df' in globals() and df is not None and 'F_Net' in df.columns:
                today_ff = df.groupby('Ticker')['F_Net'].last().sort_values(ascending=False).head(5)
                st.write("Top 5 Foreign Net Buy:")
                for tick, val in today_ff.items():
                    st.caption(f"**{tick.split('.')[0]}**: {val:,.0f} shares")
        except:
            st.caption("Data incoming...")

        st.divider()

        # --- MANUAL RUN ---
        if st.button("🚀 RUN AI PREDICTION", use_container_width=True):
            with st.status("AI is calculating...", expanded=True) as status:
                st.write("Fetching market data...")
                env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
                process = subprocess.Popen(
                    [sys.executable.replace("dashboard/app.py", "venv/bin/python3"), "main.py"],
                    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )
                for line in process.stdout:
                    st.text(line.strip())
                process.wait()
                status.update(label="AI Analysis Complete!", state="complete", expanded=False)
                st.success("Data updated successfully!")
                time.sleep(2)
                st.rerun()

        st.divider()
        st.caption("v1.2 Premium Edition")

    # ── Returns & Drawdown ──────────────────────────────────────────────────
    st.subheader("Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Return",  _pct(m.get('total_return'), sign=True))
    c2.metric("CAGR",          _pct(m.get('cagr'), sign=True))
    c3.metric("Max Drawdown",  _pct(m.get('max_drawdown')))
    c4.metric("Sharpe Ratio",  f"{m.get('sharpe_ratio', 0):.2f}")

    bench = m.get('benchmark_total_return')
    if bench is not None and not np.isnan(float(bench)):
        alpha = m.get('total_return', 0) - float(bench)
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Benchmark (IHSG)", _pct(float(bench), sign=True))
        c6.metric("Alpha vs IHSG",    _pct(alpha, sign=True))
        c7.metric("Avg Exposure",     _pct(m.get('avg_exposure')))
        c8.metric("Trades Executed",  int(m.get('num_trades', 0)))
    else:
        c5, c6, c7 = st.columns(3)
        c5.metric("Avg Exposure",    _pct(m.get('avg_exposure')))
        c6.metric("Trades Executed", int(m.get('num_trades', 0)))
        c7.metric("Win Rate",        _pct(m.get('win_rate')))

    # ── Risk Metrics Panel ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Risk Metrics")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Win Rate",        _pct(m.get('win_rate')))
    r2.metric("Profit Factor",   _inf_fmt(m.get('profit_factor')))
    r3.metric("Win / Loss Ratio",_inf_fmt(m.get('win_loss_ratio')))
    r4.metric("Avg Exposure",    _pct(m.get('avg_exposure')))

    r5, r6, r7, r8 = st.columns(4)
    r5.metric("Avg Gain / Trade", _pct(m.get('avg_gain'), sign=True))
    r6.metric("Avg Loss / Trade", _pct(m.get('avg_loss'), sign=True))
    r7.metric("Gross Profit",     f"Rp {m.get('gross_profit', 0):,.0f}")
    r8.metric("Gross Loss",       f"Rp {m.get('gross_loss', 0):,.0f}")

    # ── Model Metrics ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Model Metrics (OOF)")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("ROC-AUC",  f"{m.get('roc_auc', 0):.4f}")
    mc2.metric("Precision",f"{m.get('precision', 0):.4f}")
    mc3.metric("Recall",   f"{m.get('recall', 0):.4f}")


# ─── Equity Curve ────────────────────────────────────────────────────────────
elif page == "Equity Curve":
    st.title("Equity Curve")
    eq    = arts['equity']
    bench = arts['benchmark']

    if eq is None:
        st.warning("No equity curve found. Run `python main.py` first.")
        st.stop()

    eq['Date'] = pd.to_datetime(eq['Date'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq['Date'], y=eq['equity'],
        name='Strategy', line=dict(color='#2E86DE', width=2),
    ))
    if bench is not None and not bench.empty:
        bench['Date'] = pd.to_datetime(bench['Date'])
        fig.add_trace(go.Scatter(
            x=bench['Date'], y=bench['benchmark_equity'],
            name='IHSG Buy & Hold', line=dict(color='#E67E22', dash='dash'),
        ))
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Portfolio Value (IDR)",
        hovermode='x unified', height=480,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Exposure bar
    st.subheader("Daily Exposure")
    fig2 = px.area(eq, x='Date', y='exposure_pct', height=260)
    fig2.update_layout(yaxis_tickformat='.0%', yaxis_title='Exposure')
    st.plotly_chart(fig2, use_container_width=True)


# ─── Top Picks Today ─────────────────────────────────────────────────────────
elif page == "Top Picks Today":
    st.title("Top Picks Today")

    if picks.empty:
        st.info("No picks file found. Run `python main.py` first.")
        st.stop()

    buy = picks[picks['signal'] == 'BUY'] if 'signal' in picks.columns else pd.DataFrame()
    if buy.empty:
        st.warning("No high-confidence signals today (threshold not met).")
    else:
        st.success(f"{len(buy)} high-confidence signal(s) today.")

    display = picks.copy()
    if 'probability' in display.columns:
        display['probability'] = display['probability'].apply(lambda x: f"{x*100:.2f}%")
    if 'Close' in display.columns:
        display['Close'] = display['Close'].apply(lambda x: f"Rp {x:,.0f}")
    st.dataframe(display, use_container_width=True)


# ─── Backtest Analysis ────────────────────────────────────────────────────────
elif page == "Backtest Analysis":
    st.title("Backtest Analysis")
    eq = arts['equity']

    if eq is None:
        st.warning("No equity curve found. Run `python main.py` first.")
        st.stop()

    eq['Date'] = pd.to_datetime(eq['Date'])

    # Equity Curve (repeated here for quick access)
    bench = arts['benchmark']
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=eq['Date'], y=eq['equity'], name='Strategy', line=dict(color='#2E86DE'),
    ))
    if bench is not None and not bench.empty:
        bench['Date'] = pd.to_datetime(bench['Date'])
        fig_eq.add_trace(go.Scatter(
            x=bench['Date'], y=bench['benchmark_equity'],
            name='IHSG', line=dict(color='#E67E22', dash='dash'),
        ))
    fig_eq.update_layout(hovermode='x unified', height=380, yaxis_title='IDR')
    st.plotly_chart(fig_eq, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Daily Returns")
        colors = ['#27AE60' if r >= 0 else '#E74C3C' for r in eq['daily_return']]
        fig_ret = go.Figure(go.Bar(
            x=eq['Date'], y=eq['daily_return'], marker_color=colors,
        ))
        fig_ret.update_layout(yaxis_tickformat='.2%', height=320)
        st.plotly_chart(fig_ret, use_container_width=True)

    with col2:
        st.subheader("Drawdown")
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=eq['Date'], y=eq['drawdown'],
            fill='tozeroy', line=dict(color='#E74C3C', width=1), name='Drawdown',
        ))
        fig_dd.update_layout(yaxis_tickformat='.2%', height=320)
        st.plotly_chart(fig_dd, use_container_width=True)

    # Trade log
    trades = arts['trades']
    if trades is not None and not trades.empty:
        st.subheader(f"Trade Log ({len(trades)} trades)")

        # Exit reason summary
        if 'exit_reason' in trades.columns:
            summary = trades['exit_reason'].value_counts().reset_index()
            summary.columns = ['Exit Reason', 'Count']
            wins   = trades[trades['pnl'] > 0]
            losses = trades[trades['pnl'] <= 0]

            ec1, ec2, ec3, ec4 = st.columns(4)
            ec1.metric("TP hits",      int(summary[summary['Exit Reason']=='TP']['Count'].sum() if 'TP' in summary['Exit Reason'].values else 0))
            ec2.metric("SL hits",      int(summary[summary['Exit Reason']=='SL']['Count'].sum() if 'SL' in summary['Exit Reason'].values else 0))
            ec3.metric("MaxHold exits",int(summary[summary['Exit Reason']=='MaxHold']['Count'].sum() if 'MaxHold' in summary['Exit Reason'].values else 0))
            ec4.metric("Win Rate",     f"{len(wins)/len(trades)*100:.1f}%")

            fig_exit = px.pie(summary, names='Exit Reason', values='Count',
                              title='Exit Reason Distribution', height=280)
            st.plotly_chart(fig_exit, use_container_width=True)

        # PnL distribution
        fig_pnl = px.histogram(
            trades, x='return_pct', nbins=40, title='Return per Trade Distribution',
            color_discrete_sequence=['#2E86DE'], height=280,
        )
        fig_pnl.update_layout(xaxis_tickformat='.1%')
        st.plotly_chart(fig_pnl, use_container_width=True)

        # Table
        t = trades.copy()
        t['entry_date'] = pd.to_datetime(t['entry_date']).dt.date
        t['exit_date']  = pd.to_datetime(t['exit_date']).dt.date
        t['return_pct'] = t['return_pct'].apply(lambda x: f"{x*100:+.2f}%")
        t['pnl']        = t['pnl'].apply(lambda x: f"Rp {x:,.0f}")
        t['cost']       = t['cost'].apply(lambda x: f"Rp {x:,.0f}")
        display_cols = [c for c in [
            'ticker', 'entry_date', 'exit_date', 'days_held',
            'exit_reason', 'return_pct', 'pnl', 'cost',
        ] if c in t.columns]
        st.dataframe(t[display_cols].sort_values('exit_date', ascending=False), use_container_width=True)
    else:
        st.info("No trades recorded.")


# ─── Model Performance ────────────────────────────────────────────────────────
elif page == "Model Performance":
    st.title("Model Performance")

    roc = arts['roc']
    if roc is not None:
        st.subheader("ROC Curve (last CV fold)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=roc['fpr'], y=roc['tpr'],
            name='Model', line=dict(color='#2E86DE'),
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name='Random', line=dict(dash='dash', color='gray'),
        ))
        fig.update_layout(xaxis_title='FPR', yaxis_title='TPR', height=420)
        st.plotly_chart(fig, use_container_width=True)

    fi = arts['feat_imp']
    if fi is not None:
        st.subheader("Feature Importance")
        fi_s = fi.sort_values('importance', ascending=True)
        fig2 = px.bar(fi_s, x='importance', y='feature', orientation='h', height=500)
        st.plotly_chart(fig2, use_container_width=True)

    if arts['metrics'] is not None:
        m = arts['metrics'].iloc[0].to_dict()
        st.subheader("OOF Classification Metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("ROC-AUC",  f"{m.get('roc_auc', 0):.4f}")
        c2.metric("Precision",f"{m.get('precision', 0):.4f}")
        c3.metric("Recall",   f"{m.get('recall', 0):.4f}")
