import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(ROOT, 'data', 'artifacts')
PICKS_FILE = os.path.join(ROOT, 'data', 'latest_picks.csv')

st.set_page_config(page_title="Indo Stock Prediction", layout="wide")


@st.cache_data
def load_artifacts():
    paths = {
        'equity': os.path.join(ARTIFACT_DIR, 'equity_curve.csv'),
        'trades': os.path.join(ARTIFACT_DIR, 'trades.csv'),
        'benchmark': os.path.join(ARTIFACT_DIR, 'benchmark_curve.csv'),
        'metrics': os.path.join(ARTIFACT_DIR, 'metrics.csv'),
        'roc': os.path.join(ARTIFACT_DIR, 'roc.csv'),
        'feat_imp': os.path.join(ARTIFACT_DIR, 'feature_importances.csv'),
    }
    return {k: (pd.read_csv(p) if os.path.exists(p) else None) for k, p in paths.items()}


@st.cache_data
def load_picks():
    if os.path.exists(PICKS_FILE):
        return pd.read_csv(PICKS_FILE)
    return pd.DataFrame()


st.sidebar.title("Indo Stock Predictor")
page = st.sidebar.radio(
    "Pages",
    ["Overview", "Equity Curve", "Top Picks Today", "Backtest Analysis", "Model Performance"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Run `python main.py` to refresh data.")

artifacts = load_artifacts()
picks = load_picks()


def _fmt_pct(x, sign=False):
    if pd.isna(x):
        return "N/A"
    return f"{x*100:+.2f}%" if sign else f"{x*100:.2f}%"


if page == "Overview":
    st.title("Strategy Overview")
    if artifacts['metrics'] is None:
        st.warning("No backtest metrics found. Run `python main.py` first.")
    else:
        m = artifacts['metrics'].iloc[0].to_dict()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Return", _fmt_pct(m['total_return'], sign=True))
        c2.metric("CAGR", _fmt_pct(m['cagr'], sign=True))
        c3.metric("Win Rate", _fmt_pct(m['win_rate']))
        c4.metric("Max Drawdown", _fmt_pct(m['max_drawdown']))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Sharpe Ratio", f"{m['sharpe_ratio']:.2f}")
        c6.metric("Trades Executed", int(m['num_trades']))
        bench = m.get('benchmark_total_return')
        if pd.notna(bench):
            c7.metric("Benchmark (IHSG)", _fmt_pct(float(bench), sign=True))
            alpha = m['total_return'] - float(bench)
            c8.metric("Alpha vs IHSG", _fmt_pct(alpha, sign=True))

        st.markdown("---")
        st.subheader("Model Metrics (Out-of-Fold)")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("ROC-AUC", f"{m.get('roc_auc', 0):.4f}")
        mc2.metric("Precision", f"{m.get('precision', 0):.4f}")
        mc3.metric("Recall", f"{m.get('recall', 0):.4f}")

elif page == "Equity Curve":
    st.title("Equity Curve")
    eq = artifacts['equity']
    bench = artifacts['benchmark']
    if eq is None:
        st.warning("No equity curve found. Run `python main.py` first.")
    else:
        eq['Date'] = pd.to_datetime(eq['Date'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eq['Date'], y=eq['equity'], name='Strategy', line=dict(color='#2E86DE')
        ))
        if bench is not None and not bench.empty:
            bench['Date'] = pd.to_datetime(bench['Date'])
            fig.add_trace(go.Scatter(
                x=bench['Date'], y=bench['benchmark_equity'],
                name='IHSG Buy & Hold', line=dict(color='#E67E22', dash='dash')
            ))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Portfolio Value (IDR)",
            hovermode='x unified',
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Cash vs Market Value")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=eq['Date'], y=eq['cash'], name='Cash',
            stackgroup='one', line=dict(width=0), fillcolor='#95A5A6',
        ))
        fig2.add_trace(go.Scatter(
            x=eq['Date'], y=eq['market_value'], name='Market Value',
            stackgroup='one', line=dict(width=0), fillcolor='#27AE60',
        ))
        fig2.update_layout(yaxis_title="IDR", hovermode='x unified', height=400)
        st.plotly_chart(fig2, use_container_width=True)

elif page == "Top Picks Today":
    st.title("Top Picks Today")
    if picks.empty:
        st.info("No picks file found. Run `python main.py` first.")
    else:
        buy = picks[picks['signal'] == 'BUY'] if 'signal' in picks.columns else pd.DataFrame()
        if buy.empty:
            st.warning("No high-confidence signals today (probability >= 60%).")
        else:
            st.success(f"{len(buy)} high-confidence signal(s) today.")

        display = picks.copy()
        if 'probability' in display.columns:
            display['probability'] = display['probability'].apply(lambda x: f"{x*100:.2f}%")
        if 'Close' in display.columns:
            display['Close'] = display['Close'].apply(lambda x: f"Rp {x:,.0f}")
        st.dataframe(display, use_container_width=True)

elif page == "Backtest Analysis":
    st.title("Backtest Analysis")
    eq = artifacts['equity']
    if eq is None:
        st.warning("No equity curve found. Run `python main.py` first.")
    else:
        eq['Date'] = pd.to_datetime(eq['Date'])

        st.subheader("Daily Returns")
        fig1 = px.bar(eq, x='Date', y='daily_return')
        fig1.update_layout(yaxis_tickformat='.2%', height=350)
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Drawdown")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=eq['Date'], y=eq['drawdown'], fill='tozeroy',
            line=dict(color='#E74C3C'), name='Drawdown',
        ))
        fig2.update_layout(yaxis_tickformat='.2%', height=350)
        st.plotly_chart(fig2, use_container_width=True)

        trades = artifacts['trades']
        if trades is not None and not trades.empty:
            st.subheader(f"Trades ({len(trades)} total)")
            t = trades.copy()
            t['entry_date'] = pd.to_datetime(t['entry_date']).dt.date
            t['exit_date'] = pd.to_datetime(t['exit_date']).dt.date
            t['return_pct'] = t['return_pct'].apply(lambda x: f"{x*100:+.2f}%")
            t['pnl'] = t['pnl'].apply(lambda x: f"Rp {x:,.0f}")
            st.dataframe(
                t[['ticker', 'entry_date', 'exit_date', 'return_pct', 'pnl']].tail(50),
                use_container_width=True,
            )

elif page == "Model Performance":
    st.title("Model Performance")

    roc = artifacts['roc']
    if roc is not None:
        st.subheader("ROC Curve (last CV fold)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roc['fpr'], y=roc['tpr'], name='Model', line=dict(color='#2E86DE')))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], name='Random', line=dict(dash='dash', color='gray')
        ))
        fig.update_layout(
            xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', height=450
        )
        st.plotly_chart(fig, use_container_width=True)

    fi = artifacts['feat_imp']
    if fi is not None:
        st.subheader("Feature Importance")
        fi_sorted = fi.sort_values('importance', ascending=True)
        fig2 = px.bar(fi_sorted, x='importance', y='feature', orientation='h', height=500)
        st.plotly_chart(fig2, use_container_width=True)

    if artifacts['metrics'] is not None:
        m = artifacts['metrics'].iloc[0].to_dict()
        st.subheader("OOF Classification Metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("ROC-AUC", f"{m.get('roc_auc', 0):.4f}")
        c2.metric("Precision", f"{m.get('precision', 0):.4f}")
        c3.metric("Recall", f"{m.get('recall', 0):.4f}")
