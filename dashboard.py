import streamlit as st
import pandas as pd
import os
import sqlite3

st.set_page_config(page_title="Indo Stock Predictor", layout="wide")

PICKS_FILE = os.path.join(os.path.dirname(__file__), 'data', 'latest_picks.csv')
DB_FILE = os.path.join(os.path.dirname(__file__), 'data', 'predictions.db')

st.title("📈 Indonesian Stock Market AI Predictor")
st.markdown("Predicting the probability of a stock rising $\ge$ 2% in the next 3 days.")

# --- Today's Picks ---
st.header("🔥 Today's Top Picks")
if os.path.exists(PICKS_FILE):
    df_picks = pd.read_csv(PICKS_FILE)
    if not df_picks.empty:
        # Format the dataframe for display
        display_df = df_picks[['Date', 'Ticker', 'Close', 'Probability']].copy()
        display_df['Probability'] = (display_df['Probability'] * 100).round(2).astype(str) + '%'
        display_df['Close'] = display_df['Close'].apply(lambda x: f"Rp {x:,.0f}")
        
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("No predictions found for today.")
else:
    st.warning("No predictions file found. Please run `python main.py` first.")

# --- Historical Predictions ---
st.header("📊 Historical Predictions DB")
if os.path.exists(DB_FILE):
    try:
        conn = sqlite3.connect(DB_FILE)
        query = "SELECT Date, Ticker, Close, Probability FROM historical_picks ORDER BY Date DESC LIMIT 50"
        df_history = pd.read_sql(query, conn)
        conn.close()
        
        if not df_history.empty:
            st.dataframe(df_history, use_container_width=True)
        else:
            st.info("Database is empty.")
    except Exception as e:
        st.error(f"Could not load database: {e}")
else:
    st.info("No SQLite database found yet. It will be created on the next pipeline run.")

st.sidebar.title("About")
st.sidebar.info(
    "This dashboard is powered by an XGBoost model evaluating 30 top Indonesian stocks. "
    "Features include Macro factors (USD/IDR, S&P 500) and various statistical momentum indicators."
)
