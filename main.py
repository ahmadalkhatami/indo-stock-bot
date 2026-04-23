import os
import pandas as pd
from data.data_loader import fetch_data
from features.feature_engineering import add_features_and_labels
from model.xgboost_model import StockPredictor
from backtest.backtester import run_backtest
from bot.telegram_bot import run_bot

# Top LQ45 or popular Indonesian stocks
TICKERS = [
    'BBCA.JK', 'BBRI.JK', 'BMRI.JK', 'BBNI.JK', 'TLKM.JK',
    'ASII.JK', 'UNVR.JK', 'ICBP.JK', 'INDF.JK', 'GOTO.JK',
    'AMRT.JK', 'CPIN.JK', 'KLBF.JK', 'PGAS.JK', 'PTBA.JK',
    'ADRO.JK', 'UNTR.JK', 'ITMG.JK', 'ANTM.JK', 'MDKA.JK',
    'INKP.JK', 'TKIM.JK', 'BRPT.JK', 'TPIA.JK', 'SMGR.JK',
    'INTP.JK', 'AKRA.JK', 'MEDC.JK', 'HRUM.JK', 'ARTO.JK'
]

PICKS_FILE = os.path.join(os.path.dirname(__file__), 'data', 'latest_picks.csv')

def main():
    print("=== Indonesian Stock Market Prediction System ===")
    
    # 1. Fetch Data
    print("\n[1/5] Fetching Data...")
    df_raw = fetch_data(TICKERS, period="5y")
    if df_raw.empty:
        print("Failed to fetch data. Exiting.")
        return
        
    # 2. Feature Engineering
    print("\n[2/5] Engineering Features...")
    df_features = add_features_and_labels(df_raw)
    
    # 3. Model Training & Evaluation
    print("\n[3/5] Training Model...")
    predictor = StockPredictor()
    predictor.train_and_evaluate(df_features)
    
    # 4. Backtesting
    print("\n[4/5] Running Backtest...")
    run_backtest(df_features, predictor.model_path)
    
    # 5. Predict Today
    print("\n[5/5] Generating Today's Picks...")
    picks = predictor.predict_today(df_features)
    
    if not picks.empty:
        top_5 = picks.head(5)
        print("\n🔥 Top 5 Stocks with Highest Probability (>= 2% return in 3 days) 🔥")
        for i, row in top_5.iterrows():
            print(f"{row['Ticker']:<10} | Prob: {row['Probability']*100:>6.2f}% | Last Close: Rp {row['Close']:,.0f}")
            
        # Ensure data directory exists
        os.makedirs(os.path.dirname(PICKS_FILE), exist_ok=True)
        # Save picks for Telegram Bot
        top_5.to_csv(PICKS_FILE, index=False)
        print(f"\nPicks saved to {PICKS_FILE}")
        
        # Save to SQLite Database for the Dashboard
        import sqlite3
        DB_FILE = os.path.join(os.path.dirname(__file__), 'data', 'predictions.db')
        try:
            conn = sqlite3.connect(DB_FILE)
            top_5.to_sql('historical_picks', conn, if_exists='append', index=False)
            conn.close()
            print(f"Picks successfully saved to SQLite database at {DB_FILE}")
        except Exception as e:
            print(f"Failed to save to SQLite: {e}")
    else:
        print("No picks generated.")

    print("\nPipeline finished successfully.")
    
    # Optional: Run bot if token is provided
    bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
    if bot_token:
        print("\nStarting Telegram Bot... Press Ctrl+C to stop.")
        run_bot(bot_token)
    else:
        print("\n(Optional) To run the Telegram bot, set the TELEGRAM_BOT_TOKEN environment variable.")

if __name__ == "__main__":
    main()
