import os
import sqlite3
import pandas as pd

from data.data_loader import fetch_data, fetch_benchmark
from features.feature_engineering import add_features_and_labels
from models.xgboost_model import StockPredictor
from backtest.backtester import run_backtest, print_backtest_results
from bot.telegram_bot import run_bot

TICKERS = [
    'BBCA.JK', 'BBRI.JK', 'BMRI.JK', 'BBNI.JK', 'TLKM.JK',
    'ASII.JK', 'UNVR.JK', 'ICBP.JK', 'INDF.JK', 'GOTO.JK',
    'AMRT.JK', 'CPIN.JK', 'KLBF.JK', 'PGAS.JK', 'PTBA.JK',
    'ADRO.JK', 'UNTR.JK', 'ITMG.JK', 'ANTM.JK', 'MDKA.JK',
    'INKP.JK', 'TKIM.JK', 'BRPT.JK', 'TPIA.JK', 'SMGR.JK',
    'INTP.JK', 'AKRA.JK', 'MEDC.JK', 'HRUM.JK', 'ARTO.JK',
]

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'data')
ARTIFACT_DIR = os.path.join(DATA_DIR, 'artifacts')
PICKS_FILE = os.path.join(DATA_DIR, 'latest_picks.csv')
DB_FILE = os.path.join(DATA_DIR, 'predictions.db')

# ── Strategy parameters ────────────────────────────────────────────────────────
INITIAL_CAPITAL       = 100_000_000.0

# Entry filters
CONFIDENCE_THRESHOLD  = 0.7     # prob threshold for signals
MOMENTUM_MIN          = 1.02    # momentum_10 > 1.02 (stock above 10-day ago price by 2%)
VOLUME_SPIKE_MIN      = 1.2     # volume / 20-day avg > 1.2
TOP_PCT               = 0.05    # top 5% by probability on each day

# Portfolio
MAX_POSITIONS         = 2       # max concurrent positions
POSITION_PCT          = 0.5     # 50% of current equity per trade

# Exit rules
TAKE_PROFIT           = 0.03    # +3% TP
STOP_LOSS             = 0.015   # -1.5% SL
MAX_HOLD_DAYS         = 5       # force exit after 5 trading days


def save_artifacts(results: dict, predictor: StockPredictor) -> None:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    results['equity_curve'].to_csv(
        os.path.join(ARTIFACT_DIR, 'equity_curve.csv'), index=False
    )
    if not results['trades'].empty:
        results['trades'].to_csv(
            os.path.join(ARTIFACT_DIR, 'trades.csv'), index=False
        )
    if results['benchmark_curve'] is not None:
        results['benchmark_curve'].to_csv(
            os.path.join(ARTIFACT_DIR, 'benchmark_curve.csv'), index=False
        )

    metrics_row = {
        'total_return':           results['total_return'],
        'cagr':                   results['cagr'],
        'win_rate':               results['win_rate'],
        'max_drawdown':           results['max_drawdown'],
        'sharpe_ratio':           results['sharpe_ratio'],
        'avg_exposure':           results['avg_exposure'],
        'num_trades':             results['num_trades'],
        'avg_gain':               results['avg_gain'],
        'avg_loss':               results['avg_loss'],
        'win_loss_ratio':         results['win_loss_ratio'],
        'profit_factor':          results['profit_factor'],
        'gross_profit':           results['gross_profit'],
        'gross_loss':             results['gross_loss'],
        'benchmark_total_return': results['benchmark_total_return'],
        'roc_auc':                predictor.metrics.get('roc_auc'),
        'precision':              predictor.metrics.get('precision'),
        'recall':                 predictor.metrics.get('recall'),
    }
    pd.DataFrame([metrics_row]).to_csv(
        os.path.join(ARTIFACT_DIR, 'metrics.csv'), index=False
    )

    if predictor.roc_points is not None:
        fpr, tpr = predictor.roc_points
        pd.DataFrame({'fpr': fpr, 'tpr': tpr}).to_csv(
            os.path.join(ARTIFACT_DIR, 'roc.csv'), index=False
        )
    if predictor.feature_importances is not None:
        predictor.feature_importances.rename_axis('feature').reset_index(
            name='importance'
        ).to_csv(os.path.join(ARTIFACT_DIR, 'feature_importances.csv'), index=False)


def main():
    print("=== Indonesian Stock Market Prediction System ===")

    print("\n[1/5] Fetching Data...")
    df_raw = fetch_data(TICKERS, period="5y")
    if df_raw.empty:
        print("Failed to fetch data. Exiting.")
        return
    benchmark_df = fetch_benchmark(period="5y")

    print("\n[2/5] Engineering Features...")
    df_features = add_features_and_labels(df_raw)

    print("\n[3/5] Training Model...")
    predictor = StockPredictor(
        target='target_binary', confidence_threshold=CONFIDENCE_THRESHOLD
    )
    predictor.train_and_evaluate(df_features)

    print("\n[4/5] Running Aggressive Backtest...")
    results = run_backtest(
        df_features,
        predictor.oof_predictions,
        benchmark_df=benchmark_df,
        initial_capital=INITIAL_CAPITAL,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        momentum_min=MOMENTUM_MIN,
        volume_spike_min=VOLUME_SPIKE_MIN,
        top_pct=TOP_PCT,
        max_positions=MAX_POSITIONS,
        position_pct=POSITION_PCT,
        take_profit=TAKE_PROFIT,
        stop_loss=STOP_LOSS,
        max_hold_days=MAX_HOLD_DAYS,
    )
    print_backtest_results(results)

    print("\n[5/5] Generating Today's Picks...")
    picks = predictor.predict_today(df_features)
    os.makedirs(DATA_DIR, exist_ok=True)
    picks.to_csv(PICKS_FILE, index=False)

    buy_picks = picks[picks['signal'] == 'BUY'] if 'signal' in picks.columns else pd.DataFrame()
    if not buy_picks.empty:
        print(f"\n=== Top Picks Today (probability >= {CONFIDENCE_THRESHOLD:.0%}) ===")
        for _, r in buy_picks.head(5).iterrows():
            print(
                f"{r['Ticker']:<10} | Prob: {r['probability']*100:>6.2f}% | "
                f"Close: Rp {r['Close']:,.0f}"
            )
    else:
        print("\nNo high-confidence signals today.")

    save_artifacts(results, predictor)
    print(f"\nArtifacts saved to {ARTIFACT_DIR}")

    try:
        conn = sqlite3.connect(DB_FILE)
        picks.to_sql('historical_picks', conn, if_exists='append', index=False)
        conn.close()
    except Exception as e:
        print(f"Failed to save to SQLite: {e}")

    print("\nPipeline finished successfully.")
    print("To view the dashboard, run: streamlit run dashboard/app.py")

    bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
    if bot_token:
        print("\nStarting Telegram bot... Ctrl+C to stop.")
        run_bot(bot_token)


if __name__ == "__main__":
    main()
