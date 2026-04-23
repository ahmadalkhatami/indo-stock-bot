import os
import sqlite3
import pandas as pd
import yaml
from datetime import datetime

from data.data_loader import fetch_data, fetch_benchmark, fetch_idx_tickers
from features.feature_engineering import add_features_and_labels
from models.xgboost_model import StockPredictor
from backtest.backtester import run_backtest, print_backtest_results
from bot.telegram_bot import run_bot
from utils.logger import logger

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Load Configuration ────────────────────────────────────────────────────────
def load_config():
    config_path = os.path.join(ROOT, 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()
STRAT = CONFIG['strategy']
PATHS = CONFIG['data']
F_PATHS = CONFIG['paths']

DATA_DIR = os.path.join(ROOT, F_PATHS['data_dir'])
ARTIFACT_DIR = os.path.join(ROOT, F_PATHS['artifact_dir'])
PICKS_FILE = os.path.join(ROOT, F_PATHS['picks_file'])
DB_FILE = os.path.join(ROOT, F_PATHS['db_file'])


def save_artifacts(results: dict, predictor: StockPredictor) -> None:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    logger.info(f"Saving artifacts to {ARTIFACT_DIR}")

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
    logger.info("=== Indonesian Stock Market Prediction System Starting ===")

    logger.info("[1/5] Fetching Tickers...")
    tickers_list = []
    for cat in PATHS['tickers']:
        tickers_list.extend(fetch_idx_tickers(cat))
    
    vip_tickers = PATHS.get('vip_tickers', [])
    for t in vip_tickers:
        if t not in tickers_list:
            tickers_list.append(t)
            
    tickers = list(set(tickers_list))
    logger.info(f"Total Tickers to Analyze: {len(tickers)}")
    
    df_raw = fetch_data(tickers, period=PATHS['period'])
    if df_raw.empty:
        logger.error("Failed to fetch market data. Exiting.")
        return
    benchmark_df = fetch_benchmark(period=PATHS['period'])
    
    logger.info("[2/5] Engineering Features & Sentiment...")
    from bot.sentiment_analyzer import SentimentAnalyzer
    sentiment_score = 0.0
    try:
        analyzer = SentimentAnalyzer()
        sentiment_score = analyzer.get_market_sentiment_score()
    except Exception as e:
        logger.warning(f"Sentiment analysis failed, using neutral: {e}")
        
    df_features = add_features_and_labels(df_raw, sentiment_score=sentiment_score)

    logger.info("[3/5] Training Model with Optuna...")
    predictor = StockPredictor(
        target='target_strong', confidence_threshold=STRAT['confidence_threshold']
    )
    predictor.train_and_evaluate(df_features)

    logger.info("[4/5] Running Backtest with Config Rules...")
    results = run_backtest(
        df_features,
        predictor.oof_predictions,
        benchmark_df=benchmark_df,
        initial_capital=STRAT['initial_capital'],
        confidence_threshold=STRAT['confidence_threshold'],
        momentum_min=STRAT['momentum_min'],
        volume_spike_min=STRAT['volume_spike_min'],
        top_pct=STRAT['top_pct'],
        max_positions=STRAT['max_positions'],
        position_pct=STRAT['position_pct'],
        take_profit=STRAT['take_profit'],
        stop_loss=STRAT['stop_loss'],
        max_hold_days=STRAT['max_hold_days'],
        fee_rate=STRAT.get('fee_rate', 0.0015),
        slippage=STRAT.get('slippage', 0.001),
    )
    print_backtest_results(results)

    logger.info("[5/5] Generating Today's Picks...")
    picks = predictor.predict_today(df_features)
    os.makedirs(DATA_DIR, exist_ok=True)
    picks.to_csv(PICKS_FILE, index=False)

    buy_picks = picks[picks['signal'] == 'BUY'] if 'signal' in picks.columns else pd.DataFrame()
    if not buy_picks.empty:
        logger.info(f"Top Picks Found: {len(buy_picks)}")
        for _, r in buy_picks.head(5).iterrows():
            print(
                f"{r['Ticker']:<10} | Prob: {r['probability']*100:>6.2f}% | "
                f"Close: Rp {r['Close']:,.0f}"
            )
    else:
        logger.warning("No high-confidence signals today.")

    # WhatsApp Notification
    ws_cfg = CONFIG.get('whatsapp', {})
    if ws_cfg.get('enabled') and not buy_picks.empty:
        from utils.notifications import send_whatsapp_signal
        msg = "🚀 *IndoStockBot Signals Found!*\n\n"
        for _, r in buy_picks.head(5).iterrows():
            msg += f"• *{r['Ticker']}*\n  Prob: {r['probability']*100:.1f}%\n  Price: Rp {r['Close']:,.0f}\n\n"
        msg += "Cek dashboard untuk detail lengkap."
        send_whatsapp_signal(ws_cfg['phone'], ws_cfg['apikey'], msg)

    save_artifacts(results, predictor)

    try:
        conn = sqlite3.connect(DB_FILE)
        # Ensure column exists for production upgrade
        try:
            conn.execute("ALTER TABLE historical_picks ADD COLUMN inserted_at DATETIME")
        except:
            pass # Column already exists
            
        picks['inserted_at'] = datetime.now()
        picks.to_sql('historical_picks', conn, if_exists='append', index=False)
        conn.close()
    except Exception as e:
        logger.error(f"Failed to save to SQLite: {e}")

    logger.info("Pipeline finished successfully.")

    bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
    if bot_token:
        logger.info("Starting Telegram bot service...")
        run_bot(bot_token)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Critical error in main pipeline")
