import os
import pandas as pd
import numpy as np
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'xgboost_model.pkl')

def run_backtest(df: pd.DataFrame, model_path: str = MODEL_PATH):
    """
    Simulate strategy:
    Each day, pick top 3 stocks with highest probability.
    Hold for 3 days.
    """
    print("Initializing Backtest simulation...")
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Could not load model at {model_path}. Please train the model first.")
        return
    
    features = [
        'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Diff', 
        'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
        'BB_High', 'BB_Low', 'BB_Width', 
        'Return_1d', 'Return_3d', 'Return_7d',
        'Vol_7d', 'Vol_14d', 'Volume_Change_1d', 'Momentum_20'
    ]
    
    # Filter valid rows and drop the last 3 days where future returns are NaN
    backtest_df = df.dropna(subset=features + ['Future_Return_3d']).copy()
    
    if backtest_df.empty:
        print("Not enough data for backtesting.")
        return
    
    # Get probabilities for historical data
    X = backtest_df[features]
    backtest_df['Prob'] = model.predict_proba(X)[:, 1]
    
    daily_stats = []
    
    # Iterate day by day
    for date, group in backtest_df.groupby('Date'):
        if len(group) == 0:
            continue
        
        # Pick top 3 highest probabilities
        top_picks = group.sort_values(by='Prob', ascending=False).head(3)
        
        # Calculate the average 3-day return of these 3 picks
        avg_trade_return = top_picks['Future_Return_3d'].mean()
        
        # Win is if avg return > 0
        is_win = 1 if avg_trade_return > 0 else 0
        
        daily_stats.append({
            'Date': date,
            'Trade_Return': avg_trade_return,
            'Win': is_win
        })
        
    res_df = pd.DataFrame(daily_stats)
    if res_df.empty:
        print("No daily stats generated.")
        return
        
    res_df = res_df.sort_values(by='Date').reset_index(drop=True)
    
    # Calculate portfolio metrics
    # We allocate capital into 3 overlapping tranches because the hold period is 3 days
    # Tranche i trades on day i, i+3, i+6...
    res_df['Tranche'] = res_df.index % 3
    tranche_final_wealths = []
    
    for i in range(3):
        tranche = res_df[res_df['Tranche'] == i].copy()
        # Cumulative return for this tranche
        tranche['Cum_Return'] = (1 + tranche['Trade_Return']).cumprod()
        if not tranche.empty:
            tranche_final_wealths.append(tranche['Cum_Return'].iloc[-1])
            
    if tranche_final_wealths:
        # Final wealth of the overall portfolio is the average of the tranches
        final_wealth = np.mean(tranche_final_wealths)
        total_return = final_wealth - 1
    else:
        total_return = 0
        
    win_rate = res_df['Win'].mean()
    
    # Estimate overall equity curve for max drawdown
    # Approximate by averaging the active tranches daily
    res_df['Equity_Curve'] = (1 + res_df['Trade_Return'] / 3).cumprod()
    res_df['Peak'] = res_df['Equity_Curve'].cummax()
    res_df['Drawdown'] = (res_df['Equity_Curve'] - res_df['Peak']) / res_df['Peak']
    max_dd = res_df['Drawdown'].min()
    
    print("\n--- Backtest Results ---")
    print(f"Total Trading Days Simulated: {len(res_df)}")
    print(f"Total Return:       {total_return * 100:.2f}%")
    print(f"Win Rate:           {win_rate * 100:.2f}%")
    print(f"Max Drawdown:       {max_dd * 100:.2f}%\n")
