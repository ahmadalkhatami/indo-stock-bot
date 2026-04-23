import pandas as pd
import numpy as np


def run_backtest(
    df_features: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    benchmark_df: pd.DataFrame | None = None,
    initial_capital: float = 100_000_000.0,
    confidence_threshold: float = 0.6,
    max_positions_per_day: int = 3,
    max_concurrent: int = 9,
    hold_days: int = 3,
    fee_rate: float = 0.0015,
    slippage: float = 0.001,
) -> dict:
    """
    Realistic daily-rebalance backtest.

    Rules:
      - Each day, select up to `max_positions_per_day` stocks with prob >= threshold.
      - Each new position sized as current_equity / max_concurrent.
      - Entry at close * (1 + slippage), exit after `hold_days` at close * (1 - slippage).
      - `fee_rate` applied on both entry and exit.
      - If no signals: stay in cash.
    """
    df = df_features[['Date', 'Ticker', 'Close']].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Date', 'Ticker'])
    price_lookup = df.set_index(['Date', 'Ticker'])['Close'].to_dict()

    preds = oof_predictions.copy()
    preds['Date'] = pd.to_datetime(preds['Date'])

    trading_days = sorted(df['Date'].unique())
    day_to_idx = {d: i for i, d in enumerate(trading_days)}
    pred_days = set(preds['Date'].unique())

    daily_picks = (
        preds[preds['probability'] >= confidence_threshold]
        .sort_values(['Date', 'probability'], ascending=[True, False])
        .groupby('Date')
        .head(max_positions_per_day)
    )
    picks_by_day = {d: grp for d, grp in daily_picks.groupby('Date')}

    cash = float(initial_capital)
    positions: list[dict] = []
    equity_curve = []
    trades = []

    for day in trading_days:
        # 1. Close positions whose exit date is today
        still_open = []
        for p in positions:
            if p['exit_date'] == day:
                raw = price_lookup.get((day, p['ticker']))
                if raw is None or pd.isna(raw):
                    still_open.append(p)
                    continue
                exit_price = raw * (1 - slippage)
                proceeds = p['shares'] * exit_price * (1 - fee_rate)
                cash += proceeds
                trades.append({
                    'ticker': p['ticker'],
                    'entry_date': p['entry_date'],
                    'exit_date': day,
                    'entry_price': p['entry_price'],
                    'exit_price': exit_price,
                    'shares': p['shares'],
                    'cost': p['cost'],
                    'proceeds': proceeds,
                    'pnl': proceeds - p['cost'],
                    'return_pct': (proceeds / p['cost']) - 1,
                })
            else:
                still_open.append(p)
        positions = still_open

        # 2. Open new positions if we have signals and room
        if day in pred_days and len(positions) < max_concurrent:
            todays = picks_by_day.get(day)
            if todays is not None:
                current_equity = cash + sum(
                    p['shares'] * price_lookup.get((day, p['ticker']), p['entry_price'])
                    for p in positions
                )
                capital_per_position = current_equity / max_concurrent

                for _, row in todays.iterrows():
                    if len(positions) >= max_concurrent or cash < capital_per_position:
                        break
                    ticker = row['Ticker']
                    if any(p['ticker'] == ticker for p in positions):
                        continue
                    raw = price_lookup.get((day, ticker))
                    if raw is None or pd.isna(raw) or raw <= 0:
                        continue
                    entry_price = raw * (1 + slippage)
                    shares = (capital_per_position * (1 - fee_rate)) / entry_price
                    if shares <= 0:
                        continue
                    exit_idx = day_to_idx[day] + hold_days
                    exit_date = (
                        trading_days[exit_idx] if exit_idx < len(trading_days)
                        else trading_days[-1]
                    )
                    cash -= capital_per_position
                    positions.append({
                        'ticker': ticker,
                        'entry_date': day,
                        'exit_date': exit_date,
                        'entry_price': entry_price,
                        'shares': shares,
                        'cost': capital_per_position,
                    })

        # 3. Mark to market
        mv = 0.0
        for p in positions:
            price = price_lookup.get((day, p['ticker']))
            if price is None or pd.isna(price):
                price = p['entry_price']
            mv += p['shares'] * price
        equity_curve.append({
            'Date': day,
            'cash': cash,
            'market_value': mv,
            'equity': cash + mv,
        })

    eq_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades)

    if eq_df.empty:
        return {
            'initial_capital': initial_capital,
            'final_equity': initial_capital,
            'total_return': 0.0,
            'cagr': 0.0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'num_trades': 0,
            'benchmark_total_return': None,
            'equity_curve': eq_df,
            'trades': trades_df,
            'benchmark_curve': None,
        }

    final_equity = float(eq_df['equity'].iloc[-1])
    total_return = (final_equity / initial_capital) - 1

    eq_df['daily_return'] = eq_df['equity'].pct_change().fillna(0)
    eq_df['running_max'] = eq_df['equity'].cummax()
    eq_df['drawdown'] = (eq_df['equity'] - eq_df['running_max']) / eq_df['running_max']
    max_drawdown = float(eq_df['drawdown'].min())

    n_days = len(eq_df)
    years = n_days / 252 if n_days else 0
    cagr = (final_equity / initial_capital) ** (1 / years) - 1 if years > 0 else 0.0

    daily_ret = eq_df['daily_return']
    sharpe = (
        float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))
        if daily_ret.std() > 0 else 0.0
    )

    win_rate = float((trades_df['pnl'] > 0).mean()) if not trades_df.empty else 0.0

    benchmark_curve = None
    benchmark_total_return = None
    if benchmark_df is not None and not benchmark_df.empty:
        bench = benchmark_df.copy()
        bench['Date'] = pd.to_datetime(bench['Date'])
        bench = bench[bench['Date'].isin(eq_df['Date'])].sort_values('Date')
        if not bench.empty:
            first_price = bench['Benchmark_Close'].iloc[0]
            bench['benchmark_equity'] = (
                bench['Benchmark_Close'] / first_price * initial_capital
            )
            benchmark_curve = bench[['Date', 'benchmark_equity']].reset_index(drop=True)
            benchmark_total_return = float(
                (bench['Benchmark_Close'].iloc[-1] / first_price) - 1
            )

    return {
        'initial_capital': initial_capital,
        'final_equity': final_equity,
        'total_return': float(total_return),
        'cagr': float(cagr),
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'num_trades': len(trades_df),
        'benchmark_total_return': benchmark_total_return,
        'equity_curve': eq_df,
        'trades': trades_df,
        'benchmark_curve': benchmark_curve,
    }


def print_backtest_results(results: dict) -> None:
    print("\n=== Backtest Results ===")
    print(f"Initial Capital:  Rp {results['initial_capital']:,.0f}")
    print(f"Final Equity:     Rp {results['final_equity']:,.0f}")
    print(f"Total Return:     {results['total_return']*100:+.2f}%")
    print(f"CAGR:             {results['cagr']*100:+.2f}%")
    print(f"Win Rate:         {results['win_rate']*100:.2f}%")
    print(f"Max Drawdown:     {results['max_drawdown']*100:.2f}%")
    print(f"Sharpe Ratio:     {results['sharpe_ratio']:.2f}")
    print(f"Number of Trades: {results['num_trades']}")
    if results['benchmark_total_return'] is not None:
        print(f"Benchmark (IHSG): {results['benchmark_total_return']*100:+.2f}%")
        alpha = results['total_return'] - results['benchmark_total_return']
        print(f"Alpha:            {alpha*100:+.2f}%")
