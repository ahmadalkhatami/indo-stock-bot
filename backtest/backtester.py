import pandas as pd
import numpy as np
from typing import Optional


def run_backtest(
    df_features: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame] = None,
    initial_capital: float = 100_000_000.0,
    # Entry filters
    confidence_threshold: float = 0.7,
    momentum_min: float = 1.02,
    volume_spike_min: float = 1.2,
    top_pct: float = 0.05,
    # Portfolio
    max_positions: int = 2,
    position_pct: float = 0.5,
    # Exit rules
    take_profit: float = 0.03,
    stop_loss: float = 0.015,
    max_hold_days: int = 5,
    # Costs
    fee_rate: float = 0.0015,
    slippage: float = 0.001,
) -> dict:
    """
    Aggressive daily-bar backtest with dynamic exits.

    Entry: prob >= threshold AND momentum_10 > momentum_min AND volume_spike >
           volume_spike_min AND market bullish (IHSG trend > 1) AND top_pct by prob.

    Exit (checked each day after entry, using bar High/Low):
      - Stop loss  : Low  <= entry * (1 - stop_loss)  → exit at SL price
      - Take profit: High >= entry * (1 + take_profit) → exit at TP price
      - Max hold   : day >= max_exit_date              → exit at Close

    Position sizing: position_pct * current_equity (up to available cash).
    """
    # ── Build OHLC + feature lookups ──────────────────────────────────────────
    need = ['Date', 'Ticker', 'Close', 'High', 'Low', 'momentum_10', 'volume_spike']
    cols = [c for c in need if c in df_features.columns]
    df = df_features[cols].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Date', 'Ticker'])

    idx = ['Date', 'Ticker']
    close_lkp = df.set_index(idx)['Close'].to_dict()
    high_lkp  = df.set_index(idx)['High'].to_dict()  if 'High'  in df.columns else {}
    low_lkp   = df.set_index(idx)['Low'].to_dict()   if 'Low'   in df.columns else {}

    # ── Market regime: IHSG trend_strength > 1 ────────────────────────────────
    market_regime: dict = {}
    if benchmark_df is not None and not benchmark_df.empty:
        bench = benchmark_df.copy()
        bench['Date'] = pd.to_datetime(bench['Date'])
        if bench['Date'].dt.tz is not None:
            bench['Date'] = bench['Date'].dt.tz_localize(None)
        bench = bench.sort_values('Date')
        bench['trend'] = (
            bench['Benchmark_Close']
            / bench['Benchmark_Close'].rolling(50, min_periods=1).mean()
        )
        market_regime = dict(zip(bench['Date'], bench['trend'] > 1))

    # ── Enrich OOF predictions with feature filters ───────────────────────────
    preds = oof_predictions.copy()
    preds['Date'] = pd.to_datetime(preds['Date'])

    feat_cols = [c for c in ['momentum_10', 'volume_spike'] if c in df.columns]
    if feat_cols:
        feat = df[['Date', 'Ticker'] + feat_cols].dropna()
        preds = preds.merge(feat, on=['Date', 'Ticker'], how='left')

    # Apply entry filters
    mask = preds['probability'] >= confidence_threshold
    if 'momentum_10' in preds.columns:
        mask &= preds['momentum_10'].fillna(0) > momentum_min
    if 'volume_spike' in preds.columns:
        mask &= preds['volume_spike'].fillna(0) > volume_spike_min
    valid = preds[mask].copy()

    # Top-N% by probability within each day
    if not valid.empty and top_pct < 1.0:
        valid['_pct'] = valid.groupby('Date')['probability'].rank(pct=True)
        valid = valid[valid['_pct'] >= (1.0 - top_pct)]

    valid = valid.sort_values(['Date', 'probability'], ascending=[True, False])
    picks_by_day: dict = {d: grp for d, grp in valid.groupby('Date')}

    # ── Simulation ────────────────────────────────────────────────────────────
    trading_days = sorted(df['Date'].unique())
    day_to_idx   = {d: i for i, d in enumerate(trading_days)}

    cash      = float(initial_capital)
    positions: list[dict] = []
    eq_curve:  list[dict] = []
    trades:    list[dict] = []

    for day in trading_days:
        day_idx = day_to_idx[day]

        # ── 1. Exits (TP / SL / MaxHold) – skip entry day ─────────────────
        still_open: list[dict] = []
        for p in positions:
            if day == p['entry_date']:
                still_open.append(p)
                continue

            close_d = close_lkp.get((day, p['ticker']))
            high_d  = high_lkp.get((day, p['ticker']))
            low_d   = low_lkp.get((day, p['ticker']))

            if close_d is None or pd.isna(close_d):
                still_open.append(p)
                continue

            exit_raw    = None
            exit_reason = None

            # SL first (conservative): if low touched SL level
            if low_d is not None and not pd.isna(low_d) and low_d <= p['sl_price']:
                exit_raw    = p['sl_price']
                exit_reason = 'SL'
            # TP: if high touched TP level
            elif high_d is not None and not pd.isna(high_d) and high_d >= p['tp_price']:
                exit_raw    = p['tp_price']
                exit_reason = 'TP'
            # Max hold: force exit at close
            elif day >= p['max_exit_date']:
                exit_raw    = close_d
                exit_reason = 'MaxHold'

            if exit_raw is not None:
                exit_price = exit_raw * (1 - slippage)
                proceeds   = p['shares'] * exit_price * (1 - fee_rate)
                pnl        = proceeds - p['cost']
                cash      += proceeds
                trades.append({
                    'ticker':       p['ticker'],
                    'entry_date':   p['entry_date'],
                    'exit_date':    day,
                    'days_held':    day_idx - day_to_idx[p['entry_date']],
                    'entry_price':  round(p['entry_price'], 4),
                    'exit_price':   round(exit_price, 4),
                    'exit_reason':  exit_reason,
                    'cost':         p['cost'],
                    'proceeds':     proceeds,
                    'pnl':          pnl,
                    'return_pct':   (proceeds / p['cost']) - 1,
                })
            else:
                still_open.append(p)
        positions = still_open

        # ── 2. Open new positions ─────────────────────────────────────────
        is_bullish = market_regime.get(day, True)  # default bullish if no data
        if is_bullish and len(positions) < max_positions:
            todays = picks_by_day.get(day)
            if todays is not None:
                mv_now = sum(
                    p['shares'] * close_lkp.get((day, p['ticker']), p['entry_price'])
                    for p in positions
                )
                equity_now         = cash + mv_now
                capital_per_trade  = equity_now * position_pct

                for _, row in todays.iterrows():
                    if len(positions) >= max_positions:
                        break
                    # Need at least half the target size in cash
                    if cash < capital_per_trade * 0.5:
                        break
                    ticker = row['Ticker']
                    if any(p['ticker'] == ticker for p in positions):
                        continue
                    raw = close_lkp.get((day, ticker))
                    if raw is None or pd.isna(raw) or raw <= 0:
                        continue

                    entry_price    = raw * (1 + slippage)
                    actual_capital = min(capital_per_trade, cash)
                    shares         = (actual_capital * (1 - fee_rate)) / entry_price
                    if shares <= 0:
                        continue

                    exit_idx      = day_idx + max_hold_days
                    max_exit_date = (
                        trading_days[exit_idx] if exit_idx < len(trading_days)
                        else trading_days[-1]
                    )
                    cash -= actual_capital
                    positions.append({
                        'ticker':        ticker,
                        'entry_date':    day,
                        'entry_price':   entry_price,
                        'shares':        shares,
                        'cost':          actual_capital,
                        'tp_price':      entry_price * (1 + take_profit),
                        'sl_price':      entry_price * (1 - stop_loss),
                        'max_exit_date': max_exit_date,
                    })

        # ── 3. Mark-to-market equity ──────────────────────────────────────
        mv = sum(
            p['shares'] * close_lkp.get((day, p['ticker']), p['entry_price'])
            for p in positions
        )
        equity = cash + mv
        eq_curve.append({
            'Date':         day,
            'cash':         cash,
            'market_value': mv,
            'equity':       equity,
            'exposure_pct': mv / equity if equity > 0 else 0.0,
        })

    eq_df     = pd.DataFrame(eq_curve)
    trades_df = pd.DataFrame(trades)

    if eq_df.empty:
        return _empty_result(initial_capital)

    # ── Performance metrics ───────────────────────────────────────────────────
    final_equity = float(eq_df['equity'].iloc[-1])
    total_return = (final_equity / initial_capital) - 1

    eq_df['daily_return'] = eq_df['equity'].pct_change().fillna(0)
    eq_df['running_max']  = eq_df['equity'].cummax()
    eq_df['drawdown']     = (eq_df['equity'] - eq_df['running_max']) / eq_df['running_max']
    max_drawdown          = float(eq_df['drawdown'].min())

    n_days = len(eq_df)
    years  = n_days / 252 if n_days else 0
    cagr   = (final_equity / initial_capital) ** (1 / years) - 1 if years > 0 else 0.0

    dr     = eq_df['daily_return']
    sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0
    avg_exposure = float(eq_df['exposure_pct'].mean())

    # ── Trade metrics ─────────────────────────────────────────────────────────
    if not trades_df.empty:
        wins   = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]

        win_rate       = len(wins) / len(trades_df)
        avg_gain       = float(wins['return_pct'].mean())   if not wins.empty   else 0.0
        avg_loss       = float(losses['return_pct'].mean()) if not losses.empty else 0.0
        win_loss_ratio = abs(avg_gain / avg_loss)           if avg_loss != 0    else float('inf')
        gross_profit   = float(wins['pnl'].sum())           if not wins.empty   else 0.0
        gross_loss     = float(abs(losses['pnl'].sum()))    if not losses.empty else 0.0
        profit_factor  = gross_profit / gross_loss          if gross_loss > 0   else float('inf')

        exit_counts = trades_df['exit_reason'].value_counts().to_dict()
    else:
        win_rate = avg_gain = avg_loss = win_loss_ratio = gross_profit = gross_loss = 0.0
        profit_factor = 0.0
        exit_counts   = {}

    # ── Benchmark ─────────────────────────────────────────────────────────────
    benchmark_curve        = None
    benchmark_total_return = None
    if benchmark_df is not None and not benchmark_df.empty:
        bench = benchmark_df.copy()
        bench['Date'] = pd.to_datetime(bench['Date'])
        if bench['Date'].dt.tz is not None:
            bench['Date'] = bench['Date'].dt.tz_localize(None)
        bench = bench[bench['Date'].isin(eq_df['Date'])].sort_values('Date')
        if not bench.empty:
            fp = bench['Benchmark_Close'].iloc[0]
            bench['benchmark_equity']  = bench['Benchmark_Close'] / fp * initial_capital
            benchmark_curve            = bench[['Date', 'benchmark_equity']].reset_index(drop=True)
            benchmark_total_return     = float(bench['Benchmark_Close'].iloc[-1] / fp - 1)

    return {
        # Capital
        'initial_capital':        initial_capital,
        'final_equity':           final_equity,
        # Returns
        'total_return':           float(total_return),
        'cagr':                   float(cagr),
        # Risk
        'max_drawdown':           max_drawdown,
        'sharpe_ratio':           sharpe,
        'avg_exposure':           avg_exposure,
        # Trade stats
        'num_trades':             len(trades_df),
        'win_rate':               float(win_rate),
        'avg_gain':               avg_gain,
        'avg_loss':               avg_loss,
        'win_loss_ratio':         win_loss_ratio,
        'gross_profit':           gross_profit,
        'gross_loss':             gross_loss,
        'profit_factor':          profit_factor,
        'exit_counts':            exit_counts,
        # Benchmark
        'benchmark_total_return': benchmark_total_return,
        # DataFrames
        'equity_curve':           eq_df,
        'trades':                 trades_df,
        'benchmark_curve':        benchmark_curve,
    }


def print_backtest_results(results: dict) -> None:
    pf = results['profit_factor']
    pf_str = f"{pf:.2f}" if pf != float('inf') else "inf"

    wl = results['win_loss_ratio']
    wl_str = f"{wl:.2f}" if wl != float('inf') else "inf"

    print("\n=== Backtest Results ===")
    print(f"Initial Capital:  Rp {results['initial_capital']:>15,.0f}")
    print(f"Final Equity:     Rp {results['final_equity']:>15,.0f}")
    print(f"Total Return:     {results['total_return']*100:>+.2f}%")
    print(f"CAGR:             {results['cagr']*100:>+.2f}%")
    print(f"Max Drawdown:     {results['max_drawdown']*100:.2f}%")
    print(f"Sharpe Ratio:     {results['sharpe_ratio']:.2f}")
    print(f"Avg Exposure:     {results['avg_exposure']*100:.1f}%")
    print(f"\n--- Trade Statistics ---")
    print(f"Number of Trades: {results['num_trades']}")
    print(f"Win Rate:         {results['win_rate']*100:.2f}%")
    print(f"Avg Gain:         {results['avg_gain']*100:>+.2f}%")
    print(f"Avg Loss:         {results['avg_loss']*100:>+.2f}%")
    print(f"Win/Loss Ratio:   {wl_str}")
    print(f"Profit Factor:    {pf_str}")
    print(f"Gross Profit:     Rp {results['gross_profit']:>15,.0f}")
    print(f"Gross Loss:       Rp {results['gross_loss']:>15,.0f}")
    if results['exit_counts']:
        print(f"\n--- Exit Breakdown ---")
        for reason, count in results['exit_counts'].items():
            print(f"  {reason:<10}: {count}")
    if results['benchmark_total_return'] is not None:
        alpha = results['total_return'] - results['benchmark_total_return']
        print(f"\nBenchmark (IHSG): {results['benchmark_total_return']*100:>+.2f}%")
        print(f"Alpha:            {alpha*100:>+.2f}%")


def _empty_result(initial_capital: float) -> dict:
    return {
        'initial_capital': initial_capital, 'final_equity': initial_capital,
        'total_return': 0.0, 'cagr': 0.0, 'max_drawdown': 0.0, 'sharpe_ratio': 0.0,
        'avg_exposure': 0.0, 'num_trades': 0, 'win_rate': 0.0, 'avg_gain': 0.0,
        'avg_loss': 0.0, 'win_loss_ratio': 0.0, 'gross_profit': 0.0, 'gross_loss': 0.0,
        'profit_factor': 0.0, 'exit_counts': {}, 'benchmark_total_return': None,
        'equity_curve': pd.DataFrame(), 'trades': pd.DataFrame(), 'benchmark_curve': None,
    }
