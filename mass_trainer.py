"""
Mass Trainer + Backtester — One-Click Train & Validate
=========================================================
Downloads data for 500 stocks, trains all 3 neural network models,
then runs full backtests with 5:1 R/R targeting on each stock.

Outputs: PNL, Sharpe, win rate, profit factor, R-multiples,
equity curves, trade logs, random 3-month OOS tests, Monte Carlo.

Usage:
    python mass_trainer.py              # full 500-stock pipeline
    python mass_trainer.py --quick 50   # quick 50-stock test
"""

import warnings; warnings.filterwarnings("ignore")
import os, sys, json, time, random, argparse
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))

from gpu_compute import rsi_gpu as _rsi, macd_gpu as _macd, bollinger_gpu as _bollinger, sma_gpu as _sma, atr_gpu as _atr

from universe import ALL_STOCKS, SECTORS, CRYPTO_TICKERS
ALL_TICKERS = list(ALL_STOCKS)

# ─── CONFIG ──────────────────────────────────────────────────────
DEFAULT_N_STOCKS = 500
TRAIN_PERIOD = "5y"          # 5 years of training data
BACKTEST_PERIOD = "2y"       # 2 years for backtesting
DOWNLOAD_WORKERS = 12
BACKTEST_WORKERS = 4


# ─── DATA PROCESSING ────────────────────────────────────────────
def _process_ticker(ticker: str, period: str = "5y") -> dict:
    """Download and process one ticker into training bars."""
    try:
        from data_fetcher import fetch
        hist = fetch(ticker, period=period)
        if hist is None or hist.empty or len(hist) < 200:
            return {"ticker": ticker, "bars": [], "error": "insufficient data"}

        close = hist["Close"].values.astype(float)
        high = hist["High"].values.astype(float)
        low = hist["Low"].values.astype(float)
        volume = hist["Volume"].values.astype(float)

        # Compute all indicators once (GPU-vectorized)
        rsi_arr = _rsi(close)
        macd_l_arr, macd_s_arr, macd_h_arr = _macd(close)
        bb_u_arr, bb_m_arr, bb_l_arr = _bollinger(close)
        sma50_arr = _sma(close, 50)
        sma200_arr = _sma(close, 200)
        atr_arr = _atr(high, low, close)
        vol_avg = np.convolve(volume, np.ones(20) / 20, mode='same')

        def _s(arr, idx):
            v = arr[idx] if idx < len(arr) else 0.0
            return float(v) if not np.isnan(v) else 0.0

        bars = []
        start = max(200, 30)
        for i in range(start, len(close)):
            price = float(close[i])
            if price <= 0:
                continue
            bars.append({
                "price": price,
                "indicators": {
                    "price": price,
                    "rsi": _s(rsi_arr, i),
                    "macd": _s(macd_l_arr, i),
                    "macd_signal": _s(macd_s_arr, i),
                    "macd_hist": _s(macd_h_arr, i),
                    "bb_upper": _s(bb_u_arr, i),
                    "bb_middle": _s(bb_m_arr, i),
                    "bb_lower": _s(bb_l_arr, i),
                    "sma50": _s(sma50_arr, i),
                    "sma200": _s(sma200_arr, i),
                    "atr": _s(atr_arr, i),
                    "vol_ratio": float(volume[i]) / (float(vol_avg[i]) + 1e-10),
                },
                "price_history": {
                    "chg_1d": float((close[i] - close[i - 1]) / (close[i - 1] + 1e-10) * 100),
                    "chg_5d": float((close[i] - close[i - 5]) / (close[i - 5] + 1e-10) * 100) if i >= 5 else 0,
                    "chg_20d": float((close[i] - close[i - 20]) / (close[i - 20] + 1e-10) * 100) if i >= 20 else 0,
                },
                "future_prices": [float(x) for x in close[i + 1:i + 6]] if i + 5 < len(close) else [],
            })

        return {"ticker": ticker, "bars": bars, "count": len(bars)}
    except Exception as e:
        return {"ticker": ticker, "bars": [], "error": str(e)}


# ─── DOWNLOAD ────────────────────────────────────────────────────
def download_all(n_stocks: int = DEFAULT_N_STOCKS, period: str = TRAIN_PERIOD,
                 workers: int = DOWNLOAD_WORKERS, progress_cb=None) -> list:
    """Download training data for n_stocks in parallel."""
    tickers = list(ALL_TICKERS)
    random.shuffle(tickers)
    tickers = tickers[:n_stocks]

    print(f"\n{'=' * 70}")
    print(f"  MASS TRAINER -- Downloading {len(tickers)} stocks ({period})")
    print(f"{'=' * 70}\n")

    all_bars = []
    done = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_ticker, t, period): t for t in tickers}
        for f in as_completed(futures):
            ticker = futures[f]
            done += 1
            try:
                result = f.result()
                bars = result.get("bars", [])
                if bars:
                    all_bars.extend(bars)
                    if done % 25 == 0 or done == len(tickers):
                        pct = done / len(tickers) * 100
                        print(f"  [{done:3d}/{len(tickers)}] {pct:5.1f}% | {ticker:<6} +{len(bars):,} bars | Total: {len(all_bars):,}")
                        if progress_cb:
                            progress_cb(pct * 0.3, f"Downloaded {done}/{len(tickers)}")
                else:
                    errors += 1
            except Exception:
                errors += 1

    print(f"\n  Done: {done - errors} stocks | {errors} errors | {len(all_bars):,} total bars\n")
    return all_bars


# ─── TRAINING ────────────────────────────────────────────────────
def train_all_models(bars: list, progress_cb=None) -> dict:
    """Train NEAT, PyTorch, and EvoStrategy on the combined dataset."""
    results = {}

    print(f"\n{'=' * 70}")
    print(f"  TRAINING ALL MODELS -- {len(bars):,} bars | GPU: {'CUDA' if _has_gpu() else 'CPU'}")
    print(f"{'=' * 70}\n")

    # ─── 1. NEAT ─────────────────────────────────────────────
    print("  [1/3] Training NEAT Neuroevolution...")
    if progress_cb:
        progress_cb(35, "Training NEAT (neuroevolution)...")
    try:
        from neat_trader import get_neat_trader
        neat = get_neat_trader()
        sample = random.sample(bars, min(8000, len(bars)))
        result = neat.evolve(sample, generations=40)
        results["neat"] = result
        print(f"        Gen {result.get('total_generations')} | "
              f"Best fitness: {result.get('best_fitness', 0):.1f} | "
              f"Species: {result.get('species')} | "
              f"{result.get('elapsed_s')}s")
    except Exception as e:
        results["neat"] = {"error": str(e)}
        print(f"        NEAT error: {e}")

    # ─── 2. PyTorch LSTM ─────────────────────────────────────
    print("\n  [2/3] Training PyTorch LSTM+Attention (GPU)...")
    if progress_cb:
        progress_cb(50, "Training PyTorch LSTM on GPU...")
    try:
        from torch_predictor import get_torch_predictor
        pytorch = get_torch_predictor()
        sample = random.sample(bars, min(80000, len(bars)))
        result = pytorch.train(sample, epochs=100)
        results["pytorch"] = result
        print(f"        Epoch {result.get('total_epochs')} | "
              f"Val loss: {result.get('final_val_loss')} | "
              f"Accuracy: {result.get('direction_accuracy')}% | "
              f"Device: {result.get('device')} | "
              f"{result.get('elapsed_s')}s")
    except Exception as e:
        results["pytorch"] = {"error": str(e)}
        print(f"        PyTorch error: {e}")

    # ─── 3. Evolutionary Strategy ────────────────────────────
    print("\n  [3/3] Training Evolutionary Strategy (DEAP)...")
    if progress_cb:
        progress_cb(65, "Training EvoStrategy (genetic algorithm)...")
    try:
        from evo_optimizer import get_evo_optimizer
        evo = get_evo_optimizer()
        sample = random.sample(bars, min(15000, len(bars)))
        result = evo.evolve(sample, generations=50, pop_size=100)
        results["evo"] = result
        print(f"        Gen {result.get('total_generations')} | "
              f"Best fitness: {result.get('best_fitness', 0):.1f} | "
              f"Pop: {result.get('population_size')} | "
              f"{result.get('elapsed_s')}s")
        if result.get("best_params"):
            print(f"        Best params: SL={result['best_params'].get('stop_loss_pct', 0):.3f} "
                  f"TP={result['best_params'].get('take_profit_pct', 0):.3f} "
                  f"RSI buy<{result['best_params'].get('rsi_buy_threshold', 0):.0f} "
                  f"RSI sell>{result['best_params'].get('rsi_sell_threshold', 0):.0f}")
    except Exception as e:
        results["evo"] = {"error": str(e)}
        print(f"        Evo error: {e}")

    return results


# ─── BACKTESTING ─────────────────────────────────────────────────
def run_backtest(n_stocks: int = DEFAULT_N_STOCKS, period: str = BACKTEST_PERIOD,
                 progress_cb=None) -> dict:
    """Run full backtest on n_stocks with 5:1 R/R targeting."""
    from backtester import Backtester

    tickers = list(ALL_TICKERS)
    random.shuffle(tickers)
    tickers = tickers[:n_stocks]

    print(f"\n{'=' * 70}")
    print(f"  BACKTESTING -- {len(tickers)} stocks | {period} | 5:1 R/R")
    print(f"{'=' * 70}\n")

    bt = Backtester(config={
        "initial_capital": 100_000,
        "risk_per_trade_pct": 1.0,
        "reward_ratio": 5.0,
        "max_positions": 10,
        "min_confidence": 35,
        "min_agreement_pct": 50,
        "atr_sl_multiple": 1.5,
        "use_trailing_stop": True,
        "trailing_stop_atr": 2.0,
        "max_hold_bars": 60,
        "oos_months": 3,
        "n_random_tests": 5,
        "monte_carlo_runs": 500,
    })

    if progress_cb:
        bt.set_progress_callback(lambda pct, msg: progress_cb(70 + pct * 0.3, msg))

    results = bt.run_full_backtest(tickers, period=period, workers=BACKTEST_WORKERS, run_oos=True)

    # Print summary
    _print_backtest_summary(results)

    return results


def _print_backtest_summary(r: dict):
    """Pretty print backtest results to terminal."""
    print(f"\n{'=' * 70}")
    print(f"  BACKTEST RESULTS")
    print(f"{'=' * 70}")
    print(f"  Stocks tested:    {r.get('n_stocks_tested', 0)}")
    print(f"  Stocks traded:    {r.get('n_stocks_with_trades', 0)}")
    print(f"  Stocks profitable:{r.get('n_stocks_profitable', 0)}")
    print(f"  Total trades:     {r.get('total_trades', 0)}")
    print(f"  Winners:          {r.get('total_winners', 0)}")
    print(f"  Losers:           {r.get('total_losers', 0)}")
    print(f"  Win rate:         {r.get('win_rate', 0):.1f}%")
    print(f"  Total PNL:        ${r.get('total_pnl', 0):,.2f}")
    print(f"  Avg PNL/trade:    ${r.get('avg_pnl_per_trade', 0):,.2f}")
    print(f"  Profit factor:    {r.get('profit_factor', 0):.2f}")
    print(f"  Avg R-multiple:   {r.get('avg_r_multiple', 0):.2f}R")
    print(f"  Avg win:          ${r.get('avg_win', 0):,.2f}")
    print(f"  Avg loss:         ${r.get('avg_loss', 0):,.2f}")

    print(f"\n  Exit reasons:")
    for reason, count in r.get("exit_reasons", {}).items():
        print(f"    {reason}: {count}")

    if r.get("top_5_stocks"):
        print(f"\n  Top 5 stocks:")
        for s in r["top_5_stocks"]:
            print(f"    {s['ticker']:<6} return={s['return_pct']:+.1f}%  trades={s['trades']}  sharpe={s['sharpe']:.2f}")

    if r.get("bottom_5_stocks"):
        print(f"\n  Bottom 5 stocks:")
        for s in r["bottom_5_stocks"]:
            print(f"    {s['ticker']:<6} return={s['return_pct']:+.1f}%  trades={s['trades']}  sharpe={s['sharpe']:.2f}")

    oos = r.get("oos_summary", {})
    if oos:
        print(f"\n  Out-of-Sample (random 3-month windows):")
        print(f"    Tests run:      {oos.get('n_tests', 0)}")
        print(f"    Avg return:     {oos.get('avg_return_pct', 0):+.2f}%")
        print(f"    Median return:  {oos.get('median_return_pct', 0):+.2f}%")
        print(f"    OOS win rate:   {oos.get('win_rate', 0):.1f}%")
        print(f"    Avg Sharpe:     {oos.get('avg_sharpe', 0):.2f}")
        print(f"    Worst window:   {oos.get('worst_return', 0):+.2f}%")
        print(f"    Best window:    {oos.get('best_return', 0):+.2f}%")

    mc = r.get("monte_carlo", {})
    if mc:
        print(f"\n  Monte Carlo ({mc.get('runs', 0)} simulations):")
        print(f"    Median equity:  ${mc.get('median_final_equity', 0):,.0f}")
        print(f"    5th pctile:     ${mc.get('p5_equity', 0):,.0f}")
        print(f"    95th pctile:    ${mc.get('p95_equity', 0):,.0f}")
        print(f"    Worst case:     ${mc.get('worst_case_equity', 0):,.0f}")
        print(f"    Median max DD:  {mc.get('median_max_dd', 0):.1f}%")
        print(f"    95th DD:        {mc.get('p95_max_dd', 0):.1f}%")
        print(f"    Ruin prob:      {mc.get('ruin_probability', 0):.1f}%")

    print(f"\n  Elapsed: {r.get('elapsed_s', 0):.0f}s")
    print(f"{'=' * 70}\n")


def _has_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ─── FULL PIPELINE ───────────────────────────────────────────────
def full_pipeline(n_stocks: int = DEFAULT_N_STOCKS, progress_cb=None) -> dict:
    """
    Complete train + backtest pipeline.
    1. Download data for n_stocks
    2. Train all 3 models
    3. Backtest on each stock with 5:1 R/R
    4. Run random 3-month OOS tests
    5. Monte Carlo stress test
    """
    t0 = time.time()
    results = {"started": datetime.utcnow().isoformat() + "Z"}

    # Step 1: Download training data
    bars = download_all(n_stocks=n_stocks, period=TRAIN_PERIOD,
                        workers=DOWNLOAD_WORKERS, progress_cb=progress_cb)

    if len(bars) < 1000:
        return {"error": "Not enough training data. Check network.", "bars_collected": len(bars)}

    results["download"] = {"bars": len(bars), "stocks_ok": n_stocks}

    # Step 2: Train models
    train_results = train_all_models(bars, progress_cb=progress_cb)
    results["training"] = train_results

    # Step 3+4+5: Backtest + OOS + Monte Carlo
    backtest_results = run_backtest(n_stocks=min(n_stocks, 200), period=BACKTEST_PERIOD,
                                   progress_cb=progress_cb)
    results["backtest"] = backtest_results

    results["elapsed_s"] = round(time.time() - t0, 1)
    results["completed"] = datetime.utcnow().isoformat() + "Z"

    print(f"\n  FULL PIPELINE COMPLETE: {results['elapsed_s']:.0f}s ({results['elapsed_s']/60:.1f} min)\n")
    return results


# ─── CLI ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Mass Trainer + Backtester")
    parser.add_argument("--stocks", type=int, default=DEFAULT_N_STOCKS, help="Number of stocks (default: 500)")
    parser.add_argument("--quick", type=int, default=0, help="Quick test with N stocks")
    parser.add_argument("--backtest-only", action="store_true", help="Skip training, backtest only")
    parser.add_argument("--train-only", action="store_true", help="Train only, skip backtest")
    args = parser.parse_args()

    n = args.quick if args.quick > 0 else args.stocks

    if args.backtest_only:
        run_backtest(n_stocks=n, period=BACKTEST_PERIOD)
    elif args.train_only:
        bars = download_all(n_stocks=n)
        if len(bars) >= 1000:
            train_all_models(bars)
    else:
        full_pipeline(n_stocks=n)


if __name__ == "__main__":
    main()
