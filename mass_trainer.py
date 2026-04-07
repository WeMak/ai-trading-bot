"""
Mass Trainer — Train all neural network models on S&P 500 stocks
=================================================================
Downloads 10 years of daily data for 300+ stocks.
Combines into one giant training dataset.
Trains NEAT, PyTorch, and EvoStrategy on the combined data.

Usage:  python mass_trainer.py
"""

import warnings; warnings.filterwarnings("ignore")
import os, sys, json, time, random
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

from trading_agent import _rsi, _macd, _bollinger, _sma, _atr

from universe import ALL_STOCKS
SP500_STOCKS = ALL_STOCKS  # ~1900 stocks across all sectors


def _process_ticker(ticker: str, period: str = "10y") -> dict:
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

        # Compute all indicators once
        rsi_arr = _rsi(close)
        macd_l_arr, macd_s_arr, macd_h_arr = _macd(close)
        bb_u_arr, bb_m_arr, bb_l_arr = _bollinger(close)
        sma50_arr = _sma(close, 50)
        sma200_arr = _sma(close, 200)
        atr_arr = _atr(high, low, close)
        vol_avg = np.convolve(volume, np.ones(20)/20, mode='same')

        def _s(arr, idx):
            v = arr[idx] if idx < len(arr) else 0.0
            return float(v) if not np.isnan(v) else 0.0

        bars = []
        # Start at 200 to ensure SMA200 is valid
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
                    "chg_1d": float((close[i] - close[i-1]) / (close[i-1] + 1e-10) * 100),
                    "chg_5d": float((close[i] - close[i-5]) / (close[i-5] + 1e-10) * 100) if i >= 5 else 0,
                    "chg_20d": float((close[i] - close[i-20]) / (close[i-20] + 1e-10) * 100) if i >= 20 else 0,
                },
                "future_prices": [float(x) for x in close[i+1:i+6]] if i + 5 < len(close) else [],
            })

        return {"ticker": ticker, "bars": bars, "count": len(bars)}
    except Exception as e:
        return {"ticker": ticker, "bars": [], "error": str(e)}


def download_all(n_stocks: int = 300, period: str = "10y", workers: int = 12) -> list:
    """Download training data for n_stocks in parallel."""
    # Shuffle and pick
    tickers = list(SP500_STOCKS)
    random.shuffle(tickers)
    tickers = tickers[:n_stocks]

    print(f"\n{'='*60}")
    print(f"  MASS TRAINER — Downloading {len(tickers)} stocks ({period})")
    print(f"{'='*60}\n")

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
                    if done % 20 == 0 or done == len(tickers):
                        print(f"  [{done:3d}/{len(tickers)}] {ticker:<6} +{len(bars):,} bars | Total: {len(all_bars):,} bars")
                else:
                    errors += 1
            except Exception as e:
                errors += 1

    print(f"\n  Done: {done - errors} stocks | {errors} errors | {len(all_bars):,} total bars\n")
    return all_bars


def train_all_models(bars: list):
    """Train NEAT, PyTorch, and EvoStrategy on the combined dataset."""

    print(f"\n{'='*60}")
    print(f"  TRAINING ALL MODELS — {len(bars):,} bars")
    print(f"{'='*60}\n")

    # ─── 1. NEAT ─────────────────────────────────────────────
    print("  [1/3] Training NEAT Neuroevolution...")
    try:
        from neat_trader import get_neat_trader
        neat = get_neat_trader()
        # NEAT is slow, use a sample
        sample = random.sample(bars, min(5000, len(bars)))
        result = neat.evolve(sample, generations=30)
        print(f"    NEAT: Gen {result.get('total_generations')} | "
              f"Best fitness: {result.get('best_fitness')} | "
              f"Pop: {result.get('population_size')} | "
              f"Species: {result.get('species')} | "
              f"{result.get('elapsed_s')}s")
    except Exception as e:
        print(f"    NEAT error: {e}")

    # ─── 2. PyTorch ──────────────────────────────────────────
    print("\n  [2/3] Training PyTorch LSTM+Attention...")
    try:
        from torch_predictor import get_torch_predictor
        pytorch = get_torch_predictor()
        # PyTorch can handle more data
        sample = random.sample(bars, min(50000, len(bars)))
        # Sort by some proxy for time ordering (we mixed stocks, but within
        # each stock the bars were ordered — good enough for LSTM training)
        result = pytorch.train(sample, epochs=80)
        print(f"    PyTorch: Epoch {result.get('total_epochs')} | "
              f"Val loss: {result.get('final_val_loss')} | "
              f"Accuracy: {result.get('direction_accuracy')}% | "
              f"Device: {result.get('device')} | "
              f"{result.get('elapsed_s')}s")
    except Exception as e:
        print(f"    PyTorch error: {e}")

    # ─── 3. Evolutionary Strategy ────────────────────────────
    print("\n  [3/3] Training Evolutionary Strategy (DEAP)...")
    try:
        from evo_optimizer import get_evo_optimizer
        evo = get_evo_optimizer()
        sample = random.sample(bars, min(10000, len(bars)))
        result = evo.evolve(sample, generations=40, pop_size=80)
        print(f"    Evo: Gen {result.get('total_generations')} | "
              f"Best fitness: {result.get('best_fitness')} | "
              f"Pop: {result.get('population_size')} | "
              f"{result.get('elapsed_s')}s")
        if result.get("best_params"):
            print(f"    Best params:")
            for k, v in result["best_params"].items():
                print(f"      {k}: {v}")
    except Exception as e:
        print(f"    Evo error: {e}")


def main():
    t0 = time.time()

    # Download
    bars = download_all(n_stocks=300, period="10y", workers=12)

    if len(bars) < 1000:
        print("ERROR: Not enough training data collected. Check network connection.")
        return

    # Save raw data for future retraining
    data_path = os.path.join(os.path.dirname(__file__), "training_data.json")
    print(f"  Saving {len(bars):,} bars to training_data.json...")
    with open(data_path, "w") as f:
        json.dump({"bars": len(bars), "saved_at": datetime.utcnow().isoformat()}, f)
    # Don't save all bars (too large) — just metadata
    print(f"  Metadata saved.\n")

    # Train
    train_all_models(bars)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  COMPLETE — {elapsed:.0f}s total ({elapsed/60:.1f} min)")
    print(f"  Bars trained on: {len(bars):,}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
