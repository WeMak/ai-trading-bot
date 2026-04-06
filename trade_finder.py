"""
Trade Finder  —  Full-Stack Autonomous Trade Scanner
=====================================================
Scans 30 tickers in parallel (20 stocks + 10 crypto)
Returns top 10 opportunities ranked by signal confidence.

Watchlist
─────────
Stocks : AAPL MSFT NVDA GOOGL AMZN META TSLA AMD
         JPM BAC XOM CVX NFLX ORCL CRM COIN PLTR
         MSTR SPY QQQ
Crypto : BTC ETH SOL BNB XRP DOGE ADA AVAX LINK LTC
"""

import warnings; warnings.filterwarnings("ignore")
import concurrent.futures, time
import numpy as np
from datetime import datetime
from typing import Optional

# Import the analysis engine (no chart for speed)
from trading_agent import (
    _rsi, _macd, _bollinger, _sma, _atr, _sr_levels, _score,
    CRYPTO_MAP, analyze
)
import yfinance as yf
from datetime import timedelta

STOCKS = [
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AMD",
    "JPM","BAC","XOM","CVX","NFLX","ORCL","CRM","COIN",
    "PLTR","MSTR","SPY","QQQ",
]
CRYPTOS = ["BTC","ETH","SOL","BNB","XRP","DOGE","ADA","AVAX","LINK","LTC"]
ALL_TICKERS = STOCKS + CRYPTOS


def _analyze_quick(ticker: str) -> Optional[dict]:
    """
    Fast analysis — no chart generation.
    Returns signal dict or None on failure.
    """
    raw      = ticker.strip().upper()
    yf_sym   = CRYPTO_MAP.get(raw, raw)
    is_crypto = "-USD" in yf_sym

    try:
        end   = datetime.today()
        start = end - timedelta(days=260)
        df    = yf.Ticker(yf_sym).history(start=start, end=end, auto_adjust=True)

        if df.empty and not is_crypto:
            df = yf.Ticker(raw + "-USD").history(start=start, end=end, auto_adjust=True)
            if not df.empty:
                yf_sym, is_crypto = raw + "-USD", True

        if df.empty or len(df) < 55:
            return None

        df     = df.dropna(subset=["Close"])
        closes = df["Close"].values.astype(float)
        highs  = df["High"].values.astype(float)
        lows   = df["Low"].values.astype(float)
        volumes= df["Volume"].values.astype(float)
        price  = float(closes[-1])

        # Indicators (fast — no chart)
        rsi_arr         = _rsi(closes)
        ml, msl, mh     = _macd(closes)
        bbu, bbm, bbl   = _bollinger(closes)
        sma50_arr       = _sma(closes, 50)
        sma200_arr      = _sma(closes, 200)
        atr_arr         = _atr(highs, lows, closes)
        vol_avg         = float(np.nanmean(volumes[-20:]))
        vol_ratio       = float(volumes[-1]) / (vol_avg + 1e-10)
        _, resistances  = _sr_levels(highs, lows)

        def last(arr):
            v = arr[-1]
            return float(v) if not np.isnan(v) else float(np.nanmean(arr[-10:]))

        cur_rsi   = last(rsi_arr)
        cur_macd  = last(ml)
        cur_sig   = last(msl)
        cur_hist  = last(mh)
        cur_bbu   = last(bbu)
        cur_bbm   = last(bbm)
        cur_bbl   = last(bbl)
        cur_s50   = last(sma50_arr)
        cur_s200  = last(sma200_arr)
        cur_atr   = last(atr_arr) if not np.isnan(atr_arr[-1]) else price * 0.018

        # ATR-based levels
        if cur_rsi < 50:
            sl = price - 1.5 * cur_atr
            tp = price + 2.5 * cur_atr
        else:
            sl = price + 1.5 * cur_atr
            tp = price - 2.5 * cur_atr
        rr = round(abs(tp - price) / (abs(sl - price) + 1e-10), 2)

        ind = {
            "price":       price,
            "rsi":         round(cur_rsi,  1),
            "macd":        round(cur_macd, 6),
            "macd_signal": round(cur_sig,  6),
            "macd_hist":   round(cur_hist, 6),
            "bb_upper":    round(cur_bbu,  4),
            "bb_mid":      round(cur_bbm,  4),
            "bb_lower":    round(cur_bbl,  4),
            "sma50":       round(cur_s50,  4),
            "sma200":      round(cur_s200, 4),
            "atr":         round(cur_atr,  4),
            "vol_ratio":   round(vol_ratio, 2),
            "stop_loss":   round(sl, 4),
            "take_profit": round(tp, 4),
            "rr_ratio":    rr,
        }

        direction, confidence, notes = _score(ind)

        # Day change %
        day_chg = (closes[-1] - closes[-2]) / closes[-2] * 100 if len(closes) > 1 else 0.0
        trend   = "Up" if price > cur_s200 else "Down"
        bb_pct  = round(((price - cur_bbl) / (cur_bbu - cur_bbl + 1e-6)) * 100, 1)

        return {
            "ticker":     raw,
            "is_crypto":  is_crypto,
            "price":      round(price, 6),
            "day_chg":    round(day_chg, 2),
            "signal":     direction,
            "confidence": confidence,
            "notes":      notes,
            "trend":      trend,
            "bb_pct":     bb_pct,
            "indicators": ind,
        }

    except Exception:
        return None


def find_trades(n: int = 10, workers: int = 8) -> dict:
    """
    Scan all tickers in parallel and return the top-N trade opportunities.
    """
    t0      = time.time()
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_analyze_quick, t): t for t in ALL_TICKERS}
        for future in concurrent.futures.as_completed(futures):
            r = future.result()
            if r and r["signal"] in ("BUY", "SELL"):
                results.append(r)

    # Rank: highest confidence first; among ties prefer further from 50 RSI
    results.sort(key=lambda x: (x["confidence"], abs(x["indicators"]["rsi"] - 50)),
                 reverse=True)

    top    = results[:n]
    buys   = [r for r in top if r["signal"] == "BUY"]
    sells  = [r for r in top if r["signal"] == "SELL"]
    elapsed = round(time.time() - t0, 1)

    return {
        "timestamp":     datetime.utcnow().isoformat() + "Z",
        "elapsed_s":     elapsed,
        "scanned":       len(ALL_TICKERS),
        "signals_found": len(results),
        "top_trades":    top,
        "top_buys":      buys,
        "top_sells":     sells,
    }


if __name__ == "__main__":
    print(f"Scanning {len(ALL_TICKERS)} tickers…")
    r = find_trades(10)
    print(f"\n✅ Done in {r['elapsed_s']}s — {r['signals_found']} signals from {r['scanned']} tickers\n")
    print(f"{'#':<3} {'Ticker':<6} {'Type':<7} {'Signal':<5} {'Conf':>5} {'RSI':>5} {'Trend':>5}  Notes")
    print("─" * 80)
    for i, t in enumerate(r["top_trades"], 1):
        ind = t["indicators"]
        print(f"{i:<3} {t['ticker']:<6} {'Crypto' if t['is_crypto'] else 'Stock':<7} "
              f"{t['signal']:<5} {t['confidence']:>4.0f}% "
              f"{ind['rsi']:>5.1f} {t['trend']:>5}  {t['notes'][0][:45]}")
