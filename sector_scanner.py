"""
Sector Scanner — Top 10 stocks from every S&P sector
=====================================================
11 GICS sectors × 10 tickers each = 110 stocks scanned.
Uses yfinance for price data + technical analysis from trading_agent.
"""

import warnings; warnings.filterwarnings("ignore")
import concurrent.futures, time
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import yfinance as yf

from trading_agent import (
    _rsi, _macd, _bollinger, _sma, _atr, _sr_levels, _score, CRYPTO_MAP
)

# ── Top 10 stocks per GICS sector ────────────────────────────
from universe import SECTORS, ALL_STOCKS

ALL_SECTOR_TICKERS = ALL_STOCKS


def _analyze_ticker(ticker: str) -> Optional[dict]:
    """Quick technical analysis — no chart."""
    try:
        end = datetime.today()
        start = end - timedelta(days=260)
        from data_fetcher import fetch
        df = fetch(ticker, start=start, end=end)

        if df.empty or len(df) < 55:
            return None

        df = df.dropna(subset=["Close"])
        closes = df["Close"].values.astype(float)
        highs = df["High"].values.astype(float)
        lows = df["Low"].values.astype(float)
        volumes = df["Volume"].values.astype(float)
        price = float(closes[-1])

        rsi_arr = _rsi(closes)
        ml, msl, mh = _macd(closes)
        bbu, bbm, bbl = _bollinger(closes)
        sma50_arr = _sma(closes, 50)
        sma200_arr = _sma(closes, 200)
        atr_arr = _atr(highs, lows, closes)
        vol_avg = float(np.nanmean(volumes[-20:]))
        vol_ratio = float(volumes[-1]) / (vol_avg + 1e-10)
        supports, resistances = _sr_levels(highs, lows)

        def last(arr):
            v = arr[-1]
            return float(v) if not np.isnan(v) else float(np.nanmean(arr[-10:]))

        cur_atr = last(atr_arr) if not np.isnan(atr_arr[-1]) else price * 0.018

        if last(rsi_arr) < 50:
            sl = price - 1.5 * cur_atr
            tp = price + 2.5 * cur_atr
        else:
            sl = price + 1.5 * cur_atr
            tp = price - 2.5 * cur_atr
        rr = round(abs(tp - price) / (abs(sl - price) + 1e-10), 2)

        ind = {
            "price": price,
            "rsi": round(last(rsi_arr), 1),
            "macd": round(last(ml), 6),
            "macd_signal": round(last(msl), 6),
            "macd_hist": round(last(mh), 6),
            "bb_upper": round(last(bbu), 4),
            "bb_mid": round(last(bbm), 4),
            "bb_lower": round(last(bbl), 4),
            "sma50": round(last(sma50_arr), 4),
            "sma200": round(last(sma200_arr), 4),
            "atr": round(cur_atr, 4),
            "vol_ratio": round(vol_ratio, 2),
            "stop_loss": round(sl, 4),
            "take_profit": round(tp, 4),
            "rr_ratio": rr,
        }

        direction, confidence, notes = _score(ind)
        day_chg = (closes[-1] - closes[-2]) / closes[-2] * 100 if len(closes) > 1 else 0.0

        # 5-day and 20-day momentum
        chg_5d = (closes[-1] - closes[-5]) / closes[-5] * 100 if len(closes) >= 5 else 0.0
        chg_20d = (closes[-1] - closes[-20]) / closes[-20] * 100 if len(closes) >= 20 else 0.0

        return {
            "ticker": ticker,
            "price": round(price, 4),
            "day_chg": round(day_chg, 2),
            "chg_5d": round(chg_5d, 2),
            "chg_20d": round(chg_20d, 2),
            "signal": direction,
            "confidence": confidence,
            "notes": notes,
            "trend": "Up" if price > ind["sma200"] else "Down",
            "indicators": ind,
        }
    except Exception:
        return None


def scan_sector(sector_name: str, workers: int = 6) -> dict:
    """Scan a single sector."""
    tickers = SECTORS.get(sector_name, [])
    if not tickers:
        return {"error": f"Unknown sector: {sector_name}"}

    t0 = time.time()
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_analyze_ticker, t): t for t in tickers}
        for f in concurrent.futures.as_completed(futures):
            r = f.result()
            if r:
                results.append(r)

    results.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "sector": sector_name,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "elapsed_s": round(time.time() - t0, 1),
        "scanned": len(tickers),
        "results": results,
        "top_buy": next((r for r in results if r["signal"] == "BUY"), None),
        "top_sell": next((r for r in results if r["signal"] == "SELL"), None),
    }


def scan_all_sectors(workers: int = 10) -> dict:
    """Scan all 11 sectors in parallel."""
    t0 = time.time()
    all_results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_analyze_ticker, t): t for t in ALL_SECTOR_TICKERS}
        for f in concurrent.futures.as_completed(futures):
            r = f.result()
            if r:
                all_results.append(r)

    # Group by sector
    ticker_to_sector = {}
    for sector, tickers in SECTORS.items():
        for t in tickers:
            ticker_to_sector[t] = sector

    sector_results = {}
    for r in all_results:
        sec = ticker_to_sector.get(r["ticker"], "Unknown")
        if sec not in sector_results:
            sector_results[sec] = []
        sector_results[sec].append(r)

    # Sort each sector by confidence
    for sec in sector_results:
        sector_results[sec].sort(key=lambda x: x["confidence"], reverse=True)

    # Global top picks
    buys = [r for r in all_results if r["signal"] == "BUY"]
    sells = [r for r in all_results if r["signal"] == "SELL"]
    buys.sort(key=lambda x: x["confidence"], reverse=True)
    sells.sort(key=lambda x: x["confidence"], reverse=True)

    # Sector strength (avg momentum)
    sector_strength = {}
    for sec, items in sector_results.items():
        avg_chg = np.mean([r["day_chg"] for r in items]) if items else 0.0
        avg_5d = np.mean([r["chg_5d"] for r in items]) if items else 0.0
        buy_count = sum(1 for r in items if r["signal"] == "BUY")
        sell_count = sum(1 for r in items if r["signal"] == "SELL")
        sector_strength[sec] = {
            "avg_daily_chg": round(avg_chg, 2),
            "avg_5d_chg": round(avg_5d, 2),
            "buy_signals": buy_count,
            "sell_signals": sell_count,
            "sentiment": "Bullish" if buy_count > sell_count else "Bearish" if sell_count > buy_count else "Neutral",
        }

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "elapsed_s": round(time.time() - t0, 1),
        "total_scanned": len(all_results),
        "sectors": sector_results,
        "sector_strength": sector_strength,
        "top_buys": buys[:15],
        "top_sells": sells[:15],
    }


if __name__ == "__main__":
    print(f"Scanning {len(ALL_SECTOR_TICKERS)} tickers across {len(SECTORS)} sectors...")
    r = scan_all_sectors()
    print(f"\nDone in {r['elapsed_s']}s — {r['total_scanned']} tickers analyzed\n")

    print("SECTOR STRENGTH:")
    print(f"{'Sector':<28} {'Daily':>6} {'5-Day':>6} {'Buys':>5} {'Sells':>5} {'Sentiment'}")
    print("-" * 70)
    for sec, s in sorted(r["sector_strength"].items(), key=lambda x: x[1]["avg_5d_chg"], reverse=True):
        print(f"{sec:<28} {s['avg_daily_chg']:>+5.1f}% {s['avg_5d_chg']:>+5.1f}% {s['buy_signals']:>5} {s['sell_signals']:>5} {s['sentiment']}")

    print(f"\nTOP 10 BUYS:")
    for i, t in enumerate(r["top_buys"][:10], 1):
        print(f"  {i}. {t['ticker']:<6} ${t['price']:>10,.2f}  Conf:{t['confidence']:>4.0f}%  {t['notes'][0][:50]}")

    print(f"\nTOP 10 SELLS:")
    for i, t in enumerate(r["top_sells"][:10], 1):
        print(f"  {i}. {t['ticker']:<6} ${t['price']:>10,.2f}  Conf:{t['confidence']:>4.0f}%  {t['notes'][0][:50]}")
