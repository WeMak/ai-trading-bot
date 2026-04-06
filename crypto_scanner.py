"""
Crypto Scanner — Top 30 cryptocurrencies
==========================================
Deep technical analysis on major crypto assets.
Includes fear/greed index proxy + correlation analysis.
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

# Top 30 crypto by market cap
CRYPTO_TICKERS = {
    # Tier 1 — Large caps
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "BNB": "BNB-USD",
    "SOL": "SOL-USD",
    "XRP": "XRP-USD",
    "ADA": "ADA-USD",
    "AVAX": "AVAX-USD",
    "DOT": "DOT-USD",
    "LINK": "LINK-USD",
    "MATIC": "MATIC-USD",
    # Tier 2 — Mid caps
    "DOGE": "DOGE-USD",
    "LTC": "LTC-USD",
    "BCH": "BCH-USD",
    "UNI": "UNI-USD",
    "ATOM": "ATOM-USD",
    "XLM": "XLM-USD",
    "TRX": "TRX-USD",
    "NEAR": "NEAR-USD",
    "FIL": "FIL-USD",
    "APT": "APT-USD",
    # Tier 3 — Emerging
    "ARB": "ARB11841-USD",
    "OP": "OP-USD",
    "INJ": "INJ-USD",
    "SUI": "SUI20947-USD",
    "SEI": "SEI-USD",
    "TIA": "TIA22861-USD",
    "RENDER": "RNDR-USD",
    "FET": "FET-USD",
    "PEPE": "PEPE24478-USD",
    "WIF": "WIF-USD",
}


def _analyze_crypto(symbol: str, yf_ticker: str) -> Optional[dict]:
    """Deep technical analysis for a crypto asset."""
    try:
        end = datetime.today()
        start = end - timedelta(days=365)
        df = yf.Ticker(yf_ticker).history(start=start, end=end, auto_adjust=True)

        if df.empty or len(df) < 55:
            return None

        df = df.dropna(subset=["Close"])
        closes = df["Close"].values.astype(float)
        highs = df["High"].values.astype(float)
        lows = df["Low"].values.astype(float)
        volumes = df["Volume"].values.astype(float)
        price = float(closes[-1])

        # Technical indicators
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

        cur_atr = last(atr_arr) if not np.isnan(atr_arr[-1]) else price * 0.03

        if last(rsi_arr) < 50:
            sl = price - 2.0 * cur_atr  # wider stops for crypto
            tp = price + 3.5 * cur_atr
        else:
            sl = price + 2.0 * cur_atr
            tp = price - 3.5 * cur_atr
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

        # Price changes
        day_chg = (closes[-1] - closes[-2]) / closes[-2] * 100 if len(closes) > 1 else 0.0
        chg_7d = (closes[-1] - closes[-7]) / closes[-7] * 100 if len(closes) >= 7 else 0.0
        chg_30d = (closes[-1] - closes[-30]) / closes[-30] * 100 if len(closes) >= 30 else 0.0

        # Volatility (30-day annualized)
        returns = np.diff(np.log(closes[-31:])) if len(closes) >= 31 else np.array([0])
        volatility_30d = float(np.std(returns) * np.sqrt(365) * 100)

        # Distance from ATH
        ath = float(np.max(highs))
        drawdown = (price - ath) / ath * 100

        # BTC correlation (if not BTC itself)
        btc_corr = None
        if symbol != "BTC":
            try:
                btc_df = yf.Ticker("BTC-USD").history(start=start, end=end, auto_adjust=True)
                if not btc_df.empty and len(btc_df) > 30:
                    btc_closes = btc_df["Close"].values.astype(float)
                    min_len = min(len(closes), len(btc_closes))
                    if min_len > 30:
                        r1 = np.diff(np.log(closes[-min_len:]))
                        r2 = np.diff(np.log(btc_closes[-min_len:]))
                        btc_corr = round(float(np.corrcoef(r1, r2)[0, 1]), 3)
            except Exception:
                pass

        return {
            "symbol": symbol,
            "yf_ticker": yf_ticker,
            "price": round(price, 6),
            "day_chg": round(day_chg, 2),
            "chg_7d": round(chg_7d, 2),
            "chg_30d": round(chg_30d, 2),
            "volatility_30d": round(volatility_30d, 1),
            "ath": round(ath, 6),
            "drawdown_from_ath": round(drawdown, 1),
            "btc_correlation": btc_corr,
            "signal": direction,
            "confidence": confidence,
            "notes": notes,
            "trend": "Up" if price > ind["sma200"] else "Down",
            "indicators": ind,
            "supports": supports,
            "resistances": resistances,
        }
    except Exception:
        return None


def scan_all_crypto(workers: int = 8) -> dict:
    """Scan all crypto tickers in parallel."""
    t0 = time.time()
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_analyze_crypto, sym, yf_t): sym
                   for sym, yf_t in CRYPTO_TICKERS.items()}
        for f in concurrent.futures.as_completed(futures):
            r = f.result()
            if r:
                results.append(r)

    results.sort(key=lambda x: x["confidence"], reverse=True)

    buys = [r for r in results if r["signal"] == "BUY"]
    sells = [r for r in results if r["signal"] == "SELL"]

    # Market-wide crypto metrics
    avg_chg = np.mean([r["day_chg"] for r in results]) if results else 0
    avg_7d = np.mean([r["chg_7d"] for r in results]) if results else 0
    buy_ratio = len(buys) / len(results) if results else 0

    # Fear/greed proxy based on aggregate indicators
    avg_rsi = np.mean([r["indicators"]["rsi"] for r in results]) if results else 50
    if avg_rsi > 65 and buy_ratio > 0.6:
        fear_greed = "EXTREME_GREED"
    elif avg_rsi > 55 and buy_ratio > 0.4:
        fear_greed = "GREED"
    elif avg_rsi < 35 and buy_ratio < 0.3:
        fear_greed = "EXTREME_FEAR"
    elif avg_rsi < 45 and buy_ratio < 0.4:
        fear_greed = "FEAR"
    else:
        fear_greed = "NEUTRAL"

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "elapsed_s": round(time.time() - t0, 1),
        "total_scanned": len(results),
        "market_metrics": {
            "avg_daily_change": round(avg_chg, 2),
            "avg_7d_change": round(avg_7d, 2),
            "buy_sell_ratio": f"{len(buys)}:{len(sells)}",
            "avg_rsi": round(avg_rsi, 1),
            "fear_greed_proxy": fear_greed,
        },
        "all_results": results,
        "top_buys": buys[:10],
        "top_sells": sells[:10],
    }


if __name__ == "__main__":
    print(f"Scanning {len(CRYPTO_TICKERS)} crypto assets...")
    r = scan_all_crypto()
    print(f"\nDone in {r['elapsed_s']}s — {r['total_scanned']} analyzed\n")

    m = r["market_metrics"]
    print(f"Market: {m['fear_greed_proxy']} | Avg Daily: {m['avg_daily_change']:+.1f}% | "
          f"7D: {m['avg_7d_change']:+.1f}% | RSI: {m['avg_rsi']:.0f} | B/S: {m['buy_sell_ratio']}")

    print(f"\n{'#':<3} {'Symbol':<7} {'Price':>12} {'24h':>7} {'7d':>7} {'Signal':<5} {'Conf':>5} {'RSI':>5} {'Vol30d':>6}")
    print("-" * 75)
    for i, c in enumerate(r["all_results"][:15], 1):
        print(f"{i:<3} {c['symbol']:<7} ${c['price']:>10,.2f} {c['day_chg']:>+6.1f}% {c['chg_7d']:>+6.1f}% "
              f"{c['signal']:<5} {c['confidence']:>4.0f}% {c['indicators']['rsi']:>5.1f} {c['volatility_30d']:>5.0f}%")
