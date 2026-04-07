"""
Market Overview — Heatmap, Fear & Greed, Breadth, Watchlist
=============================================================
Inspired by Finviz, TradingView, CNN Fear & Greed.
All data from yfinance — 100% local, zero paid APIs.
"""

import warnings; warnings.filterwarnings("ignore")
import os, json, time, math
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf

WATCHLIST_PATH = os.path.join(os.path.dirname(__file__), "watchlist.json")

# ─── MARKET INDICES ─────────────────────────────────────────────
INDICES = {
    "S&P 500": "^GSPC", "NASDAQ": "^IXIC", "DOW": "^DJI",
    "Russell 2000": "^RUT", "VIX": "^VIX",
    "10Y Treasury": "^TNX", "US Dollar": "DX-Y.NYB",
    "Gold": "GC=F", "Oil (WTI)": "CL=F", "Bitcoin": "BTC-USD",
}

# Sector ETFs for heatmap
SECTOR_ETFS = {
    "Technology": "XLK", "Healthcare": "XLV", "Financials": "XLF",
    "Consumer Disc.": "XLY", "Consumer Staples": "XLP", "Energy": "XLE",
    "Industrials": "XLI", "Materials": "XLB", "Real Estate": "XLRE",
    "Utilities": "XLU", "Communication": "XLC",
}


def _safe_pct(current, previous):
    if previous and previous != 0:
        return round((current - previous) / abs(previous) * 100, 2)
    return 0


def get_market_indices() -> list:
    """Get current values for major market indices."""
    results = []
    from data_fetcher import fetch, batch_fetch_parallel
    data_dict = batch_fetch_parallel(list(INDICES.values()), period="5d", workers=10)

    for name, sym in INDICES.items():
        try:
            df = data_dict.get(sym)
            if df is None or df.empty:
                continue
            close = df["Close"].dropna()
            if len(close) < 2:
                continue
            price = float(close.iloc[-1])
            prev = float(close.iloc[-2])
            chg = _safe_pct(price, prev)
            results.append({
                "name": name, "symbol": sym,
                "price": round(price, 2), "change_pct": chg,
                "direction": "up" if chg > 0 else "down" if chg < 0 else "flat",
            })
        except Exception:
            continue
    return results


def get_sector_heatmap() -> list:
    """Sector ETF performance heatmap (1D, 1W, 1M)."""
    results = []
    etfs = list(SECTOR_ETFS.values())
    from data_fetcher import batch_fetch
    data_dict = batch_fetch(etfs, period="1mo")

    for name, sym in SECTOR_ETFS.items():
        try:
            df = data_dict.get(sym)
            if df is None or df.empty:
                continue
            close = df["Close"].dropna().values.astype(float)
            if len(close) < 5:
                continue
            price = close[-1]
            chg_1d = _safe_pct(close[-1], close[-2]) if len(close) >= 2 else 0
            chg_1w = _safe_pct(close[-1], close[-5]) if len(close) >= 5 else 0
            chg_1m = _safe_pct(close[-1], close[0])
            results.append({
                "sector": name, "etf": sym, "price": round(price, 2),
                "chg_1d": chg_1d, "chg_1w": chg_1w, "chg_1m": chg_1m,
            })
        except Exception:
            continue

    results.sort(key=lambda x: x["chg_1d"], reverse=True)
    return results


def get_fear_greed() -> dict:
    """
    Local Fear & Greed proxy using:
    - VIX level (fear gauge)
    - Market breadth (% above SMA50)
    - Put/Call ratio proxy
    - Momentum (S&P vs SMA)
    - Junk bond demand proxy
    """
    scores = []
    details = {}

    try:
        # 1. VIX — lower = greed, higher = fear
        from data_fetcher import fetch as _df
        vix = _df("^VIX", period="5d")
        if not vix.empty:
            vix_val = float(vix["Close"].iloc[-1])
            if vix_val < 15:
                s = 85  # extreme greed
            elif vix_val < 20:
                s = 65  # greed
            elif vix_val < 25:
                s = 50  # neutral
            elif vix_val < 30:
                s = 30  # fear
            else:
                s = 10  # extreme fear
            scores.append(s)
            details["vix"] = {"value": round(vix_val, 1), "score": s}
    except Exception:
        pass

    try:
        # 2. S&P 500 momentum — price vs 50-day SMA
        from data_fetcher import fetch as _df
        spy = _df("SPY", period="3mo")
        if not spy.empty and len(spy) > 50:
            close = spy["Close"].values.astype(float)
            sma50 = np.mean(close[-50:])
            ratio = (close[-1] / sma50 - 1) * 100
            s = min(90, max(10, 50 + ratio * 10))
            scores.append(s)
            details["momentum"] = {"price": round(close[-1], 2), "sma50": round(sma50, 2), "pct_above": round(ratio, 2), "score": round(s)}
    except Exception:
        pass

    try:
        # 3. Market breadth — how many of top stocks are up today
        tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "UNH", "V",
                    "XOM", "JNJ", "PG", "HD", "MA", "BAC", "PFE", "KO", "PEP", "COST"]
        from data_fetcher import batch_fetch
        data_dict = batch_fetch(tickers, period="5d")
        up = 0
        for t in tickers:
            try:
                df = data_dict.get(t)
                if df is None or df.empty:
                    continue
                c = df["Close"].dropna()
                if len(c) >= 2 and float(c.iloc[-1]) > float(c.iloc[-2]):
                    up += 1
            except Exception:
                continue
        breadth = up / len(tickers) * 100
        s = min(90, max(10, breadth))
        scores.append(s)
        details["breadth"] = {"up": up, "total": len(tickers), "pct": round(breadth, 1), "score": round(s)}
    except Exception:
        pass

    try:
        # 4. Safe haven demand — gold vs S&P ratio change
        from data_fetcher import fetch as _df
        gold = _df("GC=F", period="1mo")
        sp = _df("^GSPC", period="1mo")
        if not gold.empty and not sp.empty:
            g_chg = (float(gold["Close"].iloc[-1]) / float(gold["Close"].iloc[0]) - 1) * 100
            s_chg = (float(sp["Close"].iloc[-1]) / float(sp["Close"].iloc[0]) - 1) * 100
            # If gold outperforming stocks = fear
            diff = s_chg - g_chg
            s = min(90, max(10, 50 + diff * 5))
            scores.append(s)
            details["safe_haven"] = {"gold_1m": round(g_chg, 2), "sp500_1m": round(s_chg, 2), "score": round(s)}
    except Exception:
        pass

    # Composite
    if scores:
        composite = round(np.mean(scores))
    else:
        composite = 50

    if composite >= 75:
        label = "Extreme Greed"
    elif composite >= 60:
        label = "Greed"
    elif composite >= 40:
        label = "Neutral"
    elif composite >= 25:
        label = "Fear"
    else:
        label = "Extreme Fear"

    return {
        "score": composite, "label": label,
        "components": details,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# ─── WATCHLIST ──────────────────────────────────────────────────
def _load_watchlist() -> list:
    if os.path.exists(WATCHLIST_PATH):
        try:
            with open(WATCHLIST_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _save_watchlist(wl: list):
    with open(WATCHLIST_PATH, "w") as f:
        json.dump(wl, f, indent=2)


def get_watchlist() -> list:
    """Get watchlist with live prices and alert status."""
    wl = _load_watchlist()
    if not wl:
        return []

    tickers = [w["ticker"] for w in wl]
    from data_fetcher import batch_fetch
    data_dict = batch_fetch(tickers, period="5d")

    for w in wl:
        try:
            t = w["ticker"]
            df = data_dict.get(t)
            if df is None or df.empty:
                continue
            close = df["Close"].dropna()
            price = float(close.iloc[-1])
            prev = float(close.iloc[-2]) if len(close) >= 2 else price
            w["price"] = round(price, 2)
            w["change_pct"] = _safe_pct(price, prev)

            # Check alerts
            alerts = []
            if w.get("alert_above") and price >= w["alert_above"]:
                alerts.append(f"ABOVE ${w['alert_above']}")
            if w.get("alert_below") and price <= w["alert_below"]:
                alerts.append(f"BELOW ${w['alert_below']}")
            w["triggered_alerts"] = alerts
        except Exception:
            continue

    return wl


def add_to_watchlist(ticker: str, alert_above: float = None, alert_below: float = None, notes: str = "") -> dict:
    wl = _load_watchlist()
    # Remove if exists
    wl = [w for w in wl if w["ticker"] != ticker.upper()]
    wl.append({
        "ticker": ticker.upper(),
        "added_at": datetime.utcnow().isoformat() + "Z",
        "alert_above": alert_above,
        "alert_below": alert_below,
        "notes": notes,
    })
    _save_watchlist(wl)
    return {"status": "added", "ticker": ticker.upper(), "watchlist_size": len(wl)}


def remove_from_watchlist(ticker: str) -> dict:
    wl = _load_watchlist()
    before = len(wl)
    wl = [w for w in wl if w["ticker"] != ticker.upper()]
    _save_watchlist(wl)
    return {"status": "removed", "ticker": ticker.upper(), "removed": before - len(wl)}


def get_market_movers() -> dict:
    """Top gainers, losers, and most active from S&P 500 sample."""
    from sector_scanner import SECTORS
    all_tickers = []
    for tickers in SECTORS.values():
        all_tickers.extend(tickers)

    # Sample 50 for speed
    import random
    sample = random.sample(all_tickers, min(50, len(all_tickers)))

    from data_fetcher import batch_fetch
    data_dict = batch_fetch(sample, period="5d")
    movers = []
    for t in sample:
        try:
            df = data_dict.get(t)
            if df is None or df.empty:
                continue
            close = df["Close"].dropna()
            vol = df["Volume"].dropna()
            if len(close) < 2:
                continue
            price = float(close.iloc[-1])
            chg = _safe_pct(price, float(close.iloc[-2]))
            v = float(vol.iloc[-1]) if len(vol) > 0 else 0
            movers.append({"ticker": t, "price": round(price, 2), "change_pct": chg, "volume": v})
        except Exception:
            continue

    gainers = sorted([m for m in movers if m["change_pct"] > 0], key=lambda x: x["change_pct"], reverse=True)[:10]
    losers = sorted([m for m in movers if m["change_pct"] < 0], key=lambda x: x["change_pct"])[:10]
    active = sorted(movers, key=lambda x: x["volume"], reverse=True)[:10]

    return {"gainers": gainers, "losers": losers, "most_active": active}
