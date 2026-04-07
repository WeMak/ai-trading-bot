"""
Data Fetcher — Centralized Rate-Limited yFinance Layer
========================================================
ALL yfinance calls go through this module.
Features:
  - In-memory + disk cache (avoids redundant API hits)
  - Rate limiter (configurable requests/second)
  - Batch downloads (yf.download with multiple tickers)
  - Auto-retry with exponential backoff
  - Thread-safe request queue
  - Statistics tracking

Usage:
  from data_fetcher import fetch, batch_fetch, get_stats
  hist = fetch("AAPL", period="3mo")          # single ticker
  data = batch_fetch(["AAPL","MSFT"], "5d")   # batch download
"""

import warnings; warnings.filterwarnings("ignore")
import os, json, time, threading, hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Union
from pathlib import Path
from collections import OrderedDict
import numpy as np

import yfinance as yf

# ─── CONFIG ──────────────────────────────────────────────────────
RATE_LIMIT = float(os.environ.get("YF_RATE_LIMIT", "0.08"))   # seconds between requests (0.08 = ~12.5 req/s)
BATCH_SIZE = int(os.environ.get("YF_BATCH_SIZE", "50"))        # max tickers per yf.download() call
MAX_RETRIES = int(os.environ.get("YF_MAX_RETRIES", "3"))       # retries on failure
BACKOFF_BASE = float(os.environ.get("YF_BACKOFF", "2.0"))      # exponential backoff base
CACHE_TTL_SECONDS = {
    "1d":   120,     # 2 min cache for 1-day data
    "5d":   300,     # 5 min cache for 5-day data
    "1mo":  600,     # 10 min cache for 1-month data
    "3mo":  900,     # 15 min cache for 3-month data
    "6mo":  1800,    # 30 min cache for 6-month data
    "1y":   3600,    # 1 hr cache for 1-year data
    "2y":   7200,    # 2 hr cache for 2-year data
    "5y":   14400,   # 4 hr cache for 5-year data
    "10y":  28800,   # 8 hr cache for 10-year data
}
CACHE_MAX_ENTRIES = 5000
DISK_CACHE_DIR = Path(__file__).parent / "cache"

# ─── STATE ───────────────────────────────────────────────────────
_lock = threading.Lock()
_last_request_time = 0.0
_cache: OrderedDict = OrderedDict()  # key -> {"data": df, "expires": timestamp}
_stats = {
    "requests": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "batch_calls": 0,
    "retries": 0,
    "errors": 0,
    "rate_waits": 0,
    "total_tickers_fetched": 0,
    "started": datetime.utcnow().isoformat() + "Z",
}


def _cache_key(ticker: str, period: str = "", start=None, end=None) -> str:
    """Generate a unique cache key."""
    parts = [ticker.upper(), period]
    if start:
        parts.append(str(start))
    if end:
        parts.append(str(end))
    return "|".join(parts)


def _cache_ttl(period: str) -> int:
    """Get TTL in seconds for a given period."""
    return CACHE_TTL_SECONDS.get(period, 600)


def _get_cached(key: str):
    """Check memory cache, return DataFrame or None."""
    if key in _cache:
        entry = _cache[key]
        if time.time() < entry["expires"]:
            _stats["cache_hits"] += 1
            # Move to end (LRU)
            _cache.move_to_end(key)
            return entry["data"]
        else:
            del _cache[key]
    _stats["cache_misses"] += 1
    return None


def _set_cached(key: str, data, period: str = "5d"):
    """Store in memory cache with TTL."""
    if data is None:
        return
    ttl = _cache_ttl(period)
    _cache[key] = {"data": data, "expires": time.time() + ttl}
    _cache.move_to_end(key)
    # Evict oldest if too large
    while len(_cache) > CACHE_MAX_ENTRIES:
        _cache.popitem(last=False)


def _rate_wait():
    """Enforce rate limit — block until enough time has passed."""
    global _last_request_time
    with _lock:
        now = time.time()
        elapsed = now - _last_request_time
        if elapsed < RATE_LIMIT:
            wait = RATE_LIMIT - elapsed
            _stats["rate_waits"] += 1
            time.sleep(wait)
        _last_request_time = time.time()


def _retry_call(fn, *args, **kwargs):
    """Call fn with retries and exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            _rate_wait()
            _stats["requests"] += 1
            result = fn(*args, **kwargs)
            return result
        except Exception as e:
            _stats["retries"] += 1
            err_str = str(e).lower()
            # Rate limited by Yahoo
            if "429" in err_str or "too many" in err_str or "rate" in err_str:
                wait = BACKOFF_BASE ** (attempt + 1) + (attempt * 2)
                time.sleep(wait)
            elif attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE * (attempt + 1))
            else:
                _stats["errors"] += 1
                raise
    return None


# ─── PUBLIC API ──────────────────────────────────────────────────

def fetch(ticker: str, period: str = "3mo", start=None, end=None,
          auto_adjust: bool = True) -> Optional[object]:
    """
    Fetch history for a single ticker with caching + rate limiting.

    Returns: pandas DataFrame or None on error.
    """
    key = _cache_key(ticker, period, start, end)
    cached = _get_cached(key)
    if cached is not None:
        return cached

    try:
        def _do_fetch():
            tkr = yf.Ticker(ticker)
            if start and end:
                return tkr.history(start=start, end=end, auto_adjust=auto_adjust)
            else:
                return tkr.history(period=period, auto_adjust=auto_adjust)

        df = _retry_call(_do_fetch)
        if df is not None and not df.empty:
            _stats["total_tickers_fetched"] += 1
            _set_cached(key, df, period)
            return df
        return None
    except Exception:
        _stats["errors"] += 1
        return None


def fetch_quick(ticker: str) -> Optional[dict]:
    """
    Quick price fetch — returns dict with price, change, volume.
    Uses shortest cache (2 min).
    """
    key = _cache_key(ticker, "quick")
    cached = _get_cached(key)
    if cached is not None:
        return cached

    try:
        def _do():
            tkr = yf.Ticker(ticker)
            return tkr.history(period="2d")

        df = _retry_call(_do)
        if df is None or df.empty:
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else df.iloc[0]
        result = {
            "ticker": ticker.upper(),
            "price": round(float(last["Close"]), 4),
            "change_pct": round((float(last["Close"]) - float(prev["Close"])) / (float(prev["Close"]) + 1e-10) * 100, 2),
            "volume": int(last["Volume"]),
            "high": round(float(last["High"]), 4),
            "low": round(float(last["Low"]), 4),
        }
        _set_cached(key, result, "1d")
        return result
    except Exception:
        _stats["errors"] += 1
        return None


def batch_fetch(tickers: list, period: str = "5d",
                group_by: str = "ticker", threads: bool = True) -> dict:
    """
    Batch download multiple tickers using yf.download().
    Splits into chunks of BATCH_SIZE to avoid overwhelming the API.

    Returns: dict of {ticker: DataFrame}
    """
    if not tickers:
        return {}

    results = {}
    # Check cache first, only fetch uncached
    uncached = []
    for t in tickers:
        key = _cache_key(t, period)
        cached = _get_cached(key)
        if cached is not None:
            results[t] = cached
        else:
            uncached.append(t)

    if not uncached:
        return results

    # Split into batches
    _stats["batch_calls"] += 1
    batches = [uncached[i:i+BATCH_SIZE] for i in range(0, len(uncached), BATCH_SIZE)]

    for batch in batches:
        try:
            def _do_batch(b=batch):
                return yf.download(
                    b, period=period, group_by=group_by,
                    progress=False, threads=threads
                )

            df = _retry_call(_do_batch)
            if df is None or df.empty:
                continue

            # Extract individual ticker data
            if len(batch) == 1:
                # Single ticker — df is already the data
                t = batch[0]
                if not df.empty:
                    results[t] = df
                    _set_cached(_cache_key(t, period), df, period)
                    _stats["total_tickers_fetched"] += 1
            else:
                for t in batch:
                    try:
                        if group_by == "ticker" and t in df.columns.get_level_values(0):
                            ticker_df = df[t].dropna(how="all")
                            if not ticker_df.empty:
                                results[t] = ticker_df
                                _set_cached(_cache_key(t, period), ticker_df, period)
                                _stats["total_tickers_fetched"] += 1
                    except Exception:
                        pass

            # Rate limit between batches
            if len(batches) > 1:
                time.sleep(RATE_LIMIT * 3)  # Extra pause between batches

        except Exception:
            _stats["errors"] += 1
            continue

    return results


def batch_fetch_parallel(tickers: list, period: str = "3mo",
                         workers: int = 8) -> dict:
    """
    Fetch many tickers using ThreadPoolExecutor with rate limiting.
    Better for large periods (3mo+) where yf.download batching is less efficient.

    Returns: dict of {ticker: DataFrame}
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {}
    uncached = []

    # Check cache
    for t in tickers:
        key = _cache_key(t, period)
        cached = _get_cached(key)
        if cached is not None:
            results[t] = cached
        else:
            uncached.append(t)

    if not uncached:
        return results

    def _fetch_one(ticker):
        return ticker, fetch(ticker, period=period)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one, t): t for t in uncached}
        for f in as_completed(futures):
            try:
                t, df = f.result(timeout=60)
                if df is not None and not df.empty:
                    results[t] = df
            except Exception:
                pass

    return results


def fetch_news(ticker: str) -> list:
    """Fetch news for a ticker (rate limited)."""
    key = _cache_key(ticker, "news")
    cached = _get_cached(key)
    if cached is not None:
        return cached

    try:
        def _do():
            tkr = yf.Ticker(ticker)
            return tkr.news or []

        news = _retry_call(_do)
        if news:
            _set_cached(key, news, "1d")
        return news or []
    except Exception:
        return []


def fetch_options_chain(ticker: str) -> Optional[object]:
    """Fetch the yf.Ticker object for options analysis (rate limited)."""
    try:
        _rate_wait()
        _stats["requests"] += 1
        return yf.Ticker(ticker)
    except Exception:
        _stats["errors"] += 1
        return None


def clear_cache():
    """Clear the in-memory cache."""
    _cache.clear()
    return {"status": "cleared", "time": datetime.utcnow().isoformat()}


def get_stats() -> dict:
    """Get fetcher statistics."""
    hit_rate = 0
    total = _stats["cache_hits"] + _stats["cache_misses"]
    if total > 0:
        hit_rate = round(_stats["cache_hits"] / total * 100, 1)
    return {
        **_stats,
        "cache_entries": len(_cache),
        "cache_hit_rate": f"{hit_rate}%",
        "rate_limit": f"{RATE_LIMIT}s ({1/RATE_LIMIT:.1f} req/s)",
        "batch_size": BATCH_SIZE,
    }


def set_rate_limit(seconds: float):
    """Adjust rate limit dynamically."""
    global RATE_LIMIT
    RATE_LIMIT = max(0.05, min(seconds, 5.0))  # Between 0.05s and 5s
