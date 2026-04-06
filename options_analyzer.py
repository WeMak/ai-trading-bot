"""
Options Chain Analyzer v2.0 — Full chain, both directions
============================================================
Scans BOTH calls AND puts at every strike.
Categorizes: Weeklies | Monthlies | Half-LEAPs (3-6mo) | Full LEAPs (6mo-2yr)
Scores by: IV rank, bid-ask spread, volume/OI, moneyness, value vs premium.
Finds hidden value anywhere in the chain — up or down.
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import yfinance as yf
import concurrent.futures, time


def _iv_rank(ticker_obj, current_iv: float) -> float:
    """Estimate IV rank using historical volatility as proxy."""
    try:
        hist = ticker_obj.history(period="1y")
        if len(hist) < 30:
            return 0.5
        closes = hist["Close"].values.astype(float)
        returns = np.diff(np.log(closes))
        if len(returns) < 20:
            return 0.5
        rolling_vol = []
        for i in range(20, len(returns)):
            rv = np.std(returns[i - 20:i]) * np.sqrt(252)
            rolling_vol.append(rv)
        if not rolling_vol:
            return 0.5
        low_v = np.percentile(rolling_vol, 5)
        high_v = np.percentile(rolling_vol, 95)
        if high_v - low_v < 0.01:
            return 0.5
        rank = (current_iv - low_v) / (high_v - low_v)
        return max(0.0, min(1.0, rank))
    except Exception:
        return 0.5


def _classify_timeframe(days: int) -> str:
    """Classify option by timeframe bucket."""
    if days <= 7:
        return "weekly_0_7d"
    elif days <= 14:
        return "weekly_7_14d"
    elif days <= 45:
        return "monthly_14_45d"
    elif days <= 90:
        return "quarterly_45_90d"
    elif days <= 180:
        return "half_leap_90_180d"
    elif days <= 365:
        return "leap_180_365d"
    else:
        return "deep_leap_365d_plus"


TIMEFRAME_LABELS = {
    "weekly_0_7d": "Weekly (0-7d)",
    "weekly_7_14d": "Weekly (7-14d)",
    "monthly_14_45d": "Monthly (14-45d)",
    "quarterly_45_90d": "Quarterly (45-90d)",
    "half_leap_90_180d": "Half-LEAP (90-180d)",
    "leap_180_365d": "LEAP (180-365d)",
    "deep_leap_365d_plus": "Deep LEAP (1yr+)",
}


def _score_option(opt: dict, stock_price: float, days_to_exp: int) -> float:
    """Score any option contract 0-100. Direction-agnostic value scoring."""
    score = 0.0
    option_type = opt.get("option_type", "CALL")

    # ── Liquidity (25 pts max) ────────────────────────
    vol = opt.get("volume", 0) or 0
    oi = opt.get("openInterest", 0) or 0
    if vol > 500:
        score += 12
    elif vol > 100:
        score += 8
    elif vol > 20:
        score += 4
    if oi > 2000:
        score += 13
    elif oi > 500:
        score += 9
    elif oi > 100:
        score += 5

    # ── Bid-ask spread (15 pts max) ───────────────────
    bid = opt.get("bid", 0) or 0
    ask = opt.get("ask", 0) or 0
    mid = (bid + ask) / 2
    if mid > 0:
        spread_pct = (ask - bid) / mid
        if spread_pct < 0.03:
            score += 15
        elif spread_pct < 0.06:
            score += 12
        elif spread_pct < 0.10:
            score += 8
        elif spread_pct < 0.20:
            score += 4

    # ── IV value (15 pts max) — lower IV = cheaper option ─
    iv = opt.get("impliedVolatility", 0) or 0
    if iv > 0:
        if iv < 0.20:
            score += 15  # very cheap vol
        elif iv < 0.30:
            score += 12
        elif iv < 0.45:
            score += 8
        elif iv < 0.65:
            score += 4
        # high IV = expensive, no points

    # ── Moneyness sweet spot (20 pts max) ─────────────
    strike = opt.get("strike", stock_price)
    if option_type == "CALL":
        otm_pct = (strike - stock_price) / stock_price
    else:
        otm_pct = (stock_price - strike) / stock_price

    # Slightly OTM (1-5%) = best leverage for directional
    if 0.01 <= otm_pct <= 0.03:
        score += 20
    elif 0.03 < otm_pct <= 0.07:
        score += 15
    elif -0.02 <= otm_pct < 0.01:
        score += 17  # ATM/slightly ITM — good delta
    elif 0.07 < otm_pct <= 0.15:
        score += 8   # moderate OTM — lottery
    elif -0.05 <= otm_pct < -0.02:
        score += 12  # ITM — built-in value

    # ── Time value scoring (15 pts max) ───────────────
    if days_to_exp <= 7:
        score += 5   # risky but explosive
    elif 7 < days_to_exp <= 14:
        score += 8
    elif 14 < days_to_exp <= 45:
        score += 15  # swing sweet spot
    elif 45 < days_to_exp <= 90:
        score += 13
    elif 90 < days_to_exp <= 180:
        score += 10  # half-LEAP
    elif 180 < days_to_exp <= 365:
        score += 8   # LEAP
    else:
        score += 6   # deep LEAP

    # ── Premium affordability (10 pts max) ────────────
    last_price = opt.get("lastPrice", 0) or mid
    if 0.05 < last_price <= 1.50:
        score += 10  # very cheap
    elif last_price <= 3.00:
        score += 8
    elif last_price <= 6.00:
        score += 5
    elif last_price <= 12.00:
        score += 2

    # ── Volume/OI ratio — unusual activity bonus ─────
    if oi > 0 and vol > 0:
        vol_oi = vol / oi
        if vol_oi > 3.0:
            score += 5  # unusual activity — smart money signal
        elif vol_oi > 1.5:
            score += 3

    return min(score, 100)


def analyze_options(ticker: str, signal: str = "BOTH", min_days: int = 0,
                    max_days: int = 730, top_n: int = 10) -> dict:
    """
    Full options chain analysis.
    signal: "BOTH" scans calls+puts, "BUY" = calls only, "SELL" = puts only.
    Scans from 0 days to 2 years (LEAPs).
    """
    try:
        t0 = time.time()
        tkr = yf.Ticker(ticker)
        hist = tkr.history(period="5d")
        if hist.empty:
            return {"error": f"No price data for {ticker}", "ticker": ticker}
        stock_price = float(hist["Close"].iloc[-1])

        expiry_dates = tkr.options
        if not expiry_dates:
            return {"error": f"No options available for {ticker}", "ticker": ticker}

        now = datetime.now()
        valid_expiries = []
        for exp_str in expiry_dates:
            exp_dt = datetime.strptime(exp_str, "%Y-%m-%d")
            days = (exp_dt - now).days
            if min_days <= days <= max_days:
                valid_expiries.append((exp_str, days))

        if not valid_expiries:
            return {"error": f"No expiries in {min_days}-{max_days} day range", "ticker": ticker}

        all_calls = []
        all_puts = []

        for exp_str, days_to_exp in valid_expiries:
            try:
                chain = tkr.option_chain(exp_str)
                tf = _classify_timeframe(days_to_exp)

                # ── Calls ─────────────────────────
                if signal in ("BOTH", "BUY"):
                    for _, row in chain.calls.iterrows():
                        c = _build_contract(row, exp_str, days_to_exp, "CALL", stock_price, tf)
                        if c:
                            all_calls.append(c)

                # ── Puts ──────────────────────────
                if signal in ("BOTH", "SELL"):
                    for _, row in chain.puts.iterrows():
                        c = _build_contract(row, exp_str, days_to_exp, "PUT", stock_price, tf)
                        if c:
                            all_puts.append(c)

            except Exception:
                continue

        # Sort both by score
        all_calls.sort(key=lambda x: x["score"], reverse=True)
        all_puts.sort(key=lambda x: x["score"], reverse=True)
        all_combined = sorted(all_calls + all_puts, key=lambda x: x["score"], reverse=True)

        # IV rank
        all_ivs = [c["impliedVolatility"] for c in all_combined if c["impliedVolatility"] > 0]
        avg_iv = np.mean(all_ivs) if all_ivs else 0.3
        iv_rank = _iv_rank(tkr, avg_iv)

        # Group by timeframe
        by_timeframe = {}
        for tf_key in TIMEFRAME_LABELS:
            tf_calls = [c for c in all_calls if c["timeframe"] == tf_key][:5]
            tf_puts = [c for c in all_puts if c["timeframe"] == tf_key][:5]
            if tf_calls or tf_puts:
                by_timeframe[tf_key] = {
                    "label": TIMEFRAME_LABELS[tf_key],
                    "best_calls": tf_calls,
                    "best_puts": tf_puts,
                    "total_calls": len([c for c in all_calls if c["timeframe"] == tf_key]),
                    "total_puts": len([c for c in all_puts if c["timeframe"] == tf_key]),
                }

        # Find unusual activity (vol/OI > 3x)
        unusual = [c for c in all_combined if c.get("vol_oi_ratio", 0) > 3.0]
        unusual.sort(key=lambda x: x["vol_oi_ratio"], reverse=True)

        # Find cheapest high-score options (value plays)
        value_plays = [c for c in all_combined if c["score"] >= 50 and c["lastPrice"] <= 3.0]
        value_plays.sort(key=lambda x: x["score"], reverse=True)

        return {
            "ticker": ticker,
            "stock_price": round(stock_price, 4),
            "signal": signal,
            "iv_rank": round(iv_rank, 4),
            "avg_iv": round(avg_iv, 4),
            "total_calls_analyzed": len(all_calls),
            "total_puts_analyzed": len(all_puts),
            "total_contracts_analyzed": len(all_combined),
            "valid_expiries": len(valid_expiries),
            "top_calls": all_calls[:top_n],
            "top_puts": all_puts[:top_n],
            "top_overall": all_combined[:top_n],
            "by_timeframe": by_timeframe,
            "unusual_activity": unusual[:10],
            "value_plays": value_plays[:10],
            "elapsed_s": round(time.time() - t0, 1),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    except Exception as e:
        return {"error": str(e), "ticker": ticker}


def _build_contract(row, exp_str, days_to_exp, option_type, stock_price, tf) -> Optional[dict]:
    """Build a scored contract dict from a dataframe row."""
    contract = {
        "contractSymbol": row.get("contractSymbol", ""),
        "strike": float(row.get("strike", 0)),
        "lastPrice": float(row.get("lastPrice", 0) or 0),
        "bid": float(row.get("bid", 0) or 0),
        "ask": float(row.get("ask", 0) or 0),
        "volume": int(row.get("volume", 0) or 0),
        "openInterest": int(row.get("openInterest", 0) or 0),
        "impliedVolatility": float(row.get("impliedVolatility", 0) or 0),
        "expiry": exp_str,
        "days_to_expiry": days_to_exp,
        "option_type": option_type,
        "timeframe": tf,
        "timeframe_label": TIMEFRAME_LABELS.get(tf, tf),
        "inTheMoney": bool(row.get("inTheMoney", False)),
    }

    # Skip zero-premium contracts
    if contract["lastPrice"] <= 0 and contract["bid"] <= 0:
        return None

    contract["score"] = _score_option(contract, stock_price, days_to_exp)

    # Moneyness
    if option_type == "CALL":
        m = (contract["strike"] - stock_price) / stock_price * 100
    else:
        m = (stock_price - contract["strike"]) / stock_price * 100
    contract["moneyness_pct"] = round(m, 2)
    contract["moneyness"] = "ITM" if m < -1 else "ATM" if abs(m) < 1 else "OTM"

    # Premium / cost
    premium = contract["lastPrice"] or ((contract["bid"] + contract["ask"]) / 2)
    contract["mid_price"] = round((contract["bid"] + contract["ask"]) / 2, 4) if contract["ask"] > 0 else contract["lastPrice"]
    contract["total_cost_1c"] = round(premium * 100, 2)
    contract["total_cost_3c"] = round(premium * 100 * 3, 2)

    # Breakeven
    if option_type == "CALL":
        contract["breakeven"] = round(contract["strike"] + premium, 4)
        contract["breakeven_pct"] = round((contract["breakeven"] - stock_price) / stock_price * 100, 2)
    else:
        contract["breakeven"] = round(contract["strike"] - premium, 4)
        contract["breakeven_pct"] = round((stock_price - contract["breakeven"]) / stock_price * 100, 2)

    # Vol/OI ratio (unusual activity flag)
    oi = contract["openInterest"]
    vol = contract["volume"]
    contract["vol_oi_ratio"] = round(vol / oi, 2) if oi > 0 else 0.0

    # Bid-ask spread %
    mid = (contract["bid"] + contract["ask"]) / 2
    contract["spread_pct"] = round((contract["ask"] - contract["bid"]) / mid * 100, 1) if mid > 0 else 999

    # Intrinsic vs extrinsic value
    if option_type == "CALL":
        intrinsic = max(0, stock_price - contract["strike"])
    else:
        intrinsic = max(0, contract["strike"] - stock_price)
    contract["intrinsic_value"] = round(intrinsic, 4)
    contract["extrinsic_value"] = round(max(0, premium - intrinsic), 4)
    contract["extrinsic_pct"] = round(contract["extrinsic_value"] / premium * 100, 1) if premium > 0 else 0

    return contract


def scan_options_batch(tickers_with_signals: list, top_n: int = 5) -> list:
    """Analyze options for multiple tickers in parallel."""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {}
        for item in tickers_with_signals:
            sig = item.get("signal", "BOTH")
            f = pool.submit(analyze_options, item["ticker"], sig, top_n=top_n)
            futures[f] = item["ticker"]
        for f in concurrent.futures.as_completed(futures):
            r = f.result()
            if r and "error" not in r:
                results.append(r)
    return results


if __name__ == "__main__":
    print("Options Analyzer v2.0 — Full chain scan AAPL...")
    r = analyze_options("AAPL", "BOTH", min_days=0, max_days=730, top_n=10)
    if "error" in r:
        print(f"Error: {r['error']}")
    else:
        print(f"Stock: ${r['stock_price']}  IV Rank: {r['iv_rank']:.0%}  Avg IV: {r['avg_iv']:.0%}")
        print(f"Analyzed: {r['total_calls_analyzed']} calls + {r['total_puts_analyzed']} puts = {r['total_contracts_analyzed']} total")
        print(f"Expiries: {r['valid_expiries']}")

        print(f"\nTOP CALLS:")
        for i, c in enumerate(r["top_calls"][:5], 1):
            print(f"  {i}. CALL ${c['strike']:.0f} exp {c['expiry']} ({c['days_to_expiry']}d) "
                  f"${c['lastPrice']:.2f} Score:{c['score']:.0f} {c['moneyness']} "
                  f"IV:{c['impliedVolatility']:.0%} Vol:{c['volume']} OI:{c['openInterest']} [{c['timeframe_label']}]")

        print(f"\nTOP PUTS:")
        for i, c in enumerate(r["top_puts"][:5], 1):
            print(f"  {i}. PUT ${c['strike']:.0f} exp {c['expiry']} ({c['days_to_expiry']}d) "
                  f"${c['lastPrice']:.2f} Score:{c['score']:.0f} {c['moneyness']} "
                  f"IV:{c['impliedVolatility']:.0%} Vol:{c['volume']} OI:{c['openInterest']} [{c['timeframe_label']}]")

        if r["unusual_activity"]:
            print(f"\nUNUSUAL ACTIVITY:")
            for c in r["unusual_activity"][:5]:
                print(f"  {c['option_type']} ${c['strike']:.0f} {c['expiry']} "
                      f"Vol/OI: {c['vol_oi_ratio']:.1f}x  Vol:{c['volume']} OI:{c['openInterest']}")

        if r["value_plays"]:
            print(f"\nVALUE PLAYS (Score>50, <$3):")
            for c in r["value_plays"][:5]:
                print(f"  {c['option_type']} ${c['strike']:.0f} {c['expiry']} "
                      f"${c['lastPrice']:.2f} Score:{c['score']:.0f} {c['moneyness']}")

        print(f"\nBY TIMEFRAME:")
        for tf_key, tf_data in r["by_timeframe"].items():
            print(f"  {tf_data['label']}: {tf_data['total_calls']} calls, {tf_data['total_puts']} puts")
