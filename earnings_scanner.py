"""
Earnings Scanner — Dates, Expected Moves, 2-Sigma Deviations
==============================================================
Pulls earnings dates from yfinance.
Calculates expected move from options IV (straddle pricing).
Shows 1 and 2 standard deviation ranges.
Flags stocks with upcoming earnings for pre-earnings plays.
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import yfinance as yf
import concurrent.futures, time


def _get_earnings_date(tkr_obj) -> Optional[dict]:
    """Get next earnings date from yfinance calendar."""
    try:
        cal = tkr_obj.calendar
        if cal is None or cal.empty:
            return None
        # yfinance returns calendar as a DataFrame or dict
        if hasattr(cal, 'iloc'):
            # DataFrame format
            if "Earnings Date" in cal.index:
                dates = cal.loc["Earnings Date"]
                if hasattr(dates, 'iloc') and len(dates) > 0:
                    ed = dates.iloc[0]
                else:
                    ed = dates
                if hasattr(ed, 'isoformat'):
                    return {"date": ed.isoformat()[:10], "source": "calendar"}
        elif isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if ed:
                if isinstance(ed, list) and len(ed) > 0:
                    d = ed[0]
                    if hasattr(d, 'isoformat'):
                        return {"date": d.isoformat()[:10], "source": "calendar"}
                    return {"date": str(d)[:10], "source": "calendar"}
                elif hasattr(ed, 'isoformat'):
                    return {"date": ed.isoformat()[:10], "source": "calendar"}
    except Exception:
        pass

    # Fallback: check earnings_dates attribute
    try:
        ed = tkr_obj.earnings_dates
        if ed is not None and not ed.empty:
            future = ed[ed.index >= datetime.now()]
            if not future.empty:
                d = future.index[0]
                return {"date": d.strftime("%Y-%m-%d"), "source": "earnings_dates"}
    except Exception:
        pass

    return None


def _expected_move_from_straddle(tkr_obj, stock_price: float, earnings_date_str: str) -> Optional[dict]:
    """
    Calculate expected move from ATM straddle price.
    The straddle price = market's expected move.
    1 SD = straddle price
    2 SD = straddle price * 2
    """
    try:
        expiries = tkr_obj.options
        if not expiries:
            return None

        earnings_dt = datetime.strptime(earnings_date_str, "%Y-%m-%d")
        now = datetime.now()

        # Find the nearest expiry AFTER earnings
        best_exp = None
        best_gap = 999
        for exp_str in expiries:
            exp_dt = datetime.strptime(exp_str, "%Y-%m-%d")
            gap = (exp_dt - earnings_dt).days
            if 0 <= gap <= 7 and gap < best_gap:
                best_gap = gap
                best_exp = exp_str

        # If no post-earnings expiry, use closest one
        if not best_exp:
            for exp_str in expiries:
                exp_dt = datetime.strptime(exp_str, "%Y-%m-%d")
                gap = abs((exp_dt - earnings_dt).days)
                if gap < best_gap:
                    best_gap = gap
                    best_exp = exp_str

        if not best_exp:
            return None

        chain = tkr_obj.option_chain(best_exp)
        days_to_exp = (datetime.strptime(best_exp, "%Y-%m-%d") - now).days

        # Find ATM strikes (closest to stock price)
        calls = chain.calls
        puts = chain.puts

        if calls.empty or puts.empty:
            return None

        # ATM call
        calls_sorted = calls.iloc[(calls["strike"] - stock_price).abs().argsort()]
        atm_call = calls_sorted.iloc[0]

        # ATM put
        puts_sorted = puts.iloc[(puts["strike"] - stock_price).abs().argsort()]
        atm_put = puts_sorted.iloc[0]

        # Straddle price = ATM call + ATM put
        call_price = float(atm_call.get("lastPrice", 0) or 0)
        put_price = float(atm_put.get("lastPrice", 0) or 0)

        # Use mid if lastPrice is 0
        if call_price <= 0:
            call_price = (float(atm_call.get("bid", 0) or 0) + float(atm_call.get("ask", 0) or 0)) / 2
        if put_price <= 0:
            put_price = (float(atm_put.get("bid", 0) or 0) + float(atm_put.get("ask", 0) or 0)) / 2

        straddle = call_price + put_price
        if straddle <= 0:
            return None

        # Expected move = straddle price (1 SD)
        # For earnings specifically, multiply by ~0.85 to isolate earnings move
        # (removes time value from non-earnings days)
        earnings_days = max(1, (earnings_dt - now).days)
        if days_to_exp > earnings_days + 2:
            # Adjust: isolate the earnings-day move
            daily_theta_portion = straddle * (1 - earnings_days / max(days_to_exp, 1))
            earnings_move = straddle - daily_theta_portion * 0.5
        else:
            earnings_move = straddle * 0.85

        move_pct = (earnings_move / stock_price) * 100

        # Standard deviations
        sd1_up = stock_price + earnings_move
        sd1_down = stock_price - earnings_move
        sd2_up = stock_price + (earnings_move * 2)
        sd2_down = stock_price - (earnings_move * 2)

        # ATM IV for reference
        atm_call_iv = float(atm_call.get("impliedVolatility", 0) or 0)
        atm_put_iv = float(atm_put.get("impliedVolatility", 0) or 0)
        avg_atm_iv = (atm_call_iv + atm_put_iv) / 2

        return {
            "expiry_used": best_exp,
            "days_to_expiry": days_to_exp,
            "atm_strike": float(atm_call["strike"]),
            "atm_call_price": round(call_price, 4),
            "atm_put_price": round(put_price, 4),
            "straddle_price": round(straddle, 4),
            "expected_move": round(earnings_move, 4),
            "expected_move_pct": round(move_pct, 2),
            "atm_iv": round(avg_atm_iv, 4),
            "sd1": {
                "up": round(sd1_up, 4),
                "down": round(sd1_down, 4),
                "range_pct": round(move_pct, 2),
                "probability": "68.2%",
            },
            "sd2": {
                "up": round(sd2_up, 4),
                "down": round(sd2_down, 4),
                "range_pct": round(move_pct * 2, 2),
                "probability": "95.4%",
            },
        }
    except Exception:
        return None


def _historical_earnings_moves(tkr_obj, n_quarters: int = 8) -> Optional[dict]:
    """Analyze past earnings reactions to estimate future moves."""
    try:
        ed = tkr_obj.earnings_dates
        if ed is None or ed.empty:
            return None

        past = ed[ed.index < datetime.now()].head(n_quarters * 2)
        if past.empty:
            return None

        hist = tkr_obj.history(period="2y")
        if hist.empty:
            return None

        moves = []
        surprise_pcts = []
        for dt in past.index[:n_quarters]:
            date_str = dt.strftime("%Y-%m-%d")
            # Find the closest trading day
            try:
                mask_before = hist.index <= dt
                mask_after = hist.index >= dt
                if not mask_before.any() or not mask_after.any():
                    continue

                idx_before = hist.index[mask_before][-1]
                # Next day after earnings
                after_dates = hist.index[hist.index > dt]
                if len(after_dates) == 0:
                    continue
                idx_after = after_dates[0]

                close_before = float(hist.loc[idx_before, "Close"])
                close_after = float(hist.loc[idx_after, "Close"])
                move = (close_after - close_before) / close_before * 100
                moves.append({
                    "date": date_str,
                    "close_before": round(close_before, 2),
                    "close_after": round(close_after, 2),
                    "move_pct": round(move, 2),
                    "direction": "UP" if move > 0 else "DOWN",
                })
            except Exception:
                continue

            # EPS surprise
            try:
                eps_est = past.loc[dt].get("EPS Estimate")
                eps_act = past.loc[dt].get("Reported EPS")
                if eps_est and eps_act and float(eps_est) != 0:
                    surp = (float(eps_act) - float(eps_est)) / abs(float(eps_est)) * 100
                    surprise_pcts.append(round(surp, 2))
            except Exception:
                pass

        if not moves:
            return None

        abs_moves = [abs(m["move_pct"]) for m in moves]
        avg_move = np.mean(abs_moves)
        max_move = max(abs_moves)
        up_count = sum(1 for m in moves if m["direction"] == "UP")

        return {
            "quarters_analyzed": len(moves),
            "avg_move_pct": round(avg_move, 2),
            "max_move_pct": round(max_move, 2),
            "up_after_earnings": up_count,
            "down_after_earnings": len(moves) - up_count,
            "up_pct": round(up_count / len(moves) * 100, 0),
            "avg_eps_surprise_pct": round(np.mean(surprise_pcts), 2) if surprise_pcts else None,
            "recent_moves": moves[:8],
        }
    except Exception:
        return None


def analyze_earnings(ticker: str) -> dict:
    """Full earnings analysis for a single ticker."""
    t0 = time.time()
    try:
        from data_fetcher import fetch_options_chain
        tkr = fetch_options_chain(ticker)
        if tkr is None:
            return {"error": f"Cannot fetch {ticker}", "ticker": ticker}
        hist = tkr.history(period="5d")
        if hist.empty:
            return {"error": f"No data for {ticker}", "ticker": ticker}

        stock_price = float(hist["Close"].iloc[-1])

        # Get earnings date
        earnings_info = _get_earnings_date(tkr)
        if not earnings_info:
            return {
                "ticker": ticker,
                "stock_price": round(stock_price, 4),
                "earnings_date": None,
                "days_to_earnings": None,
                "has_earnings": False,
                "message": "No upcoming earnings date found",
                "elapsed_s": round(time.time() - t0, 1),
            }

        earnings_date = earnings_info["date"]
        days_to = (datetime.strptime(earnings_date, "%Y-%m-%d") - datetime.now()).days

        # Expected move from straddle
        expected = _expected_move_from_straddle(tkr, stock_price, earnings_date)

        # Historical earnings moves
        historical = _historical_earnings_moves(tkr)

        # Combine for best estimate
        if expected and historical:
            # Weight: 60% options-implied, 40% historical
            implied_move = expected["expected_move_pct"]
            hist_move = historical["avg_move_pct"]
            blended = implied_move * 0.6 + hist_move * 0.4
        elif expected:
            blended = expected["expected_move_pct"]
        elif historical:
            blended = historical["avg_move_pct"]
        else:
            blended = stock_price * 0.05  # fallback 5%

        # 2 SD ranges using blended estimate
        sd1_dollar = stock_price * (blended / 100)
        sd2_dollar = sd1_dollar * 2

        return {
            "ticker": ticker,
            "stock_price": round(stock_price, 4),
            "has_earnings": True,
            "earnings_date": earnings_date,
            "days_to_earnings": days_to,
            "earnings_imminent": 0 <= days_to <= 7,
            "expected_move": {
                "blended_pct": round(blended, 2),
                "blended_dollar": round(sd1_dollar, 4),
                "sd1": {
                    "up": round(stock_price + sd1_dollar, 4),
                    "down": round(stock_price - sd1_dollar, 4),
                    "range_pct": round(blended, 2),
                    "probability": "68.2%",
                    "label": "1 Standard Deviation",
                },
                "sd2": {
                    "up": round(stock_price + sd2_dollar, 4),
                    "down": round(stock_price - sd2_dollar, 4),
                    "range_pct": round(blended * 2, 2),
                    "probability": "95.4%",
                    "label": "2 Standard Deviations",
                },
            },
            "options_implied": expected,
            "historical": historical,
            "elapsed_s": round(time.time() - t0, 1),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    except Exception as e:
        return {"error": str(e), "ticker": ticker}


def scan_earnings_batch(tickers: list) -> dict:
    """Scan multiple tickers for upcoming earnings."""
    t0 = time.time()
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(analyze_earnings, t): t for t in tickers}
        for f in concurrent.futures.as_completed(futures):
            r = f.result()
            if r and not r.get("error"):
                results.append(r)

    # Sort: earnings soonest first
    with_earnings = [r for r in results if r.get("has_earnings") and r.get("days_to_earnings") is not None]
    with_earnings.sort(key=lambda x: x["days_to_earnings"])

    # Split by urgency
    imminent = [r for r in with_earnings if r.get("earnings_imminent")]
    upcoming_7_30 = [r for r in with_earnings if 7 < (r.get("days_to_earnings") or 999) <= 30]
    later = [r for r in with_earnings if (r.get("days_to_earnings") or 0) > 30]

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "elapsed_s": round(time.time() - t0, 1),
        "total_scanned": len(tickers),
        "with_earnings": len(with_earnings),
        "imminent": imminent,
        "upcoming_7_30d": upcoming_7_30,
        "later_30d_plus": later,
        "all_results": with_earnings,
    }


if __name__ == "__main__":
    # Test single ticker
    print("Earnings Scanner — Testing AAPL...")
    r = analyze_earnings("AAPL")
    if r.get("error"):
        print(f"Error: {r['error']}")
    elif r.get("has_earnings"):
        print(f"Stock: ${r['stock_price']}")
        print(f"Earnings: {r['earnings_date']} ({r['days_to_earnings']} days)")

        em = r["expected_move"]
        print(f"\nExpected Move: +/-{em['blended_pct']:.1f}% (${em['blended_dollar']:.2f})")
        print(f"  1 SD (68%): ${em['sd1']['down']:.2f} — ${em['sd1']['up']:.2f}")
        print(f"  2 SD (95%): ${em['sd2']['down']:.2f} — ${em['sd2']['up']:.2f}")

        if r.get("historical"):
            h = r["historical"]
            print(f"\nHistorical ({h['quarters_analyzed']}Q): avg {h['avg_move_pct']:.1f}%, max {h['max_move_pct']:.1f}%, up {h['up_pct']:.0f}% of time")
            for m in h["recent_moves"][:4]:
                print(f"  {m['date']}: {m['direction']} {m['move_pct']:+.1f}%")
    else:
        print(r.get("message", "No earnings data"))

    # Test batch
    print("\n\nBatch scan...")
    tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD", "NFLX", "JPM"]
    batch = scan_earnings_batch(tickers)
    print(f"Scanned {batch['total_scanned']} | With earnings: {batch['with_earnings']}")
    if batch["imminent"]:
        print(f"\nEARNINGS THIS WEEK:")
        for r in batch["imminent"]:
            em = r["expected_move"]
            print(f"  {r['ticker']:<6} {r['earnings_date']} ({r['days_to_earnings']}d) "
                  f"Expected: +/-{em['blended_pct']:.1f}%  "
                  f"1SD: ${em['sd1']['down']:.2f}-${em['sd1']['up']:.2f}  "
                  f"2SD: ${em['sd2']['down']:.2f}-${em['sd2']['up']:.2f}")
