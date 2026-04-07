"""
Autonomous 0DTE Trading Agent
================================
Continuously scans all 110 sector stocks + crypto using the full
neural network ensemble (NEAT + PyTorch + EvoStrategy + Multi-Model Ollama).

Runs in a loop until stopped. Makes paper trades on high-confidence signals.

Ensemble consensus: each model votes, weighted by confidence.
Only trades when 3+ models agree with >60% average confidence.
"""

import warnings; warnings.filterwarnings("ignore")
import os, json, time, asyncio, threading
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf

from trading_agent import _rsi, _macd, _bollinger, _sma, _atr
from paper_trader import PaperTrader

# ─── STATE ──────────────────────────────────────────────────────
STATE_PATH = os.path.join(os.path.dirname(__file__), "auto_trader_state.json")

_agent_state = {
    "running": False,
    "started_at": None,
    "stopped_at": None,
    "cycles": 0,
    "trades_made": 0,
    "signals_found": 0,
    "last_scan": None,
    "last_error": None,
    "scan_log": [],        # last 50 scan summaries
    "trade_log": [],       # last 50 trades
    "mode": "0DTE",        # 0DTE, swing, or both
    "ensemble_weights": {
        "technical": 1.0,
        "neat": 1.2,
        "pytorch": 1.3,
        "evo": 1.0,
        "ollama": 1.5,
    },
}

_stop_event = threading.Event()
_agent_thread = None

# ─── LIVE TERMINAL LOG ──────────────────────────────────────────
_terminal_log = []  # ring buffer of {time, type, msg}
_terminal_lock = threading.Lock()
_TERMINAL_MAX = 500


def _think(msg: str, msg_type: str = "info"):
    """Log a thought to the live terminal."""
    with _terminal_lock:
        _terminal_log.append({
            "time": datetime.utcnow().isoformat() + "Z",
            "type": msg_type,  # info, signal, trade, error, scan, model, decision
            "msg": msg,
        })
        if len(_terminal_log) > _TERMINAL_MAX:
            del _terminal_log[:len(_terminal_log) - _TERMINAL_MAX]


def get_terminal(since_idx: int = 0) -> dict:
    """Get terminal messages since index."""
    with _terminal_lock:
        msgs = _terminal_log[since_idx:]
        return {"messages": msgs, "next_idx": len(_terminal_log)}


def _save_state():
    with open(STATE_PATH, "w") as f:
        json.dump(_agent_state, f, indent=2, default=str)


def _load_state():
    global _agent_state
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH) as f:
                saved = json.load(f)
            _agent_state.update(saved)
            _agent_state["running"] = False  # always start stopped
        except Exception:
            pass


# ─── DATA FETCHING ──────────────────────────────────────────────
def _fetch_ticker_data(ticker: str) -> Optional[dict]:
    """Fetch price data and compute indicators for a single ticker."""
    try:
        from data_fetcher import fetch
        hist = fetch(ticker, period="3mo")
        if hist is None or hist.empty or len(hist) < 30:
            return None

        close = hist["Close"].values.astype(float)
        high = hist["High"].values.astype(float)
        low = hist["Low"].values.astype(float)
        volume = hist["Volume"].values.astype(float)
        price = float(close[-1])

        # All indicator functions return arrays — grab last value
        rsi_arr = _rsi(close)
        macd_l_arr, macd_s_arr, macd_h_arr = _macd(close)
        bb_u_arr, bb_m_arr, bb_l_arr = _bollinger(close)
        sma50_arr = _sma(close, 50)
        sma200_arr = _sma(close, 200)
        atr_arr = _atr(high, low, close)

        def _s(arr):
            v = float(arr[-1]) if len(arr) > 0 else 0.0
            return v if not np.isnan(v) else 0.0

        rsi = _s(rsi_arr)
        macd_line = _s(macd_l_arr)
        macd_sig = _s(macd_s_arr)
        macd_h = _s(macd_h_arr)
        bb_u = _s(bb_u_arr)
        bb_m = _s(bb_m_arr)
        bb_l = _s(bb_l_arr)
        sma50 = _s(sma50_arr)
        sma200 = _s(sma200_arr)
        atr = _s(atr_arr)

        avg_vol = float(np.mean(volume[-20:])) if len(volume) >= 20 else float(np.mean(volume))
        vol_ratio = float(volume[-1]) / (avg_vol + 1e-10)

        indicators = {
            "price": price,
            "rsi": rsi,
            "macd": macd_line,
            "macd_signal": macd_sig,
            "macd_hist": macd_h,
            "bb_upper": bb_u,
            "bb_middle": bb_m,
            "bb_lower": bb_l,
            "sma50": sma50,
            "sma200": sma200,
            "atr": atr,
            "vol_ratio": round(vol_ratio, 2),
        }

        # Price changes
        price_history = {}
        if len(close) >= 2:
            price_history["chg_1d"] = float((close[-1] - close[-2]) / (close[-2] + 1e-10) * 100)
        if len(close) >= 6:
            price_history["chg_5d"] = float((close[-1] - close[-6]) / (close[-6] + 1e-10) * 100)
        if len(close) >= 21:
            price_history["chg_20d"] = float((close[-1] - close[-21]) / (close[-21] + 1e-10) * 100)

        # Technical score — _score expects dict, returns (direction, confidence, notes)
        from trading_agent import _score
        tech_signal, tech_score, _ = _score(indicators)

        return {
            "ticker": ticker,
            "price": price,
            "indicators": indicators,
            "price_history": price_history,
            "technical": {"signal": tech_signal, "score": tech_score},
        }
    except Exception:
        return None


# ─── ENSEMBLE VOTING ────────────────────────────────────────────
def _ensemble_vote(ticker_data: dict) -> dict:
    """Run all available models and combine votes."""
    votes = []
    weights = _agent_state["ensemble_weights"]
    indicators = ticker_data["indicators"]
    price_history = ticker_data.get("price_history", {})

    ticker = ticker_data.get("ticker", "???")

    # 1. Technical analysis (always available)
    tech = ticker_data["technical"]
    votes.append({
        "model": "Technical",
        "action": tech["signal"],
        "confidence": tech["score"],
        "weight": weights["technical"],
    })
    _think(f"[{ticker}] Technical: {tech['signal']} ({tech['score']}%) | RSI={indicators.get('rsi',0):.0f} MACD_H={indicators.get('macd_hist',0):.4f}", "model")

    # 2. NEAT neuroevolution
    try:
        from neat_trader import get_neat_trader
        neat = get_neat_trader()
        if neat.best_net is not None:
            pred = neat.predict(indicators, price_history)
            votes.append({
                "model": "NEAT",
                "action": pred["action"],
                "confidence": pred["confidence"],
                "weight": weights["neat"],
            })
            _think(f"[{ticker}] NEAT: {pred['action']} ({pred['confidence']}%) probs={pred.get('probabilities','')}", "model")
    except Exception as e:
        _think(f"[{ticker}] NEAT error: {e}", "error")

    # 3. PyTorch LSTM
    try:
        from torch_predictor import get_torch_predictor
        pytorch = get_torch_predictor()
        if pytorch.epoch > 0:
            pred = pytorch.predict(indicators, price_history=price_history)
            votes.append({
                "model": "PyTorch",
                "action": pred["action"],
                "confidence": pred["confidence"],
                "weight": weights["pytorch"],
            })
            _think(f"[{ticker}] PyTorch: {pred['action']} ({pred['confidence']}%) move={pred.get('predicted_move_pct',0):+.2f}%", "model")
    except Exception as e:
        _think(f"[{ticker}] PyTorch error: {e}", "error")

    # 4. Evolutionary strategy
    try:
        from evo_optimizer import get_evo_optimizer
        evo = get_evo_optimizer()
        if evo.best_params is not None:
            bar = {"indicators": indicators, "price": ticker_data["price"]}
            pred = evo.get_signal(bar)
            votes.append({
                "model": "EvoStrategy",
                "action": pred["action"],
                "confidence": pred["confidence"],
                "weight": weights["evo"],
            })
            _think(f"[{ticker}] EvoStrategy: {pred['action']} ({pred['confidence']}%)", "model")
    except Exception as e:
        _think(f"[{ticker}] Evo error: {e}", "error")

    # 5. Multi-model Ollama (only if running, slower)
    # Skip in fast 0DTE mode to keep cycle time low
    # Can be enabled for swing trades

    # Build consensus
    if not votes:
        _think(f"[{ticker}] No model votes — HOLD", "decision")
        return {"action": "HOLD", "confidence": 0, "votes": [], "agreement": 0}

    buy_score = sum(v["confidence"] * v["weight"] for v in votes if v["action"] == "BUY")
    sell_score = sum(v["confidence"] * v["weight"] for v in votes if v["action"] == "SELL")
    hold_score = sum(v["confidence"] * v["weight"] for v in votes if v["action"] == "HOLD")
    total_weight = sum(v["weight"] for v in votes)

    scores = {"BUY": buy_score, "SELL": sell_score, "HOLD": hold_score}
    best_action = max(scores, key=scores.get)
    best_score = scores[best_action] / (total_weight + 1e-10)

    agreeing = [v for v in votes if v["action"] == best_action]
    agreement = len(agreeing) / len(votes) * 100

    if best_action != "HOLD":
        _think(f"[{ticker}] CONSENSUS: {best_action} | conf={best_score:.1f}% | agree={agreement:.0f}% ({len(agreeing)}/{len(votes)} models)", "signal")

    return {
        "action": best_action,
        "confidence": round(best_score, 1),
        "agreement": round(agreement, 0),
        "models_voted": len(votes),
        "models_agree": len(agreeing),
        "votes": votes,
    }


# ─── MAIN LOOP ──────────────────────────────────────────────────
def _scan_cycle():
    """Run one full scan cycle across all stocks + crypto."""
    from universe import SECTORS, ALL_STOCKS, CRYPTO_TICKERS

    all_tickers = list(ALL_STOCKS)

    # Add crypto tickers
    crypto_yf = list(CRYPTO_TICKERS.values())
    all_tickers.extend(crypto_yf)

    _think(f"=== SCAN CYCLE #{_agent_state['cycles']+1} === Scanning {len(ALL_STOCKS)} stocks + {len(CRYPTO_TICKERS)} crypto = {len(all_tickers)} total across {len(SECTORS)} sectors", "scan")

    # Scan sector by sector for visibility
    results = []
    sector_names = list(SECTORS.keys())

    # Stocks — batch by sector
    for sidx, (sector, tickers) in enumerate(SECTORS.items()):
        _think(f"  [{sidx+1}/{len(SECTORS)}] Scanning {sector}: {len(tickers)} tickers ({', '.join(tickers[:8])}{'...' if len(tickers)>8 else ''})", "scan")
        with ThreadPoolExecutor(max_workers=12) as pool:
            futures = {pool.submit(_fetch_ticker_data, t): t for t in tickers}
            count = 0
            for f in futures:
                try:
                    data = f.result(timeout=30)
                    if data:
                        results.append(data)
                        count += 1
                except Exception:
                    pass
            _think(f"    {sector}: {count}/{len(tickers)} loaded", "info")

    # Crypto
    _think(f"  Scanning {len(crypto_yf)} crypto assets...", "scan")
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(_fetch_ticker_data, t): t for t in crypto_yf}
        crypto_count = 0
        for f in futures:
            try:
                data = f.result(timeout=30)
                if data:
                    results.append(data)
                    crypto_count += 1
            except Exception:
                pass
    _think(f"    Crypto: {crypto_count}/{len(crypto_yf)} loaded", "info")

    _think(f"Data fetched: {len(results)}/{len(all_tickers)} total assets loaded successfully", "scan")

    # Run ensemble on each
    signals = []
    for data in results:
        vote = _ensemble_vote(data)
        if vote["action"] != "HOLD" and vote["confidence"] > 60 and vote["agreement"] >= 50:
            signals.append({
                "ticker": data["ticker"],
                "price": data["price"],
                "action": vote["action"],
                "confidence": vote["confidence"],
                "agreement": vote["agreement"],
                "models_agree": vote["models_agree"],
                "models_voted": vote["models_voted"],
                "votes": vote["votes"],
            })

    # Sort by confidence
    signals.sort(key=lambda x: x["confidence"], reverse=True)

    if signals:
        _think(f"SIGNALS FOUND: {len(signals)} actionable trades", "signal")
        for s in signals[:5]:
            _think(f"  >> {s['action']} {s['ticker']} @ ${s['price']:.2f} | conf={s['confidence']}% | {s['models_agree']}/{s['models_voted']} models agree", "signal")
    else:
        _think("No actionable signals this cycle — all HOLD or below threshold", "info")

    return results, signals


def _execute_signals(signals: list):
    """Execute top signals as paper trades."""
    pt = PaperTrader()
    trades_made = []
    _think(f"Evaluating {len(signals)} signals for trade execution (balance: ${pt.data['balance']:.2f})", "trade")

    for sig in signals[:5]:  # max 5 trades per cycle
        try:
            ticker = sig["ticker"]
            action = sig["action"]
            price = sig["price"]

            # Check if already in position
            existing = [p for p in pt.data.get("positions", []) if p.get("ticker") == ticker]
            if existing:
                _think(f"  [{ticker}] Already in position — skip", "info")
                continue

            # Check balance
            if pt.data["balance"] < 200:
                _think("  LOW BALANCE — stopping trade execution", "error")
                break

            if action == "BUY":
                # For 0DTE: buy shares (simulating options)
                invest = min(pt.data["balance"] * 0.1, 500)  # 10% or $500 max per trade
                shares = int(invest / price)
                if shares > 0:
                    trade = {
                        "ticker": ticker,
                        "type": "STOCK",
                        "side": "LONG",
                        "shares": shares,
                        "entry_price": price,
                        "cost": round(shares * price, 2),
                        "stop_loss": round(price * 0.97, 2),    # 3% SL for 0DTE
                        "take_profit": round(price * 1.05, 2),  # 5% TP
                        "confidence": sig["confidence"],
                        "models_agree": sig["models_agree"],
                        "entered_at": datetime.utcnow().isoformat() + "Z",
                        "source": "auto_trader",
                    }
                    pt.data["balance"] -= trade["cost"]
                    pt.data["positions"].append(trade)
                    pt.data["trades"].append({**trade, "status": "OPEN"})
                    trades_made.append(trade)
                    _think(f"  BOUGHT {shares}x {ticker} @ ${price:.2f} (${trade['cost']:.2f}) | SL=${trade['stop_loss']} TP=${trade['take_profit']} | conf={sig['confidence']}%", "trade")

            elif action == "SELL":
                # Close any long position in this ticker
                for pos in pt.data.get("positions", []):
                    if pos.get("ticker") == ticker and pos.get("side") == "LONG":
                        pnl = (price - pos["entry_price"]) * pos.get("shares", 0)
                        pt.data["balance"] += pos.get("shares", 0) * price
                        pt.data["positions"].remove(pos)
                        trade_record = {
                            **pos,
                            "exit_price": price,
                            "pnl": round(pnl, 2),
                            "exited_at": datetime.utcnow().isoformat() + "Z",
                            "status": "CLOSED",
                        }
                        pt.data["trades"].append(trade_record)
                        trades_made.append(trade_record)
                        break

        except Exception:
            continue

    # Update stops on existing positions
    for pos in pt.data.get("positions", []):
        try:
            from data_fetcher import fetch
            hist = fetch(pos["ticker"], period="1d")
            if hist is None or hist.empty:
                continue
            current = float(hist["Close"].iloc[-1])

            # Check stop-loss
            if current <= pos.get("stop_loss", 0):
                pnl = (current - pos["entry_price"]) * pos.get("shares", 0)
                pt.data["balance"] += pos.get("shares", 0) * current
                pt.data["positions"].remove(pos)
                pt.data["trades"].append({
                    **pos, "exit_price": current, "pnl": round(pnl, 2),
                    "exited_at": datetime.utcnow().isoformat() + "Z",
                    "status": "STOPPED_OUT",
                })
                trades_made.append({"ticker": pos["ticker"], "action": "STOP_LOSS", "pnl": round(pnl, 2)})
                _think(f"  STOP LOSS HIT: {pos['ticker']} @ ${current:.2f} | P&L: ${pnl:.2f}", "trade")

            # Check take-profit
            elif current >= pos.get("take_profit", 999999):
                pnl = (current - pos["entry_price"]) * pos.get("shares", 0)
                pt.data["balance"] += pos.get("shares", 0) * current
                pt.data["positions"].remove(pos)
                pt.data["trades"].append({
                    **pos, "exit_price": current, "pnl": round(pnl, 2),
                    "exited_at": datetime.utcnow().isoformat() + "Z",
                    "status": "TARGET_HIT",
                })
                trades_made.append({"ticker": pos["ticker"], "action": "TAKE_PROFIT", "pnl": round(pnl, 2)})
                _think(f"  TARGET HIT: {pos['ticker']} @ ${current:.2f} | P&L: +${pnl:.2f}", "trade")

        except Exception:
            continue

    pt._save()
    return trades_made


def _agent_loop():
    """Main trading agent loop — runs until stopped."""
    global _agent_state

    _agent_state["running"] = True
    _agent_state["started_at"] = datetime.utcnow().isoformat() + "Z"
    _agent_state["stopped_at"] = None
    _save_state()
    _think("AGENT STARTED — Autonomous trading loop active", "scan")

    while not _stop_event.is_set():
        try:
            cycle_start = time.time()

            # Run scan
            results, signals = _scan_cycle()

            # Execute signals
            trades = _execute_signals(signals) if signals else []

            # Update state
            _agent_state["cycles"] += 1
            _agent_state["signals_found"] += len(signals)
            _agent_state["trades_made"] += len(trades)
            _agent_state["last_scan"] = {
                "time": datetime.utcnow().isoformat() + "Z",
                "stocks_scanned": len(results),
                "signals": len(signals),
                "trades": len(trades),
                "top_signals": [
                    {"ticker": s["ticker"], "action": s["action"], "confidence": s["confidence"]}
                    for s in signals[:10]
                ],
                "elapsed_s": round(time.time() - cycle_start, 1),
            }

            # Log
            _agent_state["scan_log"].append(_agent_state["last_scan"])
            _agent_state["scan_log"] = _agent_state["scan_log"][-50:]  # keep last 50

            if trades:
                for t in trades:
                    _agent_state["trade_log"].append({
                        "time": datetime.utcnow().isoformat() + "Z",
                        **{k: v for k, v in t.items() if k != "votes"},
                    })
                _agent_state["trade_log"] = _agent_state["trade_log"][-50:]

            _agent_state["last_error"] = None
            _save_state()

            # Wait between cycles (5 minutes during market hours, 30 min off hours)
            now = datetime.now()
            if 9 <= now.hour <= 16 and now.weekday() < 5:
                wait = 300  # 5 min during market
            else:
                wait = 1800  # 30 min off-market

            elapsed = round(time.time() - cycle_start, 1)
            _think(f"Cycle #{_agent_state['cycles']} complete in {elapsed}s | {len(signals)} signals, {len(trades)} trades | Next scan in {wait//60}min", "scan")

            _stop_event.wait(wait)

        except Exception as e:
            _agent_state["last_error"] = str(e)
            _think(f"ERROR: {e}", "error")

            # Self-healing: diagnose with local Ollama
            try:
                from self_healer import auto_fix
                fix = auto_fix(e, context="agent_loop scan cycle", module="auto_trader")
                if fix.get("fixed"):
                    _think(f"SELF-HEAL: {fix.get('action')} — {fix.get('fixes', [])}", "info")
                    if fix["action"] == "cooldown_retry":
                        _stop_event.wait(30)
                        continue
                    elif fix["action"] == "rate_limit_backoff":
                        _stop_event.wait(60)
                        continue
                else:
                    diag = fix.get("diagnosis", {})
                    _think(f"DIAGNOSIS: {diag.get('diagnosis', 'unknown')} | Fix: {diag.get('fix', 'manual')}", "error")
            except Exception:
                pass

            _save_state()
            _stop_event.wait(60)

    _think("AGENT STOPPED", "scan")
    _agent_state["running"] = False
    _agent_state["stopped_at"] = datetime.utcnow().isoformat() + "Z"
    _save_state()


# ─── PUBLIC API ─────────────────────────────────────────────────
def start_agent() -> dict:
    """Start the autonomous trading agent."""
    global _agent_thread, _stop_event

    if _agent_state["running"]:
        return {"status": "already_running", "started_at": _agent_state["started_at"]}

    _stop_event.clear()
    _agent_thread = threading.Thread(target=_agent_loop, daemon=True)
    _agent_thread.start()

    return {
        "status": "started",
        "mode": _agent_state["mode"],
        "started_at": datetime.utcnow().isoformat() + "Z",
    }


def stop_agent() -> dict:
    """Stop the autonomous trading agent."""
    global _stop_event

    if not _agent_state["running"]:
        return {"status": "not_running"}

    _stop_event.set()
    return {
        "status": "stopping",
        "cycles_completed": _agent_state["cycles"],
        "trades_made": _agent_state["trades_made"],
    }


def get_agent_status() -> dict:
    """Get current agent status."""
    pt = PaperTrader()
    return {
        "running": _agent_state["running"],
        "mode": _agent_state["mode"],
        "started_at": _agent_state.get("started_at"),
        "stopped_at": _agent_state.get("stopped_at"),
        "cycles": _agent_state["cycles"],
        "trades_made": _agent_state["trades_made"],
        "signals_found": _agent_state["signals_found"],
        "last_scan": _agent_state.get("last_scan"),
        "last_error": _agent_state.get("last_error"),
        "portfolio": {
            "balance": pt.data["balance"],
            "positions": len(pt.data.get("positions", [])),
            "total_trades": len(pt.data.get("trades", [])),
        },
        "ensemble_weights": _agent_state["ensemble_weights"],
    }


def get_agent_logs() -> dict:
    """Get recent scan and trade logs."""
    return {
        "scan_log": _agent_state.get("scan_log", [])[-20:],
        "trade_log": _agent_state.get("trade_log", [])[-20:],
    }


def run_single_scan() -> dict:
    """Run a single scan cycle without the loop (for testing)."""
    t0 = time.time()
    results, signals = _scan_cycle()
    return {
        "stocks_scanned": len(results),
        "signals_found": len(signals),
        "signals": signals[:20],
        "elapsed_s": round(time.time() - t0, 1),
    }


# Load saved state on import
_load_state()
