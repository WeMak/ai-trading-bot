"""
Backtesting Engine — Full Trade Simulation with 5:1 R/R
=========================================================
Runs ensemble model predictions against historical data,
simulates realistic trades with strict risk management,
and produces comprehensive performance analytics.

Features:
  - 5:1 Risk/Reward targeting (configurable)
  - Random 3-month out-of-sample walk-forward tests
  - Per-stock and aggregate PNL, Sharpe, win rate, drawdown
  - Full trade log with entry/exit prices, reasons, R-multiples
  - Monte Carlo stress testing
  - Equity curve generation

Usage:
    from backtester import Backtester
    bt = Backtester()
    results = bt.run_full_backtest(tickers, period="2y")
"""

import warnings; warnings.filterwarnings("ignore")
import os, json, time, random, math
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from gpu_compute import (
    rsi_gpu as _rsi, macd_gpu as _macd, bollinger_gpu as _bollinger,
    sma_gpu as _sma, atr_gpu as _atr, compute_all_indicators,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "backtest_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── CONFIG ──────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "initial_capital": 100_000,
    "risk_per_trade_pct": 1.0,       # risk 1% of capital per trade
    "reward_ratio": 5.0,             # 5:1 R/R
    "max_positions": 10,             # max simultaneous positions
    "commission_pct": 0.001,         # 0.1% round trip
    "slippage_pct": 0.0005,          # 0.05% slippage
    "min_confidence": 35,            # min ensemble confidence to trade
    "min_agreement_pct": 50,         # min % of models agreeing
    "atr_sl_multiple": 1.5,          # stop-loss = 1.5 × ATR
    "use_trailing_stop": True,       # enable trailing stops
    "trailing_stop_atr": 2.0,        # trailing stop = 2 × ATR
    "max_hold_bars": 60,             # max days to hold
    "cooldown_bars": 2,              # bars between trades per ticker
    "oos_months": 3,                 # out-of-sample test window
    "n_random_tests": 5,             # number of random OOS windows
    "monte_carlo_runs": 500,         # MC simulations
}


# ─── BACKTESTER ──────────────────────────────────────────────────
class Backtester:
    """Full backtesting engine with ensemble predictions."""

    def __init__(self, config: dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self._log = []
        self._progress_cb = None

    def set_progress_callback(self, cb):
        """Set a callback for progress updates: cb(pct, message)."""
        self._progress_cb = cb

    def _progress(self, pct: float, msg: str):
        if self._progress_cb:
            self._progress_cb(pct, msg)

    # ── Data Preparation ─────────────────────────────────────────

    def _fetch_bars(self, ticker: str, period: str = "2y") -> dict:
        """Fetch and compute indicators for a ticker."""
        try:
            from data_fetcher import fetch
            hist = fetch(ticker, period=period)
            if hist is None or hist.empty or len(hist) < 60:
                return None

            close = hist["Close"].values.astype(float)
            high = hist["High"].values.astype(float)
            low = hist["Low"].values.astype(float)
            volume = hist["Volume"].values.astype(float)
            dates = hist.index.strftime("%Y-%m-%d").tolist()

            # Compute all indicators
            rsi_arr = _rsi(close)
            macd_l, macd_s, macd_h = _macd(close)
            bb_u, bb_m, bb_l = _bollinger(close)
            sma50_arr = _sma(close, 50)
            sma200_arr = _sma(close, 200)
            atr_arr = _atr(high, low, close)
            vol_avg = np.convolve(volume, np.ones(20) / 20, mode='same')

            def _s(arr, idx):
                v = arr[idx] if idx < len(arr) else 0.0
                return float(v) if not np.isnan(v) else 0.0

            bars = []
            for i in range(50, len(close)):  # start at 50 for SMA50
                price = float(close[i])
                if price <= 0:
                    continue
                bars.append({
                    "date": dates[i],
                    "price": price,
                    "high": float(high[i]),
                    "low": float(low[i]),
                    "close": price,
                    "volume": float(volume[i]),
                    "indicators": {
                        "price": price,
                        "rsi": _s(rsi_arr, i),
                        "macd": _s(macd_l, i),
                        "macd_signal": _s(macd_s, i),
                        "macd_hist": _s(macd_h, i),
                        "bb_upper": _s(bb_u, i),
                        "bb_middle": _s(bb_m, i),
                        "bb_lower": _s(bb_l, i),
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
                })

            return {"ticker": ticker, "bars": bars, "total_bars": len(bars)}
        except Exception as e:
            return None

    # ── Ensemble Prediction ──────────────────────────────────────

    def _get_ensemble_prediction(self, bar: dict) -> dict:
        """Run all available models on a single bar and return consensus."""
        votes = []
        indicators = bar["indicators"]
        price_history = bar.get("price_history", {})

        # 1. Technical analysis
        try:
            from trading_agent import _score
            tech_signal, tech_score, _ = _score(indicators)
            votes.append({"model": "Technical", "action": tech_signal, "confidence": tech_score, "weight": 1.0})
        except Exception:
            pass

        # 2. NEAT
        try:
            from neat_trader import get_neat_trader
            neat = get_neat_trader()
            if neat.best_net is not None:
                pred = neat.predict(indicators, price_history)
                votes.append({"model": "NEAT", "action": pred["action"], "confidence": pred["confidence"], "weight": 1.2})
        except Exception:
            pass

        # 3. PyTorch
        try:
            from torch_predictor import get_torch_predictor
            pytorch = get_torch_predictor()
            if pytorch.epoch > 0:
                pred = pytorch.predict(indicators, price_history=price_history)
                votes.append({"model": "PyTorch", "action": pred["action"], "confidence": pred["confidence"], "weight": 1.3})
        except Exception:
            pass

        # 4. EvoStrategy
        try:
            from evo_optimizer import get_evo_optimizer
            evo = get_evo_optimizer()
            if evo.best_params is not None:
                pred = evo.predict(bar)
                votes.append({"model": "Evo", "action": pred["action"], "confidence": pred["confidence"], "weight": 1.0})
        except Exception:
            pass

        if not votes:
            return {"action": "HOLD", "confidence": 0, "agreement": 0, "models_voted": 0, "votes": []}

        # Aggregate
        total_w = sum(v["weight"] for v in votes)
        buy_s = sum(v["confidence"] * v["weight"] for v in votes if v["action"] == "BUY")
        sell_s = sum(v["confidence"] * v["weight"] for v in votes if v["action"] == "SELL")
        hold_s = sum(v["confidence"] * v["weight"] for v in votes if v["action"] == "HOLD")

        scores = {"BUY": buy_s, "SELL": sell_s, "HOLD": hold_s}
        best_action = max(scores, key=scores.get)
        best_score = scores[best_action] / (total_w + 1e-10)
        models_agree = sum(1 for v in votes if v["action"] == best_action)
        agreement = (models_agree / len(votes)) * 100

        return {
            "action": best_action,
            "confidence": round(best_score, 1),
            "agreement": round(agreement, 0),
            "models_voted": len(votes),
            "models_agree": models_agree,
            "votes": votes,
        }

    # ── Single Ticker Backtest ───────────────────────────────────

    def backtest_ticker(self, ticker: str, bars: list, config: dict = None) -> dict:
        """
        Run a full backtest on a single ticker's bar data.
        Returns comprehensive trade log and performance metrics.
        """
        cfg = {**self.config, **(config or {})}
        capital = float(cfg["initial_capital"])
        balance = capital
        equity_curve = [capital]
        equity_dates = [bars[0]["date"] if bars else ""]
        positions = []  # open positions
        trades = []     # completed trades
        peak_equity = capital
        max_drawdown = 0
        max_drawdown_pct = 0
        cooldown = 0
        daily_returns = []
        bar_idx = 0

        for bar in bars:
            bar_idx += 1
            price = bar["price"]
            atr = bar["indicators"].get("atr", price * 0.02)
            if atr <= 0:
                atr = price * 0.02
            date = bar["date"]

            # Update open positions
            closed_this_bar = []
            for pos in positions:
                pos["bars_held"] += 1
                pos["current_price"] = price
                pnl = (price - pos["entry_price"]) * pos["direction_mult"] * pos["shares"]
                pos["unrealized_pnl"] = round(pnl, 2)
                pos["unrealized_pct"] = round((price / pos["entry_price"] - 1) * pos["direction_mult"] * 100, 2)

                # Trailing stop update
                if cfg["use_trailing_stop"] and pnl > 0:
                    if pos["direction_mult"] == 1:  # long
                        new_trail = price - cfg["trailing_stop_atr"] * atr
                        pos["trailing_sl"] = max(pos.get("trailing_sl", pos["stop_loss"]), new_trail)
                    else:  # short
                        new_trail = price + cfg["trailing_stop_atr"] * atr
                        pos["trailing_sl"] = min(pos.get("trailing_sl", pos["stop_loss"]), new_trail)

                effective_sl = pos.get("trailing_sl", pos["stop_loss"])

                # Check stop-loss
                hit_sl = False
                if pos["direction_mult"] == 1:
                    hit_sl = price <= effective_sl
                else:
                    hit_sl = price >= effective_sl

                # Check take-profit
                hit_tp = False
                if pos["direction_mult"] == 1:
                    hit_tp = price >= pos["take_profit"]
                else:
                    hit_tp = price <= pos["take_profit"]

                # Check max hold
                hit_max_hold = pos["bars_held"] >= cfg["max_hold_bars"]

                if hit_sl or hit_tp or hit_max_hold:
                    exit_reason = "STOP_LOSS" if hit_sl else ("TAKE_PROFIT" if hit_tp else "MAX_HOLD")
                    exit_price = price

                    # Apply slippage
                    slip = exit_price * cfg["slippage_pct"]
                    if pos["direction_mult"] == 1:
                        exit_price -= slip
                    else:
                        exit_price += slip

                    realized_pnl = (exit_price - pos["entry_price"]) * pos["direction_mult"] * pos["shares"]
                    commission = (pos["entry_price"] + exit_price) * pos["shares"] * cfg["commission_pct"]
                    net_pnl = realized_pnl - commission

                    # R-multiple (how many R's of risk was the trade)
                    risk_per_share = abs(pos["entry_price"] - pos["stop_loss"])
                    r_multiple = (realized_pnl / pos["shares"]) / (risk_per_share + 1e-10)

                    trade = {
                        "ticker": ticker,
                        "direction": "LONG" if pos["direction_mult"] == 1 else "SHORT",
                        "entry_date": pos["entry_date"],
                        "exit_date": date,
                        "entry_price": round(pos["entry_price"], 4),
                        "exit_price": round(exit_price, 4),
                        "shares": pos["shares"],
                        "stop_loss": round(pos["stop_loss"], 4),
                        "take_profit": round(pos["take_profit"], 4),
                        "bars_held": pos["bars_held"],
                        "gross_pnl": round(realized_pnl, 2),
                        "commission": round(commission, 2),
                        "net_pnl": round(net_pnl, 2),
                        "pnl_pct": round(net_pnl / (pos["position_cost"] + 1e-10) * 100, 2),
                        "r_multiple": round(r_multiple, 2),
                        "exit_reason": exit_reason,
                        "ensemble_confidence": pos["confidence"],
                        "ensemble_agreement": pos["agreement"],
                        "models_voted": pos["models_voted"],
                    }
                    trades.append(trade)
                    balance += net_pnl
                    closed_this_bar.append(pos)

            # Remove closed positions
            for cp in closed_this_bar:
                positions.remove(cp)

            # Cooldown
            if cooldown > 0:
                cooldown -= 1

            # Equity tracking
            open_pnl = sum(p["unrealized_pnl"] for p in positions)
            equity = balance + open_pnl
            equity_curve.append(round(equity, 2))
            equity_dates.append(date)

            # Drawdown
            if equity > peak_equity:
                peak_equity = equity
            dd = (peak_equity - equity) / (peak_equity + 1e-10) * 100
            if dd > max_drawdown_pct:
                max_drawdown_pct = dd
                max_drawdown = peak_equity - equity

            # Daily return
            if len(equity_curve) >= 2:
                prev = equity_curve[-2]
                daily_returns.append((equity - prev) / (prev + 1e-10))

            # Signal check — should we open a new position?
            if cooldown > 0 or len(positions) >= cfg["max_positions"]:
                continue

            # Already have a position in this ticker
            if any(p.get("ticker") == ticker for p in positions):
                continue

            prediction = self._get_ensemble_prediction(bar)
            action = prediction["action"]
            confidence = prediction["confidence"]
            agreement = prediction["agreement"]

            if action == "HOLD":
                continue
            if confidence < cfg["min_confidence"]:
                continue
            if agreement < cfg["min_agreement_pct"]:
                continue

            # Calculate position size using ATR-based stop
            direction_mult = 1 if action == "BUY" else -1
            sl_distance = cfg["atr_sl_multiple"] * atr
            risk_amount = balance * (cfg["risk_per_trade_pct"] / 100)
            shares = int(risk_amount / (sl_distance + 1e-10))
            if shares <= 0:
                continue

            # Entry with slippage
            entry_price = price + (cfg["slippage_pct"] * price * direction_mult)

            # 5:1 R/R stops
            if direction_mult == 1:
                stop_loss = entry_price - sl_distance
                take_profit = entry_price + sl_distance * cfg["reward_ratio"]
            else:
                stop_loss = entry_price + sl_distance
                take_profit = entry_price - sl_distance * cfg["reward_ratio"]

            position_cost = shares * entry_price

            # Don't risk more than we have
            if position_cost > balance * 0.95:
                shares = max(1, int(balance * 0.95 / entry_price))
                position_cost = shares * entry_price

            positions.append({
                "ticker": ticker,
                "entry_date": date,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "trailing_sl": stop_loss,
                "shares": shares,
                "direction_mult": direction_mult,
                "position_cost": position_cost,
                "bars_held": 0,
                "current_price": price,
                "unrealized_pnl": 0,
                "unrealized_pct": 0,
                "confidence": confidence,
                "agreement": agreement,
                "models_voted": prediction["models_voted"],
            })
            cooldown = cfg["cooldown_bars"]

        # Close remaining positions at last price
        for pos in positions:
            price = pos["current_price"]
            realized_pnl = (price - pos["entry_price"]) * pos["direction_mult"] * pos["shares"]
            commission = (pos["entry_price"] + price) * pos["shares"] * cfg["commission_pct"]
            net_pnl = realized_pnl - commission
            risk_per_share = abs(pos["entry_price"] - pos["stop_loss"])
            r_multiple = (realized_pnl / pos["shares"]) / (risk_per_share + 1e-10)
            trades.append({
                "ticker": ticker,
                "direction": "LONG" if pos["direction_mult"] == 1 else "SHORT",
                "entry_date": pos["entry_date"],
                "exit_date": bars[-1]["date"] if bars else "",
                "entry_price": round(pos["entry_price"], 4),
                "exit_price": round(price, 4),
                "shares": pos["shares"],
                "stop_loss": round(pos["stop_loss"], 4),
                "take_profit": round(pos["take_profit"], 4),
                "bars_held": pos["bars_held"],
                "gross_pnl": round(realized_pnl, 2),
                "commission": round(commission, 2),
                "net_pnl": round(net_pnl, 2),
                "pnl_pct": round(net_pnl / (pos["position_cost"] + 1e-10) * 100, 2),
                "r_multiple": round(r_multiple, 2),
                "exit_reason": "END_OF_DATA",
                "ensemble_confidence": pos["confidence"],
                "ensemble_agreement": pos["agreement"],
                "models_voted": pos["models_voted"],
            })
            balance += net_pnl

        # ── Compute Performance Metrics ──────────────────────────
        return self._compute_metrics(ticker, trades, equity_curve, equity_dates,
                                     daily_returns, capital, balance, max_drawdown_pct, bars)

    def _compute_metrics(self, ticker, trades, equity_curve, equity_dates,
                         daily_returns, capital, final_balance, max_drawdown_pct, bars) -> dict:
        """Compute comprehensive performance metrics."""
        total_pnl = final_balance - capital
        total_return_pct = (total_pnl / capital) * 100
        n_trades = len(trades)

        if n_trades == 0:
            return {
                "ticker": ticker,
                "total_trades": 0,
                "total_pnl": 0,
                "total_return_pct": 0,
                "sharpe": 0,
                "win_rate": 0,
                "max_drawdown_pct": 0,
                "trades": [],
                "equity_curve": equity_curve[-50:],  # last 50 points
                "equity_dates": equity_dates[-50:],
                "summary": "No trades triggered — models below confidence threshold",
            }

        winners = [t for t in trades if t["net_pnl"] > 0]
        losers = [t for t in trades if t["net_pnl"] <= 0]
        win_rate = len(winners) / n_trades * 100

        avg_win = np.mean([t["net_pnl"] for t in winners]) if winners else 0
        avg_loss = np.mean([abs(t["net_pnl"]) for t in losers]) if losers else 0
        profit_factor = (sum(t["net_pnl"] for t in winners) / (sum(abs(t["net_pnl"]) for t in losers) + 1e-10)) if losers else float('inf')

        avg_r = np.mean([t["r_multiple"] for t in trades])
        max_r = max(t["r_multiple"] for t in trades)
        min_r = min(t["r_multiple"] for t in trades)

        avg_bars_held = np.mean([t["bars_held"] for t in trades])
        avg_bars_winners = np.mean([t["bars_held"] for t in winners]) if winners else 0
        avg_bars_losers = np.mean([t["bars_held"] for t in losers]) if losers else 0

        # Sharpe ratio (annualized)
        if daily_returns and len(daily_returns) > 1:
            dr = np.array(daily_returns)
            sharpe = (np.mean(dr) / (np.std(dr) + 1e-10)) * np.sqrt(252)
        else:
            sharpe = 0

        # Sortino ratio (downside deviation only)
        if daily_returns and len(daily_returns) > 1:
            dr = np.array(daily_returns)
            downside = dr[dr < 0]
            sortino = (np.mean(dr) / (np.std(downside) + 1e-10)) * np.sqrt(252) if len(downside) > 0 else sharpe
        else:
            sortino = 0

        # Calmar ratio
        calmar = (total_return_pct / (max_drawdown_pct + 1e-10))

        # Consecutive wins/losses
        streak = 0
        max_win_streak = 0
        max_lose_streak = 0
        for t in trades:
            if t["net_pnl"] > 0:
                streak = max(0, streak) + 1
                max_win_streak = max(max_win_streak, streak)
            else:
                streak = min(0, streak) - 1
                max_lose_streak = max(max_lose_streak, abs(streak))

        # Exit reason breakdown
        exit_reasons = {}
        for t in trades:
            r = t["exit_reason"]
            exit_reasons[r] = exit_reasons.get(r, 0) + 1

        # Monthly returns
        monthly = {}
        for t in trades:
            m = t["exit_date"][:7]  # YYYY-MM
            monthly[m] = monthly.get(m, 0) + t["net_pnl"]

        # Downsample equity curve for dashboard (max 200 points)
        eq_len = len(equity_curve)
        if eq_len > 200:
            step = eq_len // 200
            eq_sampled = equity_curve[::step]
            dt_sampled = equity_dates[::step]
        else:
            eq_sampled = equity_curve
            dt_sampled = equity_dates

        return {
            "ticker": ticker,
            "period": f"{bars[0]['date']} to {bars[-1]['date']}" if bars else "",
            "total_bars": len(bars),
            "total_trades": n_trades,
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_return_pct, 2),
            "initial_capital": capital,
            "final_capital": round(final_balance, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "largest_win": round(max(t["net_pnl"] for t in trades), 2) if trades else 0,
            "largest_loss": round(min(t["net_pnl"] for t in trades), 2) if trades else 0,
            "profit_factor": round(profit_factor, 2),
            "avg_r_multiple": round(avg_r, 2),
            "best_r": round(max_r, 2),
            "worst_r": round(min_r, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "calmar_ratio": round(calmar, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "avg_bars_held": round(avg_bars_held, 1),
            "avg_bars_winners": round(avg_bars_winners, 1),
            "avg_bars_losers": round(avg_bars_losers, 1),
            "max_win_streak": max_win_streak,
            "max_lose_streak": max_lose_streak,
            "exit_reasons": exit_reasons,
            "monthly_pnl": monthly,
            "trades": trades,
            "equity_curve": eq_sampled,
            "equity_dates": dt_sampled,
        }

    # ── Random Out-of-Sample Testing ─────────────────────────────

    def random_oos_test(self, ticker: str, bars: list, n_tests: int = None) -> list:
        """
        Run N random 3-month out-of-sample backtests.
        Picks random start dates and tests on unseen data.
        """
        n_tests = n_tests or self.config["n_random_tests"]
        oos_bars = int(self.config["oos_months"] * 21)  # ~21 trading days/month
        results = []

        if len(bars) < oos_bars + 30:
            return [{"error": "Not enough bars for OOS test", "ticker": ticker}]

        # Pick random non-overlapping windows
        max_start = len(bars) - oos_bars
        starts = sorted(random.sample(range(30, max_start), min(n_tests, max_start - 30)))

        for i, start in enumerate(starts):
            window = bars[start:start + oos_bars]
            result = self.backtest_ticker(ticker, window)
            result["oos_window"] = i + 1
            result["oos_start"] = window[0]["date"]
            result["oos_end"] = window[-1]["date"]
            results.append(result)

        return results

    # ── Full Multi-Stock Backtest ─────────────────────────────────

    def run_full_backtest(self, tickers: list, period: str = "2y",
                          workers: int = 4, run_oos: bool = True) -> dict:
        """
        Master backtest: fetch data, backtest each ticker, run OOS tests.
        Returns aggregate and per-stock results.
        """
        t0 = time.time()
        n = len(tickers)
        self._progress(0, f"Starting backtest on {n} stocks...")

        # Phase 1: Fetch all data
        ticker_data = {}
        done = 0
        with ThreadPoolExecutor(max_workers=min(workers, 8)) as pool:
            futures = {pool.submit(self._fetch_bars, t, period): t for t in tickers}
            for f in as_completed(futures):
                t = futures[f]
                done += 1
                try:
                    result = f.result()
                    if result and result.get("bars"):
                        ticker_data[t] = result["bars"]
                except Exception:
                    pass
                if done % 25 == 0 or done == n:
                    self._progress(done / n * 30, f"Downloaded {done}/{n} stocks")

        self._progress(30, f"Data ready: {len(ticker_data)}/{n} stocks with data")

        # Phase 2: Backtest each ticker
        per_stock = {}
        all_trades = []
        done = 0
        total = len(ticker_data)

        for ticker, bars in ticker_data.items():
            done += 1
            result = self.backtest_ticker(ticker, bars)
            per_stock[ticker] = result
            all_trades.extend(result.get("trades", []))
            if done % 10 == 0 or done == total:
                self._progress(30 + done / total * 40, f"Backtested {done}/{total}: {ticker}")

        # Phase 3: Random OOS tests
        oos_results = {}
        if run_oos:
            done = 0
            for ticker, bars in ticker_data.items():
                done += 1
                oos = self.random_oos_test(ticker, bars)
                oos_results[ticker] = oos
                if done % 20 == 0:
                    self._progress(70 + done / total * 20, f"OOS testing {done}/{total}")

        self._progress(90, "Computing aggregate metrics...")

        # Phase 4: Aggregate metrics
        aggregate = self._aggregate_results(per_stock, all_trades, oos_results, t0)

        # Phase 5: Monte Carlo
        self._progress(95, "Running Monte Carlo simulation...")
        mc = self._monte_carlo(all_trades) if all_trades else {}
        aggregate["monte_carlo"] = mc

        aggregate["elapsed_s"] = round(time.time() - t0, 1)
        self._progress(100, f"Complete: {len(all_trades)} trades across {total} stocks")

        # Save results
        self._save_results(aggregate)

        return aggregate

    def _aggregate_results(self, per_stock, all_trades, oos_results, t0) -> dict:
        """Compute aggregate performance across all stocks."""
        n_stocks = len(per_stock)
        total_trades = len(all_trades)

        if total_trades == 0:
            return {
                "n_stocks": n_stocks, "total_trades": 0,
                "per_stock": per_stock, "oos_results": oos_results,
                "summary": "No trades triggered across any stock"
            }

        winners = [t for t in all_trades if t["net_pnl"] > 0]
        losers = [t for t in all_trades if t["net_pnl"] <= 0]
        total_pnl = sum(t["net_pnl"] for t in all_trades)
        win_rate = len(winners) / total_trades * 100

        avg_win = np.mean([t["net_pnl"] for t in winners]) if winners else 0
        avg_loss = np.mean([abs(t["net_pnl"]) for t in losers]) if losers else 0
        profit_factor = sum(t["net_pnl"] for t in winners) / (sum(abs(t["net_pnl"]) for t in losers) + 1e-10)

        r_multiples = [t["r_multiple"] for t in all_trades]
        avg_r = np.mean(r_multiples)
        expectancy = np.mean([t["net_pnl"] for t in all_trades])

        # Per-stock ranking
        ranked = sorted(per_stock.items(), key=lambda x: x[1].get("total_return_pct", 0), reverse=True)
        top_5 = [(k, v.get("total_return_pct", 0), v.get("total_trades", 0), v.get("sharpe_ratio", 0)) for k, v in ranked[:5]]
        bottom_5 = [(k, v.get("total_return_pct", 0), v.get("total_trades", 0), v.get("sharpe_ratio", 0)) for k, v in ranked[-5:]]

        # Profitable stocks
        profitable = sum(1 for k, v in per_stock.items() if v.get("total_pnl", 0) > 0)

        # OOS summary
        oos_summary = {}
        if oos_results:
            all_oos_returns = []
            all_oos_sharpe = []
            for ticker, tests in oos_results.items():
                for t in tests:
                    if isinstance(t, dict) and "total_return_pct" in t:
                        all_oos_returns.append(t["total_return_pct"])
                        all_oos_sharpe.append(t.get("sharpe_ratio", 0))
            if all_oos_returns:
                oos_summary = {
                    "n_tests": len(all_oos_returns),
                    "avg_return_pct": round(np.mean(all_oos_returns), 2),
                    "median_return_pct": round(np.median(all_oos_returns), 2),
                    "win_rate": round(sum(1 for r in all_oos_returns if r > 0) / len(all_oos_returns) * 100, 1),
                    "avg_sharpe": round(np.mean(all_oos_sharpe), 2),
                    "worst_return": round(min(all_oos_returns), 2),
                    "best_return": round(max(all_oos_returns), 2),
                }

        # Exit reason totals
        exit_totals = {}
        for t in all_trades:
            r = t["exit_reason"]
            exit_totals[r] = exit_totals.get(r, 0) + 1

        # Strip individual trades from per_stock to save space in top-level
        per_stock_summary = {}
        for k, v in per_stock.items():
            ps = {kk: vv for kk, vv in v.items() if kk not in ("trades", "equity_curve", "equity_dates")}
            ps["n_trades"] = len(v.get("trades", []))
            per_stock_summary[k] = ps

        return {
            "n_stocks_tested": n_stocks,
            "n_stocks_with_trades": sum(1 for v in per_stock.values() if v.get("total_trades", 0) > 0),
            "n_stocks_profitable": profitable,
            "total_trades": total_trades,
            "total_winners": len(winners),
            "total_losers": len(losers),
            "win_rate": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl_per_trade": round(expectancy, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "avg_r_multiple": round(avg_r, 2),
            "r_multiple_std": round(np.std(r_multiples), 2) if r_multiples else 0,
            "expectancy_per_trade": round(expectancy, 2),
            "exit_reasons": exit_totals,
            "top_5_stocks": [{"ticker": t[0], "return_pct": t[1], "trades": t[2], "sharpe": t[3]} for t in top_5],
            "bottom_5_stocks": [{"ticker": t[0], "return_pct": t[1], "trades": t[2], "sharpe": t[3]} for t in bottom_5],
            "oos_summary": oos_summary,
            "per_stock": per_stock_summary,
            "per_stock_full": per_stock,
            "oos_results": oos_results,
            "config": self.config,
        }

    # ── Monte Carlo Simulation ───────────────────────────────────

    def _monte_carlo(self, trades: list) -> dict:
        """
        Shuffle trade order N times to estimate confidence intervals
        for final equity, max drawdown, etc.
        """
        n_runs = self.config["monte_carlo_runs"]
        capital = self.config["initial_capital"]
        final_equities = []
        max_drawdowns = []
        pnls = [t["net_pnl"] for t in trades]

        for _ in range(n_runs):
            shuffled = pnls.copy()
            random.shuffle(shuffled)
            equity = capital
            peak = capital
            max_dd = 0

            for pnl in shuffled:
                equity += pnl
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / (peak + 1e-10) * 100
                if dd > max_dd:
                    max_dd = dd

            final_equities.append(equity)
            max_drawdowns.append(max_dd)

        final_equities.sort()
        max_drawdowns.sort()
        n = len(final_equities)

        return {
            "runs": n_runs,
            "median_final_equity": round(final_equities[n // 2], 2),
            "p5_equity": round(final_equities[int(n * 0.05)], 2),
            "p25_equity": round(final_equities[int(n * 0.25)], 2),
            "p75_equity": round(final_equities[int(n * 0.75)], 2),
            "p95_equity": round(final_equities[int(n * 0.95)], 2),
            "worst_case_equity": round(final_equities[0], 2),
            "best_case_equity": round(final_equities[-1], 2),
            "median_max_dd": round(max_drawdowns[n // 2], 2),
            "p95_max_dd": round(max_drawdowns[int(n * 0.95)], 2),
            "ruin_probability": round(sum(1 for e in final_equities if e < capital * 0.5) / n * 100, 1),
        }

    # ── Save Results ─────────────────────────────────────────────

    def _save_results(self, results: dict):
        """Save backtest results to JSON file."""
        filename = f"backtest_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        path = os.path.join(RESULTS_DIR, filename)

        # Create a serializable version (strip equity curves from full results)
        save_data = {k: v for k, v in results.items() if k != "per_stock_full"}
        try:
            with open(path, "w") as f:
                json.dump(save_data, f, indent=2, default=str)
        except Exception:
            pass
        return path

    # ── Load Previous Results ────────────────────────────────────

    @staticmethod
    def list_results() -> list:
        """List all saved backtest results."""
        files = []
        for f in sorted(os.listdir(RESULTS_DIR), reverse=True):
            if f.endswith(".json"):
                path = os.path.join(RESULTS_DIR, f)
                try:
                    size = os.path.getsize(path)
                    files.append({"filename": f, "size_kb": round(size / 1024, 1), "path": path})
                except Exception:
                    pass
        return files

    @staticmethod
    def load_result(filename: str) -> dict:
        """Load a saved backtest result."""
        path = os.path.join(RESULTS_DIR, filename)
        with open(path) as f:
            return json.load(f)
