"""
Paper Trading Engine — $2,000 Starting Balance
OPTIONS: 3 contracts per trade, cheap IV only (<40% rank)
CRYPTO:  5% of portfolio per trade, LONG/SHORT
Risk:    Options SL -40% / TP +80% of premium
         Crypto  SL -7%  / TP +18% of position
"""

import json, os, time
from datetime import datetime
from typing import Optional
import yfinance as yf

ACCOUNT_FILE = os.path.join(os.path.dirname(__file__), "paper_account.json")
CONTRACTS    = 3          # always buy 3 contracts
SHARES       = 100        # 1 contract = 100 shares
MAX_PREMIUM  = 6.00       # max $/share — keeps each trade under $1,800
MIN_IV_RANK  = 0.0
MAX_IV_RANK  = 0.40       # only "cheap" options (IV rank ≤ 40%)

DEFAULTS = {
    "balance":         2000.0,
    "initial_balance": 2000.0,
    "positions":       [],
    "trades":          [],
    "last_updated":    None,
}

class PaperTrader:
    def __init__(self):
        self.data = self._load()

    # ── Persistence ───────────────────────────────────────────
    def _load(self):
        if os.path.exists(ACCOUNT_FILE):
            try:
                with open(ACCOUNT_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        d = dict(DEFAULTS)
        self._write(d); return d

    def _write(self, d=None):
        with open(ACCOUNT_FILE, "w") as f:
            json.dump(d or self.data, f, indent=2, default=str)

    def _save(self):
        self.data["last_updated"] = datetime.utcnow().isoformat() + "Z"
        self._write()

    # ── Price lookup ──────────────────────────────────────────
    def _price(self, ticker: str) -> float:
        try:
            h = yf.Ticker(ticker).history(period="1d")
            if not h.empty:
                return float(h["Close"].iloc[-1])
        except Exception:
            pass
        return 0.0

    # ── Portfolio value ───────────────────────────────────────
    def portfolio_value(self) -> float:
        """Cash + mark-to-market of all open positions."""
        val = self.data["balance"]
        for p in self.data["positions"]:
            px = p.get("current_price") or p["entry_price"]
            ep = p["entry_price"]
            if not ep or not px:
                val += p["value"]; continue

            if p["asset_type"] == "OPTIONS":
                # Simulate option P&L using underlying move × leverage factor
                underlying_move = (px - ep) / ep
                if p["direction"] == "CALL":
                    opt_pct = underlying_move * 2.5
                else:
                    opt_pct = -underlying_move * 2.5
                opt_pct = max(opt_pct, -0.95)  # max loss on option = -95%
                val += p["value"] * (1 + opt_pct)
            else:
                # Crypto: direct price change
                raw = (px - ep) / ep if p["direction"] == "LONG" else (ep - px) / ep
                val += p["value"] * (1 + raw)
        return val

    # ── Entry checks ──────────────────────────────────────────
    def _can_enter(self, ticker: str, asset_type: str,
                   est_premium: float, iv_rank: float) -> tuple[bool, str]:
        if len(self.data["positions"]) >= 5:
            return False, "max 5 positions open"
        if any(p["ticker"] == ticker for p in self.data["positions"]):
            return False, f"already in {ticker}"

        if asset_type == "OPTIONS":
            # Cheap-option filter
            if iv_rank > MAX_IV_RANK:
                return False, f"IV rank {iv_rank:.0%} too expensive (max {MAX_IV_RANK:.0%})"
            if est_premium > MAX_PREMIUM:
                return False, f"premium ${est_premium:.2f}/share too high (max ${MAX_PREMIUM})"
            total_cost = est_premium * SHARES * CONTRACTS
            if total_cost > self.data["balance"]:
                return False, f"insufficient cash (need ${total_cost:.0f})"
            if self.data["balance"] < self.data["initial_balance"] * 0.10:
                return False, "account below 10% of initial"
        else:
            size = self.portfolio_value() * 0.05
            if size > self.data["balance"]:
                return False, "insufficient cash"

        return True, "ok"

    # ── Enter trade ───────────────────────────────────────────
    def enter_trade(self, ticker: str, direction: str, entry_price: float,
                    asset_type: str, signal_strength: float, reason: str,
                    est_premium: float = 0.0, iv_rank: float = 0.0,
                    strike: Optional[float] = None,
                    expiry: Optional[str] = None) -> Optional[dict]:

        ok, msg = self._can_enter(ticker, asset_type, est_premium, iv_rank)
        if not ok:
            return {"rejected": True, "reason": msg, "ticker": ticker}
        if entry_price <= 0:
            return None

        if asset_type == "OPTIONS":
            # 3 contracts × 100 shares × premium
            premium = est_premium if est_premium > 0 else max(entry_price * 0.04, 0.50)
            total_cost = premium * SHARES * CONTRACTS

            # Stop loss  → underlying moves 5% against us → option -40%
            # Take profit → underlying moves 10% in favor  → option +80%
            if direction == "CALL":
                sl_underlying = entry_price * 0.95   # stock drops 5%
                tp_underlying = entry_price * 1.10   # stock gains 10%
            else:  # PUT
                sl_underlying = entry_price * 1.05
                tp_underlying = entry_price * 0.90

            pos = {
                "id":               f"T{int(time.time()*1000)}",
                "ticker":           ticker,
                "direction":        direction,
                "asset_type":       "OPTIONS",
                "entry_price":      round(entry_price, 4),   # underlying price
                "current_price":    round(entry_price, 4),
                "est_premium":      round(premium, 4),
                "contracts":        CONTRACTS,
                "shares_total":     CONTRACTS * SHARES,
                "total_cost":       round(total_cost, 2),
                "value":            round(total_cost, 2),
                "strike":           strike,
                "expiry":           expiry,
                "stop_loss":        round(sl_underlying, 4),   # underlying SL
                "take_profit":      round(tp_underlying, 4),   # underlying TP
                "sl_pct":           -40.0,   # option loses ~40%
                "tp_pct":           +80.0,   # option gains ~80%
                "iv_rank":          round(iv_rank, 4),
                "signal_strength":  round(signal_strength, 4),
                "reason":           reason,
                "entry_time":       datetime.utcnow().isoformat() + "Z",
                "unrealized_pnl":   0.0,
                "unrealized_pnl_pct": 0.0,
                "status":           "open",
            }
            self.data["balance"] = round(self.data["balance"] - total_cost, 2)

        else:  # CRYPTO
            size = min(self.portfolio_value() * 0.05, self.data["balance"])
            size = max(size, 20.0)
            if direction == "LONG":
                sl = entry_price * 0.93
                tp = entry_price * 1.18
            else:
                sl = entry_price * 1.07
                tp = entry_price * 0.82

            pos = {
                "id":              f"T{int(time.time()*1000)}",
                "ticker":          ticker,
                "direction":       direction,
                "asset_type":      "CRYPTO",
                "entry_price":     round(entry_price, 6),
                "current_price":   round(entry_price, 6),
                "value":           round(size, 2),
                "stop_loss":       round(sl, 6),
                "take_profit":     round(tp, 6),
                "signal_strength": round(signal_strength, 4),
                "reason":          reason,
                "entry_time":      datetime.utcnow().isoformat() + "Z",
                "unrealized_pnl":  0.0,
                "unrealized_pnl_pct": 0.0,
                "status":          "open",
            }
            self.data["balance"] = round(self.data["balance"] - size, 2)

        self.data["positions"].append(pos)
        self._save()
        return pos

    # ── Exit trade ────────────────────────────────────────────
    def exit_trade(self, trade_id: str, exit_price: float,
                   reason: str = "signal") -> Optional[dict]:
        pos = next((p for p in self.data["positions"] if p["id"] == trade_id), None)
        if not pos:
            return None

        ep, val, d = pos["entry_price"], pos["value"], pos["direction"]

        if pos["asset_type"] == "OPTIONS":
            underlying_move = (exit_price - ep) / ep
            opt_pct = (underlying_move * 2.5) if d == "CALL" else (-underlying_move * 2.5)
            opt_pct = max(opt_pct, -0.95)
            pnl     = round(val * opt_pct, 2)
            pnl_pct = round(opt_pct * 100, 2)
        else:
            raw     = (exit_price - ep) / ep if d == "LONG" else (ep - exit_price) / ep
            pnl     = round(val * raw, 2)
            pnl_pct = round(raw * 100, 2)

        closed = {**pos,
            "exit_price":       round(exit_price, 6),
            "exit_time":        datetime.utcnow().isoformat() + "Z",
            "realized_pnl":     pnl,
            "realized_pnl_pct": pnl_pct,
            "exit_reason":      reason,
            "status":           "closed",
        }

        self.data["balance"]   = round(self.data["balance"] + val + pnl, 2)
        self.data["positions"] = [p for p in self.data["positions"] if p["id"] != trade_id]
        self.data["trades"].insert(0, closed)
        self.data["trades"]    = self.data["trades"][:1000]
        self._save()
        return closed

    # ── Mark-to-market all positions ──────────────────────────
    def update_positions(self) -> list:
        triggered = []
        for pos in list(self.data["positions"]):
            px = self._price(pos["ticker"])
            if not px:
                continue
            pos["current_price"] = round(px, 6)
            ep, val, d = pos["entry_price"], pos["value"], pos["direction"]

            if pos["asset_type"] == "OPTIONS":
                um = (px - ep) / ep
                opt_pct = (um * 2.5) if d == "CALL" else (-um * 2.5)
                opt_pct = max(opt_pct, -0.95)
                pos["unrealized_pnl"]     = round(val * opt_pct, 2)
                pos["unrealized_pnl_pct"] = round(opt_pct * 100, 2)

                # SL: underlying crosses stop_loss level
                sl_hit = (d == "CALL" and px <= pos["stop_loss"]) or \
                         (d == "PUT"  and px >= pos["stop_loss"])
                tp_hit = (d == "CALL" and px >= pos["take_profit"]) or \
                         (d == "PUT"  and px <= pos["take_profit"])
            else:
                raw = (px - ep) / ep if d == "LONG" else (ep - px) / ep
                pos["unrealized_pnl"]     = round(val * raw, 2)
                pos["unrealized_pnl_pct"] = round(raw * 100, 2)
                sl_hit = (d == "LONG"  and px <= pos["stop_loss"]) or \
                         (d == "SHORT" and px >= pos["stop_loss"])
                tp_hit = (d == "LONG"  and px >= pos["take_profit"]) or \
                         (d == "SHORT" and px <= pos["take_profit"])

            if sl_hit:
                c = self.exit_trade(pos["id"], px, "stop_loss")
                if c: triggered.append(c)
            elif tp_hit:
                c = self.exit_trade(pos["id"], px, "take_profit")
                if c: triggered.append(c)

        self._save()
        return triggered

    # ── Process incoming signals ──────────────────────────────
    def process_signals(self, signals: list) -> list:
        triggered = self.update_positions()
        new_trades = []
        for sig in signals:
            if sig.get("signal_strength", 0) < 0.25:
                continue
            t = self.enter_trade(
                ticker          = sig["ticker"],
                direction       = sig["direction"],
                entry_price     = sig["entry_price"],
                asset_type      = sig["asset_type"],
                signal_strength = sig["signal_strength"],
                reason          = sig.get("reason", "bot_signal"),
                est_premium     = sig.get("est_premium", 0.0),
                iv_rank         = sig.get("iv_rank", 0.0),
                strike          = sig.get("strike"),
                expiry          = sig.get("expiry"),
            )
            if t and not t.get("rejected"):
                new_trades.append(t)
        return triggered + new_trades

    # ── Summary ───────────────────────────────────────────────
    def get_summary(self) -> dict:
        port  = self.portfolio_value()
        init  = self.data["initial_balance"]
        closed= [t for t in self.data["trades"] if "realized_pnl" in t]
        wins  = [t for t in closed if t["realized_pnl"] > 0]
        total_pnl = sum(t["realized_pnl"] for t in closed)

        return {
            "balance":          round(self.data["balance"], 2),
            "portfolio_value":  round(port, 2),
            "initial_balance":  init,
            "total_pnl":        round(total_pnl, 2),
            "total_return_pct": round((port - init) / init * 100, 2),
            "open_positions":   len(self.data["positions"]),
            "total_trades":     len(closed),
            "win_rate":         round(len(wins) / len(closed) * 100, 1) if closed else 0.0,
            "positions":        self.data["positions"],
            "recent_trades":    self.data["trades"][:50],
            "last_updated":     self.data.get("last_updated"),
        }
