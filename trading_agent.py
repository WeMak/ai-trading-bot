"""
Trading AI Agent  v1.0
======================
Inputs : any stock or crypto ticker + optional question
Outputs: candlestick chart (base64 PNG) + RSI / MACD / BB / SMA analysis
         + BUY / SELL / HOLD signal with confidence + trade levels

Indicators
──────────
  RSI(14)       Momentum oscillator — oversold/overbought
  MACD(12,26,9) Trend + momentum crossover
  BB(20,2σ)     Volatility envelope — price extremes
  SMA 50/200    Trend direction + golden/death cross
  ATR(14)       Volatility for SL/TP sizing
  Volume ratio  Current volume vs 20-day average
  S/R levels    Recent swing highs and lows
"""

import warnings; warnings.filterwarnings("ignore")
import io, base64, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional

# ── Dashboard-matching dark palette ──────────────────────────
BG      = "#0f172a"
PANEL   = "#1e293b"
GRID    = "#334155"
TEXT    = "#94a3b8"
UP      = "#22c55e"
DOWN    = "#ef4444"
BB_CLR  = "#60a5fa"
SMA50_C = "#f59e0b"
SMA200_C= "#a78bfa"
RSI_CLR = "#22d3ee"
MACD_C  = "#22d3ee"
SIG_C   = "#f59e0b"

# ── Math helpers ──────────────────────────────────────────────
def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    valid = np.where(~np.isnan(arr))[0]
    if len(valid) < period:
        return out
    s = valid[0]
    if s + period - 1 >= len(arr):
        return out
    out[s + period - 1] = float(np.nanmean(arr[s:s+period]))
    alpha = 2.0 / (period + 1)
    for i in range(s + period, len(arr)):
        if not np.isnan(arr[i]):
            out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
        else:
            out[i] = out[i-1]
    return out

def _rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(closes)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    delta = np.diff(closes.astype(float))
    gain  = np.where(delta > 0,  delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    alpha = 1.0 / period
    ag = np.full(len(gain), np.nan)
    al = np.full(len(gain), np.nan)
    ag[period-1] = float(np.mean(gain[:period]))
    al[period-1] = float(np.mean(loss[:period]))
    for i in range(period, len(gain)):
        ag[i] = alpha * gain[i] + (1 - alpha) * ag[i-1]
        al[i] = alpha * loss[i] + (1 - alpha) * al[i-1]
    rs = ag / (al + 1e-10)
    rsi_vals = 100.0 - 100.0 / (1.0 + rs)
    out[1:] = rsi_vals
    return out

def _macd(closes: np.ndarray, fast=12, slow=26, sig_p=9):
    ef = _ema(closes, fast)
    es = _ema(closes, slow)
    line   = ef - es
    signal = _ema(np.where(np.isnan(line), np.nan, line), sig_p)
    hist   = line - signal
    return line, signal, hist

def _bollinger(closes: np.ndarray, period=20, nstd=2.0):
    n = len(closes)
    mid   = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(period - 1, n):
        w = closes[i - period + 1:i + 1]
        m = float(np.mean(w))
        s = float(np.std(w))
        mid[i]   = m
        upper[i] = m + nstd * s
        lower[i] = m - nstd * s
    return upper, mid, lower

def _sma(closes: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(closes), np.nan)
    for i in range(period - 1, len(closes)):
        out[i] = float(np.mean(closes[i - period + 1:i + 1]))
    return out

def _atr(highs, lows, closes, period=14) -> np.ndarray:
    n = len(closes)
    out = np.full(n, np.nan)
    if n < 2:
        return out
    tr = np.maximum(highs[1:] - lows[1:],
         np.maximum(np.abs(highs[1:] - closes[:-1]),
                    np.abs(lows[1:]  - closes[:-1])))
    if len(tr) < period:
        return out
    out[period] = float(np.mean(tr[:period]))
    alpha = 1.0 / period
    for i in range(period + 1, n):
        out[i] = alpha * tr[i - 1] + (1 - alpha) * out[i - 1]
    return out

def _sr_levels(highs, lows, lookback=80, n=3):
    h = highs[-lookback:]
    l = lows[-lookback:]
    idx_h = np.argsort(h)[-n:]
    idx_l = np.argsort(l)[:n]
    res = sorted({round(float(h[i]), 4) for i in idx_h})
    sup = sorted({round(float(l[i]), 4) for i in idx_l})
    return sup, res

# ── Signal engine ─────────────────────────────────────────────
def _score(ind: dict) -> tuple:
    score  = 0.0
    cap    = 0.0
    notes  = []

    rsi        = ind["rsi"]
    macd_line  = ind["macd"]
    macd_sig   = ind["macd_signal"]
    macd_hist  = ind["macd_hist"]
    price      = ind["price"]
    bbu        = ind["bb_upper"]
    bbl        = ind["bb_lower"]
    sma50      = ind["sma50"]
    sma200     = ind["sma200"]
    vol_ratio  = ind["vol_ratio"]

    # ── RSI  (weight 3) ──
    cap += 3
    if rsi < 30:
        score += 3;   notes.append(f"RSI {rsi:.0f} — oversold, potential bounce ↑")
    elif rsi < 42:
        score += 1.5; notes.append(f"RSI {rsi:.0f} — below midpoint, mild bullish bias")
    elif rsi > 70:
        score -= 3;   notes.append(f"RSI {rsi:.0f} — overbought, potential reversal ↓")
    elif rsi > 58:
        score -= 1.5; notes.append(f"RSI {rsi:.0f} — above midpoint, mild bearish bias")
    else:
        notes.append(f"RSI {rsi:.0f} — neutral zone")

    # ── MACD  (weight 3) ──
    cap += 3
    if macd_line > macd_sig and macd_hist > 0:
        score += 3;   notes.append("MACD bullish crossover — momentum building ↑")
    elif macd_line > macd_sig:
        score += 1.5; notes.append("MACD above signal line — leaning bullish")
    elif macd_line < macd_sig and macd_hist < 0:
        score -= 3;   notes.append("MACD bearish crossover — momentum falling ↓")
    elif macd_line < macd_sig:
        score -= 1.5; notes.append("MACD below signal line — leaning bearish")
    else:
        notes.append("MACD — neutral / converging")

    # ── Bollinger Bands  (weight 2) ──
    cap += 2
    bb_range = bbu - bbl + 1e-10
    bb_pct   = (price - bbl) / bb_range
    if bb_pct < 0.08:
        score += 2;   notes.append("Price touching lower BB — mean-reversion buy zone")
    elif bb_pct > 0.92:
        score -= 2;   notes.append("Price touching upper BB — potential pullback zone")
    elif bb_pct < 0.30:
        score += 0.7; notes.append(f"Price in lower BB zone ({bb_pct*100:.0f}%)")
    elif bb_pct > 0.70:
        score -= 0.7; notes.append(f"Price in upper BB zone ({bb_pct*100:.0f}%)")
    else:
        notes.append(f"Price mid-BB ({bb_pct*100:.0f}%) — neutral")

    # ── Trend via SMAs  (weight 2) ──
    cap += 2
    if not np.isnan(sma200):
        if price > sma50 > sma200:
            score += 2;   notes.append("Price > SMA50 > SMA200 — strong uptrend (golden alignment)")
        elif price > sma50:
            score += 1;   notes.append("Price above SMA50 — short-term uptrend")
        elif price > sma200:
            score += 0.5; notes.append("Price above SMA200 — long-term bullish")
        elif price < sma50 < sma200:
            score -= 2;   notes.append("Price < SMA50 < SMA200 — strong downtrend (death alignment)")
        elif price < sma50:
            score -= 1;   notes.append("Price below SMA50 — short-term downtrend")
        else:
            score -= 0.5; notes.append("Price below SMA200 — long-term bearish")

    # ── Volume  (weight 1) ──
    cap += 1
    if vol_ratio > 2.0:
        bonus = 0.8 if score > 0 else -0.8
        score += bonus
        notes.append(f"Volume {vol_ratio:.1f}× avg — strong confirmation")
    elif vol_ratio > 1.4:
        bonus = 0.3 if score > 0 else -0.3
        score += bonus
        notes.append(f"Volume {vol_ratio:.1f}× avg — above average")
    else:
        notes.append(f"Volume {vol_ratio:.1f}× avg — below average (weak conviction)")

    # ── Normalise ──
    norm = score / cap  # -1 to +1

    if norm > 0.25:
        direction  = "BUY"
        confidence = round(min(40 + norm * 55, 94), 1)
    elif norm < -0.25:
        direction  = "SELL"
        confidence = round(min(40 + abs(norm) * 55, 94), 1)
    else:
        direction  = "HOLD"
        confidence = round(50 - abs(norm) * 40, 1)

    return direction, confidence, notes

# ── Chart ─────────────────────────────────────────────────────
def _build_chart(df, ind, signal, ticker: str) -> str:
    n  = len(df)
    xs = np.arange(n)

    opens  = df["Open"].values.astype(float)
    closes = df["Close"].values.astype(float)
    highs  = df["High"].values.astype(float)
    lows   = df["Low"].values.astype(float)
    vols   = df["Volume"].values.astype(float)

    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    gs  = GridSpec(4, 1, figure=fig,
                   height_ratios=[5, 1.5, 1.5, 1.5], hspace=0.04)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax3 = fig.add_subplot(gs[3], sharex=ax0)

    for ax in (ax0, ax1, ax2, ax3):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=7)
        ax.yaxis.tick_right()
        ax.spines[:].set_color(GRID)
        ax.grid(color=GRID, linewidth=0.4, alpha=0.5)
        ax.set_xlim(-1, n)

    # ── Candles ──
    for i in xs:
        c = UP if closes[i] >= opens[i] else DOWN
        ax0.plot([i, i], [lows[i], highs[i]], color=c, lw=0.7, zorder=2)
        body = max(abs(closes[i] - opens[i]), highs[i] * 0.001)
        ax0.add_patch(plt.Rectangle(
            (i - 0.4, min(opens[i], closes[i])), 0.8, body,
            color=c, alpha=0.9, zorder=3))

    # ── BB ──
    ax0.plot(xs, ind["bb_upper_arr"], color=BB_CLR, lw=0.8, alpha=0.55)
    ax0.plot(xs, ind["bb_mid_arr"],   color=BB_CLR, lw=0.8, alpha=0.35, ls="--")
    ax0.plot(xs, ind["bb_lower_arr"], color=BB_CLR, lw=0.8, alpha=0.55)
    ax0.fill_between(xs, ind["bb_lower_arr"], ind["bb_upper_arr"],
                     color=BB_CLR, alpha=0.04)

    # ── SMAs ──
    ax0.plot(xs, ind["sma50_arr"],  color=SMA50_C,  lw=1.1, alpha=0.8, label="SMA 50")
    ax0.plot(xs, ind["sma200_arr"], color=SMA200_C, lw=1.1, alpha=0.8, label="SMA 200")

    # ── S/R lines ──
    for lvl in ind["supports"]:
        ax0.axhline(lvl, color=UP,   lw=0.55, ls=":", alpha=0.45)
    for lvl in ind["resistances"]:
        ax0.axhline(lvl, color=DOWN, lw=0.55, ls=":", alpha=0.45)

    # ── Signal label ──
    sig_dir, sig_conf, _ = signal
    sc = {" BUY": UP, "SELL": DOWN, "HOLD": SIG_C}[f"{sig_dir:>4}"]
    ax0.annotate(
        f"  {sig_dir}  {sig_conf:.0f}%  ",
        xy=(n - 1, closes[-1]),
        xytext=(max(n - 12, 0), closes[-1] * (1.025 if closes[-1] > opens[-1] else 0.975)),
        fontsize=10, fontweight="bold", color=sc,
        bbox=dict(boxstyle="round,pad=0.35", fc=BG, ec=sc, lw=1.5),
        arrowprops=dict(arrowstyle="->", color=sc, lw=1.2),
        zorder=10,
    )

    price = closes[-1]
    chg   = (closes[-1] - closes[-2]) / closes[-2] * 100 if n > 1 else 0
    arrow = "▲" if chg >= 0 else "▼"
    ax0.set_title(
        f"{ticker.upper()}   ${price:,.4f}   {arrow} {chg:+.2f}%   "
        f"(RSI {ind['rsi']:.0f} · {('▲ Uptrend' if price > ind['sma200'] else '▼ Downtrend')})",
        color="white", fontsize=12, fontweight="bold", loc="left", pad=8,
    )
    legend_handles = [
        mpatches.Patch(color=SMA50_C,  label="SMA 50"),
        mpatches.Patch(color=SMA200_C, label="SMA 200"),
        mpatches.Patch(color=BB_CLR,   label="Bollinger Bands"),
        mpatches.Patch(color=UP,       label="Support"),
        mpatches.Patch(color=DOWN,     label="Resistance"),
    ]
    ax0.legend(handles=legend_handles, loc="upper left",
               framealpha=0.25, fontsize=7, labelcolor=TEXT)

    # ── Volume ──
    vol_avg = float(np.nanmean(vols[-20:]))
    vcols   = [UP if closes[i] >= opens[i] else DOWN for i in xs]
    ax1.bar(xs, vols, color=vcols, alpha=0.65, width=0.8)
    ax1.axhline(vol_avg, color=TEXT, lw=0.55, ls="--", alpha=0.4)
    ax1.set_ylabel("Vol", color=TEXT, fontsize=7, rotation=0, labelpad=22)
    ax1.yaxis.set_major_formatter(FuncFormatter(
        lambda x, _: f"{x/1e9:.1f}B" if x >= 1e9
                     else f"{x/1e6:.1f}M" if x >= 1e6
                     else f"{x/1e3:.0f}K"))

    # ── RSI ──
    rsi_a = ind["rsi_arr"]
    ax2.plot(xs, rsi_a, color=RSI_CLR, lw=1.0)
    ax2.axhline(70, color=DOWN, lw=0.6, ls="--", alpha=0.55)
    ax2.axhline(30, color=UP,   lw=0.6, ls="--", alpha=0.55)
    ax2.fill_between(xs, rsi_a, 70, where=(rsi_a >= 70), color=DOWN, alpha=0.12)
    ax2.fill_between(xs, rsi_a, 30, where=(rsi_a <= 30), color=UP,   alpha=0.12)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI", color=TEXT, fontsize=7, rotation=0, labelpad=22)
    cur = ind["rsi"]
    ax2.text(n - 0.5, cur, f" {cur:.0f}", color=RSI_CLR, fontsize=7, va="center")

    # ── MACD ──
    ml  = ind["macd_arr"]
    msl = ind["macd_sig_arr"]
    mh  = ind["macd_hist_arr"]
    ax3.plot(xs, ml,  color=MACD_C, lw=1.0)
    ax3.plot(xs, msl, color=SIG_C,  lw=1.0)
    hcols = [UP if (v if not math.isnan(v or 0) else 0) >= 0 else DOWN
             for v in (mh if mh is not None else np.zeros(n))]
    ax3.bar(xs, mh, color=hcols, alpha=0.55, width=0.8)
    ax3.axhline(0, color=GRID, lw=0.5)
    ax3.set_ylabel("MACD", color=TEXT, fontsize=7, rotation=0, labelpad=22)

    # ── X-axis dates ──
    dates = df.index
    step  = max(n // 9, 1)
    ticks = xs[::step]
    ax3.set_xticks(ticks)
    ax3.set_xticklabels(
        [dates[i].strftime("%b %d") for i in ticks],
        color=TEXT, fontsize=7,
    )
    for ax in (ax0, ax1, ax2):
        plt.setp(ax.get_xticklabels(), visible=False)

    plt.tight_layout(pad=0.6)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ── Text analysis ─────────────────────────────────────────────
def _narrative(ticker, price, ind, signal, question=None) -> str:
    direction, confidence, notes = signal
    trend = "Uptrend" if price > ind["sma200"] else "Downtrend"
    bullets = "\n".join(f"• {n}" for n in notes)

    trade_block = ""
    if direction == "BUY":
        rr = ind["rr_ratio"]
        trade_block = (
            f"\n\n**Trade Setup — BUY**\n"
            f"• Entry zone:   ~${price:,.4f}\n"
            f"• Stop Loss:    ${ind['stop_loss']:,.4f}  "
            f"(ATR-based, ${price - ind['stop_loss']:,.4f} risk)\n"
            f"• Take Profit:  ${ind['take_profit']:,.4f}  "
            f"(${ind['take_profit'] - price:,.4f} reward)\n"
            f"• Risk/Reward:  1 : {rr:.1f}"
        )
    elif direction == "SELL":
        rr = ind["rr_ratio"]
        trade_block = (
            f"\n\n**Trade Setup — SHORT / EXIT**\n"
            f"• Entry zone:   ~${price:,.4f}\n"
            f"• Stop Loss:    ${ind['stop_loss']:,.4f}\n"
            f"• Take Profit:  ${ind['take_profit']:,.4f}\n"
            f"• Risk/Reward:  1 : {rr:.1f}"
        )
    else:
        trade_block = (
            f"\n\n**No trade — HOLD**\n"
            f"Indicators are mixed or in neutral territory. Wait for "
            f"RSI to reach extreme levels or a confirmed MACD crossover before entering."
        )

    answer_block = ""
    if question:
        answer_block = f"\n\n**Your question:** _{question}_\n"
        if direction == "HOLD":
            answer_block += "No clear edge right now. The indicators are balanced — patience is the trade here."
        else:
            answer_block += (
                f"The technicals lean **{direction}** with {confidence:.0f}% confidence. "
                f"Key driver: {notes[0] if notes else 'see signals above'}."
            )

    return (
        f"**{ticker.upper()} — AI Technical Analysis**\n\n"
        f"Price: ${price:,.4f}  |  Trend: {trend}  |  ATR: ${ind['atr']:,.4f}\n\n"
        f"**Indicator Signals:**\n{bullets}"
        f"{trade_block}"
        f"{answer_block}\n\n"
        f"---\n*Based on 6 months of OHLCV data · RSI/MACD/BB/SMA/ATR · Not financial advice*"
    )

# ── Main entry point ──────────────────────────────────────────
CRYPTO_MAP = {
    "BTC":"BTC-USD","ETH":"ETH-USD","SOL":"SOL-USD","BNB":"BNB-USD",
    "XRP":"XRP-USD","DOGE":"DOGE-USD","ADA":"ADA-USD","AVAX":"AVAX-USD",
    "LINK":"LINK-USD","DOT":"DOT-USD","MATIC":"MATIC-USD","LTC":"LTC-USD",
    "BCH":"BCH-USD","UNI":"UNI-USD","ATOM":"ATOM-USD","XLM":"XLM-USD",
    "TRX":"TRX-USD","NEAR":"NEAR-USD","FIL":"FIL-USD","APT":"APT-USD",
}

def analyze(ticker: str, question: Optional[str] = None) -> dict:
    raw      = ticker.strip().upper().replace("$", "")
    yf_sym   = CRYPTO_MAP.get(raw, raw)
    is_crypto = "-USD" in yf_sym

    end   = datetime.today()
    start = end - timedelta(days=260)

    from data_fetcher import fetch as _dfetch
    df = _dfetch(yf_sym, start=start, end=end)
    if (df is None or df.empty) and not is_crypto:
        df = _dfetch(raw + "-USD", start=start, end=end)
        if df is not None and not df.empty:
            yf_sym    = raw + "-USD"
            is_crypto = True

    if df is None or df.empty:
        return {"error": f"No data found for '{raw}'. Check the ticker.", "ticker": raw}

    df = df.dropna(subset=["Close"])
    if len(df) < 55:
        return {"error": f"Not enough history for {raw} ({len(df)} bars).", "ticker": raw}

    closes  = df["Close"].values.astype(float)
    opens   = df["Open"].values.astype(float)
    highs   = df["High"].values.astype(float)
    lows    = df["Low"].values.astype(float)
    volumes = df["Volume"].values.astype(float)
    price   = float(closes[-1])

    rsi_arr          = _rsi(closes)
    ml, msl, mh      = _macd(closes)
    bbu, bbm, bbl    = _bollinger(closes)
    sma50_arr        = _sma(closes, 50)
    sma200_arr       = _sma(closes, 200)
    atr_arr          = _atr(highs, lows, closes)
    vol_avg          = float(np.nanmean(volumes[-20:]))
    vol_ratio        = float(volumes[-1]) / (vol_avg + 1e-10)
    supports, resistances = _sr_levels(highs, lows)

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

    # ATR-based SL/TP
    if cur_rsi < 50:   # bullish bias
        sl = price - 1.5 * cur_atr
        tp = price + 2.5 * cur_atr
    else:              # bearish / neutral
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
        "supports":    supports,
        "resistances": resistances,
        # Arrays for chart
        "rsi_arr":       rsi_arr,
        "macd_arr":      ml,
        "macd_sig_arr":  msl,
        "macd_hist_arr": mh,
        "bb_upper_arr":  bbu,
        "bb_mid_arr":    bbm,
        "bb_lower_arr":  bbl,
        "sma50_arr":     sma50_arr,
        "sma200_arr":    sma200_arr,
    }

    signal     = _score(ind)
    chart_b64  = _build_chart(df, ind, signal, raw)
    text       = _narrative(raw, price, ind, signal, question)

    # Serialisable indicators only (no numpy arrays)
    ind_out = {k: v for k, v in ind.items() if not k.endswith("_arr")
               and k not in ("supports", "resistances")}
    ind_out["supports"]    = supports
    ind_out["resistances"] = resistances

    return {
        "ticker":     raw,
        "yf_ticker":  yf_sym,
        "is_crypto":  is_crypto,
        "price":      round(price, 6),
        "signal":     signal[0],
        "confidence": signal[1],
        "notes":      signal[2],
        "chart_b64":  chart_b64,
        "analysis":   text,
        "indicators": ind_out,
        "timestamp":  datetime.utcnow().isoformat() + "Z",
    }


if __name__ == "__main__":
    import sys, json
    t = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    q = sys.argv[2] if len(sys.argv) > 2 else None
    print(f"Analyzing {t}…")
    r = analyze(t, q)
    if "error" in r:
        print("ERROR:", r["error"])
    else:
        print(f"Signal: {r['signal']} ({r['confidence']}%)")
        print(f"Price:  ${r['price']}")
        print(f"RSI:    {r['indicators']['rsi']}")
        print("Notes:", "\n  ".join(r["notes"]))
        with open("test_chart.html", "w") as f:
            f.write(f'<img src="data:image/png;base64,{r["chart_b64"]}" style="max-width:100%">')
        print("Chart saved to test_chart.html")
