"""
Crypto NEAT Scanner — Top 20 Coins
Same NEAT architecture as stock scanner:
  Inputs:  [BTC_Relative_Strength, HV_Rank, Dist_200DMA, Momentum]
  Output:  LONG / SHORT signal with fitness score
  Evolves: 50 agents x 10 generations
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import numpy as np
import random
import copy
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

CRYPTO = {
    "BTC": "BTC-USD", "ETH": "ETH-USD", "BNB": "BNB-USD",
    "SOL": "SOL-USD", "XRP": "XRP-USD", "DOGE": "DOGE-USD",
    "ADA": "ADA-USD", "AVAX": "AVAX-USD", "LINK": "LINK-USD",
    "DOT": "DOT-USD", "MATIC": "MATIC-USD", "LTC": "LTC-USD",
    "BCH": "BCH-USD", "UNI": "UNI-USD", "ATOM": "ATOM-USD",
    "XLM": "XLM-USD", "TRX": "TRX-USD", "NEAR": "NEAR-USD",
    "FIL": "FIL-USD", "APT": "APT-USD",
}

# ── Data fetch ────────────────────────────────────────────────
def _fetch_data():
    end   = datetime.today()
    start = end - timedelta(days=420)
    symbols = list(CRYPTO.values()) + ["BTC-USD"]
    raw  = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False)
    cls  = raw["Close"].dropna(how="all")
    btc  = cls["BTC-USD"].dropna() if "BTC-USD" in cls.columns else None
    return cls, btc

# ── Features ──────────────────────────────────────────────────
def _features(symbol: str, name: str, closes, btc_closes):
    if symbol not in closes.columns:
        return None
    p = closes[symbol].dropna()
    if len(p) < 200:
        return None

    p_now  = float(p.iloc[-1])
    p252   = p.iloc[-min(252, len(p)):]
    log_ret = np.log(p252 / p252.shift(1)).dropna().values

    # 1. Relative strength vs BTC
    stk_ret = (float(p252.iloc[-1]) / float(p252.iloc[0])) - 1.0
    if btc_closes is not None and len(btc_closes) >= 2:
        btc252  = btc_closes.iloc[-min(252, len(p)):]
        btc_ret = (float(btc252.iloc[-1]) / float(btc252.iloc[0])) - 1.0
        btc_rs  = (stk_ret - btc_ret) / (abs(btc_ret) + 1e-6)
    else:
        btc_rs = stk_ret

    # 2. HV rank (IV proxy)
    hv10    = float(np.std(log_ret[-10:]) * np.sqrt(365))
    windows = [np.std(log_ret[max(0,i-10):i]) * np.sqrt(365) for i in range(10, len(log_ret)+1)]
    hv_rank = float((hv10 - min(windows)) / (max(windows) - min(windows) + 1e-9))
    hv252   = float(np.std(log_ret) * np.sqrt(365))

    # 3. Distance from 200 DMA
    ma200  = float(np.mean(p.iloc[-200:].values))
    d200   = (p_now - ma200) / ma200

    # 4. Momentum (short-term vs medium)
    r5  = float(np.sum(log_ret[-5:]))
    r20 = float(np.sum(log_ret[-20:]))
    mom = r5 / (abs(r20) + 1e-6)

    return {
        "ticker": name, "symbol": symbol,
        "price": round(p_now, 6),
        "btc_rs": btc_rs, "hv_rank": hv_rank,
        "dist_200": d200, "momentum": mom,
        "hv252": hv252, "ret_1yr": stk_ret,
    }

# ── Backtest ──────────────────────────────────────────────────
def _backtest(symbol: str, weights, closes, lookback=252):
    if symbol not in closes.columns:
        return 0.0, 0.0
    p = closes[symbol].dropna()
    if len(p) < 100:
        return 0.0, 0.0
    p = p.iloc[-min(lookback, len(p)):]
    lr = np.log(p / p.shift(1)).fillna(0).values

    equity, peak, max_dd = 1.0, 1.0, 0.0
    for i in range(20, len(p) - 4, 4):
        chunk   = lr[max(0,i-10):i]
        hv_loc  = np.std(chunk) * np.sqrt(365) if len(chunk) > 1 else 0.5
        rs_loc  = float(np.sum(lr[max(0,i-20):i]))
        iv_loc  = min(1.0, hv_loc / (np.std(lr) * np.sqrt(365) + 1e-6))
        d200_loc= float((p.iloc[i] - np.mean(p.iloc[max(0,i-200):i].values)) /
                        (np.mean(p.iloc[max(0,i-200):i].values) + 1e-6))
        mom_loc = -float(np.sum(lr[max(0,i-5):i]))

        signal = (weights[0]*rs_loc + (-weights[1])*iv_loc +
                  weights[2]*d200_loc + weights[3]*mom_loc)
        if abs(signal) < 0.05:
            continue

        fwd = float(np.sum(lr[i:i+4]))
        direction = np.sign(signal)
        pos = 0.05
        pnl = pos * abs(fwd) * 2.0 if direction * fwd > 0 else -pos * 0.55

        equity *= (1 + pnl)
        peak    = max(peak, equity)
        max_dd  = max(max_dd, (peak - equity) / peak)

    return equity - 1.0, max_dd

# ── NEAT Agent ────────────────────────────────────────────────
class Agent:
    def __init__(self, w=None):
        self.weights = np.random.uniform(-1, 1, 4) if w is None else np.array(w, float)
        self.fitness = None

    def mutate(self):
        c = copy.deepcopy(self)
        for i in range(4):
            if random.random() < 0.3:
                c.weights[i] = np.clip(c.weights[i] + np.random.normal(0, 0.3), -2, 2)
        c.fitness = None
        return c

    def crossover(self, other):
        mask = np.random.randint(0, 2, 4).astype(bool)
        return Agent(np.where(mask, self.weights, other.weights))

def _eval(agent, data, closes):
    rets, dds = [], []
    for d in data:
        r, dd = _backtest(d["symbol"], agent.weights, closes)
        rets.append(r); dds.append(dd)
    avg_r  = np.mean(rets) if rets else -1.0
    avg_dd = np.mean(dds)  if dds  else 0.0
    agent.fitness = avg_r * (0.5 if avg_dd > 0.12 else 1.0)

def _evolve(data, closes, pop=50, gens=10):
    population = [Agent() for _ in range(pop)]
    for _ in range(gens):
        for a in population:
            if a.fitness is None:
                _eval(a, data, closes)
        population.sort(key=lambda a: a.fitness or -99, reverse=True)
        elite = population[:5]
        kids  = [random.sample(elite,2)[0].crossover(random.sample(elite,2)[1])
                 for _ in range(pop//2)]
        for e in elite:
            kids += [e.mutate() for _ in range(3)]
        while len(kids) < pop - len(elite):
            kids.append(Agent())
        population = elite + kids[:pop-len(elite)]
    population.sort(key=lambda a: a.fitness or -99, reverse=True)
    return population[0]

# ── Main scan function ────────────────────────────────────────
def run_crypto_scan() -> dict:
    closes, btc = _fetch_data()

    # Compute features
    feats = []
    for name, sym in CRYPTO.items():
        f = _features(sym, name, closes, btc)
        if f:
            feats.append(f)

    if not feats:
        return {"error": "No data", "coins": [], "timestamp": datetime.utcnow().isoformat()+"Z"}

    # Evolve champion
    champ = _evolve(feats, closes)
    if champ.fitness is None:
        _eval(champ, feats, closes)

    w = champ.weights

    # Score each coin
    scored = []
    for f in feats:
        sig = (w[0]*f["btc_rs"] + (-w[1])*f["hv_rank"] +
               w[2]*f["dist_200"] + w[3]*f["momentum"])
        direction = "LONG" if sig > 0 else "SHORT"
        strength  = min(abs(sig), 1.0)
        scored.append({
            "ticker":    f["ticker"],
            "symbol":    f["symbol"],
            "price":     f["price"],
            "direction": direction,
            "signal":    round(float(sig), 4),
            "strength":  round(float(strength), 4),
            "btc_rs":    round(f["btc_rs"], 4),
            "hv_rank":   round(f["hv_rank"], 4),
            "dist_200":  round(f["dist_200"], 4),
            "momentum":  round(f["momentum"], 4),
            "hv252":     round(f["hv252"], 4),
            "ret_1yr":   round(f["ret_1yr"], 4),
        })

    scored.sort(key=lambda x: abs(x["signal"]), reverse=True)

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dna": [round(float(x), 3) for x in w.tolist()],
        "fitness": round(float(champ.fitness), 4),
        "coins": scored,
        "top_longs":  [c for c in scored if c["direction"] == "LONG"][:5],
        "top_shorts": [c for c in scored if c["direction"] == "SHORT"][:5],
    }

if __name__ == "__main__":
    import json
    print("Running crypto scan...")
    result = run_crypto_scan()
    print(f"\nChampion DNA: {result['dna']}  Fitness: {result['fitness']}")
    print(f"\nTop LONG signals:")
    for c in result["top_longs"]:
        print(f"  {c['ticker']:6s} ${c['price']:.4f}  sig={c['signal']:+.3f}  "
              f"BTC_RS={c['btc_rs']:+.2f}  HV={c['hv_rank']:.0%}  "
              f"D200={c['dist_200']:+.2%}")
    print(f"\nTop SHORT signals:")
    for c in result["top_shorts"]:
        print(f"  {c['ticker']:6s} ${c['price']:.4f}  sig={c['signal']:+.3f}")
