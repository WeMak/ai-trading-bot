import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import numpy as np
import pandas as pd
import random
import copy
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# ─── 1. UNIVERSE ───────────────────────────────────────────────────────────────
SECTORS = {
    "Technology":       ["AAPL","NVDA","MSFT","AVGO","META","TSM","ORCL","AMD","QCOM","INTC"],
    "Financials":       ["JPM","BAC","WFC","GS","MS","BLK","SCHW","C","AXP","COF"],
    "Healthcare":       ["LLY","UNH","JNJ","ABBV","MRK","TMO","ABT","DHR","BMY","AMGN"],
    "Energy":           ["XOM","CVX","COP","SLB","EOG","MPC","PSX","VLO","OXY","HAL"],
    "Consumer_Discret": ["AMZN","TSLA","HD","MCD","NKE","LOW","SBUX","TGT","BKNG","CMG"],
    "Consumer_Staples": ["WMT","PG","KO","PEP","COST","PM","MO","CL","GIS","KMB"],
    "Industrials":      ["GE","CAT","RTX","HON","UPS","BA","LMT","DE","MMM","CSX"],
    "Utilities":        ["NEE","DUK","SO","D","AEP","EXC","SRE","XEL","PEG","ED"],
    "Materials":        ["LIN","APD","SHW","ECL","NEM","FCX","NUE","ALB","IFF","MOS"],
    "Real_Estate":      ["PLD","AMT","EQIX","CCI","PSA","O","WELL","SPG","DLR","AVB"],
    "Comm_Services":    ["GOOGL","NFLX","DIS","CMCSA","T","VZ","TMUS","EA","TTWO","WBD"],
}

all_tickers = [t for s in SECTORS.values() for t in s]
print(f"Universe: {len(all_tickers)} tickers across {len(SECTORS)} sectors")

etf_map = {
    "Technology":"XLK","Financials":"XLF","Healthcare":"XLV","Energy":"XLE",
    "Consumer_Discret":"XLY","Consumer_Staples":"XLP","Industrials":"XLI",
    "Utilities":"XLU","Materials":"XLB","Real_Estate":"XLRE","Comm_Services":"XLC"
}

# ─── 2. FETCH DATA ─────────────────────────────────────────────────────────────
print("Fetching 1-year price data...")
end   = datetime.today()
start = end - timedelta(days=420)

raw    = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=False)
closes = raw["Close"].dropna(how="all")
print(f"Got {len(closes)} days, {closes.shape[1]} tickers loaded")

etfs    = list(etf_map.values())
etf_raw = yf.download(etfs, start=start, end=end, auto_adjust=True, progress=False)
etf_cls = etf_raw["Close"].dropna(how="all")

# ─── 3. FEATURE ENGINEERING ────────────────────────────────────────────────────
def compute_features(ticker, sector):
    if ticker not in closes.columns:
        return None
    p = closes[ticker].dropna()
    if len(p) < 200:
        return None

    p_now  = float(p.iloc[-1])
    p252   = p.iloc[-min(252, len(p)):]

    # Sector Relative Strength
    etf_sym = etf_map[sector]
    stk_ret = (float(p252.iloc[-1]) / float(p252.iloc[0])) - 1.0
    if etf_sym in etf_cls.columns:
        ep    = etf_cls[etf_sym].dropna().iloc[-min(252,len(p)):]
        e_ret = (float(ep.iloc[-1]) / float(ep.iloc[0])) - 1.0
        sector_rs = (stk_ret - e_ret) / (abs(e_ret) + 1e-6)
    else:
        sector_rs = stk_ret

    # IV Rank proxy via rolling HV10 percentile
    log_ret  = np.log(p252 / p252.shift(1)).dropna().values
    hv10     = float(np.std(log_ret[-10:]) * np.sqrt(252))
    hv252    = float(np.std(log_ret) * np.sqrt(252))
    windows  = [np.std(log_ret[max(0,i-10):i]) * np.sqrt(252) for i in range(10, len(log_ret)+1)]
    iv_rank  = float((hv10 - min(windows)) / (max(windows) - min(windows) + 1e-9))

    # Distance from 200DMA
    ma200   = float(np.mean(p.iloc[-200:].values))
    dist200 = (p_now - ma200) / ma200

    # Put/Call proxy
    r5  = float(np.sum(log_ret[-5:]))
    r20 = float(np.sum(log_ret[-20:]))
    pc  = -r5 / (abs(r20) + 1e-6)

    return {
        "ticker":    ticker,
        "sector":    sector,
        "price":     round(p_now, 2),
        "sector_rs": sector_rs,
        "iv_rank":   iv_rank,
        "dist_200":  dist200,
        "pc_ratio":  pc,
        "hv252":     hv252,
        "ret_1yr":   stk_ret,
    }

print("Computing features...")
features = {}
for sector, tickers in SECTORS.items():
    features[sector] = []
    for t in tickers:
        f = compute_features(t, sector)
        if f:
            features[sector].append(f)

total = sum(len(v) for v in features.values())
print(f"Valid rows: {total}")

# ─── 4. BACKTEST ENGINE ────────────────────────────────────────────────────────
def backtest(ticker, weights, lookback=252):
    if ticker not in closes.columns:
        return 0.0, 0.0
    p = closes[ticker].dropna()
    if len(p) < 100:
        return 0.0, 0.0
    p = p.iloc[-min(lookback, len(p)):]
    log_ret = np.log(p / p.shift(1)).fillna(0).values

    equity, peak, max_dd = 1.0, 1.0, 0.0
    for i in range(20, len(p) - 6, 5):
        chunk    = log_ret[max(0,i-10):i]
        hv_loc   = np.std(chunk) * np.sqrt(252) if len(chunk) > 1 else 0.2
        rs_loc   = float(np.sum(log_ret[max(0,i-20):i]))
        iv_loc   = min(1.0, hv_loc / (np.std(log_ret) * np.sqrt(252) + 1e-6))
        d200_loc = float((p.iloc[i] - np.mean(p.iloc[max(0,i-200):i].values)) /
                         (np.mean(p.iloc[max(0,i-200):i].values) + 1e-6))
        pc_loc   = -float(np.sum(log_ret[max(0,i-5):i]))

        signal = (weights[0]*rs_loc + (-weights[1])*iv_loc +
                  weights[2]*d200_loc + weights[3]*pc_loc)
        if abs(signal) < 0.05:
            continue

        fwd = float(np.sum(log_ret[i:i+5]))
        direction = np.sign(signal)
        pos = 0.05
        pnl = pos * abs(fwd) * 2.5 if direction * fwd > 0 else -pos * 0.6

        equity *= (1 + pnl)
        peak    = max(peak, equity)
        dd      = (peak - equity) / peak
        max_dd  = max(max_dd, dd)

    return equity - 1.0, max_dd

# ─── 5. GENETIC ALGORITHM ──────────────────────────────────────────────────────
class Agent:
    def __init__(self, weights=None):
        self.weights = np.random.uniform(-1,1,4) if weights is None else np.array(weights, dtype=float)
        self.fitness = None

    def mutate(self, rate=0.3):
        c = copy.deepcopy(self)
        for i in range(4):
            if random.random() < rate:
                c.weights[i] = np.clip(c.weights[i] + np.random.normal(0,0.3), -2, 2)
        c.fitness = None
        return c

    def crossover(self, other):
        mask = np.random.randint(0, 2, 4).astype(bool)
        return Agent(np.where(mask, self.weights, other.weights))

def eval_agent(agent, data):
    rets, dds = [], []
    for s in data:
        r, dd = backtest(s["ticker"], agent.weights)
        rets.append(r)
        dds.append(dd)
    avg_r  = np.mean(rets) if rets else -1.0
    avg_dd = np.mean(dds)  if dds  else 0.0
    agent.fitness = avg_r * (0.5 if avg_dd > 0.10 else 1.0)

def evolve_sector(sector, data, pop=50, gens=10):
    population = [Agent() for _ in range(pop)]
    for g in range(gens):
        for a in population:
            if a.fitness is None:
                eval_agent(a, data)
        population.sort(key=lambda a: a.fitness, reverse=True)
        elite = population[:5]
        kids  = []
        for _ in range(pop // 2):
            p1, p2 = random.sample(elite, 2)
            kids.append(p1.crossover(p2))
        for e in elite:
            for _ in range(3):
                kids.append(e.mutate())
        while len(kids) < pop - len(elite):
            kids.append(Agent())
        population = elite + kids[:pop-len(elite)]
    population.sort(key=lambda a: a.fitness if a.fitness else -99, reverse=True)
    return population[0]

# ─── 6. EVOLVE ALL SECTORS ─────────────────────────────────────────────────────
print("\n" + "="*62)
print("EVOLUTIONARY RUN  |  50 agents  |  10 generations  |  11 sectors")
print("="*62)

champions = {}
for sector, data in features.items():
    if not data:
        continue
    champ = evolve_sector(sector, data)
    if champ.fitness is None:
        eval_agent(champ, data)

    scored = []
    for s in data:
        sig = (champ.weights[0]*s["sector_rs"] + (-champ.weights[1])*s["iv_rank"] +
               champ.weights[2]*s["dist_200"] + champ.weights[3]*s["pc_ratio"])
        scored.append((s, sig))
    scored.sort(key=lambda x: x[1], reverse=True)
    best = scored[0][0]

    champions[sector] = {
        "dna":     [round(float(w),3) for w in champ.weights],
        "fitness": round(float(champ.fitness),4),
        "best":    best,
    }
    print(f"  {sector:<22} | {best['ticker']:<5} | fit={champ.fitness:+.3f} | dna={[round(float(w),2) for w in champ.weights]}")

# ─── 7. SECTOR RANKING ─────────────────────────────────────────────────────────
ranked = sorted(champions.items(), key=lambda x: x[1]["fitness"], reverse=True)

print("\n" + "="*62)
print("CHAMPION DNA — ALL 11 SECTORS (ranked by fitness)")
print("="*62)
print(f"{'Rk':<3} {'Sector':<22} {'Ticker':<6} {'Price':>8}  {'DNA [RS,IV,D200,PC]':<26} {'Fitness':>8}")
print("-"*80)
for rk, (sec, c) in enumerate(ranked, 1):
    s = c["best"]
    print(f" {rk:<3} {sec:<22} {s['ticker']:<6} ${s['price']:>7.2f}  {str(c['dna']):<26} {c['fitness']:>+8.4f}")

# ─── 8. TOP 5 RECOMMENDATIONS ──────────────────────────────────────────────────
print("\n" + "="*62)
print("TOP 5 SECTORS — OPTION TRADE RECOMMENDATIONS")
print("="*62)

for rk, (sector, c) in enumerate(ranked[:5], 1):
    stk  = c["best"]
    t    = stk["ticker"]
    px   = stk["price"]
    iv_r = stk["iv_rank"]
    rs   = stk["sector_rs"]
    d200 = stk["dist_200"]
    hv   = stk["hv252"]
    dna  = c["dna"]

    direction = "CALL" if (dna[0]*rs + dna[2]*d200) > 0 else "PUT"
    exp_days  = 30 if iv_r < 0.30 else 45
    exp_date  = (datetime.today() + timedelta(days=exp_days)).strftime("%b %d '%y")

    if direction == "CALL":
        strike = round(px * 1.02 / 5) * 5
    else:
        strike = round(px * 0.98 / 5) * 5

    T        = exp_days / 365.0
    est_prem = round(px * max(hv, 0.15) * (T**0.5) * 0.4, 2)

    reasons = []
    if iv_r < 0.30:  reasons.append(f"IV Rank {iv_r:.0%} — bottom tercile (CHEAP)")
    elif iv_r < 0.50: reasons.append(f"IV Rank {iv_r:.0%} — below median")
    if rs > 0.10:    reasons.append(f"Sector RS +{rs:.0%} outperformance")
    if abs(d200) < 0.04: reasons.append("Coiling near 200DMA")
    if not reasons:  reasons.append(f"HV={hv:.0%}; IV Rank={iv_r:.0%}")

    print(f"""
#{rk} {sector}
  Ticker  : {t} @ ${px:.2f}
  Trade   : BUY {direction}  |  Strike ${strike}  |  Expiry {exp_date}
  Est Prem: ~${est_prem:.2f}/share  (~${est_prem*100:.0f}/contract)
  DNA     : RS={dna[0]:+.3f}  IV={dna[1]:+.3f}  D200={dna[2]:+.3f}  PC={dna[3]:+.3f}
  Fitness : {c['fitness']:+.4f}
  Why Cheap: {"  //  ".join(reasons)}
  Signals  : Sector RS={rs:+.2%}  |  Dist-200DMA={d200:+.2%}  |  HV(1yr)={hv:.1%}""")

print("\n" + "="*62)
print("SCAN COMPLETE —", datetime.today().strftime("%Y-%m-%d %H:%M"))
print("="*62)
