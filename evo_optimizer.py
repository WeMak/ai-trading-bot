"""
Evolutionary Strategy Optimizer (DEAP)
=======================================
Uses genetic algorithms to evolve trading strategy parameters:
  - Entry/exit thresholds
  - Position sizing rules
  - Stop-loss / take-profit levels
  - Indicator weights and combinations
  - Timeframe selection

Each individual = a set of strategy hyperparameters.
Fitness = risk-adjusted return (Sharpe-like) on historical data.
"""

import warnings; warnings.filterwarnings("ignore")
import os, json, time, random, math, pickle
import numpy as np
from datetime import datetime
from typing import Optional
from deap import base, creator, tools, algorithms

DATA_DIR = os.path.join(os.path.dirname(__file__), "evo_data")
os.makedirs(DATA_DIR, exist_ok=True)

BEST_PARAMS_PATH = os.path.join(DATA_DIR, "best_params.json")
POPULATION_PATH = os.path.join(DATA_DIR, "population.pkl")
STATS_PATH = os.path.join(DATA_DIR, "evolution_stats.json")

# ─── STRATEGY GENOME ────────────────────────────────────────────
# Each individual has these genes (all normalized 0-1, mapped to real ranges):
GENE_SPEC = {
    "rsi_buy_threshold":    (20, 45),     # RSI below this → buy signal
    "rsi_sell_threshold":   (55, 80),     # RSI above this → sell signal
    "macd_weight":          (0.0, 2.0),   # Weight for MACD signal
    "bb_weight":            (0.0, 2.0),   # Weight for Bollinger signal
    "sma_weight":           (0.0, 2.0),   # Weight for SMA crossover
    "volume_weight":        (0.0, 2.0),   # Weight for volume signal
    "atr_weight":           (0.0, 2.0),   # Weight for ATR/volatility
    "entry_confidence":     (0.3, 0.8),   # Min confidence to enter
    "position_size":        (0.3, 0.95),  # Fraction of capital per trade
    "stop_loss_pct":        (0.02, 0.15), # Stop-loss percentage
    "take_profit_pct":      (0.03, 0.30), # Take-profit percentage
    "trailing_stop":        (0.01, 0.10), # Trailing stop distance
    "max_hold_bars":        (5, 100),     # Max bars to hold a position
    "cooldown_bars":        (1, 20),      # Bars between trades
}

N_GENES = len(GENE_SPEC)
GENE_NAMES = list(GENE_SPEC.keys())

# DEAP setup — create only if not already created
if not hasattr(creator, "EvoFitness"):
    creator.create("EvoFitness", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.EvoFitness)


def _decode_individual(ind: list) -> dict:
    """Map normalized [0,1] genes to actual parameter values."""
    params = {}
    for i, name in enumerate(GENE_NAMES):
        lo, hi = GENE_SPEC[name]
        params[name] = lo + ind[i] * (hi - lo)
    return params


def _encode_params(params: dict) -> list:
    """Map actual params back to [0,1] genes."""
    genes = []
    for name in GENE_NAMES:
        lo, hi = GENE_SPEC[name]
        val = params.get(name, (lo + hi) / 2)
        genes.append((val - lo) / (hi - lo + 1e-10))
    return genes


# ─── FITNESS EVALUATION ─────────────────────────────────────────
def _compute_signal(bar: dict, params: dict) -> tuple:
    """Compute composite signal strength from indicators using evolved weights."""
    ind = bar["indicators"]
    price = bar["price"]

    signals = []

    # RSI
    rsi = ind.get("rsi", 50)
    if rsi < params["rsi_buy_threshold"]:
        signals.append(("BUY", (params["rsi_buy_threshold"] - rsi) / params["rsi_buy_threshold"]))
    elif rsi > params["rsi_sell_threshold"]:
        signals.append(("SELL", (rsi - params["rsi_sell_threshold"]) / (100 - params["rsi_sell_threshold"] + 1e-10)))

    # MACD
    macd_hist = ind.get("macd_hist", 0)
    if macd_hist > 0:
        signals.append(("BUY", min(abs(macd_hist) / (price * 0.01 + 1e-10), 1.0) * params["macd_weight"]))
    elif macd_hist < 0:
        signals.append(("SELL", min(abs(macd_hist) / (price * 0.01 + 1e-10), 1.0) * params["macd_weight"]))

    # Bollinger Bands
    bb_upper = ind.get("bb_upper", price * 1.02)
    bb_lower = ind.get("bb_lower", price * 0.98)
    bb_pct = (price - bb_lower) / (bb_upper - bb_lower + 1e-10)
    if bb_pct < 0.2:
        signals.append(("BUY", (0.2 - bb_pct) * 5 * params["bb_weight"]))
    elif bb_pct > 0.8:
        signals.append(("SELL", (bb_pct - 0.8) * 5 * params["bb_weight"]))

    # SMA trend
    sma50 = ind.get("sma50", price)
    sma200 = ind.get("sma200", price)
    if price > sma50 > sma200:
        signals.append(("BUY", params["sma_weight"] * 0.5))
    elif price < sma50 < sma200:
        signals.append(("SELL", params["sma_weight"] * 0.5))

    # Volume
    vol_ratio = ind.get("vol_ratio", 1.0)
    if vol_ratio > 1.5:
        signals.append(("BUY" if macd_hist > 0 else "SELL", min(vol_ratio / 3, 1.0) * params["volume_weight"]))

    # Aggregate
    buy_str = sum(s for a, s in signals if a == "BUY")
    sell_str = sum(s for a, s in signals if a == "SELL")

    if buy_str > sell_str and buy_str > params["entry_confidence"]:
        return "BUY", buy_str
    elif sell_str > buy_str and sell_str > params["entry_confidence"]:
        return "SELL", sell_str
    return "HOLD", 0.0


def evaluate_strategy(individual, training_data):
    """Fitness function: simulate trading with evolved parameters."""
    try:
        return _evaluate_strategy_inner(individual, training_data)
    except (OverflowError, ValueError, FloatingPointError):
        return (0.0,)


def _evaluate_strategy_inner(individual, training_data):
    # Clamp genes to [0,1] before decoding (mutation can push outside)
    clamped = [max(0.0, min(1.0, g)) for g in individual]
    params = _decode_individual(clamped)

    balance = 10000.0
    position = 0.0
    entry_price = 0.0
    trades = 0
    wins = 0
    max_balance = balance
    max_drawdown = 0.0
    bars_in_trade = 0
    cooldown = 0
    returns = []

    cd_bars = max(1, min(int(params["cooldown_bars"]), 50))
    max_hold = max(5, min(int(params["max_hold_bars"]), 200))

    for bar in training_data:
        price = bar["price"]
        if price <= 0:
            continue

        if cooldown > 0:
            cooldown -= 1

        if position > 0:
            bars_in_trade += 1
            pnl_pct = (price - entry_price) / (entry_price + 1e-10)

            # Stop-loss
            if pnl_pct < -params["stop_loss_pct"]:
                balance += position * price
                trades += 1
                returns.append(pnl_pct)
                position = 0
                cooldown = cd_bars
                continue

            # Take-profit
            if pnl_pct > params["take_profit_pct"]:
                balance += position * price
                trades += 1
                wins += 1
                returns.append(pnl_pct)
                position = 0
                cooldown = cd_bars
                continue

            # Max hold
            if bars_in_trade >= max_hold:
                balance += position * price
                trades += 1
                if pnl_pct > 0:
                    wins += 1
                returns.append(pnl_pct)
                position = 0
                cooldown = cd_bars
                continue

        signal, strength = _compute_signal(bar, params)

        if signal == "BUY" and position == 0 and cooldown == 0:
            pos_size = max(0.01, min(params["position_size"], 0.99))
            invest = min(balance * pos_size, balance, 1e8)
            shares = int(min(invest / (price + 1e-10), 1e6))
            if shares > 0:
                position = shares
                entry_price = price
                balance -= shares * price
                bars_in_trade = 0

        elif signal == "SELL" and position > 0:
            pnl_pct = (price - entry_price) / entry_price
            balance += position * price
            trades += 1
            if pnl_pct > 0:
                wins += 1
            returns.append(pnl_pct)
            position = 0
            cooldown = cd_bars

        # Drawdown tracking
        portfolio = balance + position * price
        max_balance = max(max_balance, portfolio)
        dd = (max_balance - portfolio) / max_balance
        max_drawdown = max(max_drawdown, dd)

    # Close open position
    if position > 0:
        pnl_pct = (training_data[-1]["price"] - entry_price) / entry_price
        balance += position * training_data[-1]["price"]
        trades += 1
        if pnl_pct > 0:
            wins += 1
        returns.append(pnl_pct)

    # Fitness: Sharpe-like ratio + penalties
    profit_pct = (balance - 10000) / 10000 * 100
    win_rate = (wins / trades * 100) if trades > 0 else 0

    # Sharpe ratio approximation
    if returns:
        avg_ret = np.mean(returns)
        std_ret = np.std(returns) + 1e-10
        sharpe = avg_ret / std_ret * math.sqrt(252)
    else:
        sharpe = 0

    fitness = (
        profit_pct * 1.0
        + sharpe * 10.0
        - max_drawdown * 100
        + win_rate * 0.2
        + min(trades, 50) * 0.3  # reward more trades (up to limit)
    )

    # Cap fitness to prevent infinity
    fitness = max(-1000, min(fitness, 10000))
    if math.isnan(fitness) or math.isinf(fitness):
        fitness = 0.0

    return (fitness,)


# ─── OPTIMIZER ──────────────────────────────────────────────────
class EvoOptimizer:
    """Genetic algorithm optimizer for trading strategies."""

    def __init__(self):
        self.toolbox = base.Toolbox()
        self._setup_toolbox()
        self.best_params = None
        self.generation = 0
        self.stats = {"generations": [], "best_fitness": [], "avg_fitness": []}
        self._load()

    def _setup_toolbox(self):
        self.toolbox.register("attr_gene", random.random)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              self.toolbox.attr_gene, n=N_GENES)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.3)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def _load(self):
        if os.path.exists(BEST_PARAMS_PATH):
            with open(BEST_PARAMS_PATH) as f:
                self.best_params = json.load(f)
        if os.path.exists(STATS_PATH):
            with open(STATS_PATH) as f:
                self.stats = json.load(f)
            self.generation = len(self.stats.get("generations", []))

    def _save(self):
        if self.best_params:
            with open(BEST_PARAMS_PATH, "w") as f:
                json.dump(self.best_params, f, indent=2)
        with open(STATS_PATH, "w") as f:
            json.dump(self.stats, f, indent=2)

    def evolve(self, training_data: list, generations: int = 30, pop_size: int = 60) -> dict:
        """Run genetic algorithm to optimize strategy parameters."""
        t0 = time.time()

        if not training_data or len(training_data) < 50:
            return {"error": "Need at least 50 data points for evolution"}

        # Bind training data to fitness function
        self.toolbox.register("evaluate", evaluate_strategy, training_data=training_data)

        # Create population
        pop = self.toolbox.population(n=pop_size)

        # Seed with best known params if available
        if self.best_params:
            pop[0][:] = _encode_params(self.best_params)

        # Statistics
        stats_tracker = tools.Statistics(lambda ind: ind.fitness.values[0] if ind.fitness.valid else 0)
        stats_tracker.register("avg", np.mean)
        stats_tracker.register("max", np.max)

        hof = tools.HallOfFame(5)

        # Run evolution
        pop, logbook = algorithms.eaSimple(
            pop, self.toolbox,
            cxpb=0.7,    # crossover probability
            mutpb=0.3,   # mutation probability
            ngen=generations,
            stats=stats_tracker,
            halloffame=hof,
            verbose=False,
        )

        # Clamp genes to [0, 1]
        for ind in hof:
            for i in range(len(ind)):
                ind[i] = max(0.0, min(1.0, ind[i]))

        # Extract best
        best_ind = hof[0]
        self.best_params = _decode_individual(best_ind)
        best_fitness = best_ind.fitness.values[0]

        # Update stats
        for record in logbook:
            if isinstance(record, dict) and "gen" in record:
                self.generation += 1
                self.stats["generations"].append(self.generation)
                self.stats["best_fitness"].append(round(record.get("max", 0), 2))
                self.stats["avg_fitness"].append(round(record.get("avg", 0), 2))

        self._save()

        return {
            "generations_run": generations,
            "total_generations": self.generation,
            "population_size": pop_size,
            "best_fitness": round(best_fitness, 2),
            "best_params": {k: round(v, 4) for k, v in self.best_params.items()},
            "hall_of_fame_size": len(hof),
            "elapsed_s": round(time.time() - t0, 1),
        }

    def get_signal(self, bar: dict) -> dict:
        """Use best evolved parameters to generate a trade signal."""
        if not self.best_params:
            return {"action": "HOLD", "confidence": 0, "reason": "No evolved parameters — run evolve() first"}

        signal, strength = _compute_signal(bar, self.best_params)
        return {
            "action": signal,
            "confidence": round(min(strength * 100, 100), 1),
            "model": "EvoStrategy",
            "generation": self.generation,
            "params": {k: round(v, 4) for k, v in self.best_params.items()},
        }

    def get_status(self) -> dict:
        return {
            "model": "EvoStrategy (DEAP)",
            "generation": self.generation,
            "has_evolved": self.best_params is not None,
            "best_params": {k: round(v, 4) for k, v in self.best_params.items()} if self.best_params else None,
            "stats": {
                "total_generations": len(self.stats.get("generations", [])),
                "recent_best": self.stats.get("best_fitness", [])[-5:],
                "recent_avg": self.stats.get("avg_fitness", [])[-5:],
            }
        }


# Singleton
_optimizer = EvoOptimizer()
def get_evo_optimizer() -> EvoOptimizer:
    return _optimizer
