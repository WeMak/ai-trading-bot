"""
NEAT Neuroevolution Trading Brain
===================================
Uses NEAT (NeuroEvolution of Augmenting Topologies) to evolve
neural network trading strategies through natural selection.

Inputs (14 features):
  RSI, MACD, MACD_signal, MACD_hist, BB_%B, SMA50_ratio,
  SMA200_ratio, ATR_pct, vol_ratio, price_chg_1d, price_chg_5d,
  price_chg_20d, hour_of_day (normalized), day_of_week (normalized)

Outputs (3):
  buy_signal, sell_signal, hold_signal  (softmax → action)

Fitness = cumulative P&L on historical data with drawdown penalty.
"""

import warnings; warnings.filterwarnings("ignore")
import os, json, pickle, time, math
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import neat

DATA_DIR = os.path.join(os.path.dirname(__file__), "neat_data")
os.makedirs(DATA_DIR, exist_ok=True)

NEAT_CONFIG_PATH = os.path.join(DATA_DIR, "neat_config.txt")
BEST_GENOME_PATH = os.path.join(DATA_DIR, "best_genome.pkl")
POPULATION_PATH = os.path.join(DATA_DIR, "population.pkl")
STATS_PATH = os.path.join(DATA_DIR, "evolution_stats.json")

# Number of input features
N_INPUTS = 14
N_OUTPUTS = 3  # BUY, SELL, HOLD


def _write_neat_config():
    """Write NEAT configuration file."""
    config = f"""[NEAT]
fitness_criterion     = max
fitness_threshold     = 10000
no_fitness_termination = False
pop_size              = 50
reset_on_extinction   = True

[DefaultGenome]
num_inputs              = {N_INPUTS}
num_hidden              = 0
num_outputs             = {N_OUTPUTS}
initial_connection      = full_direct
feed_forward            = True
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5
conn_add_prob           = 0.5
conn_delete_prob        = 0.3
node_add_prob           = 0.3
node_delete_prob        = 0.2
activation_default      = tanh
activation_mutate_rate  = 0.1
activation_options      = tanh sigmoid relu
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_init_type          = gaussian
bias_max_value          = 5.0
bias_min_value          = -5.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 5.0
response_min_value      = -5.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_init_type        = gaussian
weight_max_value        = 5.0
weight_min_value        = -5.0
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1
enabled_default         = True
enabled_mutate_rate     = 0.01

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 15
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
min_species_size   = 2
"""
    with open(NEAT_CONFIG_PATH, "w") as f:
        f.write(config)
    return NEAT_CONFIG_PATH


def _prepare_features(indicators: dict, price_history: dict = None) -> np.ndarray:
    """Convert indicator dict to normalized feature vector."""
    price = indicators.get("price", 1.0)
    bb_upper = indicators.get("bb_upper", price * 1.02)
    bb_lower = indicators.get("bb_lower", price * 0.98)
    bb_range = bb_upper - bb_lower + 1e-10

    features = [
        # Momentum
        (indicators.get("rsi", 50) - 50) / 50,  # -1 to 1
        indicators.get("macd", 0) / (price * 0.01 + 1e-10),  # normalized
        indicators.get("macd_signal", 0) / (price * 0.01 + 1e-10),
        indicators.get("macd_hist", 0) / (price * 0.01 + 1e-10),
        # Bollinger %B
        (price - bb_lower) / bb_range * 2 - 1,  # -1 to 1
        # Trend
        (price / (indicators.get("sma50", price) + 1e-10)) - 1,
        (price / (indicators.get("sma200", price) + 1e-10)) - 1,
        # Volatility
        indicators.get("atr", 0) / (price + 1e-10),
        # Volume
        min(indicators.get("vol_ratio", 1.0), 5.0) / 5.0,
        # Price changes (from price_history or indicators)
        (price_history or {}).get("chg_1d", 0) / 10.0,
        (price_history or {}).get("chg_5d", 0) / 20.0,
        (price_history or {}).get("chg_20d", 0) / 30.0,
        # Time features
        datetime.now().hour / 24.0,
        datetime.now().weekday() / 6.0,
    ]
    return np.array(features, dtype=np.float32)


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-10)


class NEATTrader:
    """NEAT-evolved neural network trader."""

    def __init__(self):
        self.config = None
        self.population = None
        self.best_genome = None
        self.best_net = None
        self.generation = 0
        self.stats = {"generations": [], "best_fitness": [], "avg_fitness": []}
        self._load()

    def _load(self):
        """Load saved state."""
        if not os.path.exists(NEAT_CONFIG_PATH):
            _write_neat_config()
        self.config = neat.Config(
            neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation,
            NEAT_CONFIG_PATH
        )
        if os.path.exists(BEST_GENOME_PATH):
            with open(BEST_GENOME_PATH, "rb") as f:
                self.best_genome = pickle.load(f)
            self.best_net = neat.nn.FeedForwardNetwork.create(self.best_genome, self.config)
        if os.path.exists(STATS_PATH):
            with open(STATS_PATH) as f:
                self.stats = json.load(f)
            self.generation = len(self.stats.get("generations", []))

    def _save_best(self, genome):
        self.best_genome = genome
        self.best_net = neat.nn.FeedForwardNetwork.create(genome, self.config)
        with open(BEST_GENOME_PATH, "wb") as f:
            pickle.dump(genome, f)

    def _save_stats(self):
        with open(STATS_PATH, "w") as f:
            json.dump(self.stats, f, indent=2)

    def predict(self, indicators: dict, price_history: dict = None) -> dict:
        """Use best evolved network to make a trade decision."""
        if self.best_net is None:
            return {"action": "HOLD", "confidence": 0, "reason": "No trained genome yet — run evolve() first"}

        features = _prepare_features(indicators, price_history)
        output = self.best_net.activate(features.tolist())
        probs = _softmax(np.array(output))

        actions = ["BUY", "SELL", "HOLD"]
        idx = np.argmax(probs)
        action = actions[idx]
        confidence = float(probs[idx]) * 100

        return {
            "action": action,
            "confidence": round(confidence, 1),
            "probabilities": {a: round(float(p) * 100, 1) for a, p in zip(actions, probs)},
            "generation": self.generation,
            "model": "NEAT",
            "genome_fitness": round(float(self.best_genome.fitness), 2) if self.best_genome and self.best_genome.fitness else None,
        }

    def evolve(self, training_data: list, generations: int = 20) -> dict:
        """
        Evolve trading strategies on historical data.
        training_data: list of dicts with 'indicators', 'price', 'future_prices' keys.
        """
        t0 = time.time()

        if not training_data or len(training_data) < 30:
            return {"error": "Need at least 30 data points for training"}

        def eval_genomes(genomes, config):
            for genome_id, genome in genomes:
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                fitness = self._simulate_trading(net, training_data)
                genome.fitness = fitness

        # Create or continue population
        if self.population is None:
            self.population = neat.Population(self.config)

        # Run evolution
        best = None
        for gen in range(generations):
            self.population.run(eval_genomes, 1)

            # Track best
            best_in_gen = max(
                [g for g in self.population.population.values()],
                key=lambda g: g.fitness if g.fitness is not None else -999
            )

            avg_fit = np.mean([g.fitness for g in self.population.population.values() if g.fitness is not None])

            self.generation += 1
            self.stats["generations"].append(self.generation)
            self.stats["best_fitness"].append(round(float(best_in_gen.fitness or 0), 2))
            self.stats["avg_fitness"].append(round(float(avg_fit), 2))

            if best is None or (best_in_gen.fitness or 0) > (best.fitness or 0):
                best = best_in_gen

        # Save best
        if best:
            self._save_best(best)
        self._save_stats()

        return {
            "generations_run": generations,
            "total_generations": self.generation,
            "best_fitness": round(float(best.fitness or 0), 2) if best else 0,
            "avg_fitness": round(float(avg_fit), 2),
            "population_size": len(self.population.population),
            "species": len(self.population.species.species),
            "elapsed_s": round(time.time() - t0, 1),
            "genome_nodes": len(best.nodes) if best else 0,
            "genome_connections": len(best.connections) if best else 0,
        }

    def _simulate_trading(self, net, data: list) -> float:
        """Simulate trading with a genome on historical data. Returns fitness."""
        balance = 10000.0
        position = 0.0  # shares held
        entry_price = 0.0
        trades = 0
        wins = 0
        max_balance = balance
        max_drawdown = 0.0

        for i, bar in enumerate(data):
            features = _prepare_features(bar["indicators"], bar.get("price_history"))
            output = net.activate(features.tolist())
            probs = _softmax(np.array(output))
            action_idx = np.argmax(probs)
            confidence = probs[action_idx]

            price = bar["price"]

            if action_idx == 0 and position == 0 and confidence > 0.4:
                # BUY
                shares = int(balance * 0.95 / price)
                if shares > 0:
                    position = shares
                    entry_price = price
                    balance -= shares * price

            elif action_idx == 1 and position > 0 and confidence > 0.4:
                # SELL
                pnl = (price - entry_price) * position
                balance += position * price
                trades += 1
                if pnl > 0:
                    wins += 1
                position = 0

            # Track drawdown
            portfolio = balance + position * price
            max_balance = max(max_balance, portfolio)
            dd = (max_balance - portfolio) / max_balance
            max_drawdown = max(max_drawdown, dd)

        # Close any open position
        if position > 0:
            balance += position * data[-1]["price"]

        # Fitness: profit with drawdown penalty
        profit_pct = (balance - 10000) / 10000 * 100
        win_rate = (wins / trades * 100) if trades > 0 else 0
        fitness = profit_pct - (max_drawdown * 50) + (win_rate * 0.1) + (min(trades, 50) * 0.2)

        return fitness

    def get_status(self) -> dict:
        return {
            "model": "NEAT",
            "generation": self.generation,
            "has_trained_genome": self.best_genome is not None,
            "best_fitness": round(float(self.best_genome.fitness or 0), 2) if self.best_genome and self.best_genome.fitness else None,
            "genome_nodes": len(self.best_genome.nodes) if self.best_genome else 0,
            "genome_connections": len(self.best_genome.connections) if self.best_genome else 0,
            "stats": {
                "total_generations": len(self.stats.get("generations", [])),
                "recent_best": self.stats.get("best_fitness", [])[-5:] if self.stats.get("best_fitness") else [],
                "recent_avg": self.stats.get("avg_fitness", [])[-5:] if self.stats.get("avg_fitness") else [],
            }
        }


# Singleton
_neat_trader = NEATTrader()
def get_neat_trader() -> NEATTrader:
    return _neat_trader
