"""
PyTorch Price Prediction Network
==================================
LSTM + Attention model for short-term price direction prediction.

Inputs: sliding window of technical indicators (20 bars × 14 features)
Outputs: predicted price direction (UP/DOWN/FLAT) + magnitude

Training: online learning from live market data + periodic retraining.
"""

import warnings; warnings.filterwarnings("ignore")
import os, json, time, math
import numpy as np
from datetime import datetime
from typing import Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

DATA_DIR = os.path.join(os.path.dirname(__file__), "torch_data")
os.makedirs(DATA_DIR, exist_ok=True)

MODEL_PATH = os.path.join(DATA_DIR, "price_predictor.pt")
STATS_PATH = os.path.join(DATA_DIR, "training_stats.json")
SCALER_PATH = os.path.join(DATA_DIR, "scaler_params.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── FEATURE ENGINEERING ────────────────────────────────────────
N_FEATURES = 14
WINDOW_SIZE = 20  # bars lookback


def extract_features(indicators: dict, price: float, price_history: dict = None) -> list:
    """Convert indicator dict to feature vector (same 14 as NEAT)."""
    bb_upper = indicators.get("bb_upper", price * 1.02)
    bb_lower = indicators.get("bb_lower", price * 0.98)
    bb_range = bb_upper - bb_lower + 1e-10

    return [
        (indicators.get("rsi", 50) - 50) / 50,
        indicators.get("macd", 0) / (price * 0.01 + 1e-10),
        indicators.get("macd_signal", 0) / (price * 0.01 + 1e-10),
        indicators.get("macd_hist", 0) / (price * 0.01 + 1e-10),
        (price - bb_lower) / bb_range * 2 - 1,
        (price / (indicators.get("sma50", price) + 1e-10)) - 1,
        (price / (indicators.get("sma200", price) + 1e-10)) - 1,
        indicators.get("atr", 0) / (price + 1e-10),
        min(indicators.get("vol_ratio", 1.0), 5.0) / 5.0,
        (price_history or {}).get("chg_1d", 0) / 10.0,
        (price_history or {}).get("chg_5d", 0) / 20.0,
        (price_history or {}).get("chg_20d", 0) / 30.0,
        datetime.now().hour / 24.0,
        datetime.now().weekday() / 6.0,
    ]


# ─── DATASET ────────────────────────────────────────────────────
class PriceDataset(Dataset):
    """Sliding window dataset for price prediction."""

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        """
        features: (N, WINDOW_SIZE, N_FEATURES) — windowed sequences
        targets:  (N, 3) — [direction_up, direction_down, magnitude]
        """
        self.X = torch.FloatTensor(features)
        self.y = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_sequences(all_features: list, all_prices: list, lookahead: int = 5):
    """
    Build windowed training sequences.
    all_features: list of feature vectors (one per bar)
    all_prices: list of prices (one per bar)
    lookahead: bars to look ahead for target
    """
    X, y = [], []
    for i in range(WINDOW_SIZE, len(all_features) - lookahead):
        window = all_features[i - WINDOW_SIZE:i]
        X.append(window)

        # Target: price change over next `lookahead` bars
        current = all_prices[i]
        future = all_prices[i + lookahead]
        pct_change = (future - current) / (current + 1e-10) * 100

        # Direction: [up_prob, down_prob, magnitude]
        if pct_change > 0.5:
            direction = [1.0, 0.0]
        elif pct_change < -0.5:
            direction = [0.0, 1.0]
        else:
            direction = [0.5, 0.5]  # flat

        y.append(direction + [min(abs(pct_change), 20.0) / 20.0])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ─── MODEL ──────────────────────────────────────────────────────
class Attention(nn.Module):
    """Simple self-attention over time steps."""
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden)
        weights = torch.softmax(self.attn(lstm_out), dim=1)  # (batch, seq_len, 1)
        context = (lstm_out * weights).sum(dim=1)  # (batch, hidden)
        return context, weights.squeeze(-1)


class PricePredictor(nn.Module):
    """LSTM + Attention for price direction prediction."""

    def __init__(self, n_features=N_FEATURES, hidden_size=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.attention = Attention(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3),  # [up_prob, down_prob, magnitude]
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        out = self.fc(context)
        # Sigmoid on direction probs and magnitude
        out = torch.sigmoid(out)
        return out


# ─── TRAINER ────────────────────────────────────────────────────
class TorchPredictor:
    """Manages PyTorch model training and inference."""

    def __init__(self):
        self.model = PricePredictor().to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5
        )
        self.epoch = 0
        self.stats = {"epochs": [], "train_loss": [], "val_loss": [], "accuracy": []}
        self._load()

    def _load(self):
        """Load saved model."""
        if os.path.exists(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.epoch = checkpoint.get("epoch", 0)
        if os.path.exists(STATS_PATH):
            with open(STATS_PATH) as f:
                self.stats = json.load(f)

    def _save(self):
        """Save model checkpoint."""
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch,
        }, MODEL_PATH)
        with open(STATS_PATH, "w") as f:
            json.dump(self.stats, f, indent=2)

    def train(self, training_data: list, epochs: int = 50, val_split: float = 0.2) -> dict:
        """
        Train on historical data.
        training_data: list of dicts with 'indicators', 'price', 'price_history' keys.
        """
        t0 = time.time()

        if len(training_data) < WINDOW_SIZE + 10:
            return {"error": f"Need at least {WINDOW_SIZE + 10} data points"}

        # Extract features and prices
        all_features = []
        all_prices = []
        for bar in training_data:
            feat = extract_features(bar["indicators"], bar["price"], bar.get("price_history"))
            all_features.append(feat)
            all_prices.append(bar["price"])

        # Build sequences
        X, y = build_sequences(all_features, all_prices)
        if len(X) < 20:
            return {"error": "Not enough data after windowing"}

        # Train/val split
        split = int(len(X) * (1 - val_split))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        train_ds = PriceDataset(X_train, y_train)
        val_ds = PriceDataset(X_val, y_val)
        train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=64)

        best_val_loss = float("inf")
        patience_counter = 0

        self.model.train()
        for ep in range(epochs):
            # Training
            train_losses = []
            for xb, yb in train_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                self.optimizer.zero_grad()
                pred = self.model(xb)
                loss = self.criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_losses.append(loss.item())

            # Validation
            self.model.eval()
            val_losses = []
            correct = 0
            total = 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    pred = self.model(xb)
                    val_losses.append(self.criterion(pred, yb).item())

                    # Direction accuracy
                    pred_dir = (pred[:, 0] > pred[:, 1]).float()
                    true_dir = (yb[:, 0] > yb[:, 1]).float()
                    correct += (pred_dir == true_dir).sum().item()
                    total += len(yb)
            self.model.train()

            avg_train = np.mean(train_losses)
            avg_val = np.mean(val_losses) if val_losses else avg_train
            accuracy = correct / total * 100 if total > 0 else 0

            self.epoch += 1
            self.stats["epochs"].append(self.epoch)
            self.stats["train_loss"].append(round(avg_train, 6))
            self.stats["val_loss"].append(round(avg_val, 6))
            self.stats["accuracy"].append(round(accuracy, 1))

            self.scheduler.step(avg_val)

            # Early stopping
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_counter = 0
                self._save()
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    break

        self._save()

        return {
            "epochs_run": ep + 1,
            "total_epochs": self.epoch,
            "final_train_loss": round(avg_train, 6),
            "final_val_loss": round(avg_val, 6),
            "best_val_loss": round(best_val_loss, 6),
            "direction_accuracy": round(accuracy, 1),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "device": str(DEVICE),
            "elapsed_s": round(time.time() - t0, 1),
        }

    def predict(self, indicators: dict, recent_bars: list = None, price_history: dict = None) -> dict:
        """
        Predict price direction from current indicators + recent bar history.
        recent_bars: list of recent bar dicts (need at least WINDOW_SIZE).
        """
        if self.epoch == 0:
            return {"action": "HOLD", "confidence": 0, "reason": "No trained model — run train() first"}

        self.model.eval()

        # Build feature window
        if recent_bars and len(recent_bars) >= WINDOW_SIZE:
            window = []
            for bar in recent_bars[-WINDOW_SIZE:]:
                feat = extract_features(bar["indicators"], bar["price"], bar.get("price_history"))
                window.append(feat)
        else:
            # Fallback: repeat current features
            feat = extract_features(indicators, indicators.get("price", 1.0), price_history)
            window = [feat] * WINDOW_SIZE

        x = torch.FloatTensor([window]).to(DEVICE)

        with torch.no_grad():
            out = self.model(x)[0].cpu().numpy()

        up_prob = float(out[0])
        down_prob = float(out[1])
        magnitude = float(out[2]) * 20.0  # de-normalize

        if up_prob > down_prob + 0.1:
            action = "BUY"
            confidence = up_prob * 100
        elif down_prob > up_prob + 0.1:
            action = "SELL"
            confidence = down_prob * 100
        else:
            action = "HOLD"
            confidence = (1 - abs(up_prob - down_prob)) * 50

        return {
            "action": action,
            "confidence": round(confidence, 1),
            "predicted_move_pct": round(magnitude if action == "BUY" else -magnitude if action == "SELL" else 0, 2),
            "probabilities": {"UP": round(up_prob * 100, 1), "DOWN": round(down_prob * 100, 1)},
            "magnitude": round(magnitude, 2),
            "epoch": self.epoch,
            "model": "PyTorch-LSTM",
            "device": str(DEVICE),
        }

    def get_status(self) -> dict:
        return {
            "model": "PyTorch-LSTM+Attention",
            "device": str(DEVICE),
            "epoch": self.epoch,
            "has_trained": self.epoch > 0,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "stats": {
                "total_epochs": len(self.stats.get("epochs", [])),
                "recent_loss": self.stats.get("val_loss", [])[-5:] if self.stats.get("val_loss") else [],
                "recent_accuracy": self.stats.get("accuracy", [])[-5:] if self.stats.get("accuracy") else [],
            }
        }


# Singleton
_predictor = TorchPredictor()
def get_torch_predictor() -> TorchPredictor:
    return _predictor
