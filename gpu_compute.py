"""
GPU-Accelerated Compute Engine
================================
Hybrid CPU/GPU acceleration strategy:

INDICATORS (CPU — vectorized NumPy):
  - SMA: O(1) per element via cumsum (no loops)
  - Bollinger: stride_tricks sliding windows (no loops)
  - RSI/EMA/MACD/ATR: sequential EMA smoothing stays on CPU
  - Result: ~1.5ms per ticker — GPU transfer overhead exceeds compute

NEURAL NETWORK TRAINING (GPU — PyTorch CUDA):
  - LSTM training with mixed precision (AMP) — 3-5x faster
  - Batch size 256 with pin_memory + non_blocking transfers
  - TF32 enabled for RTX 50xx Blackwell architecture
  - cuDNN benchmark mode for auto-tuned convolutions

Usage:
    from gpu_compute import gpu, compute_indicators_batch
    print(gpu.status())
"""

import os, time, logging
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger("gpu_compute")

# ─── GPU DEVICE SETUP ───────────────────────────────────────────
import torch

class GPUEngine:
    """Central GPU resource manager."""

    def __init__(self):
        self.has_cuda = torch.cuda.is_available()
        if self.has_cuda:
            self.device = torch.device("cuda:0")
            self.gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            self.vram_total = props.total_memory / 1024**3
            self.compute_cap = torch.cuda.get_device_capability(0)
            # Enable TF32 for faster float32 ops on Ampere+ (RTX 30xx, 40xx, 50xx)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            # Enable flash attention if available
            try:
                torch.backends.cuda.enable_flash_sdp(True)
            except Exception:
                pass
            log.info(f"GPU: {self.gpu_name} | VRAM: {self.vram_total:.1f}GB | Compute: {self.compute_cap}")
        else:
            self.device = torch.device("cpu")
            self.gpu_name = "CPU only"
            self.vram_total = 0
            self.compute_cap = (0, 0)

        self._stats = {
            "indicator_ops": 0,
            "nn_training_ops": 0,
            "total_tickers_processed": 0,
            "indicator_time_ms": 0,
            "started": time.time(),
        }

    def status(self) -> dict:
        """Get GPU engine status."""
        vram_used = 0
        vram_free = 0
        gpu_util = 0
        if self.has_cuda:
            vram_used = torch.cuda.memory_allocated(0) / 1024**3
            vram_reserved = torch.cuda.memory_reserved(0) / 1024**3
            vram_free = self.vram_total - vram_reserved
        return {
            "gpu_available": self.has_cuda,
            "gpu_name": self.gpu_name,
            "vram_total_gb": round(self.vram_total, 1),
            "vram_used_gb": round(vram_used, 2),
            "vram_free_gb": round(vram_free, 1),
            "compute_capability": f"{self.compute_cap[0]}.{self.compute_cap[1]}",
            "device": str(self.device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda or "N/A",
            "nn_device": "CUDA (GPU)" if self.has_cuda else "CPU",
            "indicator_device": "CPU (vectorized NumPy)",
            "strategy": "GPU for NN training | CPU for indicators",
            "stats": self._stats.copy(),
        }


# Singleton
gpu = GPUEngine()


# ─── VECTORIZED INDICATORS (CPU — optimized NumPy) ──────────────
# Benchmark result: 1.5ms/ticker on CPU vs 184ms on GPU
# GPU overhead (transfer + kernel launch) kills perf for small arrays
# These are all loop-free where possible, using cumsum + stride tricks

def sma_gpu(closes: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average — O(n) via cumsum, zero loops."""
    n = len(closes)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return out
    cs = np.cumsum(closes.astype(np.float64))
    out[period - 1] = cs[period - 1] / period
    if n > period:
        out[period:] = (cs[period:] - cs[:-period]) / period
    return out


def bollinger_gpu(closes: np.ndarray, period: int = 20, nstd: float = 2.0):
    """Bollinger Bands — vectorized with sliding_window_view (zero loops)."""
    n = len(closes)
    mid = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    if n < period:
        return upper, mid, lower

    windows = sliding_window_view(closes.astype(np.float64), period)
    m = np.mean(windows, axis=1)
    s = np.std(windows, axis=1)
    mid[period - 1:] = m
    upper[period - 1:] = m + nstd * s
    lower[period - 1:] = m - nstd * s
    return upper, mid, lower


def rsi_gpu(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI — vectorized delta + sequential EMA (unavoidable)."""
    n = len(closes)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out

    delta = np.diff(closes.astype(np.float64))
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    alpha = 1.0 / period
    ag = np.full(len(gain), np.nan)
    al = np.full(len(gain), np.nan)
    ag[period - 1] = np.mean(gain[:period])
    al[period - 1] = np.mean(loss[:period])
    for i in range(period, len(gain)):
        ag[i] = alpha * gain[i] + (1 - alpha) * ag[i - 1]
        al[i] = alpha * loss[i] + (1 - alpha) * al[i - 1]
    rs = ag / (al + 1e-10)
    out[1:] = 100.0 - 100.0 / (1.0 + rs)
    return out


def ema_gpu(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average — sequential (inherent data dependency)."""
    out = np.full(len(arr), np.nan)
    valid = np.where(~np.isnan(arr))[0]
    if len(valid) < period:
        return out
    s = valid[0]
    if s + period - 1 >= len(arr):
        return out
    out[s + period - 1] = float(np.nanmean(arr[s:s + period]))
    alpha = 2.0 / (period + 1)
    for i in range(s + period, len(arr)):
        if not np.isnan(arr[i]):
            out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
        else:
            out[i] = out[i - 1]
    return out


def macd_gpu(closes: np.ndarray, fast=12, slow=26, sig_p=9):
    """MACD(12,26,9) — triple EMA."""
    ef = ema_gpu(closes, fast)
    es = ema_gpu(closes, slow)
    line = ef - es
    signal = ema_gpu(np.where(np.isnan(line), np.nan, line), sig_p)
    hist = line - signal
    return line, signal, hist


def atr_gpu(highs, lows, closes, period=14) -> np.ndarray:
    """Average True Range — vectorized TR, sequential EMA smoothing."""
    n = len(closes)
    out = np.full(n, np.nan)
    if n < 2:
        return out
    h = highs.astype(np.float64)
    l = lows.astype(np.float64)
    c = closes.astype(np.float64)
    tr = np.maximum(h[1:] - l[1:],
         np.maximum(np.abs(h[1:] - c[:-1]),
                    np.abs(l[1:] - c[:-1])))
    if len(tr) < period:
        return out
    out[period] = np.mean(tr[:period])
    alpha = 1.0 / period
    for i in range(period + 1, n):
        out[i] = alpha * tr[i - 1] + (1 - alpha) * out[i - 1]
    return out


# ─── BATCH INDICATOR PROCESSING ─────────────────────────────────

def compute_all_indicators(close: np.ndarray, high: np.ndarray,
                           low: np.ndarray, volume: np.ndarray) -> dict:
    """Compute all indicators for a single ticker (~1.5ms on CPU)."""
    t0 = time.perf_counter()

    rsi_arr = rsi_gpu(close)
    macd_l, macd_s, macd_h = macd_gpu(close)
    bb_u, bb_m, bb_l = bollinger_gpu(close)
    sma50_arr = sma_gpu(close, 50)
    sma200_arr = sma_gpu(close, 200)
    atr_arr = atr_gpu(high, low, close)

    def _s(arr):
        v = float(arr[-1]) if len(arr) > 0 else 0.0
        return v if not np.isnan(v) else 0.0

    avg_vol = float(np.mean(volume[-20:])) if len(volume) >= 20 else float(np.mean(volume))
    vol_ratio = float(volume[-1]) / (avg_vol + 1e-10)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    gpu._stats["indicator_ops"] += 1
    gpu._stats["indicator_time_ms"] += elapsed_ms

    return {
        "price": float(close[-1]),
        "rsi": _s(rsi_arr),
        "macd": _s(macd_l),
        "macd_signal": _s(macd_s),
        "macd_hist": _s(macd_h),
        "bb_upper": _s(bb_u),
        "bb_middle": _s(bb_m),
        "bb_lower": _s(bb_l),
        "sma50": _s(sma50_arr),
        "sma200": _s(sma200_arr),
        "atr": _s(atr_arr),
        "vol_ratio": round(vol_ratio, 2),
        "compute_ms": round(elapsed_ms, 2),
    }


def compute_indicators_batch(tickers_data: dict, workers: int = 16) -> dict:
    """
    Batch compute indicators for multiple tickers in parallel.
    Uses CPU thread pool for indicators (fast) — GPU reserved for NN training.

    tickers_data: {ticker: DataFrame} from data_fetcher.batch_fetch()
    Returns: {ticker: {indicators, price_history}}
    """
    t0 = time.perf_counter()
    results = {}

    def _process_one(item):
        ticker, df = item
        try:
            if df is None or df.empty or len(df) < 30:
                return ticker, None
            close = df["Close"].values.astype(float)
            high = df["High"].values.astype(float)
            low = df["Low"].values.astype(float)
            volume = df["Volume"].values.astype(float)
            indicators = compute_all_indicators(close, high, low, volume)

            price_history = {}
            if len(close) >= 2:
                price_history["chg_1d"] = round((close[-1] - close[-2]) / (close[-2] + 1e-10) * 100, 2)
            if len(close) >= 6:
                price_history["chg_5d"] = round((close[-1] - close[-6]) / (close[-6] + 1e-10) * 100, 2)
            if len(close) >= 21:
                price_history["chg_20d"] = round((close[-1] - close[-21]) / (close[-21] + 1e-10) * 100, 2)

            return ticker, {"indicators": indicators, "price_history": price_history}
        except Exception:
            return ticker, None

    items = list(tickers_data.items())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for ticker, result in pool.map(_process_one, items):
            if result is not None:
                results[ticker] = result

    gpu._stats["total_tickers_processed"] += len(results)
    elapsed = (time.perf_counter() - t0) * 1000
    log.info(f"Batch: {len(results)} tickers in {elapsed:.0f}ms")
    return results


# ─── GPU-ACCELERATED NEURAL NETWORK UTILITIES ───────────────────

def prepare_training_tensors(all_features: list, all_prices: list,
                             window_size: int = 20, lookahead: int = 5,
                             val_split: float = 0.2):
    """
    Prepare training tensors pre-loaded on GPU for faster training.
    Eliminates per-batch CPU→GPU transfer overhead.
    """
    from torch_predictor import build_sequences

    X, y = build_sequences(all_features, all_prices, lookahead)
    if len(X) < 20:
        return None

    split = int(len(X) * (1 - val_split))
    X_train = torch.FloatTensor(X[:split]).to(gpu.device, non_blocking=True)
    y_train = torch.FloatTensor(y[:split]).to(gpu.device, non_blocking=True)
    X_val = torch.FloatTensor(X[split:]).to(gpu.device, non_blocking=True)
    y_val = torch.FloatTensor(y[split:]).to(gpu.device, non_blocking=True)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "device": str(gpu.device),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
    }


# ─── BENCHMARK ──────────────────────────────────────────────────

def benchmark(n_tickers: int = 100, n_bars: int = 500) -> dict:
    """Benchmark indicator computation + GPU NN forward pass."""
    results = {}

    # Generate synthetic data
    np.random.seed(42)
    test_data = {}
    for i in range(n_tickers):
        base = 100 + np.random.randn() * 50
        close = base + np.cumsum(np.random.randn(n_bars) * 0.5)
        close = np.maximum(close, 1.0)
        high = close + np.abs(np.random.randn(n_bars) * 0.3)
        low = close - np.abs(np.random.randn(n_bars) * 0.3)
        volume = np.random.randint(100000, 10000000, n_bars).astype(float)
        test_data[f"TEST{i}"] = {"close": close, "high": high, "low": low, "volume": volume}

    # Indicator benchmark (CPU vectorized)
    t0 = time.perf_counter()
    for sym, d in test_data.items():
        compute_all_indicators(d["close"], d["high"], d["low"], d["volume"])
    ind_ms = (time.perf_counter() - t0) * 1000
    results["indicators_ms"] = round(ind_ms, 1)
    results["indicators_per_ticker_ms"] = round(ind_ms / n_tickers, 2)

    # GPU NN forward pass benchmark
    if gpu.has_cuda:
        from torch_predictor import PricePredictor, N_FEATURES, WINDOW_SIZE
        model = PricePredictor().to(gpu.device).eval()
        # Batch of n_tickers inference
        x = torch.randn(n_tickers, WINDOW_SIZE, N_FEATURES, device=gpu.device)
        # Warmup
        with torch.no_grad(), torch.amp.autocast("cuda"):
            _ = model(x)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for _ in range(10):
                _ = model(x)
        torch.cuda.synchronize()
        nn_ms = (time.perf_counter() - t0) * 1000 / 10
        results["nn_forward_ms"] = round(nn_ms, 1)
        results["nn_per_ticker_ms"] = round(nn_ms / n_tickers, 3)
        results["nn_device"] = str(gpu.device)

        # GPU matmul benchmark
        a = torch.randn(2048, 2048, device=gpu.device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            _ = torch.mm(a, a)
        torch.cuda.synchronize()
        matmul_ms = (time.perf_counter() - t0) * 1000 / 100
        results["gpu_matmul_2048x2048_ms"] = round(matmul_ms, 2)

    results["n_tickers"] = n_tickers
    results["n_bars"] = n_bars
    results["gpu"] = gpu.gpu_name
    results["strategy"] = "CPU vectorized indicators | GPU NN training (AMP + TF32)"

    return results
