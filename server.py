"""
NEAT Trading Bot — FastAPI Server v2.0
=======================================
Localhost-only. JWT auth. WebSocket updates.
Full pipeline: sectors + crypto + options + news + Ollama AI.
"""

import warnings, os, json, asyncio, math
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")

_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path)

import bcrypt as _bcrypt
from fastapi import FastAPI, Depends, HTTPException, status, WebSocket, WebSocketDisconnect, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from jose import JWTError, jwt
import numpy as np
import yfinance as yf

# ─── CONFIG ──────────────────────────────────────────────────────
SECRET_KEY = os.environ.get("SCANNER_SECRET", "change-me-32-bytes")
ALGORITHM = "HS256"
TOKEN_EXPIRE = 60

def _hash_pw(plain: str) -> bytes:
    return _bcrypt.hashpw(plain[:72].encode(), _bcrypt.gensalt())

def verify_password(plain: str, hashed: bytes) -> bool:
    return _bcrypt.checkpw(plain[:72].encode(), hashed)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

_default_pw = os.environ.get("SCANNER_PASS", "ChangeMe123!")
_default_user = os.environ.get("SCANNER_USER", "trader")
USERS_DB = {
    _default_user: {
        "username": _default_user,
        "hashed_password": _hash_pw(_default_pw),
        "role": "analyst",
    }
}

_pool = ThreadPoolExecutor(max_workers=4)


def _clean(obj):
    """Recursively sanitize NaN/Inf for JSON serialization."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


# ─── AUTH ────────────────────────────────────────────────────────
def create_token(data: dict) -> str:
    to_encode = data.copy()
    to_encode["exp"] = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE)
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None or username not in USERS_DB:
            raise HTTPException(status_code=401, detail="Invalid token")
        return USERS_DB[username]
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ─── APP ─────────────────────────────────────────────────────────
app = FastAPI(title="NEAT Trading Bot v2.0", docs_url="/docs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:*", "http://127.0.0.1:*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── AUTH ENDPOINTS ──────────────────────────────────────────────
@app.post("/auth/token")
def login(form: OAuth2PasswordRequestForm = Depends()):
    user = USERS_DB.get(form.username)
    if not user or not verify_password(form.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Bad credentials")
    token = create_token({"sub": form.username, "role": user["role"]})
    return {"access_token": token, "token_type": "bearer"}


# ─── SCAN ENDPOINTS ─────────────────────────────────────────────
class ScanRequest(BaseModel):
    use_ollama: bool = True
    auto_trade: bool = False

class TickerRequest(BaseModel):
    ticker: str
    with_options: bool = True
    with_news: bool = True
    with_ai: bool = True


@app.post("/scan/full")
def full_scan(req: ScanRequest, user=Depends(get_current_user)):
    """Run the complete scanning pipeline."""
    from orchestrator import get_orchestrator
    orch = get_orchestrator()
    result = orch.full_scan(use_ollama=req.use_ollama, auto_trade=req.auto_trade)
    return _clean(result)


@app.get("/scan/quick")
def quick_scan(user=Depends(get_current_user)):
    """Quick scan — 30 tickers, no AI."""
    from orchestrator import get_orchestrator
    result = get_orchestrator().quick_scan()
    return _clean(result)


@app.post("/scan/ticker")
def scan_ticker(req: TickerRequest, user=Depends(get_current_user)):
    """Deep analysis of a single ticker."""
    from orchestrator import get_orchestrator
    result = get_orchestrator().analyze_ticker(
        req.ticker, req.with_options, req.with_news, req.with_ai
    )
    return _clean(result)


@app.get("/scan/sectors")
def scan_sectors(user=Depends(get_current_user)):
    """Scan all 11 sectors."""
    from sector_scanner import scan_all_sectors
    return _clean(scan_all_sectors())


@app.get("/scan/sector/{name}")
def scan_single_sector(name: str, user=Depends(get_current_user)):
    """Scan a specific sector."""
    from sector_scanner import scan_sector
    return _clean(scan_sector(name))


@app.get("/scan/crypto")
def scan_crypto(user=Depends(get_current_user)):
    """Scan all 30 crypto assets."""
    from crypto_scanner import scan_all_crypto
    return _clean(scan_all_crypto())


@app.get("/scan/news")
def scan_news(user=Depends(get_current_user)):
    """Scan market news."""
    from news_scanner import scan_market_news
    return _clean(scan_market_news())


@app.get("/scan/news/{ticker}")
def scan_ticker_news_ep(ticker: str, user=Depends(get_current_user)):
    """Get news for a specific ticker."""
    from news_scanner import scan_ticker_news
    return _clean(scan_ticker_news(ticker, use_ollama=False))


@app.get("/options/{ticker}")
def get_options(ticker: str, signal: str = "BOTH",
                min_days: int = 0, max_days: int = 730,
                top_n: int = 10,
                user=Depends(get_current_user)):
    """Full options chain — calls + puts, weeklies to LEAPs."""
    from options_analyzer import analyze_options
    return _clean(analyze_options(ticker, signal, min_days, max_days, top_n))


# ─── EARNINGS ENDPOINTS ─────────────────────────────────────────
@app.get("/earnings/{ticker}")
def get_earnings(ticker: str, user=Depends(get_current_user)):
    """Earnings analysis — date, expected move, 2 SD ranges."""
    from earnings_scanner import analyze_earnings
    return _clean(analyze_earnings(ticker))


@app.post("/earnings/batch")
def batch_earnings(tickers: list[str], user=Depends(get_current_user)):
    """Scan multiple tickers for upcoming earnings."""
    from earnings_scanner import scan_earnings_batch
    return _clean(scan_earnings_batch(tickers))


@app.get("/earnings/scan/sectors")
def scan_sector_earnings(user=Depends(get_current_user)):
    """Scan all sector stocks for upcoming earnings."""
    from earnings_scanner import scan_earnings_batch
    from sector_scanner import ALL_SECTOR_TICKERS
    return _clean(scan_earnings_batch(ALL_SECTOR_TICKERS))


# ─── AI ENDPOINTS ───────────────────────────────────────────────
@app.get("/ai/status")
def ai_status(user=Depends(get_current_user)):
    """Check Ollama status."""
    from ollama_brain import check_ollama
    return check_ollama()


class AskRequest(BaseModel):
    ticker: str
    question: Optional[str] = None

@app.post("/ai/analyze")
def ai_analyze(req: AskRequest, user=Depends(get_current_user)):
    """Full AI analysis of a ticker with chart."""
    from trading_agent import analyze
    result = analyze(req.ticker, req.question)
    return _clean(result)


# ─── PORTFOLIO ENDPOINTS ────────────────────────────────────────
@app.get("/portfolio")
def get_portfolio(user=Depends(get_current_user)):
    """Get paper trading portfolio."""
    from orchestrator import get_orchestrator
    return _clean(get_orchestrator().get_portfolio())


@app.post("/portfolio/update")
def update_portfolio(user=Depends(get_current_user)):
    """Update all position prices."""
    from paper_trader import PaperTrader
    pt = PaperTrader()
    triggered = pt.update_positions()
    return _clean({"triggered": triggered, "summary": pt.get_summary()})


# ─── NEURAL NETWORK / AI AGENT ENDPOINTS ───────────────────────
class TrainRequest(BaseModel):
    ticker: str = "AAPL"
    period: str = "1y"
    epochs: int = 50
    generations: int = 20

@app.get("/agent/status")
def agent_status(user=Depends(get_current_user)):
    """Get autonomous trading agent status."""
    from auto_trader import get_agent_status
    return _clean(get_agent_status())

@app.post("/agent/start")
def agent_start(user=Depends(get_current_user)):
    """Start the autonomous trading agent."""
    from auto_trader import start_agent
    return _clean(start_agent())

@app.post("/agent/stop")
def agent_stop(user=Depends(get_current_user)):
    """Stop the autonomous trading agent."""
    from auto_trader import stop_agent
    return _clean(stop_agent())

@app.get("/agent/logs")
def agent_logs(user=Depends(get_current_user)):
    """Get agent scan and trade logs."""
    from auto_trader import get_agent_logs
    return _clean(get_agent_logs())

@app.post("/agent/scan-once")
def agent_scan_once(user=Depends(get_current_user)):
    """Run a single scan cycle (no loop)."""
    from auto_trader import run_single_scan
    return _clean(run_single_scan())

@app.get("/nn/status")
def nn_status(user=Depends(get_current_user)):
    """Get status of all neural network models."""
    status = {}
    try:
        from neat_trader import get_neat_trader
        status["neat"] = get_neat_trader().get_status()
    except Exception as e:
        status["neat"] = {"error": str(e)}
    try:
        from torch_predictor import get_torch_predictor
        status["pytorch"] = get_torch_predictor().get_status()
    except Exception as e:
        status["pytorch"] = {"error": str(e)}
    try:
        from evo_optimizer import get_evo_optimizer
        status["evo"] = get_evo_optimizer().get_status()
    except Exception as e:
        status["evo"] = {"error": str(e)}
    try:
        from multi_model_analyst import get_multi_analyst
        status["multi_model"] = get_multi_analyst().get_status()
    except Exception as e:
        status["multi_model"] = {"error": str(e)}
    return _clean(status)

@app.post("/nn/train/neat")
def train_neat(req: TrainRequest, user=Depends(get_current_user)):
    """Train the NEAT neuroevolution model."""
    data = _fetch_training_data(req.ticker, req.period)
    if "error" in data:
        return data
    from neat_trader import get_neat_trader
    return _clean(get_neat_trader().evolve(data["bars"], generations=req.generations))

@app.post("/nn/train/pytorch")
def train_pytorch(req: TrainRequest, user=Depends(get_current_user)):
    """Train the PyTorch LSTM model."""
    data = _fetch_training_data(req.ticker, req.period)
    if "error" in data:
        return data
    from torch_predictor import get_torch_predictor
    return _clean(get_torch_predictor().train(data["bars"], epochs=req.epochs))

@app.post("/nn/train/evo")
def train_evo(req: TrainRequest, user=Depends(get_current_user)):
    """Train the evolutionary strategy optimizer."""
    data = _fetch_training_data(req.ticker, req.period)
    if "error" in data:
        return data
    from evo_optimizer import get_evo_optimizer
    return _clean(get_evo_optimizer().evolve(data["bars"], generations=req.generations))

@app.post("/nn/train/all")
def train_all(req: TrainRequest, user=Depends(get_current_user)):
    """Train all neural network models on the same data."""
    data = _fetch_training_data(req.ticker, req.period)
    if "error" in data:
        return data
    results = {}
    try:
        from neat_trader import get_neat_trader
        results["neat"] = get_neat_trader().evolve(data["bars"], generations=req.generations)
    except Exception as e:
        results["neat"] = {"error": str(e)}
    try:
        from torch_predictor import get_torch_predictor
        results["pytorch"] = get_torch_predictor().train(data["bars"], epochs=req.epochs)
    except Exception as e:
        results["pytorch"] = {"error": str(e)}
    try:
        from evo_optimizer import get_evo_optimizer
        results["evo"] = get_evo_optimizer().evolve(data["bars"], generations=req.generations)
    except Exception as e:
        results["evo"] = {"error": str(e)}
    return _clean(results)

@app.post("/nn/predict")
def nn_predict(req: AskRequest, user=Depends(get_current_user)):
    """Get ensemble prediction for a ticker."""
    from auto_trader import _fetch_ticker_data, _ensemble_vote
    data = _fetch_ticker_data(req.ticker)
    if not data:
        return {"error": f"No data for {req.ticker}"}
    vote = _ensemble_vote(data)
    vote["ticker"] = req.ticker
    vote["price"] = data["price"]
    return _clean(vote)

@app.get("/nn/multi-model/check")
def multi_model_check(user=Depends(get_current_user)):
    """Check available Ollama models."""
    from multi_model_analyst import get_multi_analyst
    return _clean(get_multi_analyst().check_models())

@app.post("/nn/multi-model/analyze")
def multi_model_analyze(req: AskRequest, user=Depends(get_current_user)):
    """Run multi-model Ollama consensus analysis."""
    from auto_trader import _fetch_ticker_data
    data = _fetch_ticker_data(req.ticker)
    if not data:
        return {"error": f"No data for {req.ticker}"}
    from multi_model_analyst import get_multi_analyst
    return _clean(get_multi_analyst().analyze(req.ticker, data))


@app.get("/agent/terminal")
def agent_terminal(since: int = 0, user=Depends(get_current_user)):
    """Get live terminal messages from the agent."""
    from auto_trader import get_terminal
    return get_terminal(since)

@app.post("/nn/train/mass")
def train_mass(user=Depends(get_current_user)):
    """Mass-train all models on 300 S&P 500 stocks (10 years)."""
    from mass_trainer import download_all, train_all_models
    import random
    bars = download_all(n_stocks=300, period="10y", workers=12)
    if len(bars) < 1000:
        return {"error": "Not enough data collected"}
    train_all_models(bars)
    return _clean({
        "status": "complete",
        "bars_trained": len(bars),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })


def _fetch_training_data(ticker: str, period: str = "1y") -> dict:
    """Fetch and prepare training data for neural network models."""
    from trading_agent import _rsi, _macd, _bollinger, _sma, _atr
    try:
        tkr = yf.Ticker(ticker)
        hist = tkr.history(period=period)
        if hist.empty or len(hist) < 50:
            return {"error": f"Not enough data for {ticker}"}

        close = hist["Close"].values.astype(float)
        high = hist["High"].values.astype(float)
        low = hist["Low"].values.astype(float)
        volume = hist["Volume"].values.astype(float)

        # Compute indicators ONCE on the full array
        rsi_arr = _rsi(close)
        macd_l_arr, macd_s_arr, macd_h_arr = _macd(close)
        bb_u_arr, bb_m_arr, bb_l_arr = _bollinger(close)
        sma50_arr = _sma(close, 50)
        sma200_arr = _sma(close, 200)
        atr_arr = _atr(high, low, close)

        def _safe(arr, idx):
            """Extract scalar from array, handling NaN."""
            v = arr[idx] if idx < len(arr) else 0.0
            return float(v) if not np.isnan(v) else 0.0

        # Rolling volume average
        vol_avg = np.convolve(volume, np.ones(20)/20, mode='same')

        bars = []
        for i in range(30, len(close)):
            price = float(close[i])
            bars.append({
                "price": price,
                "indicators": {
                    "price": price,
                    "rsi": _safe(rsi_arr, i),
                    "macd": _safe(macd_l_arr, i),
                    "macd_signal": _safe(macd_s_arr, i),
                    "macd_hist": _safe(macd_h_arr, i),
                    "bb_upper": _safe(bb_u_arr, i),
                    "bb_middle": _safe(bb_m_arr, i),
                    "bb_lower": _safe(bb_l_arr, i),
                    "sma50": _safe(sma50_arr, i),
                    "sma200": _safe(sma200_arr, i),
                    "atr": _safe(atr_arr, i),
                    "vol_ratio": float(volume[i]) / (float(vol_avg[i]) + 1e-10),
                },
                "price_history": {
                    "chg_1d": float((close[i] - close[i-1]) / (close[i-1] + 1e-10) * 100) if i > 0 else 0,
                    "chg_5d": float((close[i] - close[i-5]) / (close[i-5] + 1e-10) * 100) if i > 5 else 0,
                    "chg_20d": float((close[i] - close[i-20]) / (close[i-20] + 1e-10) * 100) if i > 20 else 0,
                },
                "future_prices": [float(x) for x in close[i+1:i+6]] if i + 5 < len(close) else [],
            })

        return {"ticker": ticker, "bars": bars, "total_bars": len(bars)}
    except Exception as e:
        return {"error": str(e)}


# ─── MARKET OVERVIEW ENDPOINTS ──────────────────────────────────
@app.get("/market/indices")
def market_indices(user=Depends(get_current_user)):
    """Live market indices — S&P, NASDAQ, DOW, VIX, Gold, Oil, BTC."""
    from market_overview import get_market_indices
    return _clean(get_market_indices())

@app.get("/market/heatmap")
def market_heatmap(user=Depends(get_current_user)):
    """Sector ETF heatmap — 1D, 1W, 1M performance."""
    from market_overview import get_sector_heatmap
    return _clean(get_sector_heatmap())

@app.get("/market/fear-greed")
def market_fear_greed(user=Depends(get_current_user)):
    """Fear & Greed index (local proxy)."""
    from market_overview import get_fear_greed
    return _clean(get_fear_greed())

@app.get("/market/movers")
def market_movers(user=Depends(get_current_user)):
    """Top gainers, losers, most active."""
    from market_overview import get_market_movers
    return _clean(get_market_movers())

class WatchlistAdd(BaseModel):
    ticker: str
    alert_above: Optional[float] = None
    alert_below: Optional[float] = None
    notes: str = ""

@app.get("/market/watchlist")
def market_watchlist(user=Depends(get_current_user)):
    """Get watchlist with live prices."""
    from market_overview import get_watchlist
    return _clean(get_watchlist())

@app.post("/market/watchlist/add")
def market_watchlist_add(req: WatchlistAdd, user=Depends(get_current_user)):
    """Add ticker to watchlist."""
    from market_overview import add_to_watchlist
    return _clean(add_to_watchlist(req.ticker, req.alert_above, req.alert_below, req.notes))

@app.delete("/market/watchlist/{ticker}")
def market_watchlist_remove(ticker: str, user=Depends(get_current_user)):
    """Remove ticker from watchlist."""
    from market_overview import remove_from_watchlist
    return _clean(remove_from_watchlist(ticker))

# ─── SELF-HEALER ENDPOINTS ─────────────────────────────────────
@app.get("/healer/log")
def healer_log(user=Depends(get_current_user)):
    """Get self-healing error log."""
    from self_healer import get_heal_log
    return _clean(get_heal_log())

@app.get("/system/health")
def system_health(user=Depends(get_current_user)):
    """System health check — Ollama, models, disk, memory."""
    import shutil
    health = {"timestamp": datetime.utcnow().isoformat() + "Z"}
    # Ollama
    try:
        import requests as _req
        r = _req.get("http://localhost:11434/api/tags", timeout=3)
        health["ollama"] = {"status": "online", "models": len(r.json().get("models", []))}
    except Exception:
        health["ollama"] = {"status": "offline"}
    # Disk
    try:
        usage = shutil.disk_usage(os.path.dirname(__file__))
        health["disk"] = {"free_gb": round(usage.free / 1e9, 1), "total_gb": round(usage.total / 1e9, 1)}
    except Exception:
        pass
    # Heal log
    try:
        from self_healer import get_heal_log
        log = get_heal_log()
        health["recent_errors"] = len(log)
        health["last_error"] = log[-1] if log else None
    except Exception:
        health["recent_errors"] = 0
    return _clean(health)


# ─── AI CHAT ENDPOINTS ─────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    include_context: bool = True

@app.post("/chat/send")
def chat_send(req: ChatRequest, user=Depends(get_current_user)):
    """Send a message to the AI finance trader."""
    from ai_chat import chat
    return _clean(chat(req.message, req.include_context))

@app.get("/chat/history")
def chat_history(limit: int = 50, user=Depends(get_current_user)):
    """Get chat history."""
    from ai_chat import get_chat_history
    return _clean(get_chat_history(limit))

@app.post("/chat/clear")
def chat_clear(user=Depends(get_current_user)):
    """Clear chat history."""
    from ai_chat import clear_chat
    return clear_chat()

@app.post("/chat/quick/{ticker}")
def chat_quick(ticker: str, user=Depends(get_current_user)):
    """Quick AI analysis of a ticker."""
    from ai_chat import quick_analysis
    return _clean(quick_analysis(ticker))


# ─── RESULTS ENDPOINTS ──────────────────────────────────────────
@app.get("/results/{name}")
def get_results(name: str, user=Depends(get_current_user)):
    """Get saved scan results."""
    path = os.path.join(os.path.dirname(__file__), "scan_results", f"{name}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"No results for '{name}'")
    with open(path) as f:
        return json.load(f)


# ─── DASHBOARD ───────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Trading bot dashboard."""
    return DASHBOARD_HTML


DASHBOARD_HTML = open(os.path.join(os.path.dirname(__file__), "dashboard.html"), encoding="utf-8").read()


# ─── STARTUP ─────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    print("NEAT Trading Bot v2.0 — http://127.0.0.1:8000")
    print("Login: trader / ChangeMe123!")
    print("Docs:  http://127.0.0.1:8000/docs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, log_level="info", reload=True)
