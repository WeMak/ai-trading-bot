"""
NEAT Trading Bot — Secure FastAPI Server
- JWT auth, CORS locked to localhost
- WebSocket real-time trade broadcasts
- Paper trading engine ($1,000)
- Stock + Crypto NEAT scanning
- TradingView webhook receiver
"""

import warnings, os, subprocess, sys, asyncio, json
from datetime import datetime, timedelta
from typing import Optional, List
from pathlib import Path

warnings.filterwarnings("ignore")

# Load .env
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path)

import bcrypt as _bcrypt
from fastapi import FastAPI, Depends, HTTPException, status, WebSocket, WebSocketDisconnect, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from jose import JWTError, jwt

# ─── CONFIG ───────────────────────────────────────────────────���───────────────
SECRET_KEY     = os.environ.get("SCANNER_SECRET", "change-me-32-bytes")
TV_WEBHOOK_KEY = os.environ.get("TV_WEBHOOK_KEY", "tv-secret-change-me")
ALGORITHM      = "HS256"
TOKEN_EXPIRE   = 60

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

# ─── APP ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="NEAT Trading Bot", version="2.0.0", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:4173", "http://127.0.0.1:4173",
        "http://localhost:8000", "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-TV-Secret"],
)

# ─── WEBSOCKET MANAGER ───────────────────────────────────��────────────────────
class WSManager:
    def __init__(self):
        self.connections: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.connections:
            self.connections.remove(ws)

    async def broadcast(self, msg: dict):
        dead = []
        for ws in self.connections:
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

ws_manager = WSManager()

# ─── AUTH ────────────────────────────────��────────────────────────────────────
def create_token(data: dict) -> str:
    payload = {**data, "exp": datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE)}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    exc = HTTPException(status_code=401, detail="Invalid token",
                        headers={"WWW-Authenticate": "Bearer"})
    try:
        payload  = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username or username not in USERS_DB:
            raise exc
    except JWTError:
        raise exc
    return USERS_DB[username]

@app.post("/auth/token", tags=["auth"])
async def login(form: OAuth2PasswordRequestForm = Depends()):
    user = USERS_DB.get(form.username)
    if not user or not verify_password(form.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password",
                            headers={"WWW-Authenticate": "Bearer"})
    token = create_token({"sub": user["username"], "role": user["role"]})
    return {"access_token": token, "token_type": "bearer", "expires_in": TOKEN_EXPIRE * 60}

# ─── STATE ─────────────────────────────��──────────────────────────────────────
_last_stock_scan: dict = {}
_last_crypto_scan: dict = {}
_scanning = False

# ─── SCAN — STOCKS ────────────────────────────────────────────────────────────
@app.get("/scan/status", tags=["scanner"])
async def scan_status(user: dict = Depends(get_current_user)):
    return _last_stock_scan or {"status": "no_scan"}

@app.post("/scan/run", tags=["scanner"])
async def run_scan(user: dict = Depends(get_current_user)):
    global _last_stock_scan, _scanning
    if _scanning:
        return {"status": "already_running"}
    _scanning = True

    await ws_manager.broadcast({"type": "scan_start", "asset": "stocks",
                                 "time": datetime.utcnow().isoformat()+"Z"})
    script = Path(__file__).parent / "neat_scanner.py"
    if not script.exists():
        _scanning = False
        raise HTTPException(status_code=404, detail="neat_scanner.py not found")
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, timeout=360,
            cwd=str(script.parent),
        )
        _last_stock_scan = {
            "status":    "ok" if proc.returncode == 0 else "error",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "stdout":    proc.stdout[-10000:],
            "stderr":    proc.stderr[-2000:] if proc.stderr else "",
        }

        # Auto-generate paper trade signals from top picks
        signals = _extract_stock_signals(proc.stdout)
        if signals:
            await _execute_paper_trades(signals)

        await ws_manager.broadcast({"type": "scan_complete", "asset": "stocks",
                                     "time": datetime.utcnow().isoformat()+"Z",
                                     "signal_count": len(signals)})
        return _last_stock_scan
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Scanner timed out")
    finally:
        _scanning = False

# ─── SCAN — CRYPTO ───────────────────────────��────────────────────────────────
@app.get("/scan/crypto/status", tags=["crypto"])
async def crypto_status(user: dict = Depends(get_current_user)):
    return _last_crypto_scan or {"status": "no_scan"}

@app.post("/scan/crypto/run", tags=["crypto"])
async def run_crypto_scan(user: dict = Depends(get_current_user)):
    global _last_crypto_scan, _scanning
    if _scanning:
        return {"status": "already_running"}
    _scanning = True

    await ws_manager.broadcast({"type": "scan_start", "asset": "crypto",
                                 "time": datetime.utcnow().isoformat()+"Z"})
    script = Path(__file__).parent / "crypto_scanner.py"
    if not script.exists():
        _scanning = False
        raise HTTPException(status_code=404, detail="crypto_scanner.py not found")
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, timeout=300,
            cwd=str(script.parent),
        )

        # crypto_scanner prints JSON on stdout
        try:
            result = json.loads(proc.stdout.split("\n")[-2] if "\n" in proc.stdout else proc.stdout)
        except Exception:
            result = {"raw": proc.stdout[-5000:]}

        _last_crypto_scan = {
            "status":    "ok" if proc.returncode == 0 else "error",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "data":      result,
            "stderr":    proc.stderr[-2000:] if proc.stderr else "",
        }

        # Auto-generate paper trade signals from top crypto picks
        signals = _extract_crypto_signals(result)
        if signals:
            await _execute_paper_trades(signals)

        await ws_manager.broadcast({"type": "scan_complete", "asset": "crypto",
                                     "time": datetime.utcnow().isoformat()+"Z",
                                     "signal_count": len(signals)})
        return _last_crypto_scan
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Crypto scan timed out")
    finally:
        _scanning = False

# ─── PAPER TRADING ────────────────────────────────────────────────────────────
@app.get("/portfolio", tags=["trading"])
async def get_portfolio(user: dict = Depends(get_current_user)):
    from paper_trader import PaperTrader
    return PaperTrader().get_summary()

@app.get("/trades", tags=["trading"])
async def get_trades(user: dict = Depends(get_current_user)):
    from paper_trader import PaperTrader
    pt = PaperTrader()
    return {"trades": pt.data["trades"][:100],
            "total": len(pt.data["trades"])}

@app.get("/positions", tags=["trading"])
async def get_positions(user: dict = Depends(get_current_user)):
    from paper_trader import PaperTrader
    pt = PaperTrader()
    triggered = pt.update_positions()
    if triggered:
        for t in triggered:
            await ws_manager.broadcast({"type": "trade_closed", "trade": t,
                                         "time": datetime.utcnow().isoformat()+"Z"})
    return {"positions": pt.data["positions"]}

@app.post("/portfolio/reset", tags=["trading"])
async def reset_portfolio(user: dict = Depends(get_current_user)):
    """Reset paper account to $2,000."""
    import json as _json
    from paper_trader import ACCOUNT_FILE, DEFAULTS
    with open(ACCOUNT_FILE, "w") as f:
        _json.dump(dict(DEFAULTS), f, indent=2)
    await ws_manager.broadcast({"type": "portfolio_reset",
                                  "time": datetime.utcnow().isoformat()+"Z"})
    return {"status": "reset", "balance": DEFAULTS["balance"]}

# ─── TRADINGVIEW WEBHOOK ──────────────────────────────────────────────────────
@app.post("/webhook/tradingview", tags=["webhook"])
async def tradingview_webhook(request: Request):
    """
    Receive TradingView Pine Script alerts.
    Set alert message to JSON:
      {"ticker":"BTCUSDT","action":"BUY","price":"45000","strategy":"MyBot"}
    Add header X-TV-Secret: <your TV_WEBHOOK_KEY from .env>
    """
    secret = request.headers.get("X-TV-Secret", "")
    if secret != TV_WEBHOOK_KEY:
        raise HTTPException(status_code=403, detail="Invalid webhook secret")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    ticker    = body.get("ticker", "UNKNOWN").replace("USDT","").replace("USD","")
    action    = body.get("action", "").upper()
    price_str = str(body.get("price", "0")).replace(",","")
    price     = float(price_str) if price_str else 0.0
    strategy  = body.get("strategy", "TradingView")

    event = {
        "type": "tradingview_alert",
        "ticker": ticker,
        "action": action,
        "price": price,
        "strategy": strategy,
        "raw": body,
        "time": datetime.utcnow().isoformat() + "Z",
    }

    await ws_manager.broadcast(event)

    # If it's a directional signal, create a paper trade
    if action in ("BUY","LONG") and price > 0:
        from paper_trader import PaperTrader
        pt = PaperTrader()
        trade = pt.enter_trade(
            ticker=ticker + "-USD",
            direction="LONG",
            entry_price=price,
            asset_type="CRYPTO",
            signal_strength=0.7,
            reason=f"TradingView alert: {strategy}",
        )
        if trade:
            await ws_manager.broadcast({"type": "trade_opened", "trade": trade,
                                         "source": "tradingview",
                                         "time": datetime.utcnow().isoformat()+"Z"})
    elif action in ("SELL","SHORT") and price > 0:
        from paper_trader import PaperTrader
        pt = PaperTrader()
        trade = pt.enter_trade(
            ticker=ticker + "-USD",
            direction="SHORT",
            entry_price=price,
            asset_type="CRYPTO",
            signal_strength=0.7,
            reason=f"TradingView alert: {strategy}",
        )
        if trade:
            await ws_manager.broadcast({"type": "trade_opened", "trade": trade,
                                         "source": "tradingview",
                                         "time": datetime.utcnow().isoformat()+"Z"})

    return {"status": "received", "event": event}

# ─── WEBSOCKET ────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        # Send initial portfolio state on connect
        from paper_trader import PaperTrader
        summary = PaperTrader().get_summary()
        await websocket.send_json({"type": "init", "portfolio": summary,
                                    "time": datetime.utcnow().isoformat()+"Z"})
        while True:
            # Keep alive — listen for pings from client
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)

# ─── SIGNAL EXTRACTION HELPERS ────────────────────────────────────────────────
def _extract_stock_signals(stdout: str) -> list:
    """
    Parse scanner stdout → find THE #1 cheapest, highest-fitness option.
    Rules: IV rank ≤ 40%, premium ≤ $6/share, highest (fitness × cheapness) score.
    Always buys 3 contracts.
    """
    candidates = []
    blocks = stdout.split("\n#")
    for block in blocks[1:]:
        lines = block.strip().split("\n")
        try:
            ticker    = next((l.split(":")[1].strip().split("@")[0].strip()
                               for l in lines if "Ticker" in l), None)
            price_str = next((l.split("@$")[1].strip().split("\n")[0]
                               for l in lines if "Ticker" in l and "@$" in l), "0")
            direction = next((l.split("BUY ")[1].strip().split()[0]
                               for l in lines if "Trade" in l and "BUY" in l), "CALL")
            strike_str= next((l.split("Strike $")[1].strip().split()[0]
                               for l in lines if "Strike $" in l), "0")
            expiry    = next((l.split("Expiry")[1].strip()
                               for l in lines if "Expiry" in l), "")
            fitness   = float(next((l.split("Fitness")[1].strip().strip(":")
                               for l in lines if "Fitness" in l), "0"))
            # Parse IV rank and estimated premium from output
            hv_line   = next((l for l in lines if "HV(1yr)" in l), "")
            hv_val    = 0.0
            if "HV(1yr)=" in hv_line:
                try: hv_val = float(hv_line.split("HV(1yr)=")[1].split("%")[0]) / 100
                except: pass

            prem_line = next((l for l in lines if "Est Prem" in l), "")
            est_prem  = 0.0
            if "~$" in prem_line:
                try: est_prem = float(prem_line.split("~$")[1].split("/")[0].strip())
                except: pass

            # IV rank proxy: low HV rank = cheap option
            iv_rank = hv_val / 0.8 if hv_val else 0.5   # normalise roughly
            iv_rank = min(iv_rank, 1.0)

            if not ticker or fitness < 0:
                continue
            price  = float(price_str.replace(",",""))
            strike = float(strike_str.replace(",",""))
            if est_prem <= 0:
                # Fallback estimate: 4% of underlying ATM
                import math
                T = 30/365
                est_prem = round(price * max(hv_val, 0.15) * math.sqrt(T) * 0.4, 2)

            # Score = fitness weighted by cheapness (lower IV = better)
            cheapness = max(0, 0.40 - iv_rank)          # reward IV < 40%
            score     = fitness + cheapness * 2.0

            candidates.append({
                "ticker":          ticker,
                "direction":       direction,
                "entry_price":     price,
                "asset_type":      "OPTIONS",
                "signal_strength": round(min(fitness * 10 + 0.3, 1.0), 4),
                "reason":          (f"NEAT #1 pick · fit={fitness:.3f} "
                                    f"IV≈{iv_rank:.0%} prem≈${est_prem:.2f} "
                                    f"3 contracts"),
                "strike":          strike,
                "expiry":          expiry.strip(),
                "est_premium":     est_prem,
                "iv_rank":         round(iv_rank, 4),
                "score":           score,
            })
        except Exception:
            continue

    if not candidates:
        return []

    # Sort by composite score — pick only the #1 cheapest/best
    candidates.sort(key=lambda x: x["score"], reverse=True)
    top = candidates[0]
    top.pop("score", None)

    # Broadcast the top pick so dashboard can highlight it
    return [top]

def _extract_crypto_signals(result: dict) -> list:
    """Convert crypto scan results into trade signals."""
    signals = []
    coins = result.get("top_longs", []) + result.get("top_shorts", [])
    for c in coins:
        strength = c.get("strength", 0)
        if strength < 0.25:
            continue
        signals.append({
            "ticker":          c["symbol"],
            "direction":       c["direction"],
            "entry_price":     c["price"],
            "asset_type":      "CRYPTO",
            "signal_strength": strength,
            "reason":          f"NEAT crypto signal btc_rs={c.get('btc_rs',0):+.2f}",
        })
    return signals

async def _execute_paper_trades(signals: list):
    """Execute signals through paper trader and broadcast results."""
    from paper_trader import PaperTrader
    pt = PaperTrader()
    events = pt.process_signals(signals)
    for e in events:
        etype = "trade_closed" if e.get("status") == "closed" else "trade_opened"
        await ws_manager.broadcast({"type": etype, "trade": e,
                                     "time": datetime.utcnow().isoformat()+"Z"})
    if events:
        summary = pt.get_summary()
        await ws_manager.broadcast({"type": "portfolio_update", "portfolio": summary,
                                     "time": datetime.utcnow().isoformat()+"Z"})

# ─── AI AGENT ─────────────────────────────────────────────────────────────────
class AgentRequest(BaseModel):
    ticker: str
    question: Optional[str] = None

@app.post("/agent/ask", tags=["agent"])
async def agent_ask(req: AgentRequest, user: dict = Depends(get_current_user)):
    """
    Full AI analysis for a single ticker — candlestick chart + indicators + signal.
    """
    if not req.ticker or len(req.ticker.strip()) < 1:
        raise HTTPException(status_code=400, detail="ticker is required")
    try:
        from trading_agent import analyze
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: analyze(req.ticker.strip(), req.question)
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent/find-trades", tags=["agent"])
async def find_trades_endpoint(
    n: int = 10,
    user: dict = Depends(get_current_user)
):
    """
    Autonomous trade finder — scans 30 tickers (20 stocks + 10 crypto) in parallel.
    Returns top-N opportunities ranked by signal confidence.
    """
    n = max(1, min(n, 20))   # clamp to 1-20
    try:
        from trade_finder import find_trades
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: find_trades(n))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── HEALTH ────────────────────────────────────────────────────────────────���──
@app.get("/health", tags=["infra"])
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z",
            "ws_connections": len(ws_manager.connections)}

# ─── ENTRYPOINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000,
                reload=False, log_level="info")
