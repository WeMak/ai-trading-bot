"""
AI Finance Chat — Talk to Your AI Trader
==========================================
Local Ollama-powered finance assistant trained on your portfolio data,
market analysis, and trading history. Can fetch live data from the internet.
"""

import json, time, os, threading
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ─── Chat History ────────────────────────────────────────────────
_chat_history = []
_MAX_HISTORY = 100

SYSTEM_PROMPT = """You are an expert AI finance trader and analyst running locally on the user's PC.
You have access to real-time market data, technical indicators, and neural network predictions.

Your capabilities:
- Analyze stocks, crypto, options, and ETFs
- Provide technical analysis (RSI, MACD, Bollinger Bands, moving averages)
- Give trading recommendations with confidence levels
- Explain market trends, sector rotation, and macro events
- Discuss portfolio strategy, risk management, and position sizing
- Access the user's paper trading portfolio and trade history
- Run neural network predictions (NEAT, PyTorch LSTM, DEAP genetic algo)

Rules:
- Always be direct and actionable
- Include specific price levels, targets, and stop-losses when giving trade ideas
- Warn about risks — never just say "buy" without context
- Use data to back up your analysis
- If you don't know something, say so — don't make up data
- Reference the user's actual portfolio when relevant
- You ARE the trading bot — speak in first person about trades you've made

Current date: {date}
"""


def _get_portfolio_context():
    """Get current portfolio state for context."""
    try:
        from paper_trader import PaperTrader
        pt = PaperTrader()
        summary = pt.get_summary()
        return f"""
PORTFOLIO STATE:
- Balance: ${summary.get('balance', 0):,.2f}
- Portfolio Value: ${summary.get('portfolio_value', 0):,.2f}
- Total P&L: ${summary.get('total_pnl', 0):,.2f} ({summary.get('total_return_pct', 0):.1f}%)
- Open Positions: {summary.get('open_positions', 0)}
- Win Rate: {summary.get('win_rate', 0)}%
- Positions: {json.dumps(summary.get('positions', [])[:5], default=str)}
"""
    except Exception:
        return "\nPORTFOLIO: Paper trading account with $2,000 starting balance.\n"


def _get_agent_context():
    """Get agent status for context."""
    try:
        from auto_trader import get_agent_status
        status = get_agent_status()
        return f"""
AGENT STATUS:
- Running: {status.get('running', False)}
- Scan Cycles: {status.get('cycles', 0)}
- Signals Found: {status.get('signals_found', 0)}
- Trades Made: {status.get('trades_made', 0)}
"""
    except Exception:
        return "\nAGENT: Not currently running.\n"


def _get_nn_context():
    """Get neural network model status."""
    try:
        models = {}
        try:
            from neat_trader import get_neat_trader
            s = get_neat_trader().get_status()
            models["NEAT"] = f"Gen {s.get('generation', '?')}, {s.get('genome_nodes', '?')} nodes"
        except Exception:
            models["NEAT"] = "not loaded"
        try:
            from torch_predictor import get_torch_predictor
            s = get_torch_predictor().get_status()
            models["PyTorch"] = f"Epoch {s.get('epoch', '?')}, {s.get('parameters', '?')} params"
        except Exception:
            models["PyTorch"] = "not loaded"
        try:
            from evo_optimizer import get_evo_optimizer
            s = get_evo_optimizer().get_status()
            models["EvoStrategy"] = f"Gen {s.get('generation', '?')}, fitness {s.get('best_fitness', '?')}"
        except Exception:
            models["EvoStrategy"] = "not loaded"
        lines = "\n".join(f"  - {k}: {v}" for k, v in models.items())
        return f"\nNEURAL NETWORKS:\n{lines}\n"
    except Exception:
        return "\nNEURAL NETWORKS: Status unavailable.\n"


def _fetch_live_data(ticker):
    """Fetch live price data for a ticker."""
    try:
        import yfinance as yf
        tkr = yf.Ticker(ticker.upper())
        hist = tkr.history(period="5d")
        if hist.empty:
            return None
        last = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else hist.iloc[0]
        chg = (last["Close"] - prev["Close"]) / prev["Close"] * 100
        return {
            "ticker": ticker.upper(),
            "price": round(float(last["Close"]), 2),
            "change_pct": round(chg, 2),
            "volume": int(last["Volume"]),
            "high": round(float(last["High"]), 2),
            "low": round(float(last["Low"]), 2),
        }
    except Exception:
        return None


def _get_prediction(ticker):
    """Run ensemble prediction for context."""
    try:
        from auto_trader import _fetch_ticker_data, _ensemble_vote
        data = _fetch_ticker_data(ticker.upper())
        if data:
            vote = _ensemble_vote(data)
            return f"""
ENSEMBLE PREDICTION for {ticker.upper()}:
- Action: {vote.get('action', '?')}
- Confidence: {vote.get('confidence', '?')}%
- Agreement: {vote.get('agreement', '?')}%
- Models: {vote.get('models_agree', '?')}/{vote.get('models_voted', '?')} agree
- Votes: {json.dumps(vote.get('votes', []), default=str)}
"""
    except Exception:
        pass
    return ""


def _extract_tickers(msg):
    """Extract potential ticker symbols from user message."""
    import re
    # Match $AAPL style or standalone 1-5 uppercase letters
    tickers = re.findall(r'\$([A-Z]{1,5})', msg.upper())
    # Also check for common patterns
    words = msg.upper().split()
    known = {"AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA",
             "AMD", "INTC", "NFLX", "DIS", "BA", "JPM", "GS", "V", "MA",
             "BTC-USD", "ETH-USD", "SPY", "QQQ", "IWM", "VIX", "GLD", "SLV",
             "XLK", "XLV", "XLF", "XLE", "XLI", "XLY", "XLP", "XLRE", "XLB", "XLU", "XLC"}
    for w in words:
        clean = w.strip(",.?!:;\"'()[]")
        if clean in known:
            tickers.append(clean)
    return list(set(tickers))


def chat(message, include_context=True):
    """
    Send a message to the AI finance trader and get a response.

    Args:
        message: User's question/message
        include_context: Whether to include portfolio/agent/NN context

    Returns:
        dict with 'response', 'tickers_mentioned', 'live_data', 'timestamp'
    """
    import requests

    # Build context
    context_parts = []
    if include_context:
        context_parts.append(_get_portfolio_context())
        context_parts.append(_get_agent_context())
        context_parts.append(_get_nn_context())

    # Extract tickers and fetch live data
    tickers = _extract_tickers(message)
    live_data = {}
    predictions = {}
    for t in tickers[:5]:  # Limit to 5 tickers
        data = _fetch_live_data(t)
        if data:
            live_data[t] = data
            context_parts.append(f"\nLIVE DATA for {t}: Price ${data['price']}, Change {data['change_pct']:+.2f}%, Vol {data['volume']:,}")
        pred = _get_prediction(t)
        if pred:
            predictions[t] = pred
            context_parts.append(pred)

    # Build conversation for Ollama
    system = SYSTEM_PROMPT.format(date=datetime.now().strftime("%Y-%m-%d %H:%M"))
    if context_parts:
        system += "\n\n--- CURRENT CONTEXT ---" + "".join(context_parts)

    # Include recent chat history for continuity
    messages = [{"role": "system", "content": system}]
    for h in _chat_history[-10:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": message})

    # Call Ollama
    try:
        r = requests.post("http://localhost:11434/api/chat", json={
            "model": _pick_model(),
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 1024}
        }, timeout=120)
        r.raise_for_status()
        response_text = r.json().get("message", {}).get("content", "I couldn't generate a response.")
    except requests.exceptions.ConnectionError:
        response_text = "Ollama is not running. Start it with: ollama serve"
    except Exception as e:
        response_text = f"Error talking to Ollama: {str(e)}"

    # Save to history
    _chat_history.append({"role": "user", "content": message, "time": datetime.now().isoformat()})
    _chat_history.append({"role": "assistant", "content": response_text, "time": datetime.now().isoformat()})
    if len(_chat_history) > _MAX_HISTORY * 2:
        _chat_history[:] = _chat_history[-_MAX_HISTORY * 2:]

    return {
        "response": response_text,
        "tickers_mentioned": tickers,
        "live_data": live_data,
        "predictions": predictions,
        "timestamp": datetime.now().isoformat(),
    }


def _pick_model():
    """Pick the best available Ollama model."""
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        # Prefer larger models for chat
        preferred = ["gemma4:31b", "gemma4:latest", "qwen2.5-coder:32b",
                      "qwen3-coder:30b", "llama3:8b"]
        for p in preferred:
            if p in models:
                return p
        return models[0] if models else "gemma4:latest"
    except Exception:
        return "gemma4:latest"


def get_chat_history(limit=50):
    """Get recent chat history."""
    return _chat_history[-limit * 2:]


def clear_chat():
    """Clear chat history."""
    _chat_history.clear()
    return {"status": "cleared"}


def quick_analysis(ticker):
    """Quick analysis of a ticker — combines live data + prediction + AI commentary."""
    return chat(f"Give me a quick analysis of ${ticker}. What's the current setup? Should I buy, sell, or hold? Include key levels.")


# ─── Singleton ───────────────────────────────────────────────────
_instance = None
def get_ai_chat():
    global _instance
    if _instance is None:
        _instance = True  # Module-level functions, no class needed
    return True
