"""
Ollama AI Brain — Local LLM for trade analysis
================================================
Sends structured market data to your local Ollama models.
No tokens used, no data leaves your PC.
Uses qwen2.5-coder:32b for analysis (fast, smart).
"""

import warnings; warnings.filterwarnings("ignore")
import json, time
import requests
from typing import Optional

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen2.5-coder:32b"  # Best balance of speed + intelligence


def _call_ollama(prompt: str, system: str = "", temperature: float = 0.3,
                 max_tokens: int = 2000, timeout: int = 120) -> Optional[str]:
    """Send prompt to Ollama, return response text."""
    try:
        payload = {
            "model": MODEL,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json().get("response", "")
    except requests.exceptions.ConnectionError:
        return "[ERROR] Ollama not running. Start with: ollama serve"
    except Exception as e:
        return f"[ERROR] Ollama call failed: {e}"


def check_ollama() -> dict:
    """Check if Ollama is running and what models are available."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        return {"status": "online", "models": models, "using": MODEL}
    except Exception:
        return {"status": "offline", "models": [], "error": "Ollama not reachable at localhost:11434"}


def analyze_trade_signal(ticker_data: dict) -> dict:
    """
    Send technical analysis data to Ollama for AI interpretation.
    Returns structured trade recommendation.
    """
    t0 = time.time()

    system = """You are an expert quantitative trader and technical analyst.
You analyze stock/crypto market data and provide actionable trade recommendations.
Always respond in valid JSON format with these fields:
- action: "STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"
- confidence: 0-100
- timeframe: "day_trade", "swing_1_2_weeks", "position_1_3_months"
- reasoning: brief 2-3 sentence explanation
- entry_zone: [low, high] price range for entry
- stop_loss: price level
- take_profit: [target1, target2] price levels
- risk_level: "LOW", "MEDIUM", "HIGH"
- key_factors: list of 3-5 key factors driving the recommendation"""

    prompt = f"""Analyze this market data and provide a trade recommendation:

Ticker: {ticker_data.get('ticker', 'UNKNOWN')}
Price: ${ticker_data.get('price', 0):,.4f}
Day Change: {ticker_data.get('day_chg', 0):+.2f}%

Technical Indicators:
- RSI(14): {ticker_data.get('indicators', {}).get('rsi', 'N/A')}
- MACD: {ticker_data.get('indicators', {}).get('macd', 'N/A')}
- MACD Signal: {ticker_data.get('indicators', {}).get('macd_signal', 'N/A')}
- MACD Histogram: {ticker_data.get('indicators', {}).get('macd_hist', 'N/A')}
- BB Upper: ${ticker_data.get('indicators', {}).get('bb_upper', 0):,.4f}
- BB Mid: ${ticker_data.get('indicators', {}).get('bb_mid', 0):,.4f}
- BB Lower: ${ticker_data.get('indicators', {}).get('bb_lower', 0):,.4f}
- SMA 50: ${ticker_data.get('indicators', {}).get('sma50', 0):,.4f}
- SMA 200: ${ticker_data.get('indicators', {}).get('sma200', 0):,.4f}
- ATR: ${ticker_data.get('indicators', {}).get('atr', 0):,.4f}
- Volume Ratio: {ticker_data.get('indicators', {}).get('vol_ratio', 0):.2f}x avg

Signal from technical engine: {ticker_data.get('signal', 'N/A')} ({ticker_data.get('confidence', 0):.0f}% confidence)
Trend: {ticker_data.get('trend', 'N/A')}
Notes: {'; '.join(ticker_data.get('notes', []))}

Respond ONLY with valid JSON. No markdown, no code blocks."""

    response = _call_ollama(prompt, system)

    if response and response.startswith("[ERROR]"):
        return {"error": response, "elapsed_s": round(time.time() - t0, 1)}

    # Parse JSON from response
    try:
        # Try to extract JSON from response
        text = response.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
        result["model"] = MODEL
        result["elapsed_s"] = round(time.time() - t0, 1)
        return result
    except (json.JSONDecodeError, IndexError):
        return {
            "action": "HOLD",
            "confidence": 0,
            "reasoning": response[:500] if response else "No response from Ollama",
            "raw_response": response[:1000] if response else "",
            "parse_error": True,
            "model": MODEL,
            "elapsed_s": round(time.time() - t0, 1),
        }


def analyze_news_sentiment(ticker: str, headlines: list) -> dict:
    """Use Ollama to analyze news sentiment for a ticker."""
    t0 = time.time()

    system = """You are a financial news sentiment analyzer.
Analyze headlines and return JSON with:
- overall_sentiment: "VERY_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "VERY_BEARISH"
- sentiment_score: -1.0 to +1.0
- key_themes: list of main themes
- impact_assessment: "HIGH", "MEDIUM", "LOW"
- summary: 2-3 sentence summary of news sentiment
- actionable: true/false — whether news warrants action"""

    headlines_text = "\n".join(f"- {h}" for h in headlines[:20])

    prompt = f"""Analyze these recent news headlines for {ticker}:

{headlines_text}

What is the overall sentiment? How might this affect the stock price?
Respond ONLY with valid JSON."""

    response = _call_ollama(prompt, system)

    if response and response.startswith("[ERROR]"):
        return {"error": response}

    try:
        text = response.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
        result["model"] = MODEL
        result["elapsed_s"] = round(time.time() - t0, 1)
        return result
    except (json.JSONDecodeError, IndexError):
        return {
            "overall_sentiment": "NEUTRAL",
            "sentiment_score": 0.0,
            "summary": response[:300] if response else "Parse error",
            "parse_error": True,
            "model": MODEL,
            "elapsed_s": round(time.time() - t0, 1),
        }


def analyze_options_strategy(ticker: str, stock_price: float, signal: str,
                              options_data: list, indicators: dict) -> dict:
    """Use Ollama to recommend best options strategy."""
    t0 = time.time()

    system = """You are an expert options trader.
Given stock data and available options contracts, recommend the best strategy.
Return JSON with:
- strategy: name of strategy (e.g., "long_call", "bull_call_spread", "protective_put")
- contracts: list of contracts to buy/sell
- max_risk: maximum dollar risk
- max_reward: maximum dollar reward
- breakeven: breakeven price(s)
- probability_of_profit: estimated %
- reasoning: 2-3 sentences explaining why this strategy"""

    options_text = ""
    for opt in options_data[:10]:
        options_text += (
            f"  {opt['option_type']} ${opt['strike']:.0f} exp {opt['expiry']} "
            f"({opt['days_to_expiry']}d) — ${opt.get('lastPrice', 0):.2f} "
            f"IV:{opt.get('impliedVolatility', 0):.0%} "
            f"Vol:{opt.get('volume', 0)} OI:{opt.get('openInterest', 0)}\n"
        )

    prompt = f"""Recommend the best options strategy for {ticker}:

Stock Price: ${stock_price:,.2f}
Technical Signal: {signal}
RSI: {indicators.get('rsi', 'N/A')}
Trend: {'Up' if stock_price > indicators.get('sma200', 0) else 'Down'}

Available contracts:
{options_text}

Account size: $2,000 (paper trading)
Risk tolerance: moderate
Respond ONLY with valid JSON."""

    response = _call_ollama(prompt, system)

    if response and response.startswith("[ERROR]"):
        return {"error": response}

    try:
        text = response.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
        result["model"] = MODEL
        result["elapsed_s"] = round(time.time() - t0, 1)
        return result
    except (json.JSONDecodeError, IndexError):
        return {
            "strategy": "unknown",
            "reasoning": response[:300] if response else "Parse error",
            "parse_error": True,
            "model": MODEL,
            "elapsed_s": round(time.time() - t0, 1),
        }


def generate_market_brief(sector_data: dict, top_trades: list) -> dict:
    """Generate a daily market brief using Ollama."""
    t0 = time.time()

    system = """You are a senior market strategist writing a daily trading brief.
Return JSON with:
- market_outlook: "BULLISH", "NEUTRAL", "BEARISH"
- summary: 3-4 sentence market overview
- top_sectors: list of strongest sectors with reasoning
- avoid_sectors: list of weakest sectors
- trade_ideas: list of 3-5 specific actionable trade ideas with entry/exit levels
- risk_factors: list of key risks to watch
- day_trade_picks: 2-3 tickers best suited for day trading today"""

    sector_summary = ""
    for sec, data in (sector_data.get("sector_strength", {}) or {}).items():
        sector_summary += f"  {sec}: {data['sentiment']} (daily:{data['avg_daily_chg']:+.1f}%, 5d:{data['avg_5d_chg']:+.1f}%, buys:{data['buy_signals']}, sells:{data['sell_signals']})\n"

    trades_summary = ""
    for t in top_trades[:10]:
        trades_summary += f"  {t['ticker']} — {t['signal']} {t['confidence']:.0f}% — RSI:{t['indicators']['rsi']:.0f} — {t['notes'][0][:60]}\n"

    prompt = f"""Generate a daily trading brief based on this market data:

SECTOR ANALYSIS:
{sector_summary}

TOP TRADE SIGNALS:
{trades_summary}

Provide a comprehensive but concise daily brief.
Respond ONLY with valid JSON."""

    response = _call_ollama(prompt, system, max_tokens=3000)

    if response and response.startswith("[ERROR]"):
        return {"error": response}

    try:
        text = response.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
        result["model"] = MODEL
        result["elapsed_s"] = round(time.time() - t0, 1)
        return result
    except (json.JSONDecodeError, IndexError):
        return {
            "market_outlook": "NEUTRAL",
            "summary": response[:500] if response else "Parse error",
            "parse_error": True,
            "model": MODEL,
            "elapsed_s": round(time.time() - t0, 1),
        }


if __name__ == "__main__":
    print("Checking Ollama status...")
    status = check_ollama()
    print(f"Status: {status['status']}")
    if status["status"] == "online":
        print(f"Models: {', '.join(status['models'])}")
        print(f"Using: {status['using']}")

        # Test with sample data
        test_data = {
            "ticker": "AAPL",
            "price": 178.50,
            "day_chg": +1.2,
            "signal": "BUY",
            "confidence": 72,
            "trend": "Up",
            "notes": ["RSI 38 — below midpoint, mild bullish bias", "MACD bullish crossover"],
            "indicators": {
                "rsi": 38.0,
                "macd": 0.5,
                "macd_signal": -0.2,
                "macd_hist": 0.7,
                "bb_upper": 185.0,
                "bb_mid": 178.0,
                "bb_lower": 171.0,
                "sma50": 176.0,
                "sma200": 170.0,
                "atr": 3.2,
                "vol_ratio": 1.5,
            }
        }
        print("\nTesting trade analysis...")
        result = analyze_trade_signal(test_data)
        print(json.dumps(result, indent=2))
    else:
        print(f"Error: {status.get('error', 'Unknown')}")
