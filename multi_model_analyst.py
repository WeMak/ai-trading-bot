"""
Multi-Model Ollama Analyst
============================
Runs all available Ollama models in parallel on the same trade signal,
then combines their outputs into a consensus recommendation.

Models: gpt-oss:120b, qwen3-coder:30b, qwen2.5-coder:32b
Each model independently analyzes the same data.
Consensus = weighted vote across all models.
"""

import warnings; warnings.filterwarnings("ignore")
import os, json, time, asyncio, re
from datetime import datetime
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor
import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# Models in priority order (larger = higher weight)
MODELS = [
    {"name": "gpt-oss:120b", "weight": 1.5, "timeout": 120},
    {"name": "qwen3-coder:30b", "weight": 1.0, "timeout": 90},
    {"name": "qwen2.5-coder:32b", "weight": 1.0, "timeout": 90},
]


def _get_available_models() -> list:
    """Check which models are actually available in Ollama."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            installed = [m["name"] for m in r.json().get("models", [])]
            return [m for m in MODELS if m["name"] in installed or m["name"].split(":")[0] in [n.split(":")[0] for n in installed]]
    except Exception:
        pass
    return []


def _query_model(model_name: str, prompt: str, timeout: int = 90) -> Optional[dict]:
    """Query a single Ollama model and parse its JSON response."""
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 1024},
            },
            timeout=timeout,
        )
        if r.status_code != 200:
            return None

        text = r.json().get("response", "")

        # Try to parse JSON from response
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: extract key signals from text
        text_lower = text.lower()
        if "strong buy" in text_lower:
            action, conf = "BUY", 85
        elif "buy" in text_lower:
            action, conf = "BUY", 65
        elif "strong sell" in text_lower:
            action, conf = "SELL", 85
        elif "sell" in text_lower:
            action, conf = "SELL", 65
        else:
            action, conf = "HOLD", 50

        return {
            "action": action,
            "confidence": conf,
            "reasoning": text[:500],
        }

    except Exception as e:
        return {"error": str(e), "model": model_name}


def _build_analysis_prompt(ticker: str, data: dict) -> str:
    """Build the analysis prompt with all available data."""
    indicators = data.get("indicators", {})
    price = data.get("price", 0)
    news = data.get("news", [])
    options = data.get("options", {})

    prompt = f"""You are a professional trading analyst. Analyze this data and provide a trading recommendation.

TICKER: {ticker}
PRICE: ${price:.2f}

TECHNICAL INDICATORS:
- RSI(14): {indicators.get('rsi', 'N/A')}
- MACD: {indicators.get('macd', 'N/A')} | Signal: {indicators.get('macd_signal', 'N/A')} | Hist: {indicators.get('macd_hist', 'N/A')}
- SMA50: {indicators.get('sma50', 'N/A')} | SMA200: {indicators.get('sma200', 'N/A')}
- BB Upper: {indicators.get('bb_upper', 'N/A')} | Lower: {indicators.get('bb_lower', 'N/A')}
- ATR: {indicators.get('atr', 'N/A')}
- Volume Ratio: {indicators.get('vol_ratio', 'N/A')}

SIGNAL: {data.get('signal', 'N/A')} | Score: {data.get('score', 'N/A')}/100
"""

    if news:
        prompt += "\nRECENT NEWS:\n"
        for n in news[:5]:
            prompt += f"- {n.get('title', 'N/A')} (sentiment: {n.get('sentiment', 'N/A')})\n"

    if options:
        prompt += f"\nOPTIONS: IV Rank={options.get('iv_rank', 'N/A')}%, "
        prompt += f"Put/Call Ratio={options.get('put_call_ratio', 'N/A')}\n"

    prompt += """
Respond with ONLY a JSON object (no other text):
{
  "action": "BUY" or "SELL" or "HOLD",
  "confidence": 0-100,
  "reasoning": "2-3 sentence explanation",
  "risk_level": "LOW" or "MEDIUM" or "HIGH",
  "target_price": estimated target or null,
  "stop_loss": recommended stop-loss or null,
  "timeframe": "intraday" or "swing" or "position"
}"""

    return prompt


class MultiModelAnalyst:
    """Runs multiple Ollama models in parallel for consensus trading signals."""

    def __init__(self):
        self.available_models = []
        self.last_check = None
        self._pool = ThreadPoolExecutor(max_workers=3)

    def check_models(self) -> dict:
        """Check which models are available."""
        self.available_models = _get_available_models()
        self.last_check = datetime.utcnow().isoformat()
        return {
            "ollama_url": OLLAMA_URL,
            "available_models": [m["name"] for m in self.available_models],
            "total_models": len(MODELS),
            "available": len(self.available_models),
            "checked_at": self.last_check,
        }

    def analyze(self, ticker: str, data: dict) -> dict:
        """Run all available models in parallel and build consensus."""
        t0 = time.time()

        if not self.available_models:
            self.check_models()

        if not self.available_models:
            return {
                "error": "No Ollama models available",
                "recommendation": {"action": "HOLD", "confidence": 0},
            }

        prompt = _build_analysis_prompt(ticker, data)

        # Query all models in parallel
        futures = {}
        for model in self.available_models:
            future = self._pool.submit(_query_model, model["name"], prompt, model["timeout"])
            futures[future] = model

        results = []
        for future in futures:
            model = futures[future]
            try:
                result = future.result(timeout=model["timeout"] + 10)
                if result and not result.get("error"):
                    result["model"] = model["name"]
                    result["weight"] = model["weight"]
                    results.append(result)
            except Exception as e:
                results.append({"model": model["name"], "error": str(e)})

        # Build consensus
        consensus = self._build_consensus(results)

        return {
            "ticker": ticker,
            "consensus": consensus,
            "model_results": results,
            "models_queried": len(self.available_models),
            "models_responded": len([r for r in results if not r.get("error")]),
            "elapsed_s": round(time.time() - t0, 1),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    def _build_consensus(self, results: list) -> dict:
        """Combine model outputs into weighted consensus."""
        valid = [r for r in results if not r.get("error") and r.get("action")]

        if not valid:
            return {"action": "HOLD", "confidence": 0, "agreement": 0, "reason": "No valid model responses"}

        # Weighted voting
        buy_weight = 0
        sell_weight = 0
        hold_weight = 0
        total_weight = 0
        confidences = []

        for r in valid:
            w = r.get("weight", 1.0)
            conf = r.get("confidence", 50) / 100.0
            total_weight += w

            action = r.get("action", "HOLD").upper()
            if action == "BUY":
                buy_weight += w * conf
            elif action == "SELL":
                sell_weight += w * conf
            else:
                hold_weight += w * conf

            confidences.append(r.get("confidence", 50))

        # Determine consensus action
        scores = {"BUY": buy_weight, "SELL": sell_weight, "HOLD": hold_weight}
        consensus_action = max(scores, key=scores.get)
        consensus_strength = scores[consensus_action] / (total_weight + 1e-10)

        # Agreement: how many models agree
        actions = [r.get("action", "HOLD").upper() for r in valid]
        agreement = actions.count(consensus_action) / len(actions) * 100

        # Average confidence of agreeing models
        agreeing_confs = [r.get("confidence", 50) for r in valid if r.get("action", "").upper() == consensus_action]
        avg_confidence = sum(agreeing_confs) / len(agreeing_confs) if agreeing_confs else 50

        # Collect reasoning
        reasons = [r.get("reasoning", "") for r in valid if r.get("reasoning")]

        # Targets from agreeing models
        targets = [r.get("target_price") for r in valid if r.get("target_price") and r.get("action", "").upper() == consensus_action]
        stops = [r.get("stop_loss") for r in valid if r.get("stop_loss") and r.get("action", "").upper() == consensus_action]

        return {
            "action": consensus_action,
            "confidence": round(avg_confidence, 1),
            "strength": round(consensus_strength * 100, 1),
            "agreement_pct": round(agreement, 0),
            "votes": {a: actions.count(a) for a in set(actions)},
            "avg_target": round(sum(targets) / len(targets), 2) if targets else None,
            "avg_stop": round(sum(stops) / len(stops), 2) if stops else None,
            "risk_levels": [r.get("risk_level", "MEDIUM") for r in valid],
            "reasoning": reasons,
        }

    def get_status(self) -> dict:
        if not self.last_check:
            self.check_models()
        return {
            "model": "MultiModel-Consensus",
            "ollama_url": OLLAMA_URL,
            "available_models": [m["name"] for m in self.available_models],
            "total_configured": len(MODELS),
            "last_check": self.last_check,
        }


# Singleton
_analyst = MultiModelAnalyst()
def get_multi_analyst() -> MultiModelAnalyst:
    return _analyst
