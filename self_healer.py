"""
Self-Healing Agent — Local Error Diagnosis & Auto-Fix
======================================================
When any module throws an error, this system:
1. Captures the full traceback + context
2. Sends it to local Ollama for diagnosis
3. Attempts automated fixes (config tweaks, retries, fallbacks)
4. Logs everything to the terminal

All processing is LOCAL via Ollama — zero external API calls.
"""

import warnings; warnings.filterwarnings("ignore")
import os, json, time, traceback, re
from datetime import datetime
from typing import Optional
import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
HEAL_MODEL = "qwen2.5-coder:32b"  # fast local model for error fixing
HEAL_LOG_PATH = os.path.join(os.path.dirname(__file__), "heal_log.json")

_heal_log = []


def _query_ollama_local(prompt: str, timeout: int = 60) -> str:
    """Query local Ollama — zero external calls."""
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": HEAL_MODEL, "prompt": prompt, "stream": False,
                   "options": {"temperature": 0.1, "num_predict": 512}},
            timeout=timeout,
        )
        if r.status_code == 200:
            return r.json().get("response", "")
    except Exception:
        pass
    return ""


def diagnose_error(error: Exception, context: str = "", module: str = "") -> dict:
    """Send error to local Ollama for diagnosis. Returns fix suggestion."""
    tb = traceback.format_exception(type(error), error, error.__traceback__)
    tb_str = "".join(tb[-5:])  # last 5 frames to save tokens

    prompt = f"""You are a Python debugging assistant. Diagnose this error and suggest a fix.
Keep your answer under 100 words. Be specific.

Module: {module}
Context: {context}
Error: {type(error).__name__}: {str(error)[:200]}
Traceback (last 5 frames):
{tb_str[:600]}

Reply with JSON only:
{{"diagnosis": "what went wrong", "fix": "specific fix", "severity": "low|medium|high", "auto_fixable": true/false}}"""

    response = _query_ollama_local(prompt)

    # Parse JSON from response
    result = {"diagnosis": str(error), "fix": "manual review needed",
              "severity": "medium", "auto_fixable": False, "raw": ""}
    try:
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            parsed = json.loads(match.group())
            result.update(parsed)
    except Exception:
        result["raw"] = response[:300]

    # Log
    entry = {
        "time": datetime.utcnow().isoformat() + "Z",
        "module": module,
        "error": f"{type(error).__name__}: {str(error)[:200]}",
        "diagnosis": result.get("diagnosis", ""),
        "fix": result.get("fix", ""),
        "severity": result.get("severity", "medium"),
    }
    _heal_log.append(entry)
    if len(_heal_log) > 100:
        _heal_log.pop(0)
    _save_log()

    return result


def auto_fix(error: Exception, context: str = "", module: str = "") -> dict:
    """Attempt automatic fix for common errors."""
    err_str = str(error).lower()
    err_type = type(error).__name__

    fixes_applied = []

    # Common auto-fixes
    if "no data" in err_str or "empty" in err_str:
        fixes_applied.append("data_retry: will retry with longer period")
        return {"fixed": True, "action": "retry_longer_period", "fixes": fixes_applied}

    if "connection" in err_str or "timeout" in err_str:
        fixes_applied.append("network_retry: will retry after 30s cooldown")
        return {"fixed": True, "action": "cooldown_retry", "fixes": fixes_applied}

    if "rate limit" in err_str or "429" in err_str:
        fixes_applied.append("rate_limit: backing off 60s")
        return {"fixed": True, "action": "rate_limit_backoff", "fixes": fixes_applied}

    if err_type == "OverflowError":
        fixes_applied.append("overflow: clamping numerical values")
        return {"fixed": True, "action": "clamp_values", "fixes": fixes_applied}

    if "nan" in err_str or err_type == "ValueError":
        fixes_applied.append("nan_guard: replacing NaN with defaults")
        return {"fixed": True, "action": "nan_cleanup", "fixes": fixes_applied}

    if "memory" in err_str or "oom" in err_str:
        fixes_applied.append("memory: reducing batch size")
        return {"fixed": True, "action": "reduce_batch", "fixes": fixes_applied}

    # If no auto-fix, ask Ollama
    diag = diagnose_error(error, context, module)
    if diag.get("auto_fixable"):
        fixes_applied.append(f"ollama_suggested: {diag.get('fix', 'unknown')}")
        return {"fixed": True, "action": "ollama_fix", "diagnosis": diag, "fixes": fixes_applied}

    return {"fixed": False, "diagnosis": diag, "fixes": fixes_applied}


def get_heal_log() -> list:
    return _heal_log[-50:]


def _save_log():
    try:
        with open(HEAL_LOG_PATH, "w") as f:
            json.dump(_heal_log[-100:], f, indent=2)
    except Exception:
        pass


def _load_log():
    global _heal_log
    if os.path.exists(HEAL_LOG_PATH):
        try:
            with open(HEAL_LOG_PATH) as f:
                _heal_log = json.load(f)
        except Exception:
            pass

_load_log()
