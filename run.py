"""
NEAT Trading Bot — Standalone Launcher
=======================================
Run this script to start the ENTIRE trading system without Claude Code.
  python run.py

What it does:
  1. Checks Ollama is running (starts it if not)
  2. Starts the FastAPI server + dashboard
  3. Auto-starts the autonomous trading agent
  4. Opens the dashboard in your browser
  5. Runs forever — Ctrl+C to stop

No Claude Code needed. Fully local.
"""

import subprocess, sys, os, time, threading, signal, webbrowser, json
from pathlib import Path

HOST = "127.0.0.1"
PORT = 8000
BASE_DIR = Path(__file__).parent
URL = f"http://{HOST}:{PORT}"

# ─── Colors for terminal ─────────────────────────────────────────
class C:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

def banner():
    print(f"""{C.CYAN}{C.BOLD}
╔══════════════════════════════════════════════════════╗
║           NEAT Trading Bot v2.0 — Standalone        ║
║         Local AI-Powered Autonomous Trader           ║
╠══════════════════════════════════════════════════════╣
║  Neural Networks: NEAT + PyTorch LSTM + DEAP Evo    ║
║  AI Models:       Gemma4 31B + Gemma4 + Qwen (vote) ║
║  Self-Healing:    Auto-fix errors via local Ollama   ║
║  Dashboard:       {URL:<37s} ║
╚══════════════════════════════════════════════════════╝{C.RESET}
""")

def log(msg, level="info"):
    colors = {"info": C.CYAN, "ok": C.GREEN, "warn": C.YELLOW, "err": C.RED}
    ts = time.strftime("%H:%M:%S")
    print(f"{C.DIM}{ts}{C.RESET} {colors.get(level, C.CYAN)}{msg}{C.RESET}")


# ─── Check dependencies ──────────────────────────────────────────
def check_deps():
    log("Checking Python dependencies...")
    missing = []
    for mod in ["fastapi", "uvicorn", "jose", "bcrypt", "numpy", "yfinance"]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        log(f"Installing missing: {', '.join(missing)}", "warn")
        req_file = BASE_DIR / "requirements.txt"
        if req_file.exists():
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_file), "-q"], check=False)
        else:
            subprocess.run([sys.executable, "-m", "pip", "install"] + missing + ["-q"], check=False)
    log("Dependencies OK", "ok")


# ─── Check Ollama ────────────────────────────────────────────────
def check_ollama():
    log("Checking Ollama...")
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        log(f"Ollama online — {len(models)} models: {', '.join(models[:5])}", "ok")
        return True
    except Exception:
        log("Ollama not running. Attempting to start...", "warn")
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(3)
            import requests
            r = requests.get("http://localhost:11434/api/tags", timeout=3)
            log("Ollama started successfully", "ok")
            return True
        except Exception:
            log("Could not start Ollama. AI features will be limited.", "err")
            log("Install from: https://ollama.com", "warn")
            return False


# ─── Start server ────────────────────────────────────────────────
def start_server():
    log(f"Starting server on {URL}...")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server:app",
         "--host", HOST, "--port", str(PORT), "--log-level", "warning"],
        cwd=str(BASE_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    # Wait for server to be ready
    for i in range(30):
        try:
            import requests
            r = requests.get(f"{URL}/docs", timeout=2)
            if r.status_code == 200:
                log(f"Server running at {URL}", "ok")
                return proc
        except Exception:
            pass
        time.sleep(1)
    log("Server failed to start!", "err")
    return proc


# ─── Auto-start agent ────────────────────────────────────────────
def auto_start_agent():
    """Start the autonomous trading agent after server is up."""
    time.sleep(3)
    try:
        import requests
        # Login
        r = requests.post(f"{URL}/auth/token",
                          data={"username": "trader", "password": "ChangeMe123!"}, timeout=5)
        token = r.json().get("access_token")
        if not token:
            log("Auto-login failed", "err")
            return
        headers = {"Authorization": f"Bearer {token}"}

        # Check agent status
        r = requests.get(f"{URL}/agent/status", headers=headers, timeout=5)
        status = r.json()
        if status.get("running"):
            log("Agent already running", "ok")
            return

        # Start agent
        r = requests.post(f"{URL}/agent/start", headers=headers, timeout=10)
        log("Autonomous trading agent STARTED", "ok")
        log("Agent will scan all sectors + crypto continuously", "info")
    except Exception as e:
        log(f"Agent auto-start failed: {e}", "err")


# ─── Terminal monitor ────────────────────────────────────────────
def terminal_monitor():
    """Print agent terminal output to this console."""
    time.sleep(5)
    idx = 0
    try:
        import requests
        # Login
        r = requests.post(f"{URL}/auth/token",
                          data={"username": "trader", "password": "ChangeMe123!"}, timeout=5)
        token = r.json().get("access_token")
        headers = {"Authorization": f"Bearer {token}"}

        while True:
            try:
                r = requests.get(f"{URL}/agent/terminal?since={idx}", headers=headers, timeout=5)
                d = r.json()
                for m in d.get("messages", []):
                    t = m.get("time", "")[-8:]
                    msg = m.get("msg", "")
                    mtype = m.get("type", "info")
                    color = {
                        "scan": C.CYAN, "signal": C.GREEN, "trade": C.YELLOW,
                        "model": "\033[95m", "decision": "\033[94m", "error": C.RED
                    }.get(mtype, C.DIM)
                    print(f"  {C.DIM}{t}{C.RESET} {color}{msg}{C.RESET}")
                idx = d.get("next_idx", idx)
            except Exception:
                pass
            time.sleep(3)
    except Exception:
        pass


# ─── Main ────────────────────────────────────────────────────────
def main():
    banner()
    check_deps()
    check_ollama()

    server_proc = start_server()
    if not server_proc:
        sys.exit(1)

    # Auto-start agent in background
    threading.Thread(target=auto_start_agent, daemon=True).start()

    # Terminal monitor in background
    threading.Thread(target=terminal_monitor, daemon=True).start()

    # Open browser
    log("Opening dashboard in browser...", "info")
    time.sleep(1)
    webbrowser.open(URL)

    print(f"""
{C.GREEN}{C.BOLD}System is running!{C.RESET}
{C.CYAN}Dashboard:{C.RESET} {URL}
{C.CYAN}API Docs:{C.RESET}  {URL}/docs
{C.CYAN}Login:{C.RESET}     trader / ChangeMe123!

{C.DIM}Press Ctrl+C to stop everything.{C.RESET}
{C.DIM}Agent terminal output will appear below:{C.RESET}
{"─" * 50}
""")

    def shutdown(sig, frame):
        print(f"\n{C.YELLOW}Shutting down...{C.RESET}")
        server_proc.terminate()
        server_proc.wait(timeout=5)
        print(f"{C.GREEN}Stopped. Goodbye!{C.RESET}")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Keep alive
    try:
        server_proc.wait()
    except KeyboardInterrupt:
        shutdown(None, None)


if __name__ == "__main__":
    main()
