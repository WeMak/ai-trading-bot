"""
Master Orchestrator — Autonomous Trading Pipeline
====================================================
Coordinates all scanners into a single pipeline:
1. Scan all 11 sectors (110 stocks)
2. Scan 30 crypto assets
3. Pull news sentiment
4. Run Ollama AI on top picks
5. Analyze options chains
6. Generate trade recommendations
7. Execute via paper trader

Runs on a schedule or on-demand.
"""

import warnings; warnings.filterwarnings("ignore")
import json, time, os, threading
from datetime import datetime
from typing import Optional

from sector_scanner import scan_all_sectors, scan_sector, ALL_SECTOR_TICKERS
from crypto_scanner import scan_all_crypto
from news_scanner import scan_ticker_news, scan_market_news, scan_news_batch
from options_analyzer import analyze_options, scan_options_batch
from earnings_scanner import analyze_earnings, scan_earnings_batch
from ollama_brain import (
    check_ollama, analyze_trade_signal, generate_market_brief
)
from trading_agent import analyze
from trade_finder import find_trades
from paper_trader import PaperTrader

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "scan_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


class TradingOrchestrator:
    def __init__(self):
        self.paper = PaperTrader()
        self.last_scan = None
        self.is_running = False
        self._lock = threading.Lock()

    def _save_result(self, name: str, data: dict):
        """Save scan result to disk."""
        path = os.path.join(RESULTS_DIR, f"{name}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def full_scan(self, use_ollama: bool = True, auto_trade: bool = False) -> dict:
        """
        Run the complete scanning pipeline.
        Returns combined results from all scanners.
        """
        with self._lock:
            if self.is_running:
                return {"error": "Scan already in progress"}
            self.is_running = True

        t0 = time.time()
        results = {"timestamp": datetime.utcnow().isoformat() + "Z", "stages": {}}

        try:
            # ── Stage 1: Sector Scan ─────────────────────────────
            print("[1/6] Scanning 110 stocks across 11 sectors...")
            sector_data = scan_all_sectors()
            results["stages"]["sectors"] = {
                "elapsed_s": sector_data["elapsed_s"],
                "scanned": sector_data["total_scanned"],
                "sector_strength": sector_data["sector_strength"],
                "top_buys": sector_data["top_buys"][:10],
                "top_sells": sector_data["top_sells"][:10],
            }
            self._save_result("sectors", sector_data)
            print(f"    Done: {sector_data['total_scanned']} stocks in {sector_data['elapsed_s']}s")

            # ── Stage 2: Crypto Scan ─────────────────────────────
            print("[2/6] Scanning 30 crypto assets...")
            crypto_data = scan_all_crypto()
            results["stages"]["crypto"] = {
                "elapsed_s": crypto_data["elapsed_s"],
                "scanned": crypto_data["total_scanned"],
                "market_metrics": crypto_data["market_metrics"],
                "top_buys": crypto_data["top_buys"][:5],
                "top_sells": crypto_data["top_sells"][:5],
            }
            self._save_result("crypto", crypto_data)
            print(f"    Done: {crypto_data['total_scanned']} cryptos in {crypto_data['elapsed_s']}s")

            # ── Stage 3: News Scan ───────────────────────────────
            print("[3/6] Scanning market news...")
            market_news = scan_market_news()

            # Get news for top picks
            top_tickers = [t["ticker"] for t in sector_data["top_buys"][:5]]
            top_tickers += [t["symbol"] for t in crypto_data["top_buys"][:3]]
            ticker_news = scan_news_batch(top_tickers, use_ollama=False)

            results["stages"]["news"] = {
                "market_mood": market_news["market_mood"],
                "market_sentiment": market_news["market_sentiment"],
                "total_headlines": market_news["total_headlines"],
                "trending": market_news["trending_tickers"][:10],
                "ticker_news": [{
                    "ticker": n["ticker"],
                    "sentiment": n["sentiment"],
                    "score": n["sentiment_score"],
                    "headlines": len(n["headlines"]),
                } for n in ticker_news],
            }
            self._save_result("news", {
                "market": market_news,
                "ticker_news": ticker_news,
            })
            print(f"    Done: {market_news['total_headlines']} headlines, mood: {market_news['market_mood']}")

            # ── Stage 4: Earnings Scan ───────────────────────────
            print("[4/7] Scanning earnings dates + expected moves...")
            earnings_tickers = [t["ticker"] for t in sector_data.get("top_buys", [])[:15]]
            earnings_tickers += [t["ticker"] for t in sector_data.get("top_sells", [])[:10]]
            # Dedupe
            earnings_tickers = list(dict.fromkeys(earnings_tickers))
            earnings_data = scan_earnings_batch(earnings_tickers)
            results["stages"]["earnings"] = {
                "scanned": earnings_data["total_scanned"],
                "with_earnings": earnings_data["with_earnings"],
                "imminent": earnings_data["imminent"],
                "upcoming_7_30d": earnings_data["upcoming_7_30d"],
            }
            self._save_result("earnings", earnings_data)
            print(f"    Done: {earnings_data['with_earnings']} stocks with upcoming earnings, {len(earnings_data['imminent'])} this week")

            # ── Stage 5: Options Analysis ────────────────────────
            print("[5/7] Analyzing full options chains (calls + puts, weeklies to LEAPs)...")
            options_targets = []
            for t in sector_data["top_buys"][:5]:
                options_targets.append({"ticker": t["ticker"], "signal": "BOTH"})
            for t in sector_data["top_sells"][:3]:
                options_targets.append({"ticker": t["ticker"], "signal": "BOTH"})
            # Also scan earnings-imminent tickers
            for e in earnings_data["imminent"][:3]:
                if e["ticker"] not in [o["ticker"] for o in options_targets]:
                    options_targets.append({"ticker": e["ticker"], "signal": "BOTH"})

            options_results = scan_options_batch(options_targets, top_n=5)
            results["stages"]["options"] = [{
                "ticker": o["ticker"],
                "iv_rank": o["iv_rank"],
                "total_calls": o["total_calls_analyzed"],
                "total_puts": o["total_puts_analyzed"],
                "total_analyzed": o["total_contracts_analyzed"],
                "top_calls": o["top_calls"][:3],
                "top_puts": o["top_puts"][:3],
                "unusual_activity": o.get("unusual_activity", [])[:3],
                "value_plays": o.get("value_plays", [])[:3],
            } for o in options_results]
            self._save_result("options", options_results)
            print(f"    Done: {len(options_results)} tickers, {sum(o['total_contracts_analyzed'] for o in options_results)} contracts analyzed")

            # ── Stage 6: Ollama AI Analysis ──────────────────────
            ai_results = []
            if use_ollama:
                ollama_status = check_ollama()
                if ollama_status["status"] == "online":
                    print("[6/7] Running AI analysis on top picks...")
                    all_picks = sector_data["top_buys"][:3] + crypto_data["top_buys"][:2]
                    for pick in all_picks:
                        print(f"    AI analyzing {pick.get('ticker', pick.get('symbol', '?'))}...")
                        ai = analyze_trade_signal(pick)
                        ai["ticker"] = pick.get("ticker", pick.get("symbol"))
                        ai_results.append(ai)

                    print("    Generating market brief...")
                    all_top = sector_data["top_buys"][:5] + sector_data["top_sells"][:3]
                    brief = generate_market_brief(sector_data, all_top)
                    results["stages"]["ai_brief"] = brief
                    self._save_result("ai_brief", brief)
                else:
                    print("[6/7] Ollama offline — skipping AI analysis")
            else:
                print("[6/7] Ollama disabled — skipping AI analysis")

            results["stages"]["ai_analysis"] = ai_results
            if ai_results:
                self._save_result("ai_analysis", ai_results)

            # ── Stage 7: Trade Recommendations ───────────────────
            print("[7/7] Compiling trade recommendations...")
            recommendations = self._compile_recommendations(
                sector_data, crypto_data, ticker_news, options_results, ai_results
            )
            results["recommendations"] = recommendations
            self._save_result("recommendations", recommendations)

            # ── Auto-trade if enabled ────────────────────────────
            if auto_trade and recommendations["actionable"]:
                print("    Auto-executing paper trades...")
                trade_results = self._execute_trades(recommendations)
                results["trades_executed"] = trade_results

            # ── Summary ──────────────────────────────────────────
            results["elapsed_s"] = round(time.time() - t0, 1)
            results["summary"] = {
                "total_assets_scanned": sector_data["total_scanned"] + crypto_data["total_scanned"],
                "stock_buys": len(sector_data["top_buys"]),
                "stock_sells": len(sector_data["top_sells"]),
                "crypto_buys": len(crypto_data["top_buys"]),
                "crypto_sells": len(crypto_data["top_sells"]),
                "market_mood": market_news["market_mood"],
                "crypto_fear_greed": crypto_data["market_metrics"]["fear_greed_proxy"],
                "earnings_imminent": len(earnings_data.get("imminent", [])),
                "earnings_upcoming": earnings_data.get("with_earnings", 0),
                "options_analyzed": sum(o.get("total_contracts_analyzed", 0) for o in options_results),
                "ai_enhanced": len(ai_results) > 0,
                "recommendations": len(recommendations.get("trades", [])),
            }

            self.last_scan = results
            self._save_result("full_scan", results)
            print(f"\nFull scan complete in {results['elapsed_s']}s")
            print(f"Scanned {results['summary']['total_assets_scanned']} assets")
            print(f"Found {results['summary']['recommendations']} trade recommendations")

        except Exception as e:
            results["error"] = str(e)
            print(f"Scan error: {e}")

        finally:
            with self._lock:
                self.is_running = False

        return results

    def _compile_recommendations(self, sector_data, crypto_data,
                                  ticker_news, options_data, ai_data) -> dict:
        """Merge all signals into ranked trade recommendations."""
        trades = []

        # Process stock picks
        for pick in sector_data.get("top_buys", [])[:10]:
            ticker = pick["ticker"]

            # Find matching news
            news_sent = 0.0
            for n in ticker_news:
                if n["ticker"] == ticker:
                    news_sent = n["sentiment_score"]
                    break

            # Find matching options
            best_option = None
            for opt in options_data:
                if opt.get("ticker") == ticker and opt.get("top_picks"):
                    best_option = opt["top_picks"][0]
                    break

            # Find matching AI analysis
            ai_rec = None
            for ai in ai_data:
                if ai.get("ticker") == ticker:
                    ai_rec = ai
                    break

            # Composite score
            tech_score = pick["confidence"]
            news_boost = news_sent * 10  # -10 to +10
            ai_boost = 0
            if ai_rec and not ai_rec.get("parse_error"):
                ai_conf = ai_rec.get("confidence", 50)
                if ai_rec.get("action", "").endswith("BUY"):
                    ai_boost = (ai_conf - 50) * 0.3
                elif ai_rec.get("action", "").endswith("SELL"):
                    ai_boost = -(ai_conf - 50) * 0.3

            composite = min(99, max(1, tech_score + news_boost + ai_boost))

            trade = {
                "ticker": ticker,
                "asset_type": "STOCK",
                "signal": pick["signal"],
                "composite_score": round(composite, 1),
                "technical_confidence": pick["confidence"],
                "news_sentiment": round(news_sent, 3),
                "ai_recommendation": ai_rec.get("action") if ai_rec else None,
                "price": pick["price"],
                "indicators": pick["indicators"],
                "notes": pick["notes"],
                "best_option": best_option,
            }
            trades.append(trade)

        # Process crypto picks
        for pick in crypto_data.get("top_buys", [])[:5]:
            trade = {
                "ticker": pick["symbol"],
                "asset_type": "CRYPTO",
                "signal": pick["signal"],
                "composite_score": pick["confidence"],
                "technical_confidence": pick["confidence"],
                "news_sentiment": 0.0,
                "price": pick["price"],
                "indicators": pick["indicators"],
                "notes": pick["notes"],
                "best_option": None,
            }
            trades.append(trade)

        # Sort by composite score
        trades.sort(key=lambda x: x["composite_score"], reverse=True)

        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "trades": trades[:15],
            "actionable": len([t for t in trades if t["composite_score"] > 65]) > 0,
            "high_confidence": [t for t in trades if t["composite_score"] > 75][:5],
        }

    def _execute_trades(self, recommendations: dict) -> list:
        """Execute high-confidence trades via paper trader."""
        executed = []
        for trade in recommendations.get("high_confidence", []):
            if trade["composite_score"] < 70:
                continue

            if trade["asset_type"] == "STOCK" and trade["best_option"]:
                # Enter options trade
                opt = trade["best_option"]
                result = self.paper.enter_trade(
                    ticker=trade["ticker"],
                    direction="CALL" if trade["signal"] == "BUY" else "PUT",
                    entry_price=trade["price"],
                    asset_type="OPTIONS",
                    signal_strength=trade["composite_score"] / 100,
                    reason=f"AI scan: {trade['notes'][0][:50]}",
                    est_premium=opt.get("lastPrice", 2.0),
                    iv_rank=0.3,
                    strike=opt.get("strike"),
                    expiry=opt.get("expiry"),
                )
                if result:
                    executed.append(result)

            elif trade["asset_type"] == "CRYPTO":
                result = self.paper.enter_trade(
                    ticker=trade["ticker"] + "-USD",
                    direction="LONG" if trade["signal"] == "BUY" else "SHORT",
                    entry_price=trade["price"],
                    asset_type="CRYPTO",
                    signal_strength=trade["composite_score"] / 100,
                    reason=f"AI scan: {trade['notes'][0][:50]}",
                )
                if result:
                    executed.append(result)

        return executed

    def quick_scan(self) -> dict:
        """Fast scan — just the original 30 tickers, no AI."""
        return find_trades(10)

    def analyze_ticker(self, ticker: str, with_options: bool = True,
                        with_news: bool = True, with_ai: bool = True) -> dict:
        """Deep analysis of a single ticker."""
        t0 = time.time()
        result = {"ticker": ticker}

        # Technical analysis with chart
        tech = analyze(ticker)
        result["technical"] = tech

        if with_news:
            news = scan_ticker_news(ticker, use_ollama=False)
            result["news"] = news

        if with_options and not tech.get("is_crypto"):
            # Full chain — both calls and puts
            opts = analyze_options(ticker, "BOTH", min_days=0, max_days=730, top_n=10)
            result["options"] = opts

        # Earnings analysis (stocks only)
        if not tech.get("is_crypto"):
            earnings = analyze_earnings(ticker)
            result["earnings"] = earnings

        if with_ai:
            ollama = check_ollama()
            if ollama["status"] == "online":
                ai = analyze_trade_signal(tech)
                result["ai_analysis"] = ai

        result["elapsed_s"] = round(time.time() - t0, 1)
        return result

    def get_portfolio(self) -> dict:
        """Get paper trading portfolio status."""
        self.paper.update_positions()
        return self.paper.get_summary()


# Singleton
_orchestrator = TradingOrchestrator()


def get_orchestrator() -> TradingOrchestrator:
    return _orchestrator


if __name__ == "__main__":
    import sys
    orch = get_orchestrator()

    if len(sys.argv) > 1 and sys.argv[1] == "full":
        print("Running FULL SCAN pipeline...")
        result = orch.full_scan(use_ollama=True, auto_trade=False)
        print(json.dumps(result.get("summary", {}), indent=2))
    elif len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
        print(f"Deep analysis: {ticker}")
        result = orch.analyze_ticker(ticker)
        if "technical" in result and "error" not in result["technical"]:
            t = result["technical"]
            print(f"Signal: {t['signal']} ({t['confidence']}%)")
            print(f"Price: ${t['price']:,.4f}")
        if "ai_analysis" in result:
            ai = result["ai_analysis"]
            print(f"AI says: {ai.get('action', 'N/A')} — {ai.get('reasoning', '')[:100]}")
    else:
        print("Quick scan (30 tickers)...")
        result = orch.quick_scan()
        print(f"Done in {result['elapsed_s']}s — {result['signals_found']} signals")
        for i, t in enumerate(result["top_trades"][:5], 1):
            print(f"  {i}. {t['ticker']:<6} {t['signal']:<5} {t['confidence']:>4.0f}%")
