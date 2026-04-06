"""
News Sentiment Scanner — Free, local, no API keys
===================================================
Scrapes financial news from free RSS feeds and Yahoo Finance.
Uses Ollama for sentiment analysis — 100% local.
"""

import warnings; warnings.filterwarnings("ignore")
import re, time, json
import concurrent.futures
from datetime import datetime
from typing import Optional
import requests
from xml.etree import ElementTree

# Free RSS feeds for financial news
RSS_FEEDS = {
    "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
    "marketwatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "cnbc": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "reuters_business": "https://www.reutersagency.com/feed/?taxonomy=best-sectors&post_type=best",
    "investing_com": "https://www.investing.com/rss/news.rss",
}


def _fetch_rss(url: str, timeout: int = 10) -> list:
    """Fetch and parse an RSS feed."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Trading Bot RSS Reader"
        }
        r = requests.get(url, timeout=timeout, headers=headers)
        r.raise_for_status()
        root = ElementTree.fromstring(r.content)

        items = []
        for item in root.iter("item"):
            title = item.findtext("title", "")
            desc = item.findtext("description", "")
            link = item.findtext("link", "")
            pub_date = item.findtext("pubDate", "")
            if title:
                items.append({
                    "title": title.strip(),
                    "description": desc.strip()[:200] if desc else "",
                    "link": link.strip(),
                    "published": pub_date.strip(),
                })
        return items
    except Exception:
        return []


def _fetch_yahoo_news(ticker: str) -> list:
    """Get news for a specific ticker from Yahoo Finance."""
    try:
        import yfinance as yf
        tkr = yf.Ticker(ticker)
        news = tkr.news or []
        items = []
        for n in news[:15]:
            items.append({
                "title": n.get("title", ""),
                "description": n.get("summary", "")[:200] if n.get("summary") else "",
                "link": n.get("link", ""),
                "published": datetime.fromtimestamp(n.get("providerPublishTime", 0)).isoformat() if n.get("providerPublishTime") else "",
                "source": n.get("publisher", "Yahoo Finance"),
            })
        return items
    except Exception:
        return []


def _keyword_sentiment(text: str) -> float:
    """Quick keyword-based sentiment score (-1 to +1). Fast fallback when Ollama is slow."""
    text = text.lower()

    bullish = [
        "surge", "rally", "soar", "jump", "gain", "rise", "bullish", "upgrade",
        "beat", "record", "high", "growth", "profit", "strong", "buy", "outperform",
        "positive", "upside", "breakout", "momentum", "innovation", "partnership",
        "dividend", "expansion", "revenue growth", "earnings beat", "analyst upgrade"
    ]
    bearish = [
        "crash", "plunge", "fall", "drop", "decline", "bearish", "downgrade",
        "miss", "loss", "weak", "sell", "underperform", "negative", "risk",
        "lawsuit", "investigation", "warning", "layoff", "recession", "debt",
        "bankruptcy", "scandal", "fraud", "earnings miss", "analyst downgrade"
    ]

    bull_count = sum(1 for word in bullish if word in text)
    bear_count = sum(1 for word in bearish if word in text)

    total = bull_count + bear_count
    if total == 0:
        return 0.0
    return round((bull_count - bear_count) / total, 3)


def scan_ticker_news(ticker: str, use_ollama: bool = True) -> dict:
    """Get and analyze news for a specific ticker."""
    t0 = time.time()

    # Get ticker-specific news from Yahoo
    headlines = _fetch_yahoo_news(ticker)

    if not headlines:
        return {
            "ticker": ticker,
            "headline_count": 0,
            "sentiment": "NEUTRAL",
            "sentiment_score": 0.0,
            "headlines": [],
            "elapsed_s": round(time.time() - t0, 1),
        }

    # Quick keyword sentiment for each headline
    for h in headlines:
        h["keyword_sentiment"] = _keyword_sentiment(h["title"] + " " + h.get("description", ""))

    avg_keyword = sum(h["keyword_sentiment"] for h in headlines) / len(headlines)

    # Ollama deep analysis if available
    ai_analysis = None
    if use_ollama:
        try:
            from ollama_brain import analyze_news_sentiment
            titles = [h["title"] for h in headlines]
            ai_analysis = analyze_news_sentiment(ticker, titles)
        except Exception:
            pass

    # Combined sentiment
    if ai_analysis and not ai_analysis.get("parse_error"):
        sentiment_score = ai_analysis.get("sentiment_score", avg_keyword)
        sentiment = ai_analysis.get("overall_sentiment", "NEUTRAL")
    else:
        sentiment_score = avg_keyword
        if sentiment_score > 0.3:
            sentiment = "BULLISH"
        elif sentiment_score > 0.1:
            sentiment = "SLIGHTLY_BULLISH"
        elif sentiment_score < -0.3:
            sentiment = "BEARISH"
        elif sentiment_score < -0.1:
            sentiment = "SLIGHTLY_BEARISH"
        else:
            sentiment = "NEUTRAL"

    return {
        "ticker": ticker,
        "headline_count": len(headlines),
        "sentiment": sentiment,
        "sentiment_score": round(sentiment_score, 3),
        "headlines": headlines[:10],
        "ai_analysis": ai_analysis,
        "elapsed_s": round(time.time() - t0, 1),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


def scan_market_news() -> dict:
    """Scan general market news from RSS feeds."""
    t0 = time.time()
    all_headlines = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futures = {}
        for name, url in RSS_FEEDS.items():
            futures[pool.submit(_fetch_rss, url)] = name
        for f in concurrent.futures.as_completed(futures):
            source = futures[f]
            items = f.result()
            for item in items:
                item["source"] = source
            all_headlines.extend(items)

    # Score each headline
    for h in all_headlines:
        h["keyword_sentiment"] = _keyword_sentiment(h["title"] + " " + h.get("description", ""))

    # Overall market sentiment
    if all_headlines:
        avg = sum(h["keyword_sentiment"] for h in all_headlines) / len(all_headlines)
    else:
        avg = 0.0

    # Extract mentioned tickers
    ticker_mentions = {}
    ticker_pattern = re.compile(r'\b[A-Z]{2,5}\b')
    common_words = {"THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN",
                    "HER", "WAS", "ONE", "OUR", "OUT", "HAS", "NEW", "NOW", "OLD",
                    "SEE", "WAY", "MAY", "WHO", "GDP", "CEO", "IPO", "ETF", "SEC",
                    "FED", "FBI", "CIA", "USA", "FDA", "NYSE", "DOW", "API", "RSS",
                    "IRS", "IMF", "WHO", "NATO"}
    for h in all_headlines:
        matches = ticker_pattern.findall(h["title"])
        for m in matches:
            if m not in common_words and len(m) >= 2:
                if m not in ticker_mentions:
                    ticker_mentions[m] = {"count": 0, "sentiments": []}
                ticker_mentions[m]["count"] += 1
                ticker_mentions[m]["sentiments"].append(h["keyword_sentiment"])

    # Top mentioned tickers
    trending = []
    for ticker, data in sorted(ticker_mentions.items(), key=lambda x: x[1]["count"], reverse=True)[:20]:
        avg_sent = sum(data["sentiments"]) / len(data["sentiments"])
        trending.append({
            "ticker": ticker,
            "mentions": data["count"],
            "avg_sentiment": round(avg_sent, 3),
        })

    return {
        "total_headlines": len(all_headlines),
        "market_sentiment": round(avg, 3),
        "market_mood": "BULLISH" if avg > 0.15 else "BEARISH" if avg < -0.15 else "NEUTRAL",
        "trending_tickers": trending,
        "top_headlines": sorted(all_headlines, key=lambda x: abs(x["keyword_sentiment"]), reverse=True)[:15],
        "elapsed_s": round(time.time() - t0, 1),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


def scan_news_batch(tickers: list, use_ollama: bool = False) -> list:
    """Scan news for multiple tickers in parallel."""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(scan_ticker_news, t, use_ollama): t for t in tickers}
        for f in concurrent.futures.as_completed(futures):
            r = f.result()
            if r:
                results.append(r)
    results.sort(key=lambda x: abs(x["sentiment_score"]), reverse=True)
    return results


if __name__ == "__main__":
    print("Scanning market news...")
    market = scan_market_news()
    print(f"Found {market['total_headlines']} headlines")
    print(f"Market mood: {market['market_mood']} ({market['market_sentiment']:+.3f})")

    if market["trending_tickers"]:
        print(f"\nTrending tickers:")
        for t in market["trending_tickers"][:10]:
            print(f"  {t['ticker']:<6} mentions:{t['mentions']:>3}  sentiment:{t['avg_sentiment']:+.3f}")

    print(f"\nTop headlines:")
    for h in market["top_headlines"][:10]:
        sent = h["keyword_sentiment"]
        icon = "+" if sent > 0 else "-" if sent < 0 else " "
        print(f"  [{icon}] {h['title'][:80]}")

    # Test ticker-specific news
    print("\n\nScanning AAPL news...")
    aapl = scan_ticker_news("AAPL", use_ollama=False)
    print(f"Sentiment: {aapl['sentiment']} ({aapl['sentiment_score']:+.3f})")
    for h in aapl["headlines"][:5]:
        print(f"  [{h['keyword_sentiment']:+.2f}] {h['title'][:70]}")
