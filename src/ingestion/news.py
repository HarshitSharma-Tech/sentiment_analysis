"""
Free News Ingestion Module
Collects recent market news using 100% free public RSS feeds (CoinDesk, Cointelegraph, Yahoo Finance).
Provides fallbacks for local CSV/JSON imports and optional Tavily/Exa connectors.
Deduplicates articles, extracts UTC publication timestamps, and computes headline sentiment.
Kept strictly separate from the primary quantitative price prediction model features.
"""

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import feedparser
import pandas as pd
import requests

from src.database import DatabaseManager
from src.sentiment import SentimentAnalyzer, get_analyzer

logger = logging.getLogger(__name__)


class NewsIngestion:
    """Collects and standardizes financial news from free RSS feeds and optional search APIs."""

    DEFAULT_FEEDS = [
        {"name": "CoinDesk", "url": "https://www.coindesk.com/arc/outboundfeeds/rss/"},
        {"name": "Cointelegraph", "url": "https://cointelegraph.com/rss"},
        {"name": "Google News Crypto", "url": "https://news.google.com/rss/search?q=Bitcoin+crypto&hl=en-US&gl=US&ceid=US:en"},
    ]

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        analyzer: Optional[SentimentAnalyzer] = None,
        tavily_api_key: Optional[str] = None,
        exa_api_key: Optional[str] = None,
    ):
        self.db = db_manager
        self.analyzer = analyzer or get_analyzer()
        self.tavily_api_key = tavily_api_key or ""
        self.exa_api_key = exa_api_key or ""

    @staticmethod
    def generate_article_id(url: str, title: str) -> str:
        """Create a deterministic hash identifier for deduplication."""
        raw = f"{url.strip().lower()}|{title.strip().lower()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def fetch_rss_feeds(
        self,
        feeds: Optional[List[Dict[str, str]]] = None,
        max_per_feed: int = 25,
    ) -> List[Dict[str, Any]]:
        """
        Fetch news from publicly accessible financial RSS feeds (100% Free, no API keys).
        """
        active_feeds = feeds or self.DEFAULT_FEEDS
        articles: List[Dict[str, Any]] = []
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        for feed_info in active_feeds:
            name = feed_info["name"]
            url = feed_info["url"]
            logger.info(f"Fetching RSS feed: {name} ({url})...")
            try:
                resp = requests.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                    timeout=5,
                )
                if resp.status_code != 200:
                    logger.warning(f"Feed {name} returned HTTP {resp.status_code}. Skipping.")
                    continue
                parsed = feedparser.parse(resp.content)
                entries = parsed.entries[:max_per_feed]
                logger.info(f"Retrieved {len(entries)} entries from {name}.")

                for entry in entries:
                    title = getattr(entry, "title", "").strip()
                    if not title:
                        continue

                    link = getattr(entry, "link", "")
                    summary = getattr(entry, "summary", "") or getattr(entry, "description", "")
                    # Clean html tags from summary if present
                    summary = pd.Series([summary]).str.replace(r"<[^<]+?>", "", regex=True).iloc[0].strip()

                    # Published timestamp
                    pub_ts = None
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        pub_ts = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                    elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                        pub_ts = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        pub_ts = now_utc

                    # Sentiment analysis of headline + summary
                    sentiment_text = f"{title}. {summary[:200]}"
                    score = self.analyzer.analyze(sentiment_text)["compound"]
                    article_id = self.generate_article_id(link, title)

                    articles.append({
                        "article_id": article_id,
                        "title": title,
                        "source": name,
                        "published_at": pub_ts,
                        "url": link,
                        "summary": summary[:500],
                        "sentiment_compound": score,
                        "collected_at": now_utc,
                    })
            except Exception as e:
                logger.warning(f"Failed to fetch RSS feed {name} ({url}): {e}")

        # Deduplicate by article_id
        unique_articles = {a["article_id"]: a for a in articles}.values()
        article_list = list(unique_articles)
        logger.info(f"Total unique RSS news articles processed: {len(article_list)}")

        if self.db and article_list:
            self.db.save_news_articles(article_list)

        return article_list

    def fetch_tavily(self, query: str = "Bitcoin BTC crypto price news", max_results: int = 10) -> List[Dict[str, Any]]:
        """Optional connector for Tavily Search API if credentials are provided."""
        if not self.tavily_api_key:
            logger.info("Tavily API key not supplied (TAVILY_API_KEY). Using free RSS feeds.")
            return []

        logger.info("Querying Tavily News API...")
        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.tavily_api_key,
                    "query": query,
                    "topic": "news",
                    "max_results": max_results,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
        except Exception as e:
            logger.warning(f"Tavily search request failed: {e}")
            return []

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        articles = []
        for r in results:
            title = r.get("title", "")
            url = r.get("url", "")
            content = r.get("content", "")
            pub = r.get("published_date") or now_utc
            score = self.analyzer.analyze(f"{title}. {content[:200]}")["compound"]
            articles.append({
                "article_id": self.generate_article_id(url, title),
                "title": title,
                "source": "Tavily",
                "published_at": pub,
                "url": url,
                "summary": content[:500],
                "sentiment_compound": score,
                "collected_at": now_utc,
            })

        if self.db and articles:
            self.db.save_news_articles(articles)
        return articles

    def fetch_exa(self, query: str = "Bitcoin cryptocurrency market outlook", num_results: int = 10) -> List[Dict[str, Any]]:
        """Optional connector for Exa Search API if credentials are provided."""
        if not self.exa_api_key:
            logger.info("Exa API key not supplied (EXA_API_KEY). Using free RSS feeds.")
            return []

        logger.info("Querying Exa Search API...")
        try:
            resp = requests.post(
                "https://api.exa.ai/search",
                headers={"x-api-key": self.exa_api_key, "Content-Type": "application/json"},
                json={"query": query, "type": "neural", "useAutoprompt": True, "numResults": num_results},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
        except Exception as e:
            logger.warning(f"Exa search request failed: {e}")
            return []

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        articles = []
        for r in results:
            title = r.get("title", "")
            url = r.get("url", "")
            pub = r.get("publishedDate") or now_utc
            score = self.analyzer.analyze(title)["compound"]
            articles.append({
                "article_id": self.generate_article_id(url, title),
                "title": title,
                "source": "Exa",
                "published_at": pub,
                "url": url,
                "summary": title,
                "sentiment_compound": score,
                "collected_at": now_utc,
            })

        if self.db and articles:
            self.db.save_news_articles(articles)
        return articles

    def import_from_json(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Import news articles from local JSON file."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"News JSON file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            items = json.load(f)

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        articles = []
        for it in items:
            title = str(it.get("title", "")).strip()
            if not title:
                continue
            url = str(it.get("url", ""))
            score = it.get("sentiment_compound")
            if score is None:
                score = self.analyzer.analyze(title)["compound"]

            articles.append({
                "article_id": str(it.get("article_id") or self.generate_article_id(url, title)),
                "title": title,
                "source": str(it.get("source", "LocalImport")),
                "published_at": it.get("published_at") or now_utc,
                "url": url,
                "summary": str(it.get("summary", "")),
                "sentiment_compound": float(score),
                "collected_at": now_utc,
            })

        if self.db and articles:
            self.db.save_news_articles(articles)
        return articles
