"""
Modular Apify X / Twitter Connector
Integrates with Apify Actor API when legitimate credentials are provided.
If credentials are absent, fails gracefully with informative logging and
delegates to free local imports or sample data.
"""

from datetime import datetime, timezone
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.database import DatabaseManager
from src.sentiment import SentimentAnalyzer, get_analyzer

logger = logging.getLogger(__name__)


class ApifyXConnector:
    """Optional Apify connector for collecting recent/historical X posts."""

    def __init__(
        self,
        api_token: Optional[str] = None,
        db_manager: Optional[DatabaseManager] = None,
        analyzer: Optional[SentimentAnalyzer] = None,
    ):
        self.api_token = os.getenv("APIFY_API_TOKEN", "") if api_token is None else api_token
        self.db = db_manager
        self.analyzer = analyzer or get_analyzer()

    def is_configured(self) -> bool:
        """Check whether an Apify API token has been supplied."""
        return bool(self.api_token and len(self.api_token.strip()) > 5)

    def fetch_posts(
        self,
        query: str = "Bitcoin OR BTC",
        max_posts: int = 200,
        actor_id: str = "apidojo/tweet-scraper",
        query_ticker: str = "BTC",
    ) -> List[Dict[str, Any]]:
        """
        Fetch posts from Apify REST API if credentials are provided.
        Accepts actor_id in both 'owner/actor' and 'owner~actor' formats.
        Returns empty list with explanation if unconfigured.
        """
        if not self.is_configured():
            logger.info(
                "Apify token not detected in environment (APIFY_API_TOKEN). "
                "Skipping live Apify scraping and using free local import pipeline."
            )
            return []

        # Apify REST API requires owner~actor format in URL paths
        normalized_actor = actor_id.replace("/", "~")
        logger.info(f"Triggering Apify Actor '{normalized_actor}' for query '{query}' (limit: {max_posts})...")
        endpoint = f"https://api.apify.com/v2/acts/{normalized_actor}/run-sync-get-dataset-items"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "searchTerms": [query],
            "maxItems": max_posts,
            "sort": "Latest",
        }

        try:
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=120)
            if resp.status_code == 401:
                logger.error("Apify authentication failed: Invalid or expired API token.")
                return []
            resp.raise_for_status()
            items = resp.json()
        except Exception as e:
            logger.error(f"Apify request failed: {e}. Check network connection or actor limits.")
            return []

        if not isinstance(items, list):
            logger.warning(f"Unexpected Apify response structure: {type(items)}")
            return []

        posts: List[Dict[str, Any]] = []
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        for item in items:
            # Handles both standard text and modern Apify fullText
            text = str(
                item.get("fullText")
                or item.get("full_text")
                or item.get("text")
                or ""
            ).strip()
            if not text:
                continue

            post_id = str(item.get("id_str") or item.get("id") or f"apify_{hash(text[:40])}")
            # Handles createdAt (ISO or UTC string) and created_at
            created_at = item.get("createdAt") or item.get("created_at") or now_utc

            # Parse timestamp to UTC standard string
            try:
                import pandas as pd
                ts_str = pd.to_datetime(created_at, utc=True).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                ts_str = now_utc

            # Engagement metrics: supports camelCase (apidojo) and snake_case
            likes = int(item.get("likeCount") or item.get("favorite_count") or item.get("likes") or 0)
            reposts = int(item.get("retweetCount") or item.get("retweet_count") or item.get("reposts") or 0)
            views = int(item.get("viewCount") or item.get("view_count") or item.get("views") or 0)

            # Author follower count: supports nested author dict or user dict
            author_obj = item.get("author") if isinstance(item.get("author"), dict) else (
                item.get("user") if isinstance(item.get("user"), dict) else {}
            )
            followers = int(
                author_obj.get("followers")
                or author_obj.get("followersCount")
                or author_obj.get("followers_count")
                or item.get("author_followers")
                or 0
            )

            # Sentiment scoring via crypto-augmented VADER NLP
            scores = self.analyzer.analyze(text)

            post_dict = {
                "post_id": post_id,
                "timestamp": ts_str,
                "text": text,
                "likes": likes,
                "reposts": reposts,
                "views": views,
                "author_followers": followers,
                "query_ticker": query_ticker,
                "sentiment_compound": scores["compound"],
                "sentiment_pos": scores["pos"],
                "sentiment_neu": scores["neu"],
                "sentiment_neg": scores["neg"],
                "source": "apify",
                "collected_at": now_utc,
            }
            posts.append(post_dict)

        logger.info(f"Successfully processed {len(posts)} posts from Apify.")
        if self.db and posts:
            self.db.save_x_posts(posts)

        return posts
