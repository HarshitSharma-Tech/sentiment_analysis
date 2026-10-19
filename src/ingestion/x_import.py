"""
Free Local X / Twitter Import Pipeline
Loads X/Twitter posts from CSV and JSON files without requiring API credentials.
Normalizes heterogeneous field names, computes VADER sentiment, deduplicates,
and persists records to the SQLite database.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from src.database import DatabaseManager
from src.sentiment import SentimentAnalyzer, get_analyzer

logger = logging.getLogger(__name__)


class XPostImporter:
    """Imports and standardizes X/Twitter posts from local files."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        analyzer: Optional[SentimentAnalyzer] = None,
    ):
        self.db = db_manager
        self.analyzer = analyzer or get_analyzer()

    def _normalize_post_dict(self, raw: Dict[str, Any], default_source: str) -> Dict[str, Any]:
        """Normalize varying column/field keys from different export formats."""
        # 1. Post ID
        post_id = str(raw.get("post_id") or raw.get("id") or raw.get("tweet_id") or "")
        if not post_id:
            # Fallback deterministic pseudo-id from timestamp and text snippet
            ts = raw.get("timestamp") or raw.get("created_at") or ""
            txt = raw.get("text") or raw.get("content") or ""
            post_id = f"gen_{hash(str(ts) + str(txt[:30]))}"

        # 2. Timestamp
        raw_ts = raw.get("timestamp") or raw.get("created_at") or raw.get("date") or datetime.now(timezone.utc).isoformat()
        try:
            ts_dt = pd.to_datetime(raw_ts, utc=True)
            ts_str = ts_dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        # 3. Text content
        text = str(raw.get("text") or raw.get("content") or raw.get("full_text") or "").strip()

        # 4. Engagement metrics
        likes = int(raw.get("likes") or raw.get("favorite_count") or raw.get("like_count") or 0)
        reposts = int(raw.get("reposts") or raw.get("retweet_count") or raw.get("retweets") or 0)
        views = int(raw.get("views") or raw.get("view_count") or raw.get("impressions") or 0)
        followers = int(raw.get("author_followers") or raw.get("followers_count") or raw.get("user_followers") or 0)
        query_ticker = str(raw.get("query_ticker") or raw.get("ticker") or "BTC").upper()
        source = str(raw.get("source") or default_source)

        # 5. Sentiment scores (compute if not present)
        compound = raw.get("sentiment_compound")
        pos = raw.get("sentiment_pos")
        neu = raw.get("sentiment_neu")
        neg = raw.get("sentiment_neg")

        if compound is None or pos is None:
            sentiment = self.analyzer.analyze(text)
            compound = sentiment["compound"]
            pos = sentiment["pos"]
            neu = sentiment["neu"]
            neg = sentiment["neg"]

        collected_at = raw.get("collected_at") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        return {
            "post_id": post_id,
            "timestamp": ts_str,
            "text": text,
            "likes": max(0, likes),
            "reposts": max(0, reposts),
            "views": max(0, views),
            "author_followers": max(0, followers),
            "query_ticker": query_ticker,
            "sentiment_compound": float(compound),
            "sentiment_pos": float(pos),
            "sentiment_neu": float(neu),
            "sentiment_neg": float(neg),
            "source": source,
            "collected_at": collected_at,
        }

    def import_from_json(self, file_path: Union[str, Path], source_label: str = "local_json") -> List[Dict[str, Any]]:
        """Import posts from JSON array or JSON Lines file."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"JSON file not found: {p}")

        posts: List[Dict[str, Any]] = []
        with open(p, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content.startswith("["):
                raw_data = json.loads(content)
            else:
                raw_data = [json.loads(line) for line in content.splitlines() if line.strip()]

        for item in raw_data:
            if isinstance(item, dict):
                norm = self._normalize_post_dict(item, default_source=source_label)
                if norm["text"]:
                    posts.append(norm)

        logger.info(f"Loaded {len(posts)} posts from JSON: {p.name}")
        if self.db and posts:
            self.db.save_x_posts(posts)
        return posts

    def import_from_csv(self, file_path: Union[str, Path], source_label: str = "local_csv") -> List[Dict[str, Any]]:
        """Import posts from CSV file."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"CSV file not found: {p}")

        df = pd.read_csv(p)
        posts: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            norm = self._normalize_post_dict(row.to_dict(), default_source=source_label)
            if norm["text"]:
                posts.append(norm)

        logger.info(f"Loaded {len(posts)} posts from CSV: {p.name}")
        if self.db and posts:
            self.db.save_x_posts(posts)
        return posts
