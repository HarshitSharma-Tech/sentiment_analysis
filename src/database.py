"""
SQLite Database Layer
Handles persistent local storage for X posts, market prices, news articles,
aggregated features, predictions, and evaluation metrics.
Thread-safe, parameterized queries with duplicate prevention.
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite operations for the CrowdWisdom forecasting pipeline."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Create and return a new SQLite connection with foreign keys enabled."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self) -> None:
        """Initialize database schema from database/schema.sql."""
        schema_path = Path(__file__).resolve().parent.parent / "database" / "schema.sql"
        if not schema_path.exists():
            raise FileNotFoundError(f"Database schema file not found at {schema_path}")

        with open(schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        with self.get_connection() as conn:
            conn.executescript(schema_sql)
            conn.commit()
        logger.info(f"Database initialized successfully at {self.db_path}")

    # =========================================================================
    # X / Twitter Posts
    # =========================================================================

    def save_x_posts(self, posts: List[Dict[str, Any]]) -> int:
        """
        Insert or ignore X posts into database.
        Returns the number of rows inserted.
        """
        if not posts:
            return 0

        query = """
        INSERT OR IGNORE INTO x_posts (
            post_id, timestamp, text, likes, reposts, views,
            author_followers, query_ticker, sentiment_compound,
            sentiment_pos, sentiment_neu, sentiment_neg, source, collected_at
        ) VALUES (
            :post_id, :timestamp, :text, :likes, :reposts, :views,
            :author_followers, :query_ticker, :sentiment_compound,
            :sentiment_pos, :sentiment_neu, :sentiment_neg, :source, :collected_at
        );
        """
        with self.get_connection() as conn:
            cursor = conn.executemany(query, posts)
            conn.commit()
            logger.info(f"Saved {cursor.rowcount} new X posts to database.")
            return cursor.rowcount

    def get_x_posts(
        self,
        ticker: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> pd.DataFrame:
        """Retrieve X posts as a Pandas DataFrame sorted chronologically."""
        query = "SELECT * FROM x_posts WHERE 1=1"
        params: List[Any] = []

        if ticker:
            query += " AND query_ticker = ?"
            params.append(ticker)
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp ASC;"

        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if not df.empty and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    # =========================================================================
    # Historical Market Prices
    # =========================================================================

    def save_prices(
        self,
        df: pd.DataFrame,
        ticker: str,
        frequency: str = "1h",
        source: str = "yfinance",
    ) -> int:
        """
        Insert or replace historical OHLCV prices into SQLite.
        Expects df with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        or DatetimeIndex named 'timestamp'.
        """
        if df.empty:
            return 0

        df_to_save = df.copy()
        if "timestamp" not in df_to_save.columns:
            if isinstance(df_to_save.index, pd.DatetimeIndex):
                df_to_save = df_to_save.reset_index().rename(columns={"index": "timestamp", "Date": "timestamp", "Datetime": "timestamp"})
            else:
                raise ValueError("DataFrame must contain 'timestamp' column or DatetimeIndex")

        now_utc = datetime.now(timezone.utc).isoformat()
        records = []
        for _, row in df_to_save.iterrows():
            ts_str = pd.to_datetime(row["timestamp"], utc=True).strftime("%Y-%m-%d %H:%M:%S")
            records.append({
                "ticker": ticker,
                "timestamp": ts_str,
                "frequency": frequency,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0.0)),
                "source": source,
                "collected_at": now_utc,
            })

        query = """
        INSERT OR REPLACE INTO historical_prices (
            ticker, timestamp, frequency, open, high, low, close, volume, source, collected_at
        ) VALUES (
            :ticker, :timestamp, :frequency, :open, :high, :low, :close, :volume, :source, :collected_at
        );
        """
        with self.get_connection() as conn:
            cursor = conn.executemany(query, records)
            conn.commit()
            logger.info(f"Saved {cursor.rowcount} price records for {ticker} ({frequency}).")
            return cursor.rowcount

    def get_prices(
        self,
        ticker: str,
        frequency: str = "1h",
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> pd.DataFrame:
        """Retrieve historical prices for given ticker and frequency as a sorted DataFrame."""
        query = "SELECT * FROM historical_prices WHERE ticker = ? AND frequency = ?"
        params: List[Any] = [ticker, frequency]

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp ASC;"

        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if not df.empty and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp").sort_index()
        return df

    # =========================================================================
    # News Articles
    # =========================================================================

    def save_news_articles(self, articles: List[Dict[str, Any]]) -> int:
        """Insert or ignore news articles."""
        if not articles:
            return 0

        query = """
        INSERT OR IGNORE INTO news_articles (
            article_id, title, source, published_at, url, summary,
            sentiment_compound, collected_at
        ) VALUES (
            :article_id, :title, :source, :published_at, :url, :summary,
            :sentiment_compound, :collected_at
        );
        """
        with self.get_connection() as conn:
            cursor = conn.executemany(query, articles)
            conn.commit()
            logger.info(f"Saved {cursor.rowcount} news articles.")
            return cursor.rowcount

    def get_news_articles(self, limit: int = 50) -> pd.DataFrame:
        """Retrieve recent news articles ordered by publication date descending."""
        query = "SELECT * FROM news_articles ORDER BY published_at DESC LIMIT ?;"
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=[limit])
        return df

    # =========================================================================
    # Aggregated Sentiment
    # =========================================================================

    def save_aggregated_sentiment(self, df: pd.DataFrame, frequency: str = "1h") -> int:
        """Store computed aggregated sentiment features."""
        if df.empty:
            return 0

        df_to_save = df.copy()
        if "timestamp" not in df_to_save.columns:
            df_to_save = df_to_save.reset_index().rename(columns={"index": "timestamp"})

        records = []
        for _, row in df_to_save.iterrows():
            ts_str = pd.to_datetime(row["timestamp"], utc=True).strftime("%Y-%m-%d %H:%M:%S")
            records.append({
                "timestamp": ts_str,
                "frequency": frequency,
                "post_count": int(row.get("post_count", 0)),
                "mean_compound": float(row.get("mean_compound", 0.0)),
                "pos_proportion": float(row.get("pos_proportion", 0.0)),
                "neu_proportion": float(row.get("neu_proportion", 0.0)),
                "neg_proportion": float(row.get("neg_proportion", 0.0)),
                "sentiment_dispersion": float(row.get("sentiment_dispersion", 0.0)),
                "total_likes": int(row.get("total_likes", 0)),
                "total_reposts": int(row.get("total_reposts", 0)),
                "total_views": int(row.get("total_views", 0)),
                "engagement_weighted_sentiment": float(row.get("engagement_weighted_sentiment", 0.0)),
            })

        query = """
        INSERT OR REPLACE INTO aggregated_sentiment (
            timestamp, frequency, post_count, mean_compound, pos_proportion,
            neu_proportion, neg_proportion, sentiment_dispersion, total_likes,
            total_reposts, total_views, engagement_weighted_sentiment
        ) VALUES (
            :timestamp, :frequency, :post_count, :mean_compound, :pos_proportion,
            :neu_proportion, :neg_proportion, :sentiment_dispersion, :total_likes,
            :total_reposts, :total_views, :engagement_weighted_sentiment
        );
        """
        with self.get_connection() as conn:
            cursor = conn.executemany(query, records)
            conn.commit()
            return cursor.rowcount

    def get_aggregated_sentiment(self, frequency: str = "1h") -> pd.DataFrame:
        """Fetch aggregated sentiment by frequency as a DatetimeIndexed DataFrame."""
        query = "SELECT * FROM aggregated_sentiment WHERE frequency = ? ORDER BY timestamp ASC;"
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=[frequency])
        if not df.empty and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp").sort_index()
        return df

    # =========================================================================
    # Model Predictions & Evaluation Results
    # =========================================================================

    def save_predictions(self, predictions: List[Dict[str, Any]]) -> int:
        """Save out-of-sample prediction records."""
        if not predictions:
            return 0

        query = """
        INSERT OR REPLACE INTO model_predictions (
            model_name, horizon, forecast_timestamp, target_timestamp,
            actual_price, predicted_price, actual_return, predicted_return,
            run_id, created_at
        ) VALUES (
            :model_name, :horizon, :forecast_timestamp, :target_timestamp,
            :actual_price, :predicted_price, :actual_return, :predicted_return,
            :run_id, :created_at
        );
        """
        with self.get_connection() as conn:
            cursor = conn.executemany(query, predictions)
            conn.commit()
            return cursor.rowcount

    def get_predictions(
        self,
        run_id: Optional[str] = None,
        model_name: Optional[str] = None,
        horizon: Optional[int] = None,
    ) -> pd.DataFrame:
        """Retrieve model predictions DataFrame."""
        query = "SELECT * FROM model_predictions WHERE 1=1"
        params: List[Any] = []
        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
        if horizon is not None:
            query += " AND horizon = ?"
            params.append(horizon)

        query += " ORDER BY forecast_timestamp ASC, horizon ASC;"
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def save_evaluation_results(self, results: List[Dict[str, Any]]) -> int:
        """Save walk-forward cross-validation evaluation results."""
        if not results:
            return 0

        query = """
        INSERT INTO evaluation_results (
            run_id, model_name, horizon, frequency, train_bars, test_bars,
            mae, rmse, mape, directional_accuracy, r2_score,
            baseline_mae, baseline_rmse, sharpe_ratio, sortino_ratio,
            max_drawdown, cumulative_return, win_rate, total_trades, created_at
        ) VALUES (
            :run_id, :model_name, :horizon, :frequency, :train_bars, :test_bars,
            :mae, :rmse, :mape, :directional_accuracy, :r2_score,
            :baseline_mae, :baseline_rmse, :sharpe_ratio, :sortino_ratio,
            :max_drawdown, :cumulative_return, :win_rate, :total_trades, :created_at
        );
        """
        with self.get_connection() as conn:
            cursor = conn.executemany(query, results)
            conn.commit()
            return cursor.rowcount

    def get_evaluation_results(self, run_id: Optional[str] = None) -> pd.DataFrame:
        """Fetch evaluation results table."""
        query = "SELECT * FROM evaluation_results WHERE 1=1"
        params: List[Any] = []
        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
        query += " ORDER BY id DESC;"
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)
