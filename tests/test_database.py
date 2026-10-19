"""Unit tests for SQLite database manager."""
import tempfile
from pathlib import Path
import pandas as pd
import pytest
from src.database import DatabaseManager


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        manager = DatabaseManager(db_path)
        yield manager


def test_init_db(temp_db):
    with temp_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        assert "x_posts" in tables
        assert "historical_prices" in tables
        assert "news_articles" in tables
        assert "aggregated_sentiment" in tables
        assert "model_predictions" in tables
        assert "evaluation_results" in tables


def test_save_and_get_x_posts(temp_db):
    sample_posts = [
        {
            "post_id": "1001",
            "timestamp": "2024-01-01 10:00:00",
            "text": "Bitcoin to the moon! $BTC",
            "likes": 50,
            "reposts": 10,
            "views": 1000,
            "author_followers": 500,
            "query_ticker": "BTC",
            "sentiment_compound": 0.8,
            "sentiment_pos": 0.6,
            "sentiment_neu": 0.4,
            "sentiment_neg": 0.0,
            "source": "test",
            "collected_at": "2024-01-01 10:05:00",
        }
    ]
    inserted = temp_db.save_x_posts(sample_posts)
    assert inserted == 1

    # Duplicate insertion should be ignored
    inserted_dup = temp_db.save_x_posts(sample_posts)
    assert inserted_dup == 0

    df = temp_db.get_x_posts(ticker="BTC")
    assert len(df) == 1
    assert df.iloc[0]["post_id"] == "1001"


def test_save_and_get_prices(temp_db):
    dates = pd.date_range("2024-01-01 00:00:00", periods=5, freq="1h", tz="UTC")
    df_prices = pd.DataFrame({
        "open": [40000.0, 40100.0, 40200.0, 40150.0, 40300.0],
        "high": [40200.0, 40300.0, 40350.0, 40250.0, 40400.0],
        "low": [39900.0, 40050.0, 40100.0, 40100.0, 40200.0],
        "close": [40100.0, 40200.0, 40150.0, 40300.0, 40350.0],
        "volume": [100.0, 120.0, 80.0, 95.0, 110.0],
    }, index=dates)

    saved = temp_db.save_prices(df_prices, ticker="BTC-USD", frequency="1h", source="test")
    assert saved == 5

    retrieved = temp_db.get_prices(ticker="BTC-USD", frequency="1h")
    assert len(retrieved) == 5
    assert retrieved["close"].iloc[0] == 40100.0
