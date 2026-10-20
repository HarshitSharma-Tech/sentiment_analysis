"""
Unit tests for ApifyXConnector in src/ingestion/apify_x.py
Verifies configuration detection, graceful free fallback, response parsing, and error handling.
"""

from unittest.mock import MagicMock, patch
import pytest
from src.ingestion.apify_x import ApifyXConnector
from src.database import DatabaseManager


def test_apify_unconfigured_fallback():
    """Verify that unconfigured Apify connector returns empty list without error."""
    connector = ApifyXConnector(api_token="")
    assert not connector.is_configured()
    posts = connector.fetch_posts(query="Bitcoin", max_posts=10)
    assert posts == []


def test_apify_configured_detection():
    """Verify that valid token is recognized."""
    connector = ApifyXConnector(api_token="apify_api_mock_valid_token_12345")
    assert connector.is_configured()


@patch("src.ingestion.apify_x.requests.post")
def test_apify_successful_fetch(mock_post, tmp_path):
    """Verify parsing of valid Apify dataset items into structured posts."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {
            "id_str": "tweet_1001",
            "full_text": "Bitcoin is super bullish today! HODL to the moon!",
            "created_at": "2024-10-18T14:30:00Z",
            "favorite_count": 120,
            "retweet_count": 45,
            "view_count": 1500,
            "user": {"followers_count": 5000},
        },
        {
            "id_str": "tweet_1002",
            "full_text": "Major dump incoming on crypto markets, be careful.",
            "created_at": "2024-10-18T14:35:00Z",
            "favorite_count": 25,
            "retweet_count": 5,
            "view_count": 400,
            "user": {"followers_count": 1200},
        },
    ]
    mock_post.return_value = mock_response

    db_path = tmp_path / "test_apify.db"
    db = DatabaseManager(str(db_path))

    connector = ApifyXConnector(api_token="apify_api_mock_token_123", db_manager=db)
    posts = connector.fetch_posts(query="Bitcoin", max_posts=2)

    assert len(posts) == 2
    assert posts[0]["post_id"] == "tweet_1001"
    assert posts[0]["likes"] == 120
    assert posts[0]["reposts"] == 45
    assert posts[0]["views"] == 1500
    assert posts[0]["author_followers"] == 5000
    assert posts[0]["source"] == "apify"
    assert "sentiment_compound" in posts[0]
    assert posts[0]["sentiment_compound"] > 0  # Bullish sentiment

    assert posts[1]["post_id"] == "tweet_1002"
    assert posts[1]["sentiment_compound"] < 0  # Bearish sentiment

    # Verify saved into SQLite database
    saved_posts = db.get_x_posts()
    assert len(saved_posts) == 2


@patch("src.ingestion.apify_x.requests.post")
def test_apify_api_error_handling(mock_post):
    """Verify graceful failure when Apify API returns 401 Unauthorized or throws exception."""
    # Test 401
    mock_resp_401 = MagicMock()
    mock_resp_401.status_code = 401
    mock_post.return_value = mock_resp_401

    connector = ApifyXConnector(api_token="invalid_token_12345")
    posts = connector.fetch_posts(query="Bitcoin", max_posts=10)
    assert posts == []

    # Test Exception (e.g. connection timeout)
    mock_post.side_effect = Exception("Connection timeout to Apify")
    posts_exc = connector.fetch_posts(query="Bitcoin", max_posts=10)
    assert posts_exc == []
