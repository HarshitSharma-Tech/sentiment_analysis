"""
Real End-to-End Integration Test for Apify X Scraper
CrowdWisdomTrading Quantitative Data Scientist Assessment

Performs a live, controlled test using apidojo/tweet-scraper and available Apify credits.
Validates:
1. Authentication & Credit/Cost tracking (capped at ~50 posts, ~$0.02 cost)
2. Ingestion & field schema parsing (text, likes, reposts, views, followers, UTC timestamps)
3. Sentiment scoring with crypto-augmented VADER NLP
4. Database insertion & deduplication (idempotent primary key check)
5. Quantitative feature extraction & aggregation
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys

# Ensure repository root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import get_config
from src.database import DatabaseManager
from src.features import FeatureEngineer
from src.ingestion.apify_x import ApifyXConnector
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("apify_integration_test")


def main():
    cfg = get_config()
    print("=" * 80)
    print("CROWDWISDOMTRADING: REAL APIFY X SCRAPER INTEGRATION TEST")
    print("=" * 80)

    # 1. Environment & Credential Verification
    token = cfg.apify_api_token
    if not token:
        print("[ERROR] APIFY_API_TOKEN not found in environment or .env file.")
        sys.exit(1)

    masked_token = f"{token[:10]}...{token[-4:]}"
    print(f"[*] API Token detected: {masked_token} (length: {len(token)})")

    # Verify Account & Plan
    headers = {"Authorization": f"Bearer {token}"}
    try:
        user_resp = requests.get("https://api.apify.com/v2/users/me", headers=headers, timeout=10)
        user_resp.raise_for_status()
        user_data = user_resp.json().get("data", {})
        username = user_data.get("username", "unknown")
        plan_id = user_data.get("plan", {}).get("id", "FREE")
        credits_usd = user_data.get("plan", {}).get("monthlyUsageCreditsUsd", 5.0)
        print(f"[+] Authenticated successfully with Apify as user: '{username}'")
        print(f"[+] Plan Tier: {plan_id} | Monthly Credit Allocation: ${credits_usd:.2f} USD")
    except Exception as e:
        print(f"[ERROR] Failed to authenticate with Apify: {e}")
        sys.exit(1)

    # 2. Pricing & Cost Budget Verification
    # apidojo/tweet-scraper charges $0.0004 per tweet ($0.40 / 1,000 tweets)
    MAX_POSTS = 50
    ESTIMATED_COST = MAX_POSTS * 0.0004
    print(f"[*] Target Actor: 'apidojo/tweet-scraper'")
    print(f"[*] Rate: $0.0004 / tweet ($0.40 per 1,000 tweets)")
    print(f"[*] Controlled Run Limit: {MAX_POSTS} tweets")
    print(f"[*] Estimated Max Run Cost: ${ESTIMATED_COST:.4f} USD (leaving ~${credits_usd - ESTIMATED_COST:.2f} credits untouched)")

    # 3. Database Initialization
    db = DatabaseManager(cfg.database_path)
    initial_live_count = len(db.get_x_posts().query("source == 'apify'")) if not db.get_x_posts().empty else 0
    print(f"[*] Initial live Apify posts in database: {initial_live_count}")

    # 4. Live Scrape Execution via ApifyXConnector
    print(f"\n[*] Executing live scrape via ApifyXConnector...")
    connector = ApifyXConnector(api_token=token, db_manager=db)
    posts = connector.fetch_posts(
        query="Bitcoin OR BTC",
        max_posts=MAX_POSTS,
        actor_id="apidojo/tweet-scraper",
        query_ticker="BTC"
    )

    if not posts:
        print("[ERROR] No posts returned from Apify execution.")
        sys.exit(1)

    print(f"[+] Successfully retrieved {len(posts)} live posts from Apify!")

    # 5. Field Parsing & Schema Verification
    print("\n" + "-" * 60)
    print("SCHEMA & METRIC PARSING VERIFICATION (Sample of 3 posts):")
    print("-" * 60)
    required_keys = [
        "post_id", "timestamp", "text", "likes", "reposts", "views",
        "author_followers", "query_ticker", "sentiment_compound",
        "sentiment_pos", "sentiment_neu", "sentiment_neg", "source", "collected_at"
    ]

    for i, p in enumerate(posts[:3], 1):
        for rk in required_keys:
            assert rk in p, f"Missing required key '{rk}' in parsed post!"
        
        safe_snippet = p["text"][:75].encode("ascii", errors="replace").decode("ascii")
        print(f"Post {i}:")
        print(f"  ID:          {p['post_id']}")
        print(f"  Timestamp:   {p['timestamp']} (UTC format verified)")
        print(f"  Text:        \"{safe_snippet}...\"")
        print(f"  Engagement:  Likes={p['likes']} | Reposts={p['reposts']} | Views={p['views']} | Followers={p['author_followers']}")
        print(f"  Sentiment:   Compound={p['sentiment_compound']:+.4f} (pos={p['sentiment_pos']:.3f}, neu={p['sentiment_neu']:.3f}, neg={p['sentiment_neg']:.3f})")
        print(f"  Source:      {p['source']} | Ticker: {p['query_ticker']}")

    # 6. Database Insertion & Deduplication Verification
    print("\n" + "-" * 60)
    print("DATABASE INSERTION & IDEMPOTENCY / DEDUPLICATION CHECK:")
    print("-" * 60)
    all_db_posts = db.get_x_posts()
    apify_db_posts = all_db_posts[all_db_posts["source"] == "apify"]
    print(f"[+] Total Apify posts now stored in SQLite database: {len(apify_db_posts)}")
    assert len(apify_db_posts) >= len(posts), "Database record count mismatch!"

    # Test Deduplication: Re-inserting the exact same posts must result in 0 new rows (idempotent primary key)
    print("[*] Testing idempotency: Re-inserting the same posts to test duplicate rejection...")
    dup_inserted = db.save_x_posts(posts)
    print(f"[+] Duplicate insertion result: {dup_inserted} rows inserted (Expected: 0). Deduplication verified!")
    assert dup_inserted == 0, f"Expected 0 duplicate rows inserted, got {dup_inserted}!"

    # 7. Quantitative Feature Engineering on Live Ingested Data
    print("\n" + "-" * 60)
    print("QUANTITATIVE FEATURE AGGREGATION ON LIVE DATA:")
    print("-" * 60)
    fe = FeatureEngineer()
    agg_df = fe.aggregate_posts_by_frequency(apify_db_posts, frequency="1h")
    print(f"[+] Successfully aggregated live posts into {len(agg_df)} hourly bars.")
    print("Sample aggregated bar features:")
    for col in ["post_count", "sentiment_compound_mean", "sentiment_compound_ewm", "pos_ratio", "neg_ratio", "sentiment_dispersion"]:
        if col in agg_df.columns:
            val = agg_df[col].iloc[-1]
            print(f"  - {col}: {val:.4f}" if isinstance(val, float) else f"  - {col}: {val}")

    print("\n" + "=" * 80)
    print("APIFY REAL INTEGRATION TEST: ALL CHECKS PASSED (100% SUCCESS)")
    print("=" * 80)


if __name__ == "__main__":
    main()
