"""
Reproducible Sample Data Generator
Generates realistic crypto microblog posts and financial news across a 180-day window.
Calibrated to Bitcoin market regimes (rallies, selloffs, consolidations).
Explicitly labeled as reproducible test data to support 100% free offline execution.
"""

from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import random
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

POST_TEMPLATES = [
    # Bullish
    ("Bitcoin looking extremely strong here! Breaking past resistance with heavy spot volume. $BTC to the moon! 🚀🔥", 1.0),
    ("Huge whale accumulation detected on-chain. $BTC supply on exchanges dropping to multi-year lows. Bullish!", 0.9),
    ("BTC ETF inflows just hit a new daily record. Institutional demand is relentless. Accumulating more here. 📈", 0.85),
    ("Clear bullish divergence on the 4H chart for Bitcoin. Target is the previous ATH. LFG! $BTC", 0.8),
    ("Buying the dip here. BTC fundamentals have never looked better. Hodl strong! 💎🙌", 0.75),
    ("Mining hash rate at all-time highs. Network security is impenetrable. $BTC is digital gold.", 0.7),
    ("Macro setup turning ultra bullish. Rate cuts incoming and Bitcoin halving supply squeeze kicking in. 🚀", 0.85),
    ("Just stacked another 0.1 $BTC. DCA every single week regardless of price. Wealth preservation.", 0.6),

    # Neutral
    ("Bitcoin currently consolidating around the 20-day moving average. Awaiting breakout direction. $BTC", 0.0),
    ("BTC funding rates remain neutral across major derivatives exchanges. Market in equilibrium.", 0.0),
    ("Upcoming CPI inflation print this Wednesday. Traders expecting volatility in crypto and equities. $BTC", 0.05),
    ("Bitcoin market dominance at 54.2%. Altcoins tracking sideways. Watching the weekly close closely.", 0.0),
    ("Miner revenue distribution analysis for Q3: how transaction fees impact network security post-halving.", 0.0),
    ("Order book liquidity depth on BTC-USD pairs showing clustered bids at support and asks at resistance.", 0.0),
    ("Daily Bitcoin technical outlook: Bollinger bands tightening, indicating a potential volatility expansion.", 0.02),

    # Bearish
    ("Heavy spot selling pressure hitting Bitcoin. Broken key support line. Bearish continuation likely. 📉", -0.85),
    ("Over $250M in crypto long positions liquidated in the last hour! Brutal flush. Stay safe out there. ⚠️", -0.9),
    ("Regulatory FUD spreading again. SEC investigating major crypto custodians. Expecting lower prices for $BTC.", -0.8),
    ("Macro headwinds returning: dollar index surging and yields climbing. Risky assets like Bitcoin dumping.", -0.75),
    ("BTC rejection at the key resistance level. Double top pattern confirmed on daily timeframe. Rekt.", -0.85),
    ("Whale wallet moved 10,000 BTC to exchange. Dump incoming, protect your capital.", -0.8),
    ("Bear market blues... Bitcoin volume is drying up completely. Dead cat bounce over?", -0.7),
]

SAMPLE_NEWS = [
    {"title": "Bitcoin Surges Toward Key Resistance as Spot ETF Inflows Accelerate", "source": "CoinDesk", "sentiment": 0.65},
    {"title": "Federal Reserve Signals Cautious Rate Path; Crypto Markets Hold Support", "source": "Yahoo Finance", "sentiment": 0.15},
    {"title": "Over $300M in Crypto Derivatives Liquidated Following Sharp Flash Crash", "source": "Cointelegraph", "sentiment": -0.75},
    {"title": "Institutional Crypto Custody Adoption Reaches Historic Milestone", "source": "CoinDesk", "sentiment": 0.55},
    {"title": "Global Regulators Debate Unified Framework for Digital Asset Custodians", "source": "Yahoo Finance", "sentiment": 0.0},
    {"title": "Bitcoin Mining Efficiency Hits Record High Despite Energy Market Volatility", "source": "Cointelegraph", "sentiment": 0.40},
    {"title": "Market Watch: Bitcoin Consolidates Near Recent Highs as Traders Eye Macro Data", "source": "CoinDesk", "sentiment": 0.10},
    {"title": "Macroeconomic Uncertainty Prompts Short-Term De-risking in Crypto Assets", "source": "Yahoo Finance", "sentiment": -0.45},
]


def generate_sample_data(days: int = 180, posts_per_day: int = 12) -> None:
    """Generate realistic sample X posts and news articles."""
    random.seed(42)
    np.random.seed(42)

    sample_dir = Path(__file__).resolve().parent.parent / "data" / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    end_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_time = end_time - timedelta(days=days)

    posts = []
    current_time = start_time

    post_id_counter = 1000000000

    # Simulate regime phases (bull, consolidation, bear)
    regime = 0.2  # baseline positive drift

    while current_time < end_time:
        # Occasionally shift regime
        if random.random() < 0.05:
            regime = random.choice([0.6, -0.4, 0.1, -0.1])

        # Hourly posts (poisson distributed)
        hour_posts = max(0, int(np.random.poisson(posts_per_day / 24.0)))

        for _ in range(hour_posts):
            post_id_counter += 1
            # Sample template based on regime
            if regime > 0.3:
                weights = [0.65, 0.20, 0.15]
            elif regime < -0.2:
                weights = [0.15, 0.25, 0.60]
            else:
                weights = [0.35, 0.40, 0.25]

            cat_choices = [
                [t for t in POST_TEMPLATES if t[1] > 0.3],
                [t for t in POST_TEMPLATES if -0.3 <= t[1] <= 0.3],
                [t for t in POST_TEMPLATES if t[1] < -0.3],
            ]
            chosen_cat = random.choices(cat_choices, weights=weights)[0]
            text, _ = random.choice(chosen_cat)

            # Power-law engagement
            likes = int(np.random.pareto(1.8) * 15)
            reposts = int(likes * random.uniform(0.1, 0.4))
            views = int(likes * random.uniform(20, 150)) + random.randint(100, 500)
            followers = int(np.random.pareto(1.5) * 500) + random.randint(50, 2000)

            # Add random minutes within the hour
            post_ts = current_time + timedelta(minutes=random.randint(0, 59), seconds=random.randint(0, 59))

            posts.append({
                "post_id": str(post_id_counter),
                "timestamp": post_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "text": text,
                "likes": likes,
                "reposts": reposts,
                "views": views,
                "author_followers": followers,
                "query_ticker": "BTC",
                "source": "sample_reproducible",
                "collected_at": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            })

        current_time += timedelta(hours=1)

    logger.info(f"Generated {len(posts)} reproducible sample posts across {days} days.")

    # Save JSON and CSV
    json_path = sample_dir / "btc_sample_x_posts.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(posts, f, indent=2)

    df_posts = pd.DataFrame(posts)
    csv_path = sample_dir / "btc_sample_x_posts.csv"
    df_posts.to_csv(csv_path, index=False)

    # Save News
    news_items = []
    for i, n in enumerate(SAMPLE_NEWS):
        news_ts = end_time - timedelta(days=random.randint(0, 14), hours=random.randint(1, 23))
        news_items.append({
            "article_id": f"sample_news_{i+1:03d}",
            "title": n["title"],
            "source": n["source"],
            "published_at": news_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "url": f"https://example.com/crypto-news/{i+1}",
            "summary": f"{n['title']}. Market participants are actively tracking liquidity conditions and institutional sentiment.",
            "sentiment_compound": n["sentiment"],
            "collected_at": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    news_json_path = sample_dir / "btc_sample_news.json"
    with open(news_json_path, "w", encoding="utf-8") as f:
        json.dump(news_items, f, indent=2)

    logger.info(f"Sample datasets saved to {sample_dir}")


if __name__ == "__main__":
    generate_sample_data(days=180, posts_per_day=16)
