"""
Global Configuration and Settings
CrowdWisdomTrading Quantitative Data Scientist Assessment
Free-first, fully reproducible architecture.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Base workspace directory
BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass
class Config:
    """Project-wide settings and parameters."""

    # Paths
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    sample_dir: Path = BASE_DIR / "data" / "sample"
    raw_dir: Path = BASE_DIR / "data" / "raw"
    processed_dir: Path = BASE_DIR / "data" / "processed"
    database_dir: Path = BASE_DIR / "database"
    database_path: Path = BASE_DIR / "database" / "crowdwisdom.db"
    models_dir: Path = BASE_DIR / "models"
    reports_dir: Path = BASE_DIR / "reports"
    figures_dir: Path = BASE_DIR / "reports" / "figures"

    # Asset & Market Data
    asset_ticker: str = os.getenv("ASSET_TICKER", "BTC-USD")
    data_frequency: str = os.getenv("DATA_FREQUENCY", "1h")  # '1h' or '1d'
    history_days: int = 180  # Target window: 180 days

    # Forecast Horizons (n+0 to n+5)
    horizons: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])

    # Sentiment Configuration (VADER standard thresholds)
    vader_pos_threshold: float = 0.05
    vader_neg_threshold: float = -0.05
    sentiment_rolling_windows: List[int] = field(default_factory=lambda: [3, 6, 12, 24])

    # Price Feature Configuration
    price_rolling_windows: List[int] = field(default_factory=lambda: [3, 6, 12, 24])
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Walk-Forward Cross Validation
    wf_min_train_size: int = 480  # 20 days of hourly bars minimum training history
    wf_step_size: int = 168       # Step by 1 week (168 hourly bars)
    wf_test_size: int = 168       # Predict 1 week out-of-sample per step

    # Backtesting & Quantitative Strategy
    initial_capital: float = 100_000.0
    transaction_cost_bps: float = 10.0  # 10 bps = 0.10% per transaction (slippage + taker fee)
    signal_threshold: float = 0.001     # 0.10% expected return minimum to trigger entry
    risk_free_rate: float = 0.04        # 4.0% annualized risk-free rate

    # News RSS Feeds (Free, no API keys needed)
    rss_feeds: List[dict] = field(default_factory=lambda: [
        {"name": "CoinDesk", "url": "https://www.coindesk.com/arc/outboundfeeds/rss/"},
        {"name": "Cointelegraph", "url": "https://cointelegraph.com/rss"},
        {"name": "Google News Crypto", "url": "https://news.google.com/rss/search?q=Bitcoin+crypto&hl=en-US&gl=US&ceid=US:en"},
    ])

    # Optional API Keys (defaults to None / empty)
    apify_api_token: str = os.getenv("APIFY_API_TOKEN", "")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    exa_api_key: str = os.getenv("EXA_API_KEY", "")

    def __post_init__(self):
        """Ensure all required runtime directories exist."""
        for path in [
            self.data_dir,
            self.sample_dir,
            self.raw_dir,
            self.processed_dir,
            self.database_dir,
            self.models_dir,
            self.reports_dir,
            self.figures_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


def get_config() -> Config:
    """Return a single initialized Config instance."""
    return Config()
