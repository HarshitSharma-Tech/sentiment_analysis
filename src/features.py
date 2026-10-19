"""
Quantitative Feature Engineering Module
Computes strictly point-in-time, leakage-free features from:
1. Hourly/Daily aggregated X/Twitter sentiment metrics
2. Historical market prices and technical momentum/volatility indicators

Strict Guarantees:
- Zero forward-looking lookahead bias
- Past-only rolling statistics
- Engagement weighting with logarithmic clipping
- Curated high-signal feature set (avoids feature proliferation & noise overfitting)
- Supports explicit ablation: Price-Only vs. Price+Sentiment
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Feature column registries for ablation comparisons
PRICE_FEATURE_COLS = [
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "return_24",
    "price_volatility_6",
    "price_volatility_24",
    "sma_ratio_6",
    "sma_ratio_24",
    "macd_diff",
    "rsi_14",
    "volume_change_1",
    "volume_ratio_24",
]

SENTIMENT_FEATURE_COLS = [
    "mean_compound",
    "sentiment_dispersion",
    "pos_proportion",
    "neg_proportion",
    "post_count_log",
    "post_volume_change_1",
    "engagement_weighted_sentiment",
    "sentiment_rolling_mean_6",
    "sentiment_rolling_mean_24",
    "sentiment_momentum",
    "sentiment_price_interaction",
]

ALL_FEATURE_COLS = PRICE_FEATURE_COLS + SENTIMENT_FEATURE_COLS


class FeatureEngineer:
    """Computes curated price-derived and sentiment feature matrices."""

    def __init__(
        self,
        sentiment_windows: Optional[List[int]] = None,
        price_windows: Optional[List[int]] = None,
        rsi_period: int = 14,
    ):
        self.sentiment_windows = sentiment_windows or [6, 24]
        self.price_windows = price_windows or [6, 24]
        self.rsi_period = rsi_period

    # =========================================================================
    # 1. Sentiment Aggregation & Feature Extraction
    # =========================================================================

    @staticmethod
    def aggregate_posts_by_frequency(
        posts_df: pd.DataFrame,
        frequency: str = "1h",
    ) -> pd.DataFrame:
        """
        Aggregate individual X posts into regular chronological time buckets (e.g. 1h or 1d).
        Computes mean sentiment, pos/neu/neg proportions, dispersion, engagement metrics,
        and engagement-weighted sentiment with logarithmic clipping.
        """
        if posts_df.empty:
            return pd.DataFrame()

        df = posts_df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp")
            else:
                raise ValueError("posts_df must contain timestamp column or index.")

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df = df.sort_index()

        # Engagement weight calculation with safe upper bound
        # weight = log1p(likes + 2*reposts + 0.05*views + 0.01*followers)
        # Cap weight at 20.0 to prevent an extreme outlier post from dominating
        likes = df["likes"].clip(lower=0)
        reposts = df["reposts"].clip(lower=0)
        views = df["views"].clip(lower=0)
        followers = df["author_followers"].clip(lower=0)

        raw_eng = likes + (2.0 * reposts) + (0.05 * views) + (0.01 * followers)
        df["eng_weight"] = np.clip(np.log1p(raw_eng), a_min=0.1, a_max=20.0)

        compound = df["sentiment_compound"]
        df["weighted_compound"] = compound * df["eng_weight"]
        df["is_pos"] = (compound >= 0.05).astype(float)
        df["is_neu"] = ((compound > -0.05) & (compound < 0.05)).astype(float)
        df["is_neg"] = (compound <= -0.05).astype(float)

        # Resample into regular frequency buckets
        grouped = df.resample(frequency)
        counts = grouped["sentiment_compound"].count()

        agg = pd.DataFrame(index=counts.index)
        agg["post_count"] = counts
        agg["mean_compound"] = grouped["sentiment_compound"].mean().fillna(0.0)
        agg["sentiment_dispersion"] = grouped["sentiment_compound"].std().fillna(0.0)

        total_posts = agg["post_count"].replace(0, np.nan)
        agg["pos_proportion"] = (grouped["is_pos"].sum() / total_posts).fillna(0.0)
        agg["neu_proportion"] = (grouped["is_neu"].sum() / total_posts).fillna(1.0)
        agg["neg_proportion"] = (grouped["is_neg"].sum() / total_posts).fillna(0.0)

        agg["total_likes"] = grouped["likes"].sum().fillna(0)
        agg["total_reposts"] = grouped["reposts"].sum().fillna(0)
        agg["total_views"] = grouped["views"].sum().fillna(0)

        sum_weight = grouped["eng_weight"].sum()
        sum_weighted_score = grouped["weighted_compound"].sum()
        agg["engagement_weighted_sentiment"] = np.where(
            sum_weight > 0, sum_weighted_score / np.maximum(sum_weight, 1e-12), agg["mean_compound"]
        )

        agg.index = pd.to_datetime(agg.index, utc=True)
        agg.index.name = "timestamp"
        return agg.sort_index()

    def build_sentiment_features(self, agg_sentiment: pd.DataFrame) -> pd.DataFrame:
        """
        Derive dynamic temporal sentiment signals (past-only rolling averages, volume deltas).
        """
        if agg_sentiment.empty:
            return pd.DataFrame()

        feats = pd.DataFrame(index=agg_sentiment.index)

        # Core point-in-time sentiment signals
        feats["mean_compound"] = agg_sentiment["mean_compound"].fillna(0.0)
        feats["sentiment_dispersion"] = agg_sentiment["sentiment_dispersion"].fillna(0.0)
        feats["pos_proportion"] = agg_sentiment["pos_proportion"].fillna(0.0)
        feats["neg_proportion"] = agg_sentiment["neg_proportion"].fillna(0.0)
        feats["post_count_log"] = np.log1p(agg_sentiment["post_count"].fillna(0.0))
        feats["post_volume_change_1"] = agg_sentiment["post_count"].diff().fillna(0.0)
        feats["engagement_weighted_sentiment"] = agg_sentiment["engagement_weighted_sentiment"].fillna(0.0)

        # Rolling sentiment trends
        feats["sentiment_rolling_mean_6"] = (
            agg_sentiment["mean_compound"].rolling(window=6, min_periods=1).mean().fillna(0.0)
        )
        feats["sentiment_rolling_mean_24"] = (
            agg_sentiment["mean_compound"].rolling(window=24, min_periods=1).mean().fillna(0.0)
        )
        feats["sentiment_momentum"] = feats["sentiment_rolling_mean_6"] - feats["sentiment_rolling_mean_24"]

        return feats

    # =========================================================================
    # 2. Price-Derived Features
    # =========================================================================

    def build_price_features(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive technical indicators, returns, rolling volatility, and momentum.
        Assumes prices_df has columns: ['open', 'high', 'low', 'close', 'volume'].
        All features are strictly point-in-time up to the bar timestamp.
        """
        if prices_df.empty:
            return pd.DataFrame()

        p = prices_df.copy().sort_index()
        close = p["close"]
        volume = p["volume"]

        feats = pd.DataFrame(index=p.index)

        # 1. Multi-horizon lagged returns
        feats["return_1"] = (close / close.shift(1) - 1.0).fillna(0.0)
        feats["return_3"] = (close / close.shift(3) - 1.0).fillna(0.0)
        feats["return_6"] = (close / close.shift(6) - 1.0).fillna(0.0)
        feats["return_12"] = (close / close.shift(12) - 1.0).fillna(0.0)
        feats["return_24"] = (close / close.shift(24) - 1.0).fillna(0.0)

        # 2. Rolling price return volatility
        feats["price_volatility_6"] = feats["return_1"].rolling(window=6, min_periods=2).std().fillna(0.0)
        feats["price_volatility_24"] = feats["return_1"].rolling(window=24, min_periods=2).std().fillna(0.0)

        # 3. Moving Average Ratios (Close / SMA - 1)
        sma_6 = close.rolling(window=6, min_periods=1).mean()
        sma_24 = close.rolling(window=24, min_periods=1).mean()
        feats["sma_ratio_6"] = (close / sma_6 - 1.0).fillna(0.0)
        feats["sma_ratio_24"] = (close / sma_24 - 1.0).fillna(0.0)

        # 4. MACD Signal Difference
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        feats["macd_diff"] = ((macd - macd_signal) / close).fillna(0.0)

        # 5. Relative Strength Index (RSI 14)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=self.rsi_period, min_periods=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period, min_periods=self.rsi_period).mean()

        rs = np.where(avg_loss > 1e-12, avg_gain / avg_loss, np.where(avg_gain > 1e-12, 1000.0, 1.0))
        rsi = 100.0 - (100.0 / (1.0 + rs))
        feats["rsi_14"] = (pd.Series(rsi, index=p.index).fillna(50.0)) / 100.0  # Normalized to [0, 1]

        # 6. Volume Dynamics
        feats["volume_change_1"] = volume.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        vol_sma_24 = volume.rolling(window=24, min_periods=1).mean().replace(0, np.nan)
        feats["volume_ratio_24"] = (volume / vol_sma_24).fillna(1.0)

        return feats

    # =========================================================================
    # 3. Synchronized Feature Matrix Construction
    # =========================================================================

    def merge_features(
        self,
        prices_df: pd.DataFrame,
        posts_df: Optional[pd.DataFrame] = None,
        agg_sentiment: Optional[pd.DataFrame] = None,
        frequency: str = "1h",
    ) -> pd.DataFrame:
        """
        Merge price features and sentiment features aligned by timestamp index.
        Creates sentiment-price interaction feature.
        """
        if prices_df.empty:
            raise ValueError("prices_df cannot be empty")

        price_feats = self.build_price_features(prices_df)

        if agg_sentiment is None or agg_sentiment.empty:
            if posts_df is not None and not posts_df.empty:
                agg_sentiment = self.aggregate_posts_by_frequency(posts_df, frequency=frequency)
            else:
                agg_sentiment = pd.DataFrame(index=price_feats.index)

        sent_feats = self.build_sentiment_features(agg_sentiment)

        # Left-join onto price index so every price observation has a corresponding feature row
        merged = price_feats.join(sent_feats, how="left")

        # Fill sentiment gaps with neutral values for periods where no tweets occurred
        for col in SENTIMENT_FEATURE_COLS:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0.0)
            elif col != "sentiment_price_interaction":
                merged[col] = 0.0

        # Interaction feature: Sentiment confirming or diverging from recent return
        merged["sentiment_price_interaction"] = (merged["mean_compound"] * merged["return_1"]).fillna(0.0)

        # Reference price columns for target construction
        merged["close"] = prices_df["close"]
        merged["open"] = prices_df["open"]
        merged["high"] = prices_df["high"]
        merged["low"] = prices_df["low"]
        merged["volume"] = prices_df["volume"]

        return merged.sort_index()
