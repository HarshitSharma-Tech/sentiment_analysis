"""
Quantitative Feature Engineering Module
Computes strictly point-in-time, leakage-free features from:
1. Hourly/Daily aggregated X/Twitter sentiment metrics
2. Historical market prices and technical momentum/volatility indicators

Strict Guarantees:
- Zero forward-looking lookahead bias
- Consistent UTC chronological timestamps
- Engagement weighting with robust clipping against viral post anomalies
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Computes sentiment and price-derived feature matrices."""

    def __init__(
        self,
        sentiment_windows: Optional[List[int]] = None,
        price_windows: Optional[List[int]] = None,
        rsi_period: int = 14,
    ):
        self.sentiment_windows = sentiment_windows or [3, 6, 12, 24]
        self.price_windows = price_windows or [3, 6, 12, 24]
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
        Derive dynamic temporal sentiment signals (lags, rolling averages, volume deltas).
        Zero lookahead: all rolling statistics use only past observations.
        """
        if agg_sentiment.empty:
            return pd.DataFrame()

        feats = agg_sentiment.copy()

        # 1. Post Volume Dynamics
        feats["post_volume_change_1"] = feats["post_count"].diff().fillna(0.0)
        feats["post_volume_pct_change_1"] = (
            feats["post_count"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )

        # 2. Lagged Sentiment Scores (1, 2, 3 periods back)
        for lag in [1, 2, 3]:
            feats[f"sentiment_compound_lag_{lag}"] = feats["mean_compound"].shift(lag).fillna(0.0)
            feats[f"eng_sentiment_lag_{lag}"] = feats["engagement_weighted_sentiment"].shift(lag).fillna(0.0)
            feats[f"post_count_lag_{lag}"] = feats["post_count"].shift(lag).fillna(0.0)

        # 3. Rolling Sentiment Means and Volatility
        for w in self.sentiment_windows:
            feats[f"sentiment_rolling_mean_{w}"] = (
                feats["mean_compound"].rolling(window=w, min_periods=1).mean().fillna(0.0)
            )
            feats[f"sentiment_rolling_std_{w}"] = (
                feats["mean_compound"].rolling(window=w, min_periods=1).std().fillna(0.0)
            )
            feats[f"post_count_rolling_sum_{w}"] = (
                feats["post_count"].rolling(window=w, min_periods=1).sum().fillna(0.0)
            )

        # 4. Sentiment Momentum (Fast - Slow)
        if len(self.sentiment_windows) >= 2:
            fast_w = self.sentiment_windows[0]
            slow_w = self.sentiment_windows[-1]
            feats["sentiment_momentum"] = (
                feats[f"sentiment_rolling_mean_{fast_w}"] - feats[f"sentiment_rolling_mean_{slow_w}"]
            )

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

        # 1. Log Returns and Percentage Changes
        feats["return_1"] = (close / close.shift(1) - 1.0).fillna(0.0)
        feats["log_return_1"] = np.log(close / close.shift(1)).fillna(0.0)

        for w in [3, 6, 12, 24]:
            feats[f"return_{w}"] = (close / close.shift(w) - 1.0).fillna(0.0)
            feats[f"price_volatility_{w}"] = feats["return_1"].rolling(window=w, min_periods=2).std().fillna(0.0)

        # 2. Moving Average Ratios (Close / SMA - 1)
        for w in self.price_windows:
            sma = close.rolling(window=w, min_periods=1).mean()
            feats[f"sma_ratio_{w}"] = (close / sma - 1.0).fillna(0.0)

        # 3. Exponential Moving Averages (EMA 12 and 26)
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()

        feats["macd_normalized"] = (macd / close).fillna(0.0)
        feats["macd_signal_normalized"] = (macd_signal / close).fillna(0.0)
        feats["macd_diff"] = (macd - macd_signal) / close

        # 4. Relative Strength Index (RSI)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=self.rsi_period, min_periods=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period, min_periods=self.rsi_period).mean()

        # Handle zero loss (pure uptrend -> RSI = 100) or zero gain (pure downtrend -> RSI = 0)
        rs = np.where(avg_loss > 1e-12, avg_gain / avg_loss, np.where(avg_gain > 1e-12, 1000.0, 1.0))
        rsi = 100.0 - (100.0 / (1.0 + rs))
        feats["rsi_14"] = (pd.Series(rsi, index=p.index).fillna(50.0)) / 100.0  # Normalized to [0, 1]

        # 5. Price Momentum & Range
        feats["high_low_spread"] = ((p["high"] - p["low"]) / close).fillna(0.0)
        feats["close_open_spread"] = ((close - p["open"]) / close).fillna(0.0)

        # 6. Volume Dynamics
        vol_sma_24 = volume.rolling(window=24, min_periods=1).mean().replace(0, np.nan)
        feats["volume_ratio_24"] = (volume / vol_sma_24).fillna(1.0)
        feats["volume_change_1"] = (volume.pct_change()).replace([np.inf, -np.inf], np.nan).fillna(0.0)

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
        If sentiment is sparse, gaps are sensibly filled with neutral baseline values.
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
        sentiment_fill_zeros = [c for c in sent_feats.columns if "post_count" in c or "volume" in c]
        sentiment_fill_neutral = [c for c in sent_feats.columns if c not in sentiment_fill_zeros]

        for col in sentiment_fill_zeros:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0.0)
        for col in sentiment_fill_neutral:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0.0)

        # Include raw close price and volume for reference and target generation
        merged["close"] = prices_df["close"]
        merged["volume"] = prices_df["volume"]
        merged["open"] = prices_df["open"]

        return merged.sort_index()
