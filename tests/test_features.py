"""Unit tests for feature engineering."""
import numpy as np
import pandas as pd
import pytest
from src.features import FeatureEngineer, PRICE_FEATURE_COLS, SENTIMENT_FEATURE_COLS, ALL_FEATURE_COLS


def test_aggregate_posts():
    dates = pd.date_range("2024-01-01 10:00:00", periods=4, freq="15min", tz="UTC")
    posts_df = pd.DataFrame({
        "timestamp": dates,
        "sentiment_compound": [0.5, 0.8, -0.2, 0.0],
        "likes": [10, 100, 5, 2],
        "reposts": [2, 20, 1, 0],
        "views": [200, 2000, 100, 50],
        "author_followers": [500, 5000, 300, 100],
    })

    fe = FeatureEngineer()
    agg = fe.aggregate_posts_by_frequency(posts_df, frequency="1h")

    assert len(agg) == 1
    assert agg["post_count"].iloc[0] == 4
    assert 0.0 < agg["mean_compound"].iloc[0] < 1.0
    assert agg["pos_proportion"].iloc[0] == 0.5
    assert agg["neu_proportion"].iloc[0] == 0.25
    assert agg["neg_proportion"].iloc[0] == 0.25


def test_price_features():
    dates = pd.date_range("2024-01-01 00:00:00", periods=30, freq="1h", tz="UTC")
    closes = np.linspace(40000, 43000, 30)
    prices_df = pd.DataFrame({
        "open": closes - 50,
        "high": closes + 100,
        "low": closes - 100,
        "close": closes,
        "volume": np.full(30, 1000.0),
    }, index=dates)

    fe = FeatureEngineer()
    price_feats = fe.build_price_features(prices_df)

    assert "return_1" in price_feats.columns
    assert "rsi_14" in price_feats.columns
    assert "macd_diff" in price_feats.columns
    assert not price_feats["return_1"].isna().any()
    # In an uptrend, RSI should be above 0.5 (normalized)
    assert price_feats["rsi_14"].iloc[-1] > 0.5


def test_feature_registries():
    assert len(PRICE_FEATURE_COLS) == 13
    assert len(SENTIMENT_FEATURE_COLS) == 11
    assert len(ALL_FEATURE_COLS) == 24
    assert "sentiment_price_interaction" in SENTIMENT_FEATURE_COLS
