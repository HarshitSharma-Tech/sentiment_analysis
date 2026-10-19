"""Unit tests for multi-horizon target engineering."""
import numpy as np
import pandas as pd
import pytest
from src.targets import TargetEngineer


def test_target_generation():
    dates = pd.date_range("2024-01-01 00:00:00", periods=10, freq="1h", tz="UTC")
    closes = np.array([100.0, 105.0, 110.0, 108.0, 112.0, 115.0, 120.0, 118.0, 122.0, 125.0])
    df = pd.DataFrame({"close": closes, "feat1": np.ones(10)}, index=dates)

    te = TargetEngineer()
    df_targets = te.create_targets(df)

    # Test n+0: target is current close, cutoff is previous close
    # At index 1 (close=105.0): cutoff should be 100.0, target should be 105.0, return is +5%
    assert df_targets["cutoff_price_h0"].iloc[1] == 100.0
    assert df_targets["target_price_h0"].iloc[1] == 105.0
    assert pytest.approx(df_targets["target_return_h0"].iloc[1], 0.0001) == 0.05

    # Test n+1: target is next close, cutoff is current close
    # At index 0 (close=100.0): cutoff is 100.0, target is 105.0, return is +5%
    assert df_targets["cutoff_price_h1"].iloc[0] == 100.0
    assert df_targets["target_price_h1"].iloc[0] == 105.0
    assert pytest.approx(df_targets["target_return_h1"].iloc[0], 0.0001) == 0.05

    # Test n+5: target is close 5 periods ahead
    # At index 0: target should be index 5 (115.0), return is +15%
    assert df_targets["target_price_h5"].iloc[0] == 115.0
    assert pytest.approx(df_targets["target_return_h5"].iloc[0], 0.0001) == 0.15


def test_horizon_dataset_extraction():
    dates = pd.date_range("2024-01-01 00:00:00", periods=10, freq="1h", tz="UTC")
    closes = np.linspace(100, 200, 10)
    df = pd.DataFrame({"close": closes, "f1": np.arange(10)}, index=dates)

    te = TargetEngineer()
    df_targets = te.create_targets(df)

    X_0, y_ret_0, y_p_0, p_cut_0 = te.get_horizon_dataset(df_targets, horizon=0, feature_cols=["f1"])
    # Index 0 is NaN for n+0 because cutoff is shifted by 1
    assert len(X_0) == 9

    X_1, y_ret_1, y_p_1, p_cut_1 = te.get_horizon_dataset(df_targets, horizon=1, feature_cols=["f1"])
    # Last index is NaN for n+1 because target is shifted by -1
    assert len(X_1) == 9
