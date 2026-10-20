"""Unit tests for multi-horizon target engineering."""
import numpy as np
import pandas as pd
import pytest
from src.targets import TargetEngineer


def test_target_generation():
    dates = pd.date_range("2024-01-01 00:00:00", periods=10, freq="1h", tz="UTC")
    closes = np.array([100.0, 105.0, 110.0, 108.0, 112.0, 115.0, 120.0, 118.0, 122.0, 125.0])
    opens = np.array([99.0, 101.0, 106.0, 109.0, 108.0, 113.0, 116.0, 119.0, 117.0, 123.0])
    df = pd.DataFrame({"close": closes, "open": opens, "feat1": np.ones(10)}, index=dates)

    te = TargetEngineer()
    df_targets = te.create_targets(df)

    # Base cutoff price is strictly close(t-1)
    assert df_targets["cutoff_price"].iloc[1] == 100.0

    # Test n+0: target is current close(t)
    # At index 1: cutoff is close(0)=100.0, target is close(1)=105.0, return is (105-100)/100 = 0.05
    assert df_targets["target_price_h0"].iloc[1] == 105.0
    expected_ret_h0 = (105.0 - 100.0) / 100.0
    assert pytest.approx(df_targets["target_return_h0"].iloc[1], 0.0001) == expected_ret_h0

    # Inverse price reconstruction check: P_hat = cutoff * (1 + R_hat)
    reconstructed_p0 = df_targets["cutoff_price"].iloc[1] * (1.0 + df_targets["target_return_h0"].iloc[1])
    assert pytest.approx(reconstructed_p0, 0.0001) == 105.0

    # Test n+1: target is next close(t+1) = close(2) = 110.0
    assert df_targets["target_price_h1"].iloc[1] == 110.0
    expected_ret_h1 = (110.0 - 100.0) / 100.0
    assert pytest.approx(df_targets["target_return_h1"].iloc[1], 0.0001) == expected_ret_h1
    reconstructed_p1 = df_targets["cutoff_price"].iloc[1] * (1.0 + df_targets["target_return_h1"].iloc[1])
    assert pytest.approx(reconstructed_p1, 0.0001) == 110.0

    # Verify that target returns for h0 and h1 are strictly distinct
    assert df_targets["target_return_h0"].iloc[1] != df_targets["target_return_h1"].iloc[1]

    # Test n+5: target is close 5 periods ahead: index 1 + 5 = 6 (120.0)
    assert df_targets["target_price_h5"].iloc[1] == 120.0
    expected_ret_h5 = (120.0 - 100.0) / 100.0
    assert pytest.approx(df_targets["target_return_h5"].iloc[1], 0.0001) == expected_ret_h5


def test_horizon_specs_and_realization_offsets():
    te = TargetEngineer()
    specs = te.get_specs()
    assert len(specs) == 6
    for h in range(6):
        assert specs[h].horizon == h
        # Realization offset: h=0 realizes at t+1h, h=1 realizes at t+2h, etc.
        assert specs[h].realization_offset_hours == h + 1


def test_horizon_dataset_extraction():
    dates = pd.date_range("2024-01-01 00:00:00", periods=10, freq="1h", tz="UTC")
    closes = np.linspace(100, 200, 10)
    df = pd.DataFrame({"close": closes, "open": closes - 1.0, "f1": np.arange(10)}, index=dates)

    te = TargetEngineer()
    df_targets = te.create_targets(df)

    # For horizon 0: features shifted by 1 means index 0 is dropped (9 valid rows)
    X_0, y_ret_0, y_p_0, p_cut_0 = te.get_horizon_dataset(df_targets, horizon=0, feature_cols=["f1"])
    assert len(X_0) == 9

    # For horizon 1: target shifted by -1 means last row is dropped, plus index 0 shifted (8 valid rows)
    X_1, y_ret_1, y_p_1, p_cut_1 = te.get_horizon_dataset(df_targets, horizon=1, feature_cols=["f1"])
    assert len(X_1) == 8

    # For horizon 5: 10 - 5 - 1 = 4 valid rows
    X_5, y_ret_5, y_p_5, p_cut_5 = te.get_horizon_dataset(df_targets, horizon=5, feature_cols=["f1"])
    assert len(X_5) == 4
