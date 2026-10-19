"""Unit tests for walk-forward validation and directional accuracy."""
import numpy as np
import pandas as pd
import pytest
from src.targets import TargetEngineer
from src.validate import WalkForwardValidator


def test_directional_accuracy():
    actual = np.array([0.02, -0.01, 0.05, -0.03, 0.01])
    # Predicted with 4 out of 5 matching signs
    pred = np.array([0.01, -0.02, 0.03, -0.01, -0.02])
    acc = WalkForwardValidator.calculate_directional_accuracy(actual, pred)
    assert acc == 80.0


def test_walk_forward_evaluation_mock():
    # 300 periods mock data
    dates = pd.date_range("2024-01-01 00:00:00", periods=300, freq="1h", tz="UTC")
    closes = 40000.0 + np.cumsum(np.random.normal(0, 50, 300))
    df = pd.DataFrame({
        "close": closes,
        "feat1": np.random.normal(0, 1, 300),
        "feat2": np.random.normal(0, 1, 300),
    }, index=dates)

    te = TargetEngineer()
    df_targets = te.create_targets(df)

    validator = WalkForwardValidator(min_train_size=100, step_size=24, test_size=24)
    summary_h1, preds_h1 = validator.evaluate_model_horizon(
        model_name="ridge",
        horizon=1,
        df_with_targets=df_targets,
        feature_cols=["feat1", "feat2"],
        run_id="test_run",
    )

    assert summary_h1["test_bars"] > 100
    assert summary_h1["mae"] > 0
    assert summary_h1["directional_accuracy"] >= 0.0
    assert len(preds_h1) == summary_h1["test_bars"]


def test_distinct_metrics_across_horizons():
    dates = pd.date_range("2024-01-01 00:00:00", periods=250, freq="1h", tz="UTC")
    np.random.seed(42)
    closes = 50000.0 + np.cumsum(np.random.normal(0, 100, 250))
    df = pd.DataFrame({
        "close": closes,
        "open": closes - np.random.uniform(5, 20, 250),
        "f1": np.random.normal(0, 1, 250),
    }, index=dates)

    te = TargetEngineer()
    df_targets = te.create_targets(df)

    validator = WalkForwardValidator(min_train_size=80, step_size=24, test_size=24)
    summary_h0, _ = validator.evaluate_model_horizon("persistence", 0, df_targets, ["f1"], "run_test")
    summary_h1, _ = validator.evaluate_model_horizon("persistence", 1, df_targets, ["f1"], "run_test")
    summary_h2, _ = validator.evaluate_model_horizon("persistence", 2, df_targets, ["f1"], "run_test")

    # Baseline MAE must be distinct and increase with horizon span
    assert summary_h0["mae"] != summary_h1["mae"]
    assert summary_h1["mae"] != summary_h2["mae"]
    assert summary_h0["mae"] < summary_h2["mae"]
