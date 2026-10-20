"""Unit tests for quantitative strategy backtesting and risk metrics."""
import numpy as np
import pandas as pd
import pytest
from src.backtest import StrategyBacktester


def test_strategy_backtester_metrics():
    dates = pd.date_range("2024-01-01 00:00:00", periods=100, freq="1h", tz="UTC")
    # Alternating positive and negative returns
    actual_returns = np.array([0.005, -0.003] * 50)
    # Model correctly predicts sign
    pred_returns = np.array([0.004, -0.002] * 50)

    preds_df = pd.DataFrame({
        "model_name": ["ridge"] * 100,
        "horizon": [1] * 100,
        "forecast_timestamp": dates.strftime("%Y-%m-%d %H:%M:%S"),
        "actual_return": actual_returns,
        "predicted_return": pred_returns,
    })

    backtester = StrategyBacktester(
        initial_capital=100_000.0,
        transaction_cost_bps=5.0,
        signal_threshold=0.001,
        frequency="1h",
    )
    res = backtester.run_backtest(preds_df, model_name="ridge", horizon=1)

    assert res.total_bars == 100
    assert res.total_trades > 0
    assert res.win_rate_pct > 80.0
    assert res.sharpe_ratio > 0
    assert res.max_drawdown_pct <= 0.0
    assert 0.0 <= res.exposure_pct <= 100.0
    assert res.turnover_pct >= 0.0
    assert len(res.no_trade_curve) == 100


def test_evaluate_all_models_summary():
    dates = pd.date_range("2024-01-01 00:00:00", periods=50, freq="1h", tz="UTC")
    preds_df = pd.DataFrame({
        "model_name": ["ridge"] * 50 + ["random_forest"] * 50,
        "horizon": [1] * 100,
        "forecast_timestamp": list(dates.strftime("%Y-%m-%d %H:%M:%S")) * 2,
        "actual_return": [0.002, -0.001] * 50,
        "predicted_return": [0.003, -0.001] * 50,
    })

    backtester = StrategyBacktester(initial_capital=100_000.0, frequency="1h")
    summary = backtester.evaluate_all_models(preds_df, horizon=1)
    assert len(summary) >= 3  # no_trade_baseline, ridge, random_forest
    assert "no_trade_baseline" in summary["model_name"].values
    assert "exposure_pct" in summary.columns
    assert "turnover_pct" in summary.columns


def test_zero_volatility_handling():
    # If returns are completely flat (zero volatility)
    dates = pd.date_range("2024-01-01 00:00:00", periods=50, freq="1h", tz="UTC")
    preds_df = pd.DataFrame({
        "model_name": ["ridge"] * 50,
        "horizon": [1] * 50,
        "forecast_timestamp": dates.strftime("%Y-%m-%d %H:%M:%S"),
        "actual_return": np.zeros(50),
        "predicted_return": np.zeros(50),
    })

    backtester = StrategyBacktester(initial_capital=100_000.0, frequency="1h")
    # Must not throw ZeroDivisionError
    res = backtester.run_backtest(preds_df, model_name="ridge", horizon=1)
    assert res.sharpe_ratio == 0.0
    assert res.sortino_ratio == 0.0
    assert res.cumulative_return_pct == 0.0
    assert res.exposure_pct == 0.0
    assert hasattr(res, "autocorr_lag1")


def test_split_type_filtering():
    dates = pd.date_range("2024-01-01 00:00:00", periods=40, freq="1h", tz="UTC")
    preds_df = pd.DataFrame({
        "model_name": ["ridge"] * 40,
        "horizon": [1] * 40,
        "split_type": ["dev_walk_forward"] * 25 + ["frozen_holdout"] * 15,
        "forecast_timestamp": dates.strftime("%Y-%m-%d %H:%M:%S"),
        "actual_return": [0.002, -0.001] * 20,
        "predicted_return": [0.003, -0.001] * 20,
    })

    backtester = StrategyBacktester(initial_capital=100_000.0, frequency="1h")
    res_holdout = backtester.run_backtest(preds_df, model_name="ridge", horizon=1, split_type="frozen_holdout")
    assert res_holdout.total_bars == 15
    assert res_holdout.split_type == "frozen_holdout"

    res_dev = backtester.run_backtest(preds_df, model_name="ridge", horizon=1, split_type="dev_walk_forward")
    assert res_dev.total_bars == 25

