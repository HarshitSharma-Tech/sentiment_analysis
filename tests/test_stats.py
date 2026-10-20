"""Unit tests for time-series statistical hypothesis testing."""
import numpy as np
import pytest
from src.stats import (
    diebold_mariano_test,
    benjamini_hochberg_fdr,
    holm_bonferroni,
    calculate_majority_class_baseline,
)


def test_diebold_mariano_identical_forecasts():
    # If baseline and model are identical, d_mean = 0 and statistic = 0
    actual = np.array([100, 102, 101, 105, 107, 104, 108, 110, 109, 112], dtype=float)
    baseline = actual + np.ones(10)
    model = actual + np.ones(10)

    res = diebold_mariano_test(actual, baseline, model, horizon=1)
    assert res.statistic == 0.0
    assert res.p_value_two_sided == 1.0
    assert not res.is_significant_5pct


def test_diebold_mariano_superior_model():
    # Model has much smaller errors than baseline
    np.random.seed(42)
    actual = np.cumsum(np.random.normal(0, 1, 100))
    baseline = actual + np.random.normal(0, 5, 100)  # large error
    model = actual + np.random.normal(0, 0.5, 100)   # small error

    res = diebold_mariano_test(actual, baseline, model, horizon=1)
    assert res.statistic > 0.0
    assert res.mean_loss_diff > 0.0
    assert res.p_value_one_sided < 0.001
    assert res.is_significant_5pct
    assert res.is_significant_1pct


def test_benjamini_hochberg_fdr():
    # 5 tests with varied p-values
    p_vals = [0.001, 0.008, 0.03, 0.15, 0.60]
    rejections, q_vals = benjamini_hochberg_fdr(p_vals, alpha=0.05)

    assert len(rejections) == 5
    assert len(q_vals) == 5
    # First two p-values should be rejected under FDR=0.05
    assert rejections[0] == True
    assert rejections[1] == True
    assert rejections[4] == False
    assert q_vals[0] <= q_vals[1] <= q_vals[2]


def test_holm_bonferroni():
    p_vals = [0.001, 0.01, 0.04, 0.20]
    rejections, adj_p = holm_bonferroni(p_vals, alpha=0.05)
    assert rejections[0] == True
    assert len(adj_p) == 4


def test_majority_class_baseline():
    # 7 positive returns, 3 negative returns -> majority class baseline is 70%
    returns = np.array([0.01, 0.02, -0.01, 0.03, 0.04, -0.02, 0.01, 0.02, -0.05, 0.03])
    base = calculate_majority_class_baseline(returns)
    assert base == 70.0
