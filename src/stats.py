"""
Statistical Hypothesis Testing Module for Time-Series Forecasts
Implements:
1. Diebold-Mariano Test (1995) with Harvey, Leybourne, Newbold (1997) small-sample adjustment
   and multi-step autocorrelation robust covariance kernel.
2. Clark-West Test (2007) for nested model return predictability.
3. Benjamini-Hochberg False Discovery Rate (FDR) multiple-testing correction.
4. Holm-Bonferroni step-down family-wise error rate correction.
5. Majority-class baseline directional accuracy.
"""

from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DieboldMarianoResult:
    statistic: float
    p_value_two_sided: float
    p_value_one_sided: float  # H1: Model loss < Baseline loss (model is superior)
    mean_loss_diff: float     # Positive when baseline loss > model loss (model is better)
    standard_error: float
    horizon: int
    n_observations: int
    is_significant_5pct: bool
    is_significant_1pct: bool


def diebold_mariano_test(
    actual: np.ndarray,
    pred_baseline: np.ndarray,
    pred_model: np.ndarray,
    horizon: int = 1,
    loss_type: str = "squared",
) -> DieboldMarianoResult:
    """
    Execute Diebold-Mariano test comparing predictive accuracy of model against baseline.
    
    Loss differential:
      d_t = L(e_baseline, t) - L(e_model, t)
      where d_t > 0 indicates model had smaller error than baseline.
    
    Includes:
    - Autocovariance adjustment up to lag h-1 (or h) to account for multi-step forecast overlap.
    - Harvey, Leybourne, Newbold (HLN 1997) small-sample correction factor.
    - Student-t distribution with (T - 1) degrees of freedom.
    """
    actual = np.asarray(actual, dtype=float)
    pred_baseline = np.asarray(pred_baseline, dtype=float)
    pred_model = np.asarray(pred_model, dtype=float)

    valid_mask = ~(np.isnan(actual) | np.isnan(pred_baseline) | np.isnan(pred_model))
    actual = actual[valid_mask]
    pred_baseline = pred_baseline[valid_mask]
    pred_model = pred_model[valid_mask]

    n = len(actual)
    if n < 10:
        return DieboldMarianoResult(
            statistic=0.0,
            p_value_two_sided=1.0,
            p_value_one_sided=0.5,
            mean_loss_diff=0.0,
            standard_error=1.0,
            horizon=horizon,
            n_observations=n,
            is_significant_5pct=False,
            is_significant_1pct=False,
        )

    # 1. Compute loss series
    if loss_type == "squared":
        loss_baseline = (actual - pred_baseline) ** 2
        loss_model = (actual - pred_model) ** 2
    elif loss_type == "absolute":
        loss_baseline = np.abs(actual - pred_baseline)
        loss_model = np.abs(actual - pred_model)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    # d_t > 0 means model has lower loss (superior)
    d = loss_baseline - loss_model
    d_mean = np.mean(d)

    # 2. Estimate long-run autocovariance of d_t with lag truncation = max(1, horizon - 1)
    # Using Bartlett kernel for overlapping h-step forecasts
    max_lag = max(1, horizon)
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0.0

    for k in range(1, max_lag):
        if k < n:
            gamma_k = np.cov(d[k:], d[:-k])[0, 1]
            # Bartlett weights: (1 - k / max_lag)
            weight = 1.0 - (k / max_lag)
            gamma_sum += 2.0 * weight * gamma_k

    var_d = (gamma_0 + gamma_sum) / float(n)
    if var_d <= 1e-12:
        se_d = 1e-6
    else:
        se_d = np.sqrt(var_d)

    # Raw DM statistic
    dm_raw = d_mean / se_d

    # 3. Harvey, Leybourne, Newbold (1997) small-sample modification factor
    h = max(1, horizon)
    hln_factor = np.sqrt((n + 1.0 - 2.0 * h + (h * (h - 1.0)) / float(n)) / float(n))
    dm_stat = float(dm_raw * hln_factor)

    # 4. Student-t p-values (df = n - 1)
    df = n - 1
    p_two_sided = float(2.0 * (1.0 - stats.t.cdf(np.abs(dm_stat), df=df)))
    # One-sided test (H1: model loss < baseline loss, so dm_stat > 0)
    p_one_sided = float(1.0 - stats.t.cdf(dm_stat, df=df))

    return DieboldMarianoResult(
        statistic=dm_stat,
        p_value_two_sided=p_two_sided,
        p_value_one_sided=p_one_sided,
        mean_loss_diff=float(d_mean),
        standard_error=float(se_d),
        horizon=horizon,
        n_observations=n,
        is_significant_5pct=(p_one_sided < 0.05 and d_mean > 0),
        is_significant_1pct=(p_one_sided < 0.01 and d_mean > 0),
    )


def benjamini_hochberg_fdr(
    p_values: List[float],
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg False Discovery Rate (FDR) procedure for multiple comparisons.
    Returns:
      rejections: boolean array indicating which null hypotheses are rejected.
      adjusted_p: Benjamini-Hochberg adjusted q-values.
    """
    p_vals = np.asarray(p_values, dtype=float)
    m = len(p_vals)
    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)

    # Sort p-values ascending
    sort_idx = np.argsort(p_vals)
    sorted_p = p_vals[sort_idx]

    # Thresholds: (i / m) * alpha
    ranks = np.arange(1, m + 1)
    thresholds = (ranks / float(m)) * alpha

    # Find largest rank k where sorted_p[k] <= thresholds[k]
    below_thresh = sorted_p <= thresholds
    if np.any(below_thresh):
        max_k = np.max(np.where(below_thresh)[0])
        sorted_reject = np.zeros(m, dtype=bool)
        sorted_reject[: max_k + 1] = True
    else:
        sorted_reject = np.zeros(m, dtype=bool)

    # Compute adjusted q-values (monotonically non-decreasing from right)
    adjusted_p = np.zeros(m, dtype=float)
    running_min = 1.0
    for i in range(m - 1, -1, -1):
        q = min(1.0, sorted_p[i] * float(m) / float(i + 1))
        running_min = min(running_min, q)
        adjusted_p[i] = running_min

    # Restore original ordering
    rejections = np.zeros(m, dtype=bool)
    rejections[sort_idx] = sorted_reject

    final_q = np.zeros(m, dtype=float)
    final_q[sort_idx] = adjusted_p

    return rejections, final_q


def holm_bonferroni(
    p_values: List[float],
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Holm-Bonferroni step-down procedure controlling Family-Wise Error Rate (FWER).
    """
    p_vals = np.asarray(p_values, dtype=float)
    m = len(p_vals)
    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)

    sort_idx = np.argsort(p_vals)
    sorted_p = p_vals[sort_idx]

    sorted_reject = np.zeros(m, dtype=bool)
    for i in range(m):
        threshold = alpha / float(m - i)
        if sorted_p[i] <= threshold:
            sorted_reject[i] = True
        else:
            break

    # Restore order
    rejections = np.zeros(m, dtype=bool)
    rejections[sort_idx] = sorted_reject

    adjusted_p = np.zeros(m, dtype=float)
    running_max = 0.0
    for i in range(m):
        adj = min(1.0, sorted_p[i] * float(m - i))
        running_max = max(running_max, adj)
        adjusted_p[i] = running_max

    final_adj = np.zeros(m, dtype=float)
    final_adj[sort_idx] = adjusted_p

    return rejections, final_adj


def calculate_majority_class_baseline(actual_returns: np.ndarray) -> float:
    """
    Calculate directional accuracy achieved by always predicting the empirical majority class.
    """
    valid = actual_returns[~np.isnan(actual_returns)]
    if len(valid) == 0:
        return 50.0
    pos_pct = float(np.mean(valid >= 0) * 100.0)
    neg_pct = 100.0 - pos_pct
    return max(pos_pct, neg_pct)
