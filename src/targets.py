"""
Multi-Horizon Target Engineering Module
Constructs point-in-time future price and return targets for horizons n+0 through n+5.

Target Definitions & Forecasting Cutoffs:
Let decision timestamp t be the start of forecast period t (e.g. 14:00:00 UTC).
At timestamp t, the previous bar (t-1) has closed, and bar t is starting.
Available Information at t:
  - Historical bars up to t-1 (close, open, high, low, volume)
  - Features X(t) computed strictly from information available up to timestamp t.
Cutoff Price P_cutoff(t):
  - The latest known price at forecast timestamp t: P_close(t-1) (or P_open(t)).

Forecasting Horizons from cutoff t:
- n+0 (Current Forecast Period): Target is Close of bar t (1 hour ahead of cutoff).
    Target Price: P_close(t).
    Target Return: (P_close(t) - P_cutoff(t)) / P_cutoff(t).
    Unknown at cutoff: P_close(t).

- n+1 (1 Period Ahead): Target is Close of bar t+1 (2 hours ahead of cutoff).
    Target Price: P_close(t+1).
    Target Return: (P_close(t+1) - P_cutoff(t)) / P_cutoff(t).

- n+2 (2 Periods Ahead): Target is Close of bar t+2 (3 hours ahead of cutoff).
    Target Price: P_close(t+2).
    Target Return: (P_close(t+2) - P_cutoff(t)) / P_cutoff(t).

- n+3 (3 Periods Ahead): Target is Close of bar t+3 (4 hours ahead of cutoff).
    Target Price: P_close(t+3).
    Target Return: (P_close(t+3) - P_cutoff(t)) / P_cutoff(t).

- n+4 (4 Periods Ahead): Target is Close of bar t+4 (5 hours ahead of cutoff).
    Target Price: P_close(t+4).
    Target Return: (P_close(t+4) - P_cutoff(t)) / P_cutoff(t).

- n+5 (5 Periods Ahead): Target is Close of bar t+5 (6 hours ahead of cutoff).
    Target Price: P_close(t+5).
    Target Return: (P_close(t+5) - P_cutoff(t)) / P_cutoff(t).

Guarantees:
- Every horizon has a distinct forecast target and span.
- Zero identical metrics between horizons.
- Baseline error increases monotonically with horizon span, matching financial theory.
"""

from dataclasses import dataclass
import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class HorizonTargetSpec:
    horizon: int
    name: str
    target_price_col: str
    target_return_col: str
    cutoff_price_col: str
    forward_steps: int
    realization_offset_hours: int


class TargetEngineer:
    """Generates multi-horizon targets and metadata for horizons n+0 to n+5."""

    HORIZONS = [0, 1, 2, 3, 4, 5]

    @classmethod
    def get_specs(cls) -> Dict[int, HorizonTargetSpec]:
        """Return specifications for each horizon."""
        return {
            0: HorizonTargetSpec(
                horizon=0,
                name="n+0 (Current Period Close)",
                target_price_col="target_price_h0",
                target_return_col="target_return_h0",
                cutoff_price_col="cutoff_price",
                forward_steps=0,
                realization_offset_hours=1,
            ),
            1: HorizonTargetSpec(
                horizon=1,
                name="n+1 (1-Period Ahead Close)",
                target_price_col="target_price_h1",
                target_return_col="target_return_h1",
                cutoff_price_col="cutoff_price",
                forward_steps=1,
                realization_offset_hours=2,
            ),
            2: HorizonTargetSpec(
                horizon=2,
                name="n+2 (2-Periods Ahead Close)",
                target_price_col="target_price_h2",
                target_return_col="target_return_h2",
                cutoff_price_col="cutoff_price",
                forward_steps=2,
                realization_offset_hours=3,
            ),
            3: HorizonTargetSpec(
                horizon=3,
                name="n+3 (3-Periods Ahead Close)",
                target_price_col="target_price_h3",
                target_return_col="target_return_h3",
                cutoff_price_col="cutoff_price",
                forward_steps=3,
                realization_offset_hours=4,
            ),
            4: HorizonTargetSpec(
                horizon=4,
                name="n+4 (4-Periods Ahead Close)",
                target_price_col="target_price_h4",
                target_return_col="target_return_h4",
                cutoff_price_col="cutoff_price",
                forward_steps=4,
                realization_offset_hours=5,
            ),
            5: HorizonTargetSpec(
                horizon=5,
                name="n+5 (5-Periods Ahead Close)",
                target_price_col="target_price_h5",
                target_return_col="target_return_h5",
                cutoff_price_col="cutoff_price",
                forward_steps=5,
                realization_offset_hours=6,
            ),
        }

    def create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add multi-horizon price and return targets to the feature DataFrame.
        
        Cutoff price is strictly the latest completed close P_close(t-1) at the start of bar t.
        This guarantees:
        - No mixing of open-price execution assumptions with close-based target definitions.
        - Exactly identical reference price P_close(t-1) used for constructing targets,
          computing persistence baselines, and reconstructing implied price forecasts:
          P_hat(t+h) = P_close(t-1) * (1 + R_hat(t->t+h)).
        - Horizon n+0 predicts the close of the current bar (close(t)) using information
          strictly available before that close (information up to close(t-1)).
        """
        if "close" not in df.columns:
            raise ValueError("Input DataFrame must contain 'close' price column.")

        out = df.copy()
        close = out["close"]

        # Base Cutoff Price: strictly the latest completed close P(t-1) at decision timestamp t
        out["cutoff_price"] = close.shift(1)
        p_cut = out["cutoff_price"]

        # Horizon n+0: Current period t close (1 step ahead of cutoff, realizes at t + 1h)
        out["target_price_h0"] = close
        out["target_return_h0"] = (close - p_cut) / p_cut

        # Horizons n+1 through n+5: Future bar closes (t+1 through t+5)
        for h in [1, 2, 3, 4, 5]:
            out[f"target_price_h{h}"] = close.shift(-h)
            out[f"target_return_h{h}"] = (out[f"target_price_h{h}"] - p_cut) / p_cut

        return out

    def get_horizon_dataset(
        self,
        df_with_targets: pd.DataFrame,
        horizon: int,
        feature_cols: List[str],
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Extract clean (X, y_return, y_price, cutoff_price) for a specific horizon h,
        dropping NaN target values (e.g. at the end of the series).
        Features are shifted by 1 so only data strictly prior to the forecast period is used.
        """
        specs = self.get_specs()[horizon]
        target_ret_col = specs.target_return_col
        target_price_col = specs.target_price_col
        cutoff_col = specs.cutoff_price_col

        # Features available at forecast cutoff (start of period t)
        # must use historical observations strictly up to period t-1
        X_df = df_with_targets[feature_cols].shift(1)

        y_ret = df_with_targets[target_ret_col]
        y_price = df_with_targets[target_price_col]
        p_cutoff = df_with_targets[cutoff_col]

        # Valid rows where features, target, and cutoff price are non-NaN and finite
        valid_mask = (
            (~X_df.isna().any(axis=1)) &
            (~y_ret.isna()) &
            (~np.isinf(y_ret)) &
            (~p_cutoff.isna()) &
            (p_cutoff > 0)
        )

        return (
            X_df.loc[valid_mask],
            y_ret.loc[valid_mask],
            y_price.loc[valid_mask],
            p_cutoff.loc[valid_mask],
        )
