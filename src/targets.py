"""
Multi-Horizon Target Engineering Module
Constructs future price and return targets for horizons n+0 through n+5.

Target Definitions & Forecasting Cutoffs:
- n+0 (Current Period Forecast):
    Cutoff: At bar opening or prior bar close (t-1).
    Target: Closing price of current period P(t).
    Unknown at cutoff: P(t) close.
    Target Return: (P(t) - P(t-1)) / P(t-1).
    Baseline Prediction: P(t-1) (0% return).

- n+1 through n+5 (Forward Horizons):
    Cutoff: Bar close P(t).
    Target: Future closing price P(t + h).
    Target Return: (P(t + h) - P(t)) / P(t).
    Baseline Prediction: P(t) (0% return).

Strictly prevents lookahead leakage: all future targets are aligned with
the exact cutoff timestamp when the prediction is generated.
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
    shift_periods: int


class TargetEngineer:
    """Generates multi-horizon targets and metadata for horizons n+0 to n+5."""

    HORIZONS = [0, 1, 2, 3, 4, 5]

    @classmethod
    def get_specs(cls) -> Dict[int, HorizonTargetSpec]:
        """Return specifications for each horizon."""
        return {
            0: HorizonTargetSpec(
                horizon=0,
                name="n+0 (Current Bar Close)",
                target_price_col="target_price_h0",
                target_return_col="target_return_h0",
                cutoff_price_col="cutoff_price_h0",
                shift_periods=0,
            ),
            1: HorizonTargetSpec(
                horizon=1,
                name="n+1 (1-Period Ahead)",
                target_price_col="target_price_h1",
                target_return_col="target_return_h1",
                cutoff_price_col="cutoff_price_h1",
                shift_periods=1,
            ),
            2: HorizonTargetSpec(
                horizon=2,
                name="n+2 (2-Periods Ahead)",
                target_price_col="target_price_h2",
                target_return_col="target_return_h2",
                cutoff_price_col="cutoff_price_h2",
                shift_periods=2,
            ),
            3: HorizonTargetSpec(
                horizon=3,
                name="n+3 (3-Periods Ahead)",
                target_price_col="target_price_h3",
                target_return_col="target_return_h3",
                cutoff_price_col="cutoff_price_h3",
                shift_periods=3,
            ),
            4: HorizonTargetSpec(
                horizon=4,
                name="n+4 (4-Periods Ahead)",
                target_price_col="target_price_h4",
                target_return_col="target_return_h4",
                cutoff_price_col="cutoff_price_h4",
                shift_periods=4,
            ),
            5: HorizonTargetSpec(
                horizon=5,
                name="n+5 (5-Periods Ahead)",
                target_price_col="target_price_h5",
                target_return_col="target_return_h5",
                cutoff_price_col="cutoff_price_h5",
                shift_periods=5,
            ),
        }

    def create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add multi-horizon price and return targets to the feature DataFrame.
        Requires 'close' column in df.
        """
        if "close" not in df.columns:
            raise ValueError("Input DataFrame must contain 'close' price column.")

        out = df.copy()
        close = out["close"]

        # n+0: Forecast current bar close given state at bar start
        # Cutoff price is previous close P(t-1)
        out["cutoff_price_h0"] = close.shift(1)
        out["target_price_h0"] = close
        out["target_return_h0"] = (close - out["cutoff_price_h0"]) / out["cutoff_price_h0"]

        # n+1 through n+5: Forecast future closes given current bar close P(t)
        for h in [1, 2, 3, 4, 5]:
            out[f"cutoff_price_h{h}"] = close
            out[f"target_price_h{h}"] = close.shift(-h)
            out[f"target_return_h{h}"] = (out[f"target_price_h{h}"] - close) / close

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
        """
        specs = self.get_specs()[horizon]
        target_ret_col = specs.target_return_col
        target_price_col = specs.target_price_col
        cutoff_col = specs.cutoff_price_col

        # For n+0, features must NOT use period t's close!
        # Features for n+0 must be shifted by 1 so only info up to t-1 is used
        if horizon == 0:
            X_df = df_with_targets[feature_cols].shift(1)
        else:
            X_df = df_with_targets[feature_cols]

        y_ret = df_with_targets[target_ret_col]
        y_price = df_with_targets[target_price_col]
        p_cutoff = df_with_targets[cutoff_col]

        # Valid rows where both features and target are non-NaN
        valid_mask = (~X_df.isna().any(axis=1)) & (~y_ret.isna()) & (~p_cutoff.isna())

        return (
            X_df.loc[valid_mask],
            y_ret.loc[valid_mask],
            y_price.loc[valid_mask],
            p_cutoff.loc[valid_mask],
        )
