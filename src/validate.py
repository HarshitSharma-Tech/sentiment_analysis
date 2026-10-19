"""
Chronological Walk-Forward Validation Engine
Validates time-series forecasting models with strict quant-grade leak prevention:
- Expanding/rolling chronological training windows
- Purged training boundaries (eliminates multi-step overlapping target leakage)
- Preprocessing (StandardScaler) fitted strictly on training slices
- Out-of-sample evaluation on unseen future observations
- Benchmarks every model against the Naive Persistence baseline across all 6 horizons
- Reports MAE, RMSE, MAPE, Directional Accuracy (hit rate), and R²
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.database import DatabaseManager
from src.targets import TargetEngineer
from src.train import ModelFactory

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """Executes chronological walk-forward cross validation with target purging."""

    def __init__(
        self,
        min_train_size: int = 480,
        step_size: int = 168,
        test_size: int = 168,
        expanding: bool = True,
        db_manager: Optional[DatabaseManager] = None,
    ):
        self.min_train_size = min_train_size
        self.step_size = step_size
        self.test_size = test_size
        self.expanding = expanding
        self.db = db_manager
        self.target_engineer = TargetEngineer()

    @staticmethod
    def calculate_directional_accuracy(actual_returns: np.ndarray, pred_returns: np.ndarray) -> float:
        """
        Compute percentage of correct directional calls.
        Direction is positive if return >= 0, negative if return < 0.
        """
        if len(actual_returns) == 0:
            return 0.0
        matches = np.sum((pred_returns >= 0) == (actual_returns >= 0))
        return float(matches / len(actual_returns) * 100.0)

    def evaluate_model_horizon(
        self,
        model_name: str,
        horizon: int,
        df_with_targets: pd.DataFrame,
        feature_cols: List[str],
        run_id: str,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Execute purged walk-forward cross validation for a single model and horizon.
        """
        X_df, y_ret, y_price, p_cutoff = self.target_engineer.get_horizon_dataset(
            df_with_targets=df_with_targets,
            horizon=horizon,
            feature_cols=feature_cols,
        )

        n_samples = len(X_df)
        if n_samples < self.min_train_size + self.test_size:
            raise ValueError(
                f"Insufficient samples ({n_samples}) for min_train={self.min_train_size} + test={self.test_size}."
            )

        all_preds: List[Dict[str, Any]] = []
        actual_prices_list: List[float] = []
        pred_prices_list: List[float] = []
        actual_rets_list: List[float] = []
        pred_rets_list: List[float] = []
        base_prices_list: List[float] = []

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        # Walk-forward loop
        split_idx = self.min_train_size
        fold_count = 0

        # Purge window: for horizon h, target uses price at t+h
        # To prevent overlap with test set starting at split_idx,
        # train targets must end h steps before split_idx
        purge_steps = horizon

        while split_idx < n_samples:
            test_end_idx = min(split_idx + self.test_size, n_samples)
            if test_end_idx - split_idx < 1:
                break

            train_start_idx = 0 if self.expanding else max(0, split_idx - self.min_train_size)
            train_end_idx = max(train_start_idx + 24, split_idx - purge_steps)

            # Training slices (strictly purged of forward target overlap)
            X_train = X_df.iloc[train_start_idx:train_end_idx]
            y_ret_train = y_ret.iloc[train_start_idx:train_end_idx]

            # Test slices (strictly future unseen bars)
            X_test = X_df.iloc[split_idx:test_end_idx]
            y_ret_test = y_ret.iloc[split_idx:test_end_idx]
            y_price_test = y_price.iloc[split_idx:test_end_idx]
            p_cutoff_test = p_cutoff.iloc[split_idx:test_end_idx]

            # Fit scaler strictly on training split
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Fit model strictly on training data
            model = ModelFactory.create_model(model_name)
            model.fit(X_train_scaled, y_ret_train)

            # Predict returns out-of-sample
            if model_name == "persistence":
                pred_ret = np.zeros(len(X_test), dtype=float)
            else:
                pred_ret = model.predict(X_test_scaled)

            # Derive implied predicted prices from cutoff price
            pred_price = p_cutoff_test.values * (1.0 + pred_ret)
            base_price = p_cutoff_test.values  # Persistence baseline: price unchanged

            # Collect individual predictions
            for i, ts in enumerate(X_test.index):
                actual_p = float(y_price_test.iloc[i])
                pred_p = float(pred_price[i])
                act_r = float(y_ret_test.iloc[i])
                prd_r = float(pred_ret[i])
                base_p = float(base_price[i])

                actual_prices_list.append(actual_p)
                pred_prices_list.append(pred_p)
                actual_rets_list.append(act_r)
                pred_rets_list.append(prd_r)
                base_prices_list.append(base_p)

                # Target timestamp is h periods ahead of forecast timestamp ts
                target_ts = ts + pd.Timedelta(hours=horizon)

                all_preds.append({
                    "model_name": model_name,
                    "horizon": horizon,
                    "forecast_timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "target_timestamp": target_ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "actual_price": actual_p,
                    "predicted_price": pred_p,
                    "actual_return": act_r,
                    "predicted_return": prd_r,
                    "run_id": run_id,
                    "created_at": now_utc,
                })

            split_idx += self.step_size
            fold_count += 1

        # Calculate Out-of-Sample Metrics across all test folds
        act_arr = np.array(actual_prices_list)
        pred_arr = np.array(pred_prices_list)
        base_arr = np.array(base_prices_list)
        act_ret_arr = np.array(actual_rets_list)
        pred_ret_arr = np.array(pred_rets_list)

        mae = float(mean_absolute_error(act_arr, pred_arr))
        rmse = float(np.sqrt(mean_squared_error(act_arr, pred_arr)))
        mape = float(np.mean(np.abs((act_arr - pred_arr) / act_arr)) * 100.0)
        dir_acc = self.calculate_directional_accuracy(act_ret_arr, pred_ret_arr)
        r2 = float(r2_score(act_arr, pred_arr)) if len(act_arr) > 1 else 0.0

        base_mae = float(mean_absolute_error(act_arr, base_arr))
        base_rmse = float(np.sqrt(mean_squared_error(act_arr, base_arr)))

        summary = {
            "run_id": run_id,
            "model_name": model_name,
            "horizon": horizon,
            "test_bars": len(act_arr),
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "directional_accuracy": dir_acc,
            "r2_score": r2,
            "baseline_mae": base_mae,
            "baseline_rmse": base_rmse,
            "mae_improvement_pct": float((base_mae - mae) / base_mae * 100.0) if base_mae > 0 else 0.0,
            "rmse_improvement_pct": float((base_rmse - rmse) / base_rmse * 100.0) if base_rmse > 0 else 0.0,
        }

        return summary, all_preds

    def run_full_validation(
        self,
        df_with_targets: pd.DataFrame,
        feature_cols: List[str],
        models: Optional[List[str]] = None,
        horizons: Optional[List[int]] = None,
        run_id: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run complete walk-forward validation matrix across models and horizons.
        """
        active_models = models or ["persistence", "ridge", "random_forest", "hist_gradient_boosting"]
        active_horizons = horizons or [0, 1, 2, 3, 4, 5]
        active_run_id = run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        logger.info(
            f"Starting purged walk-forward validation: {len(active_models)} models, "
            f"{len(active_horizons)} horizons, min_train={self.min_train_size}, step={self.step_size}."
        )

        all_summaries: List[Dict[str, Any]] = []
        all_predictions: List[Dict[str, Any]] = []

        for model_name in active_models:
            for h in active_horizons:
                logger.info(f"Evaluating {model_name} on horizon n+{h}...")
                summary, preds = self.evaluate_model_horizon(
                    model_name=model_name,
                    horizon=h,
                    df_with_targets=df_with_targets,
                    feature_cols=feature_cols,
                    run_id=active_run_id,
                )
                all_summaries.append(summary)
                all_predictions.extend(preds)

        # Save predictions to SQLite
        if self.db:
            self.db.save_predictions(all_predictions)

        metrics_df = pd.DataFrame(all_summaries)
        preds_df = pd.DataFrame(all_predictions)

        logger.info(f"Walk-forward validation completed for run {active_run_id}.")
        return metrics_df, preds_df
