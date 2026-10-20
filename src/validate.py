"""
Chronological Walk-Forward Validation & Holdout Engine
Validates time-series forecasting models with institutional quant-grade leak prevention:
- Chronological separation: 70% Development (Walk-Forward CV) / 30% Frozen Holdout
- Purged training boundaries (eliminates multi-step overlapping target leakage)
- 5-bar embargo window (eliminates auto-regressive rolling feature overlap)
- Preprocessing (StandardScaler) fitted strictly on training slices
- Target realization timestamps correctly set to close of target bar (ts + horizon + 1h)
- Diebold-Mariano test (HLN 1997 adjusted) comparing models against Persistence baseline
- Benjamini-Hochberg FDR multiple-testing correction across hypotheses
- Majority-class baseline directional accuracy calculation
- Benchmarks every model against Naive Persistence across all 6 horizons
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.database import DatabaseManager
from src.stats import (
    benjamini_hochberg_fdr,
    calculate_majority_class_baseline,
    diebold_mariano_test,
)
from src.targets import TargetEngineer
from src.train import ModelFactory

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """Executes chronological walk-forward cross validation and frozen holdout evaluation."""

    def __init__(
        self,
        min_train_size: int = 480,
        step_size: int = 168,
        test_size: int = 168,
        embargo_bars: int = 5,
        expanding: bool = True,
        db_manager: Optional[DatabaseManager] = None,
    ):
        self.min_train_size = min_train_size
        self.step_size = step_size
        self.test_size = test_size
        self.embargo_bars = embargo_bars
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
        max_sample_idx: Optional[int] = None,
        split_type: str = "dev_walk_forward",
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Execute purged walk-forward cross validation for a single model and horizon.
        Restricted to max_sample_idx (e.g. 70% development window).
        """
        X_df, y_ret, y_price, p_cutoff = self.target_engineer.get_horizon_dataset(
            df_with_targets=df_with_targets,
            horizon=horizon,
            feature_cols=feature_cols,
        )

        n_samples_total = len(X_df)
        max_idx = min(max_sample_idx, n_samples_total) if max_sample_idx else n_samples_total

        if max_idx < self.min_train_size + self.test_size:
            raise ValueError(
                f"Insufficient samples ({max_idx}) for min_train={self.min_train_size} + test={self.test_size}."
            )

        all_preds: List[Dict[str, Any]] = []
        actual_prices_list: List[float] = []
        pred_prices_list: List[float] = []
        actual_rets_list: List[float] = []
        pred_rets_list: List[float] = []
        base_prices_list: List[float] = []

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        # Walk-forward loop inside development bounds
        split_idx = self.min_train_size
        fold_count = 0

        # Purging & Embargo:
        # Target for horizon h uses close(t+h). To avoid overlap with test fold starting at split_idx,
        # train targets must end h steps before split_idx, PLUS embargo_bars to clear rolling feature memory.
        purge_steps = horizon

        while split_idx < max_idx:
            test_end_idx = min(split_idx + self.test_size, max_idx)
            if test_end_idx - split_idx < 1:
                break

            train_start_idx = 0 if self.expanding else max(0, split_idx - self.min_train_size)
            train_end_idx = max(train_start_idx + 24, split_idx - purge_steps - self.embargo_bars)

            # Training slices (strictly purged and embargoed)
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

            # Derive implied predicted prices from cutoff price P_close(t-1)
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

                # Realization timestamp is strictly the close of target bar t+h: ts + (horizon + 1)h
                target_ts = ts + pd.Timedelta(hours=horizon + 1)

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
                    "split_type": split_type,
                    "created_at": now_utc,
                })

            split_idx += self.step_size
            fold_count += 1

        # Calculate Out-of-Sample Metrics across all development test folds
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
        maj_baseline = calculate_majority_class_baseline(act_ret_arr)

        # Diebold-Mariano test comparing model to persistence baseline
        dm_result = diebold_mariano_test(
            actual=act_arr,
            pred_baseline=base_arr,
            pred_model=pred_arr,
            horizon=horizon + 1,
            loss_type="squared",
        )

        summary = {
            "run_id": run_id,
            "split_type": split_type,
            "model_name": model_name,
            "horizon": horizon,
            "test_bars": len(act_arr),
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "directional_accuracy": dir_acc,
            "majority_class_baseline": maj_baseline,
            "r2_score": r2,
            "baseline_mae": base_mae,
            "baseline_rmse": base_rmse,
            "mae_improvement_pct": float((base_mae - mae) / base_mae * 100.0) if base_mae > 0 else 0.0,
            "rmse_improvement_pct": float((base_rmse - rmse) / base_rmse * 100.0) if base_rmse > 0 else 0.0,
            "dm_statistic": dm_result.statistic,
            "dm_p_value": dm_result.p_value_two_sided,
            "dm_p_value_one_sided": dm_result.p_value_one_sided,
            "dm_significant_5pct": dm_result.is_significant_5pct,
        }

        return summary, all_preds

    def evaluate_frozen_holdout(
        self,
        model_name: str,
        horizon: int,
        df_with_targets: pd.DataFrame,
        feature_cols: List[str],
        dev_end_idx: int,
        run_id: str,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Evaluate model strictly once on the frozen chronological holdout [dev_end_idx, N).
        Trained strictly on development period [0, dev_end_idx - horizon - embargo).
        """
        X_df, y_ret, y_price, p_cutoff = self.target_engineer.get_horizon_dataset(
            df_with_targets=df_with_targets,
            horizon=horizon,
            feature_cols=feature_cols,
        )

        n_samples = len(X_df)
        if dev_end_idx >= n_samples - 24:
            raise ValueError(f"Holdout size ({n_samples - dev_end_idx}) is too small.")

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        # Training slice: development set with purge and embargo
        train_end_idx = max(24, dev_end_idx - horizon - self.embargo_bars)
        X_train = X_df.iloc[:train_end_idx]
        y_ret_train = y_ret.iloc[:train_end_idx]

        # Holdout slice: strictly quarantined future observations
        X_holdout = X_df.iloc[dev_end_idx:]
        y_ret_holdout = y_ret.iloc[dev_end_idx:]
        y_price_holdout = y_price.iloc[dev_end_idx:]
        p_cutoff_holdout = p_cutoff.iloc[dev_end_idx:]

        # Preprocessing fitted strictly on development data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_holdout_scaled = scaler.transform(X_holdout)

        # Fit model on development data
        model = ModelFactory.create_model(model_name)
        model.fit(X_train_scaled, y_ret_train)

        if model_name == "persistence":
            pred_ret = np.zeros(len(X_holdout), dtype=float)
        else:
            pred_ret = model.predict(X_holdout_scaled)

        pred_price = p_cutoff_holdout.values * (1.0 + pred_ret)
        base_price = p_cutoff_holdout.values

        holdout_preds: List[Dict[str, Any]] = []
        for i, ts in enumerate(X_holdout.index):
            act_p = float(y_price_holdout.iloc[i])
            prd_p = float(pred_price[i])
            act_r = float(y_ret_holdout.iloc[i])
            prd_r = float(pred_ret[i])
            target_ts = ts + pd.Timedelta(hours=horizon + 1)

            holdout_preds.append({
                "model_name": model_name,
                "horizon": horizon,
                "forecast_timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "target_timestamp": target_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "actual_price": act_p,
                "predicted_price": prd_p,
                "actual_return": act_r,
                "predicted_return": prd_r,
                "run_id": run_id,
                "split_type": "frozen_holdout",
                "created_at": now_utc,
            })

        act_arr = y_price_holdout.values
        pred_arr = pred_price
        base_arr = base_price
        act_ret_arr = y_ret_holdout.values

        mae = float(mean_absolute_error(act_arr, pred_arr))
        rmse = float(np.sqrt(mean_squared_error(act_arr, pred_arr)))
        mape = float(np.mean(np.abs((act_arr - pred_arr) / act_arr)) * 100.0)
        dir_acc = self.calculate_directional_accuracy(act_ret_arr, pred_ret)
        r2 = float(r2_score(act_arr, pred_arr)) if len(act_arr) > 1 else 0.0

        base_mae = float(mean_absolute_error(act_arr, base_arr))
        base_rmse = float(np.sqrt(mean_squared_error(act_arr, base_arr)))
        maj_baseline = calculate_majority_class_baseline(act_ret_arr)

        dm_result = diebold_mariano_test(
            actual=act_arr,
            pred_baseline=base_arr,
            pred_model=pred_arr,
            horizon=horizon + 1,
            loss_type="squared",
        )

        summary = {
            "run_id": run_id,
            "split_type": "frozen_holdout",
            "model_name": model_name,
            "horizon": horizon,
            "test_bars": len(act_arr),
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "directional_accuracy": dir_acc,
            "majority_class_baseline": maj_baseline,
            "r2_score": r2,
            "baseline_mae": base_mae,
            "baseline_rmse": base_rmse,
            "mae_improvement_pct": float((base_mae - mae) / base_mae * 100.0) if base_mae > 0 else 0.0,
            "rmse_improvement_pct": float((base_rmse - rmse) / base_rmse * 100.0) if base_rmse > 0 else 0.0,
            "dm_statistic": dm_result.statistic,
            "dm_p_value": dm_result.p_value_two_sided,
            "dm_p_value_one_sided": dm_result.p_value_one_sided,
            "dm_significant_5pct": dm_result.is_significant_5pct,
        }

        return summary, holdout_preds

    def run_full_validation(
        self,
        df_with_targets: pd.DataFrame,
        feature_cols: List[str],
        models: Optional[List[str]] = None,
        horizons: Optional[List[int]] = None,
        run_id: Optional[str] = None,
        holdout_ratio: float = 0.30,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run complete quantitative validation pipeline:
        1. 70% Development Set: Purged & Embargoed Walk-Forward Cross Validation.
        2. 30% Frozen Holdout: Evaluated once with models trained on development data.
        3. Multiple-testing Benjamini-Hochberg FDR adjustments applied across all tests.
        Returns:
          (dev_metrics_df, holdout_metrics_df, all_preds_df)
        """
        active_models = models or ["persistence", "ridge", "random_forest", "hist_gradient_boosting"]
        active_horizons = horizons or [0, 1, 2, 3, 4, 5]
        active_run_id = run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        # Determine total sample count from horizon 0 dataset
        X_sample, _, _, _ = self.target_engineer.get_horizon_dataset(df_with_targets, 0, feature_cols)
        n_samples = len(X_sample)
        dev_end_idx = int(n_samples * (1.0 - holdout_ratio))

        logger.info(
            f"Starting Research-Grade Validation: Total={n_samples} bars | "
            f"Development (70%)={dev_end_idx} bars | Frozen Holdout (30%)={n_samples - dev_end_idx} bars | "
            f"Purge=h steps, Embargo={self.embargo_bars} bars."
        )

        dev_summaries: List[Dict[str, Any]] = []
        holdout_summaries: List[Dict[str, Any]] = []
        all_predictions: List[Dict[str, Any]] = []

        # 1. Development Walk-Forward Evaluation
        for model_name in active_models:
            for h in active_horizons:
                logger.info(f"[Dev Walk-Forward] Evaluating {model_name} on horizon n+{h}...")
                s_dev, p_dev = self.evaluate_model_horizon(
                    model_name=model_name,
                    horizon=h,
                    df_with_targets=df_with_targets,
                    feature_cols=feature_cols,
                    run_id=active_run_id,
                    max_sample_idx=dev_end_idx,
                    split_type="dev_walk_forward",
                )
                dev_summaries.append(s_dev)
                all_predictions.extend(p_dev)

        # 2. Frozen Holdout Evaluation
        for model_name in active_models:
            for h in active_horizons:
                logger.info(f"[Frozen Holdout] Evaluating {model_name} on horizon n+{h}...")
                s_hold, p_hold = self.evaluate_frozen_holdout(
                    model_name=model_name,
                    horizon=h,
                    df_with_targets=df_with_targets,
                    feature_cols=feature_cols,
                    dev_end_idx=dev_end_idx,
                    run_id=active_run_id,
                )
                holdout_summaries.append(s_hold)
                all_predictions.extend(p_hold)

        # 3. Benjamini-Hochberg FDR adjustments
        dev_df = pd.DataFrame(dev_summaries)
        hold_df = pd.DataFrame(holdout_summaries)

        # Calculate FDR adjusted p-values across models for non-persistence models
        for df_target in [dev_df, hold_df]:
            p_vals = df_target["dm_p_value_one_sided"].values
            rejections, q_vals = benjamini_hochberg_fdr(p_vals, alpha=0.05)
            df_target["dm_fdr_q_value"] = q_vals
            df_target["dm_fdr_significant"] = rejections

        # Save all predictions to SQLite
        if self.db:
            self.db.save_predictions(all_predictions)

        all_preds_df = pd.DataFrame(all_predictions)
        logger.info(f"Validation completed for run {active_run_id}. Saved {len(all_preds_df)} predictions.")

        return dev_df, hold_df, all_preds_df
