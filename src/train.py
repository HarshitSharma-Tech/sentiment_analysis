"""
Model Training and Persistence Module
Implements regularized, variance-controlled multi-horizon model architectures:
1. Persistence Baseline (Zero-return benchmark)
2. Ridge Regression (L2 regularized with RidgeCV tuning on training folds)
3. Random Forest Regressor (Constrained depth, sqrt feature sampling, large min leaf)
4. HistGradientBoostingRegressor (Constrained depth, strong L2 penalty, conservative learning rate)

Saves trained model artifacts, scalers, and metadata using Joblib.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class PersistenceModel:
    """Zero-return benchmark predicting no price change from cutoff."""

    def __init__(self):
        self.is_fitted = True

    def fit(self, X: Any, y: Any) -> "PersistenceModel":
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        n_samples = len(X)
        return np.zeros(n_samples, dtype=float)


class ModelFactory:
    """Instantiates standardized regressors tuned for noisy financial time-series."""

    SUPPORTED_MODELS = ["persistence", "ridge", "random_forest", "hist_gradient_boosting"]

    @classmethod
    def create_model(cls, model_name: str, random_state: int = 42):
        name = model_name.lower().strip()
        if name == "persistence":
            return PersistenceModel()
        elif name == "ridge":
            # RidgeCV automatically cross-validates alpha on training folds to shrink noise
            return RidgeCV(alphas=[10.0, 50.0, 100.0, 500.0, 1000.0])
        elif name == "random_forest":
            return RandomForestRegressor(
                n_estimators=50,
                max_depth=4,
                min_samples_leaf=25,
                max_features="sqrt",
                random_state=random_state,
                n_jobs=-1,
            )
        elif name in ("hist_gradient_boosting", "histgradientboosting"):
            return HistGradientBoostingRegressor(
                max_iter=50,
                max_depth=3,
                min_samples_leaf=30,
                l2_regularization=25.0,
                learning_rate=0.03,
                random_state=random_state,
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}. Supported: {cls.SUPPORTED_MODELS}")


class MultiHorizonPipeline:
    """Manages models across all six forecast horizons (n+0 to n+5)."""

    def __init__(
        self,
        model_name: str = "ridge",
        horizons: Optional[List[int]] = None,
        models_dir: Optional[Path] = None,
    ):
        self.model_name = model_name
        self.horizons = horizons or [0, 1, 2, 3, 4, 5]
        self.models_dir = Path(models_dir) if models_dir else Path("models")
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.models: Dict[int, Any] = {}
        self.scalers: Dict[int, StandardScaler] = {}
        self.feature_names: Dict[int, List[str]] = {}

    def fit_horizon(
        self,
        horizon: int,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:
        """Fit scaler and model for a single horizon strictly on training data."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        model = ModelFactory.create_model(self.model_name)
        model.fit(X_scaled, y_train)

        self.models[horizon] = model
        self.scalers[horizon] = scaler
        self.feature_names[horizon] = list(X_train.columns)

    def predict_horizon(
        self,
        horizon: int,
        X: pd.DataFrame,
        cutoff_prices: pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict return and implied price for a given horizon.
        Returns:
            (predicted_returns, predicted_prices)
        """
        if horizon not in self.models:
            raise ValueError(f"Model for horizon {horizon} has not been fitted.")

        scaler = self.scalers[horizon]
        model = self.models[horizon]

        X_scaled = scaler.transform(X)
        pred_returns = model.predict(X_scaled)
        pred_prices = cutoff_prices.values * (1.0 + pred_returns)

        return pred_returns, pred_prices

    def save(self, run_id: str) -> Path:
        """Persist model bundle, scalers, and metadata to disk."""
        target_file = self.models_dir / f"{self.model_name}_{run_id}.joblib"
        bundle = {
            "model_name": self.model_name,
            "run_id": run_id,
            "horizons": self.horizons,
            "models": self.models,
            "scalers": self.scalers,
            "feature_names": self.feature_names,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        joblib.dump(bundle, target_file)
        logger.info(f"Saved model pipeline bundle to {target_file}")
        return target_file

    @classmethod
    def load(cls, file_path: Path) -> "MultiHorizonPipeline":
        """Load a saved pipeline bundle from disk."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}")
        bundle = joblib.load(p)
        pipeline = cls(model_name=bundle["model_name"], horizons=bundle["horizons"])
        pipeline.models = bundle["models"]
        pipeline.scalers = bundle["scalers"]
        pipeline.feature_names = bundle["feature_names"]
        return pipeline
