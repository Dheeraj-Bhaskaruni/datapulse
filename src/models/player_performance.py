"""Player performance prediction model."""

import logging
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class PlayerPerformanceModel(BaseModel):
    """Gradient Boosting regressor for predicting player fantasy points."""

    def __init__(self, n_estimators: int = 200, learning_rate: float = 0.05, max_depth: int = 4):
        super().__init__()
        self._model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
        )
        self.feature_importances_: np.ndarray = np.array([])
        self.cv_metrics: Dict[str, Any] = {}

    def train(self, X: pd.DataFrame, y: pd.Series) -> "PlayerPerformanceModel":
        """Train the model and compute cross-validation metrics."""
        self.feature_names = list(X.columns)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self._model.fit(X_train, y_train)

        y_pred = self._model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        cv_scores = cross_val_score(self._model, X, y, cv=5, scoring='neg_mean_squared_error')
        cv_rmse = float(np.sqrt(-cv_scores.mean()))

        self.cv_metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'cv_rmse': cv_rmse,
            'cv_rmse_std': float(np.sqrt(-cv_scores).std()),
        }

        self.feature_importances_ = self._model.feature_importances_
        self._trained = True

        logger.info(f"PlayerPerformanceModel trained — RMSE: {rmse:.3f}, R²: {r2:.3f}")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        return self._model.predict(df[self.feature_names] if self.feature_names else df)

    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        joblib.dump(self, Path(path) / "model.joblib")

    def load(self, path: str) -> "PlayerPerformanceModel":
        loaded = joblib.load(Path(path) / "model.joblib")
        self.__dict__.update(loaded.__dict__)
        return self
