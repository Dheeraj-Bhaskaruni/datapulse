"""Market outcome prediction model with calibration."""

import logging
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class MarketPredictorModel(BaseModel):
    """Calibrated GBM classifier for market outcome prediction."""

    def __init__(self, n_estimators: int = 200, learning_rate: float = 0.05, max_depth: int = 4):
        super().__init__()
        self._base_model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
        )
        self._model = None
        self.cv_metrics: Dict[str, Any] = {}

    def train(self, X: pd.DataFrame, y: pd.Series) -> "MarketPredictorModel":
        """Train the calibrated classifier."""
        self.feature_names = list(X.columns)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self._model = CalibratedClassifierCV(self._base_model, cv=5, method='isotonic')
        self._model.fit(X_train, y_train)

        y_pred = self._model.predict(X_test)
        y_proba = self._model.predict_proba(X_test)[:, 1]

        self.cv_metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'log_loss': float(log_loss(y_test, y_proba)),
            'brier_score': float(brier_score_loss(y_test, y_proba)),
            'roc_auc': float(roc_auc_score(y_test, y_proba)),
        }

        self._trained = True
        logger.info(f"MarketPredictorModel trained — AUC: {self.cv_metrics['roc_auc']:.3f}")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        return self._model.predict(df[self.feature_names] if self.feature_names else df)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        return self._model.predict_proba(df[self.feature_names] if self.feature_names else df)

    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        joblib.dump(self, Path(path) / "model.joblib")

    def load(self, path: str) -> "MarketPredictorModel":
        loaded = joblib.load(Path(path) / "model.joblib")
        self.__dict__.update(loaded.__dict__)
        return self
