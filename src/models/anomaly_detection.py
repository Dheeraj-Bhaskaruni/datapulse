"""Anomaly detection model using Isolation Forest."""

import logging
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class AnomalyDetectionModel(BaseModel):
    """Isolation Forest model for unsupervised anomaly detection."""

    def __init__(self, n_estimators: int = 200, contamination: float = 0.05):
        super().__init__()
        self._model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        self._scaler = StandardScaler()

    def train(self, X: pd.DataFrame) -> "AnomalyDetectionModel":
        """Fit the anomaly detector (unsupervised — no y required)."""
        self.feature_names = list(X.columns)
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled)
        self._trained = True
        logger.info(f"AnomalyDetectionModel trained on {X.shape[0]} samples, {X.shape[1]} features")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return labels: 1 for normal, -1 for anomaly."""
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        X = df[self.feature_names] if self.feature_names else df
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled)

    def anomaly_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Return raw anomaly scores. More negative = more anomalous."""
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        X = df[self.feature_names] if self.feature_names else df
        X_scaled = self._scaler.transform(X)
        return self._model.score_samples(X_scaled)

    @staticmethod
    def statistical_anomalies(
        df: pd.DataFrame, columns: List[str], z_threshold: float = 3.0
    ) -> pd.DataFrame:
        """Flag anomalies per column using z-score thresholding.

        Returns a copy of df with boolean flag columns and a 'total_anomaly_flags' summary column.
        """
        result = df.copy()
        for col in columns:
            if col in result.columns:
                z_scores = np.abs((result[col] - result[col].mean()) / result[col].std())
                result[f"{col}_anomaly"] = z_scores > z_threshold

        flag_cols = [f"{col}_anomaly" for col in columns if col in df.columns]
        result["total_anomaly_flags"] = result[flag_cols].sum(axis=1)
        return result

    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        joblib.dump(self, Path(path) / "model.joblib")

    def load(self, path: str) -> "AnomalyDetectionModel":
        loaded = joblib.load(Path(path) / "model.joblib")
        self.__dict__.update(loaded.__dict__)
        return self
