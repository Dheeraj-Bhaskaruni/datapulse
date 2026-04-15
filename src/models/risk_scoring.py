"""User risk scoring classification model."""

import logging
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class RiskScoringModel(BaseModel):
    """Random Forest classifier for user risk categorization."""

    def __init__(self, n_estimators: int = 200, max_depth: int = 8):
        super().__init__()
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
        )
        self.feature_importances_: np.ndarray = np.array([])
        self.cv_metrics: Dict[str, Any] = {}

    def train(self, X: pd.DataFrame, y: pd.Series) -> "RiskScoringModel":
        """Train the classifier and compute cross-validation metrics."""
        self.feature_names = list(X.columns)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self._model.fit(X_train, y_train)

        y_pred = self._model.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))

        cv_scores = cross_val_score(self._model, X, y, cv=5, scoring='accuracy')
        self.cv_metrics = {
            'accuracy': accuracy,
            'f1': f1,
            'cv_accuracy': float(cv_scores.mean()),
            'cv_accuracy_std': float(cv_scores.std()),
        }

        self.feature_importances_ = self._model.feature_importances_
        self._trained = True

        logger.info(f"RiskScoringModel trained — accuracy: {accuracy:.3f}, F1: {f1:.3f}")
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

    def load(self, path: str) -> "RiskScoringModel":
        loaded = joblib.load(Path(path) / "model.joblib")
        self.__dict__.update(loaded.__dict__)
        return self
