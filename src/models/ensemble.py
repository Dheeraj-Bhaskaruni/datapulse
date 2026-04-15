"""Weighted ensemble model wrapper."""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class EnsembleModel:
    """Weighted ensemble that aggregates predictions from multiple models."""

    def __init__(self, models: List[BaseModel], weights: Optional[List[float]] = None):
        if not models:
            raise ValueError("At least one model is required.")
        self.models = models
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models.")
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return weighted average of model predictions."""
        predictions = np.array([model.predict(df) for model in self.models])
        return np.average(predictions, axis=0, weights=self.weights)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return weighted average of class probability predictions."""
        probas = np.array([model.predict_proba(df) for model in self.models])
        return np.average(probas, axis=0, weights=self.weights)
