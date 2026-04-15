"""Base model class for DataPulse models."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all DataPulse models."""

    def __init__(self):
        self.version: str = "1.0.0"
        self.feature_names: List[str] = []
        self._model = None
        self._trained = False

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return predictions for input DataFrame."""

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return class probabilities (classifiers only)."""
        raise NotImplementedError("predict_proba is not available for this model.")

    def anomaly_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Return anomaly scores (anomaly models only)."""
        raise NotImplementedError("anomaly_scores is not available for this model.")

    def save(self, path: str) -> None:
        """Save model artifacts to directory using joblib."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, save_dir / "model.joblib")
        logger.info(f"Model saved to {save_dir}")

    def load(self, path: str) -> "BaseModel":
        """Load model artifacts from directory using joblib."""
        load_path = Path(path) / "model.joblib"
        loaded = joblib.load(load_path)
        self.__dict__.update(loaded.__dict__)
        logger.info(f"Model loaded from {load_path}")
        return self
