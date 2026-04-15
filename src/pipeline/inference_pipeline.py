"""Inference pipeline for real-time predictions."""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

from src.models.player_performance import PlayerPerformanceModel
from src.models.risk_scoring import RiskScoringModel
from src.models.anomaly_detection import AnomalyDetectionModel

logger = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    """Raised when a required model is not found on disk."""
    pass


class InferencePipeline:
    """Handles model loading and real-time inference."""

    def __init__(self, model_path: str = "models"):
        self.model_path = Path(model_path)
        self._player_model: PlayerPerformanceModel = None
        self._risk_model: RiskScoringModel = None
        self._anomaly_model: AnomalyDetectionModel = None
        self._load_models()

    def _load_models(self) -> None:
        """Load all available models from disk."""
        registry = {
            'player_performance': (PlayerPerformanceModel, '_player_model'),
            'risk_scoring': (RiskScoringModel, '_risk_model'),
            'anomaly_detection': (AnomalyDetectionModel, '_anomaly_model'),
        }
        for name, (cls, attr) in registry.items():
            model_dir = self.model_path / name
            if model_dir.exists():
                try:
                    model = cls()
                    model.load(str(model_dir))
                    setattr(self, attr, model)
                    logger.info(f"Loaded model: {name}")
                except Exception as e:
                    logger.error(f"Failed to load model {name}: {e}")

    def _require(self, model, name: str):
        """Raise if a model wasn't loaded."""
        if model is None:
            raise ModelNotFoundError(
                f"Model '{name}' not found at {self.model_path / name}. "
                f"Run 'make train' locally to generate model artifacts."
            )
        return model

    def predict_player(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Predict player performance."""
        model = self._require(self._player_model, 'player_performance')
        df = pd.DataFrame([features])
        prediction = model.predict(df)
        return {
            'prediction': float(prediction[0]),
            'model_version': model.version,
        }

    def score_risk(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Score user risk."""
        model = self._require(self._risk_model, 'risk_scoring')
        df = pd.DataFrame([features])
        prediction = model.predict(df)
        proba = model.predict_proba(df)
        return {
            'risk_category': int(prediction[0]),
            'probabilities': proba[0].tolist(),
            'model_version': model.version,
        }

    def detect_anomaly(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Detect if a data point is anomalous."""
        model = self._require(self._anomaly_model, 'anomaly_detection')
        df = pd.DataFrame([features])
        prediction = model.predict(df)
        scores = model.anomaly_scores(df)
        return {
            'is_anomaly': bool(prediction[0] == -1),
            'anomaly_score': float(scores[0]),
            'model_version': model.version,
        }

    def available_models(self) -> List[str]:
        """List loaded models."""
        loaded = []
        if self._player_model:
            loaded.append('player_performance')
        if self._risk_model:
            loaded.append('risk_scoring')
        if self._anomaly_model:
            loaded.append('anomaly_detection')
        return loaded
