"""Training pipeline for all DataPulse models."""

import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.models.player_performance import PlayerPerformanceModel
from src.models.risk_scoring import RiskScoringModel
from src.models.market_predictor import MarketPredictorModel
from src.models.anomaly_detection import AnomalyDetectionModel

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Orchestrates training for all DataPulse models."""

    def __init__(self, data_path: str, model_path: str):
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)

    def _load_csv(self, name: str) -> pd.DataFrame:
        candidates = [
            self.data_path / f"{name}.csv",
            self.data_path / "sample" / f"{name}.csv",
        ]
        for path in candidates:
            if path.exists():
                logger.info(f"Loading {path}")
                return pd.read_csv(path)
        raise FileNotFoundError(f"Could not find {name}.csv under {self.data_path}")

    def _save_metadata(self, model_dir: Path, metadata: dict) -> None:
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def train_player_performance(self) -> PlayerPerformanceModel:
        """Train player performance regression model."""
        df = self._load_csv("players")

        feature_cols = [
            "games_played", "points_avg", "assists_avg", "rebounds_avg",
            "steals_avg", "blocks_avg", "turnovers_avg", "fg_pct",
            "salary", "consistency_score",
        ]
        feature_cols = [c for c in feature_cols if c in df.columns]
        target_col = "fantasy_points"

        df = df[feature_cols + [target_col]].dropna()
        X = df[feature_cols]
        y = df[target_col]

        model = PlayerPerformanceModel()
        model.train(X, y)

        save_dir = self.model_path / "player_performance"
        model.save(str(save_dir))
        self._save_metadata(save_dir, {
            "model_type": "PlayerPerformanceModel",
            "version": model.version,
            "feature_names": feature_cols,
            "cv_metrics": model.cv_metrics,
            "n_samples": len(df),
        })
        logger.info("PlayerPerformanceModel saved.")
        return model

    def train_risk_scoring(self) -> RiskScoringModel:
        """Train user risk scoring classification model."""
        df = self._load_csv("user_profiles")

        feature_cols = [
            "total_entries", "win_rate", "avg_entry_fee",
            "total_wagered", "total_won", "net_profit",
        ]
        feature_cols = [c for c in feature_cols if c in df.columns]

        df = df[feature_cols + ["risk_score"]].dropna()
        X = df[feature_cols]

        # Bin risk_score into 3 classes: low / medium / high
        y = pd.cut(df["risk_score"], bins=[-1, 33, 66, 100], labels=[0, 1, 2]).astype(int)

        model = RiskScoringModel()
        model.train(X, y)

        save_dir = self.model_path / "risk_scoring"
        model.save(str(save_dir))
        self._save_metadata(save_dir, {
            "model_type": "RiskScoringModel",
            "version": model.version,
            "feature_names": feature_cols,
            "cv_metrics": model.cv_metrics,
            "n_samples": len(df),
        })
        logger.info("RiskScoringModel saved.")
        return model

    def train_market_predictor(self) -> MarketPredictorModel:
        """Train market outcome prediction model."""
        df = self._load_csv("market_odds")

        feature_cols = ["opening_line", "closing_line", "opening_odds", "closing_odds"]
        feature_cols = [c for c in feature_cols if c in df.columns]
        target_col = "result"

        # Encode market_type if present
        if "market_type" in df.columns:
            le = LabelEncoder()
            df["market_type_enc"] = le.fit_transform(df["market_type"].astype(str))
            feature_cols.append("market_type_enc")

        df = df[feature_cols + [target_col]].dropna()
        X = df[feature_cols]
        y = df[target_col].astype(int)

        model = MarketPredictorModel()
        model.train(X, y)

        save_dir = self.model_path / "market_predictor"
        model.save(str(save_dir))
        self._save_metadata(save_dir, {
            "model_type": "MarketPredictorModel",
            "version": model.version,
            "feature_names": feature_cols,
            "cv_metrics": model.cv_metrics,
            "n_samples": len(df),
        })
        logger.info("MarketPredictorModel saved.")
        return model

    def train_anomaly_detection(self) -> AnomalyDetectionModel:
        """Train anomaly detection model on user profiles."""
        df = self._load_csv("user_profiles")

        feature_cols = [
            "total_entries", "win_rate", "avg_entry_fee",
            "total_wagered", "total_won", "net_profit",
        ]
        feature_cols = [c for c in feature_cols if c in df.columns]

        df = df[feature_cols].dropna()
        X = df[feature_cols]

        model = AnomalyDetectionModel()
        model.train(X)

        save_dir = self.model_path / "anomaly_detection"
        model.save(str(save_dir))
        self._save_metadata(save_dir, {
            "model_type": "AnomalyDetectionModel",
            "version": model.version,
            "feature_names": feature_cols,
            "n_samples": len(df),
        })
        logger.info("AnomalyDetectionModel saved.")
        return model

    def run_all(self) -> None:
        """Train and save all models."""
        logger.info("Starting full training pipeline...")
        self.train_player_performance()
        self.train_risk_scoring()
        self.train_market_predictor()
        self.train_anomaly_detection()
        logger.info("All models trained and saved successfully.")
