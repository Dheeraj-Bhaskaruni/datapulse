"""Simple feature store for storing and retrieving computed features."""

import pandas as pd
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class FeatureStore:
    """Simple file-based feature store."""

    def __init__(self, store_path: str = "data/features"):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.store_path / "metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        return {"feature_sets": {}}

    def _save_metadata(self) -> None:
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def save_features(self, name: str, df: pd.DataFrame, description: str = "") -> str:
        """Save a feature set to the store."""
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{version}.parquet"
        filepath = self.store_path / filename

        df.to_parquet(filepath, index=False)

        self.metadata["feature_sets"][name] = {
            "latest_version": version,
            "file": filename,
            "columns": list(df.columns),
            "shape": list(df.shape),
            "description": description,
            "created_at": datetime.now().isoformat(),
            "checksum": hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest(),
        }
        self._save_metadata()
        logger.info(f"Saved feature set '{name}' ({df.shape}) to {filepath}")
        return str(filepath)

    def load_features(self, name: str, version: Optional[str] = None) -> pd.DataFrame:
        """Load a feature set from the store."""
        if name not in self.metadata["feature_sets"]:
            raise KeyError(f"Feature set '{name}' not found. Available: {list(self.metadata['feature_sets'].keys())}")

        info = self.metadata["feature_sets"][name]
        filepath = self.store_path / info["file"]

        if not filepath.exists():
            raise FileNotFoundError(f"Feature file not found: {filepath}")

        df = pd.read_parquet(filepath)
        logger.info(f"Loaded feature set '{name}' ({df.shape})")
        return df

    def list_feature_sets(self) -> Dict[str, Any]:
        """List all available feature sets."""
        return {
            name: {k: v for k, v in info.items() if k != 'checksum'}
            for name, info in self.metadata["feature_sets"].items()
        }

    def delete_feature_set(self, name: str) -> None:
        """Delete a feature set."""
        if name in self.metadata["feature_sets"]:
            filepath = self.store_path / self.metadata["feature_sets"][name]["file"]
            if filepath.exists():
                filepath.unlink()
            del self.metadata["feature_sets"][name]
            self._save_metadata()
            logger.info(f"Deleted feature set '{name}'")
