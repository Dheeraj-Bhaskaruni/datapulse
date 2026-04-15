"""Data ingestion module for loading data from various sources."""

import pandas as pd
import logging
import time
from pathlib import Path
from typing import Optional, Dict, List
from functools import wraps

logger = logging.getLogger(__name__)


def retry(max_retries: int = 3, delay: float = 1.0):
    """Retry decorator with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    wait_time = delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
            raise last_exception
        return wrapper
    return decorator


class DataLoader:
    """Universal data loader supporting multiple formats and sources."""

    SUPPORTED_FORMATS = {'.csv', '.json', '.parquet', '.xlsx', '.tsv'}

    def __init__(self, base_path: Optional[str] = None):
        self.base_path = Path(base_path) if base_path else Path("data")
        logger.info(f"DataLoader initialized with base_path: {self.base_path}")

    def load(self, source: str, format: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Load data from a file path or URL."""
        path = Path(source) if not source.startswith("http") else None

        if path:
            if not path.is_absolute() and not path.exists():
                # Try prepending base_path only if the path doesn't exist as-is
                candidate = self.base_path / path
                if candidate.exists():
                    path = candidate
            return self._load_file(path, format, **kwargs)
        else:
            return self._load_url(source, format, **kwargs)

    def _load_file(self, path: Path, format: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Load data from a local file."""
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        ext = format or path.suffix.lower()
        logger.info(f"Loading {ext} file: {path}")

        loaders = {
            '.csv': lambda: pd.read_csv(path, **kwargs),
            '.tsv': lambda: pd.read_csv(path, sep='\t', **kwargs),
            '.json': lambda: pd.read_json(path, **kwargs),
            '.parquet': lambda: pd.read_parquet(path, **kwargs),
            '.xlsx': lambda: pd.read_excel(path, **kwargs),
        }

        loader = loaders.get(ext)
        if not loader:
            raise ValueError(f"Unsupported format: {ext}. Supported: {self.SUPPORTED_FORMATS}")

        df = loader()
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns from {path.name}")
        return df

    @retry(max_retries=3, delay=1.0)
    def _load_url(self, url: str, format: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Load data from a URL with retry logic."""
        logger.info(f"Loading data from URL: {url}")
        if format == '.json' or url.endswith('.json'):
            return pd.read_json(url, **kwargs)
        return pd.read_csv(url, **kwargs)

    def load_multiple(self, sources: List[str], **kwargs) -> Dict[str, pd.DataFrame]:
        """Load multiple data sources into a dictionary."""
        result = {}
        for source in sources:
            name = Path(source).stem
            result[name] = self.load(source, **kwargs)
        return result

    def get_sample_data(self, dataset: str) -> pd.DataFrame:
        """Load a sample dataset by name."""
        sample_path = self.base_path / "sample" / f"{dataset}.csv"
        return self.load(str(sample_path))

    def list_available(self, directory: str = "sample") -> List[str]:
        """List available datasets in a directory."""
        target = self.base_path / directory
        if not target.exists():
            return []
        return [f.stem for f in target.glob("*.*") if f.suffix in self.SUPPORTED_FORMATS]
