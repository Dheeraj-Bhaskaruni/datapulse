"""Data preprocessing and cleaning module."""

import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Tuple, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


class DataCleaner:
    """Handles data cleaning operations."""

    def __init__(self):
        self.cleaning_report: Dict[str, Any] = {}

    def clean(self, df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
        """Run full cleaning pipeline."""
        df = df.copy()
        initial_shape = df.shape

        df = self.remove_duplicates(df)
        df = self.handle_missing_values(df, config)
        df = self.fix_dtypes(df)
        df = self.remove_outliers(df)

        self.cleaning_report['initial_shape'] = initial_shape
        self.cleaning_report['final_shape'] = df.shape
        self.cleaning_report['rows_removed'] = initial_shape[0] - df.shape[0]

        logger.info(f"Cleaning complete: {initial_shape} -> {df.shape}")
        return df

    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """Remove duplicate rows."""
        n_before = len(df)
        df = df.drop_duplicates(subset=subset)
        n_removed = n_before - len(df)
        if n_removed > 0:
            logger.info(f"Removed {n_removed} duplicate rows")
        return df

    def handle_missing_values(self, df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
        """Handle missing values with configurable strategies."""
        missing_pct = df.isnull().sum() / len(df) * 100
        high_missing = missing_pct[missing_pct > 50].index.tolist()

        if high_missing:
            logger.warning(f"Columns with >50% missing: {high_missing}")
            df = df.drop(columns=high_missing)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        if len(numeric_cols) > 0 and df[numeric_cols].isnull().any().any():
            strategy = (config or {}).get('numeric_impute', 'median')
            imputer = SimpleImputer(strategy=strategy)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        if len(categorical_cols) > 0 and df[categorical_cols].isnull().any().any():
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'unknown')

        return df

    def fix_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Auto-detect and fix data types."""
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                    logger.info(f"Converted {col} to datetime")
                    continue
                except (ValueError, TypeError):
                    pass
                try:
                    df[col] = pd.to_numeric(df[col])
                    logger.info(f"Converted {col} to numeric")
                except (ValueError, TypeError):
                    pass
        return df

    def remove_outliers(self, df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """Remove outliers using IQR or z-score method."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        mask = pd.Series(True, index=df.index)

        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                mask &= (df[col] >= Q1 - threshold * IQR) & (df[col] <= Q3 + threshold * IQR)
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                mask &= z_scores < threshold

        n_removed = (~mask).sum()
        if n_removed > 0:
            logger.info(f"Removed {n_removed} outlier rows using {method} method")
        return df[mask].reset_index(drop=True)


class FeatureEngineer:
    """Handles feature engineering operations."""

    def __init__(self):
        self.scalers: Dict[str, Any] = {}
        self.encoders: Dict[str, LabelEncoder] = {}

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run feature engineering pipeline."""
        df = df.copy()
        df = self._add_datetime_features(df)
        df = self._add_interaction_features(df)
        df = self._add_statistical_features(df)
        return df

    def _add_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from datetime columns."""
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        for col in datetime_cols:
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            df[f'{col}_hour'] = df[col].dt.hour
            df[f'{col}_is_weekend'] = df[col].dt.dayofweek.isin([5, 6]).astype(int)
            logger.info(f"Added datetime features for {col}")
        return df

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            for i in range(min(len(numeric_cols), 5)):
                for j in range(i + 1, min(len(numeric_cols), 5)):
                    col_a, col_b = numeric_cols[i], numeric_cols[j]
                    ratio_name = f'{col_a}_to_{col_b}_ratio'
                    denominator = df[col_b].replace(0, np.nan)
                    df[ratio_name] = df[col_a] / denominator
        return df

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling and aggregate statistical features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:5]:
            df[f'{col}_zscore'] = (df[col] - df[col].mean()) / df[col].std()
            df[f'{col}_pct_rank'] = df[col].rank(pct=True)
        return df

    def scale_features(self, df: pd.DataFrame, columns: List[str], method: str = 'standard') -> pd.DataFrame:
        """Scale numeric features."""
        df = df.copy()
        scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
        df[columns] = scaler.fit_transform(df[columns])
        self.scalers[method] = scaler
        return df

    def encode_categorical(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Encode categorical variables."""
        df = df.copy()
        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in columns:
            if df[col].nunique() <= 10:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            else:
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
                self.encoders[col] = encoder
        return df
