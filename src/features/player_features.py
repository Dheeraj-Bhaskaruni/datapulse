"""Player performance feature engineering."""

import pandas as pd
import numpy as np
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


class PlayerFeatureGenerator:
    """Generate player performance features."""

    def __init__(self, window_sizes: Optional[List[int]] = None):
        self.window_sizes = window_sizes or [3, 5, 10]

    def generate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all player features."""
        df = df.copy()
        df = self.add_rolling_averages(df)
        df = self.add_consistency_metrics(df)
        df = self.add_trend_indicators(df)
        df = self.add_value_metrics(df)
        df = self.add_percentile_ranks(df)
        logger.info(f"Generated player features. Shape: {df.shape}")
        return df

    def add_rolling_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling average features for key stats."""
        stat_cols = [c for c in ['points_avg', 'assists_avg', 'rebounds_avg', 'fantasy_points'] if c in df.columns]
        for col in stat_cols:
            for window in self.window_sizes:
                df[f'{col}_rolling_{window}'] = df.groupby('player_id')[col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
        return df

    def add_consistency_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add consistency score based on coefficient of variation."""
        if 'fantasy_points' in df.columns:
            grouped = df.groupby('player_id')['fantasy_points']
            df['fp_std'] = grouped.transform('std').fillna(0)
            df['fp_mean'] = grouped.transform('mean')
            df['consistency_score'] = 1 - (df['fp_std'] / df['fp_mean'].replace(0, np.nan)).fillna(0)
            df['consistency_score'] = df['consistency_score'].clip(0, 1)
            df['floor'] = grouped.transform(lambda x: x.quantile(0.1))
            df['ceiling'] = grouped.transform(lambda x: x.quantile(0.9))
            df['upside_ratio'] = (df['ceiling'] - df['fp_mean']) / df['fp_mean'].replace(0, np.nan)
        return df

    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend direction indicators."""
        if 'fantasy_points' in df.columns:
            for window in [3, 5]:
                rolling = df.groupby('player_id')['fantasy_points'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                overall = df.groupby('player_id')['fantasy_points'].transform('mean')
                df[f'trend_{window}g'] = np.where(
                    rolling > overall * 1.05, 1,
                    np.where(rolling < overall * 0.95, -1, 0)
                )
        return df

    def add_value_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add salary value metrics."""
        if 'fantasy_points' in df.columns and 'salary' in df.columns:
            df['points_per_dollar'] = df['fantasy_points'] / df['salary'].replace(0, np.nan)
            df['value_score'] = (
                (df['points_per_dollar'] - df['points_per_dollar'].mean()) / df['points_per_dollar'].std()
            )
            salary_median = df['salary'].median()
            df['salary_tier'] = pd.cut(
                df['salary'],
                bins=[0, salary_median * 0.6, salary_median, salary_median * 1.4, float('inf')],
                labels=['budget', 'mid', 'premium', 'elite'],
            )
        return df

    def add_percentile_ranks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add position-based percentile rankings."""
        if 'position' in df.columns and 'fantasy_points' in df.columns:
            df['position_rank'] = df.groupby('position')['fantasy_points'].rank(pct=True)
            df['overall_rank'] = df['fantasy_points'].rank(pct=True)
        return df
