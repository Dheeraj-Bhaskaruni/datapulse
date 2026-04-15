"""User behavior feature engineering."""

import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class UserFeatureGenerator:
    """Generate user behavior and risk features."""

    def generate_all(self, profiles_df: pd.DataFrame, entries_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate all user features."""
        df = profiles_df.copy()
        df = self.add_engagement_features(df)
        df = self.add_financial_features(df)
        df = self.add_risk_indicators(df)
        if entries_df is not None:
            df = self.add_behavioral_patterns(df, entries_df)
        logger.info(f"Generated user features. Shape: {df.shape}")
        return df

    def add_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add user engagement metrics."""
        if 'join_date' in df.columns:
            df['join_date'] = pd.to_datetime(df['join_date'])
            df['account_age_days'] = (pd.Timestamp.now() - df['join_date']).dt.days
            if 'total_entries' in df.columns:
                df['entries_per_day'] = df['total_entries'] / df['account_age_days'].replace(0, 1)
            if 'total_contests' in df.columns:
                df['contests_per_day'] = df['total_contests'] / df['account_age_days'].replace(0, 1)
        if 'last_active' in df.columns:
            df['last_active'] = pd.to_datetime(df['last_active'])
            df['days_since_active'] = (pd.Timestamp.now() - df['last_active']).dt.days
            df['is_dormant'] = (df['days_since_active'] > 30).astype(int)
        return df

    def add_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add financial behavior features."""
        if 'total_wagered' in df.columns and 'total_won' in df.columns:
            df['roi'] = (df['total_won'] - df['total_wagered']) / df['total_wagered'].replace(0, np.nan)
            if 'net_profit' in df.columns:
                df['profit_margin'] = df['net_profit'] / df['total_wagered'].replace(0, np.nan)
            else:
                df['profit_margin'] = df['roi']
        if 'avg_entry_fee' in df.columns and 'total_wagered' in df.columns:
            df['fee_to_wagered_ratio'] = df['avg_entry_fee'] / (
                df['total_wagered'] / df['total_entries'].replace(0, 1)
            )
        if 'win_rate' in df.columns:
            df['win_rate_tier'] = pd.cut(
                df['win_rate'],
                bins=[0, 0.3, 0.45, 0.55, 0.7, 1.0],
                labels=['low', 'below_avg', 'average', 'above_avg', 'high'],
            )
        return df

    def add_risk_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add risk scoring indicators."""
        risk_score = pd.Series(0.0, index=df.index)

        if 'win_rate' in df.columns:
            risk_score += np.where(df['win_rate'] > 0.65, 2, np.where(df['win_rate'] > 0.55, 1, 0))
        if 'roi' in df.columns:
            risk_score += np.where(df['roi'] > 0.15, 2, np.where(df['roi'] > 0.05, 1, 0))
        if 'total_wagered' in df.columns:
            high_volume = df['total_wagered'].quantile(0.9)
            risk_score += np.where(df['total_wagered'] > high_volume, 2, 0)
        if 'entries_per_day' in df.columns:
            risk_score += np.where(df['entries_per_day'] > 10, 1, 0)

        df['computed_risk_score'] = risk_score
        max_risk = risk_score.max() if risk_score.max() > 0 else 1
        df['risk_level'] = pd.cut(
            risk_score,
            bins=[-1, max_risk * 0.33, max_risk * 0.66, float('inf')],
            labels=['low', 'medium', 'high'],
        )
        return df

    def add_behavioral_patterns(self, profiles_df: pd.DataFrame, entries_df: pd.DataFrame) -> pd.DataFrame:
        """Add behavioral pattern features from entry data."""
        if 'user_id' in entries_df.columns:
            entry_stats = entries_df.groupby('user_id').agg(
                avg_score=('total_score', 'mean'),
                score_std=('total_score', 'std'),
                max_payout=('payout', 'max'),
                avg_payout=('payout', 'mean'),
                entry_count=('entry_id', 'count'),
            ).reset_index()
            entry_stats['score_consistency'] = 1 - (
                entry_stats['score_std'] / entry_stats['avg_score'].replace(0, np.nan)
            ).fillna(0)
            profiles_df = profiles_df.merge(entry_stats, on='user_id', how='left')
        return profiles_df
