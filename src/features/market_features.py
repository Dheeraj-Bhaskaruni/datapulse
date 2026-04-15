"""Market and pricing feature engineering."""

import pandas as pd
import numpy as np
import logging


logger = logging.getLogger(__name__)


class MarketFeatureGenerator:
    """Generate market analytics features."""

    def generate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all market features."""
        df = df.copy()
        df = self.add_line_movement(df)
        df = self.add_implied_probability(df)
        df = self.add_expected_value(df)
        df = self.add_closing_line_value(df)
        df = self.add_market_efficiency(df)
        logger.info(f"Generated market features. Shape: {df.shape}")
        return df

    def add_line_movement(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate line movement between open and close."""
        if 'opening_line' in df.columns and 'closing_line' in df.columns:
            df['line_movement'] = df['closing_line'] - df['opening_line']
            df['line_movement_pct'] = df['line_movement'] / df['opening_line'].replace(0, np.nan).abs()
            df['line_moved_direction'] = np.sign(df['line_movement'])
            df['significant_move'] = (df['line_movement_pct'].abs() > 0.05).astype(int)
        return df

    def add_implied_probability(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert American odds to implied probability."""
        if 'closing_odds' in df.columns:
            odds = df['closing_odds']
            df['implied_probability'] = np.where(
                odds > 0,
                100 / (odds + 100),
                odds.abs() / (odds.abs() + 100),
            )
            df['implied_probability'] = df['implied_probability'].clip(0, 1)
        if 'opening_odds' in df.columns:
            odds = df['opening_odds']
            df['opening_implied_prob'] = np.where(
                odds > 0,
                100 / (odds + 100),
                odds.abs() / (odds.abs() + 100),
            )
        return df

    def add_expected_value(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate expected value of bets."""
        if 'implied_probability' in df.columns and 'closing_odds' in df.columns:
            payout_multiplier = np.where(
                df['closing_odds'] > 0,
                df['closing_odds'] / 100,
                100 / df['closing_odds'].abs(),
            )
            # EV = (prob * payout) - ((1 - prob) * stake)
            df['expected_value'] = (
                (df['implied_probability'] * payout_multiplier) - (1 - df['implied_probability'])
            )
            df['ev_positive'] = (df['expected_value'] > 0).astype(int)
        return df

    def add_closing_line_value(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate closing line value (CLV)."""
        if 'opening_implied_prob' in df.columns and 'implied_probability' in df.columns:
            df['clv'] = df['opening_implied_prob'] - df['implied_probability']
            df['clv_pct'] = df['clv'] / df['opening_implied_prob'].replace(0, np.nan)
            df['beat_closing_line'] = (df['clv'] > 0).astype(int)
        return df

    def add_market_efficiency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market efficiency indicators."""
        if 'result' in df.columns and 'implied_probability' in df.columns:
            df['prediction_correct'] = (
                ((df['implied_probability'] > 0.5) & (df['result'] == 1))
                | ((df['implied_probability'] <= 0.5) & (df['result'] == 0))
            ).astype(int)
            df['calibration_error'] = (df['implied_probability'] - df['result']).abs()
        return df
