"""Time series analysis module."""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from scipy import stats

logger = logging.getLogger(__name__)


class TimeSeriesAnalyzer:
    """Time series analysis toolkit."""

    def __init__(self, df: pd.DataFrame, date_col: str, value_col: str):
        self.df = df.sort_values(date_col).copy()
        self.date_col = date_col
        self.value_col = value_col
        self.df[date_col] = pd.to_datetime(self.df[date_col])

    def decompose(self, period: Optional[int] = None) -> Dict[str, pd.Series]:
        """Decompose time series into trend, seasonal, and residual."""
        values = self.df[self.value_col].values
        n = len(values)

        if period is None:
            period = min(7, n // 3) if n > 21 else 3

        # Trend using moving average
        trend = pd.Series(values).rolling(window=period, center=True).mean()

        # Seasonal: average of detrended values by position in period
        detrended = values - trend.values
        seasonal = np.zeros(n)
        for i in range(period):
            indices = range(i, n, period)
            vals = [detrended[j] for j in indices if not np.isnan(detrended[j])]
            if vals:
                seasonal_val = np.mean(vals)
                for j in indices:
                    seasonal[j] = seasonal_val

        residual = values - trend.values - seasonal

        return {
            'trend': pd.Series(trend.values, index=self.df.index),
            'seasonal': pd.Series(seasonal, index=self.df.index),
            'residual': pd.Series(residual, index=self.df.index),
            'original': pd.Series(values, index=self.df.index),
        }

    def stationarity_test(self) -> Dict[str, Any]:
        """Test for stationarity by comparing halves of the series."""
        values = self.df[self.value_col].dropna().values
        n = len(values)
        half = n // 2
        first_half = values[:half]
        second_half = values[half:]

        t_stat, p_value = stats.ttest_ind(first_half, second_half)
        var_ratio = first_half.var() / second_half.var() if second_half.var() > 0 else float('inf')

        return {
            'is_stationary': p_value > 0.05 and 0.5 < var_ratio < 2.0,
            'mean_first_half': float(first_half.mean()),
            'mean_second_half': float(second_half.mean()),
            'variance_ratio': float(var_ratio),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
        }

    def autocorrelation(self, max_lags: int = 20) -> Dict[str, list]:
        """Calculate autocorrelation function."""
        values = self.df[self.value_col].dropna().values
        n = len(values)
        max_lags = min(max_lags, n // 2)
        mean = values.mean()

        acf_values = []
        for lag in range(max_lags + 1):
            if lag == 0:
                acf_values.append(1.0)
            else:
                covariance = np.mean((values[lag:] - mean) * (values[:-lag] - mean))
                variance = np.var(values)
                acf_values.append(float(covariance / variance) if variance > 0 else 0)

        confidence = 1.96 / np.sqrt(n)
        return {
            'lags': list(range(max_lags + 1)),
            'acf': acf_values,
            'confidence_upper': float(confidence),
            'confidence_lower': float(-confidence),
        }

    def forecast_simple(self, periods: int = 10, method: str = 'exponential') -> pd.DataFrame:
        """Simple forecasting using exponential smoothing."""
        values = self.df[self.value_col].dropna().values

        if method == 'exponential':
            alpha = 0.3
            forecast = [values[-1]]
            level = values[-1]
            for _ in range(periods):
                level = alpha * forecast[-1] + (1 - alpha) * level
                forecast.append(level)
            forecast = forecast[1:]
        else:
            # Moving average
            window = min(5, len(values))
            ma = np.mean(values[-window:])
            forecast = [ma] * periods

        last_date = self.df[self.date_col].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)

        std = np.std(values[-min(20, len(values)):])
        return pd.DataFrame({
            'date': future_dates,
            'forecast': forecast,
            'lower_bound': [f - 1.96 * std for f in forecast],
            'upper_bound': [f + 1.96 * std for f in forecast],
        })

    def summary(self) -> Dict[str, Any]:
        """Get time series summary."""
        values = self.df[self.value_col].dropna()
        return {
            'n_observations': len(values),
            'date_range': {
                'start': str(self.df[self.date_col].min()),
                'end': str(self.df[self.date_col].max()),
            },
            'mean': float(values.mean()),
            'trend_direction': 'upward' if values.iloc[-1] > values.iloc[0] else 'downward',
            'volatility': float(values.std() / values.mean()) if values.mean() != 0 else 0,
            'min': float(values.min()),
            'max': float(values.max()),
        }
