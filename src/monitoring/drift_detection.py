"""Data and model drift detection module."""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List
from scipy import stats

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detect data drift and model performance degradation."""

    @staticmethod
    def psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
        """Population Stability Index (PSI) for drift detection."""
        ref = np.array(reference, dtype=float)
        cur = np.array(current, dtype=float)

        breakpoints = np.percentile(ref, np.linspace(0, 100, n_bins + 1))
        breakpoints = np.unique(breakpoints)

        ref_counts = np.histogram(ref, bins=breakpoints)[0] / len(ref)
        cur_counts = np.histogram(cur, bins=breakpoints)[0] / len(cur)

        ref_counts = np.clip(ref_counts, 0.001, None)
        cur_counts = np.clip(cur_counts, 0.001, None)

        psi_value = float(np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts)))

        return {
            'psi': psi_value,
            'interpretation': (
                'No drift' if psi_value < 0.1 else
                'Moderate drift' if psi_value < 0.25 else
                'Significant drift'
            ),
            'drifted': psi_value >= 0.1,
        }

    @staticmethod
    def ks_test(reference: np.ndarray, current: np.ndarray) -> Dict[str, Any]:
        """Kolmogorov-Smirnov test for distribution comparison."""
        stat, p_value = stats.ks_2samp(reference, current)
        return {
            'ks_statistic': float(stat),
            'p_value': float(p_value),
            'drifted': p_value < 0.05,
            'interpretation': (
                'Distributions differ significantly' if p_value < 0.05 else 'Distributions are similar'
            ),
        }

    def check_data_drift(
        self, reference_df: pd.DataFrame, current_df: pd.DataFrame, columns: list = None
    ) -> Dict[str, Any]:
        """Check drift across multiple columns."""
        if columns is None:
            columns = reference_df.select_dtypes(include=[np.number]).columns.tolist()

        drift_results = {}
        drifted_columns = []

        for col in columns:
            if col in reference_df.columns and col in current_df.columns:
                ref_data = reference_df[col].dropna().values
                cur_data = current_df[col].dropna().values
                if len(ref_data) > 0 and len(cur_data) > 0:
                    psi_result = self.psi(ref_data, cur_data)
                    ks_result = self.ks_test(ref_data, cur_data)
                    drift_results[col] = {'psi': psi_result, 'ks_test': ks_result}
                    if psi_result['drifted'] or ks_result['drifted']:
                        drifted_columns.append(col)

        return {
            'columns_checked': len(columns),
            'columns_drifted': len(drifted_columns),
            'drifted_columns': drifted_columns,
            'overall_drift': len(drifted_columns) > 0,
            'details': drift_results,
        }

    @staticmethod
    def performance_drift(
        historical_metrics: List[float], current_metric: float, window: int = 10
    ) -> Dict[str, Any]:
        """Detect model performance degradation."""
        recent = historical_metrics[-window:]
        mean_perf = np.mean(recent)
        std_perf = np.std(recent)

        z_score = (current_metric - mean_perf) / std_perf if std_perf > 0 else 0
        degraded = z_score < -2

        return {
            'current_metric': float(current_metric),
            'historical_mean': float(mean_perf),
            'historical_std': float(std_perf),
            'z_score': float(z_score),
            'degraded': degraded,
            'interpretation': (
                'Performance degraded significantly' if degraded else 'Performance within normal range'
            ),
        }
