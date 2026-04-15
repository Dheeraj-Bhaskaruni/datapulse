"""Automated Exploratory Data Analysis module."""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class AutoEDA:
    """Automated exploratory data analysis."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.report: Dict[str, Any] = {}

    def run_full_analysis(self) -> Dict[str, Any]:
        """Run complete EDA pipeline."""
        self.report = {
            'overview': self.get_overview(),
            'numeric_summary': self.get_numeric_summary(),
            'categorical_summary': self.get_categorical_summary(),
            'missing_values': self.get_missing_analysis(),
            'correlations': self.get_correlations(),
            'distributions': self.get_distribution_stats(),
            'outliers': self.detect_outliers(),
        }
        logger.info("Full EDA analysis complete")
        return self.report

    def get_overview(self) -> Dict[str, Any]:
        """Get dataset overview."""
        return {
            'shape': {'rows': self.df.shape[0], 'columns': self.df.shape[1]},
            'dtypes': self.df.dtypes.astype(str).to_dict(),
            'memory_usage_mb': float(self.df.memory_usage(deep=True).sum() / 1024**2),
            'numeric_columns': self.df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': self.df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime_columns': self.df.select_dtypes(include=['datetime64']).columns.tolist(),
            'duplicated_rows': int(self.df.duplicated().sum()),
        }

    def get_numeric_summary(self) -> Dict[str, Any]:
        """Get summary statistics for numeric columns."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return {}

        stats = numeric_df.describe().T
        stats['skewness'] = numeric_df.skew()
        stats['kurtosis'] = numeric_df.kurtosis()
        stats['cv'] = stats['std'] / stats['mean'].replace(0, np.nan)
        return stats.to_dict()

    def get_categorical_summary(self) -> Dict[str, Any]:
        """Get summary for categorical columns."""
        cat_df = self.df.select_dtypes(include=['object', 'category'])
        if cat_df.empty:
            return {}

        summary = {}
        for col in cat_df.columns:
            summary[col] = {
                'n_unique': int(cat_df[col].nunique()),
                'top_values': cat_df[col].value_counts().head(10).to_dict(),
                'null_count': int(cat_df[col].isnull().sum()),
            }
        return summary

    def get_missing_analysis(self) -> Dict[str, Any]:
        """Analyze missing values."""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)

        return {
            'total_missing': int(missing.sum()),
            'columns_with_missing': {
                col: {'count': int(missing[col]), 'percentage': float(missing_pct[col])}
                for col in missing[missing > 0].index
            },
            'complete_rows': int((~self.df.isnull().any(axis=1)).sum()),
            'complete_row_pct': float((~self.df.isnull().any(axis=1)).mean() * 100),
        }

    def get_correlations(self) -> Dict[str, Any]:
        """Get correlation analysis."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return {}

        corr_matrix = numeric_df.corr()
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                val = corr_matrix.iloc[i, j]
                if abs(val) > 0.7:
                    high_corr.append({
                        'col_1': corr_matrix.columns[i],
                        'col_2': corr_matrix.columns[j],
                        'correlation': round(float(val), 4),
                    })

        return {
            'correlation_matrix': corr_matrix.round(4).to_dict(),
            'high_correlations': sorted(high_corr, key=lambda x: abs(x['correlation']), reverse=True),
        }

    def get_distribution_stats(self) -> Dict[str, Any]:
        """Get distribution statistics for numeric columns."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        stats = {}
        for col in numeric_df.columns:
            data = numeric_df[col].dropna()
            stats[col] = {
                'mean': float(data.mean()),
                'median': float(data.median()),
                'mode': float(data.mode().iloc[0]) if not data.mode().empty else None,
                'std': float(data.std()),
                'skewness': float(data.skew()),
                'kurtosis': float(data.kurtosis()),
                'q25': float(data.quantile(0.25)),
                'q75': float(data.quantile(0.75)),
                'iqr': float(data.quantile(0.75) - data.quantile(0.25)),
            }
        return stats

    def detect_outliers(self, method: str = 'iqr') -> Dict[str, Any]:
        """Detect outliers in numeric columns."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        outlier_info = {}
        for col in numeric_df.columns:
            data = numeric_df[col].dropna()
            Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            n_outliers = ((data < lower) | (data > upper)).sum()
            if n_outliers > 0:
                outlier_info[col] = {
                    'n_outliers': int(n_outliers),
                    'pct_outliers': float(n_outliers / len(data) * 100),
                    'lower_bound': float(lower),
                    'upper_bound': float(upper),
                }
        return outlier_info
