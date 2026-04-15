"""Statistical testing module."""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from scipy import stats

logger = logging.getLogger(__name__)


class StatisticalTester:
    """Statistical testing toolkit."""

    @staticmethod
    def t_test(group_a: np.ndarray, group_b: np.ndarray, paired: bool = False) -> Dict[str, Any]:
        """Perform t-test between two groups."""
        group_a = np.array(group_a, dtype=float)
        group_b = np.array(group_b, dtype=float)

        if paired:
            stat, p_value = stats.ttest_rel(group_a, group_b)
        else:
            stat, p_value = stats.ttest_ind(group_a, group_b)

        effect_size = (group_a.mean() - group_b.mean()) / np.sqrt(
            (group_a.std()**2 + group_b.std()**2) / 2
        )

        return {
            'test': 'paired_t_test' if paired else 'independent_t_test',
            't_statistic': float(stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'effect_size_cohens_d': float(effect_size),
            'group_a_mean': float(group_a.mean()),
            'group_b_mean': float(group_b.mean()),
            'mean_difference': float(group_a.mean() - group_b.mean()),
        }

    @staticmethod
    def chi_square_test(observed: pd.DataFrame) -> Dict[str, Any]:
        """Perform chi-square test of independence."""
        chi2, p_value, dof, expected = stats.chi2_contingency(observed)
        return {
            'test': 'chi_square',
            'chi2_statistic': float(chi2),
            'p_value': float(p_value),
            'degrees_of_freedom': int(dof),
            'significant': p_value < 0.05,
            'cramers_v': float(np.sqrt(chi2 / (observed.values.sum() * (min(observed.shape) - 1)))),
        }

    @staticmethod
    def anova(groups: list) -> Dict[str, Any]:
        """Perform one-way ANOVA."""
        stat, p_value = stats.f_oneway(*groups)
        grand_mean = np.concatenate(groups).mean()
        n_total = sum(len(g) for g in groups)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        ss_total = sum(np.sum((g - grand_mean)**2) for g in groups)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        return {
            'test': 'one_way_anova',
            'f_statistic': float(stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'eta_squared': float(eta_squared),
            'n_groups': len(groups),
        }

    @staticmethod
    def normality_test(data: np.ndarray) -> Dict[str, Any]:
        """Test for normality using Shapiro-Wilk test."""
        data = np.array(data, dtype=float)
        sample = data[:5000] if len(data) > 5000 else data
        stat, p_value = stats.shapiro(sample)
        return {
            'test': 'shapiro_wilk',
            'statistic': float(stat),
            'p_value': float(p_value),
            'is_normal': p_value > 0.05,
            'n_samples': len(sample),
        }

    @staticmethod
    def bootstrap_ci(
        data: np.ndarray,
        statistic: str = 'mean',
        n_bootstrap: int = 10000,
        ci: float = 0.95,
    ) -> Dict[str, float]:
        """Calculate bootstrap confidence interval."""
        data = np.array(data, dtype=float)
        stat_fn = {'mean': np.mean, 'median': np.median, 'std': np.std}.get(statistic, np.mean)

        bootstrap_stats = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(stat_fn(sample))

        bootstrap_stats = np.array(bootstrap_stats)
        alpha = (1 - ci) / 2

        return {
            'statistic': statistic,
            'point_estimate': float(stat_fn(data)),
            'ci_lower': float(np.percentile(bootstrap_stats, alpha * 100)),
            'ci_upper': float(np.percentile(bootstrap_stats, (1 - alpha) * 100)),
            'ci_level': ci,
            'n_bootstrap': n_bootstrap,
            'std_error': float(bootstrap_stats.std()),
        }

    @staticmethod
    def ab_test(control: np.ndarray, treatment: np.ndarray, metric: str = 'conversion') -> Dict[str, Any]:
        """Perform A/B test analysis."""
        control = np.array(control, dtype=float)
        treatment = np.array(treatment, dtype=float)

        result = StatisticalTester.t_test(treatment, control)
        lift = (treatment.mean() - control.mean()) / control.mean() if control.mean() != 0 else 0

        result.update({
            'test': 'ab_test',
            'metric': metric,
            'control_mean': float(control.mean()),
            'treatment_mean': float(treatment.mean()),
            'lift': float(lift),
            'lift_pct': float(lift * 100),
            'recommendation': (
                'Deploy treatment' if (result['significant'] and lift > 0) else
                'Keep control' if result['significant'] else
                'Gather more data'
            ),
        })
        return result
