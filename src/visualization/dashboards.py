"""Dashboard layout helpers and composite visualization builders."""

import pandas as pd
import logging
from typing import Dict
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .plots import PlotFactory

logger = logging.getLogger(__name__)


class DashboardBuilder:
    """Builds multi-panel dashboard figures."""

    @classmethod
    def overview_dashboard(cls, datasets: Dict[str, pd.DataFrame]) -> go.Figure:
        """Build a multi-panel overview dashboard."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Fantasy Points Distribution",
                "Win Rate Distribution",
                "Contest Types",
                "Salary vs Fantasy Points",
            ),
        )

        if 'players' in datasets:
            players = datasets['players']
            fig.add_trace(
                go.Histogram(x=players['fantasy_points'], nbinsx=30, marker_color='#6366F1', opacity=0.8),
                row=1, col=1,
            )

        if 'user_profiles' in datasets:
            users = datasets['user_profiles']
            fig.add_trace(
                go.Histogram(x=users['win_rate'], nbinsx=25, marker_color='#8B5CF6', opacity=0.8),
                row=1, col=2,
            )

        if 'contests' in datasets:
            contests = datasets['contests']
            type_counts = contests['contest_type'].value_counts()
            fig.add_trace(
                go.Bar(x=type_counts.index.tolist(), y=type_counts.values.tolist(), marker_color='#EC4899'),
                row=2, col=1,
            )

        if 'players' in datasets:
            players = datasets['players']
            fig.add_trace(
                go.Scatter(
                    x=players['salary'], y=players['fantasy_points'],
                    mode='markers', marker=dict(color='#10B981', opacity=0.5, size=5),
                ),
                row=2, col=2,
            )

        fig.update_layout(
            title="DataPulse Overview Dashboard",
            template='plotly_dark',
            paper_bgcolor='#0F172A',
            plot_bgcolor='#1E293B',
            font=dict(color='#E2E8F0'),
            showlegend=False,
            height=700,
        )
        return fig

    @classmethod
    def model_performance_dashboard(cls, metrics_dict: Dict[str, Dict]) -> go.Figure:
        """Build a model comparison dashboard."""
        model_names = list(metrics_dict.keys())
        metric_keys = ['accuracy', 'f1', 'auc_roc', 'val_r2']

        fig = make_subplots(
            rows=1, cols=len(model_names),
            subplot_titles=model_names,
        )

        for i, (model_name, metrics) in enumerate(metrics_dict.items(), 1):
            available = {k: v for k, v in metrics.items() if k in metric_keys and isinstance(v, float)}
            if available:
                fig.add_trace(
                    go.Bar(
                        x=list(available.keys()),
                        y=list(available.values()),
                        marker_color=PlotFactory.THEME['palette'][i - 1],
                        name=model_name,
                    ),
                    row=1, col=i,
                )

        fig.update_layout(
            title="Model Performance Comparison",
            template='plotly_dark',
            paper_bgcolor='#0F172A',
            showlegend=False,
            height=400,
        )
        return fig
