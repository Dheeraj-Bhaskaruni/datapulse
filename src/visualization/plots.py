"""Reusable visualization functions using Plotly."""

import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class PlotFactory:
    """Factory for creating standardized visualizations."""

    THEME = {
        'primary': '#6366F1',
        'secondary': '#8B5CF6',
        'success': '#10B981',
        'danger': '#EF4444',
        'warning': '#F59E0B',
        'info': '#3B82F6',
        'bg': '#0F172A',
        'card_bg': '#1E293B',
        'text': '#E2E8F0',
        'grid': '#334155',
        'palette': ['#6366F1', '#8B5CF6', '#EC4899', '#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#06B6D4'],
    }

    @classmethod
    def _apply_theme(cls, fig: go.Figure) -> go.Figure:
        """Apply dark theme to figure."""
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=cls.THEME['bg'],
            plot_bgcolor=cls.THEME['card_bg'],
            font=dict(color=cls.THEME['text'], family='Inter, sans-serif'),
            xaxis=dict(gridcolor=cls.THEME['grid']),
            yaxis=dict(gridcolor=cls.THEME['grid']),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        return fig

    @classmethod
    def distribution_plot(cls, data: pd.Series, title: str = "Distribution", nbins: int = 30) -> go.Figure:
        """Create a distribution histogram with KDE overlay."""
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=data, nbinsx=nbins, name='Histogram',
            marker_color=cls.THEME['primary'], opacity=0.7,
        ))
        fig.update_layout(title=title, xaxis_title=data.name or 'Value', yaxis_title='Count')
        return cls._apply_theme(fig)

    @classmethod
    def correlation_heatmap(cls, df: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
        """Create a correlation heatmap."""
        corr = df.select_dtypes(include=[np.number]).corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu_r',
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate='%{text}',
        ))
        fig.update_layout(title=title, width=700, height=600)
        return cls._apply_theme(fig)

    @classmethod
    def time_series_plot(
        cls, df: pd.DataFrame, date_col: str, value_cols: List[str], title: str = "Time Series"
    ) -> go.Figure:
        """Create a time series line plot."""
        fig = go.Figure()
        for i, col in enumerate(value_cols):
            color = cls.THEME['palette'][i % len(cls.THEME['palette'])]
            fig.add_trace(go.Scatter(
                x=df[date_col], y=df[col], mode='lines', name=col,
                line=dict(color=color, width=2),
            ))
        fig.update_layout(
            title=title, xaxis_title='Date', yaxis_title='Value', hovermode='x unified'
        )
        return cls._apply_theme(fig)

    @classmethod
    def bar_chart(cls, data: pd.Series, title: str = "Bar Chart", orientation: str = 'v') -> go.Figure:
        """Create a bar chart."""
        if orientation == 'h':
            fig = go.Figure(go.Bar(
                x=data.values, y=data.index, orientation='h',
                marker_color=cls.THEME['primary'],
            ))
        else:
            fig = go.Figure(go.Bar(x=data.index, y=data.values, marker_color=cls.THEME['primary']))
        fig.update_layout(title=title)
        return cls._apply_theme(fig)

    @classmethod
    def scatter_plot(
        cls,
        df: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        size: Optional[str] = None,
        title: str = "Scatter Plot",
    ) -> go.Figure:
        """Create a scatter plot."""
        fig = px.scatter(
            df, x=x, y=y, color=color, size=size, title=title,
            color_discrete_sequence=cls.THEME['palette'],
        )
        return cls._apply_theme(fig)

    @classmethod
    def box_plot(cls, df: pd.DataFrame, x: str, y: str, title: str = "Box Plot") -> go.Figure:
        """Create a box plot."""
        fig = px.box(df, x=x, y=y, title=title, color=x,
                     color_discrete_sequence=cls.THEME['palette'])
        return cls._apply_theme(fig)

    @classmethod
    def confusion_matrix_plot(
        cls, cm: np.ndarray, labels: Optional[List[str]] = None, title: str = "Confusion Matrix"
    ) -> go.Figure:
        """Create a confusion matrix heatmap."""
        if labels is None:
            labels = [str(i) for i in range(len(cm))]
        fig = go.Figure(data=go.Heatmap(
            z=cm, x=labels, y=labels, colorscale='Blues',
            text=cm, texttemplate='%{text}',
        ))
        fig.update_layout(title=title, xaxis_title='Predicted', yaxis_title='Actual')
        return cls._apply_theme(fig)

    @classmethod
    def roc_curve_plot(cls, fpr: list, tpr: list, auc: float, title: str = "ROC Curve") -> go.Figure:
        """Create an ROC curve plot."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={auc:.4f})',
            line=dict(color=cls.THEME['primary'], width=2),
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines', name='Random',
            line=dict(color=cls.THEME['danger'], width=1, dash='dash'),
        ))
        fig.update_layout(
            title=title, xaxis_title='False Positive Rate', yaxis_title='True Positive Rate'
        )
        return cls._apply_theme(fig)

    @classmethod
    def kpi_card(
        cls, value: float, label: str, delta: Optional[float] = None,
        prefix: str = "", suffix: str = "",
    ) -> go.Figure:
        """Create a KPI indicator card."""
        fig = go.Figure(go.Indicator(
            mode="number+delta" if delta else "number",
            value=value,
            title={"text": label, "font": {"size": 16}},
            delta={"reference": value - delta, "relative": True} if delta else None,
            number={"prefix": prefix, "suffix": suffix, "font": {"size": 36}},
        ))
        fig.update_layout(height=150, margin=dict(l=20, r=20, t=50, b=20))
        return cls._apply_theme(fig)

    @classmethod
    def gauge_chart(cls, value: float, title: str = "Gauge", max_val: float = 100) -> go.Figure:
        """Create a gauge chart."""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': title},
            gauge={
                'axis': {'range': [0, max_val]},
                'bar': {'color': cls.THEME['primary']},
                'steps': [
                    {'range': [0, max_val * 0.33], 'color': '#064E3B'},
                    {'range': [max_val * 0.33, max_val * 0.66], 'color': '#78350F'},
                    {'range': [max_val * 0.66, max_val], 'color': '#7F1D1D'},
                ],
            },
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
        return cls._apply_theme(fig)
