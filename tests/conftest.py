"""Shared test fixtures."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_players():
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        'player_id': range(1, n + 1),
        'name': [f'Player_{i}' for i in range(1, n + 1)],
        'team': np.random.choice(['TeamA', 'TeamB', 'TeamC', 'TeamD'], n),
        'position': np.random.choice(['PG', 'SG', 'SF', 'PF', 'C'], n),
        'games_played': np.random.randint(20, 82, n),
        'points_avg': np.random.uniform(5, 35, n).round(1),
        'assists_avg': np.random.uniform(1, 12, n).round(1),
        'rebounds_avg': np.random.uniform(2, 14, n).round(1),
        'fantasy_points': np.random.uniform(10, 60, n).round(1),
        'salary': np.random.randint(3500, 11000, n),
        'consistency_score': np.random.uniform(0.3, 0.95, n).round(3),
    })


@pytest.fixture
def sample_users():
    np.random.seed(42)
    n = 30
    return pd.DataFrame({
        'user_id': range(1, n + 1),
        'username': [f'user_{i}' for i in range(1, n + 1)],
        'total_entries': np.random.randint(10, 5000, n),
        'win_rate': np.random.uniform(0.2, 0.8, n).round(3),
        'avg_entry_fee': np.random.uniform(1, 100, n).round(2),
        'total_wagered': np.random.uniform(100, 100000, n).round(2),
        'total_won': np.random.uniform(50, 120000, n).round(2),
        'net_profit': np.random.uniform(-20000, 30000, n).round(2),
        'risk_score': np.random.uniform(0, 100, n).round(1),
        'join_date': pd.date_range('2022-01-01', periods=n, freq='10D'),
        'last_active': pd.date_range('2024-01-01', periods=n, freq='D'),
    })


@pytest.fixture
def sample_market():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'event_id': range(1, n + 1),
        'market_type': np.random.choice(['spread', 'moneyline', 'over_under', 'player_prop'], n),
        'opening_line': np.random.uniform(-10, 10, n).round(1),
        'closing_line': np.random.uniform(-10, 10, n).round(1),
        'opening_odds': np.random.choice([-110, -105, 100, 110, 150, -150, -200, 200], n),
        'closing_odds': np.random.choice([-110, -105, 100, 110, 150, -150, -200, 200], n),
        'result': np.random.randint(0, 2, n),
    })


@pytest.fixture
def data_path():
    return str(Path(__file__).resolve().parent.parent / "data")
