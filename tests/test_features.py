"""Tests for feature engineering."""

import pytest
import pandas as pd
import numpy as np
from src.features.player_features import PlayerFeatureGenerator
from src.features.market_features import MarketFeatureGenerator
from src.features.user_features import UserFeatureGenerator
from src.features.feature_store import FeatureStore


class TestPlayerFeatures:
    def test_generate_all_expands_columns(self, sample_players):
        gen = PlayerFeatureGenerator()
        result = gen.generate_all(sample_players)
        assert len(result.columns) > len(sample_players.columns)

    def test_consistency_score_bounded(self, sample_players):
        gen = PlayerFeatureGenerator()
        result = gen.add_consistency_metrics(sample_players)
        if 'consistency_score' in result.columns:
            valid = result['consistency_score'].dropna()
            assert valid.between(0, 1).all()

    def test_value_metrics_created(self, sample_players):
        gen = PlayerFeatureGenerator()
        result = gen.add_value_metrics(sample_players)
        assert 'points_per_dollar' in result.columns
        assert 'value_score' in result.columns

    def test_percentile_ranks(self, sample_players):
        gen = PlayerFeatureGenerator()
        result = gen.add_percentile_ranks(sample_players)
        assert 'position_rank' in result.columns
        assert 'overall_rank' in result.columns

    def test_trend_indicators(self, sample_players):
        gen = PlayerFeatureGenerator()
        result = gen.add_trend_indicators(sample_players)
        if 'trend_3g' in result.columns:
            assert result['trend_3g'].isin([-1, 0, 1]).all()


class TestMarketFeatures:
    def test_generate_all(self, sample_market):
        gen = MarketFeatureGenerator()
        result = gen.generate_all(sample_market)
        assert 'line_movement' in result.columns
        assert 'implied_probability' in result.columns

    def test_implied_probability_bounded(self, sample_market):
        gen = MarketFeatureGenerator()
        result = gen.add_implied_probability(sample_market)
        if 'implied_probability' in result.columns:
            valid = result['implied_probability'].dropna()
            assert valid.between(0, 1).all()

    def test_expected_value_computed(self, sample_market):
        gen = MarketFeatureGenerator()
        result = gen.generate_all(sample_market)
        if 'expected_value' in result.columns:
            assert result['expected_value'].notna().any()

    def test_line_movement_sign(self, sample_market):
        gen = MarketFeatureGenerator()
        result = gen.add_line_movement(sample_market)
        if 'line_moved_direction' in result.columns:
            assert result['line_moved_direction'].isin([-1.0, 0.0, 1.0]).all()


class TestUserFeatures:
    def test_generate_all_creates_risk_level(self, sample_users):
        gen = UserFeatureGenerator()
        result = gen.generate_all(sample_users)
        assert 'risk_level' in result.columns

    def test_computed_risk_score_non_negative(self, sample_users):
        gen = UserFeatureGenerator()
        result = gen.add_risk_indicators(sample_users)
        assert 'computed_risk_score' in result.columns
        assert (result['computed_risk_score'] >= 0).all()

    def test_engagement_features(self, sample_users):
        gen = UserFeatureGenerator()
        result = gen.add_engagement_features(sample_users)
        assert 'account_age_days' in result.columns
        assert 'days_since_active' in result.columns

    def test_financial_features_roi(self, sample_users):
        gen = UserFeatureGenerator()
        result = gen.add_financial_features(sample_users)
        assert 'roi' in result.columns


class TestFeatureStore:
    def test_save_and_load_roundtrip(self, tmp_path, sample_players):
        store = FeatureStore(str(tmp_path))
        store.save_features("test_features", sample_players, "Unit test data")
        loaded = store.load_features("test_features")
        assert len(loaded) == len(sample_players)
        assert list(loaded.columns) == list(sample_players.columns)

    def test_list_feature_sets(self, tmp_path, sample_players):
        store = FeatureStore(str(tmp_path))
        store.save_features("test", sample_players)
        feature_sets = store.list_feature_sets()
        assert "test" in feature_sets

    def test_key_error_on_missing_feature_set(self, tmp_path):
        store = FeatureStore(str(tmp_path))
        with pytest.raises(KeyError):
            store.load_features("nonexistent_feature_set")

    def test_delete_feature_set(self, tmp_path, sample_players):
        store = FeatureStore(str(tmp_path))
        store.save_features("to_delete", sample_players)
        store.delete_feature_set("to_delete")
        assert "to_delete" not in store.list_feature_sets()
