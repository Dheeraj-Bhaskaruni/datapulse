"""Tests for data ingestion and preprocessing."""

import pytest
import pandas as pd
import numpy as np
from src.data.ingestion import DataLoader
from src.data.preprocessing import DataCleaner, FeatureEngineer
from src.data.validation import DataValidator, SCHEMAS


class TestDataLoader:
    def test_load_csv(self, tmp_path):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        path = tmp_path / "test.csv"
        df.to_csv(path, index=False)
        loader = DataLoader(str(tmp_path))
        result = loader.load(str(path))
        assert len(result) == 3
        assert list(result.columns) == ['a', 'b']

    def test_load_nonexistent_file(self):
        loader = DataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/file.csv")

    def test_list_available(self, tmp_path):
        (tmp_path / "test_data.csv").write_text("a,b\n1,2\n3,4")
        loader = DataLoader(str(tmp_path.parent))
        available = loader.list_available(tmp_path.name)
        assert 'test_data' in available

    def test_load_multiple(self, tmp_path):
        for name in ['alpha', 'beta']:
            df = pd.DataFrame({'x': [1, 2]})
            df.to_csv(tmp_path / f"{name}.csv", index=False)
        loader = DataLoader(str(tmp_path))
        result = loader.load_multiple([str(tmp_path / "alpha.csv"), str(tmp_path / "beta.csv")])
        assert 'alpha' in result
        assert 'beta' in result


class TestDataCleaner:
    def test_remove_duplicates(self):
        df = pd.DataFrame({'a': [1, 1, 2], 'b': [3, 3, 4]})
        cleaner = DataCleaner()
        result = cleaner.remove_duplicates(df)
        assert len(result) == 2

    def test_handle_missing_values_numeric(self):
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, np.nan]})
        cleaner = DataCleaner()
        result = cleaner.handle_missing_values(df)
        assert result.isnull().sum().sum() == 0

    def test_handle_missing_values_categorical(self):
        df = pd.DataFrame({'a': ['x', None, 'z']})
        cleaner = DataCleaner()
        result = cleaner.handle_missing_values(df)
        assert result['a'].isnull().sum() == 0

    def test_full_clean_pipeline(self, sample_players):
        cleaner = DataCleaner()
        result = cleaner.clean(sample_players)
        assert len(result) > 0
        assert 'initial_shape' in cleaner.cleaning_report

    def test_cleaning_report_populated(self, sample_players):
        cleaner = DataCleaner()
        cleaner.clean(sample_players)
        report = cleaner.cleaning_report
        assert 'initial_shape' in report
        assert 'final_shape' in report
        assert 'rows_removed' in report


class TestFeatureEngineer:
    def test_encode_categorical_low_cardinality(self):
        df = pd.DataFrame({'position': ['PG', 'SG', 'SF', 'PF', 'C'] * 10})
        fe = FeatureEngineer()
        result = fe.encode_categorical(df)
        assert result.select_dtypes(include=['object']).shape[1] == 0

    def test_scale_features_standard(self, sample_players):
        fe = FeatureEngineer()
        cols = ['points_avg', 'assists_avg']
        result = fe.scale_features(sample_players, cols, method='standard')
        assert abs(result[cols[0]].mean()) < 0.1

    def test_scale_features_minmax(self, sample_players):
        fe = FeatureEngineer()
        cols = ['salary']
        result = fe.scale_features(sample_players, cols, method='minmax')
        assert result[cols[0]].min() >= -1e-9
        assert result[cols[0]].max() <= 1 + 1e-9

    def test_create_features(self, sample_players):
        fe = FeatureEngineer()
        result = fe.create_features(sample_players)
        assert len(result.columns) > len(sample_players.columns)


class TestDataValidator:
    def test_valid_players(self, sample_players):
        validator = DataValidator()
        result = validator.validate(sample_players, SCHEMAS['players'])
        assert result.is_valid

    def test_missing_columns(self):
        df = pd.DataFrame({'a': [1, 2]})
        validator = DataValidator()
        result = validator.validate(df, SCHEMAS['players'])
        assert not result.is_valid

    def test_validation_result_summary(self, sample_players):
        validator = DataValidator()
        result = validator.validate(sample_players, SCHEMAS['players'])
        summary = result.summary()
        assert 'PASSED' in summary or 'FAILED' in summary

    def test_value_range_violation(self):
        df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'username': ['a', 'b', 'c'],
            'total_entries': [10, 20, 30],
            'win_rate': [1.5, 0.5, -0.1],  # out of range
        })
        validator = DataValidator()
        result = validator.validate(df, SCHEMAS['user_profiles'])
        assert result.checks_failed > 0
