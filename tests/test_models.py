"""Tests for DataPulse ML models."""

import pytest
import numpy as np
import pandas as pd

from src.models.player_performance import PlayerPerformanceModel
from src.models.risk_scoring import RiskScoringModel
from src.models.anomaly_detection import AnomalyDetectionModel
from src.models.evaluation import ModelEvaluator



@pytest.fixture
def player_features():
    np.random.seed(0)
    n = 100
    return pd.DataFrame({
        'games_played': np.random.randint(20, 82, n),
        'points_avg': np.random.uniform(5, 35, n),
        'assists_avg': np.random.uniform(1, 12, n),
        'rebounds_avg': np.random.uniform(2, 14, n),
        'steals_avg': np.random.uniform(0.3, 2.5, n),
        'blocks_avg': np.random.uniform(0.1, 2.5, n),
        'turnovers_avg': np.random.uniform(0.5, 4.5, n),
        'fg_pct': np.random.uniform(0.38, 0.62, n),
        'salary': np.random.randint(3500, 11000, n),
        'consistency_score': np.random.uniform(0.3, 0.95, n),
    })


@pytest.fixture
def player_target(player_features):
    np.random.seed(0)
    return (
        player_features['points_avg'] * 1.0
        + player_features['assists_avg'] * 1.5
        + player_features['rebounds_avg'] * 1.25
        + np.random.normal(0, 2, len(player_features))
    )


@pytest.fixture
def risk_features():
    np.random.seed(1)
    n = 150
    return pd.DataFrame({
        'total_entries': np.random.randint(10, 5000, n),
        'win_rate': np.random.uniform(0.2, 0.8, n),
        'avg_entry_fee': np.random.uniform(1, 100, n),
        'total_wagered': np.random.uniform(100, 100000, n),
        'total_won': np.random.uniform(50, 120000, n),
        'net_profit': np.random.uniform(-20000, 30000, n),
    })


@pytest.fixture
def risk_target():
    np.random.seed(1)
    return pd.Series(np.random.randint(0, 3, 150))


@pytest.fixture
def anomaly_features():
    np.random.seed(2)
    n = 200
    return pd.DataFrame({
        'total_entries': np.random.randint(10, 5000, n),
        'win_rate': np.random.uniform(0.2, 0.8, n),
        'avg_entry_fee': np.random.uniform(1, 100, n),
        'total_wagered': np.random.uniform(100, 100000, n),
        'total_won': np.random.uniform(50, 120000, n),
        'net_profit': np.random.uniform(-20000, 30000, n),
    })



class TestPlayerPerformanceModel:
    def test_train_and_predict(self, player_features, player_target):
        model = PlayerPerformanceModel()
        model.train(player_features, player_target)
        preds = model.predict(player_features)
        assert len(preds) == len(player_features)
        assert isinstance(preds, np.ndarray)

    def test_untrained_model_raises(self, player_features):
        model = PlayerPerformanceModel()
        with pytest.raises(RuntimeError):
            model.predict(player_features)

    def test_feature_importance_shape(self, player_features, player_target):
        model = PlayerPerformanceModel()
        model.train(player_features, player_target)
        assert model.feature_importances_.shape == (len(player_features.columns),)

    def test_save_and_load(self, player_features, player_target, tmp_path):
        model = PlayerPerformanceModel()
        model.train(player_features, player_target)
        expected = model.predict(player_features)

        save_dir = str(tmp_path / "player_model")
        model.save(save_dir)

        loaded = PlayerPerformanceModel()
        loaded.load(save_dir)
        result = loaded.predict(player_features)

        np.testing.assert_allclose(result, expected)

    def test_cv_metrics_present(self, player_features, player_target):
        model = PlayerPerformanceModel()
        model.train(player_features, player_target)
        assert isinstance(model.cv_metrics, dict)
        assert len(model.cv_metrics) > 0
        assert 'rmse' in model.cv_metrics



class TestRiskScoringModel:
    def test_train_and_predict(self, risk_features, risk_target):
        model = RiskScoringModel()
        model.train(risk_features, risk_target)
        preds = model.predict(risk_features)
        assert len(preds) == len(risk_features)

    def test_predict_proba_shape(self, risk_features, risk_target):
        model = RiskScoringModel()
        model.train(risk_features, risk_target)
        proba = model.predict_proba(risk_features)
        n_classes = len(np.unique(risk_target))
        assert proba.shape == (len(risk_features), n_classes)

    def test_feature_importance(self, risk_features, risk_target):
        model = RiskScoringModel()
        model.train(risk_features, risk_target)
        assert model.feature_importances_.shape == (len(risk_features.columns),)



class TestAnomalyDetectionModel:
    def test_train_and_predict(self, anomaly_features):
        model = AnomalyDetectionModel()
        model.train(anomaly_features)
        preds = model.predict(anomaly_features)
        assert len(preds) == len(anomaly_features)
        assert set(preds).issubset({-1, 1})

    def test_anomaly_scores_negative_is_anomalous(self, anomaly_features):
        model = AnomalyDetectionModel()
        model.train(anomaly_features)
        scores = model.anomaly_scores(anomaly_features)
        labels = model.predict(anomaly_features)
        # anomalies (label == -1) should have lower scores than normals (label == 1)
        anomaly_score_mean = scores[labels == -1].mean() if (labels == -1).any() else None
        normal_score_mean = scores[labels == 1].mean() if (labels == 1).any() else None
        if anomaly_score_mean is not None and normal_score_mean is not None:
            assert anomaly_score_mean < normal_score_mean

    def test_statistical_anomalies(self, anomaly_features):
        result = AnomalyDetectionModel.statistical_anomalies(
            anomaly_features, columns=['total_wagered', 'net_profit'], z_threshold=2.0
        )
        assert 'total_anomaly_flags' in result.columns
        assert 'total_wagered_anomaly' in result.columns
        assert 'net_profit_anomaly' in result.columns
        assert len(result) == len(anomaly_features)

    def test_untrained_anomaly_scores_raises(self, anomaly_features):
        model = AnomalyDetectionModel()
        with pytest.raises(RuntimeError):
            model.anomaly_scores(anomaly_features)



class TestModelEvaluator:
    @pytest.fixture
    def reg_data(self):
        np.random.seed(42)
        y_true = np.random.uniform(10, 60, 100)
        y_pred = y_true + np.random.normal(0, 3, 100)
        return y_true, y_pred

    @pytest.fixture
    def clf_data(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        y_proba = np.column_stack([
            np.random.uniform(0.2, 0.8, 200),
            np.random.uniform(0.2, 0.8, 200),
        ])
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
        y_pred = y_proba.argmax(axis=1)
        return y_true, y_pred, y_proba

    def test_regression_metrics_keys(self, reg_data):
        y_true, y_pred = reg_data
        evaluator = ModelEvaluator()
        result = evaluator.evaluate_regression(y_true, y_pred)
        for key in ('rmse', 'mae', 'r2', 'mape'):
            assert key in result

    def test_regression_r2_bounded(self, reg_data):
        y_true, y_pred = reg_data
        evaluator = ModelEvaluator()
        # r2 can be negative for bad models, but should be a float
        result = evaluator.evaluate_regression(y_true, y_pred)
        assert isinstance(result['r2'], float)
        # For a trivially bad predictor, r2 can be very negative; check it's finite
        assert np.isfinite(result['r2'])

    def test_classification_metrics_keys(self, clf_data):
        y_true, y_pred, y_proba = clf_data
        evaluator = ModelEvaluator()
        result = evaluator.evaluate_classification(y_true, y_pred, y_proba)
        for key in ('accuracy', 'precision', 'recall', 'f1'):
            assert key in result

    def test_roc_data_structure(self, clf_data):
        y_true, y_pred, y_proba = clf_data
        evaluator = ModelEvaluator()
        result = evaluator.evaluate_classification(y_true, y_pred, y_proba)
        assert 'roc_auc' in result
        assert 'roc_curve' in result
        roc = result['roc_curve']
        assert 'fpr' in roc
        assert 'tpr' in roc
        assert len(roc['fpr']) == len(roc['tpr'])

    def test_profit_curve_length(self, clf_data):
        y_true, y_pred, y_proba = clf_data
        evaluator = ModelEvaluator()
        n_thresholds = 50
        result = evaluator.profit_curve(y_true, y_proba[:, 1], n_thresholds=n_thresholds)
        assert 'thresholds' in result
        assert 'profits' in result
        assert len(result['thresholds']) == n_thresholds
        assert len(result['profits']) == n_thresholds
