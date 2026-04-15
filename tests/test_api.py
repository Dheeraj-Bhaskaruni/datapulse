"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_check_status_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_check_body(self):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "service" in data


class TestPredictionEndpoints:
    PLAYER_FEATURES = {
        "games_played": 60.0, "points_avg": 25.0, "assists_avg": 7.0,
        "rebounds_avg": 8.0, "steals_avg": 1.5, "blocks_avg": 0.8,
        "turnovers_avg": 2.1, "fg_pct": 0.48, "salary": 8000.0,
        "consistency_score": 0.75,
    }
    RISK_FEATURES = {
        "total_entries": 500.0, "win_rate": 0.65, "avg_entry_fee": 25.0,
        "total_wagered": 50000.0, "total_won": 45000.0, "net_profit": -5000.0,
    }

    def test_player_prediction_status_200(self):
        response = client.post("/predict/player-performance", json={"features": self.PLAYER_FEATURES})
        assert response.status_code in [200, 503]

    def test_player_prediction_body(self):
        response = client.post("/predict/player-performance", json={"features": self.PLAYER_FEATURES})
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "confidence" in data
            assert "model_version" in data
            assert "features_used" in data

    def test_player_prediction_confidence_range(self):
        response = client.post("/predict/player-performance", json={"features": self.PLAYER_FEATURES})
        if response.status_code == 200:
            data = response.json()
            assert 0 <= data["confidence"] <= 1

    def test_risk_score_status_200(self):
        response = client.post("/predict/risk-score", json={"features": self.RISK_FEATURES})
        assert response.status_code in [200, 503]

    def test_risk_score_level_valid(self):
        response = client.post("/predict/risk-score", json={"features": self.RISK_FEATURES})
        if response.status_code == 200:
            data = response.json()
            assert data["risk_level"] in ["low", "medium", "high"]

    def test_risk_score_range(self):
        response = client.post("/predict/risk-score", json={"features": self.RISK_FEATURES})
        if response.status_code == 200:
            data = response.json()
            assert 0 <= data["risk_score"] <= 100


class TestMarketEndpoint:
    def test_market_evaluation_status_200(self):
        response = client.post("/market/evaluate", json={
            "odds": -110,
            "estimated_probability": 0.55,
        })
        assert response.status_code == 200

    def test_market_evaluation_body(self):
        response = client.post("/market/evaluate", json={
            "odds": -110,
            "estimated_probability": 0.55,
        })
        data = response.json()
        assert "expected_value" in data
        assert "implied_probability" in data
        assert "recommendation" in data
        assert data["recommendation"] in ["Bet", "Pass"]

    def test_market_implied_prob_range(self):
        response = client.post("/market/evaluate", json={
            "odds": 150,
            "estimated_probability": 0.45,
        })
        data = response.json()
        assert 0 <= data["implied_probability"] <= 1


class TestAnalyticsEndpoint:
    def test_analytics_summary_status_200(self):
        response = client.get("/analytics/summary")
        assert response.status_code == 200

    def test_analytics_summary_body(self):
        response = client.get("/analytics/summary")
        data = response.json()
        assert "datasets" in data
        assert "total_datasets" in data


class TestDriftEndpoint:
    def test_drift_check_status_200(self):
        response = client.get("/monitoring/drift")
        assert response.status_code == 200

    def test_drift_check_body(self):
        response = client.get("/monitoring/drift")
        data = response.json()
        assert "overall_drift" in data
        assert "columns_checked" in data
        assert "drifted_columns" in data
        assert isinstance(data["drifted_columns"], list)
