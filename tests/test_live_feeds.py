"""Tests for live sports data feeds."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.data.live_feeds import (
    OpenF1Client, CricketDataClient, BallDontLieClient,
    OddsAPIClient, LiveFeedsManager,
)


class TestOpenF1Client:
    """Tests for the OpenF1 (Formula 1) client — no API key needed."""

    def test_init(self):
        client = OpenF1Client()
        assert client.BASE_URL == "https://api.openf1.org/v1"

    @patch("src.data.live_feeds.OpenF1Client._get")
    def test_get_drivers(self, mock_get):
        mock_get.return_value = [
            {"driver_number": 1, "broadcast_name": "M VERSTAPPEN", "team_name": "Red Bull Racing"},
            {"driver_number": 44, "broadcast_name": "L HAMILTON", "team_name": "Ferrari"},
        ]
        client = OpenF1Client()
        df = client.get_drivers(session_key="latest")
        assert len(df) == 2
        assert "driver_number" in df.columns

    @patch("src.data.live_feeds.OpenF1Client._get")
    def test_get_meetings(self, mock_get):
        mock_get.return_value = [
            {"meeting_name": "Bahrain Grand Prix", "year": 2025, "country_name": "Bahrain"},
        ]
        client = OpenF1Client()
        df = client.get_meetings(year=2025)
        assert len(df) == 1

    @patch("src.data.live_feeds.OpenF1Client._get")
    def test_get_laps(self, mock_get):
        mock_get.return_value = [
            {"driver_number": 1, "lap_number": 1, "lap_duration": 92.5},
            {"driver_number": 1, "lap_number": 2, "lap_duration": 90.3},
        ]
        client = OpenF1Client()
        df = client.get_laps(session_key="latest")
        assert len(df) == 2
        assert df["lap_duration"].dtype in ["float64", "int64"]

    @patch("src.data.live_feeds.OpenF1Client._get")
    def test_get_weather(self, mock_get):
        mock_get.return_value = [
            {"air_temperature": 28.5, "track_temperature": 42.1, "humidity": 55},
        ]
        client = OpenF1Client()
        df = client.get_weather(session_key="latest")
        assert len(df) == 1
        assert "air_temperature" in df.columns

    @patch("src.data.live_feeds.OpenF1Client._get")
    def test_get_stints(self, mock_get):
        mock_get.return_value = [
            {"driver_number": 1, "compound": "SOFT", "lap_start": 1, "lap_end": 15},
        ]
        client = OpenF1Client()
        df = client.get_stints(session_key="latest")
        assert len(df) == 1

    @patch("src.data.live_feeds.OpenF1Client._get")
    def test_empty_response(self, mock_get):
        mock_get.return_value = []
        client = OpenF1Client()
        df = client.get_drivers()
        assert df.empty

    @patch("src.data.live_feeds.OpenF1Client._get")
    def test_race_summary(self, mock_get):
        mock_get.return_value = [{"test": "data"}]
        client = OpenF1Client()
        summary = client.get_race_summary(session_key="latest")
        assert "drivers" in summary
        assert "laps" in summary
        assert "weather" in summary


class TestCricketDataClient:
    """Tests for the CricketData.org client."""

    def test_init_without_key(self):
        client = CricketDataClient()
        assert client.api_key is None

    def test_init_with_key(self):
        client = CricketDataClient(api_key="test_key")
        assert client.api_key == "test_key"

    def test_check_key_raises(self):
        client = CricketDataClient()
        with pytest.raises(ValueError, match="CricketData API key not configured"):
            client._check_key()

    def test_set_api_key(self):
        client = CricketDataClient()
        client.set_api_key("new_key")
        assert client.api_key == "new_key"

    @patch("src.data.live_feeds.CricketDataClient._get")
    def test_get_current_matches(self, mock_get):
        mock_get.return_value = {
            "status": "success",
            "data": [
                {
                    "id": "123", "name": "India vs Australia", "status": "Live",
                    "venue": "MCG", "date": "2025-01-15", "matchType": "T20",
                    "teams": ["India", "Australia"], "score": [{"r": 180, "w": 4, "o": 18.2}],
                    "matchStarted": True, "matchEnded": False,
                },
            ],
        }
        client = CricketDataClient(api_key="test")
        df = client.get_current_matches()
        assert len(df) == 1
        assert df.iloc[0]["team_1"] == "India"

    @patch("src.data.live_feeds.CricketDataClient._get")
    def test_get_series(self, mock_get):
        mock_get.return_value = {
            "status": "success",
            "data": [{"id": "1", "name": "IPL 2025", "startDate": "2025-03-20"}],
        }
        client = CricketDataClient(api_key="test")
        df = client.get_series()
        assert len(df) == 1

    @patch("src.data.live_feeds.CricketDataClient._get")
    def test_search_players(self, mock_get):
        mock_get.return_value = {
            "status": "success",
            "data": [{"id": "100", "name": "Virat Kohli", "country": "India"}],
        }
        client = CricketDataClient(api_key="test")
        df = client.search_players("Virat")
        assert len(df) == 1
        assert df.iloc[0]["name"] == "Virat Kohli"

    def test_format_score(self):
        client = CricketDataClient()
        assert client._format_score([{"r": 180, "w": 4, "o": 18.2}], 0) == "180/4 (18.2 ov)"
        assert client._format_score([], 0) == ""
        assert client._format_score([{"r": 200, "w": 10, "o": 48}], 1) == ""


class TestBallDontLieClient:
    """Tests for the Ball Don't Lie (NBA) client."""

    def test_init_without_key(self):
        client = BallDontLieClient()
        assert client.api_key is None

    def test_check_key_raises(self):
        client = BallDontLieClient()
        with pytest.raises(ValueError, match="BallDontLie API key not configured"):
            client._check_key()

    @patch("src.data.live_feeds.BallDontLieClient._get")
    def test_get_teams(self, mock_get):
        mock_get.return_value = {
            "data": [
                {"id": 1, "full_name": "Atlanta Hawks", "abbreviation": "ATL", "conference": "East"},
            ],
        }
        client = BallDontLieClient(api_key="test")
        df = client.get_teams()
        assert len(df) == 1

    @patch("src.data.live_feeds.BallDontLieClient._get")
    def test_get_players(self, mock_get):
        mock_get.return_value = {
            "data": [
                {
                    "id": 115, "first_name": "LeBron", "last_name": "James",
                    "position": "F", "team": {"full_name": "Los Angeles Lakers", "abbreviation": "LAL",
                                               "conference": "West", "division": "Pacific"},
                },
            ],
        }
        client = BallDontLieClient(api_key="test")
        df = client.get_players(search="LeBron")
        assert len(df) == 1
        assert df.iloc[0]["first_name"] == "LeBron"

    @patch("src.data.live_feeds.BallDontLieClient._get")
    def test_get_games(self, mock_get):
        mock_get.return_value = {
            "data": [
                {
                    "id": 1, "date": "2025-01-15", "season": 2024, "status": "Final",
                    "period": 4, "time": "", "home_team": {"full_name": "Lakers"},
                    "home_team_score": 110, "visitor_team": {"full_name": "Celtics"},
                    "visitor_team_score": 105, "postseason": False,
                },
            ],
        }
        client = BallDontLieClient(api_key="test")
        df = client.get_games(dates=["2025-01-15"])
        assert len(df) == 1


class TestOddsAPIClient:
    """Tests for The Odds API client."""

    def test_sport_keys(self):
        assert OddsAPIClient.SPORT_KEYS["nba"] == "basketball_nba"
        assert OddsAPIClient.SPORT_KEYS["cricket_ipl"] == "cricket_ipl"

    def test_check_key_raises(self):
        client = OddsAPIClient()
        with pytest.raises(ValueError, match="Odds API key not configured"):
            client._check_key()

    @patch("src.data.live_feeds.OddsAPIClient._get")
    def test_get_odds(self, mock_get):
        mock_get.return_value = [
            {
                "id": "event1", "sport_key": "basketball_nba",
                "home_team": "Lakers", "away_team": "Celtics",
                "commence_time": "2025-01-15T02:00:00Z",
                "bookmakers": [
                    {
                        "title": "DraftKings",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Lakers", "price": -150},
                                    {"name": "Celtics", "price": +130},
                                ],
                            },
                        ],
                    },
                ],
            },
        ]
        client = OddsAPIClient(api_key="test")
        df = client.get_odds(sport="nba")
        assert len(df) == 2  # Two outcomes
        assert "bookmaker" in df.columns

    @patch("src.data.live_feeds.OddsAPIClient._get")
    def test_get_sports(self, mock_get):
        mock_get.return_value = [
            {"key": "basketball_nba", "title": "NBA", "active": True},
        ]
        client = OddsAPIClient(api_key="test")
        df = client.get_sports()
        assert len(df) == 1


class TestLiveFeedsManager:
    """Tests for the unified LiveFeedsManager."""

    def test_init(self):
        manager = LiveFeedsManager()
        assert manager.f1 is not None
        assert manager.cricket is not None
        assert manager.nba is not None
        assert manager.odds is not None

    def test_api_status(self):
        manager = LiveFeedsManager(cricket_key="test_key")
        status = manager.get_api_status()
        assert status["f1_openf1"]["configured"] is True
        assert status["f1_openf1"]["key_required"] is False
        assert status["cricket"]["configured"] is True
        assert status["nba"]["configured"] is False

    def test_init_with_keys(self):
        manager = LiveFeedsManager(cricket_key="ck", nba_key="nk", odds_key="ok")
        assert manager.cricket.api_key == "ck"
        assert manager.nba.api_key == "nk"
        assert manager.odds.api_key == "ok"
