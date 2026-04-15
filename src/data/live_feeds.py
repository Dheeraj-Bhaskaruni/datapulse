"""
Live sports data feeds module.

Integrates with free APIs:
- OpenF1 API (api.openf1.org) — Formula 1 data, NO key required
- CricketData.org API — Cricket data, free tier (key required, free signup)
- Ball Don't Lie API (balldontlie.io) — NBA data, free tier (key required)
- The Odds API — Live betting odds, free tier (key required)
"""

import logging
import time
import requests
import pandas as pd
from typing import Optional, Dict, Any, List
from functools import wraps

logger = logging.getLogger(__name__)


def api_retry(max_retries: int = 3, delay: float = 1.0):
    """Retry decorator with exponential backoff for API calls."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    wait = delay * (2 ** attempt)
                    logger.warning(f"API call failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait}s...")
                    time.sleep(wait)
            logger.error(f"API call failed after {max_retries} attempts: {last_exception}")
            raise last_exception
        return wrapper
    return decorator


# Formula 1 — OpenF1 API (completely free, no key needed)


class OpenF1Client:
    """
    Client for the OpenF1 API — real-time and historical F1 data.

    Base URL: https://api.openf1.org/v1/
    Rate limit: 3 req/s, 30 req/min (free tier)
    No API key required.
    Data available from 2023 season onwards.
    """

    BASE_URL = "https://api.openf1.org/v1"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    @api_retry(max_retries=2, delay=0.5)
    def _get(self, endpoint: str, params: Optional[Dict] = None) -> List[Dict]:
        """Make a GET request to the OpenF1 API."""
        url = f"{self.BASE_URL}/{endpoint}"
        resp = self.session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_drivers(self, session_key: Optional[str] = None) -> pd.DataFrame:
        """Get driver information. Pass session_key for a specific session, or 'latest'."""
        params = {}
        if session_key:
            params["session_key"] = session_key
        data = self._get("drivers", params)
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        logger.info(f"Fetched {len(df)} driver records from OpenF1")
        return df

    def get_latest_drivers(self) -> pd.DataFrame:
        """Get drivers from the most recent session."""
        return self.get_drivers(session_key="latest")

    def get_meetings(self, year: Optional[int] = None) -> pd.DataFrame:
        """Get F1 meetings (race weekends) for a given year."""
        params = {}
        if year:
            params["year"] = year
        data = self._get("meetings", params)
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)

    def get_sessions(self, meeting_key: Optional[int] = None,
                     session_type: Optional[str] = None,
                     year: Optional[int] = None) -> pd.DataFrame:
        """
        Get sessions (Practice, Qualifying, Race, Sprint).

        Args:
            meeting_key: Filter by specific meeting
            session_type: 'Practice', 'Qualifying', 'Race', 'Sprint'
            year: Filter by year
        """
        params = {}
        if meeting_key:
            params["meeting_key"] = meeting_key
        if session_type:
            params["session_type"] = session_type
        if year:
            params["year"] = year
        data = self._get("sessions", params)
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)

    def get_laps(self, session_key: str = "latest",
                 driver_number: Optional[int] = None,
                 lap_number: Optional[int] = None) -> pd.DataFrame:
        """Get lap time data for a session."""
        params = {"session_key": session_key}
        if driver_number:
            params["driver_number"] = driver_number
        if lap_number:
            params["lap_number"] = lap_number
        data = self._get("laps", params)
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if 'lap_duration' in df.columns:
            df['lap_duration'] = pd.to_numeric(df['lap_duration'], errors='coerce')
        logger.info(f"Fetched {len(df)} lap records from OpenF1")
        return df

    def get_positions(self, session_key: str = "latest",
                      driver_number: Optional[int] = None) -> pd.DataFrame:
        """Get position data during a session."""
        params = {"session_key": session_key}
        if driver_number:
            params["driver_number"] = driver_number
        data = self._get("position", params)
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)

    def get_pit_stops(self, session_key: str = "latest",
                      driver_number: Optional[int] = None) -> pd.DataFrame:
        """Get pit stop data."""
        params = {"session_key": session_key}
        if driver_number:
            params["driver_number"] = driver_number
        data = self._get("pit", params)
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)

    def get_stints(self, session_key: str = "latest",
                   driver_number: Optional[int] = None) -> pd.DataFrame:
        """Get stint data (tire compound, stint duration)."""
        params = {"session_key": session_key}
        if driver_number:
            params["driver_number"] = driver_number
        data = self._get("stints", params)
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)

    def get_weather(self, session_key: str = "latest") -> pd.DataFrame:
        """Get weather data for a session."""
        params = {"session_key": session_key}
        data = self._get("weather", params)
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)

    def get_race_control(self, session_key: str = "latest") -> pd.DataFrame:
        """Get race control messages (flags, penalties, etc.)."""
        params = {"session_key": session_key}
        data = self._get("race_control", params)
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)

    def get_intervals(self, session_key: str = "latest",
                      driver_number: Optional[int] = None) -> pd.DataFrame:
        """Get gap/interval data between drivers."""
        params = {"session_key": session_key}
        if driver_number:
            params["driver_number"] = driver_number
        data = self._get("intervals", params)
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)

    def get_race_summary(self, session_key: str = "latest") -> Dict[str, pd.DataFrame]:
        """Get a complete race summary with drivers, laps, positions, and stints."""
        return {
            "drivers": self.get_drivers(session_key),
            "laps": self.get_laps(session_key),
            "positions": self.get_positions(session_key),
            "stints": self.get_stints(session_key),
            "weather": self.get_weather(session_key),
        }

    def get_season_calendar(self, year: int = 2025) -> pd.DataFrame:
        """Get the full race calendar for a season."""
        meetings = self.get_meetings(year=year)
        if meetings.empty:
            return meetings
        cols = [c for c in ['meeting_name', 'meeting_official_name', 'location',
                            'country_name', 'date_start', 'year'] if c in meetings.columns]
        return meetings[cols] if cols else meetings

# CRICKET — CricketData.org API (free tier, key required)


class CricketDataClient:
    """
    Client for the CricketData.org API.

    Free tier includes: current matches, match info, series info, player stats.
    Sign up at: https://cricketdata.org/
    """

    BASE_URL = "https://api.cricapi.com/v1"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.session = requests.Session()
        if not api_key:
            logger.warning("CricketData API key not set. Use set_api_key() or pass key to constructor.")

    def set_api_key(self, key: str) -> None:
        """Set the API key."""
        self.api_key = key

    def _check_key(self) -> None:
        if not self.api_key:
            raise ValueError(
                "CricketData API key not configured. "
                "Get a free key at https://cricketdata.org/ and set CRICKET_API_KEY in .env"
            )

    @api_retry(max_retries=2, delay=1.0)
    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make a GET request to the CricketData API."""
        self._check_key()
        url = f"{self.BASE_URL}/{endpoint}"
        all_params = {"apikey": self.api_key}
        if params:
            all_params.update(params)
        resp = self.session.get(url, params=all_params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "success":
            raise ValueError(f"API error: {data.get('status', 'unknown')} - {data.get('info', '')}")
        return data

    def get_current_matches(self) -> pd.DataFrame:
        """Get all current/live cricket matches."""
        data = self._get("currentMatches")
        matches = data.get("data", [])
        if not matches:
            return pd.DataFrame()
        rows = []
        for m in matches:
            rows.append({
                "match_id": m.get("id"),
                "name": m.get("name"),
                "status": m.get("status"),
                "venue": m.get("venue"),
                "date": m.get("date"),
                "date_time_gmt": m.get("dateTimeGMT"),
                "match_type": m.get("matchType"),
                "team_1": m.get("teams", [None, None])[0] if m.get("teams") else None,
                "team_2": m.get("teams", [None, None])[1] if len(m.get("teams", [])) > 1 else None,
                "score_1": self._format_score(m.get("score", []), 0),
                "score_2": self._format_score(m.get("score", []), 1),
                "series_id": m.get("series_id"),
                "match_started": m.get("matchStarted"),
                "match_ended": m.get("matchEnded"),
            })
        df = pd.DataFrame(rows)
        logger.info(f"Fetched {len(df)} current cricket matches")
        return df

    def _format_score(self, scores: list, index: int) -> str:
        """Format a score entry."""
        if not scores or index >= len(scores):
            return ""
        s = scores[index]
        if isinstance(s, dict):
            runs = s.get("r", "")
            wickets = s.get("w", "")
            overs = s.get("o", "")
            return f"{runs}/{wickets} ({overs} ov)" if runs else ""
        return str(s)

    def get_match_info(self, match_id: str) -> Dict[str, Any]:
        """Get detailed info for a specific match."""
        data = self._get("match_info", {"id": match_id})
        return data.get("data", {})

    def get_match_scorecard(self, match_id: str) -> Dict[str, Any]:
        """Get full scorecard for a match."""
        data = self._get("match_scoreCard", {"id": match_id})
        return data.get("data", {})

    def get_series(self) -> pd.DataFrame:
        """Get list of current/upcoming cricket series."""
        data = self._get("series")
        series_list = data.get("data", [])
        if not series_list:
            return pd.DataFrame()
        rows = []
        for s in series_list:
            rows.append({
                "series_id": s.get("id"),
                "name": s.get("name"),
                "start_date": s.get("startDate"),
                "end_date": s.get("endDate"),
                "odi": s.get("odi", 0),
                "t20": s.get("t20", 0),
                "test": s.get("test", 0),
                "squads": s.get("squads", 0),
                "matches": s.get("matches", 0),
            })
        return pd.DataFrame(rows)

    def search_players(self, name: str) -> pd.DataFrame:
        """Search for cricket players by name."""
        data = self._get("players", {"search": name})
        players = data.get("data", [])
        if not players:
            return pd.DataFrame()
        rows = []
        for p in players:
            rows.append({
                "player_id": p.get("id"),
                "name": p.get("name"),
                "country": p.get("country"),
            })
        return pd.DataFrame(rows)

    def get_player_stats(self, player_id: str) -> Dict[str, Any]:
        """Get detailed stats for a player."""
        data = self._get("players_info", {"id": player_id})
        return data.get("data", {})

    def get_match_list(self) -> pd.DataFrame:
        """Get list of upcoming and recent matches."""
        data = self._get("matches")
        matches = data.get("data", [])
        if not matches:
            return pd.DataFrame()
        rows = []
        for m in matches:
            rows.append({
                "match_id": m.get("id"),
                "name": m.get("name"),
                "status": m.get("status"),
                "venue": m.get("venue"),
                "date": m.get("date"),
                "match_type": m.get("matchType"),
                "team_1": m.get("teams", [None])[0] if m.get("teams") else None,
                "team_2": m.get("teams", [None, None])[1] if len(m.get("teams", [])) > 1 else None,
                "match_started": m.get("matchStarted"),
                "match_ended": m.get("matchEnded"),
            })
        return pd.DataFrame(rows)

# NBA — Ball Don't Lie API (free tier, key required)


class BallDontLieClient:
    """
    Client for the Ball Don't Lie API — NBA stats and data.

    Free tier: 5 requests/minute.
    Sign up at: https://www.balldontlie.io/
    """

    BASE_URL = "https://api.balldontlie.io/v1"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": api_key})
        else:
            logger.warning("BallDontLie API key not set. Use set_api_key() or pass key to constructor.")

    def set_api_key(self, key: str) -> None:
        """Set the API key."""
        self.api_key = key
        self.session.headers.update({"Authorization": key})

    def _check_key(self) -> None:
        if not self.api_key:
            raise ValueError(
                "BallDontLie API key not configured. "
                "Get a free key at https://www.balldontlie.io/ and set NBA_API_KEY in .env"
            )

    @api_retry(max_retries=2, delay=1.5)
    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make a GET request to the BallDontLie API."""
        self._check_key()
        url = f"{self.BASE_URL}/{endpoint}"
        resp = self.session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_teams(self) -> pd.DataFrame:
        """Get all NBA teams."""
        data = self._get("teams")
        teams = data.get("data", [])
        if not teams:
            return pd.DataFrame()
        return pd.DataFrame(teams)

    def get_players(self, search: Optional[str] = None,
                    per_page: int = 25, cursor: Optional[int] = None) -> pd.DataFrame:
        """Get NBA players with optional search."""
        params = {"per_page": per_page}
        if search:
            params["search"] = search
        if cursor:
            params["cursor"] = cursor
        data = self._get("players", params)
        players = data.get("data", [])
        if not players:
            return pd.DataFrame()
        rows = []
        for p in players:
            team = p.get("team", {}) or {}
            rows.append({
                "player_id": p.get("id"),
                "first_name": p.get("first_name"),
                "last_name": p.get("last_name"),
                "position": p.get("position"),
                "height": p.get("height"),
                "weight": p.get("weight"),
                "jersey_number": p.get("jersey_number"),
                "college": p.get("college"),
                "country": p.get("country"),
                "draft_year": p.get("draft_year"),
                "draft_round": p.get("draft_round"),
                "draft_number": p.get("draft_number"),
                "team_name": team.get("full_name"),
                "team_abbreviation": team.get("abbreviation"),
                "conference": team.get("conference"),
                "division": team.get("division"),
            })
        return pd.DataFrame(rows)

    def get_games(self, dates: Optional[List[str]] = None,
                  seasons: Optional[List[int]] = None,
                  team_ids: Optional[List[int]] = None,
                  per_page: int = 25) -> pd.DataFrame:
        """Get NBA games with optional filters."""
        params = {"per_page": per_page}
        if dates:
            params["dates[]"] = dates
        if seasons:
            params["seasons[]"] = seasons
        if team_ids:
            params["team_ids[]"] = team_ids
        data = self._get("games", params)
        games = data.get("data", [])
        if not games:
            return pd.DataFrame()
        rows = []
        for g in games:
            home = g.get("home_team", {}) or {}
            visitor = g.get("visitor_team", {}) or {}
            rows.append({
                "game_id": g.get("id"),
                "date": g.get("date"),
                "season": g.get("season"),
                "status": g.get("status"),
                "period": g.get("period"),
                "time": g.get("time"),
                "home_team": home.get("full_name"),
                "home_score": g.get("home_team_score"),
                "visitor_team": visitor.get("full_name"),
                "visitor_score": g.get("visitor_team_score"),
                "postseason": g.get("postseason"),
            })
        return pd.DataFrame(rows)

    def get_stats(self, player_ids: Optional[List[int]] = None,
                  game_ids: Optional[List[int]] = None,
                  seasons: Optional[List[int]] = None,
                  per_page: int = 25) -> pd.DataFrame:
        """Get player box score stats."""
        params = {"per_page": per_page}
        if player_ids:
            params["player_ids[]"] = player_ids
        if game_ids:
            params["game_ids[]"] = game_ids
        if seasons:
            params["seasons[]"] = seasons
        data = self._get("stats", params)
        stats = data.get("data", [])
        if not stats:
            return pd.DataFrame()
        rows = []
        for s in stats:
            player = s.get("player", {}) or {}
            team = s.get("team", {}) or {}
            rows.append({
                "player_name": f"{player.get('first_name', '')} {player.get('last_name', '')}",
                "player_id": player.get("id"),
                "team": team.get("full_name"),
                "game_id": s.get("game", {}).get("id") if isinstance(s.get("game"), dict) else s.get("game"),
                "min": s.get("min"),
                "pts": s.get("pts"),
                "ast": s.get("ast"),
                "reb": s.get("reb"),
                "stl": s.get("stl"),
                "blk": s.get("blk"),
                "turnover": s.get("turnover"),
                "fg_pct": s.get("fg_pct"),
                "fg3_pct": s.get("fg3_pct"),
                "ft_pct": s.get("ft_pct"),
                "fgm": s.get("fgm"),
                "fga": s.get("fga"),
                "fg3m": s.get("fg3m"),
                "fg3a": s.get("fg3a"),
                "ftm": s.get("ftm"),
                "fta": s.get("fta"),
                "oreb": s.get("oreb"),
                "dreb": s.get("dreb"),
                "pf": s.get("pf"),
            })
        return pd.DataFrame(rows)

    def get_season_averages(self, season: int, player_ids: List[int]) -> pd.DataFrame:
        """Get season averages for specified players."""
        params = {"season": season, "player_ids[]": player_ids}
        data = self._get("season_averages", params)
        averages = data.get("data", [])
        if not averages:
            return pd.DataFrame()
        return pd.DataFrame(averages)

# ODDS — The Odds API (free tier, key required)


class OddsAPIClient:
    """
    Client for The Odds API — live betting odds from 40+ sportsbooks.

    Free tier: 500 requests/month.
    Sign up at: https://the-odds-api.com/
    """

    BASE_URL = "https://api.the-odds-api.com/v4"

    SPORT_KEYS = {
        "nba": "basketball_nba",
        "nfl": "americanfootball_nfl",
        "mlb": "baseball_mlb",
        "nhl": "icehockey_nhl",
        "epl": "soccer_epl",
        "cricket_ipl": "cricket_ipl",
        "cricket_test": "cricket_test_match",
        "cricket_odi": "cricket_odi",
        "cricket_t20": "cricket_international_t20",
        "mma": "mma_mixed_martial_arts",
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.session = requests.Session()
        self.remaining_requests: Optional[int] = None
        if not api_key:
            logger.warning("Odds API key not set. Get a free key at https://the-odds-api.com/")

    def set_api_key(self, key: str) -> None:
        self.api_key = key

    def _check_key(self) -> None:
        if not self.api_key:
            raise ValueError(
                "Odds API key not configured. "
                "Get a free key at https://the-odds-api.com/ and set ODDS_API_KEY in .env"
            )

    @api_retry(max_retries=2, delay=1.0)
    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        """Make a GET request to The Odds API."""
        self._check_key()
        url = f"{self.BASE_URL}/{endpoint}"
        all_params = {"apiKey": self.api_key}
        if params:
            all_params.update(params)
        resp = self.session.get(url, params=all_params, timeout=15)
        resp.raise_for_status()

        # Track remaining quota
        self.remaining_requests = resp.headers.get("x-requests-remaining")
        if self.remaining_requests:
            logger.info(f"Odds API remaining requests: {self.remaining_requests}")

        return resp.json()

    def get_sports(self) -> pd.DataFrame:
        """Get list of available sports."""
        data = self._get("sports")
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)

    def get_odds(self, sport: str = "nba", regions: str = "us",
                 markets: str = "h2h", odds_format: str = "american") -> pd.DataFrame:
        """
        Get live odds for a sport.

        Args:
            sport: Sport key (use SPORT_KEYS dict or raw key like 'basketball_nba')
            regions: 'us', 'uk', 'eu', 'au' (comma-separated for multiple)
            markets: 'h2h' (moneyline), 'spreads', 'totals'
            odds_format: 'american' or 'decimal'
        """
        sport_key = self.SPORT_KEYS.get(sport, sport)
        params = {
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
        }
        data = self._get(f"sports/{sport_key}/odds", params)
        if not data:
            return pd.DataFrame()

        rows = []
        for event in data:
            base = {
                "event_id": event.get("id"),
                "sport": event.get("sport_key"),
                "home_team": event.get("home_team"),
                "away_team": event.get("away_team"),
                "commence_time": event.get("commence_time"),
            }
            for bookmaker in event.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    for outcome in market.get("outcomes", []):
                        rows.append({
                            **base,
                            "bookmaker": bookmaker.get("title"),
                            "market": market.get("key"),
                            "selection": outcome.get("name"),
                            "price": outcome.get("price"),
                            "point": outcome.get("point"),
                        })

        df = pd.DataFrame(rows)
        logger.info(f"Fetched {len(df)} odds entries for {sport_key}")
        return df

    def get_cricket_odds(self, league: str = "cricket_ipl",
                         markets: str = "h2h") -> pd.DataFrame:
        """Get cricket betting odds."""
        return self.get_odds(sport=league, markets=markets)

    def get_scores(self, sport: str = "nba", days_from: int = 1) -> pd.DataFrame:
        """Get recent scores for a sport."""
        sport_key = self.SPORT_KEYS.get(sport, sport)
        params = {"daysFrom": days_from}
        data = self._get(f"sports/{sport_key}/scores", params)
        if not data:
            return pd.DataFrame()
        rows = []
        for event in data:
            row = {
                "event_id": event.get("id"),
                "sport": event.get("sport_key"),
                "home_team": event.get("home_team"),
                "away_team": event.get("away_team"),
                "commence_time": event.get("commence_time"),
                "completed": event.get("completed"),
            }
            scores = event.get("scores")
            if scores:
                for score in scores:
                    if score.get("name") == event.get("home_team"):
                        row["home_score"] = score.get("score")
                    elif score.get("name") == event.get("away_team"):
                        row["away_score"] = score.get("score")
            rows.append(row)
        return pd.DataFrame(rows)

# Unified Live Feeds Manager


class LiveFeedsManager:
    """Unified manager for all live sports data feeds."""

    def __init__(self, cricket_key: Optional[str] = None,
                 nba_key: Optional[str] = None,
                 odds_key: Optional[str] = None):
        import os

        self.f1 = OpenF1Client()  # No key needed
        self.cricket = CricketDataClient(api_key=cricket_key or os.environ.get("CRICKET_API_KEY"))
        self.nba = BallDontLieClient(api_key=nba_key or os.environ.get("NBA_API_KEY"))
        self.odds = OddsAPIClient(api_key=odds_key or os.environ.get("ODDS_API_KEY"))

    def get_api_status(self) -> Dict[str, Dict[str, Any]]:
        """Check which APIs are configured and available."""
        status = {
            "f1_openf1": {
                "configured": True,  # Always available, no key needed
                "key_required": False,
                "description": "Formula 1 real-time data",
            },
            "cricket": {
                "configured": bool(self.cricket.api_key),
                "key_required": True,
                "signup_url": "https://cricketdata.org/",
                "description": "Cricket live scores & stats",
            },
            "nba": {
                "configured": bool(self.nba.api_key),
                "key_required": True,
                "signup_url": "https://www.balldontlie.io/",
                "description": "NBA stats & games",
            },
            "odds": {
                "configured": bool(self.odds.api_key),
                "key_required": True,
                "signup_url": "https://the-odds-api.com/",
                "description": "Live betting odds (40+ sportsbooks)",
            },
        }
        return status

    def fetch_all_available(self) -> Dict[str, pd.DataFrame]:
        """Fetch data from all configured APIs. Skips unconfigured ones gracefully."""
        results = {}

        # F1 is always available
        try:
            results["f1_drivers"] = self.f1.get_latest_drivers()
            results["f1_meetings"] = self.f1.get_meetings(year=2025)
        except Exception as e:
            logger.warning(f"F1 data fetch failed: {e}")

        # Cricket (if configured)
        if self.cricket.api_key:
            try:
                results["cricket_matches"] = self.cricket.get_current_matches()
            except Exception as e:
                logger.warning(f"Cricket data fetch failed: {e}")

        # NBA (if configured)
        if self.nba.api_key:
            try:
                results["nba_teams"] = self.nba.get_teams()
            except Exception as e:
                logger.warning(f"NBA data fetch failed: {e}")

        # Odds (if configured)
        if self.odds.api_key:
            try:
                results["odds_nba"] = self.odds.get_odds(sport="nba")
            except Exception as e:
                logger.warning(f"Odds data fetch failed: {e}")

        return results
