"""
data_pipeline.py - Comprehensive NBA Data Pipeline

Uses Balldontlie GOAT API for:
- Live DraftKings player props
- Player game logs and season averages
- Usage rates and advanced stats
- Injury reports
- Team schedules (for back-to-back detection)
- Box scores and pace data

Now with SQLite caching for faster performance!
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import requests
import numpy as np

# Import cache (optional - will work without it)
try:
    from cache_db import get_cache, CacheDB
    CACHE_ENABLED = True
except ImportError:
    CACHE_ENABLED = False
    get_cache = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PlayerProp:
    """Live DraftKings prop line"""
    player_id: int
    player_name: str
    prop_type: str  # points, rebounds, assists, threes, etc.
    line: float
    over_odds: int  # American odds
    under_odds: int
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class GameLog:
    """Single game performance"""
    game_id: int
    date: str
    opponent: str
    home: bool
    minutes: float
    pts: int
    reb: int
    ast: int
    stl: int
    blk: int
    fg3m: int  # 3-pointers made
    turnover: int
    fga: int
    fgm: int
    fta: int
    ftm: int


@dataclass
class PlayerProfile:
    """Complete player data for projection"""
    id: int
    name: str
    team: str
    team_id: int
    position: str

    # Season averages
    season_avg: Dict[str, float] = field(default_factory=dict)

    # Rolling averages
    last_5_avg: Dict[str, float] = field(default_factory=dict)
    last_10_avg: Dict[str, float] = field(default_factory=dict)

    # Standard deviations (for edge calculation)
    std_dev: Dict[str, float] = field(default_factory=dict)

    # Per-minute rates
    per_minute: Dict[str, float] = field(default_factory=dict)

    # Usage data
    usage_pct: float = 0.0

    # Home/Away splits
    home_avg: Dict[str, float] = field(default_factory=dict)
    away_avg: Dict[str, float] = field(default_factory=dict)

    # Raw game logs for correlation
    game_logs: List[GameLog] = field(default_factory=list)

    # Injury status
    status: str = "active"  # active, out, questionable, doubtful, probable
    injury_note: str = ""


@dataclass
class TeamProfile:
    """Team-level data for matchup adjustments"""
    id: int
    name: str
    abbreviation: str

    # Pace and efficiency
    pace: float = 100.0  # possessions per game
    offensive_rating: float = 110.0
    defensive_rating: float = 110.0

    # Position defense (points allowed vs position)
    def_vs_pg: float = 1.0  # multiplier vs league avg
    def_vs_sg: float = 1.0
    def_vs_sf: float = 1.0
    def_vs_pf: float = 1.0
    def_vs_c: float = 1.0

    # Schedule info
    last_game_date: Optional[str] = None
    is_back_to_back: bool = False
    days_rest: int = 1  # Days since last game (0=B2B, 1=normal, 2+=extra rest)


@dataclass
class GameContext:
    """Full context for a game"""
    game_id: int
    home_team: TeamProfile
    away_team: TeamProfile
    game_time: datetime
    spread: float = 0.0  # positive = home favored
    total: float = 220.0

    # All player props for this game
    props: List[PlayerProp] = field(default_factory=list)

    # Player profiles for both teams
    players: Dict[str, PlayerProfile] = field(default_factory=dict)

    # Injury report
    injuries: List[Dict] = field(default_factory=list)


# =============================================================================
# API CLIENT
# =============================================================================

class BallDontLieClient:
    """
    Client for Balldontlie GOAT API

    Endpoints used:
    - /v1/players - Player search and info
    - /v1/stats - Game logs
    - /v1/season_averages - Season stats
    - /v1/season_averages?type=usage - Usage rates
    - /v1/stats/advanced - Pace, ratings
    - /v1/player_injuries - Injury report
    - /v1/games - Schedule
    - /v1/box_scores - Full box scores
    - /v2/odds/player_props - LIVE DraftKings lines
    """

    BASE_URL_V1 = "https://api.balldontlie.io/v1"
    BASE_URL_V2 = "https://api.balldontlie.io/v2"

    # Team name to ID mapping
    TEAM_IDS = {
        "Atlanta Hawks": 1, "Boston Celtics": 2, "Brooklyn Nets": 3,
        "Charlotte Hornets": 4, "Chicago Bulls": 5, "Cleveland Cavaliers": 6,
        "Dallas Mavericks": 7, "Denver Nuggets": 8, "Detroit Pistons": 9,
        "Golden State Warriors": 10, "Houston Rockets": 11, "Indiana Pacers": 12,
        "LA Clippers": 13, "Los Angeles Lakers": 14, "Memphis Grizzlies": 15,
        "Miami Heat": 16, "Milwaukee Bucks": 17, "Minnesota Timberwolves": 18,
        "New Orleans Pelicans": 19, "New York Knicks": 20, "Oklahoma City Thunder": 21,
        "Orlando Magic": 22, "Philadelphia 76ers": 23, "Phoenix Suns": 24,
        "Portland Trail Blazers": 25, "Sacramento Kings": 26, "San Antonio Spurs": 27,
        "Toronto Raptors": 28, "Utah Jazz": 29, "Washington Wizards": 30,
        # Abbreviation mappings
        "ATL": 1, "BOS": 2, "BKN": 3, "CHA": 4, "CHI": 5, "CLE": 6,
        "DAL": 7, "DEN": 8, "DET": 9, "GSW": 10, "HOU": 11, "IND": 12,
        "LAC": 13, "LAL": 14, "MEM": 15, "MIA": 16, "MIL": 17, "MIN": 18,
        "NOP": 19, "NYK": 20, "OKC": 21, "ORL": 22, "PHI": 23, "PHX": 24,
        "POR": 25, "SAC": 26, "SAS": 27, "TOR": 28, "UTA": 29, "WAS": 30
    }

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {"Authorization": api_key}
        self._cache: Dict[str, Tuple[datetime, any]] = {}
        self._cache_ttl = timedelta(minutes=5)

    def _make_request(self, endpoint: str, params: Dict = None, version: int = 1, raw_url: str = None) -> Optional[Dict]:
        """Make authenticated request with caching"""
        base_url = self.BASE_URL_V1 if version == 1 else self.BASE_URL_V2
        cache_key = f"{endpoint}:{str(params)}:{raw_url}"

        # Check cache
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if datetime.now() - cached_time < self._cache_ttl:
                return cached_data

        try:
            # Use raw_url if provided (for URLs with special chars like [])
            if raw_url:
                url = raw_url
            else:
                url = f"{base_url}/{endpoint}"

            response = requests.get(
                url,
                headers=self.headers,
                params=params or {} if not raw_url else None,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            # Cache successful responses
            self._cache[cache_key] = (datetime.now(), data)
            return data

        except requests.RequestException as e:
            logger.error(f"API request failed: {endpoint} - {e}")
            return None

    # -------------------------------------------------------------------------
    # Player Data
    # -------------------------------------------------------------------------

    def _cache_player_result(self, player: Dict, search_name: str):
        """Cache a player search result"""
        if CACHE_ENABLED and player:
            cache = get_cache()
            player_id = player.get("id")
            if player_id:
                cache.set_player(player_id, player, search_name)

    def search_player(self, name: str) -> Optional[Dict]:
        """Search for a player by name (with SQLite caching)"""
        original_name = name.strip()
        search_name_lower = original_name.lower()
        parts = original_name.split()

        # Check SQLite cache first
        if CACHE_ENABLED:
            cache = get_cache()
            cached = cache.get_player_by_name(original_name)
            if cached is not None:
                return cached

        # Try full name first
        data = self._make_request("players", {"search": original_name})
        if data and data.get("data"):
            # Try to find exact match first
            for player in data["data"]:
                full_name = f"{player.get('first_name', '')} {player.get('last_name', '')}".lower()
                if search_name_lower == full_name:
                    return player
            # Try partial match
            for player in data["data"]:
                full_name = f"{player.get('first_name', '')} {player.get('last_name', '')}".lower()
                if search_name_lower in full_name or full_name in search_name_lower:
                    return player
            return data["data"][0]

        # Full name search failed - try LAST NAME search (works better for stars like "Curry", "James")
        if len(parts) >= 2:
            first_name = parts[0].lower()
            last_name = parts[-1]  # Use last word as last name
            expected_last = ' '.join(parts[1:]).lower()  # Full last name including Jr., etc.

            # Search by last name (more reliable for star players)
            data = self._make_request("players", {"search": last_name})
            if data and data.get("data"):
                # Find exact first+last match
                for player in data["data"]:
                    player_first = player.get("first_name", "").lower()
                    player_last = player.get("last_name", "").lower()
                    if player_first == first_name and player_last == expected_last:
                        return player
                # Try matching first name only
                for player in data["data"]:
                    player_first = player.get("first_name", "").lower()
                    if player_first == first_name:
                        return player
                # Try partial first name match
                for player in data["data"]:
                    player_first = player.get("first_name", "").lower()
                    if first_name in player_first or player_first in first_name:
                        return player

            # If last name search failed, try first name search
            data = self._make_request("players", {"search": first_name})
            if data and data.get("data"):
                # Find exact last name match
                for player in data["data"]:
                    player_last = player.get("last_name", "").lower()
                    if player_last == expected_last:
                        return player
                # Try partial match on last name
                for player in data["data"]:
                    player_last = player.get("last_name", "").lower()
                    if expected_last in player_last or player_last in expected_last:
                        return player
                # Fall back to starts-with match
                base_last = parts[1].lower()
                for player in data["data"]:
                    if player.get("last_name", "").lower().startswith(base_last[:3]):
                        return player

        # If still not found, strip suffix and try again
        suffixes = ['Jr.', 'Jr', 'Sr.', 'Sr', 'II', 'III', 'IV', 'V']
        clean_name = original_name
        for suffix in suffixes:
            if clean_name.endswith(' ' + suffix):
                clean_name = clean_name[:-len(suffix)-1].strip()
                break

        if clean_name != original_name:
            data = self._make_request("players", {"search": clean_name})
            if data and data.get("data"):
                for player in data["data"]:
                    full_name = f"{player.get('first_name', '')} {player.get('last_name', '')}".lower()
                    if clean_name.lower() in full_name or full_name in clean_name.lower():
                        return player
                return data["data"][0]

        return None

    def get_player_by_id(self, player_id: int) -> Optional[Dict]:
        """Get player info by ID"""
        data = self._make_request(f"players/{player_id}")
        return data.get("data") if data else None

    def get_team_players(self, team_id: int) -> List[Dict]:
        """Get all players on a team"""
        data = self._make_request("players", {
            "team_ids[]": team_id,
            "per_page": 30
        })
        return data.get("data", []) if data else []

    # -------------------------------------------------------------------------
    # Game Logs & Stats
    # -------------------------------------------------------------------------

    def get_player_game_logs(
        self,
        player_id: int,
        season: int = 2025,
        last_n: int = 20
    ) -> List[Dict]:
        """Fetch recent game logs for a player (with SQLite caching)"""
        # Check SQLite cache first
        if CACHE_ENABLED:
            cache = get_cache()
            cached_count = cache.get_game_logs_count(player_id)
            if cached_count >= last_n:
                cached_logs = cache.get_game_logs(player_id, last_n)
                if cached_logs:
                    return cached_logs

        # Request more games to ensure we get recent ones (API returns oldest first)
        data = self._make_request("stats", {
            "player_ids[]": player_id,
            "seasons[]": season,
            "per_page": 100  # Fetch full season, then take most recent
        })

        if not data or not data.get("data"):
            return []

        # Sort by date descending to get most recent games
        logs = data["data"]
        logs.sort(key=lambda x: x.get("game", {}).get("date", ""), reverse=True)
        result = logs[:last_n]

        # Cache the game logs
        if CACHE_ENABLED and result:
            cache = get_cache()
            cache.set_game_logs(player_id, logs)  # Cache all logs, not just last_n

        return result

    def get_season_averages(self, player_id: int, season: int = 2025) -> Optional[Dict]:
        """Get season averages for a player (with SQLite caching)"""
        # Check SQLite cache first
        if CACHE_ENABLED:
            cache = get_cache()
            cached = cache.get_season_average(player_id, season)
            if cached is not None:
                return cached

        data = self._make_request("season_averages", {
            "season": season,
            "player_id": player_id  # API uses singular player_id, not player_ids[]
        })
        if data and data.get("data"):
            result = data["data"][0]
            # Cache the result
            if CACHE_ENABLED:
                cache = get_cache()
                cache.set_season_average(player_id, result, season)
            return result
        return None

    def get_usage_stats(self, player_id: int, season: int = 2025) -> Optional[Dict]:
        """Get usage rate and advanced averages"""
        data = self._make_request("season_averages", {
            "season": season,
            "player_id": player_id,  # API uses singular player_id
            "type": "usage"
        })
        if data and data.get("data"):
            return data["data"][0]
        return None

    def get_advanced_stats(self, player_id: int, season: int = 2025) -> List[Dict]:
        """Get advanced stats per game (pace, ratings)"""
        data = self._make_request("stats/advanced", {
            "player_ids[]": player_id,
            "seasons[]": season,
            "per_page": 20
        })
        return data.get("data", []) if data else []

    # -------------------------------------------------------------------------
    # Team & Schedule Data
    # -------------------------------------------------------------------------

    def get_team(self, team_id: int) -> Optional[Dict]:
        """Get team info"""
        data = self._make_request(f"teams/{team_id}")
        return data.get("data") if data else None

    def get_team_schedule(
        self,
        team_id: int,
        start_date: str = None,
        end_date: str = None
    ) -> List[Dict]:
        """Get team's game schedule"""
        params = {"team_ids[]": team_id, "per_page": 20}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        data = self._make_request("games", params)
        return data.get("data", []) if data else []

    def get_todays_games(self) -> List[Dict]:
        """Get all games for today"""
        today = datetime.now().strftime("%Y-%m-%d")
        data = self._make_request("games", {
            "start_date": today,
            "end_date": today
        })
        return data.get("data", []) if data else []

    def get_box_score(self, game_id: int) -> Optional[Dict]:
        """Get full box score for a game"""
        today = datetime.now().strftime("%Y-%m-%d")
        data = self._make_request("box_scores", {"game_ids[]": game_id})
        if data and data.get("data"):
            return data["data"][0] if data["data"] else None
        return None

    # -------------------------------------------------------------------------
    # Injuries
    # -------------------------------------------------------------------------

    def get_injuries(self, team_id: int = None) -> List[Dict]:
        """Get current injury report (with SQLite caching)"""
        # Check SQLite cache first
        if CACHE_ENABLED and team_id:
            cache = get_cache()
            cached = cache.get_injuries(team_id)
            if cached is not None:
                return cached

        params = {}
        if team_id:
            params["team_ids[]"] = team_id
        data = self._make_request("player_injuries", params)
        result = data.get("data", []) if data else []

        # Cache the result
        if CACHE_ENABLED and team_id and result:
            cache = get_cache()
            cache.set_injuries(team_id, result)

        return result

    # -------------------------------------------------------------------------
    # Live DraftKings Props (GOAT feature)
    # -------------------------------------------------------------------------

    def get_todays_odds(self, date: str = None) -> List[Dict]:
        """
        Get all game odds for a specific date

        Returns odds from all vendors (filter for draftkings)
        Also provides game_ids needed for player props
        Handles pagination to get all results
        """
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        all_odds = []
        cursor = None

        # Fetch all pages
        while True:
            # Build URL with optional cursor for pagination
            if cursor:
                raw_url = f"{self.BASE_URL_V2}/odds?dates[]={date}&cursor={cursor}&per_page=100"
            else:
                raw_url = f"{self.BASE_URL_V2}/odds?dates[]={date}&per_page=100"

            data = self._make_request("odds", {}, version=2, raw_url=raw_url)

            if not data or not data.get("data"):
                break

            all_odds.extend(data["data"])

            # Check for more pages
            meta = data.get("meta", {})
            cursor = meta.get("next_cursor")
            if not cursor:
                break

        return all_odds

    def get_draftkings_game_odds(self, date: str = None) -> Dict[int, Dict]:
        """
        Get DraftKings spreads and totals for all games on a date

        Returns: {game_id: {"spread": float, "total": float, "home_team": str}}
        """
        all_odds = self.get_todays_odds(date)

        dk_odds = {}
        for odds in all_odds:
            if odds.get("vendor") == "draftkings":
                game_id = odds.get("game_id")
                dk_odds[game_id] = {
                    "spread": float(odds.get("spread_home_value") or 0),
                    "total": float(odds.get("total_value") or 220),
                    "spread_home_odds": odds.get("spread_home_odds", -110),
                    "spread_away_odds": odds.get("spread_away_odds", -110),
                    "total_over_odds": odds.get("total_over_odds", -110),
                    "total_under_odds": odds.get("total_under_odds", -110),
                }

        return dk_odds

    def get_player_props(
        self,
        game_id: int,
        vendor: str = "draftkings"
    ) -> List[Dict]:
        """
        Get LIVE player props for a specific game

        Filters for over_under market type (excludes milestones)
        Returns props with player_id, prop_type, line, over_odds, under_odds
        """
        # Use raw URL to avoid encoding issues with []
        raw_url = f"{self.BASE_URL_V2}/odds/player_props?game_id={game_id}&vendors[]={vendor}"
        data = self._make_request(
            "odds/player_props",
            {},
            version=2,
            raw_url=raw_url
        )

        if not data or not data.get("data"):
            return []

        # Filter for over_under props only (not milestones)
        props = []
        for prop in data["data"]:
            market = prop.get("market", {})
            if market.get("type") == "over_under":
                props.append({
                    "player_id": prop.get("player_id"),
                    "prop_type": prop.get("prop_type"),
                    "line": float(prop.get("line_value", 0)),
                    "over_odds": market.get("over_odds", -110),
                    "under_odds": market.get("under_odds", -110),
                    "updated_at": prop.get("updated_at")
                })

        return props

    def get_game_odds(self, game_id: int) -> Optional[Dict]:
        """Get game lines (spread, total) for a specific game from DraftKings"""
        # Get all odds for today and filter for this game
        all_odds = self.get_todays_odds()

        for odds in all_odds:
            if odds.get("game_id") == game_id and odds.get("vendor") == "draftkings":
                return {
                    "spread": float(odds.get("spread_home_value") or 0),
                    "total": float(odds.get("total_value") or 220),
                }

        return None


# =============================================================================
# DATA PIPELINE - ORCHESTRATOR
# =============================================================================

class DataPipeline:
    """
    Main orchestrator that builds complete game context

    Fetches and combines:
    - Player profiles with stats, variance, splits
    - Team profiles with pace, defense
    - Live props from DraftKings
    - Injuries and schedule (back-to-back)
    """

    STAT_KEYS = ["pts", "reb", "ast", "stl", "blk", "fg3m", "turnover"]

    def __init__(self, api_key: str):
        self.client = BallDontLieClient(api_key)
        # NBA season is referred to by the year it starts (2025-26 season = 2025)
        self.current_season = 2025

        # Caches
        self._player_cache: Dict[int, PlayerProfile] = {}
        self._team_cache: Dict[int, TeamProfile] = {}

    # -------------------------------------------------------------------------
    # Build Player Profile
    # -------------------------------------------------------------------------

    def build_player_profile(self, player_name: str) -> Optional[PlayerProfile]:
        """
        Build complete player profile with:
        - Season averages
        - Rolling averages (L5, L10)
        - Standard deviations
        - Per-minute rates
        - Home/away splits
        - Usage rate
        """
        # Search for player
        player_data = self.client.search_player(player_name)
        if not player_data:
            logger.warning(f"Player not found: {player_name}")
            return None

        player_id = player_data["id"]

        # Cache player data for future lookups
        self.client._cache_player_result(player_data, player_name)

        # Check cache
        if player_id in self._player_cache:
            return self._player_cache[player_id]

        # Create base profile
        profile = PlayerProfile(
            id=player_id,
            name=f"{player_data['first_name']} {player_data['last_name']}",
            team=player_data.get("team", {}).get("full_name", ""),
            team_id=player_data.get("team", {}).get("id", 0),
            position=player_data.get("position", "")
        )

        # Fetch game logs
        raw_logs = self.client.get_player_game_logs(player_id, self.current_season, 20)
        if not raw_logs:
            logger.warning(f"No game logs for {player_name}")
            return profile

        # Convert to GameLog objects
        game_logs = []
        for log in raw_logs:
            game_info = log.get("game", {})
            home_team_id = game_info.get("home_team_id")
            player_team_id = log.get("team", {}).get("id")

            # Parse minutes (comes as "MM:SS" string)
            min_str = log.get("min", "0:00") or "0:00"
            try:
                if ":" in min_str:
                    parts = min_str.split(":")
                    minutes = int(parts[0]) + int(parts[1]) / 60
                else:
                    minutes = float(min_str)
            except:
                minutes = 0.0

            # Determine opponent
            if player_team_id == home_team_id:
                opponent_id = game_info.get("visitor_team_id")
                is_home = True
            else:
                opponent_id = home_team_id
                is_home = False

            game_logs.append(GameLog(
                game_id=game_info.get("id", 0),
                date=game_info.get("date", ""),
                opponent=str(opponent_id),
                home=is_home,
                minutes=minutes,
                pts=log.get("pts", 0) or 0,
                reb=log.get("reb", 0) or 0,
                ast=log.get("ast", 0) or 0,
                stl=log.get("stl", 0) or 0,
                blk=log.get("blk", 0) or 0,
                fg3m=log.get("fg3m", 0) or 0,
                turnover=log.get("turnover", 0) or 0,
                fga=log.get("fga", 0) or 0,
                fgm=log.get("fgm", 0) or 0,
                fta=log.get("fta", 0) or 0,
                ftm=log.get("ftm", 0) or 0
            ))

        profile.game_logs = game_logs

        # Filter out DNP games (0 minutes) for averaging
        played_games = [g for g in game_logs if g.minutes > 0]

        # Calculate rolling averages (only games where player actually played)
        profile.last_5_avg = self._calculate_averages(played_games[:5])
        profile.last_10_avg = self._calculate_averages(played_games[:10])

        # Calculate standard deviations (only played games)
        profile.std_dev = self._calculate_std_dev(played_games[:15])

        # Calculate per-minute rates (only played games)
        profile.per_minute = self._calculate_per_minute(played_games[:10])

        # Calculate home/away splits (only played games)
        home_games = [g for g in played_games if g.home]
        away_games = [g for g in played_games if not g.home]
        profile.home_avg = self._calculate_averages(home_games[:10])
        profile.away_avg = self._calculate_averages(away_games[:10])

        # Get season averages
        season_avg = self.client.get_season_averages(player_id, self.current_season)
        if season_avg:
            profile.season_avg = {
                "pts": season_avg.get("pts", 0) or 0,
                "reb": season_avg.get("reb", 0) or 0,
                "ast": season_avg.get("ast", 0) or 0,
                "stl": season_avg.get("stl", 0) or 0,
                "blk": season_avg.get("blk", 0) or 0,
                "fg3m": season_avg.get("fg3m", 0) or 0,
                "turnover": season_avg.get("turnover", 0) or 0,
                "min": float(season_avg.get("min", "0").split(":")[0]) if season_avg.get("min") else 0,
                "games_played": season_avg.get("games_played", 0) or 0
            }

        # Get usage rate
        usage_stats = self.client.get_usage_stats(player_id, self.current_season)
        if usage_stats:
            profile.usage_pct = usage_stats.get("usg_pct", 0) or 0

        # Check injury status
        injuries = self.client.get_injuries(profile.team_id)
        for injury in injuries:
            if injury.get("player", {}).get("id") == player_id:
                profile.status = injury.get("status", "active").lower()
                profile.injury_note = injury.get("comment", "")
                break

        # Cache and return
        self._player_cache[player_id] = profile
        return profile

    def _calculate_averages(self, logs: List[GameLog]) -> Dict[str, float]:
        """Calculate average stats from game logs"""
        if not logs:
            return {}

        totals = {stat: 0.0 for stat in self.STAT_KEYS}
        totals["min"] = 0.0

        for log in logs:
            totals["pts"] += log.pts
            totals["reb"] += log.reb
            totals["ast"] += log.ast
            totals["stl"] += log.stl
            totals["blk"] += log.blk
            totals["fg3m"] += log.fg3m
            totals["turnover"] += log.turnover
            totals["min"] += log.minutes

        n = len(logs)
        return {k: round(v / n, 2) for k, v in totals.items()}

    def _calculate_std_dev(self, logs: List[GameLog]) -> Dict[str, float]:
        """Calculate standard deviation for each stat"""
        if len(logs) < 3:
            return {}

        stats = {stat: [] for stat in self.STAT_KEYS}
        stats["min"] = []

        for log in logs:
            stats["pts"].append(log.pts)
            stats["reb"].append(log.reb)
            stats["ast"].append(log.ast)
            stats["stl"].append(log.stl)
            stats["blk"].append(log.blk)
            stats["fg3m"].append(log.fg3m)
            stats["turnover"].append(log.turnover)
            stats["min"].append(log.minutes)

        # Use sample std_dev (ddof=1) for unbiased estimate
        # Population std (ddof=0) underestimates true variance
        return {k: round(float(np.std(v, ddof=1)), 2) for k, v in stats.items() if len(v) > 1}

    def _calculate_per_minute(self, logs: List[GameLog]) -> Dict[str, float]:
        """Calculate per-minute rates"""
        if not logs:
            return {}

        totals = {stat: 0.0 for stat in self.STAT_KEYS}
        total_minutes = 0.0

        for log in logs:
            if log.minutes > 0:
                totals["pts"] += log.pts
                totals["reb"] += log.reb
                totals["ast"] += log.ast
                totals["stl"] += log.stl
                totals["blk"] += log.blk
                totals["fg3m"] += log.fg3m
                totals["turnover"] += log.turnover
                total_minutes += log.minutes

        if total_minutes == 0:
            return {}

        return {k: round(v / total_minutes, 3) for k, v in totals.items()}

    # -------------------------------------------------------------------------
    # Build Team Profile
    # -------------------------------------------------------------------------

    def build_team_profile(self, team_name: str) -> Optional[TeamProfile]:
        """
        Build team profile with:
        - Pace and efficiency ratings
        - Position defense multipliers
        - Back-to-back status
        """
        team_id = self.client.TEAM_IDS.get(team_name)
        if not team_id:
            logger.warning(f"Team not found: {team_name}")
            return None

        # Check cache
        if team_id in self._team_cache:
            return self._team_cache[team_id]

        team_data = self.client.get_team(team_id)
        if not team_data:
            return None

        profile = TeamProfile(
            id=team_id,
            name=team_data.get("full_name", ""),
            abbreviation=team_data.get("abbreviation", "")
        )

        # Get recent games to calculate pace/ratings
        # and check for back-to-back
        today = datetime.now()
        yesterday = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        week_ago = (today - timedelta(days=7)).strftime("%Y-%m-%d")

        schedule = self.client.get_team_schedule(
            team_id,
            start_date=week_ago,
            end_date=today.strftime("%Y-%m-%d")
        )

        if schedule:
            # Find most recent game and calculate days rest
            recent_games = []
            for game in schedule:
                game_date_str = game.get("date", "")[:10]
                if game_date_str and game_date_str < today.strftime("%Y-%m-%d"):
                    recent_games.append(game_date_str)

            if recent_games:
                # Sort to get most recent
                recent_games.sort(reverse=True)
                last_game = recent_games[0]
                profile.last_game_date = last_game

                # Calculate days since last game
                last_game_dt = datetime.strptime(last_game, "%Y-%m-%d")
                days_diff = (today - last_game_dt).days
                profile.days_rest = days_diff

                # B2B is 1 day rest (played yesterday)
                if days_diff == 1:
                    profile.is_back_to_back = True

        # Calculate team pace/defense from box scores
        # (In production, could use team-level advanced stats)
        team_players = self.client.get_team_players(team_id)
        if team_players:
            # Sample a few players to estimate team pace and defensive rating
            pace_samples = []
            def_rating_samples = []
            off_rating_samples = []

            for player in team_players[:5]:
                adv_stats = self.client.get_advanced_stats(
                    player["id"],
                    self.current_season
                )
                for stat in adv_stats[:3]:
                    if stat.get("pace"):
                        pace_samples.append(stat["pace"])
                    if stat.get("defensive_rating"):
                        def_rating_samples.append(stat["defensive_rating"])
                    if stat.get("offensive_rating"):
                        off_rating_samples.append(stat["offensive_rating"])

            if pace_samples:
                profile.pace = round(np.mean(pace_samples), 1)
            if def_rating_samples:
                profile.defensive_rating = round(np.mean(def_rating_samples), 1)
            if off_rating_samples:
                profile.offensive_rating = round(np.mean(off_rating_samples), 1)

        # Cache and return
        self._team_cache[team_id] = profile
        return profile

    # -------------------------------------------------------------------------
    # Build Full Game Context
    # -------------------------------------------------------------------------

    def build_game_context(
        self,
        home_team: str,
        away_team: str,
        player_names: List[str] = None
    ) -> Optional[GameContext]:
        """
        Build complete game context for betting analysis

        Includes:
        - Team profiles (pace, defense, b2b status)
        - Player profiles (stats, variance, splits)
        - Live DraftKings props
        - Injury report
        """
        # Build team profiles
        home_profile = self.build_team_profile(home_team)
        away_profile = self.build_team_profile(away_team)

        if not home_profile or not away_profile:
            return None

        # Find today's game between these teams
        games = self.client.get_todays_games()
        game_id = None
        game_time = datetime.now()

        for game in games:
            h_team = game.get("home_team", {}).get("full_name", "")
            a_team = game.get("visitor_team", {}).get("full_name", "")
            if (h_team == home_team and a_team == away_team) or \
               (h_team == away_team and a_team == home_team):
                game_id = game.get("id")
                game_time = datetime.fromisoformat(
                    game.get("date", "").replace("Z", "+00:00")
                ) if game.get("date") else datetime.now()
                break

        context = GameContext(
            game_id=game_id or 0,
            home_team=home_profile,
            away_team=away_profile,
            game_time=game_time
        )

        # Get game odds (spread, total) from DraftKings
        if game_id:
            odds = self.client.get_game_odds(game_id)
            if odds:
                context.spread = odds.get("spread", 0)
                context.total = odds.get("total", 220)

        # Build player profiles
        if player_names:
            for name in player_names:
                profile = self.build_player_profile(name)
                if profile:
                    context.players[name] = profile

        # Get injuries for both teams
        home_injuries = self.client.get_injuries(home_profile.id)
        away_injuries = self.client.get_injuries(away_profile.id)
        context.injuries = home_injuries + away_injuries

        # Get live props from DraftKings
        if game_id:
            raw_props = self.client.get_player_props(game_id)

            # Cache for player name lookups
            player_names_cache = {}

            for prop in raw_props:
                player_id = prop.get("player_id")

                # Look up player name if not cached
                if player_id not in player_names_cache:
                    player_data = self.client.get_player_by_id(player_id)
                    if player_data:
                        player_names_cache[player_id] = f"{player_data.get('first_name', '')} {player_data.get('last_name', '')}"
                    else:
                        player_names_cache[player_id] = f"Player {player_id}"

                context.props.append(PlayerProp(
                    player_id=player_id,
                    player_name=player_names_cache[player_id],
                    prop_type=prop.get("prop_type", ""),
                    line=prop.get("line", 0),
                    over_odds=prop.get("over_odds", -110),
                    under_odds=prop.get("under_odds", -110)
                ))

        return context

    def _find_under_odds(self, outcomes: List[Dict]) -> int:
        """Find under odds from outcomes list"""
        for outcome in outcomes:
            if outcome.get("name") == "Under":
                return outcome.get("price", -110)
        return -110

    # -------------------------------------------------------------------------
    # Active Roster Detection (from DraftKings props)
    # -------------------------------------------------------------------------

    def get_active_players_for_game(self, game_id: int) -> List[Dict]:
        """
        Get active players for a game using DraftKings props.

        This is the BEST way to get active rosters - if DK has props on a player,
        they're playing tonight. Much better than using stale team roster data.

        Returns list of {player_id, player_name, team_id, props}
        """
        raw_props = self.client.get_player_props(game_id)

        # Group props by player
        player_props = {}
        for prop in raw_props:
            player_id = prop.get("player_id")
            if player_id not in player_props:
                player_props[player_id] = {
                    "player_id": player_id,
                    "player_name": None,
                    "team_id": None,
                    "props": []
                }
            player_props[player_id]["props"].append(prop)

        # Look up player details
        active_players = []
        for player_id, data in player_props.items():
            player_data = self.client.get_player_by_id(player_id)
            if player_data:
                data["player_name"] = f"{player_data.get('first_name', '')} {player_data.get('last_name', '')}"
                data["team_id"] = player_data.get("team", {}).get("id")
                data["team"] = player_data.get("team", {}).get("full_name", "")
                data["position"] = player_data.get("position", "")
            else:
                data["player_name"] = f"Player {player_id}"

            active_players.append(data)

        # Sort by number of props (more props = more important player)
        active_players.sort(key=lambda x: len(x["props"]), reverse=True)

        return active_players

    def get_active_player_names_for_game(self, game_id: int) -> List[str]:
        """Get just the names of active players for a game"""
        active = self.get_active_players_for_game(game_id)
        return [p["player_name"] for p in active if p["player_name"]]

    # -------------------------------------------------------------------------
    # Quick Access Methods
    # -------------------------------------------------------------------------

    def get_player_props_for_game(
        self,
        home_team: str,
        away_team: str
    ) -> List[PlayerProp]:
        """Get all player props for a specific game"""
        context = self.build_game_context(home_team, away_team)
        return context.props if context else []

    def get_injuries_for_teams(self, team_names: List[str]) -> List[Dict]:
        """Get injuries for specific teams"""
        all_injuries = []
        for team_name in team_names:
            team_id = self.client.TEAM_IDS.get(team_name)
            if team_id:
                injuries = self.client.get_injuries(team_id)
                all_injuries.extend(injuries)
        return all_injuries

    def is_back_to_back(self, team_name: str) -> bool:
        """Check if team is on back-to-back"""
        profile = self.build_team_profile(team_name)
        return profile.is_back_to_back if profile else False

    def clear_cache(self):
        """Clear all caches"""
        self._player_cache.clear()
        self._team_cache.clear()
        self.client._cache.clear()


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    API_KEY = os.getenv("BALLDONTLIE_API_KEY")
    if not API_KEY:
        print("Error: BALLDONTLIE_API_KEY not found in environment")
        exit(1)

    pipeline = DataPipeline(API_KEY)

    # Example: Build player profile
    print("\n=== Player Profile ===")
    profile = pipeline.build_player_profile("Jaylen Brown")
    if profile:
        print(f"Name: {profile.name}")
        print(f"Team: {profile.team}")
        print(f"Position: {profile.position}")
        print(f"Usage %: {profile.usage_pct}")
        print(f"Last 5 avg: {profile.last_5_avg}")
        print(f"Last 10 avg: {profile.last_10_avg}")
        print(f"Std Dev: {profile.std_dev}")
        print(f"Per Minute: {profile.per_minute}")
        print(f"Home avg: {profile.home_avg}")
        print(f"Away avg: {profile.away_avg}")
        print(f"Status: {profile.status}")

    # Example: Check today's games
    print("\n=== Today's Games ===")
    games = pipeline.client.get_todays_games()
    for game in games:
        home = game.get("home_team", {}).get("full_name", "")
        away = game.get("visitor_team", {}).get("full_name", "")
        print(f"  {away} @ {home}")
