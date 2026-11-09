"""
data_pipeline.py - Comprehensive NBA Data Pipeline

Uses Balldontlie GOAT API for:
- Player game logs and season averages
- Usage rates and advanced stats
- Injury reports
- Team schedules (for back-to-back detection)
- Box scores and pace data

Uses The Odds API for:
- Live DraftKings player props (primary source)
- Real PRA lines (not synthetic)

Falls back to Balldontlie for props if The Odds API key not set.

Now with SQLite caching for faster performance!
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import requests
import numpy as np
import pandas as pd

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required, can use env vars directly

# NBA.com tracking stats via nba_api
try:
    from nba_api.stats.endpoints import (
        leaguehustlestatsplayer,
        leaguedashptstats,
        playerestimatedmetrics,
    )
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False
    logging.warning("nba_api not installed - tracking stats unavailable")

# Cleaning the Glass CSV support
import glob as glob_module  # For finding CSV files
CTG_CSV_AVAILABLE = True  # We always have CSV support via pandas

# Import cache (optional - will work without it)
try:
    from cache_db import get_cache, CacheDB
    CACHE_ENABLED = True
except ImportError:
    CACHE_ENABLED = False
    get_cache = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The Odds API key (for live props - primary source)
ODDS_API_KEY = os.environ.get("ODDS_API_KEY")


def get_current_nba_season() -> int:
    """Return the current NBA season year based on today's date.
    NBA seasons span Oct-June: 2026-27 season -> returns 2026."""
    now = datetime.now()
    if now.month >= 10:
        return now.year
    else:
        return now.year - 1


def get_current_nba_season_str() -> str:
    """Return season string like '2026-27'."""
    year = get_current_nba_season()
    return f"{year}-{str(year + 1)[-2:]}"


# =============================================================================
# THE ODDS API CLIENT (Live Props)
# =============================================================================

class OddsAPIClient:
    """
    Client for The Odds API - fetches live DraftKings player props

    This is the PRIMARY source for props because it includes:
    - Real PRA (pts_rebs_asts) lines
    - More complete prop coverage
    - Accurate odds
    """

    BASE_URL = "https://api.the-odds-api.com/v4"
    SPORT_KEY = "basketball_nba"

    # Market key to our prop_type mapping
    MARKET_MAP = {
        "player_points": "points",
        "player_rebounds": "rebounds",
        "player_assists": "assists",
        "player_threes": "threes",
        "player_points_rebounds_assists": "pts_rebs_asts",
        "player_steals": "steals",
        "player_blocks": "blocks",
        "player_turnovers": "turnovers",
        "player_double_double": "double_double",
    }

    # All player prop markets we want
    PROP_MARKETS = [
        "player_points",
        "player_rebounds",
        "player_assists",
        "player_threes",
        "player_points_rebounds_assists",
        "player_steals",
        "player_blocks",
        "player_turnovers",
    ]

    def __init__(self, api_key: str = None):
        self.api_key = api_key or ODDS_API_KEY
        self._cache: Dict[str, Tuple[datetime, any]] = {}
        self._cache_ttl = timedelta(minutes=2)  # Props change frequently

    def get_events(self) -> List[Dict]:
        """Get all upcoming NBA events"""
        url = f"{self.BASE_URL}/sports/{self.SPORT_KEY}/events"
        params = {"apiKey": self.api_key}

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Odds API events request failed: {e}")
            return []

    def find_event_id(self, home_team: str, away_team: str) -> Optional[str]:
        """Find The Odds API event ID for a game by team names"""
        events = self.get_events()

        # Normalize team names for matching
        def normalize(name: str) -> str:
            return name.lower().replace("la clippers", "los angeles clippers")

        home_norm = normalize(home_team)
        away_norm = normalize(away_team)

        for event in events:
            event_home = normalize(event.get("home_team", ""))
            event_away = normalize(event.get("away_team", ""))

            # Check both directions (API might swap home/away)
            if (home_norm in event_home or event_home in home_norm) and \
               (away_norm in event_away or event_away in away_norm):
                return event.get("id")
            if (away_norm in event_home or event_home in away_norm) and \
               (home_norm in event_away or event_away in home_norm):
                return event.get("id")

        return None

    def get_player_props(self, event_id: str, bookmaker: str = "draftkings") -> List[Dict]:
        """
        Get live player props for a specific event

        Returns list of dicts with: player_name, prop_type, line, over_odds, under_odds
        """
        cache_key = f"props:{event_id}:{bookmaker}"

        # Check cache
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if datetime.now() - cached_time < self._cache_ttl:
                return cached_data

        url = f"{self.BASE_URL}/sports/{self.SPORT_KEY}/events/{event_id}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": ",".join(self.PROP_MARKETS),
            "bookmakers": bookmaker,
            "oddsFormat": "american",
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            logger.error(f"Odds API props request failed: {e}")
            return []

        # Parse the response into our format
        props = self._parse_props(data)

        # Cache the result
        self._cache[cache_key] = (datetime.now(), props)

        logger.info(f"Fetched {len(props)} props from The Odds API for event {event_id}")
        return props

    def _parse_props(self, data: Dict) -> List[Dict]:
        """Parse The Odds API response into our prop format"""
        props = []

        bookmakers = data.get("bookmakers", [])
        if not bookmakers:
            return props

        for bookmaker in bookmakers:
            for market in bookmaker.get("markets", []):
                market_key = market.get("key", "")
                prop_type = self.MARKET_MAP.get(market_key)

                if not prop_type:
                    continue

                # Group outcomes by player (Over/Under pairs)
                player_lines = {}
                for outcome in market.get("outcomes", []):
                    player_name = outcome.get("description", "")
                    direction = outcome.get("name", "")  # "Over" or "Under"
                    odds = outcome.get("price", -110)
                    line = outcome.get("point", 0)

                    if player_name not in player_lines:
                        player_lines[player_name] = {"line": line, "player_name": player_name}

                    if direction == "Over":
                        player_lines[player_name]["over_odds"] = odds
                    elif direction == "Under":
                        player_lines[player_name]["under_odds"] = odds

                # Create prop records for complete pairs
                for player_name, line_data in player_lines.items():
                    if "over_odds" in line_data and "under_odds" in line_data:
                        props.append({
                            "player_name": player_name,
                            "prop_type": prop_type,
                            "line": float(line_data["line"]),
                            "over_odds": line_data["over_odds"],
                            "under_odds": line_data["under_odds"],
                            "source": "odds_api",
                        })

        return props


# =============================================================================
# PBP STATS CLIENT (Free API)
# =============================================================================

class PBPStatsClient:
    """
    Client for PBPStats.com API (free, no auth required)

    PBP Stats provides play-by-play derived advanced stats:
    - Shot zone breakdown (AtRim, ShortMid, LongMid, Corner3, Arc3)
    - Shot quality metrics (ShotQualityAvg)
    - Assisted vs unassisted percentages
    - Turnover breakdown (BadPass, LostBall)
    - Self offensive rebounds (ownmiss)
    - Shooting fouls drawn rate
    - Usage, TS%, eFG%
    """

    BASE_URL = "https://api.pbpstats.com"

    def __init__(self, season: str = None):
        self.season = season or get_current_nba_season_str()
        self._cache: Dict[str, Tuple[datetime, Dict]] = {}
        self._cache_ttl = timedelta(hours=6)
        self._player_cache: Dict[str, Dict] = {}  # name -> stats

    def _get_cached(self, cache_key: str) -> Optional[Dict]:
        if cache_key in self._cache:
            cached_time, data = self._cache[cache_key]
            if datetime.now() - cached_time < self._cache_ttl:
                return data
        return None

    def _fetch(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Fetch data from PBP Stats API"""
        cache_key = f"{endpoint}_{str(params)}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            import time
            time.sleep(0.5)  # Rate limiting
            url = f"{self.BASE_URL}/{endpoint}"
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                self._cache[cache_key] = (datetime.now(), data)
                return data
        except Exception as e:
            logger.warning(f"PBP Stats fetch failed: {e}")
        return None

    def fetch_all_player_totals(self) -> Dict[str, Dict]:
        """
        Fetch totals for all players.
        Returns dict mapping lowercase player name -> stats dict
        """
        if self._player_cache:
            return self._player_cache

        data = self._fetch("get-totals/nba", {
            "Season": self.season,
            "SeasonType": "Regular Season",
            "Type": "Player"
        })

        if not data or 'multi_row_table_data' not in data:
            return {}

        for player in data['multi_row_table_data']:
            name = player.get('Name', '').lower()
            if name:
                self._player_cache[name] = player

        logger.info(f"Loaded PBP Stats for {len(self._player_cache)} players")
        return self._player_cache

    def get_player_stats(self, player_name: str) -> Optional[Dict]:
        """Get PBP Stats for a specific player"""
        all_stats = self.fetch_all_player_totals()
        name_lower = player_name.lower()

        # Exact match
        if name_lower in all_stats:
            return all_stats[name_lower]

        # Partial match
        for stored_name, stats in all_stats.items():
            if name_lower in stored_name or stored_name in name_lower:
                return stats

        return None

    def clear_cache(self):
        self._cache.clear()
        self._player_cache.clear()

    def get_player_pbp_stats(self, player_name: str) -> Optional['PBPStats']:
        """Get PBP Stats as a structured object"""
        raw = self.get_player_stats(player_name)
        if not raw:
            return None

        games = raw.get('GamesPlayed', 1) or 1

        return PBPStats(
            # Shot zone (convert to per-game)
            at_rim_fga=raw.get('AtRimFGA', 0) / games,
            at_rim_accuracy=raw.get('AtRimAccuracy', 0),
            at_rim_freq=raw.get('AtRimFrequency', 0),
            short_mid_fga=raw.get('ShortMidRangeFGA', 0) / games,
            short_mid_accuracy=raw.get('ShortMidRangeAccuracy', 0),
            long_mid_fga=raw.get('LongMidRangeFGA', 0) / games,
            long_mid_accuracy=raw.get('LongMidRangeAccuracy', 0),
            corner3_fga=raw.get('Corner3FGA', 0) / games,
            corner3_accuracy=raw.get('Corner3Accuracy', 0),
            arc3_fga=raw.get('Arc3FGA', 0) / games,
            arc3_accuracy=raw.get('Arc3Accuracy', 0),

            # Shot creation
            assisted_2s_pct=raw.get('Assisted2sPct', 0),
            assisted_3s_pct=raw.get('Assisted3sPct', 0),
            shot_quality_avg=raw.get('ShotQualityAvg', 0),

            # Efficiency
            ts_pct=raw.get('TsPct', 0),
            efg_pct=raw.get('EfgPct', 0),
            usage=raw.get('Usage', 0),

            # Turnovers (per-game)
            bad_pass_turnovers=raw.get('BadPassTurnovers', 0) / games,
            lost_ball_turnovers=raw.get('LostBallTurnovers', 0) / games,
            live_ball_tov_pct=raw.get('LiveBallTurnoverPct', 0),

            # Rebounding
            self_oreb=raw.get('SelfOReb', 0) / games,
            off_reb_pct=raw.get('OffFGReboundPct', 0),
            def_reb_pct=raw.get('DefFGReboundPct', 0),

            # Fouls
            shooting_fouls_drawn_pct=raw.get('ShootingFoulsDrawnPct', 0),
            fouls_drawn=raw.get('FoulsDrawn', 0) / games,
            offensive_fouls=raw.get('Offensive Fouls', 0) / games,

            # Blocks
            blocks=raw.get('Blocks', 0) / games,
            blocked_2s=raw.get('Blocked2s', 0) / games,
            fg2a_pct_blocked=raw.get('FG2APctBlocked', 0),

            # Steals
            bad_pass_steals=raw.get('BadPassSteals', 0) / games,
            lost_ball_steals=raw.get('LostBallSteals', 0) / games,

            # On/Off
            on_off_rtg=raw.get('OnOffRtg', 0),
            on_def_rtg=raw.get('OnDefRtg', 0),

            # Assists breakdown (per-game)
            at_rim_assists=raw.get('AtRimAssists', 0) / games,
            three_pt_assists=raw.get('ThreePtAssists', 0) / games,
            assist_points=raw.get('AssistPoints', 0) / games,
        )


# =============================================================================
# CLEANING THE GLASS CLIENT (CSV-based)
# =============================================================================

class CleaningTheGlassClient:
    """
    Client for Cleaning the Glass (cleaningtheglass.com)

    Loads data from downloaded CSV files. User should download the CSV from
    CTG website and place it in the project directory.

    Expected filename pattern: players_offensive_overview_*.csv

    Key stats provided:
    - Percentile rankings for all stats (league context)
    - PSA (Points per Shot Attempt = TS% * 2)
    - Usage rate and percentile
    - Assist % and percentile
    - Turnover % and percentile (lower is better)
    """

    def __init__(self, csv_dir: str = None):
        """
        Initialize CTG client with CSV directory.

        Args:
            csv_dir: Directory containing CTG CSV files. Defaults to current directory.
        """
        self.csv_dir = csv_dir or os.path.dirname(__file__)
        self._cache: Dict[str, 'CTGStats'] = {}
        self._loaded = False

    def _find_latest_csv(self) -> Optional[str]:
        """Find the most recent CTG CSV file in the directory."""
        patterns = [
            os.path.join(self.csv_dir, "players_offensive_overview_*.csv"),
            os.path.join(self.csv_dir, "*ctg*.csv"),
            os.path.join(self.csv_dir, "*cleaning*glass*.csv"),
        ]

        all_files = []
        for pattern in patterns:
            all_files.extend(glob_module.glob(pattern))

        if not all_files:
            return None

        # Return the most recently modified file
        return max(all_files, key=os.path.getmtime)

    def _load_csv(self) -> bool:
        """Load CTG data from CSV file."""
        if self._loaded and self._cache:
            return True

        csv_path = self._find_latest_csv()
        if not csv_path:
            logger.warning("No CTG CSV file found. Download from cleaningtheglass.com and place in project directory.")
            return False

        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loading CTG data from: {csv_path}")

            # Map CSV columns to CTGStats
            # Expected columns: Player, Age, Team, Pos, MIN, MPG,
            #                   Usage Rank, Usage, PSA Rank, PSA,
            #                   AST% Rank, AST%, AST:Usg Rank, AST:Usg,
            #                   TOV% Rank, TOV%

            for _, row in df.iterrows():
                try:
                    player_name = str(row.get('Player', '')).strip().lower()
                    if not player_name:
                        continue

                    stats = CTGStats(
                        team=str(row.get('Team', '')),
                        position=str(row.get('Pos', '')),
                        minutes_total=float(row.get('MIN', 0)),
                        mpg=float(row.get('MPG', 0)),
                        # Usage
                        usage_rate=self._parse_pct(row.get('Usage', 0)),
                        usage_percentile=int(row.get('Usage Rank', 50)),
                        # Scoring efficiency
                        psa=float(row.get('PSA', 0)),
                        psa_percentile=int(row.get('PSA Rank', 50)),
                        # Assists
                        ast_rate=self._parse_pct(row.get('AST%', 0)),
                        ast_percentile=int(row.get('AST% Rank', 50)),
                        # Assist to usage ratio (playmaking efficiency)
                        ast_to_usage=float(row.get('AST:Usg', 0)),
                        ast_to_usage_percentile=int(row.get('AST:Usg Rank', 50)),
                        # Turnovers (NOTE: higher rank = lower TOV% = better)
                        tov_rate=self._parse_pct(row.get('TOV%', 0)),
                        tov_percentile=int(row.get('TOV% Rank', 50)),
                    )

                    self._cache[player_name] = stats

                except Exception as e:
                    logger.debug(f"Error parsing CTG row for {row.get('Player', 'unknown')}: {e}")
                    continue

            logger.info(f"Loaded CTG data for {len(self._cache)} players")
            self._loaded = True
            return True

        except Exception as e:
            logger.error(f"Failed to load CTG CSV: {e}")
            return False

    def _parse_pct(self, value) -> float:
        """Parse percentage value, handling % suffix."""
        if pd.isna(value):
            return 0.0
        s = str(value).replace('%', '').strip()
        try:
            return float(s)
        except (ValueError, TypeError):
            return 0.0

    def get_player_stats(self, player_name: str) -> Optional['CTGStats']:
        """Get CTG stats for a specific player."""
        self._load_csv()

        name_lower = player_name.lower().strip()

        # Exact match
        if name_lower in self._cache:
            return self._cache[name_lower]

        # Partial match (handle name variations)
        for stored_name, stats in self._cache.items():
            # Check if one name contains the other
            if name_lower in stored_name or stored_name in name_lower:
                return stats

            # Check last name match (for "LeBron James" vs "James, LeBron")
            name_parts = name_lower.split()
            stored_parts = stored_name.split()
            if len(name_parts) >= 2 and len(stored_parts) >= 2:
                if name_parts[-1] == stored_parts[-1] and name_parts[0][0] == stored_parts[0][0]:
                    return stats

        return None

    def fetch_player_stats(self) -> Dict[str, 'CTGStats']:
        """Get all player stats (loads from CSV if needed)."""
        self._load_csv()
        return self._cache

    def clear_cache(self):
        """Clear the in-memory cache."""
        self._cache.clear()
        self._loaded = False

    def reload(self):
        """Force reload data from CSV."""
        self.clear_cache()
        return self._load_csv()


@dataclass
class CTGStats:
    """
    Cleaning the Glass stats (from downloaded CSV)

    Premium analytics with garbage-time filtering and percentile context.
    Percentile rankings provide league context (0-100, higher = better).

    NOTE: For TOV%, higher percentile rank means LOWER turnover rate (better).
    """
    # Player info
    team: str = ""
    position: str = ""  # Big, Wing, Forward, Combo, Point
    minutes_total: float = 0.0
    mpg: float = 0.0

    # Scoring efficiency
    psa: float = 0.0  # Points per Shot Attempt (similar to TS% * 2)
    psa_percentile: int = 50  # League percentile (0-100)

    # Usage (% of team plays used while on court)
    usage_rate: float = 0.0  # As percentage (e.g., 25.0 = 25%)
    usage_percentile: int = 50

    # Playmaking / Assists
    ast_rate: float = 0.0  # AST% - % of teammate FGs assisted
    ast_percentile: int = 50

    # Assist-to-Usage ratio (playmaking efficiency)
    ast_to_usage: float = 0.0  # Higher = more efficient playmaker
    ast_to_usage_percentile: int = 50

    # Turnovers (lower rate = better, but higher percentile = better)
    tov_rate: float = 0.0  # TOV% - turnovers per 100 plays
    tov_percentile: int = 50  # Higher = fewer turnovers = better


@dataclass
class CTGTeamStats:
    """
    Cleaning the Glass TEAM-level stats (from downloaded CSVs)

    These are loaded from the league_* CSV files:
    - league_four_factors_*.csv
    - league_offense_transition_*.csv
    - league_defense_transition_*.csv
    - league_offense_halfcourt_and_putbacks_*.csv
    - league_defense_halfcourt_and_putbacks_*.csv
    """
    team_name: str = ""

    # Four Factors - Offense
    off_pts_poss: float = 115.9       # Pts per 100 possessions
    off_efg_pct: float = 54.6         # Offensive eFG%
    off_efg_rank: int = 15            # eFG% rank (1-30)
    off_tov_pct: float = 14.6         # Turnover %
    off_tov_rank: int = 15
    off_orb_pct: float = 29.2         # Offensive rebound %
    off_orb_rank: int = 15
    off_ft_rate: float = 21.4         # Free throw rate
    off_ft_rank: int = 15

    # Four Factors - Defense
    def_pts_poss: float = 115.9       # Pts allowed per 100 possessions
    def_efg_pct: float = 54.6         # Opponent eFG% allowed
    def_efg_rank: int = 15
    def_tov_pct: float = 14.6         # Forced turnover %
    def_tov_rank: int = 15
    def_orb_pct: float = 29.2         # Opponent ORB% allowed
    def_orb_rank: int = 15
    def_ft_rate: float = 21.4         # Opponent FT rate
    def_ft_rank: int = 15

    # Transition Offense
    transition_freq: float = 15.6     # % of plays in transition
    transition_pts_play: float = 125.3  # Pts per transition play
    transition_rank: int = 15

    # Transition Defense
    def_transition_pts: float = 125.3   # Transition pts allowed
    def_transition_rank: int = 15

    # Halfcourt Offense
    halfcourt_pts_play: float = 98.1    # Pts per halfcourt play
    halfcourt_rank: int = 15

    # Halfcourt Defense
    def_halfcourt_pts: float = 98.1     # Halfcourt pts allowed
    def_halfcourt_rank: int = 15


class CTGTeamClient:
    """
    Client for Cleaning the Glass TEAM-level data (from CSVs)

    Loads data from:
    - league_four_factors_*.csv
    - league_offense_transition_*.csv
    - league_defense_transition_*.csv
    - league_offense_halfcourt_and_putbacks_*.csv
    - league_defense_halfcourt_and_putbacks_*.csv
    """

    def __init__(self, csv_dir: str = None):
        self.csv_dir = csv_dir or os.path.dirname(__file__)
        self._cache: Dict[str, CTGTeamStats] = {}
        self._loaded = False

    def _find_csv(self, pattern: str) -> Optional[str]:
        """Find the most recent CSV matching pattern."""
        search_pattern = os.path.join(self.csv_dir, pattern)
        files = glob_module.glob(search_pattern)
        if not files:
            return None
        return max(files, key=os.path.getmtime)

    def _parse_pct(self, val) -> float:
        """Parse percentage string to float."""
        if pd.isna(val) or val == '':
            return 0.0
        if isinstance(val, str):
            return float(val.replace('%', '').strip())
        return float(val)

    def _load_all(self) -> bool:
        """Load all CTG team CSVs."""
        if self._loaded and self._cache:
            return True

        # Load Four Factors
        ff_path = self._find_csv("league_four_factors_*.csv")
        if ff_path:
            try:
                df = pd.read_csv(ff_path)
                logger.info(f"Loading CTG team four factors from: {ff_path}")

                for _, row in df.iterrows():
                    team = str(row.get('Team', '')).strip()
                    if not team or team == 'Average':
                        continue

                    # Get or create team stats
                    if team not in self._cache:
                        self._cache[team] = CTGTeamStats(team_name=team)

                    stats = self._cache[team]
                    stats.off_pts_poss = float(row.get('OFFENSE: Pts/Poss', 115.9) or 115.9)
                    stats.off_efg_pct = self._parse_pct(row.get('OFFENSE: eFG%', 54.6))
                    stats.off_efg_rank = int(row.get('OFFENSE: eFG% Rank', 15) or 15)
                    stats.off_tov_pct = self._parse_pct(row.get('OFFENSE: TOV%', 14.6))
                    stats.off_tov_rank = int(row.get('OFFENSE: TOV% Rank', 15) or 15)
                    stats.off_orb_pct = self._parse_pct(row.get('OFFENSE: ORB%', 29.2))
                    stats.off_orb_rank = int(row.get('OFFENSE: ORB% Rank', 15) or 15)
                    stats.off_ft_rate = float(row.get('OFFENSE: FT Rate', 21.4) or 21.4)
                    stats.off_ft_rank = int(row.get('OFFENSE: FT Rate Rank', 15) or 15)

                    stats.def_pts_poss = float(row.get('DEFENSE: Pts/Poss', 115.9) or 115.9)
                    stats.def_efg_pct = self._parse_pct(row.get('DEFENSE: eFG%', 54.6))
                    stats.def_efg_rank = int(row.get('DEFENSE: eFG% Rank', 15) or 15)
                    stats.def_tov_pct = self._parse_pct(row.get('DEFENSE: TOV%', 14.6))
                    stats.def_tov_rank = int(row.get('DEFENSE: TOV% Rank', 15) or 15)
                    stats.def_orb_pct = self._parse_pct(row.get('DEFENSE: ORB%', 29.2))
                    stats.def_orb_rank = int(row.get('DEFENSE: ORB% Rank', 15) or 15)
                    stats.def_ft_rate = float(row.get('DEFENSE: FT Rate', 21.4) or 21.4)
                    stats.def_ft_rank = int(row.get('DEFENSE: FT Rate Rank', 15) or 15)

                logger.info(f"Loaded CTG four factors for {len(self._cache)} teams")
            except Exception as e:
                logger.warning(f"Error loading CTG four factors: {e}")

        # Load Transition Offense
        trans_off_path = self._find_csv("league_offense_transition_*.csv")
        if trans_off_path:
            try:
                df = pd.read_csv(trans_off_path)
                for _, row in df.iterrows():
                    team = str(row.get('Team', '')).strip()
                    if team not in self._cache:
                        continue
                    stats = self._cache[team]
                    stats.transition_freq = self._parse_pct(row.get('ALL TRANSITION: Freq', 15.6))
                    stats.transition_pts_play = float(row.get('ALL TRANSITION: Pts/Play', 125.3) or 125.3)
                    stats.transition_rank = int(row.get('ALL TRANSITION: Pts/Play Rank', 15) or 15)
            except Exception as e:
                logger.warning(f"Error loading CTG transition offense: {e}")

        # Load Transition Defense
        trans_def_path = self._find_csv("league_defense_transition_*.csv")
        if trans_def_path:
            try:
                df = pd.read_csv(trans_def_path)
                for _, row in df.iterrows():
                    team = str(row.get('Team', '')).strip()
                    if team not in self._cache:
                        continue
                    stats = self._cache[team]
                    stats.def_transition_pts = float(row.get('ALL TRANSITION: Pts/Play', 125.3) or 125.3)
                    stats.def_transition_rank = int(row.get('ALL TRANSITION: Pts/Play Rank', 15) or 15)
            except Exception as e:
                logger.warning(f"Error loading CTG transition defense: {e}")

        # Load Halfcourt Offense
        hc_off_path = self._find_csv("league_offense_halfcourt_and_putbacks_*.csv")
        if hc_off_path:
            try:
                df = pd.read_csv(hc_off_path)
                for _, row in df.iterrows():
                    team = str(row.get('Team', '')).strip()
                    if team not in self._cache:
                        continue
                    stats = self._cache[team]
                    stats.halfcourt_pts_play = float(row.get('HALFCOURT: Pts/Play', 98.1) or 98.1)
                    stats.halfcourt_rank = int(row.get('HALFCOURT: Pts/Play Rank', 15) or 15)
            except Exception as e:
                logger.warning(f"Error loading CTG halfcourt offense: {e}")

        # Load Halfcourt Defense
        hc_def_path = self._find_csv("league_defense_halfcourt_and_putbacks_*.csv")
        if hc_def_path:
            try:
                df = pd.read_csv(hc_def_path)
                for _, row in df.iterrows():
                    team = str(row.get('Team', '')).strip()
                    if team not in self._cache:
                        continue
                    stats = self._cache[team]
                    stats.def_halfcourt_pts = float(row.get('HALFCOURT: Pts/Play', 98.1) or 98.1)
                    stats.def_halfcourt_rank = int(row.get('HALFCOURT: Pts/Play Rank', 15) or 15)
            except Exception as e:
                logger.warning(f"Error loading CTG halfcourt defense: {e}")

        self._loaded = True
        return len(self._cache) > 0

    def get_team_stats(self, team_name: str) -> Optional[CTGTeamStats]:
        """Get CTG stats for a specific team."""
        self._load_all()

        # Try exact match first
        if team_name in self._cache:
            return self._cache[team_name]

        # Try partial match (e.g., "Boston" matches "Boston Celtics")
        team_lower = team_name.lower()
        for stored_name, stats in self._cache.items():
            if team_lower in stored_name.lower() or stored_name.lower() in team_lower:
                return stats

        return None

    def get_all_teams(self) -> Dict[str, CTGTeamStats]:
        """Get stats for all teams."""
        self._load_all()
        return self._cache

    def clear_cache(self):
        """Clear the cache."""
        self._cache.clear()
        self._loaded = False


@dataclass
class CTGLineupStats:
    """
    Cleaning the Glass LINEUP-level stats (from lineups_* CSVs)

    Represents stats for a specific 5-man lineup.
    """
    team: str = ""
    players: List[str] = field(default_factory=list)  # PG, SG, SF, PF, C
    possessions: int = 0

    # Net rating and point differential
    net_rating: float = 0.0
    net_rating_rank: int = 50

    # Offensive stats
    off_pts_poss: float = 115.9       # Pts per 100 possessions
    off_efg_pct: float = 54.5         # Offensive eFG%
    off_tov_pct: float = 14.6         # Turnover %
    off_orb_pct: float = 29.2         # Offensive rebound %
    off_ft_rate: float = 21.3         # Free throw rate

    # Defensive stats
    def_pts_poss: float = 115.9       # Pts allowed per 100 possessions
    def_efg_pct: float = 54.5         # Opponent eFG%
    def_tov_pct: float = 14.6         # Forced turnover %
    def_orb_pct: float = 29.2         # Opponent ORB%
    def_ft_rate: float = 21.3         # Opponent FT rate


class CTGLineupClient:
    """
    Client for Cleaning the Glass LINEUP-level data (from CSVs)

    Loads data from lineups_four_factors_*.csv and matches by player names.
    """

    def __init__(self, csv_dir: str = None):
        self.csv_dir = csv_dir or os.path.dirname(__file__)
        self._cache: List[CTGLineupStats] = []
        self._loaded = False

    def _find_csv(self, pattern: str) -> Optional[str]:
        """Find the most recent CSV matching pattern."""
        search_pattern = os.path.join(self.csv_dir, pattern)
        files = glob_module.glob(search_pattern)
        if not files:
            return None
        return max(files, key=os.path.getmtime)

    def _parse_pct(self, val) -> float:
        """Parse percentage string to float."""
        if pd.isna(val) or val == '':
            return 0.0
        if isinstance(val, str):
            return float(val.replace('%', '').strip())
        return float(val)

    def _normalize_name(self, name: str) -> str:
        """Normalize player name for matching."""
        if pd.isna(name) or not name:
            return ""
        return str(name).strip().lower()

    def _load_lineups(self) -> bool:
        """Load lineup data from CSV."""
        if self._loaded and self._cache:
            return True

        ff_path = self._find_csv("lineups_four_factors_*.csv")
        if not ff_path:
            logger.warning("No lineups_four_factors CSV found")
            return False

        try:
            df = pd.read_csv(ff_path)
            logger.info(f"Loading CTG lineup data from: {ff_path}")

            for _, row in df.iterrows():
                team = str(row.get('Team', '')).strip()
                if not team or team == 'League Averages':
                    continue

                # Get player names (columns: PG, SG, SF, PF, C)
                players = [
                    self._normalize_name(row.get('PG', '')),
                    self._normalize_name(row.get('SG', '')),
                    self._normalize_name(row.get('SF', '')),
                    self._normalize_name(row.get('PF', '')),
                    self._normalize_name(row.get('C', ''))
                ]

                # Skip if missing players
                if not all(players):
                    continue

                # Parse diff column for net rating
                diff_str = str(row.get('Diff', '+0')).replace('+', '')
                try:
                    net_rating = float(diff_str) if diff_str else 0.0
                except (ValueError, TypeError):
                    net_rating = 0.0

                lineup = CTGLineupStats(
                    team=team,
                    players=players,
                    possessions=int(row.get('Poss', 0) or 0),
                    net_rating=net_rating,
                    net_rating_rank=int(row.get('Diff Rank', 50) or 50),
                    off_pts_poss=float(row.get('OFFENSE: Pts/Poss', 115.9) or 115.9),
                    off_efg_pct=self._parse_pct(row.get('OFFENSE: eFG%', 54.5)),
                    off_tov_pct=self._parse_pct(row.get('OFFENSE: TOV%', 14.6)),
                    off_orb_pct=self._parse_pct(row.get('OFFENSE: ORB%', 29.2)),
                    off_ft_rate=float(row.get('OFFENSE: FT Rate', 21.3) or 21.3),
                    def_pts_poss=float(row.get('DEFENSE: Pts/Poss', 115.9) or 115.9),
                    def_efg_pct=self._parse_pct(row.get('DEFENSE: eFG%', 54.5)),
                    def_tov_pct=self._parse_pct(row.get('DEFENSE: TOV%', 14.6)),
                    def_orb_pct=self._parse_pct(row.get('DEFENSE: ORB%', 29.2)),
                    def_ft_rate=float(row.get('DEFENSE: FT Rate', 21.3) or 21.3),
                )
                self._cache.append(lineup)

            logger.info(f"Loaded {len(self._cache)} CTG lineups")
            self._loaded = True
            return True
        except Exception as e:
            logger.warning(f"Error loading CTG lineups: {e}")
            return False

    def find_lineup(self, player_names: List[str], team: str = None) -> Optional[CTGLineupStats]:
        """
        Find a lineup that contains the given players.

        Args:
            player_names: List of 5 player names (starters)
            team: Optional team name to filter by

        Returns:
            CTGLineupStats if found, None otherwise
        """
        self._load_lineups()

        if not player_names or len(player_names) < 5:
            return None

        # Normalize input names
        input_names = set(self._normalize_name(n) for n in player_names if n)

        best_match = None
        best_score = 0

        for lineup in self._cache:
            # Team filter
            if team and team.lower() not in lineup.team.lower():
                continue

            # Count matching players
            lineup_names = set(lineup.players)
            matches = len(input_names & lineup_names)

            # Perfect match (all 5 players)
            if matches == 5:
                return lineup

            # Track best partial match
            if matches > best_score:
                best_score = matches
                best_match = lineup

        # Return best match if at least 3 players match
        if best_score >= 3:
            return best_match

        return None

    def find_lineups_with_player(self, player_name: str, team: str = None) -> List[CTGLineupStats]:
        """Find all lineups containing a specific player."""
        self._load_lineups()

        normalized = self._normalize_name(player_name)
        results = []

        for lineup in self._cache:
            if team and team.lower() not in lineup.team.lower():
                continue
            if normalized in lineup.players:
                results.append(lineup)

        return sorted(results, key=lambda x: x.possessions, reverse=True)

    def get_best_lineup_for_player(self, player_name: str, team: str = None) -> Optional[CTGLineupStats]:
        """Get the highest-minutes lineup for a player."""
        lineups = self.find_lineups_with_player(player_name, team)
        return lineups[0] if lineups else None

    def clear_cache(self):
        """Clear the cache."""
        self._cache.clear()
        self._loaded = False


@dataclass
class BDLAdvancedStats:
    """
    Balldontlie Advanced Stats from /v1/stats/advanced endpoint

    These are per-game advanced metrics averaged over recent games.
    """
    # Player Impact Estimate (overall value metric, 0-1 scale)
    pie: float = 0.0

    # Efficiency metrics
    true_shooting_pct: float = 0.0      # TS% - most comprehensive efficiency
    effective_fg_pct: float = 0.0       # eFG% - 3PT-weighted FG%

    # Ratings (per 100 possessions)
    offensive_rating: float = 110.0     # Points produced per 100 poss
    defensive_rating: float = 110.0     # Points allowed per 100 poss
    net_rating: float = 0.0             # Off rating - Def rating

    # Rebounding percentages
    off_reb_pct: float = 0.0            # % of available OREBs grabbed
    def_reb_pct: float = 0.0            # % of available DREBs grabbed
    reb_pct: float = 0.0                # Total rebound %

    # Playmaking
    ast_pct: float = 0.0                # % of teammate FGs assisted
    ast_ratio: float = 0.0              # AST per 100 possessions
    ast_to_tov: float = 0.0             # AST/TOV ratio

    # Usage and pace
    usage_pct: float = 0.0              # % of team plays used
    pace: float = 100.0                 # Possessions per 48 min

    # Turnover efficiency
    tov_ratio: float = 0.0              # TOVs per 100 possessions

    # Sample size
    games_sampled: int = 0


@dataclass
class PBPStats:
    """
    Play-by-play derived stats from PBPStats.com

    These stats provide deeper shot and play quality analysis.
    """
    # Shot zone breakdown (FGA and accuracy)
    at_rim_fga: float = 0.0
    at_rim_accuracy: float = 0.0
    at_rim_freq: float = 0.0
    short_mid_fga: float = 0.0
    short_mid_accuracy: float = 0.0
    long_mid_fga: float = 0.0
    long_mid_accuracy: float = 0.0
    corner3_fga: float = 0.0
    corner3_accuracy: float = 0.0
    arc3_fga: float = 0.0
    arc3_accuracy: float = 0.0

    # Shot creation metrics
    assisted_2s_pct: float = 0.0  # % of 2s that are assisted
    assisted_3s_pct: float = 0.0  # % of 3s that are assisted
    shot_quality_avg: float = 0.0  # Expected value of shots taken

    # Efficiency
    ts_pct: float = 0.0
    efg_pct: float = 0.0
    usage: float = 0.0

    # Turnover breakdown
    bad_pass_turnovers: float = 0.0
    lost_ball_turnovers: float = 0.0
    live_ball_tov_pct: float = 0.0  # % of TOVs that are live ball

    # Rebounding
    self_oreb: float = 0.0  # Own miss rebounds (ownmiss)
    off_reb_pct: float = 0.0
    def_reb_pct: float = 0.0

    # Fouls
    shooting_fouls_drawn_pct: float = 0.0
    fouls_drawn: float = 0.0
    offensive_fouls: float = 0.0

    # Blocks
    blocks: float = 0.0
    blocked_2s: float = 0.0
    fg2a_pct_blocked: float = 0.0

    # Steals breakdown
    bad_pass_steals: float = 0.0
    lost_ball_steals: float = 0.0

    # On/Off ratings (if available)
    on_off_rtg: float = 0.0
    on_def_rtg: float = 0.0

    # Assists breakdown
    at_rim_assists: float = 0.0
    three_pt_assists: float = 0.0
    assist_points: float = 0.0


# =============================================================================
# DATA CLASSES - TrackingStats (needed by NBATrackingClient)
# =============================================================================

@dataclass
class TrackingStats:
    """
    NBA.com tracking stats from nba_api

    These are the advanced tracking stats that significantly improve projections:
    - Hustle stats: deflections, loose balls, contested shots, box outs
    - Passing: potential assists, passes made/received, ast_to_pass_pct
    - Rebounding: contested/uncontested splits, rebound chances
    - Possessions: touches, time of possession, dribbles per touch
    - Shot types: catch & shoot vs pull-up splits
    - Drives: drive frequency, efficiency, assists from drives
    - Estimated metrics: offensive/defensive rating, usage, pace
    """
    # Hustle stats (critical for steals/blocks props)
    deflections: float = 0.0
    loose_balls_recovered: float = 0.0
    contested_shots: float = 0.0
    contested_shots_2pt: float = 0.0
    contested_shots_3pt: float = 0.0
    charges_drawn: float = 0.0
    screen_assists: float = 0.0
    box_outs: float = 0.0
    off_box_outs: float = 0.0
    def_box_outs: float = 0.0

    # Passing stats (critical for assists props)
    passes_made: float = 0.0
    passes_received: float = 0.0
    potential_assists: float = 0.0  # KEY - assists that should have been
    secondary_assists: float = 0.0
    ft_assists: float = 0.0
    ast_points_created: float = 0.0
    ast_to_pass_pct: float = 0.0  # Efficiency of passes to assists
    ast_adj: float = 0.0  # Adjusted assists

    # Rebounding tracking (critical for rebounds props)
    oreb_chances: float = 0.0
    oreb_chance_pct: float = 0.0
    oreb_contest_pct: float = 0.0
    dreb_chances: float = 0.0
    dreb_chance_pct: float = 0.0
    dreb_contest_pct: float = 0.0
    avg_reb_dist: float = 0.0  # Average distance to rebound

    # Possession/Touch stats (usage indicator)
    touches: float = 0.0
    front_ct_touches: float = 0.0
    time_of_poss: float = 0.0
    avg_sec_per_touch: float = 0.0
    avg_drib_per_touch: float = 0.0
    pts_per_touch: float = 0.0
    elbow_touches: float = 0.0
    post_touches: float = 0.0
    paint_touches: float = 0.0

    # Shot type breakdown (critical for points/threes props)
    catch_shoot_fga: float = 0.0
    catch_shoot_fg_pct: float = 0.0
    catch_shoot_fg3_pct: float = 0.0
    pull_up_fga: float = 0.0
    pull_up_fg_pct: float = 0.0
    pull_up_fg3_pct: float = 0.0

    # Drive stats (critical for points props)
    drives: float = 0.0
    drive_pts: float = 0.0
    drive_fg_pct: float = 0.0
    drive_fta: float = 0.0  # Free throws from drives
    drive_ast: float = 0.0  # Assists from drives
    drive_tov: float = 0.0  # Turnovers from drives
    drive_pf: float = 0.0   # Fouls drawn from drives

    # Defensive tracking
    def_rim_fga: float = 0.0  # Shots defended at rim
    def_rim_fg_pct: float = 0.0  # Opponent FG% at rim when defending

    # Speed/Distance (workload indicator)
    dist_miles: float = 0.0
    avg_speed: float = 0.0
    avg_speed_off: float = 0.0
    avg_speed_def: float = 0.0

    # Estimated advanced metrics (NBA.com calculations)
    e_off_rating: float = 0.0
    e_def_rating: float = 0.0
    e_net_rating: float = 0.0
    e_ast_ratio: float = 0.0
    e_oreb_pct: float = 0.0
    e_dreb_pct: float = 0.0
    e_reb_pct: float = 0.0
    e_tov_pct: float = 0.0
    e_usg_pct: float = 0.0
    e_pace: float = 0.0


# =============================================================================
# NBA.COM TRACKING CLIENT (via nba_api)
# =============================================================================

class NBATrackingClient:
    """
    Client for NBA.com tracking stats via nba_api

    Fetches advanced tracking data that's not available from Balldontlie:
    - Hustle stats (deflections, loose balls, contested shots)
    - Passing (potential assists, ast_to_pass_pct)
    - Rebounding (contested %, chances, distance)
    - Possessions (touches, time of possession)
    - Shot types (catch & shoot, pull-up)
    - Drives (frequency, efficiency)
    - Speed/Distance
    - Estimated metrics (off/def rating, etc.)
    """

    def __init__(self, season: str = None):
        self.season = season or get_current_nba_season_str()
        self._cache: Dict[str, Tuple[datetime, pd.DataFrame]] = {}
        self._cache_ttl = timedelta(hours=6)  # Tracking stats update less frequently
        self._player_cache: Dict[int, TrackingStats] = {}

    def _get_cached_df(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get cached DataFrame if still valid"""
        if cache_key in self._cache:
            cached_time, df = self._cache[cache_key]
            if datetime.now() - cached_time < self._cache_ttl:
                return df
        return None

    def _cache_df(self, cache_key: str, df: pd.DataFrame):
        """Cache a DataFrame"""
        self._cache[cache_key] = (datetime.now(), df)

    def _safe_fetch(self, fetch_func, cache_key: str) -> Optional[pd.DataFrame]:
        """Safely fetch data with caching and error handling"""
        # Check cache first
        cached = self._get_cached_df(cache_key)
        if cached is not None:
            return cached

        try:
            import time
            time.sleep(0.6)  # Rate limiting for NBA.com
            result = fetch_func()
            df = result.get_data_frames()[0]
            self._cache_df(cache_key, df)
            return df
        except Exception as e:
            logger.warning(f"NBA tracking fetch failed ({cache_key}): {e}")
            return None

    def fetch_all_tracking_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch all tracking data in one batch

        Returns dict with keys: hustle, passing, defense, rebounding,
        possessions, catch_shoot, pull_up, drives, speed, estimated_metrics
        """
        if not NBA_API_AVAILABLE:
            logger.warning("nba_api not available - cannot fetch tracking stats")
            return {}

        data = {}

        # Hustle stats
        data['hustle'] = self._safe_fetch(
            lambda: leaguehustlestatsplayer.LeagueHustleStatsPlayer(
                season=self.season,
                per_mode_time='PerGame'
            ),
            f'hustle_{self.season}'
        )

        # Passing stats
        data['passing'] = self._safe_fetch(
            lambda: leaguedashptstats.LeagueDashPtStats(
                season=self.season,
                pt_measure_type='Passing',
                player_or_team='Player',
                per_mode_simple='PerGame'
            ),
            f'passing_{self.season}'
        )

        # Defense stats
        data['defense'] = self._safe_fetch(
            lambda: leaguedashptstats.LeagueDashPtStats(
                season=self.season,
                pt_measure_type='Defense',
                player_or_team='Player',
                per_mode_simple='PerGame'
            ),
            f'defense_{self.season}'
        )

        # Rebounding tracking
        data['rebounding'] = self._safe_fetch(
            lambda: leaguedashptstats.LeagueDashPtStats(
                season=self.season,
                pt_measure_type='Rebounding',
                player_or_team='Player',
                per_mode_simple='PerGame'
            ),
            f'rebounding_{self.season}'
        )

        # Possessions/Touches
        data['possessions'] = self._safe_fetch(
            lambda: leaguedashptstats.LeagueDashPtStats(
                season=self.season,
                pt_measure_type='Possessions',
                player_or_team='Player',
                per_mode_simple='PerGame'
            ),
            f'possessions_{self.season}'
        )

        # Catch & Shoot
        data['catch_shoot'] = self._safe_fetch(
            lambda: leaguedashptstats.LeagueDashPtStats(
                season=self.season,
                pt_measure_type='CatchShoot',
                player_or_team='Player',
                per_mode_simple='PerGame'
            ),
            f'catch_shoot_{self.season}'
        )

        # Pull Up shots
        data['pull_up'] = self._safe_fetch(
            lambda: leaguedashptstats.LeagueDashPtStats(
                season=self.season,
                pt_measure_type='PullUpShot',
                player_or_team='Player',
                per_mode_simple='PerGame'
            ),
            f'pull_up_{self.season}'
        )

        # Drives
        data['drives'] = self._safe_fetch(
            lambda: leaguedashptstats.LeagueDashPtStats(
                season=self.season,
                pt_measure_type='Drives',
                player_or_team='Player',
                per_mode_simple='PerGame'
            ),
            f'drives_{self.season}'
        )

        # Speed/Distance
        data['speed'] = self._safe_fetch(
            lambda: leaguedashptstats.LeagueDashPtStats(
                season=self.season,
                pt_measure_type='SpeedDistance',
                player_or_team='Player',
                per_mode_simple='PerGame'
            ),
            f'speed_{self.season}'
        )

        # Estimated metrics
        data['estimated_metrics'] = self._safe_fetch(
            lambda: playerestimatedmetrics.PlayerEstimatedMetrics(
                season=self.season
            ),
            f'estimated_metrics_{self.season}'
        )

        logger.info(f"Fetched tracking data: {[k for k, v in data.items() if v is not None]}")
        return data

    def get_player_tracking_stats(self, player_id: int, player_name: str = None) -> Optional[TrackingStats]:
        """
        Get tracking stats for a specific player

        Args:
            player_id: NBA player ID (from Balldontlie, need to map)
            player_name: Player name for matching (more reliable)

        Returns:
            TrackingStats object with all available tracking data
        """
        # Check cache
        if player_id in self._player_cache:
            return self._player_cache[player_id]

        # Fetch all tracking data if not already cached
        all_data = self.fetch_all_tracking_data()
        if not all_data:
            return None

        # Build tracking stats from all sources
        stats = TrackingStats()

        # Helper to find player row by name (more reliable than ID matching)
        def find_player_row(df: pd.DataFrame) -> Optional[pd.Series]:
            if df is None or player_name is None:
                return None
            # Try exact match first
            mask = df['PLAYER_NAME'].str.lower() == player_name.lower()
            if mask.any():
                return df[mask].iloc[0]
            # Try partial match
            name_parts = player_name.lower().split()
            if len(name_parts) >= 2:
                first, last = name_parts[0], name_parts[-1]
                mask = df['PLAYER_NAME'].str.lower().str.contains(last)
                matches = df[mask]
                for _, row in matches.iterrows():
                    row_name = row['PLAYER_NAME'].lower()
                    if first in row_name:
                        return row
            return None

        # Hustle stats
        if all_data.get('hustle') is not None:
            row = find_player_row(all_data['hustle'])
            if row is not None:
                stats.deflections = float(row.get('DEFLECTIONS', 0) or 0)
                stats.loose_balls_recovered = float(row.get('LOOSE_BALLS_RECOVERED', 0) or 0)
                stats.contested_shots = float(row.get('CONTESTED_SHOTS', 0) or 0)
                stats.contested_shots_2pt = float(row.get('CONTESTED_SHOTS_2PT', 0) or 0)
                stats.contested_shots_3pt = float(row.get('CONTESTED_SHOTS_3PT', 0) or 0)
                stats.charges_drawn = float(row.get('CHARGES_DRAWN', 0) or 0)
                stats.screen_assists = float(row.get('SCREEN_ASSISTS', 0) or 0)
                stats.box_outs = float(row.get('BOX_OUTS', 0) or 0)
                stats.off_box_outs = float(row.get('OFF_BOXOUTS', 0) or 0)
                stats.def_box_outs = float(row.get('DEF_BOXOUTS', 0) or 0)

        # Passing stats
        if all_data.get('passing') is not None:
            row = find_player_row(all_data['passing'])
            if row is not None:
                stats.passes_made = float(row.get('PASSES_MADE', 0) or 0)
                stats.passes_received = float(row.get('PASSES_RECEIVED', 0) or 0)
                stats.potential_assists = float(row.get('POTENTIAL_AST', 0) or 0)
                stats.secondary_assists = float(row.get('SECONDARY_AST', 0) or 0)
                stats.ft_assists = float(row.get('FT_AST', 0) or 0)
                stats.ast_points_created = float(row.get('AST_POINTS_CREATED', 0) or 0)
                stats.ast_to_pass_pct = float(row.get('AST_TO_PASS_PCT', 0) or 0)
                stats.ast_adj = float(row.get('AST_ADJ', 0) or 0)

        # Defense stats
        if all_data.get('defense') is not None:
            row = find_player_row(all_data['defense'])
            if row is not None:
                stats.def_rim_fga = float(row.get('DEF_RIM_FGA', 0) or 0)
                stats.def_rim_fg_pct = float(row.get('DEF_RIM_FG_PCT', 0) or 0)

        # Rebounding tracking
        if all_data.get('rebounding') is not None:
            row = find_player_row(all_data['rebounding'])
            if row is not None:
                stats.oreb_chances = float(row.get('OREB_CHANCES', 0) or 0)
                stats.oreb_chance_pct = float(row.get('OREB_CHANCE_PCT', 0) or 0)
                stats.oreb_contest_pct = float(row.get('OREB_CONTEST_PCT', 0) or 0)
                stats.dreb_chances = float(row.get('DREB_CHANCES', 0) or 0)
                stats.dreb_chance_pct = float(row.get('DREB_CHANCE_PCT', 0) or 0)
                stats.dreb_contest_pct = float(row.get('DREB_CONTEST_PCT', 0) or 0)
                stats.avg_reb_dist = float(row.get('AVG_REB_DIST', 0) or 0)

        # Possessions/Touches
        if all_data.get('possessions') is not None:
            row = find_player_row(all_data['possessions'])
            if row is not None:
                stats.touches = float(row.get('TOUCHES', 0) or 0)
                stats.front_ct_touches = float(row.get('FRONT_CT_TOUCHES', 0) or 0)
                stats.time_of_poss = float(row.get('TIME_OF_POSS', 0) or 0)
                stats.avg_sec_per_touch = float(row.get('AVG_SEC_PER_TOUCH', 0) or 0)
                stats.avg_drib_per_touch = float(row.get('AVG_DRIB_PER_TOUCH', 0) or 0)
                stats.pts_per_touch = float(row.get('PTS_PER_TOUCH', 0) or 0)
                stats.elbow_touches = float(row.get('ELBOW_TOUCHES', 0) or 0)
                stats.post_touches = float(row.get('POST_TOUCHES', 0) or 0)
                stats.paint_touches = float(row.get('PAINT_TOUCHES', 0) or 0)

        # Catch & Shoot
        if all_data.get('catch_shoot') is not None:
            row = find_player_row(all_data['catch_shoot'])
            if row is not None:
                stats.catch_shoot_fga = float(row.get('CATCH_SHOOT_FGA', 0) or 0)
                stats.catch_shoot_fg_pct = float(row.get('CATCH_SHOOT_FG_PCT', 0) or 0)
                stats.catch_shoot_fg3_pct = float(row.get('CATCH_SHOOT_FG3_PCT', 0) or 0)

        # Pull Up
        if all_data.get('pull_up') is not None:
            row = find_player_row(all_data['pull_up'])
            if row is not None:
                stats.pull_up_fga = float(row.get('PULL_UP_FGA', 0) or 0)
                stats.pull_up_fg_pct = float(row.get('PULL_UP_FG_PCT', 0) or 0)
                stats.pull_up_fg3_pct = float(row.get('PULL_UP_FG3_PCT', 0) or 0)

        # Drives
        if all_data.get('drives') is not None:
            row = find_player_row(all_data['drives'])
            if row is not None:
                stats.drives = float(row.get('DRIVES', 0) or 0)
                stats.drive_pts = float(row.get('DRIVE_PTS', 0) or 0)
                stats.drive_fg_pct = float(row.get('DRIVE_FG_PCT', 0) or 0)
                stats.drive_fta = float(row.get('DRIVE_FTA', 0) or 0)
                stats.drive_ast = float(row.get('DRIVE_AST', 0) or 0)
                stats.drive_tov = float(row.get('DRIVE_TOV', 0) or 0)
                stats.drive_pf = float(row.get('DRIVE_PF', 0) or 0)

        # Speed/Distance
        if all_data.get('speed') is not None:
            row = find_player_row(all_data['speed'])
            if row is not None:
                stats.dist_miles = float(row.get('DIST_MILES', 0) or 0)
                stats.avg_speed = float(row.get('AVG_SPEED', 0) or 0)
                stats.avg_speed_off = float(row.get('AVG_SPEED_OFF', 0) or 0)
                stats.avg_speed_def = float(row.get('AVG_SPEED_DEF', 0) or 0)

        # Estimated metrics
        if all_data.get('estimated_metrics') is not None:
            row = find_player_row(all_data['estimated_metrics'])
            if row is not None:
                stats.e_off_rating = float(row.get('E_OFF_RATING', 0) or 0)
                stats.e_def_rating = float(row.get('E_DEF_RATING', 0) or 0)
                stats.e_net_rating = float(row.get('E_NET_RATING', 0) or 0)
                stats.e_ast_ratio = float(row.get('E_AST_RATIO', 0) or 0)
                stats.e_oreb_pct = float(row.get('E_OREB_PCT', 0) or 0)
                stats.e_dreb_pct = float(row.get('E_DREB_PCT', 0) or 0)
                stats.e_reb_pct = float(row.get('E_REB_PCT', 0) or 0)
                stats.e_tov_pct = float(row.get('E_TOV_PCT', 0) or 0)
                stats.e_usg_pct = float(row.get('E_USG_PCT', 0) or 0)
                stats.e_pace = float(row.get('E_PACE', 0) or 0)

        # Cache the result
        self._player_cache[player_id] = stats
        return stats

    def clear_cache(self):
        """Clear all caches"""
        self._cache.clear()
        self._player_cache.clear()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PlayerProp:
    """Live prop line — best available across all books"""
    player_id: int
    player_name: str
    prop_type: str  # points, rebounds, assists, threes, etc.
    line: float
    over_odds: int  # American odds (best available)
    under_odds: int
    last_updated: datetime = field(default_factory=datetime.now)
    # Cross-book data (populated when PropLine multi-book data available)
    best_over_book: str = "draftkings"
    best_under_book: str = "draftkings"
    dk_line: float = 0.0
    dk_over_odds: int = -110
    dk_under_odds: int = -110
    all_books: Dict = field(default_factory=dict)


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
    last_3_avg: Dict[str, float] = field(default_factory=dict)
    last_5_avg: Dict[str, float] = field(default_factory=dict)
    last_10_avg: Dict[str, float] = field(default_factory=dict)

    # Standard deviations (for edge calculation)
    std_dev: Dict[str, float] = field(default_factory=dict)

    # Per-minute rates
    per_minute: Dict[str, float] = field(default_factory=dict)

    # Usage data
    usage_pct: float = 0.0

    # Home/Away splits (full sample)
    home_avg: Dict[str, float] = field(default_factory=dict)
    away_avg: Dict[str, float] = field(default_factory=dict)

    # Home/Away L5 splits (more recent)
    home_l5_avg: Dict[str, float] = field(default_factory=dict)
    away_l5_avg: Dict[str, float] = field(default_factory=dict)

    # Opponent-specific history: {opponent_id: {stat_key: avg, "games": count}}
    opponent_history: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Raw game logs for correlation
    game_logs: List[GameLog] = field(default_factory=list)

    # Injury status
    status: str = "active"  # active, out, questionable, doubtful, probable
    injury_note: str = ""

    # NBA.com tracking stats (from nba_api)
    tracking: Optional[TrackingStats] = None

    # PBP Stats (from pbpstats.com)
    pbp: Optional[PBPStats] = None

    # Cleaning the Glass stats (subscription)
    ctg: Optional[CTGStats] = None

    # Balldontlie Advanced Stats
    bdl_advanced: Optional[BDLAdvancedStats] = None

    # Roster-filtered averages (games where currently injured teammates were also out)
    # This prevents double-counting injury boosts
    roster_filtered_l3_avg: Dict[str, float] = field(default_factory=dict)
    roster_filtered_l5_avg: Dict[str, float] = field(default_factory=dict)
    roster_filtered_games_count: int = 0  # How many games with this roster
    roster_filtering_applied: bool = False  # True if we have enough filtered games


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
# PROJECTION FEATURES (for ML model input)
# =============================================================================

@dataclass
class ProjectionFeatures:
    """
    All features for ML projection model (~66 features)

    This consolidates data from all sources into a single feature vector
    for the stacked ensemble (LightGBM + XGBoost + CatBoost).
    """

    # === BASE STATS (from game logs) ===
    l3_avg: float = 0.0       # Last 3 game average for this stat
    l5_avg: float = 0.0       # Last 5 game average
    l10_avg: float = 0.0      # Last 10 game average
    season_avg: float = 0.0   # Season average
    std_dev: float = 0.0      # Variance for this stat
    per_minute: float = 0.0   # Rate stat (stat per minute)

    # === MINUTES CONTEXT ===
    proj_minutes: float = 0.0
    avg_minutes_l5: float = 0.0
    minutes_trend: float = 0.0  # L3 vs L10 difference (positive = trending up)

    # === GAME CONTEXT ===
    is_home: bool = False
    opp_def_rating: float = 110.0
    pace_factor: float = 1.0
    spread: float = 0.0
    total: float = 220.0
    is_b2b: bool = False
    days_rest: int = 1

    # === BALLDONTLIE ADVANCED ===
    pie: float = 0.0                    # Player Impact Estimate (0-1)
    true_shooting_pct: float = 0.0      # True Shooting %
    efg_pct: float = 0.0                # Effective FG%
    net_rating: float = 0.0             # On-court point differential
    off_reb_pct: float = 0.0            # Offensive rebound %
    def_reb_pct: float = 0.0            # Defensive rebound %
    ast_ratio: float = 0.0              # AST per 100 possessions
    ast_to_tov: float = 0.0             # AST/TOV ratio
    has_bdl_advanced: bool = False

    # === CTG PLAYER FEATURES ===
    ctg_psa_percentile: int = 50        # Points per Shot Attempt percentile
    ctg_usage_percentile: int = 50      # Usage rate percentile
    ctg_ast_percentile: int = 50        # Assist % percentile
    ctg_ast_to_usage: float = 0.0       # Playmaking efficiency
    ctg_tov_percentile: int = 50        # Turnover % percentile (higher = fewer TOVs)
    has_ctg: bool = False

    # === CTG TEAM-LEVEL ===
    # Own team offense
    team_off_efg_rank: int = 15         # Team's offensive eFG% rank (1-30)
    team_off_tov_pct: float = 14.6      # Team's turnover %
    team_transition_freq: float = 15.6  # % of plays in transition
    team_transition_pts_play: float = 125.3  # Transition pts per play

    # Opponent defense (matchup context)
    opp_def_efg_rank: int = 15          # Opponent's defensive eFG% rank
    opp_def_tov_forced_pct: float = 14.6  # Opponent's forced turnover %
    opp_def_orb_allowed_pct: float = 29.2   # Opponent ORB% allowed
    opp_def_transition_pts: float = 125.3   # Opponent transition pts allowed
    opp_def_halfcourt_pts: float = 98.1     # Opponent halfcourt pts allowed
    opp_def_rim_pct: float = 66.5       # Opponent FG% allowed at rim
    opp_def_three_pct: float = 36.1     # Opponent 3PT% allowed
    has_ctg_team: bool = False

    # === LINEUP-SPECIFIC (from NBA.com + CTG) ===
    lineup_net_rating: float = 0.0      # This 5-man's net rating
    lineup_off_efg: float = 54.5        # Lineup's offensive eFG%
    lineup_off_pts_poss: float = 115.9  # Lineup's pts per 100 poss
    lineup_def_pts_poss: float = 115.9  # Lineup's defensive pts allowed
    lineup_possessions: int = 0         # Sample size (for confidence)
    lineup_found: bool = False          # True if exact lineup found in CSV

    # === TRACKING FEATURES ===
    track_drives: float = 0.0           # Drives per game
    track_drive_pts: float = 0.0        # Points from drives
    track_potential_assists: float = 0.0  # Potential assists (should-be assists)
    track_oreb_chance_pct: float = 0.0  # Offensive rebound chance %
    track_dreb_chance_pct: float = 0.0  # Defensive rebound chance %
    track_deflections: float = 0.0      # Deflections per game
    track_contested_shots: float = 0.0  # Contested shots per game
    track_catch_shoot_pct: float = 0.0  # Catch & shoot FG%
    track_pull_up_pct: float = 0.0      # Pull-up FG%
    has_tracking: bool = False

    # === PBP FEATURES ===
    pbp_shot_quality: float = 0.0       # Shot quality average
    pbp_at_rim_pct: float = 0.0         # % of shots at rim
    pbp_corner3_pct: float = 0.0        # % of 3s from corner
    pbp_arc3_pct: float = 0.0           # % of 3s from arc
    pbp_bad_pass_tov_pct: float = 0.0   # % of TOVs from bad passes
    pbp_assisted_2s_pct: float = 0.0    # % of 2s that are assisted
    pbp_assisted_3s_pct: float = 0.0    # % of 3s that are assisted
    has_pbp: bool = False

    # === INJURY/ROLE CONTEXT ===
    injury_boost: float = 1.0           # Multiplier from teammate injuries
    role_change: int = 0                # -1=reduced, 0=same, 1=expanded

    # === OPPONENT HISTORY ===
    games_vs_opp: int = 0               # Games played vs this opponent
    avg_vs_opp: float = 0.0             # Avg stat vs this opponent (if 5+ games)

    def to_array(self) -> np.ndarray:
        """Convert to feature array for ML model (~66 features)"""
        return np.array([
            # Base stats (6)
            self.l3_avg, self.l5_avg, self.l10_avg, self.season_avg,
            self.std_dev, self.per_minute,
            # Minutes (3)
            self.proj_minutes, self.avg_minutes_l5, self.minutes_trend,
            # Game context (7)
            float(self.is_home), self.opp_def_rating, self.pace_factor,
            self.spread, self.total, float(self.is_b2b), float(self.days_rest),
            # BDL Advanced (9)
            self.pie, self.true_shooting_pct, self.efg_pct, self.net_rating,
            self.off_reb_pct, self.def_reb_pct, self.ast_ratio, self.ast_to_tov,
            float(self.has_bdl_advanced),
            # CTG Player (6)
            float(self.ctg_psa_percentile), float(self.ctg_usage_percentile),
            float(self.ctg_ast_percentile), self.ctg_ast_to_usage,
            float(self.ctg_tov_percentile), float(self.has_ctg),
            # CTG Team (12)
            float(self.team_off_efg_rank), self.team_off_tov_pct,
            self.team_transition_freq, self.team_transition_pts_play,
            float(self.opp_def_efg_rank), self.opp_def_tov_forced_pct,
            self.opp_def_orb_allowed_pct, self.opp_def_transition_pts,
            self.opp_def_halfcourt_pts, self.opp_def_rim_pct,
            self.opp_def_three_pct, float(self.has_ctg_team),
            # Lineup (6)
            self.lineup_net_rating, self.lineup_off_efg,
            self.lineup_off_pts_poss, self.lineup_def_pts_poss,
            float(self.lineup_possessions), float(self.lineup_found),
            # Tracking (10)
            self.track_drives, self.track_drive_pts,
            self.track_potential_assists, self.track_oreb_chance_pct,
            self.track_dreb_chance_pct, self.track_deflections,
            self.track_contested_shots, self.track_catch_shoot_pct,
            self.track_pull_up_pct, float(self.has_tracking),
            # PBP (8)
            self.pbp_shot_quality, self.pbp_at_rim_pct,
            self.pbp_corner3_pct, self.pbp_arc3_pct,
            self.pbp_bad_pass_tov_pct, self.pbp_assisted_2s_pct,
            self.pbp_assisted_3s_pct, float(self.has_pbp),
            # Context (4)
            self.injury_boost, float(self.role_change),
            float(self.games_vs_opp), self.avg_vs_opp
        ])  # Total: 66 features


# Feature names for interpretability
PROJECTION_FEATURE_NAMES = [
    # Base stats (6)
    'l3_avg', 'l5_avg', 'l10_avg', 'season_avg', 'std_dev', 'per_minute',
    # Minutes (3)
    'proj_minutes', 'avg_minutes_l5', 'minutes_trend',
    # Game context (7)
    'is_home', 'opp_def_rating', 'pace_factor', 'spread', 'total', 'is_b2b', 'days_rest',
    # BDL Advanced (9)
    'pie', 'true_shooting_pct', 'efg_pct', 'net_rating',
    'off_reb_pct', 'def_reb_pct', 'ast_ratio', 'ast_to_tov', 'has_bdl_advanced',
    # CTG Player (6)
    'ctg_psa_percentile', 'ctg_usage_percentile', 'ctg_ast_percentile',
    'ctg_ast_to_usage', 'ctg_tov_percentile', 'has_ctg',
    # CTG Team (12)
    'team_off_efg_rank', 'team_off_tov_pct', 'team_transition_freq', 'team_transition_pts_play',
    'opp_def_efg_rank', 'opp_def_tov_forced_pct', 'opp_def_orb_allowed_pct',
    'opp_def_transition_pts', 'opp_def_halfcourt_pts', 'opp_def_rim_pct',
    'opp_def_three_pct', 'has_ctg_team',
    # Lineup (6)
    'lineup_net_rating', 'lineup_off_efg', 'lineup_off_pts_poss',
    'lineup_def_pts_poss', 'lineup_possessions', 'lineup_found',
    # Tracking (10)
    'track_drives', 'track_drive_pts', 'track_potential_assists',
    'track_oreb_chance_pct', 'track_dreb_chance_pct', 'track_deflections',
    'track_contested_shots', 'track_catch_shoot_pct', 'track_pull_up_pct', 'has_tracking',
    # PBP (8)
    'pbp_shot_quality', 'pbp_at_rim_pct', 'pbp_corner3_pct', 'pbp_arc3_pct',
    'pbp_bad_pass_tov_pct', 'pbp_assisted_2s_pct', 'pbp_assisted_3s_pct', 'has_pbp',
    # Context (4)
    'injury_boost', 'role_change', 'games_vs_opp', 'avg_vs_opp'
]


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
        season: int = None,
        last_n: int = 20
    ) -> List[Dict]:
        """Fetch recent game logs for a player (with SQLite caching)"""
        if season is None:
            season = get_current_nba_season()
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

        current_logs = data.get("data", []) if data else []

        # Early season supplement: if current season has < 10 games,
        # pull previous season to give projections enough data
        if len(current_logs) < 10:
            prev_season = season - 1
            if prev_season >= 2024:
                needed = last_n - len(current_logs)
                if needed > 0:
                    logger.info(f"Only {len(current_logs)} logs in {season} for player {player_id}, supplementing from {prev_season}")
                    prev_data = self._make_request("stats", {
                        "player_ids[]": player_id,
                        "seasons[]": prev_season,
                        "per_page": 100
                    })
                    if prev_data and prev_data.get("data"):
                        current_logs.extend(prev_data["data"])

        if not current_logs:
            return []

        # Sort by date descending to get most recent games
        logs = current_logs
        logs.sort(key=lambda x: x.get("game", {}).get("date", ""), reverse=True)
        result = logs[:last_n]

        # Cache the game logs
        if CACHE_ENABLED and result:
            cache = get_cache()
            cache.set_game_logs(player_id, logs)  # Cache all logs, not just last_n

        return result

    def get_season_averages(self, player_id: int, season: int = None) -> Optional[Dict]:
        """Get season averages for a player (with SQLite caching)"""
        if season is None:
            season = get_current_nba_season()
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
        result = data["data"][0] if data and data.get("data") else None

        # Early season: if current season avg is based on < 5 games,
        # use previous season as more reliable baseline
        games_played = result.get("games_played", 0) if result else 0
        if games_played < 5:
            prev_season = season - 1
            if prev_season >= 2024:
                prev_data = self._make_request("season_averages", {
                    "season": prev_season,
                    "player_id": player_id
                })
                if prev_data and prev_data.get("data"):
                    prev_result = prev_data["data"][0]
                    if prev_result.get("games_played", 0) >= 5:
                        logger.info(f"Using {prev_season} season avg for player {player_id} ({games_played} games in {season})")
                        result = prev_result

        if result:
            if CACHE_ENABLED:
                cache = get_cache()
                cache.set_season_average(player_id, result, season)
            return result

        return None

    def get_usage_stats(self, player_id: int, season: int = None) -> Optional[Dict]:
        """Get usage rate and advanced averages"""
        if season is None:
            season = get_current_nba_season()
        data = self._make_request("season_averages", {
            "season": season,
            "player_id": player_id,  # API uses singular player_id
            "type": "usage"
        })
        if data and data.get("data"):
            return data["data"][0]
        return None

    def get_advanced_stats(self, player_id: int, season: int = None) -> List[Dict]:
        """Get advanced stats per game (pace, ratings). Falls back to previous season early in year."""
        if season is None:
            season = get_current_nba_season()
        data = self._make_request("stats/advanced", {
            "player_ids[]": player_id,
            "seasons[]": season,
            "per_page": 20
        })
        result = data.get("data", []) if data else []
        if not result and season - 1 >= 2024:
            logger.info(f"No advanced stats for player {player_id} in {season}, falling back to {season-1}")
            data = self._make_request("stats/advanced", {
                "player_ids[]": player_id,
                "seasons[]": season - 1,
                "per_page": 20
            })
            result = data.get("data", []) if data else []
        return result

    def get_bdl_advanced_stats(self, player_id: int, season: int = None, num_games: int = 10) -> Optional[BDLAdvancedStats]:
        """
        Get aggregated BDL Advanced Stats for a player

        Fetches per-game advanced stats and averages them over recent games.
        Returns a BDLAdvancedStats object ready for ML features.
        """
        raw_stats = self.get_advanced_stats(player_id, season)

        if not raw_stats:
            return None

        # Take most recent num_games
        recent = raw_stats[:num_games]
        n = len(recent)

        if n == 0:
            return None

        # Helper to safely average a field
        def avg(field: str) -> float:
            values = [g.get(field, 0) or 0 for g in recent]
            return sum(values) / n if n > 0 else 0.0

        return BDLAdvancedStats(
            pie=avg("pie"),
            true_shooting_pct=avg("true_shooting_percentage"),
            effective_fg_pct=avg("effective_field_goal_percentage"),
            offensive_rating=avg("offensive_rating"),
            defensive_rating=avg("defensive_rating"),
            net_rating=avg("net_rating"),
            off_reb_pct=avg("offensive_rebound_percentage"),
            def_reb_pct=avg("defensive_rebound_percentage"),
            reb_pct=avg("rebound_percentage"),
            ast_pct=avg("assist_percentage"),
            ast_ratio=avg("assist_ratio"),
            ast_to_tov=avg("assist_to_turnover"),
            usage_pct=avg("usage_percentage"),
            pace=avg("pace"),
            tov_ratio=avg("turnover_ratio"),
            games_sampled=n
        )

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
        """Get all games for today (includes tomorrow's UTC date for late games)"""
        from datetime import timedelta
        today = datetime.now().strftime("%Y-%m-%d")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        # Check both today and tomorrow to catch late-night games (UTC offset)
        data = self._make_request("games", {
            "start_date": today,
            "end_date": tomorrow
        })
        games = data.get("data", []) if data else []

        # Filter to only include games that haven't finished yet or are today
        active_games = []
        for g in games:
            status = g.get("status", "")
            game_date = g.get("date", "")[:10]
            # Include if: scheduled for today/tomorrow, or currently in progress
            if status != "Final" or game_date == today:
                active_games.append(g)

        return active_games

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

        # Generate synthetic PRA (pts_rebs_asts) props
        # DraftKings API doesn't return combo props, so we create them from individual lines
        synthetic_pra = self._generate_synthetic_pra(props)
        props.extend(synthetic_pra)

        return props

    def _generate_synthetic_pra(self, props: List[Dict]) -> List[Dict]:
        """
        Generate synthetic PRA (pts_rebs_asts) props from individual points, rebounds, assists lines.

        For players who have all 3 individual props, creates a combo prop with:
        - Line = points_line + rebounds_line + assists_line
        - Odds estimated at -110/-110 (standard juice)

        These are marked as synthetic for tracking purposes.
        """
        # Group props by player_id
        player_props = {}
        for prop in props:
            pid = prop.get("player_id")
            if pid not in player_props:
                player_props[pid] = {}
            player_props[pid][prop.get("prop_type")] = prop

        synthetic = []
        for player_id, player_data in player_props.items():
            # Check if player has all 3 required props
            if "points" in player_data and "rebounds" in player_data and "assists" in player_data:
                pts_line = player_data["points"]["line"]
                reb_line = player_data["rebounds"]["line"]
                ast_line = player_data["assists"]["line"]

                # Create synthetic PRA prop
                pra_line = pts_line + reb_line + ast_line

                synthetic.append({
                    "player_id": player_id,
                    "prop_type": "pts_rebs_asts",
                    "line": pra_line,
                    "over_odds": -110,  # Standard juice estimate
                    "under_odds": -110,
                    "updated_at": player_data["points"].get("updated_at"),
                    "synthetic": True,  # Mark as synthetic for tracking
                    "component_lines": {
                        "points": pts_line,
                        "rebounds": reb_line,
                        "assists": ast_line
                    }
                })

        if synthetic:
            logging.info(f"Generated {len(synthetic)} synthetic PRA props from individual lines")

        return synthetic

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
        self.odds_client = OddsAPIClient()  # The Odds API for live props
        season_str = get_current_nba_season_str()
        self.tracking_client = NBATrackingClient(season=season_str)  # NBA.com tracking stats
        self.pbp_client = PBPStatsClient(season=season_str)  # PBP Stats API
        self.ctg_client = CleaningTheGlassClient()  # Cleaning the Glass player stats
        self.ctg_team_client = CTGTeamClient()  # Cleaning the Glass team stats
        self.ctg_lineup_client = CTGLineupClient()  # Cleaning the Glass lineup stats
        self.current_season = get_current_nba_season()

        # PropLine client for fast bulk props (optional)
        try:
            from propline_client import PropLineClient
            self.propline_client = PropLineClient()
            self.use_propline = self.propline_client.api_key is not None
            if self.use_propline:
                logger.info("PropLine client initialized (bulk props enabled)")
        except ImportError:
            self.propline_client = None
            self.use_propline = False

        # Caches
        self._player_cache: Dict[int, PlayerProfile] = {}
        self._team_cache: Dict[int, TeamProfile] = {}
        self._propline_bulk_cache: Optional[Dict] = None  # Cached bulk props from PropLine

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
            except (ValueError, TypeError, IndexError):
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
        profile.last_3_avg = self._calculate_averages(played_games[:3])
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

        # Calculate home/away L5 splits (more recent)
        profile.home_l5_avg = self._calculate_averages(home_games[:5])
        profile.away_l5_avg = self._calculate_averages(away_games[:5])

        # Calculate opponent-specific history (all played games)
        profile.opponent_history = self._calculate_opponent_history(played_games)

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

        # Get usage rate from advanced stats (the usage endpoint doesn't include it)
        adv_stats = self.client.get_advanced_stats(player_id, self.current_season)
        if adv_stats:
            # Average usage_percentage across recent games
            usage_vals = [g.get("usage_percentage", 0) for g in adv_stats if g.get("usage_percentage")]
            if usage_vals:
                avg_usage = sum(usage_vals) / len(usage_vals)
                # API returns as decimal (0.25 = 25%), convert to percentage
                profile.usage_pct = avg_usage * 100 if avg_usage < 1 else avg_usage

        # Check injury status
        injuries = self.client.get_injuries(profile.team_id)
        for injury in injuries:
            if injury.get("player", {}).get("id") == player_id:
                profile.status = injury.get("status", "active").lower()
                profile.injury_note = injury.get("comment", "")
                break

        # Tracking/PBP/CTG stats — SKIPPED when adjustments are disabled
        # These add 5-15s per player with no proven value (honest eval showed they add noise)
        # Re-enable in adjustment_config.json if needed in the future
        try:
            import json as _json
            _cfg_path = os.path.join(os.path.dirname(__file__), 'adjustment_config.json')
            with open(_cfg_path) as _f:
                _adj_cfg = _json.load(_f)
        except Exception:
            _adj_cfg = {}

        if _adj_cfg.get("tracking", {}).get("enabled", False) and NBA_API_AVAILABLE:
            tracking_stats = self.tracking_client.get_player_tracking_stats(
                player_id=player_id, player_name=profile.name
            )
            if tracking_stats:
                profile.tracking = tracking_stats

        if _adj_cfg.get("pbp", {}).get("enabled", False):
            pbp_stats = self.pbp_client.get_player_pbp_stats(profile.name)
            if pbp_stats:
                profile.pbp = pbp_stats

        if _adj_cfg.get("ctg", {}).get("enabled", False):
            ctg_stats = self.ctg_client.get_player_stats(profile.name)
            if ctg_stats:
                profile.ctg = ctg_stats

        # Fetch Balldontlie Advanced stats (PIE, TS%, eFG%, net_rating, etc.)
        bdl_advanced = self.client.get_bdl_advanced_stats(player_id, self.current_season)
        if bdl_advanced:
            profile.bdl_advanced = bdl_advanced
            logger.debug(f"Loaded BDL advanced for {profile.name}: pie={bdl_advanced.pie:.3f}, ts={bdl_advanced.true_shooting_pct:.1f}%")

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

    def _calculate_opponent_history(self, logs: List[GameLog]) -> Dict[str, Dict[str, float]]:
        """
        Calculate per-opponent averages from game logs.
        Groups games by opponent and calculates averages for each.

        Returns: {opponent_id: {stat_key: avg, "games": count}}
        """
        if not logs:
            return {}

        # Group games by opponent
        opponent_games: Dict[str, List[GameLog]] = {}
        for log in logs:
            opp_id = log.opponent
            if opp_id not in opponent_games:
                opponent_games[opp_id] = []
            opponent_games[opp_id].append(log)

        # Calculate averages per opponent
        result = {}
        for opp_id, games in opponent_games.items():
            if len(games) >= 1:  # At least 1 game against this opponent
                avgs = self._calculate_averages(games)
                avgs["games"] = len(games)
                result[opp_id] = avgs

        return result

    # -------------------------------------------------------------------------
    # Roster-Aware Game Log Filtering
    # -------------------------------------------------------------------------

    def get_player_last_game_date(self, player_name: str) -> Optional[str]:
        """
        Get the date of a player's most recent game.
        Used to determine when an injured player last played.

        Returns: Date string (YYYY-MM-DD) or None if no games found
        """
        player_data = self.client.search_player(player_name)
        if not player_data:
            return None

        player_id = player_data["id"]
        raw_logs = self.client.get_player_game_logs(player_id, self.current_season, 5)

        if not raw_logs:
            return None

        # Find most recent game where they actually played (minutes > 0)
        for log in raw_logs:
            min_str = log.get("min", "0:00") or "0:00"
            try:
                if ":" in min_str:
                    parts = min_str.split(":")
                    minutes = int(parts[0]) + int(parts[1]) / 60
                else:
                    minutes = float(min_str)
            except (ValueError, TypeError, IndexError):
                minutes = 0

            if minutes > 0:
                game_date = log.get("game", {}).get("date", "")
                if game_date:
                    # Return just the date portion (YYYY-MM-DD)
                    return game_date[:10] if len(game_date) >= 10 else game_date

        return None

    def filter_games_by_injured_roster(
        self,
        game_logs: List[GameLog],
        injured_teammate_last_games: Dict[str, str]
    ) -> List[GameLog]:
        """
        Filter game logs to only include games where injured teammates were also out.

        This prevents double-counting injury boosts:
        - If Ja Morant has been out since 2026-01-05
        - We filter Desmond Bane's logs to only games AFTER 2026-01-05
        - Those games already reflect Bane's production without Ja

        Args:
            game_logs: Player's game log history
            injured_teammate_last_games: {player_name: last_game_date}
                e.g., {"Ja Morant": "2026-01-05", "Marcus Smart": "2026-01-10"}

        Returns:
            Filtered list of games where ALL injured teammates were also out
        """
        if not injured_teammate_last_games:
            return game_logs

        # Find the LATEST "last game" among injured teammates
        # We need games AFTER this date (when all injured players were out)
        latest_injury_cutoff = None
        for player_name, last_game in injured_teammate_last_games.items():
            if last_game:
                if latest_injury_cutoff is None or last_game > latest_injury_cutoff:
                    latest_injury_cutoff = last_game

        if not latest_injury_cutoff:
            return game_logs

        # Filter to games AFTER the cutoff date
        # These are games where all currently injured teammates were also out
        filtered = []
        for log in game_logs:
            game_date = log.date[:10] if len(log.date) >= 10 else log.date
            if game_date > latest_injury_cutoff:
                filtered.append(log)

        logger.debug(
            f"Roster filter: {len(game_logs)} games -> {len(filtered)} games "
            f"(after {latest_injury_cutoff})"
        )

        return filtered

    def apply_roster_filtering_to_profile(
        self,
        profile: PlayerProfile,
        injured_teammate_names: List[str]
    ) -> None:
        """
        Apply roster-aware filtering to a player profile.

        Calculates averages from games where injured teammates were also out,
        preventing double-counting of injury boosts.

        Args:
            profile: Player profile to update (modified in place)
            injured_teammate_names: List of OUT teammate names (same team)
        """
        if not injured_teammate_names or not profile.game_logs:
            profile.roster_filtering_applied = False
            return

        # Get last game date for each injured teammate
        injured_last_games = {}
        for name in injured_teammate_names:
            last_game = self.get_player_last_game_date(name)
            if last_game:
                injured_last_games[name] = last_game
                logger.debug(f"Injured teammate {name} last played: {last_game}")

        if not injured_last_games:
            profile.roster_filtering_applied = False
            return

        # Filter game logs to games where injured teammates were also out
        played_games = [g for g in profile.game_logs if g.minutes > 0]
        filtered_games = self.filter_games_by_injured_roster(played_games, injured_last_games)

        profile.roster_filtered_games_count = len(filtered_games)

        # Need at least 3 games with this roster for meaningful averages
        if len(filtered_games) >= 3:
            profile.roster_filtered_l3_avg = self._calculate_averages(filtered_games[:3])
            profile.roster_filtered_l5_avg = self._calculate_averages(filtered_games[:5])
            profile.roster_filtering_applied = True

            logger.info(
                f"Roster-filtered averages for {profile.name}: "
                f"{len(filtered_games)} games with current roster. "
                f"L3: {profile.roster_filtered_l3_avg.get('pts', 0):.1f}pts vs "
                f"Unfiltered L3: {profile.last_3_avg.get('pts', 0):.1f}pts"
            )
        else:
            profile.roster_filtering_applied = False
            logger.debug(
                f"Not enough roster-filtered games for {profile.name}: "
                f"{len(filtered_games)} games (need 3+)"
            )

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

        # Get live props - try PropLine bulk first, then Odds API, then Balldontlie
        raw_props = []
        props_source = "none"

        # Try PropLine first (fastest - one bulk call for ALL games)
        if self.use_propline:
            propline_props = self._get_propline_props_for_game(home_team, away_team)
            if propline_props:
                raw_props = propline_props
                props_source = "propline"
                logger.info(f"Using PropLine for props ({len(raw_props)} props)")

        # Fallback to The Odds API
        if not raw_props:
            odds_api_event_id = self.odds_client.find_event_id(home_team, away_team)
            if odds_api_event_id:
                raw_props = self.odds_client.get_player_props(odds_api_event_id)
                if raw_props:
                    props_source = "odds_api"
                    logger.info(f"Using The Odds API for props ({len(raw_props)} props)")

        # Fallback to Balldontlie if nothing else worked
        if not raw_props and game_id:
            logger.info("Falling back to Balldontlie for props")
            raw_props = self.client.get_player_props(game_id)
            props_source = "balldontlie"

        # Process props based on source
        if props_source in ("propline", "odds_api"):
            # PropLine/Odds API return player_name directly; may include cross-book data
            for prop in raw_props:
                context.props.append(PlayerProp(
                    player_id=0,
                    player_name=prop.get("player_name", ""),
                    prop_type=prop.get("prop_type", ""),
                    line=prop.get("line", 0),
                    over_odds=prop.get("over_odds", -110),
                    under_odds=prop.get("under_odds", -110),
                    best_over_book=prop.get("best_over_book", "draftkings"),
                    best_under_book=prop.get("best_under_book", "draftkings"),
                    dk_line=prop.get("dk_line", prop.get("line", 0)),
                    dk_over_odds=prop.get("dk_over_odds", prop.get("over_odds", -110)),
                    dk_under_odds=prop.get("dk_under_odds", prop.get("under_odds", -110)),
                    all_books=prop.get("all_books", {}),
                ))
        elif props_source == "balldontlie":
            # Balldontlie returns player_id, need to look up names
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
    # PropLine Integration (fast bulk props)
    # -------------------------------------------------------------------------

    def _ensure_propline_bulk_cache(self):
        """Load PropLine bulk props into cache if not already loaded."""
        if self._propline_bulk_cache is None and self.use_propline:
            self._propline_bulk_cache = self.propline_client.get_all_props()

    def _get_propline_props_for_game(self, home_team: str, away_team: str) -> List[Dict]:
        """
        Get props for a specific game using BEST lines across all bookmakers.

        Uses PropLine's multi-book data to find the best available line per prop.
        Each prop dict includes best_over_book, best_under_book, all_books for display.
        Falls back to DK-only if multi-book fetch fails.
        """
        if not self.use_propline:
            return []

        # Try best-lines-all-books (60s TTL — fresh enough for live, fast enough for batch)
        import time as _time
        cache_age = _time.time() - getattr(self, '_best_lines_ts', 0)
        if cache_age > 60 or not hasattr(self, '_best_lines_cache') or self._best_lines_cache is None:
            try:
                self._best_lines_cache = self.propline_client.get_best_lines_all_books()
                self._best_lines_ts = _time.time()
            except Exception as e:
                logger.debug(f"Best-lines fetch failed, falling back to DK: {e}")
                if not hasattr(self, '_best_lines_cache'):
                    self._best_lines_cache = {}

        if self._best_lines_cache:
            event_id = self.propline_client.find_event_id(home_team, away_team)
            if event_id and event_id in self._best_lines_cache:
                return self._best_lines_cache[event_id]

        # Fallback: DK-only from bulk cache
        self._ensure_propline_bulk_cache()
        if not self._propline_bulk_cache:
            return []

        event_id = self.propline_client.find_event_id(home_team, away_team)
        if event_id and event_id in self._propline_bulk_cache:
            return self._propline_bulk_cache[event_id]

        if event_id:
            props = self.propline_client.get_game_props(event_id)
            if props:
                return props

        return []

    def get_propline_active_players(self, home_team: str, away_team: str) -> List[Dict]:
        """
        Get active players for a game using PropLine props data.

        If DK has props on a player, they're playing. This avoids the slow
        Balldontlie player-by-player lookups.

        Returns: [{"player_name": "Jalen Brunson"}, ...]
        """
        props = self._get_propline_props_for_game(home_team, away_team)
        seen = set()
        players = []
        for prop in props:
            name = prop.get("player_name", "")
            if name and name not in seen:
                seen.add(name)
                players.append({"player_name": name})
        return players

    # -------------------------------------------------------------------------
    # Active Roster Detection (from DraftKings props)
    # -------------------------------------------------------------------------

    def get_active_players_for_game(self, game_id: int, home_team: str = None, away_team: str = None) -> List[Dict]:
        """
        Get active players for a game using DraftKings props.

        This is the BEST way to get active rosters - if DK has props on a player,
        they're playing tonight. Much better than using stale team roster data.

        If home_team/away_team are provided and PropLine is available, uses the
        fast PropLine path (no per-player API lookups needed).

        Returns list of {player_id, player_name, team_id, props}
        """
        # Try PropLine first if team names are available (fast path)
        if self.use_propline and home_team and away_team:
            propline_players = self.get_propline_active_players(home_team, away_team)
            if propline_players:
                logger.info(f"PropLine: {len(propline_players)} active players for {away_team} @ {home_team}")
                return propline_players

        # Fallback to Balldontlie (slow path - per-player API lookups)
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

    def get_active_players_from_odds_api(self, home_team: str, away_team: str) -> List[Dict]:
        """
        Get active players for a game using The Odds API.

        This is the BEST way to get active rosters - if DK has props on a player,
        they're playing tonight.

        Returns list of {player_name, props}
        """
        event_id = self.odds_client.find_event_id(home_team, away_team)
        if not event_id:
            logger.warning(f"No Odds API event found for {away_team} @ {home_team}")
            return []

        raw_props = self.odds_client.get_player_props(event_id)

        # Group props by player name
        player_props = {}
        for prop in raw_props:
            player_name = prop.get("player_name")
            if player_name not in player_props:
                player_props[player_name] = {
                    "player_name": player_name,
                    "props": []
                }
            player_props[player_name]["props"].append(prop)

        # Sort by number of props (more props = more important player)
        active_players = list(player_props.values())
        active_players.sort(key=lambda x: len(x["props"]), reverse=True)

        return active_players

    def get_active_player_names_from_odds_api(self, home_team: str, away_team: str) -> List[str]:
        """Get just the names of active players using The Odds API"""
        active = self.get_active_players_from_odds_api(home_team, away_team)
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
        if NBA_API_AVAILABLE:
            self.tracking_client.clear_cache()
        self.pbp_client.clear_cache()
        self.ctg_client.clear_cache()
        self._propline_bulk_cache = None
        self._best_lines_cache = None
        if self.propline_client:
            self.propline_client.clear_cache()


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
