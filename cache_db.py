#!/usr/bin/env python3
"""
SQLite Cache Database for NBA Betting Model

Caches:
- Player profiles (24hr TTL)
- Game logs (permanent - historical data doesn't change)
- Season averages (1hr TTL)
- Injuries (15min TTL - changes frequently)
- Team data (24hr TTL)
"""

import sqlite3
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "cache.db")

# TTL in seconds
TTL_PLAYER = 24 * 60 * 60      # 24 hours
TTL_GAME_LOG = 365 * 24 * 60 * 60  # 1 year (permanent for historical)
TTL_SEASON_AVG = 60 * 60       # 1 hour
TTL_INJURIES = 15 * 60         # 15 minutes
TTL_TEAM = 24 * 60 * 60        # 24 hours
TTL_PROPS = 5 * 60             # 5 minutes (live data)


class CacheDB:
    """SQLite cache for API responses"""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Players table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS players (
                id INTEGER PRIMARY KEY,
                name TEXT,
                team TEXT,
                team_id INTEGER,
                position TEXT,
                data TEXT,
                updated_at REAL
            )
        """)

        # Player name lookup (for fuzzy matching)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_names (
                search_name TEXT PRIMARY KEY,
                player_id INTEGER,
                updated_at REAL
            )
        """)

        # Game logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS game_logs (
                player_id INTEGER,
                game_id INTEGER,
                game_date TEXT,
                data TEXT,
                updated_at REAL,
                PRIMARY KEY (player_id, game_id)
            )
        """)

        # Season averages
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS season_averages (
                player_id INTEGER,
                season INTEGER,
                data TEXT,
                updated_at REAL,
                PRIMARY KEY (player_id, season)
            )
        """)

        # Injuries
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS injuries (
                team_id INTEGER PRIMARY KEY,
                data TEXT,
                updated_at REAL
            )
        """)

        # Teams
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS teams (
                id INTEGER PRIMARY KEY,
                name TEXT,
                abbreviation TEXT,
                data TEXT,
                updated_at REAL
            )
        """)

        # Props cache (short TTL)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS props (
                game_id INTEGER PRIMARY KEY,
                data TEXT,
                updated_at REAL
            )
        """)

        # Games cache
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS games (
                id INTEGER PRIMARY KEY,
                date TEXT,
                home_team TEXT,
                away_team TEXT,
                data TEXT,
                updated_at REAL
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_name ON players(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_game_logs_date ON game_logs(game_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_date ON games(date)")

        # =====================================================================
        # HISTORICAL ODDS TABLES (from The Odds API)
        # These store REAL betting lines for backtesting
        # =====================================================================

        # Historical games from The Odds API
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historical_games (
                event_id TEXT PRIMARY KEY,
                sport_key TEXT,
                commence_time TEXT,
                home_team TEXT,
                away_team TEXT,
                snapshot_time TEXT,
                created_at REAL
            )
        """)

        # Historical player props with real betting lines
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historical_player_props (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT,
                bookmaker TEXT,
                market TEXT,
                player_name TEXT,
                line REAL,
                over_odds INTEGER,
                under_odds INTEGER,
                snapshot_time TEXT,
                created_at REAL,
                UNIQUE(event_id, bookmaker, market, player_name, line)
            )
        """)

        # Indexes for historical odds queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hist_props_player ON historical_player_props(player_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hist_props_event ON historical_player_props(event_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hist_props_market ON historical_player_props(market)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hist_games_date ON historical_games(commence_time)")

        # Track API usage for The Odds API
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS odds_api_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint TEXT,
                credits_used INTEGER,
                credits_remaining INTEGER,
                timestamp TEXT,
                notes TEXT
            )
        """)

        conn.commit()
        conn.close()

    def _is_expired(self, updated_at: float, ttl: int) -> bool:
        """Check if cache entry is expired"""
        if updated_at is None:
            return True
        return (time.time() - updated_at) > ttl

    # -------------------------------------------------------------------------
    # Player Methods
    # -------------------------------------------------------------------------

    def get_player(self, player_id: int) -> Optional[Dict]:
        """Get player by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT data, updated_at FROM players WHERE id = ?", (player_id,))
        row = cursor.fetchone()
        conn.close()

        if row and not self._is_expired(row[1], TTL_PLAYER):
            return json.loads(row[0])
        return None

    def get_player_by_name(self, name: str) -> Optional[Dict]:
        """Get player by name (uses name lookup cache)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # First check name lookup cache
        search_name = name.lower().strip()
        cursor.execute(
            "SELECT player_id, updated_at FROM player_names WHERE search_name = ?",
            (search_name,)
        )
        row = cursor.fetchone()

        if row and not self._is_expired(row[1], TTL_PLAYER):
            player_id = row[0]
            cursor.execute("SELECT data, updated_at FROM players WHERE id = ?", (player_id,))
            player_row = cursor.fetchone()
            conn.close()
            if player_row and not self._is_expired(player_row[1], TTL_PLAYER):
                return json.loads(player_row[0])
            return None

        conn.close()
        return None

    def set_player(self, player_id: int, data: Dict, name: str = None):
        """Cache player data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = time.time()

        cursor.execute("""
            INSERT OR REPLACE INTO players (id, name, team, team_id, position, data, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            player_id,
            data.get("first_name", "") + " " + data.get("last_name", ""),
            data.get("team", {}).get("full_name", ""),
            data.get("team", {}).get("id", 0),
            data.get("position", ""),
            json.dumps(data),
            now
        ))

        # Also cache name lookup
        if name:
            search_name = name.lower().strip()
            cursor.execute("""
                INSERT OR REPLACE INTO player_names (search_name, player_id, updated_at)
                VALUES (?, ?, ?)
            """, (search_name, player_id, now))

        conn.commit()
        conn.close()

    # -------------------------------------------------------------------------
    # Game Log Methods
    # -------------------------------------------------------------------------

    def get_game_logs(self, player_id: int, limit: int = 20) -> List[Dict]:
        """Get cached game logs for player"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT data FROM game_logs
            WHERE player_id = ?
            ORDER BY game_date DESC
            LIMIT ?
        """, (player_id, limit))
        rows = cursor.fetchall()
        conn.close()

        return [json.loads(row[0]) for row in rows]

    def get_game_logs_count(self, player_id: int) -> int:
        """Get count of cached game logs"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM game_logs WHERE player_id = ?", (player_id,))
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def set_game_logs(self, player_id: int, logs: List[Dict]):
        """Cache game logs for player"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = time.time()

        for log in logs:
            game_id = log.get("game", {}).get("id", 0)
            game_date = log.get("game", {}).get("date", "")

            cursor.execute("""
                INSERT OR REPLACE INTO game_logs (player_id, game_id, game_date, data, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (player_id, game_id, game_date, json.dumps(log), now))

        conn.commit()
        conn.close()

    # -------------------------------------------------------------------------
    # Season Average Methods
    # -------------------------------------------------------------------------

    def get_season_average(self, player_id: int, season: int = 2025) -> Optional[Dict]:
        """Get cached season average"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT data, updated_at FROM season_averages WHERE player_id = ? AND season = ?",
            (player_id, season)
        )
        row = cursor.fetchone()
        conn.close()

        if row and not self._is_expired(row[1], TTL_SEASON_AVG):
            return json.loads(row[0])
        return None

    def set_season_average(self, player_id: int, data: Dict, season: int = 2025):
        """Cache season average"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO season_averages (player_id, season, data, updated_at)
            VALUES (?, ?, ?, ?)
        """, (player_id, season, json.dumps(data), time.time()))
        conn.commit()
        conn.close()

    # -------------------------------------------------------------------------
    # Injuries Methods
    # -------------------------------------------------------------------------

    def get_injuries(self, team_id: int) -> Optional[List[Dict]]:
        """Get cached injuries for team"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT data, updated_at FROM injuries WHERE team_id = ?", (team_id,))
        row = cursor.fetchone()
        conn.close()

        if row and not self._is_expired(row[1], TTL_INJURIES):
            return json.loads(row[0])
        return None

    def set_injuries(self, team_id: int, data: List[Dict]):
        """Cache injuries for team"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO injuries (team_id, data, updated_at)
            VALUES (?, ?, ?)
        """, (team_id, json.dumps(data), time.time()))
        conn.commit()
        conn.close()

    # -------------------------------------------------------------------------
    # Props Methods
    # -------------------------------------------------------------------------

    def get_props(self, game_id: int) -> Optional[List[Dict]]:
        """Get cached props for game"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT data, updated_at FROM props WHERE game_id = ?", (game_id,))
        row = cursor.fetchone()
        conn.close()

        if row and not self._is_expired(row[1], TTL_PROPS):
            return json.loads(row[0])
        return None

    def set_props(self, game_id: int, data: List[Dict]):
        """Cache props for game"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO props (game_id, data, updated_at)
            VALUES (?, ?, ?)
        """, (game_id, json.dumps(data), time.time()))
        conn.commit()
        conn.close()

    # -------------------------------------------------------------------------
    # Games Methods
    # -------------------------------------------------------------------------

    def get_games_for_date(self, date: str) -> Optional[List[Dict]]:
        """Get cached games for date"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT data FROM games WHERE date = ?", (date,))
        rows = cursor.fetchall()
        conn.close()

        if rows:
            return [json.loads(row[0]) for row in rows]
        return None

    def set_games(self, games: List[Dict]):
        """Cache games"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = time.time()

        for game in games:
            cursor.execute("""
                INSERT OR REPLACE INTO games (id, date, home_team, away_team, data, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                game.get("id"),
                game.get("date"),
                game.get("home_team", {}).get("full_name", ""),
                game.get("visitor_team", {}).get("full_name", ""),
                json.dumps(game),
                now
            ))

        conn.commit()
        conn.close()

    # -------------------------------------------------------------------------
    # Historical Odds Methods (The Odds API)
    # -------------------------------------------------------------------------

    def save_historical_game(self, event_id: str, sport_key: str, commence_time: str,
                             home_team: str, away_team: str, snapshot_time: str):
        """Save a historical game from The Odds API"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO historical_games
            (event_id, sport_key, commence_time, home_team, away_team, snapshot_time, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (event_id, sport_key, commence_time, home_team, away_team, snapshot_time, time.time()))
        conn.commit()
        conn.close()

    def save_historical_player_prop(self, event_id: str, bookmaker: str, market: str,
                                    player_name: str, line: float, over_odds: int,
                                    under_odds: int, snapshot_time: str):
        """Save a historical player prop from The Odds API"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO historical_player_props
            (event_id, bookmaker, market, player_name, line, over_odds, under_odds, snapshot_time, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (event_id, bookmaker, market, player_name, line, over_odds, under_odds, snapshot_time, time.time()))
        conn.commit()
        conn.close()

    def save_historical_props_batch(self, props: List[Dict]):
        """Save multiple player props in a single transaction (more efficient)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = time.time()

        for prop in props:
            cursor.execute("""
                INSERT OR REPLACE INTO historical_player_props
                (event_id, bookmaker, market, player_name, line, over_odds, under_odds, snapshot_time, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prop['event_id'], prop['bookmaker'], prop['market'],
                prop['player_name'], prop['line'], prop['over_odds'],
                prop['under_odds'], prop['snapshot_time'], now
            ))

        conn.commit()
        conn.close()

    def get_historical_props_for_game(self, event_id: str, bookmaker: str = None) -> List[Dict]:
        """Get all historical props for a game"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if bookmaker:
            cursor.execute("""
                SELECT market, player_name, line, over_odds, under_odds, snapshot_time
                FROM historical_player_props
                WHERE event_id = ? AND bookmaker = ?
            """, (event_id, bookmaker))
        else:
            cursor.execute("""
                SELECT bookmaker, market, player_name, line, over_odds, under_odds, snapshot_time
                FROM historical_player_props
                WHERE event_id = ?
            """, (event_id,))

        rows = cursor.fetchall()
        conn.close()

        if bookmaker:
            return [{"market": r[0], "player_name": r[1], "line": r[2],
                     "over_odds": r[3], "under_odds": r[4], "snapshot_time": r[5]} for r in rows]
        else:
            return [{"bookmaker": r[0], "market": r[1], "player_name": r[2], "line": r[3],
                     "over_odds": r[4], "under_odds": r[5], "snapshot_time": r[6]} for r in rows]

    def get_historical_games_for_date(self, date: str) -> List[Dict]:
        """Get all historical games for a date (YYYY-MM-DD format)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT event_id, sport_key, commence_time, home_team, away_team, snapshot_time
            FROM historical_games
            WHERE commence_time LIKE ?
            ORDER BY commence_time
        """, (f"{date}%",))
        rows = cursor.fetchall()
        conn.close()

        return [{"event_id": r[0], "sport_key": r[1], "commence_time": r[2],
                 "home_team": r[3], "away_team": r[4], "snapshot_time": r[5]} for r in rows]

    def get_prop_for_player_game(self, player_name: str, event_id: str,
                                  market: str, bookmaker: str = "draftkings") -> Optional[Dict]:
        """Get a specific prop for a player in a game"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT line, over_odds, under_odds, snapshot_time
            FROM historical_player_props
            WHERE player_name = ? AND event_id = ? AND market = ? AND bookmaker = ?
        """, (player_name, event_id, market, bookmaker))
        row = cursor.fetchone()
        conn.close()

        if row:
            return {"line": row[0], "over_odds": row[1], "under_odds": row[2], "snapshot_time": row[3]}
        return None

    def log_odds_api_usage(self, endpoint: str, credits_used: int, credits_remaining: int, notes: str = ""):
        """Log API usage for tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO odds_api_usage (endpoint, credits_used, credits_remaining, timestamp, notes)
            VALUES (?, ?, ?, ?, ?)
        """, (endpoint, credits_used, credits_remaining, datetime.now().isoformat(), notes))
        conn.commit()
        conn.close()

    def get_odds_api_usage_total(self) -> int:
        """Get total credits used from The Odds API"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT SUM(credits_used) FROM odds_api_usage")
        result = cursor.fetchone()[0]
        conn.close()
        return result or 0

    def get_historical_odds_stats(self) -> Dict:
        """Get statistics on historical odds data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM historical_games")
        games = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM historical_player_props")
        props = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT player_name) FROM historical_player_props")
        players = cursor.fetchone()[0]

        cursor.execute("SELECT MIN(commence_time), MAX(commence_time) FROM historical_games")
        date_range = cursor.fetchone()

        conn.close()

        return {
            "games": games,
            "props": props,
            "unique_players": players,
            "earliest_game": date_range[0],
            "latest_game": date_range[1],
            "credits_used": self.get_odds_api_usage_total()
        }

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def clear_expired(self):
        """Clear all expired entries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = time.time()

        # Clear expired players
        cursor.execute("DELETE FROM players WHERE ? - updated_at > ?", (now, TTL_PLAYER))
        cursor.execute("DELETE FROM player_names WHERE ? - updated_at > ?", (now, TTL_PLAYER))

        # Clear expired season averages
        cursor.execute("DELETE FROM season_averages WHERE ? - updated_at > ?", (now, TTL_SEASON_AVG))

        # Clear expired injuries
        cursor.execute("DELETE FROM injuries WHERE ? - updated_at > ?", (now, TTL_INJURIES))

        # Clear expired props
        cursor.execute("DELETE FROM props WHERE ? - updated_at > ?", (now, TTL_PROPS))

        # Game logs are permanent, don't clear
        # Games are permanent, don't clear

        conn.commit()
        conn.close()

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}
        for table in ["players", "player_names", "game_logs", "season_averages", "injuries", "props", "games"]:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = cursor.fetchone()[0]

        conn.close()
        return stats

    def clear_all(self):
        """Clear all cache data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for table in ["players", "player_names", "game_logs", "season_averages", "injuries", "props", "games"]:
            cursor.execute(f"DELETE FROM {table}")
        conn.commit()
        conn.close()


# Global cache instance
_cache = None


def get_cache() -> CacheDB:
    """Get global cache instance"""
    global _cache
    if _cache is None:
        _cache = CacheDB()
    return _cache


if __name__ == "__main__":
    # Test the cache
    cache = get_cache()
    print(f"Cache database: {DB_PATH}")
    print(f"Stats: {cache.get_stats()}")
