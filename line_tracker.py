#!/usr/bin/env python3
"""
Line Movement Tracker

Tracks opening vs closing lines for player props to:
1. Detect line movement direction
2. Calculate Closing Line Value (CLV)
3. Identify sharp money moves (line moves against public betting)
4. Provide historical line data for model features

Usage:
    from line_tracker import LineTracker

    tracker = LineTracker()

    # Record a line when first seen
    tracker.record_line(
        game_id=12345,
        player_name="LeBron James",
        prop_type="points",
        line=25.5,
        over_odds=-110,
        under_odds=-110
    )

    # Get line movement info
    movement = tracker.get_line_movement(12345, "LeBron James", "points")
    # Returns: {"opening_line": 25.5, "current_line": 26.5, "movement": +1.0, ...}

    # Calculate CLV for a pick
    clv = tracker.calculate_clv(
        game_id=12345,
        player_name="LeBron James",
        prop_type="points",
        pick_direction="OVER",
        pick_line=25.5
    )
"""

import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List, Tuple

logger = logging.getLogger(__name__)


class LineTracker:
    """
    SQLite-based line movement tracker for player props.
    """

    def __init__(self, db_path: str = None):
        """
        Initialize the line tracker.

        Args:
            db_path: Path to SQLite database. Defaults to 'lines.db' in project root.
        """
        if db_path is None:
            db_path = str(Path(__file__).parent / 'lines.db')

        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Main line movements table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS line_movements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER NOT NULL,
                player_name TEXT NOT NULL,
                prop_type TEXT NOT NULL,
                opening_line REAL,
                opening_over_odds INTEGER,
                opening_under_odds INTEGER,
                current_line REAL,
                current_over_odds INTEGER,
                current_under_odds INTEGER,
                first_seen TIMESTAMP,
                last_updated TIMESTAMP,
                game_date TEXT,
                UNIQUE(game_id, player_name, prop_type)
            )
        """)

        # Line history for tracking changes over time
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS line_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER NOT NULL,
                player_name TEXT NOT NULL,
                prop_type TEXT NOT NULL,
                line REAL,
                over_odds INTEGER,
                under_odds INTEGER,
                recorded_at TIMESTAMP
            )
        """)

        # Create indexes for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_movements_game
            ON line_movements(game_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_movements_player
            ON line_movements(player_name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_history_game
            ON line_history(game_id)
        """)

        conn.commit()
        conn.close()

    def record_line(self, game_id: int, player_name: str, prop_type: str,
                    line: float, over_odds: int = -110, under_odds: int = -110,
                    game_date: str = None) -> Dict:
        """
        Record a line. If this is the first time seeing it, store as opening.
        If we've seen it before, update as current line.

        Returns:
            Dict with opening line, current line, and movement info
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        # Check if we already have this line
        cursor.execute("""
            SELECT opening_line, opening_over_odds, opening_under_odds,
                   current_line, current_over_odds, current_under_odds,
                   first_seen
            FROM line_movements
            WHERE game_id = ? AND player_name = ? AND prop_type = ?
        """, (game_id, player_name, prop_type))

        existing = cursor.fetchone()

        if existing is None:
            # First time seeing this prop - record as opening
            cursor.execute("""
                INSERT INTO line_movements
                (game_id, player_name, prop_type, opening_line, opening_over_odds,
                 opening_under_odds, current_line, current_over_odds, current_under_odds,
                 first_seen, last_updated, game_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (game_id, player_name, prop_type, line, over_odds, under_odds,
                  line, over_odds, under_odds, now, now, game_date or now[:10]))

            # Also record in history
            cursor.execute("""
                INSERT INTO line_history
                (game_id, player_name, prop_type, line, over_odds, under_odds, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (game_id, player_name, prop_type, line, over_odds, under_odds, now))

            conn.commit()
            conn.close()

            return {
                "is_new": True,
                "opening_line": line,
                "current_line": line,
                "movement": 0.0,
                "opening_over_odds": over_odds,
                "current_over_odds": over_odds,
                "odds_movement": 0
            }

        else:
            # We've seen this before - update current
            (opening_line, opening_over, opening_under,
             old_line, old_over, old_under, first_seen) = existing

            # Only record history if line actually changed
            if line != old_line or over_odds != old_over:
                cursor.execute("""
                    INSERT INTO line_history
                    (game_id, player_name, prop_type, line, over_odds, under_odds, recorded_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (game_id, player_name, prop_type, line, over_odds, under_odds, now))

            # Update current line
            cursor.execute("""
                UPDATE line_movements
                SET current_line = ?, current_over_odds = ?, current_under_odds = ?,
                    last_updated = ?
                WHERE game_id = ? AND player_name = ? AND prop_type = ?
            """, (line, over_odds, under_odds, now, game_id, player_name, prop_type))

            conn.commit()
            conn.close()

            movement = line - opening_line
            odds_movement = over_odds - opening_over

            return {
                "is_new": False,
                "opening_line": opening_line,
                "current_line": line,
                "movement": movement,
                "opening_over_odds": opening_over,
                "current_over_odds": over_odds,
                "odds_movement": odds_movement
            }

    def get_line_movement(self, game_id: int, player_name: str,
                          prop_type: str) -> Optional[Dict]:
        """
        Get line movement info for a specific prop.

        Returns:
            Dict with opening line, current line, movement, and CLV signals
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT opening_line, opening_over_odds, opening_under_odds,
                   current_line, current_over_odds, current_under_odds,
                   first_seen, last_updated
            FROM line_movements
            WHERE game_id = ? AND player_name = ? AND prop_type = ?
        """, (game_id, player_name, prop_type))

        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        (opening_line, opening_over, opening_under,
         current_line, current_over, current_under,
         first_seen, last_updated) = row

        movement = current_line - opening_line

        # Determine if this is sharp money (line moved significantly)
        is_sharp_over = movement > 0.5  # Line moved up = sharp on OVER
        is_sharp_under = movement < -0.5  # Line moved down = sharp on UNDER

        # Odds movement
        over_odds_shift = current_over - opening_over
        under_odds_shift = current_under - opening_under

        return {
            "opening_line": opening_line,
            "current_line": current_line,
            "movement": movement,
            "movement_pct": round((movement / opening_line) * 100, 2) if opening_line else 0,
            "opening_over_odds": opening_over,
            "current_over_odds": current_over,
            "opening_under_odds": opening_under,
            "current_under_odds": current_under,
            "over_odds_shift": over_odds_shift,
            "under_odds_shift": under_odds_shift,
            "is_sharp_over": is_sharp_over,
            "is_sharp_under": is_sharp_under,
            "first_seen": first_seen,
            "last_updated": last_updated
        }

    def calculate_clv(self, game_id: int, player_name: str, prop_type: str,
                      pick_direction: str, pick_line: float) -> Optional[Dict]:
        """
        Calculate Closing Line Value (CLV) for a pick.

        CLV measures whether you beat the closing line.
        Positive CLV = you got a better line than the market settled on.

        Args:
            game_id: Game ID
            player_name: Player name
            prop_type: Prop type
            pick_direction: "OVER" or "UNDER"
            pick_line: The line you took

        Returns:
            Dict with CLV calculation and interpretation
        """
        movement_info = self.get_line_movement(game_id, player_name, prop_type)

        if movement_info is None:
            return None

        closing_line = movement_info['current_line']

        if pick_direction.upper() == 'OVER':
            # For OVER: CLV positive if closing > pick line (you got lower)
            clv = closing_line - pick_line
            beat_close = clv > 0
        else:
            # For UNDER: CLV positive if closing < pick line (you got higher)
            clv = pick_line - closing_line
            beat_close = clv > 0

        # Interpret the CLV
        if clv >= 1.0:
            interpretation = "Excellent CLV - you significantly beat the close"
        elif clv >= 0.5:
            interpretation = "Good CLV - you beat the close"
        elif clv > 0:
            interpretation = "Slight positive CLV"
        elif clv == 0:
            interpretation = "No CLV - picked at closing line"
        elif clv > -0.5:
            interpretation = "Slight negative CLV"
        else:
            interpretation = "Poor CLV - line moved against you"

        return {
            "pick_line": pick_line,
            "closing_line": closing_line,
            "pick_direction": pick_direction.upper(),
            "clv": round(clv, 2),
            "clv_pct": round((clv / pick_line) * 100, 2) if pick_line else 0,
            "beat_close": beat_close,
            "interpretation": interpretation,
            "line_movement": movement_info['movement'],
            "is_sharp_side": (
                (pick_direction.upper() == 'OVER' and movement_info['is_sharp_over']) or
                (pick_direction.upper() == 'UNDER' and movement_info['is_sharp_under'])
            )
        }

    def get_all_movements_for_game(self, game_id: int) -> List[Dict]:
        """Get all line movements for a specific game."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT player_name, prop_type, opening_line, current_line,
                   opening_over_odds, current_over_odds,
                   opening_under_odds, current_under_odds
            FROM line_movements
            WHERE game_id = ?
            ORDER BY player_name, prop_type
        """, (game_id,))

        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            movement = row[3] - row[2]  # current - opening
            results.append({
                "player_name": row[0],
                "prop_type": row[1],
                "opening_line": row[2],
                "current_line": row[3],
                "movement": movement,
                "opening_over_odds": row[4],
                "current_over_odds": row[5]
            })

        return results

    def get_line_history(self, game_id: int, player_name: str,
                         prop_type: str) -> List[Dict]:
        """Get full line history for a specific prop."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT line, over_odds, under_odds, recorded_at
            FROM line_history
            WHERE game_id = ? AND player_name = ? AND prop_type = ?
            ORDER BY recorded_at
        """, (game_id, player_name, prop_type))

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "line": row[0],
                "over_odds": row[1],
                "under_odds": row[2],
                "recorded_at": row[3]
            }
            for row in rows
        ]

    def cleanup_old_data(self, days_to_keep: int = 30):
        """Remove line data older than specified days."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff = (datetime.now() - timedelta(days=days_to_keep)).isoformat()

        cursor.execute("""
            DELETE FROM line_movements WHERE last_updated < ?
        """, (cutoff,))

        cursor.execute("""
            DELETE FROM line_history WHERE recorded_at < ?
        """, (cutoff,))

        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        logger.info(f"Cleaned up {deleted} old line records")
        return deleted


# Convenience function for easy import
_line_tracker_instance = None


def get_line_tracker() -> LineTracker:
    """Get singleton LineTracker instance."""
    global _line_tracker_instance
    if _line_tracker_instance is None:
        _line_tracker_instance = LineTracker()
    return _line_tracker_instance


if __name__ == "__main__":
    # Demo usage
    tracker = LineTracker()

    # Simulate recording lines at different times
    print("Recording opening line...")
    result = tracker.record_line(
        game_id=12345,
        player_name="LeBron James",
        prop_type="points",
        line=25.5,
        over_odds=-110,
        under_odds=-110
    )
    print(f"  Result: {result}")

    # Simulate line movement
    print("\nRecording updated line...")
    result = tracker.record_line(
        game_id=12345,
        player_name="LeBron James",
        prop_type="points",
        line=26.5,
        over_odds=-120,
        under_odds=+100
    )
    print(f"  Result: {result}")

    # Get movement info
    print("\nGetting line movement...")
    movement = tracker.get_line_movement(12345, "LeBron James", "points")
    print(f"  Movement: {movement}")

    # Calculate CLV for OVER pick at 25.5
    print("\nCalculating CLV for OVER 25.5...")
    clv = tracker.calculate_clv(12345, "LeBron James", "points", "OVER", 25.5)
    print(f"  CLV: {clv}")

    # Calculate CLV for UNDER pick at 26.5
    print("\nCalculating CLV for UNDER 26.5...")
    clv = tracker.calculate_clv(12345, "LeBron James", "points", "UNDER", 26.5)
    print(f"  CLV: {clv}")
