"""
CLV (Closing Line Value) Tracking System

Tracks opening lines, closing lines, bet recommendations, and actual results
to compute CLV -- the single most important process metric for sports bettors.

If you consistently beat the closing line, you are profitable long-term
regardless of short-term variance.

Tables:
    line_snapshots      - Timestamped line observations (initial, movement, closing)
    bet_recommendations - Our model's picks with probability/edge at time of pick
    clv_results         - Computed CLV + actual results for each tracked bet

Usage:
    from web.clv_tracker import CLVTracker
    tracker = CLVTracker()

    # Log a line snapshot
    tracker.log_snapshot(game_id=123, player="LeBron James", prop_type="points",
                         line=25.5, over_odds=-110, under_odds=-110,
                         snapshot_type="initial")

    # Log a recommendation
    tracker.log_recommendation(game_id=123, player="LeBron James",
                                prop_type="points", side="over",
                                our_probability=0.58, edge=0.05,
                                line=25.5, odds=-110, model_version="v6")

    # Compute CLV after closing lines are captured
    tracker.compute_clv_for_date("2026-11-09")

    # Get CLV report
    report = tracker.get_clv_report(days=30)
"""

import sqlite3
import logging
import os
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List, Any

logger = logging.getLogger(__name__)

# Project root -- this file lives at web/clv_tracker.py
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DB_PATH = os.path.join(_PROJECT_ROOT, "data", "clv_tracking.db")


def _american_to_decimal(odds):
    """Convert American odds to decimal odds."""
    if odds is None or odds == 0:
        return 1.909  # default -110
    if odds > 0:
        return 1 + odds / 100.0
    return 1 + 100.0 / abs(odds)


def _american_to_implied(odds):
    """Convert American odds to implied probability (no-vig)."""
    if odds is None or odds == 0:
        return 0.5238  # -110
    if odds > 0:
        return 100.0 / (odds + 100)
    return abs(odds) / (abs(odds) + 100.0)


def _decimal_to_implied(decimal_odds):
    """Convert decimal odds to implied probability."""
    if decimal_odds <= 1:
        return 1.0
    return 1.0 / decimal_odds


class CLVTracker:
    """
    SQLite-backed CLV tracking system.

    Stores line snapshots, bet recommendations, and computed CLV results.
    All timestamps are ISO 8601 strings in local time.
    """

    def __init__(self, db_path=None):
        if db_path is None:
            db_path = _DEFAULT_DB_PATH
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _get_conn(self):
        """Get a connection with row_factory for dict-like access."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS line_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER NOT NULL,
                game_date TEXT,
                player TEXT NOT NULL,
                prop_type TEXT NOT NULL,
                book TEXT DEFAULT 'draftkings',
                line REAL NOT NULL,
                over_odds INTEGER,
                under_odds INTEGER,
                snapshot_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                best_book TEXT,
                best_odds INTEGER
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS bet_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER NOT NULL,
                game_date TEXT,
                player TEXT NOT NULL,
                prop_type TEXT NOT NULL,
                recommended_side TEXT NOT NULL,
                our_probability REAL,
                edge REAL,
                line_at_recommendation REAL,
                odds_at_recommendation INTEGER,
                over_odds INTEGER,
                under_odds INTEGER,
                predicted_value REAL,
                confidence REAL,
                kelly_fraction REAL,
                recommended_units REAL,
                best_book TEXT,
                best_odds INTEGER,
                model_version TEXT,
                timestamp TEXT NOT NULL
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS clv_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bet_id INTEGER NOT NULL,
                game_id INTEGER NOT NULL,
                game_date TEXT,
                player TEXT NOT NULL,
                prop_type TEXT NOT NULL,
                recommended_side TEXT NOT NULL,
                line_at_pick REAL,
                odds_at_pick INTEGER,
                closing_line REAL,
                closing_odds INTEGER,
                clv_line REAL,
                clv_implied REAL,
                clv_cents REAL,
                actual_result REAL,
                won INTEGER,
                push INTEGER DEFAULT 0,
                profit REAL,
                computed_at TEXT,
                FOREIGN KEY (bet_id) REFERENCES bet_recommendations(id)
            )
        """)

        # Indexes for fast lookups
        c.execute("""CREATE INDEX IF NOT EXISTS idx_snapshots_game
                     ON line_snapshots(game_id, player, prop_type)""")
        c.execute("""CREATE INDEX IF NOT EXISTS idx_snapshots_date
                     ON line_snapshots(game_date)""")
        c.execute("""CREATE INDEX IF NOT EXISTS idx_snapshots_type
                     ON line_snapshots(snapshot_type)""")
        c.execute("""CREATE INDEX IF NOT EXISTS idx_recs_game
                     ON bet_recommendations(game_id, player, prop_type)""")
        c.execute("""CREATE INDEX IF NOT EXISTS idx_recs_date
                     ON bet_recommendations(game_date)""")
        c.execute("""CREATE INDEX IF NOT EXISTS idx_clv_bet
                     ON clv_results(bet_id)""")
        c.execute("""CREATE INDEX IF NOT EXISTS idx_clv_date
                     ON clv_results(game_date)""")

        conn.commit()
        conn.close()
        logger.info("CLV tracking database initialized at %s", self.db_path)

    # ------------------------------------------------------------------
    # Line Snapshot Logging
    # ------------------------------------------------------------------

    def log_snapshot(self, game_id, player, prop_type, line, over_odds=-110,
                     under_odds=-110, snapshot_type="initial", book="draftkings",
                     game_date=None, best_book=None, best_odds=None):
        """
        Log a line snapshot.

        Args:
            snapshot_type: "initial" (first seen), "movement" (line changed),
                           "closing" (final line before tipoff), "poll" (periodic check)
        """
        if game_date is None:
            game_date = datetime.now().strftime("%Y-%m-%d")
        now = datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # For 'initial' type, check if we already have one for this prop
        if snapshot_type == "initial":
            c.execute("""
                SELECT id FROM line_snapshots
                WHERE game_id = ? AND player = ? AND prop_type = ?
                  AND book = ? AND snapshot_type = 'initial'
            """, (game_id, player, prop_type, book))
            if c.fetchone():
                # Already have an initial snapshot; downgrade to "poll"
                snapshot_type = "poll"

        c.execute("""
            INSERT INTO line_snapshots
            (game_id, game_date, player, prop_type, book, line, over_odds,
             under_odds, snapshot_type, timestamp, best_book, best_odds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (game_id, game_date, player, prop_type, book, line, over_odds,
              under_odds, snapshot_type, now, best_book, best_odds))

        conn.commit()
        conn.close()

    def log_snapshots_bulk(self, snapshots):
        """
        Log multiple line snapshots in a single transaction.

        Args:
            snapshots: list of dicts with keys matching log_snapshot params
        """
        if not snapshots:
            return

        now = datetime.now().isoformat()
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        for snap in snapshots:
            game_date = snap.get("game_date") or datetime.now().strftime("%Y-%m-%d")
            st = snap.get("snapshot_type", "poll")

            # Dedup initial snapshots
            if st == "initial":
                c.execute("""
                    SELECT id FROM line_snapshots
                    WHERE game_id = ? AND player = ? AND prop_type = ?
                      AND book = ? AND snapshot_type = 'initial'
                """, (snap["game_id"], snap["player"], snap["prop_type"],
                      snap.get("book", "draftkings")))
                if c.fetchone():
                    st = "poll"

            c.execute("""
                INSERT INTO line_snapshots
                (game_id, game_date, player, prop_type, book, line, over_odds,
                 under_odds, snapshot_type, timestamp, best_book, best_odds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snap["game_id"], game_date, snap["player"], snap["prop_type"],
                snap.get("book", "draftkings"), snap["line"],
                snap.get("over_odds", -110), snap.get("under_odds", -110),
                st, now, snap.get("best_book"), snap.get("best_odds"),
            ))

        conn.commit()
        conn.close()
        logger.debug("Logged %d line snapshots", len(snapshots))

    # ------------------------------------------------------------------
    # Bet Recommendation Logging
    # ------------------------------------------------------------------

    def log_recommendation(self, game_id, player, prop_type, side,
                           our_probability, edge, line, odds,
                           over_odds=-110, under_odds=-110,
                           predicted_value=None, confidence=None,
                           kelly_fraction=None, recommended_units=None,
                           best_book=None, best_odds=None,
                           model_version=None, game_date=None):
        """
        Log a bet recommendation from our model.

        Returns the recommendation ID for later CLV computation.
        """
        if game_date is None:
            game_date = datetime.now().strftime("%Y-%m-%d")
        now = datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Dedup: don't log the same recommendation twice for same game/player/prop/side/date
        c.execute("""
            SELECT id FROM bet_recommendations
            WHERE game_id = ? AND player = ? AND prop_type = ?
              AND recommended_side = ? AND game_date = ?
        """, (game_id, player, prop_type, side, game_date))
        existing = c.fetchone()
        if existing:
            conn.close()
            return existing[0]

        c.execute("""
            INSERT INTO bet_recommendations
            (game_id, game_date, player, prop_type, recommended_side,
             our_probability, edge, line_at_recommendation, odds_at_recommendation,
             over_odds, under_odds, predicted_value, confidence,
             kelly_fraction, recommended_units, best_book, best_odds,
             model_version, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            game_id, game_date, player, prop_type, side,
            our_probability, edge, line, odds,
            over_odds, under_odds, predicted_value, confidence,
            kelly_fraction, recommended_units, best_book, best_odds,
            model_version, now,
        ))

        rec_id = c.lastrowid
        conn.commit()
        conn.close()
        logger.debug("Logged recommendation #%d: %s %s %s %s", rec_id, player, prop_type, side, line)
        return rec_id

    def log_recommendations_bulk(self, recs):
        """
        Log multiple recommendations in a single transaction.

        Args:
            recs: list of dicts with keys matching log_recommendation params

        Returns:
            number of new recommendations logged
        """
        if not recs:
            return 0

        now = datetime.now().isoformat()
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        logged = 0

        for rec in recs:
            game_date = rec.get("game_date") or datetime.now().strftime("%Y-%m-%d")
            side = rec.get("side", rec.get("recommended_side", "over"))

            # Dedup
            c.execute("""
                SELECT id FROM bet_recommendations
                WHERE game_id = ? AND player = ? AND prop_type = ?
                  AND recommended_side = ? AND game_date = ?
            """, (rec["game_id"], rec["player"], rec["prop_type"], side, game_date))
            if c.fetchone():
                continue

            c.execute("""
                INSERT INTO bet_recommendations
                (game_id, game_date, player, prop_type, recommended_side,
                 our_probability, edge, line_at_recommendation, odds_at_recommendation,
                 over_odds, under_odds, predicted_value, confidence,
                 kelly_fraction, recommended_units, best_book, best_odds,
                 model_version, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rec["game_id"], game_date, rec["player"], rec["prop_type"], side,
                rec.get("our_probability", 0.5), rec.get("edge", 0),
                rec.get("line", 0), rec.get("odds", -110),
                rec.get("over_odds", -110), rec.get("under_odds", -110),
                rec.get("predicted_value"), rec.get("confidence"),
                rec.get("kelly_fraction"), rec.get("recommended_units"),
                rec.get("best_book"), rec.get("best_odds"),
                rec.get("model_version"), now,
            ))
            logged += 1

        conn.commit()
        conn.close()
        if logged:
            logger.info("Logged %d new bet recommendations", logged)
        return logged

    # ------------------------------------------------------------------
    # Closing Line Capture
    # ------------------------------------------------------------------

    def log_closing_lines(self, game_id, props_by_key, game_date=None):
        """
        Log closing lines for a game (called near tipoff).

        Args:
            game_id: BDL game ID
            props_by_key: {"player|prop_type": {"line": X, "over_odds": Y, "under_odds": Z}}
        """
        if not props_by_key:
            return

        if game_date is None:
            game_date = datetime.now().strftime("%Y-%m-%d")

        now = datetime.now().isoformat()
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        for pkey, pdata in props_by_key.items():
            parts = pkey.split("|", 1)
            if len(parts) != 2:
                continue
            player, prop_type = parts

            # Check if we already have a closing snapshot
            c.execute("""
                SELECT id FROM line_snapshots
                WHERE game_id = ? AND player = ? AND prop_type = ?
                  AND snapshot_type = 'closing'
            """, (game_id, player, prop_type))
            if c.fetchone():
                continue

            c.execute("""
                INSERT INTO line_snapshots
                (game_id, game_date, player, prop_type, book, line, over_odds,
                 under_odds, snapshot_type, timestamp)
                VALUES (?, ?, ?, ?, 'draftkings', ?, ?, ?, 'closing', ?)
            """, (
                game_id, game_date, player, prop_type,
                pdata.get("line", 0),
                pdata.get("over_odds", -110),
                pdata.get("under_odds", -110),
                now,
            ))

        conn.commit()
        conn.close()
        logger.info("Logged closing lines for game %d (%d props)", game_id, len(props_by_key))

    # ------------------------------------------------------------------
    # CLV Computation
    # ------------------------------------------------------------------

    def compute_clv_for_game(self, game_id, actual_results=None):
        """
        Compute CLV for all recommendations on a game.

        Args:
            game_id: BDL game ID
            actual_results: optional dict of {"player|prop_type": {"actual": X, "hit": bool}}

        Returns:
            list of computed CLV result dicts
        """
        conn = self._get_conn()
        c = conn.cursor()

        # Get recommendations for this game
        c.execute("""
            SELECT * FROM bet_recommendations
            WHERE game_id = ?
        """, (game_id,))
        recs = [dict(row) for row in c.fetchall()]

        if not recs:
            conn.close()
            return []

        results = []
        for rec in recs:
            player = rec["player"]
            prop_type = rec["prop_type"]
            side = rec["recommended_side"]
            pick_line = rec["line_at_recommendation"]
            pick_odds = rec["odds_at_recommendation"]

            # Find closing line (last snapshot, preferring "closing" type)
            c.execute("""
                SELECT line, over_odds, under_odds, snapshot_type
                FROM line_snapshots
                WHERE game_id = ? AND player = ? AND prop_type = ?
                ORDER BY
                    CASE WHEN snapshot_type = 'closing' THEN 0 ELSE 1 END,
                    timestamp DESC
                LIMIT 1
            """, (game_id, player, prop_type))
            closing_row = c.fetchone()

            if not closing_row:
                continue

            closing_line = closing_row["line"]
            closing_over = closing_row["over_odds"]
            closing_under = closing_row["under_odds"]

            # Determine closing odds for our side
            if side.lower() == "over":
                closing_odds = closing_over or -110
            else:
                closing_odds = closing_under or -110

            # Compute CLV metrics
            # 1. Line CLV: positive = we got a better number
            if side.lower() == "over":
                clv_line = closing_line - pick_line  # closing went UP = we got lower = good
            else:
                clv_line = pick_line - closing_line  # closing went DOWN = we got higher = good

            # 2. Implied probability CLV
            pick_implied = _american_to_implied(pick_odds)
            closing_implied = _american_to_implied(closing_odds)
            clv_implied = pick_implied - closing_implied  # positive = we got better price

            # 3. CLV in cents (industry standard: difference in implied probabilities)
            pick_decimal = _american_to_decimal(pick_odds)
            closing_decimal = _american_to_decimal(closing_odds)
            clv_cents = (1.0 / pick_decimal - 1.0 / closing_decimal) * 100

            # For OVER bets: negative clv_cents means we beat the close
            # For UNDER bets: same logic applies to the under odds
            # Actually, clv_cents convention: positive = we got better odds
            # If our implied was LOWER than closing implied, that means
            # the market moved toward our side (we were right) = positive CLV
            # So: clv_cents = (closing_implied - pick_implied) * 100
            # This way positive = the closing line agreed with us
            clv_cents = (closing_implied - pick_implied) * 100

            # Actual result
            won = None
            push = 0
            profit = None
            actual_result = None

            if actual_results:
                pkey = "{}|{}".format(player, prop_type)
                ar = actual_results.get(pkey)
                if ar:
                    actual_result = ar.get("actual")
                    if actual_result is not None and pick_line:
                        if actual_result == pick_line:
                            won = 0
                            push = 1
                            profit = 0.0
                        elif side.lower() == "over":
                            won = 1 if actual_result > pick_line else 0
                        else:
                            won = 1 if actual_result < pick_line else 0

                        if won == 1:
                            profit = _american_to_decimal(pick_odds) - 1
                        elif won == 0 and push == 0:
                            profit = -1.0

            # Check if CLV result already exists
            c.execute("""
                SELECT id FROM clv_results WHERE bet_id = ?
            """, (rec["id"],))
            existing = c.fetchone()

            now_iso = datetime.now().isoformat()

            if existing:
                # Update existing
                c.execute("""
                    UPDATE clv_results SET
                        closing_line = ?, closing_odds = ?,
                        clv_line = ?, clv_implied = ?, clv_cents = ?,
                        actual_result = ?, won = ?, push = ?, profit = ?,
                        computed_at = ?
                    WHERE bet_id = ?
                """, (
                    closing_line, closing_odds,
                    round(clv_line, 3), round(clv_implied, 5), round(clv_cents, 3),
                    actual_result, won, push, profit,
                    now_iso, rec["id"],
                ))
            else:
                c.execute("""
                    INSERT INTO clv_results
                    (bet_id, game_id, game_date, player, prop_type, recommended_side,
                     line_at_pick, odds_at_pick, closing_line, closing_odds,
                     clv_line, clv_implied, clv_cents,
                     actual_result, won, push, profit, computed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    rec["id"], game_id, rec["game_date"], player, prop_type, side,
                    pick_line, pick_odds, closing_line, closing_odds,
                    round(clv_line, 3), round(clv_implied, 5), round(clv_cents, 3),
                    actual_result, won, push, profit, now_iso,
                ))

            results.append({
                "player": player,
                "prop_type": prop_type,
                "side": side,
                "pick_line": pick_line,
                "pick_odds": pick_odds,
                "closing_line": closing_line,
                "closing_odds": closing_odds,
                "clv_line": round(clv_line, 3),
                "clv_implied": round(clv_implied, 5),
                "clv_cents": round(clv_cents, 3),
                "actual_result": actual_result,
                "won": won,
                "profit": profit,
            })

        conn.commit()
        conn.close()
        logger.info("Computed CLV for game %d: %d results", game_id, len(results))
        return results

    def compute_clv_for_date(self, date_str, actual_results_by_game=None):
        """
        Compute CLV for all recommendations on a given date.

        Args:
            date_str: YYYY-MM-DD
            actual_results_by_game: {game_id: {"player|prop_type": {"actual": X, "hit": bool}}}
        """
        conn = self._get_conn()
        c = conn.cursor()

        c.execute("""
            SELECT DISTINCT game_id FROM bet_recommendations
            WHERE game_date = ?
        """, (date_str,))
        game_ids = [row["game_id"] for row in c.fetchall()]
        conn.close()

        all_results = []
        for gid in game_ids:
            actual = (actual_results_by_game or {}).get(gid)
            results = self.compute_clv_for_game(gid, actual)
            all_results.extend(results)

        return all_results

    # ------------------------------------------------------------------
    # CLV Reporting
    # ------------------------------------------------------------------

    def get_clv_report(self, date_str=None, days=None):
        """
        Get CLV summary statistics.

        Args:
            date_str: specific date (YYYY-MM-DD), or None for range
            days: number of days to look back (7, 30, or None for all-time)

        Returns:
            dict with summary stats and per-bet details
        """
        conn = self._get_conn()
        c = conn.cursor()

        where_clause = "1=1"
        params = []

        if date_str:
            where_clause = "game_date = ?"
            params.append(date_str)
        elif days:
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            where_clause = "game_date >= ?"
            params.append(cutoff)

        c.execute("""
            SELECT * FROM clv_results
            WHERE {}
            ORDER BY game_date DESC, player
        """.format(where_clause), params)
        rows = [dict(row) for row in c.fetchall()]

        if not rows:
            conn.close()
            return {
                "total_bets": 0,
                "avg_clv_cents": 0,
                "avg_clv_line": 0,
                "pct_beat_close": 0,
                "by_prop_type": {},
                "by_edge_bucket": {},
                "daily_trend": [],
                "bets": [],
            }

        # Overall stats
        clv_cents_list = [r["clv_cents"] for r in rows if r["clv_cents"] is not None]
        clv_line_list = [r["clv_line"] for r in rows if r["clv_line"] is not None]
        beat_close = [r for r in rows if r["clv_cents"] is not None and r["clv_cents"] > 0]

        won_bets = [r for r in rows if r["won"] == 1]
        lost_bets = [r for r in rows if r["won"] == 0 and r["push"] == 0]
        total_graded = len(won_bets) + len(lost_bets)

        avg_clv_cents = sum(clv_cents_list) / len(clv_cents_list) if clv_cents_list else 0
        avg_clv_line = sum(clv_line_list) / len(clv_line_list) if clv_line_list else 0
        pct_beat = len(beat_close) / len(clv_cents_list) if clv_cents_list else 0
        win_rate = len(won_bets) / total_graded if total_graded else None

        # By prop type
        by_prop = {}
        for r in rows:
            pt = r["prop_type"]
            if pt not in by_prop:
                by_prop[pt] = {"clv_cents": [], "clv_line": [], "won": 0, "lost": 0, "count": 0}
            by_prop[pt]["count"] += 1
            if r["clv_cents"] is not None:
                by_prop[pt]["clv_cents"].append(r["clv_cents"])
            if r["clv_line"] is not None:
                by_prop[pt]["clv_line"].append(r["clv_line"])
            if r["won"] == 1:
                by_prop[pt]["won"] += 1
            elif r["won"] == 0 and r["push"] == 0:
                by_prop[pt]["lost"] += 1

        by_prop_summary = {}
        for pt, data in by_prop.items():
            by_prop_summary[pt] = {
                "count": data["count"],
                "avg_clv_cents": round(sum(data["clv_cents"]) / len(data["clv_cents"]), 3) if data["clv_cents"] else 0,
                "avg_clv_line": round(sum(data["clv_line"]) / len(data["clv_line"]), 3) if data["clv_line"] else 0,
                "pct_beat_close": round(len([c for c in data["clv_cents"] if c > 0]) / len(data["clv_cents"]), 3) if data["clv_cents"] else 0,
                "win_rate": round(data["won"] / (data["won"] + data["lost"]), 3) if (data["won"] + data["lost"]) else None,
            }

        # By edge bucket
        # Get recommendations to find edge at pick time
        rec_ids = [r["bet_id"] for r in rows]
        edge_map = {}
        if rec_ids:
            placeholders = ",".join(["?"] * len(rec_ids))
            c.execute("""
                SELECT id, edge FROM bet_recommendations
                WHERE id IN ({})
            """.format(placeholders), rec_ids)
            for row in c.fetchall():
                edge_map[row["id"]] = row["edge"]

        by_edge = {}
        for r in rows:
            edge = edge_map.get(r["bet_id"], 0) or 0
            edge_pct = abs(edge * 100) if abs(edge) < 1 else abs(edge)
            if edge_pct < 3:
                bucket = "0-3%"
            elif edge_pct < 5:
                bucket = "3-5%"
            elif edge_pct < 8:
                bucket = "5-8%"
            elif edge_pct < 12:
                bucket = "8-12%"
            else:
                bucket = "12%+"

            if bucket not in by_edge:
                by_edge[bucket] = {"clv_cents": [], "won": 0, "lost": 0, "count": 0}
            by_edge[bucket]["count"] += 1
            if r["clv_cents"] is not None:
                by_edge[bucket]["clv_cents"].append(r["clv_cents"])
            if r["won"] == 1:
                by_edge[bucket]["won"] += 1
            elif r["won"] == 0 and r["push"] == 0:
                by_edge[bucket]["lost"] += 1

        by_edge_summary = {}
        for bucket, data in by_edge.items():
            by_edge_summary[bucket] = {
                "count": data["count"],
                "avg_clv_cents": round(sum(data["clv_cents"]) / len(data["clv_cents"]), 3) if data["clv_cents"] else 0,
                "pct_beat_close": round(len([c for c in data["clv_cents"] if c > 0]) / len(data["clv_cents"]), 3) if data["clv_cents"] else 0,
                "win_rate": round(data["won"] / (data["won"] + data["lost"]), 3) if (data["won"] + data["lost"]) else None,
            }

        # Daily trend
        daily = {}
        for r in rows:
            d = r["game_date"]
            if d not in daily:
                daily[d] = {"clv_cents": [], "won": 0, "lost": 0, "count": 0}
            daily[d]["count"] += 1
            if r["clv_cents"] is not None:
                daily[d]["clv_cents"].append(r["clv_cents"])
            if r["won"] == 1:
                daily[d]["won"] += 1
            elif r["won"] == 0 and r["push"] == 0:
                daily[d]["lost"] += 1

        daily_trend = []
        for d in sorted(daily.keys()):
            data = daily[d]
            daily_trend.append({
                "date": d,
                "count": data["count"],
                "avg_clv_cents": round(sum(data["clv_cents"]) / len(data["clv_cents"]), 3) if data["clv_cents"] else 0,
                "pct_beat_close": round(len([c for c in data["clv_cents"] if c > 0]) / len(data["clv_cents"]), 3) if data["clv_cents"] else 0,
                "win_rate": round(data["won"] / (data["won"] + data["lost"]), 3) if (data["won"] + data["lost"]) else None,
            })

        # CLV vs win rate correlation
        clv_win_correlation = self._compute_clv_win_correlation(rows)

        conn.close()

        return {
            "total_bets": len(rows),
            "total_graded": total_graded,
            "avg_clv_cents": round(avg_clv_cents, 3),
            "avg_clv_line": round(avg_clv_line, 3),
            "pct_beat_close": round(pct_beat, 3),
            "win_rate": round(win_rate, 3) if win_rate is not None else None,
            "total_profit": round(sum(r["profit"] or 0 for r in rows), 2),
            "by_prop_type": by_prop_summary,
            "by_edge_bucket": by_edge_summary,
            "daily_trend": daily_trend,
            "clv_win_correlation": clv_win_correlation,
            "bets": rows[:200],  # Cap for API response size
        }

    def get_clv_summary(self):
        """
        Get CLV summary for multiple time windows (7d, 30d, all-time).

        Returns:
            dict with keys "last_7d", "last_30d", "all_time"
        """
        return {
            "last_7d": self.get_clv_report(days=7),
            "last_30d": self.get_clv_report(days=30),
            "all_time": self.get_clv_report(),
        }

    def _compute_clv_win_correlation(self, rows):
        """
        Compute correlation between CLV and win rate.

        Groups bets into CLV buckets and computes win rate per bucket.
        """
        # Only use graded bets
        graded = [r for r in rows if r["won"] is not None and r["clv_cents"] is not None]
        if len(graded) < 5:
            return None

        # Bucket by CLV
        buckets = {}
        for r in graded:
            clv = r["clv_cents"]
            if clv < -3:
                bk = "< -3"
            elif clv < -1:
                bk = "-3 to -1"
            elif clv < 0:
                bk = "-1 to 0"
            elif clv < 1:
                bk = "0 to 1"
            elif clv < 3:
                bk = "1 to 3"
            else:
                bk = "> 3"

            if bk not in buckets:
                buckets[bk] = {"won": 0, "total": 0}
            buckets[bk]["total"] += 1
            if r["won"] == 1:
                buckets[bk]["won"] += 1

        result = []
        for bk in ["< -3", "-3 to -1", "-1 to 0", "0 to 1", "1 to 3", "> 3"]:
            if bk in buckets and buckets[bk]["total"] > 0:
                result.append({
                    "clv_bucket": bk,
                    "count": buckets[bk]["total"],
                    "win_rate": round(buckets[bk]["won"] / buckets[bk]["total"], 3),
                })

        return result

    # ------------------------------------------------------------------
    # Stats helpers
    # ------------------------------------------------------------------

    def get_recent_recommendations(self, days=3):
        """Get recent recommendations that may need closing lines."""
        conn = self._get_conn()
        c = conn.cursor()

        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        c.execute("""
            SELECT DISTINCT game_id, game_date FROM bet_recommendations
            WHERE game_date >= ?
            ORDER BY game_date DESC
        """, (cutoff,))
        games = [dict(row) for row in c.fetchall()]
        conn.close()
        return games

    def get_games_needing_closing_lines(self):
        """
        Find games where we have recommendations but no closing line snapshots.

        Returns list of game_ids that need closing lines captured.
        """
        conn = self._get_conn()
        c = conn.cursor()

        c.execute("""
            SELECT DISTINCT r.game_id, r.game_date
            FROM bet_recommendations r
            LEFT JOIN (
                SELECT DISTINCT game_id
                FROM line_snapshots
                WHERE snapshot_type = 'closing'
            ) cls ON r.game_id = cls.game_id
            WHERE cls.game_id IS NULL
              AND r.game_date >= ?
            ORDER BY r.game_date DESC
        """, ((datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),))

        result = [dict(row) for row in c.fetchall()]
        conn.close()
        return result

    def get_games_needing_results(self):
        """
        Find games where we have CLV computed but no actual results.

        Returns list of game_ids that need results graded.
        """
        conn = self._get_conn()
        c = conn.cursor()

        c.execute("""
            SELECT DISTINCT game_id, game_date
            FROM clv_results
            WHERE won IS NULL
              AND game_date >= ?
            ORDER BY game_date DESC
        """, ((datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),))

        result = [dict(row) for row in c.fetchall()]
        conn.close()
        return result

    def get_tracking_stats(self):
        """
        Get overall tracking system statistics.

        Returns counts of snapshots, recommendations, and CLV results.
        """
        conn = self._get_conn()
        c = conn.cursor()

        stats = {}

        c.execute("SELECT COUNT(*) as cnt FROM line_snapshots")
        stats["total_snapshots"] = c.fetchone()["cnt"]

        c.execute("SELECT COUNT(*) as cnt FROM line_snapshots WHERE snapshot_type = 'closing'")
        stats["closing_snapshots"] = c.fetchone()["cnt"]

        c.execute("SELECT COUNT(*) as cnt FROM bet_recommendations")
        stats["total_recommendations"] = c.fetchone()["cnt"]

        c.execute("SELECT COUNT(*) as cnt FROM clv_results")
        stats["total_clv_results"] = c.fetchone()["cnt"]

        c.execute("SELECT COUNT(*) as cnt FROM clv_results WHERE won IS NOT NULL")
        stats["graded_results"] = c.fetchone()["cnt"]

        c.execute("SELECT MIN(game_date) as first_date, MAX(game_date) as last_date FROM bet_recommendations")
        row = c.fetchone()
        stats["first_date"] = row["first_date"]
        stats["last_date"] = row["last_date"]

        # Games tracked
        c.execute("SELECT COUNT(DISTINCT game_id) as cnt FROM bet_recommendations")
        stats["games_tracked"] = c.fetchone()["cnt"]

        conn.close()
        return stats


# Singleton instance
_clv_tracker = None


def get_clv_tracker():
    """Get singleton CLVTracker instance."""
    global _clv_tracker
    if _clv_tracker is None:
        _clv_tracker = CLVTracker()
    return _clv_tracker
