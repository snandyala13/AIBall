#!/usr/bin/env python3
"""
v2/minutes_ml_model.py - ML-trained Minutes Prediction Model

"Props are minutes bets, not stats bets."

This is a dedicated LightGBM model that predicts actual minutes played.
It replaces the heuristic-based minutes_model.py for training and evaluation,
and feeds into the MinutesFirstPredictor for the two-stage architecture:

  Stage 1: minutes_model.predict(features) -> projected_minutes
  Stage 2: projected_stat = per_minute_rate * projected_minutes
  Edge:    residual = projected_stat - DK_line

Key minutes features (beyond what the stat model already has):
- Recent minutes (L3, L5, L10), trend, variance, min/max
- Games since injury return (minute-restriction ramp-up detection)
- Blowout probability (spread-based)
- Back-to-back fatigue proxy
- Rotation stability (minutes_std / minutes_mean)
- Game total (higher totals -> slightly different minute distributions)
- Overtime probability proxy (close spread -> more OT chance)
- Foul trouble proxy (personal fouls per game)

Usage:
    python3 -I -m v2.minutes_ml_model             # Train + evaluate
    python3 -I -m v2.minutes_ml_model --from-cache # Use cached features
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: lightgbm not installed. Run: pip install lightgbm")
    sys.exit(1)

try:
    import joblib
except ImportError:
    print("ERROR: joblib not installed. Run: pip install joblib")
    sys.exit(1)

from scipy import stats as scipy_stats

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models" / "minutes_model"

GAME_DATA_FILES = [
    OUTPUTS / "backtest_2023_24_data.json",
    OUTPUTS / "backtest_2024_25_data.json",
    OUTPUTS / "backtest_oos_data.json",
]


# =============================================================================
# Feature builder for minutes prediction
# =============================================================================

class MinutesFeatureBuilder:
    """Build features specifically optimized for minutes prediction.

    This is separate from the stat feature builder because:
    1. The target is different (minutes, not stat - line)
    2. We need features that don't exist in the stat model
       (e.g., foul rate, minutes ramp-up detection, rotation stability)
    3. We want to be explicit about what matters for minutes
    """

    def __init__(self):
        self.games = []
        self.player_history = defaultdict(list)
        self.game_lookup = {}
        self._loaded = False

    def load_games(self):
        """Load all game data from JSON files."""
        all_games = []
        for path in GAME_DATA_FILES:
            if not path.exists():
                logger.warning("Missing: %s", path)
                continue
            logger.info("Loading %s ...", path.name)
            with open(path) as f:
                data = json.load(f)
            games = data.get("games", [])
            logger.info("  -> %d games", len(games))
            all_games.extend(games)

        all_games.sort(key=lambda g: g["date"])
        self.games = all_games
        self.game_lookup = {g["game_id"]: g for g in all_games}
        logger.info("Total games loaded: %d", len(self.games))

    def build_player_history(self):
        """Build per-player chronological game logs from box scores."""
        self.player_history.clear()

        # Also build team schedule for B2B detection
        self._team_schedule = defaultdict(list)  # {team: [(date, game_id)]}

        for game in self.games:
            date = game["date"]
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            game_id = game.get("game_id")
            home_score = game.get("home_score", 0)
            away_score = game.get("away_score", 0)
            box = game.get("box_score", {})

            # Track team schedules for B2B
            if home:
                self._team_schedule[home].append((date, game_id))
            if away:
                self._team_schedule[away].append((date, game_id))

            for player_name, stats in box.items():
                entry = {
                    "date": date,
                    "game_id": game_id,
                    "home_team": home,
                    "away_team": away,
                    "home_score": home_score,
                    "away_score": away_score,
                    "min": stats.get("min", 0),
                    "pts": stats.get("pts", 0),
                    "reb": stats.get("reb", 0),
                    "ast": stats.get("ast", 0),
                    "fg3m": stats.get("fg3m", 0),
                    "stl": stats.get("stl", 0),
                    "blk": stats.get("blk", 0),
                    "turnover": stats.get("turnover", 0),
                    "fga": stats.get("fga", 0),
                    "fgm": stats.get("fgm", 0),
                    "fta": stats.get("fta", 0),
                    "ftm": stats.get("ftm", 0),
                    "fg3a": stats.get("fg3a", 0),
                    "pf": stats.get("pf", 0),
                }
                self.player_history[player_name].append(entry)

        # Sort each player's history
        for player in self.player_history:
            self.player_history[player].sort(key=lambda g: g["date"])

        # Sort team schedules
        for team in self._team_schedule:
            self._team_schedule[team].sort()

        logger.info(
            "Built history for %d players (avg %.1f games/player)",
            len(self.player_history),
            np.mean([len(v) for v in self.player_history.values()]),
        )
        self._loaded = True

    def ensure_loaded(self):
        if not self._loaded:
            self.load_games()
            self.build_player_history()

    # -----------------------------------------------------------------
    # Core: build features for a single player-game
    # -----------------------------------------------------------------

    def get_minutes_features(
        self,
        player_name: str,
        game_date: str,
        game_context: dict,
    ) -> Optional[dict]:
        """Build features for predicting minutes in a specific game.

        Args:
            player_name: e.g. "LeBron James"
            game_date: ISO date string
            game_context: dict with spread, total, is_home, etc.

        Returns:
            dict of features, or None if insufficient history
        """
        history = self.player_history.get(player_name, [])
        prior_games = [g for g in history if g["date"] < game_date]

        if len(prior_games) < 3:
            return None

        # Most recent first
        prior = list(reversed(prior_games))
        feats = {}

        # ==================================================================
        # 1. MINUTES ROLLING AVERAGES
        # ==================================================================
        mins = [g["min"] for g in prior]
        mins_played = [m for m in mins if m > 0]  # Exclude DNPs

        l3 = mins[:3]
        l5 = mins[:5] if len(mins) >= 5 else mins
        l10 = mins[:10] if len(mins) >= 10 else mins
        l20 = mins[:20] if len(mins) >= 20 else mins
        season = mins

        feats["mins_l3"] = np.mean(l3)
        feats["mins_l5"] = np.mean(l5)
        feats["mins_l10"] = np.mean(l10)
        feats["mins_l20"] = np.mean(l20)
        feats["mins_season"] = np.mean(season)

        # Minutes when played (excluding DNPs)
        if len(mins_played) >= 3:
            played_l5 = mins_played[:5] if len(mins_played) >= 5 else mins_played
            played_l10 = mins_played[:10] if len(mins_played) >= 10 else mins_played
            feats["mins_when_played_l5"] = np.mean(played_l5)
            feats["mins_when_played_l10"] = np.mean(played_l10)
        else:
            feats["mins_when_played_l5"] = feats["mins_l5"]
            feats["mins_when_played_l10"] = feats["mins_l10"]

        # ==================================================================
        # 2. MINUTES VARIANCE / STABILITY
        # ==================================================================
        feats["mins_std_l10"] = np.std(l10) if len(l10) >= 3 else np.std(season)
        feats["mins_std_season"] = np.std(season) if len(season) >= 5 else feats["mins_std_l10"]
        feats["mins_min_l5"] = min(l5)
        feats["mins_max_l5"] = max(l5)
        feats["mins_range_l5"] = max(l5) - min(l5)
        feats["mins_median_l10"] = float(np.median(l10))

        # Rotation stability: low CV = stable role, high CV = volatile
        mean_l10 = max(feats["mins_l10"], 0.1)
        feats["mins_cv_l10"] = feats["mins_std_l10"] / mean_l10
        feats["mins_cv_season"] = feats["mins_std_season"] / max(feats["mins_season"], 0.1)

        # ==================================================================
        # 3. MINUTES TREND (role change detection)
        # ==================================================================
        feats["mins_trend_l3_l10"] = feats["mins_l3"] - feats["mins_l10"]
        feats["mins_trend_l5_l20"] = feats["mins_l5"] - feats["mins_l20"]
        feats["mins_trend_l3_season"] = feats["mins_l3"] - feats["mins_season"]

        # Exponentially weighted minutes (recent games matter more)
        if len(mins) >= 5:
            weights = np.array([0.95 ** i for i in range(min(10, len(mins)))])
            ewm_mins = np.average(mins[:len(weights)], weights=weights)
            feats["mins_ewm_l10"] = ewm_mins
        else:
            feats["mins_ewm_l10"] = feats["mins_l5"]

        # ==================================================================
        # 4. DNP / RETURN FROM INJURY DETECTION
        # ==================================================================
        # Count DNPs in recent games
        dnp_l5 = sum(1 for m in l5 if m == 0)
        dnp_l10 = sum(1 for m in l10 if m == 0)
        feats["dnp_count_l5"] = dnp_l5
        feats["dnp_count_l10"] = dnp_l10

        # Games since last DNP (proxy for "games since return from injury")
        games_since_dnp = 0
        for m in mins:
            if m == 0:
                break
            games_since_dnp += 1
        feats["games_since_dnp"] = min(games_since_dnp, 50)

        # Minute restriction ramp-up detection:
        # If player recently returned (games_since_dnp < 5), check if minutes
        # are ramping up game over game
        if games_since_dnp > 0 and games_since_dnp <= 5:
            recent_since_return = mins[:games_since_dnp]
            if len(recent_since_return) >= 2:
                # Is each game more minutes than the last? (ramp-up)
                ramp_up = all(
                    recent_since_return[i] >= recent_since_return[i + 1] - 1
                    for i in range(len(recent_since_return) - 1)
                )
                feats["minute_ramp_up"] = 1 if ramp_up else 0
                feats["ramp_up_slope"] = (
                    (recent_since_return[0] - recent_since_return[-1])
                    / max(len(recent_since_return) - 1, 1)
                )
            else:
                feats["minute_ramp_up"] = 0
                feats["ramp_up_slope"] = 0.0
        else:
            feats["minute_ramp_up"] = 0
            feats["ramp_up_slope"] = 0.0

        # ==================================================================
        # 5. FOUL TROUBLE PROXY
        # ==================================================================
        pf_list = [g.get("pf", 0) for g in prior[:10]]
        feats["pf_per_game_l10"] = np.mean(pf_list) if pf_list else 0.0
        if len(mins_played) >= 3:
            pf_per_min = [
                g.get("pf", 0) / max(g["min"], 1)
                for g in prior[:10]
                if g["min"] > 0
            ]
            feats["pf_per_minute_l10"] = np.mean(pf_per_min) if pf_per_min else 0.0
        else:
            feats["pf_per_minute_l10"] = 0.0

        # High foul rate: played fewer minutes due to fouls recently?
        # Games with 5+ fouls in L10
        high_foul_games = sum(1 for g in prior[:10] if g.get("pf", 0) >= 5)
        feats["high_foul_games_l10"] = high_foul_games

        # ==================================================================
        # 6. GAME CONTEXT FEATURES
        # ==================================================================
        spread = game_context.get("spread", 0.0)
        total = game_context.get("total", 224.0)
        is_home = game_context.get("is_home", None)

        feats["spread"] = spread
        feats["abs_spread"] = abs(spread)
        feats["is_favorite"] = 1 if spread < 0 else 0
        feats["vegas_total"] = total
        feats["is_home"] = 1 if is_home else (0 if is_home is False else 0.5)

        # Blowout risk (spread magnitude bucketed)
        feats["blowout_risk"] = 1 if abs(spread) > 10 else 0
        feats["big_favorite"] = 1 if spread < -10 else 0
        feats["big_underdog"] = 1 if spread > 10 else 0

        # Close game = overtime probability
        feats["close_game"] = 1 if abs(spread) <= 3 else 0
        feats["ot_probability_proxy"] = max(0, 0.06 - 0.01 * abs(spread))

        # ==================================================================
        # 7. BACK-TO-BACK / SCHEDULE FEATURES
        # ==================================================================
        is_b2b = game_context.get("is_b2b", 0)
        days_rest = game_context.get("days_rest", 1)
        feats["is_b2b"] = is_b2b
        feats["days_rest"] = min(days_rest, 5)

        # ==================================================================
        # 8. ROLE INDICATORS
        # ==================================================================
        # Classify player role from recent minutes
        ref_mins = feats["mins_l5"]
        feats["is_starter"] = 1 if ref_mins >= 28 else 0
        feats["is_rotation"] = 1 if 18 <= ref_mins < 28 else 0
        feats["is_bench"] = 1 if ref_mins < 18 else 0

        # Season games played (experience / load context)
        feats["games_played"] = min(len(prior), 82)

        # High-minutes game percentage in L10
        feats["high_mins_pct_l10"] = np.mean([1 if m >= 30 else 0 for m in l10])

        # ==================================================================
        # 9. SCORING CONTEXT (minutes correlate with usage)
        # ==================================================================
        pts_l5 = [g["pts"] for g in prior[:5]]
        feats["pts_per_game_l5"] = np.mean(pts_l5) if pts_l5 else 0.0

        # Usage proxy: FGA + FTA per minute
        if len(mins_played) >= 3:
            usage_vals = [
                (g["fga"] + 0.44 * g["fta"]) / max(g["min"], 1)
                for g in prior[:10]
                if g["min"] > 0
            ]
            feats["usage_proxy_l10"] = np.mean(usage_vals) if usage_vals else 0.0
        else:
            feats["usage_proxy_l10"] = 0.0

        # ==================================================================
        # 10. SCORE DIFFERENTIAL / BLOWOUT HISTORY
        # ==================================================================
        # How often does this player's team get blown out or blow out?
        differentials = []
        for g in prior[:20]:
            home_s = g.get("home_score", 0)
            away_s = g.get("away_score", 0)
            diff = abs(home_s - away_s)
            differentials.append(diff)
        if differentials:
            feats["avg_game_margin_l20"] = np.mean(differentials)
            feats["blowout_pct_l20"] = np.mean([1 if d >= 20 else 0 for d in differentials])
        else:
            feats["avg_game_margin_l20"] = 10.0
            feats["blowout_pct_l20"] = 0.1

        # ==================================================================
        # 11. INTERACTION FEATURES
        # ==================================================================
        # Starter on B2B = biggest minutes impact
        feats["starter_b2b"] = feats["is_starter"] * feats["is_b2b"]
        # Favorite starter = blowout risk
        feats["starter_big_fav"] = feats["is_starter"] * feats["big_favorite"]
        # Underdog starter = potential extra minutes
        feats["starter_big_dog"] = feats["is_starter"] * feats["big_underdog"]
        # Recent return from injury
        feats["recent_return"] = 1 if 0 < games_since_dnp <= 5 else 0
        # Ramp-up on recent return
        feats["ramp_up_recent"] = feats["minute_ramp_up"] * feats["recent_return"]

        return feats

    # -----------------------------------------------------------------
    # Build training dataset
    # -----------------------------------------------------------------

    def build_training_data(self) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        """Build feature matrix with target = actual minutes played.

        Each row is a player-game. Uses ALL players in ALL games (not just
        those with prop lines), giving us ~280K training samples.

        Returns:
            (X: DataFrame of features,
             y: ndarray of actual minutes,
             meta: DataFrame with date, player, game_id, team info)
        """
        self.ensure_loaded()

        rows_X = []
        rows_y = []
        rows_meta = []
        skipped = 0

        for game_idx, game in enumerate(self.games):
            if game_idx > 0 and game_idx % 500 == 0:
                logger.info(
                    "  Processing game %d/%d (%.0f%%) -- %d samples built",
                    game_idx, len(self.games), 100 * game_idx / len(self.games),
                    len(rows_X),
                )

            date = game["date"]
            game_id = game.get("game_id")
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            home_score = game.get("home_score", 0)
            away_score = game.get("away_score", 0)
            box = game.get("box_score", {})

            # Determine B2B for each team
            home_b2b = self._is_b2b(home, date)
            away_b2b = self._is_b2b(away, date)
            home_rest = self._days_rest(home, date)
            away_rest = self._days_rest(away, date)

            for player_name, stats in box.items():
                actual_mins = stats.get("min", 0)

                # Build game context
                # We don't have spread/total in historical data, so use proxies
                # Score differential as a proxy for what the spread would have been
                score_diff = home_score - away_score

                # Determine which team this player is on
                # Heuristic: check which team's history has this player more
                is_home = self._guess_player_team(player_name, game)

                if is_home is None:
                    # Can't determine team, use neutral context
                    is_b2b = 0
                    days_rest = 1
                    spread = 0.0
                elif is_home:
                    is_b2b = home_b2b
                    days_rest = home_rest
                    spread = 0.0  # No spread data in historical
                else:
                    is_b2b = away_b2b
                    days_rest = away_rest
                    spread = 0.0

                game_context = {
                    "spread": spread,
                    "total": 224.0,  # Default, no total in data
                    "is_home": is_home,
                    "is_b2b": is_b2b,
                    "days_rest": days_rest,
                }

                feats = self.get_minutes_features(player_name, date, game_context)
                if feats is None:
                    skipped += 1
                    continue

                rows_X.append(feats)
                rows_y.append(actual_mins)
                rows_meta.append({
                    "date": date,
                    "player": player_name,
                    "game_id": game_id,
                    "home_team": home,
                    "away_team": away,
                    "is_home": is_home,
                    "actual_mins": actual_mins,
                })

        logger.info(
            "Minutes training data: %d samples, %d skipped (no history)",
            len(rows_X), skipped,
        )

        X = pd.DataFrame(rows_X).fillna(0)
        y = np.array(rows_y, dtype=np.float64)
        meta = pd.DataFrame(rows_meta)

        return X, y, meta

    # -----------------------------------------------------------------
    # Build training data from legs (for comparison with stat model)
    # -----------------------------------------------------------------

    def build_legs_training_data(self) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        """Build training data from legs files (only players with prop lines).

        This is a subset of build_training_data() that matches the stat model's
        training set, allowing apples-to-apples comparison.

        Returns:
            (X: features, y: actual minutes, meta: with date/player/game_id/actual_mins
             + prop_type/line/actual_stat for stat comparison)
        """
        self.ensure_loaded()

        LEGS_FILES = [
            OUTPUTS / "backtest_2023_24_legs.json",
            OUTPUTS / "backtest_2024_25_legs.json",
            OUTPUTS / "backtest_oos_legs.json",
        ]

        all_legs = []
        for path in LEGS_FILES:
            if not path.exists():
                continue
            with open(path) as f:
                legs = json.load(f)
            all_legs.extend(legs)
        all_legs.sort(key=lambda x: x["date"])

        # De-duplicate: for the same player-game, we only need one minutes
        # prediction (even if they have multiple props). Keep track.
        seen = set()

        rows_X = []
        rows_y = []
        rows_meta = []
        skipped = 0

        for leg in all_legs:
            player = leg["player"]
            date = leg["date"]
            game_id = leg.get("game_id")
            key = (player, game_id)

            # For minutes training, we just need one entry per player-game
            # But for evaluation we want all legs
            actual = leg.get("actual")
            if actual is None:
                skipped += 1
                continue

            # Look up actual minutes from the game's box score
            game = self.game_lookup.get(game_id)
            if not game:
                skipped += 1
                continue

            box = game.get("box_score", {})
            player_stats = box.get(player)
            if not player_stats:
                skipped += 1
                continue

            actual_mins = player_stats.get("min", 0)

            # Determine team
            matchup = leg.get("matchup", "")
            is_home = None
            home_team = game.get("home_team", "")
            away_team = game.get("away_team", "")

            # B2B detection
            is_b2b = 0
            days_rest = 1
            player_team = self._guess_player_team(player, game)
            if player_team is not None:
                is_home = player_team
                team_name = home_team if is_home else away_team
                is_b2b = self._is_b2b(team_name, date)
                days_rest = self._days_rest(team_name, date)

            game_context = {
                "spread": 0.0,
                "total": 224.0,
                "is_home": is_home,
                "is_b2b": is_b2b,
                "days_rest": days_rest,
            }

            # Only build features once per player-game for minutes model
            if key not in seen:
                feats = self.get_minutes_features(player, date, game_context)
                if feats is None:
                    skipped += 1
                    continue
                seen.add(key)
            else:
                # Skip duplicate player-game for minutes (but keep for meta)
                continue

            rows_X.append(feats)
            rows_y.append(actual_mins)
            rows_meta.append({
                "date": date,
                "player": player,
                "game_id": game_id,
                "actual_mins": actual_mins,
                "prop_type": leg["prop_type"],
                "line": leg.get("line", 0),
                "actual_stat": float(actual),
                "over_odds": leg.get("over_odds", -110),
                "under_odds": leg.get("under_odds", -110),
            })

        logger.info(
            "Legs-based minutes data: %d unique player-games, %d skipped",
            len(rows_X), skipped,
        )

        X = pd.DataFrame(rows_X).fillna(0)
        y = np.array(rows_y, dtype=np.float64)
        meta = pd.DataFrame(rows_meta)

        return X, y, meta

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _is_b2b(self, team: str, date: str) -> int:
        """Check if this game is a back-to-back for the team."""
        schedule = self._team_schedule.get(team, [])
        for i, (d, _) in enumerate(schedule):
            if d == date and i > 0:
                prev_date = schedule[i - 1][0]
                # Check if previous game was yesterday
                from datetime import datetime, timedelta
                try:
                    curr = datetime.strptime(date, "%Y-%m-%d")
                    prev = datetime.strptime(prev_date, "%Y-%m-%d")
                    if (curr - prev).days == 1:
                        return 1
                except ValueError:
                    pass
                return 0
        return 0

    def _days_rest(self, team: str, date: str) -> int:
        """Compute days since last game for a team."""
        schedule = self._team_schedule.get(team, [])
        for i, (d, _) in enumerate(schedule):
            if d == date and i > 0:
                prev_date = schedule[i - 1][0]
                from datetime import datetime
                try:
                    curr = datetime.strptime(date, "%Y-%m-%d")
                    prev = datetime.strptime(prev_date, "%Y-%m-%d")
                    return (curr - prev).days
                except ValueError:
                    pass
                return 1
        return 1

    def _guess_player_team(self, player_name: str, game: dict) -> Optional[bool]:
        """Guess if player is on home team based on game context.

        Returns True if home, False if away, None if can't determine.
        Uses a simple heuristic: look at how many games this player has
        appeared in for each team.
        """
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        history = self.player_history.get(player_name, [])

        if not history:
            return None

        # Count games where player appeared on each team's side
        home_count = 0
        away_count = 0
        for g in history[-20:]:  # Use recent history
            if g.get("home_team") == home:
                home_count += 1
            if g.get("away_team") == home:
                away_count += 1

        if home_count > away_count:
            return True  # More likely home team
        elif away_count > home_count:
            return False  # More likely away team
        return None


# =============================================================================
# Blowout Risk Sub-Model
# =============================================================================

class BlowoutRiskModel:
    """Predicts probability of different game margin buckets.

    Input: spread, team ratings, game total
    Output: P(close), P(moderate), P(blowout)

    Blowout -> minute reduction for starters, increase for bench.
    """

    # Historical margin distributions by spread bucket
    # (derived from ~10 years of NBA data)
    MARGIN_PROBS = {
        # abs_spread: (p_close_0_9, p_moderate_10_19, p_blowout_20plus)
        0: (0.55, 0.30, 0.15),
        3: (0.50, 0.30, 0.20),
        5: (0.42, 0.33, 0.25),
        7: (0.35, 0.33, 0.32),
        10: (0.25, 0.35, 0.40),
        12: (0.20, 0.32, 0.48),
        15: (0.12, 0.30, 0.58),
    }

    @classmethod
    def predict(cls, spread: float) -> dict:
        """Return margin probabilities for this spread."""
        abs_s = abs(spread)

        # Interpolate between known spread buckets
        spreads = sorted(cls.MARGIN_PROBS.keys())
        if abs_s <= spreads[0]:
            probs = cls.MARGIN_PROBS[spreads[0]]
        elif abs_s >= spreads[-1]:
            probs = cls.MARGIN_PROBS[spreads[-1]]
        else:
            # Linear interpolation
            for i in range(len(spreads) - 1):
                if spreads[i] <= abs_s < spreads[i + 1]:
                    frac = (abs_s - spreads[i]) / (spreads[i + 1] - spreads[i])
                    p1 = cls.MARGIN_PROBS[spreads[i]]
                    p2 = cls.MARGIN_PROBS[spreads[i + 1]]
                    probs = tuple(
                        p1[j] + frac * (p2[j] - p1[j]) for j in range(3)
                    )
                    break
            else:
                probs = cls.MARGIN_PROBS[spreads[-1]]

        return {
            "p_close": probs[0],
            "p_moderate": probs[1],
            "p_blowout": probs[2],
        }


# =============================================================================
# Minutes Model Trainer
# =============================================================================

class MinutesModelTrainer:
    """Train and evaluate the minutes prediction model."""

    def __init__(self, X: pd.DataFrame, y: np.ndarray, meta: pd.DataFrame):
        self.X = X
        self.y = y
        self.meta = meta

    @classmethod
    def from_scratch(cls):
        """Build features from raw game data."""
        fb = MinutesFeatureBuilder()
        fb.load_games()
        fb.build_player_history()
        X, y, meta = fb.build_training_data()
        return cls(X, y, meta)

    @classmethod
    def from_legs(cls):
        """Build from legs data (for apples-to-apples stat model comparison)."""
        fb = MinutesFeatureBuilder()
        fb.load_games()
        fb.build_player_history()
        X, y, meta = fb.build_legs_training_data()
        return cls(X, y, meta)

    def train(self) -> dict:
        """Train the minutes model with walk-forward time split.

        Returns dict with model, metrics, feature importance.
        """
        # Sort by date
        dates = self.meta["date"]
        sort_idx = dates.argsort()
        X = self.X.iloc[sort_idx].reset_index(drop=True)
        y = self.y[sort_idx]
        meta = self.meta.iloc[sort_idx].reset_index(drop=True)
        dates = meta["date"]

        n = len(X)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        X_train, y_train = X.iloc[:train_end], y[:train_end]
        X_val, y_val = X.iloc[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X.iloc[val_end:], y[val_end:]

        dates_train = dates.iloc[:train_end]
        dates_test = dates.iloc[val_end:]
        meta_test = meta.iloc[val_end:]

        logger.info(
            "Split: Train=%d (to %s), Val=%d (to %s), Test=%d (to %s)",
            len(X_train), dates_train.iloc[-1],
            len(X_val), dates.iloc[val_end - 1],
            len(X_test), dates_test.iloc[-1],
        )

        # Sample weights: favor recent data
        max_date = dates_train.max()
        days_ago = (pd.to_datetime(max_date) - pd.to_datetime(dates_train)).dt.days
        weights = np.exp(-0.693 * days_ago.values / 90.0)
        weights = weights / weights.mean()

        # --- Main model: Huber loss (robust to outliers like DNPs) ---
        model = lgb.LGBMRegressor(
            objective="huber",
            n_estimators=1000,
            max_depth=7,
            learning_rate=0.05,
            num_leaves=50,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
        )
        model.fit(
            X_train, y_train,
            sample_weight=weights,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )

        # --- Quantile models for uncertainty ---
        q10_model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=0.10,
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=40,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
        )
        q10_model.fit(
            X_train, y_train,
            sample_weight=weights,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )

        q90_model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=0.90,
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=40,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
        )
        q90_model.fit(
            X_train, y_train,
            sample_weight=weights,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )

        # ==================================================================
        # EVALUATE on test set
        # ==================================================================
        preds = model.predict(X_test)
        q10_preds = q10_model.predict(X_test)
        q90_preds = q90_model.predict(X_test)

        # Clip predictions to valid range
        preds = np.clip(preds, 0, 53)
        q10_preds = np.clip(q10_preds, 0, 53)
        q90_preds = np.clip(q90_preds, 0, 53)

        # -- Minutes prediction accuracy --
        residuals = preds - y_test
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals ** 2))
        mape = np.mean(np.abs(residuals) / np.maximum(y_test, 1)) * 100

        # Baseline: predict L5 average (what you'd do without a model)
        baseline_preds = X_test["mins_l5"].values
        baseline_mae = np.mean(np.abs(baseline_preds - y_test))
        baseline_rmse = np.sqrt(np.mean((baseline_preds - y_test) ** 2))

        # Baseline 2: L3 average
        l3_preds = X_test["mins_l3"].values
        l3_mae = np.mean(np.abs(l3_preds - y_test))

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-9)

        # Quantile calibration
        in_range = np.mean((y_test >= q10_preds) & (y_test <= q90_preds))

        # -- Conditional accuracy (the important stuff) --
        # B2B games
        b2b_mask = X_test["is_b2b"].values == 1
        if b2b_mask.sum() > 0:
            b2b_model_mae = np.mean(np.abs(preds[b2b_mask] - y_test[b2b_mask]))
            b2b_baseline_mae = np.mean(np.abs(baseline_preds[b2b_mask] - y_test[b2b_mask]))
        else:
            b2b_model_mae = b2b_baseline_mae = 0.0

        # Recent return from injury
        return_mask = X_test["recent_return"].values == 1
        if return_mask.sum() > 0:
            return_model_mae = np.mean(np.abs(preds[return_mask] - y_test[return_mask]))
            return_baseline_mae = np.mean(np.abs(baseline_preds[return_mask] - y_test[return_mask]))
        else:
            return_model_mae = return_baseline_mae = 0.0

        # Starters
        starter_mask = X_test["is_starter"].values == 1
        if starter_mask.sum() > 0:
            starter_model_mae = np.mean(np.abs(preds[starter_mask] - y_test[starter_mask]))
            starter_baseline_mae = np.mean(np.abs(baseline_preds[starter_mask] - y_test[starter_mask]))
        else:
            starter_model_mae = starter_baseline_mae = 0.0

        # High variance (DNPs in recent games)
        dnp_mask = X_test["dnp_count_l10"].values > 0
        if dnp_mask.sum() > 0:
            dnp_model_mae = np.mean(np.abs(preds[dnp_mask] - y_test[dnp_mask]))
            dnp_baseline_mae = np.mean(np.abs(baseline_preds[dnp_mask] - y_test[dnp_mask]))
        else:
            dnp_model_mae = dnp_baseline_mae = 0.0

        # Minutes deviation analysis: cases where actual differs from L5 by 5+ min
        deviation_mask = np.abs(y_test - baseline_preds) >= 5
        if deviation_mask.sum() > 0:
            dev_model_mae = np.mean(np.abs(preds[deviation_mask] - y_test[deviation_mask]))
            dev_baseline_mae = np.mean(np.abs(baseline_preds[deviation_mask] - y_test[deviation_mask]))
        else:
            dev_model_mae = dev_baseline_mae = 0.0

        # Feature importance
        importances = model.feature_importances_
        feat_names = X_train.columns.tolist()
        imp_pairs = sorted(zip(feat_names, importances), key=lambda x: -x[1])

        metrics = {
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "model_mae": float(mae),
            "model_rmse": float(rmse),
            "model_mape": float(mape),
            "baseline_l5_mae": float(baseline_mae),
            "baseline_l5_rmse": float(baseline_rmse),
            "baseline_l3_mae": float(l3_mae),
            "beats_l5_baseline": bool(mae < baseline_mae),
            "mae_improvement": float(baseline_mae - mae),
            "r_squared": float(r2),
            "quantile_calib_80": float(in_range),
            "best_iteration": model.best_iteration_,
            "conditional": {
                "b2b": {
                    "n": int(b2b_mask.sum()),
                    "model_mae": float(b2b_model_mae),
                    "baseline_mae": float(b2b_baseline_mae),
                },
                "recent_return": {
                    "n": int(return_mask.sum()),
                    "model_mae": float(return_model_mae),
                    "baseline_mae": float(return_baseline_mae),
                },
                "starters": {
                    "n": int(starter_mask.sum()),
                    "model_mae": float(starter_model_mae),
                    "baseline_mae": float(starter_baseline_mae),
                },
                "recent_dnp": {
                    "n": int(dnp_mask.sum()),
                    "model_mae": float(dnp_model_mae),
                    "baseline_mae": float(dnp_baseline_mae),
                },
                "minutes_deviation_5plus": {
                    "n": int(deviation_mask.sum()),
                    "model_mae": float(dev_model_mae),
                    "baseline_mae": float(dev_baseline_mae),
                },
            },
            "top_features": [
                {"name": name, "importance": int(imp)}
                for name, imp in imp_pairs[:20]
            ],
        }

        # Print results
        self._print_results(metrics)

        # Save models
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODELS_DIR / "minutes_lgb_huber.joblib")
        joblib.dump(q10_model, MODELS_DIR / "minutes_q10.joblib")
        joblib.dump(q90_model, MODELS_DIR / "minutes_q90.joblib")

        # Save metadata
        model_meta = {
            "version": "v1_minutes_first",
            "description": "Dedicated LightGBM minutes prediction model",
            "feature_names": feat_names,
            "n_features": len(feat_names),
            "sigma": float(mae),  # Use MAE as sigma for downstream P(over) calc
            "metrics": metrics,
        }
        with open(MODELS_DIR / "model_metadata.json", "w") as f:
            json.dump(model_meta, f, indent=2, default=str)

        logger.info("Models saved to %s", MODELS_DIR)

        return {
            "model": model,
            "q10_model": q10_model,
            "q90_model": q90_model,
            "metrics": metrics,
            "X_test": X_test,
            "y_test": y_test,
            "meta_test": meta_test,
            "preds": preds,
        }

    def _print_results(self, metrics: dict):
        """Pretty-print training results."""
        print(f"\n{'='*70}")
        print(f"  MINUTES MODEL - Test Set Results")
        print(f"{'='*70}")
        print(f"  Train: {metrics['n_train']:,}  Val: {metrics['n_val']:,}  Test: {metrics['n_test']:,}")
        print(f"\n  --- Overall Accuracy ---")
        print(f"  Model MAE:       {metrics['model_mae']:.2f} minutes")
        print(f"  Baseline L5 MAE: {metrics['baseline_l5_mae']:.2f} minutes")
        print(f"  Baseline L3 MAE: {metrics['baseline_l3_mae']:.2f} minutes")
        beats = "YES" if metrics['beats_l5_baseline'] else "NO"
        print(f"  Beats L5 baseline: {beats} (improvement: {metrics['mae_improvement']:+.3f} min)")
        print(f"  Model RMSE:      {metrics['model_rmse']:.2f} minutes")
        print(f"  R-squared:       {metrics['r_squared']:.4f}")
        print(f"  P10-P90 calib:   {metrics['quantile_calib_80']:.1%} (target: 80%)")
        print(f"  Early stopping:  iteration {metrics['best_iteration']}")

        print(f"\n  --- Conditional Accuracy (where minutes model MATTERS) ---")
        for scenario, data in metrics["conditional"].items():
            if data["n"] > 0:
                delta = data["baseline_mae"] - data["model_mae"]
                beats = "+" if delta > 0 else "-"
                print(f"  {scenario:25s}: N={data['n']:5d}  "
                      f"Model={data['model_mae']:.2f}  "
                      f"Baseline={data['baseline_mae']:.2f}  "
                      f"Delta={delta:+.3f} [{beats}]")

        print(f"\n  --- Top 15 Features ---")
        for i, feat in enumerate(metrics["top_features"][:15], 1):
            print(f"    {i:2d}. {feat['name']:30s} {feat['importance']:6d}")

        print(f"{'='*70}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train minutes prediction model")
    parser.add_argument("--from-cache", action="store_true",
                        help="Use cached features (not supported yet, builds from scratch)")
    parser.add_argument("--legs-only", action="store_true",
                        help="Train only on player-games with prop lines")
    args = parser.parse_args()

    print("=" * 70)
    print("  Minutes-First Prediction Model - Stage 1: Minutes Predictor")
    print("  Target: actual minutes played")
    print("  Architecture: dedicated LightGBM for minutes prediction")
    print("=" * 70)

    t0 = time.time()

    if args.legs_only:
        trainer = MinutesModelTrainer.from_legs()
    else:
        trainer = MinutesModelTrainer.from_scratch()

    result = trainer.train()

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
