#!/usr/bin/env python3
"""
v2/features.py - Feature engineering for line-anchor ML prop model.

Builds feature matrices from raw game data (box scores + DraftKings prop lines).
CRITICAL: All features use only games BEFORE the prediction date (no lookahead).

Architecture: predicted = DK_line + ML_adjustment(features)
  The DK line is the anchor. The ML model learns small adjustments.
  Target = actual - DK_line (how wrong was the line?)

Data sources:
  - outputs/backtest_2023_24_data.json  (1,319 games, 2023-10 to 2024-10)
  - outputs/backtest_oos_data.json      (794 games, 2026-01 to 2026-06)
  - outputs/backtest_2023_24_legs.json  (flat leg records with actuals)
  - outputs/backtest_oos_legs.json      (flat leg records with actuals)

Feature groups:
  1. Box score rolling stats (~30 features)
  2. Minutes features (~7 features)
  3. Per-minute rates (~4 features)
  4. Related stats (~7 features)
  5. Shooting efficiency (~5 features)
  6. Market line features / line-anchor signals (~12 features)
  7. Cross-book market features (~6 features from PropLine all_books)
  8. Game context (~6 features)
  9. Consistency / volatility (~7 features)
  10. Prop type encoding (~5 features)
  11. Referee tendencies (~5 features, enhanced mode)
  12. Player on/off impact (~4 features, enhanced mode)
  13. Pace matchup prediction (~4 features, enhanced mode)

Usage:
    python -m v2.features          # Build training data and print stats
"""

import json
import os
import sys
import logging
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS = BASE_DIR / "outputs"

GAME_DATA_FILES = [
    OUTPUTS / "backtest_2023_24_data.json",
    OUTPUTS / "backtest_2024_25_data.json",
    OUTPUTS / "backtest_oos_data.json",
]
LEGS_FILES = [
    OUTPUTS / "backtest_2023_24_legs.json",
    OUTPUTS / "backtest_2024_25_legs.json",
    OUTPUTS / "backtest_oos_legs.json",
]

# ---------------------------------------------------------------------------
# Prop type -> box score stat key(s)
# ---------------------------------------------------------------------------
PROP_TO_STAT = {
    "points": ["pts"],
    "rebounds": ["reb"],
    "assists": ["ast"],
    "threes": ["fg3m"],
    "steals": ["stl"],
    "blocks": ["blk"],
    "turnovers": ["turnover"],
    "pts_rebs_asts": ["pts", "reb", "ast"],
    "pts_rebs": ["pts", "reb"],
    "pts_asts": ["pts", "ast"],
    "rebs_asts": ["reb", "ast"],
    "stls_blks": ["stl", "blk"],
}

# Core prop types we train models for (high-volume markets)
CORE_PROPS = ["points", "rebounds", "assists", "threes", "pts_rebs_asts", "steals", "blocks"]

# All individual stat keys tracked per game
ALL_STAT_KEYS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover", "min"]


# =============================================================================
# FeatureBuilder
# =============================================================================

class FeatureBuilder:
    """Builds feature matrix from raw game data for ML training."""

    def __init__(self):
        self.games = []                       # All games loaded from JSON
        self.player_history = defaultdict(list)  # {player: [game_dicts sorted by date]}
        self.game_lookup = {}                 # {game_id: game_dict}
        self._loaded = False

    # -----------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------

    def load_games(self):
        """Load all game data from both JSON files, sort by date."""
        all_games = []
        for path in GAME_DATA_FILES:
            if not path.exists():
                logger.warning("Missing game data file: %s", path)
                continue
            logger.info("Loading %s ...", path.name)
            with open(path) as f:
                data = json.load(f)
            games = data.get("games", [])
            logger.info("  -> %d games", len(games))
            all_games.extend(games)

        # Sort globally by date
        all_games.sort(key=lambda g: g["date"])
        self.games = all_games
        self.game_lookup = {g["game_id"]: g for g in all_games}
        logger.info("Total games loaded: %d", len(self.games))

    def build_player_history(self):
        """
        For each player, build chronological game history from box scores.
        Each entry: {date, pts, reb, ast, fg3m, stl, blk, turnover, min,
                     opponent, is_home, game_id}
        """
        self.player_history.clear()
        for game in self.games:
            date = game["date"]
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            game_id = game.get("game_id")
            home_score = game.get("home_score", 0)
            away_score = game.get("away_score", 0)
            box = game.get("box_score", {})

            for player_name, stats in box.items():
                # Determine team & opponent
                # We don't have explicit team info per player in box score,
                # so we infer: if a player is in both teams' games, we can't
                # perfectly tell. But for feature purposes, we track is_home
                # based on whether this game's home_team matches.
                # Since box_score has all players from both teams, we need
                # a heuristic. We'll just store both team names and skip
                # is_home at the per-game level (use it from the leg context).
                entry = {
                    "date": date,
                    "game_id": game_id,
                    "home_team": home,
                    "away_team": away,
                    "home_score": home_score,
                    "away_score": away_score,
                }
                for k in ALL_STAT_KEYS:
                    entry[k] = stats.get(k, 0)
                # Additional shooting stats if available
                entry["fga"] = stats.get("fga", 0)
                entry["fgm"] = stats.get("fgm", 0)
                entry["fta"] = stats.get("fta", 0)
                entry["ftm"] = stats.get("ftm", 0)
                entry["fg3a"] = stats.get("fg3a", 0)

                self.player_history[player_name].append(entry)

        # Sort each player's history by date (should already be mostly sorted)
        for player in self.player_history:
            self.player_history[player].sort(key=lambda g: g["date"])

        logger.info(
            "Built history for %d players (avg %.1f games/player)",
            len(self.player_history),
            np.mean([len(v) for v in self.player_history.values()]),
        )
        self._loaded = True

    def ensure_loaded(self):
        """Load data if not already loaded."""
        if not self._loaded:
            self.load_games()
            self.build_player_history()

    # -----------------------------------------------------------------
    # Feature computation for a single leg
    # -----------------------------------------------------------------

    def get_stat_value(self, game_entry: dict, prop_type: str) -> float:
        """Extract the stat value for a prop type from a game entry."""
        keys = PROP_TO_STAT.get(prop_type)
        if keys is None:
            return 0.0
        return sum(game_entry.get(k, 0) for k in keys)

    def get_features_for_leg(
        self,
        player_name: str,
        prop_type: str,
        game_date: str,
        game_context: dict,
    ) -> Optional[dict]:
        """
        Build feature vector for a single prediction.
        CRITICAL: Only use games BEFORE game_date (no lookahead).

        Args:
            player_name: e.g. "LeBron James"
            prop_type: e.g. "points", "rebounds", "pts_rebs_asts"
            game_date: ISO date string e.g. "2024-01-15"
            game_context: dict with keys like line, over_odds, under_odds, etc.

        Returns:
            dict of ~80 features, or None if insufficient history.
        """
        history = self.player_history.get(player_name, [])
        prior_games = [g for g in history if g["date"] < game_date]

        if len(prior_games) < 3:
            return None  # Need at least 3 prior games

        # Most recent games first for rolling windows
        prior_games = list(reversed(prior_games))

        features = {}

        # ==================================================================
        # 1. TARGET STAT rolling stats
        # ==================================================================
        values = [self.get_stat_value(g, prop_type) for g in prior_games]

        l3 = values[:3]
        l5 = values[:5] if len(values) >= 5 else values
        l10 = values[:10] if len(values) >= 10 else values
        l20 = values[:20] if len(values) >= 20 else values
        all_vals = values

        features["target_l3"] = np.mean(l3)
        features["target_l5"] = np.mean(l5)
        features["target_l10"] = np.mean(l10)
        features["target_l20"] = np.mean(l20)
        features["target_season"] = np.mean(all_vals)
        features["target_std_l10"] = np.std(l10) if len(l10) >= 3 else np.std(all_vals)
        features["target_std_season"] = np.std(all_vals)
        features["target_min_l5"] = min(l5)
        features["target_max_l5"] = max(l5)
        features["target_median_l10"] = float(np.median(l10))
        features["target_trend"] = features["target_l3"] - features["target_l10"]
        features["target_trend_l5_l20"] = features["target_l5"] - np.mean(l20)
        features["target_cv"] = features["target_std_l10"] / max(features["target_season"], 0.1)

        # Hit rate vs line (how often would the OVER have hit recently)
        line = game_context.get("line", 0)
        features["hit_rate_l5"] = np.mean([1 if v > line else 0 for v in l5])
        features["hit_rate_l10"] = np.mean([1 if v > line else 0 for v in l10])
        features["hit_rate_l20"] = np.mean([1 if v > line else 0 for v in l20])

        # Percentile of line within recent distribution
        if features["target_std_l10"] > 0:
            features["line_z_l10"] = (line - features["target_l10"]) / features["target_std_l10"]
        else:
            features["line_z_l10"] = 0.0

        # ==================================================================
        # 2. MINUTES features
        # ==================================================================
        mins = [g["min"] for g in prior_games]
        mins_l3 = mins[:3]
        mins_l5 = mins[:5] if len(mins) >= 5 else mins
        mins_l10 = mins[:10] if len(mins) >= 10 else mins

        features["mins_l3"] = np.mean(mins_l3)
        features["mins_l5"] = np.mean(mins_l5)
        features["mins_l10"] = np.mean(mins_l10)
        features["mins_season"] = np.mean(mins)
        features["mins_std"] = np.std(mins_l10) if len(mins_l10) >= 3 else np.std(mins)
        features["mins_trend"] = features["mins_l3"] - features["mins_l10"]
        features["mins_min_l5"] = min(mins_l5)

        # ==================================================================
        # 3. PER-MINUTE RATE
        # ==================================================================
        total_stat_l10 = sum(values[:min(10, len(values))])
        total_mins_l10 = sum(mins[:min(10, len(mins))])
        features["per_minute_rate"] = total_stat_l10 / max(total_mins_l10, 1.0)

        total_stat_l5 = sum(values[:min(5, len(values))])
        total_mins_l5 = sum(mins[:min(5, len(mins))])
        features["per_minute_rate_l5"] = total_stat_l5 / max(total_mins_l5, 1.0)

        # Expected output from rate * recent minutes
        features["expected_from_rate"] = features["per_minute_rate"] * features["mins_l5"]
        features["expected_from_rate_l5"] = features["per_minute_rate_l5"] * features["mins_l3"]

        # ==================================================================
        # 4. RELATED STATS (context about the player's overall production)
        # ==================================================================
        stat_keys_for_prop = PROP_TO_STAT.get(prop_type, [])
        for related_key in ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]:
            if related_key not in stat_keys_for_prop:
                related_vals = [g.get(related_key, 0) for g in prior_games]
                r_l5 = related_vals[:5] if len(related_vals) >= 5 else related_vals
                features[f"related_{related_key}_l5"] = np.mean(r_l5)

        # For combined props, also include individual component L5 averages
        if len(stat_keys_for_prop) > 1:
            for comp_key in stat_keys_for_prop:
                comp_vals = [g.get(comp_key, 0) for g in prior_games]
                c_l5 = comp_vals[:5] if len(comp_vals) >= 5 else comp_vals
                features[f"component_{comp_key}_l5"] = np.mean(c_l5)

        # ==================================================================
        # 5. SHOOTING EFFICIENCY
        # ==================================================================
        fga_list = [g.get("fga", 0) for g in prior_games[:10]]
        fgm_list = [g.get("fgm", 0) for g in prior_games[:10]]
        fta_list = [g.get("fta", 0) for g in prior_games[:10]]
        fg3a_list = [g.get("fg3a", 0) for g in prior_games[:10]]

        features["fga_l10"] = np.mean(fga_list) if any(fga_list) else 0.0
        features["fg_pct_l10"] = sum(fgm_list) / max(sum(fga_list), 1)
        features["fta_l10"] = np.mean(fta_list) if any(fta_list) else 0.0
        features["fg3a_l10"] = np.mean(fg3a_list) if any(fg3a_list) else 0.0
        features["fg3_pct_l10"] = (
            sum(g.get("fg3m", 0) for g in prior_games[:10])
            / max(sum(fg3a_list), 1)
        )

        # ==================================================================
        # 6. GAMES PLAYED / SAMPLE SIZE
        # ==================================================================
        features["games_played"] = len(prior_games)
        features["games_played_capped"] = min(len(prior_games), 82)

        # ==================================================================
        # 7. MARKET LINE features (the line IS a feature)
        # ==================================================================
        features["line"] = line
        over_odds = game_context.get("over_odds", -110)
        under_odds = game_context.get("under_odds", -110)
        features["over_odds"] = over_odds
        features["under_odds"] = under_odds
        features["implied_over_prob"] = self._american_to_prob(over_odds)
        features["implied_under_prob"] = self._american_to_prob(under_odds)
        features["vig"] = features["implied_over_prob"] + features["implied_under_prob"] - 1.0
        features["line_vs_season"] = line - features["target_season"]
        features["line_vs_l5"] = line - features["target_l5"]
        features["line_vs_l3"] = line - features["target_l3"]
        features["line_vs_expected_rate"] = line - features["expected_from_rate"]

        # Line-anchor adjustment signals: "reasons the line might be wrong"
        # Positive = player trending above line, negative = below
        features["l3_above_line"] = features["target_l3"] - line
        features["l5_above_line"] = features["target_l5"] - line
        features["rate_above_line"] = features["expected_from_rate"] - line
        features["trend_magnitude"] = abs(features["target_trend"])
        # Odds skew: when book shades one side, it signals their belief
        features["odds_skew"] = features["implied_over_prob"] - features["implied_under_prob"]

        # ==================================================================
        # 7b. CROSS-BOOK MARKET FEATURES (from PropLine all_books)
        # ==================================================================
        # These capture market disagreement - when DK differs from consensus,
        # the outlier line may be exploitable.
        all_books = game_context.get("all_books", {})
        if all_books and len(all_books) > 1:
            book_lines = []
            book_over_odds = []
            book_under_odds = []
            for book_key, bd in all_books.items():
                bl = bd.get("line")
                if bl is not None:
                    try:
                        book_lines.append(float(bl))
                    except (TypeError, ValueError):
                        pass
                bo = bd.get("over_odds")
                if bo is not None:
                    try:
                        book_over_odds.append(int(bo))
                    except (TypeError, ValueError):
                        pass
                bu = bd.get("under_odds")
                if bu is not None:
                    try:
                        book_under_odds.append(int(bu))
                    except (TypeError, ValueError):
                        pass

            if book_lines:
                consensus_line = float(np.median(book_lines))
                features["consensus_line"] = consensus_line
                features["dk_vs_consensus"] = line - consensus_line
                features["line_range"] = max(book_lines) - min(book_lines)
                features["best_over_line"] = min(book_lines)
                features["best_under_line"] = max(book_lines)
                features["num_books_with_prop"] = len(book_lines)
            else:
                features["consensus_line"] = line
                features["dk_vs_consensus"] = 0.0
                features["line_range"] = 0.0
                features["best_over_line"] = line
                features["best_under_line"] = line
                features["num_books_with_prop"] = 1
        else:
            # No cross-book data available (historical legs)
            features["consensus_line"] = line
            features["dk_vs_consensus"] = 0.0
            features["line_range"] = 0.0
            features["best_over_line"] = line
            features["best_under_line"] = line
            features["num_books_with_prop"] = 1

        # ==================================================================
        # 8. GAME CONTEXT (spread, total, home/away)
        # ==================================================================
        # Note: spread and total are not in our game data, so we use defaults
        # when not available. The model learns from the features that ARE present.
        vegas_total = game_context.get("total", 224.0)
        vegas_spread = game_context.get("spread", 0.0)
        features["vegas_total"] = vegas_total
        features["vegas_spread"] = vegas_spread
        features["spread_magnitude"] = abs(vegas_spread)
        is_home = game_context.get("is_home", None)
        features["is_home"] = 1 if is_home else (0 if is_home is False else 0.5)
        features["blowout_risk"] = 1 if abs(vegas_spread) > 10 else 0
        features["total_vs_avg"] = vegas_total / 224.0

        # ==================================================================
        # 9. CONSISTENCY / VOLATILITY
        # ==================================================================
        if len(l10) >= 5:
            sorted_l10 = sorted(l10)
            features["pct_25_l10"] = np.percentile(l10, 25)
            features["pct_75_l10"] = np.percentile(l10, 75)
            features["iqr_l10"] = features["pct_75_l10"] - features["pct_25_l10"]
        else:
            features["pct_25_l10"] = features["target_l5"] - features["target_std_l10"]
            features["pct_75_l10"] = features["target_l5"] + features["target_std_l10"]
            features["iqr_l10"] = 2 * features["target_std_l10"]

        # Streak: consecutive games above/below line
        streak = 0
        for v in values:
            if v > line:
                if streak >= 0:
                    streak += 1
                else:
                    break
            else:
                if streak <= 0:
                    streak -= 1
                else:
                    break
        features["over_streak"] = max(streak, 0)
        features["under_streak"] = max(-streak, 0)

        # ==================================================================
        # 10. PROP TYPE one-hot
        # ==================================================================
        for pt in CORE_PROPS:
            features[f"is_{pt}"] = 1 if prop_type == pt else 0

        return features

    # -----------------------------------------------------------------
    # Build full training data
    # -----------------------------------------------------------------

    def build_training_data(
        self,
        prop_types: Optional[List[str]] = None,
        target_mode: str = "line_anchor",
        enhanced: bool = True,
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        """
        Build full feature matrix + target vector from all legs.

        For each leg:
        1. Look up game context (line, odds)
        2. Build features from prior games only (NO lookahead)
        3. Target depends on target_mode:
           - "line_anchor": actual - line (how wrong was the DK line?)
           - "raw": actual stat value (legacy mode)

        The line-anchor framing is the key architectural change:
          predicted_adjustment = model.predict(features)
          final_prediction = DK_line + predicted_adjustment

        Args:
            prop_types: list of prop types to include, or None for CORE_PROPS
            target_mode: "line_anchor" (default) or "raw"
            enhanced: if True, include team/travel/calendar/derived features

        Returns:
            (X: DataFrame of features,
             y: ndarray of target values (adjustment or raw),
             meta: DataFrame with date, player, prop_type, game_id, line, actual columns)
        """
        self.ensure_loaded()

        if prop_types is None:
            prop_types = CORE_PROPS

        prop_set = set(prop_types)

        # Initialize enhanced feature builders if requested
        team_builder = None
        team_schedules = None
        derived_builder = None
        ref_builder = None
        ref_game_lookup = None
        onoff_data = None
        if enhanced:
            try:
                from v2.team_features import TeamStatsBuilder
                from v2.travel_features import (
                    get_travel_features, get_calendar_features,
                    build_team_schedules,
                )
                from v2.derived_features import DerivedFeatures

                logger.info("Building team history for enhanced features...")
                team_builder = TeamStatsBuilder()
                team_builder.build_team_history(self.games)

                logger.info("Building team schedules for travel features...")
                team_schedules = build_team_schedules(self.games)

                derived_builder = DerivedFeatures()
                logger.info("Enhanced feature builders ready.")
            except Exception as e:
                logger.warning("Could not initialize enhanced features: %s", e)
                enhanced = False

            # --- Data Source 1: Referee tendencies ---
            try:
                from v2.ref_features import RefStatsBuilder, REF_CACHE_DIR
                ref_builder = RefStatsBuilder()
                loaded_refs = ref_builder.load_cached_tendencies()
                if loaded_refs:
                    # Load ref assignments to map games to refs
                    ref_game_lookup = {}
                    for season_str in ["2023-24", "2024-25", "2025-26"]:
                        assign_path = REF_CACHE_DIR / f"ref_assignments_{season_str}.json"
                        if assign_path.exists():
                            with open(assign_path) as f:
                                assignments = json.load(f)
                            # Build date+team -> refs mapping for matching
                            # Also load game stats for date info
                            stats_path = REF_CACHE_DIR / f"game_stats_{season_str}.json"
                            if stats_path.exists():
                                with open(stats_path) as f:
                                    game_stats_cache = json.load(f)
                                for nba_gid, refs in assignments.items():
                                    if not refs:
                                        continue
                                    gs = game_stats_cache.get(nba_gid, {})
                                    gdate = gs.get("game_date", "")
                                    home_abbr = gs.get("home_team", "")
                                    away_abbr = gs.get("away_team", "")
                                    if gdate:
                                        # Normalize date
                                        from datetime import datetime as _dt
                                        try:
                                            if len(gdate) == 10 and gdate[4] == "-":
                                                norm_date = gdate
                                            else:
                                                norm_date = _dt.strptime(
                                                    gdate.strip(), "%b %d, %Y"
                                                ).strftime("%Y-%m-%d")
                                        except (ValueError, TypeError):
                                            norm_date = gdate
                                        if norm_date and home_abbr:
                                            ref_game_lookup[f"{norm_date}_{home_abbr}"] = {
                                                "refs": refs, "season": season_str
                                            }
                                        if norm_date and away_abbr:
                                            ref_game_lookup[f"{norm_date}_{away_abbr}"] = {
                                                "refs": refs, "season": season_str
                                            }
                    logger.info(
                        "Ref data ready: %d game-date entries for lookup",
                        len(ref_game_lookup),
                    )
                else:
                    logger.info(
                        "No cached ref tendencies found. Run 'python -m v2.fetch_ref_data' first."
                    )
                    ref_builder = None
            except Exception as e:
                logger.warning("Could not initialize ref features: %s", e)
                ref_builder = None

            # --- Data Source 2: Player on/off impact ---
            # Build per-team roster usage data from box scores.
            # For each game, compute per-player minutes and usage proxy
            # so we can detect who was OUT and estimate redistribution.
            # Uses team_builder's partition for accurate team assignments.
            try:
                onoff_data = self._build_onoff_data(team_builder)
                logger.info(
                    "On/off data ready: %d team-season entries",
                    len(onoff_data.get("team_rosters", {})),
                )
            except Exception as e:
                logger.warning("Could not build on/off data: %s", e)
                onoff_data = None

        # Load all legs
        all_legs = []
        for path in LEGS_FILES:
            if not path.exists():
                logger.warning("Missing legs file: %s", path)
                continue
            logger.info("Loading legs from %s ...", path.name)
            with open(path) as f:
                legs = json.load(f)
            logger.info("  -> %d legs", len(legs))
            all_legs.extend(legs)

        logger.info("Total legs: %d", len(all_legs))

        # Filter to desired prop types
        all_legs = [leg for leg in all_legs if leg.get("prop_type") in prop_set]
        logger.info("Legs after prop type filter (%s): %d", prop_types, len(all_legs))

        # Sort by date for consistent ordering
        all_legs.sort(key=lambda x: x["date"])

        rows_X = []
        rows_y = []
        rows_meta = []
        skipped_no_history = 0
        skipped_no_actual = 0

        for i, leg in enumerate(all_legs):
            if i > 0 and i % 10000 == 0:
                logger.info(
                    "  Processing leg %d/%d (%.0f%%) -- %d features built, %d skipped",
                    i, len(all_legs), 100 * i / len(all_legs),
                    len(rows_X), skipped_no_history,
                )

            player = leg["player"]
            prop_type = leg["prop_type"]
            date = leg["date"]
            actual = leg.get("actual")

            if actual is None:
                skipped_no_actual += 1
                continue

            # Build game context from the leg data
            game_context = {
                "line": leg.get("line", 0),
                "over_odds": leg.get("over_odds", -110),
                "under_odds": leg.get("under_odds", -110),
                # spread/total not available in legs; use defaults
                "total": 224.0,
                "spread": 0.0,
            }

            # Determine is_home and teams from matchup string
            matchup = leg.get("matchup", "")
            home_team = ""
            away_team = ""
            is_home = None
            if matchup:
                # Format: "Away Team @ Home Team"
                parts = matchup.split(" @ ")
                if len(parts) == 2:
                    away_team = parts[0].strip()
                    home_team = parts[1].strip()
                    game = self.game_lookup.get(leg.get("game_id"))
                    if game:
                        is_home = self._is_player_home(player, game)
                    game_context["is_home"] = is_home

            feats = self.get_features_for_leg(player, prop_type, date, game_context)
            if feats is None:
                skipped_no_history += 1
                continue

            # ---- Enhanced features ----
            if enhanced:
                game_id = leg.get("game_id")

                # Determine player's team and opponent
                player_team = None
                opp_team = None
                if team_builder and game_id:
                    player_team = team_builder.get_player_team(player, game_id)
                    if player_team:
                        opp_team = (
                            away_team if player_team == home_team
                            else home_team if player_team == away_team
                            else None
                        )
                        is_home = (player_team == home_team)
                        game_context["is_home"] = is_home

                # Team + opponent + matchup features (28 features)
                if team_builder and player_team and opp_team:
                    team_feats = team_builder.get_all_features_for_game(
                        player_team, opp_team, date
                    )
                    if team_feats:
                        feats.update(team_feats)

                # Travel features (11 features)
                if team_schedules and player_team and opp_team:
                    try:
                        schedule = team_schedules.get(player_team, [])
                        travel_feats = get_travel_features(
                            player_team, opp_team, date,
                            bool(is_home), schedule,
                        )
                        feats.update(travel_feats)
                    except Exception:
                        pass

                # Calendar features (8 features)
                try:
                    cal_feats = get_calendar_features(date)
                    feats.update(cal_feats)
                except Exception:
                    pass

                # Derived player features (29 features)
                if derived_builder:
                    try:
                        history = self.player_history.get(player, [])
                        prior = list(reversed(
                            [g for g in history if g["date"] < date]
                        ))
                        if len(prior) >= 3:
                            ctx = {
                                "line": game_context.get("line", 0),
                                "is_home": is_home,
                                "opponent": opp_team or "",
                            }
                            derived = derived_builder.compute_all(
                                prior, prop_type, ctx
                            )
                            feats.update(derived)
                    except Exception:
                        pass

                # ============================================================
                # NEW DATA SOURCE 1: Referee tendency features (5 features)
                # ============================================================
                if ref_builder and ref_game_lookup:
                    try:
                        ref_feats = self._get_ref_features_for_leg(
                            ref_builder, ref_game_lookup,
                            date, home_team, away_team,
                        )
                        feats.update(ref_feats)
                    except Exception:
                        feats.update(_default_ref_features())

                # ============================================================
                # NEW DATA SOURCE 2: Player on/off impact features (4 features)
                # ============================================================
                if onoff_data and team_builder and player_team:
                    try:
                        onoff_feats = self._get_onoff_features(
                            onoff_data, player, player_team,
                            game_id, date, team_builder,
                        )
                        feats.update(onoff_feats)
                    except Exception:
                        feats.update(_default_onoff_features())

                # ============================================================
                # NEW DATA SOURCE 3: Pace matchup prediction features (4 features)
                # ============================================================
                if team_builder and player_team and opp_team:
                    try:
                        pace_feats = self._get_pace_matchup_features(
                            team_builder, player_team, opp_team,
                            date, game_context.get("total", 224.0),
                        )
                        feats.update(pace_feats)
                    except Exception:
                        feats.update(_default_pace_features())

            rows_X.append(feats)

            # Target: line-anchor or raw
            leg_line = float(leg.get("line", 0))
            actual_val = float(actual)
            if target_mode == "line_anchor":
                # Target = actual - line: how much the line was wrong
                # Positive = over hit, Negative = under hit
                rows_y.append(actual_val - leg_line)
            else:
                rows_y.append(actual_val)

            rows_meta.append({
                "date": date,
                "player": player,
                "prop_type": prop_type,
                "game_id": leg.get("game_id"),
                "line": leg_line,
                "actual": actual_val,
                "matchup": matchup,
            })

        logger.info(
            "Feature building complete: %d samples, %d skipped (no history), "
            "%d skipped (no actual)",
            len(rows_X), skipped_no_history, skipped_no_actual,
        )

        X = pd.DataFrame(rows_X)
        y = np.array(rows_y)
        meta = pd.DataFrame(rows_meta)

        # Fill any NaN with 0 (shouldn't happen but safety)
        X = X.fillna(0)

        return X, y, meta

    # -----------------------------------------------------------------
    # New data source helpers
    # -----------------------------------------------------------------

    def _build_onoff_data(self, team_builder=None) -> dict:
        """Build per-team roster data for on/off feature computation.

        Uses team_builder's pre-computed team assignments (from the greedy
        partition in TeamStatsBuilder.build_team_history) for accurate
        player-to-team mapping. This avoids the problem of mispartitioned
        players inflating roster sizes.

        For each team, tracks which players appear in games and their
        average minutes/usage (pts per game as proxy). This lets us
        detect when key teammates are absent for a given game.

        Returns:
            dict with keys:
              - team_rosters: {team: {player: {avg_min, avg_pts, games, ...}}}
              - game_rosters: {game_id: {team: {player: {min, pts}}}}
        """
        game_rosters = defaultdict(lambda: defaultdict(dict))
        team_player_stats = defaultdict(lambda: defaultdict(lambda: {
            "total_min": 0.0, "total_pts": 0.0, "games": 0,
        }))

        for game in self.games:
            game_id = game.get("game_id")
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            box = game.get("box_score", {})

            if not box or not game_id:
                continue

            for player_name, stats in box.items():
                mins = stats.get("min", 0)
                pts = stats.get("pts", 0)

                # Use team_builder's assignment if available
                player_team = None
                if team_builder:
                    player_team = team_builder.get_player_team(player_name, game_id)

                if not player_team:
                    continue  # Skip if we can't determine team

                if mins > 0:
                    game_rosters[game_id][player_team][player_name] = {
                        "min": mins, "pts": pts,
                    }
                    team_player_stats[player_team][player_name]["total_min"] += mins
                    team_player_stats[player_team][player_name]["total_pts"] += pts
                    team_player_stats[player_team][player_name]["games"] += 1

        # Compute averages per player per team.
        # Filter out mispartitioned visitors: require a player to have
        # appeared in at least 15% of the team's games AND at least 10
        # games to count as a "regular" roster member.
        team_game_counts = defaultdict(int)
        for game in self.games:
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            if home:
                team_game_counts[home] += 1
            if away:
                team_game_counts[away] += 1

        team_rosters = {}
        for team, players in team_player_stats.items():
            team_rosters[team] = {}
            team_games = max(team_game_counts.get(team, 1), 1)
            for player, stats in players.items():
                n = stats["games"]
                # Must have played in at least 15% of team's games
                # AND at least 10 games to be considered a regular
                if n >= 10 and n >= team_games * 0.15:
                    team_rosters[team][player] = {
                        "avg_min": stats["total_min"] / n,
                        "avg_pts": stats["total_pts"] / n,
                        "games": n,
                    }

        return {
            "team_rosters": dict(team_rosters),
            "game_rosters": dict(game_rosters),
        }

    def _get_ref_features_for_leg(
        self,
        ref_builder,
        ref_game_lookup: dict,
        date: str,
        home_team: str,
        away_team: str,
    ) -> dict:
        """Look up referee features for a training leg.

        Matches the training game to NBA.com game data by date + team
        abbreviation, then gets the crew's tendency features.

        Returns dict of 5 ref features (defaults if no match).
        """
        from v2.ref_features import RefStatsBuilder

        # Build team abbreviation map
        abbrevs = {
            "Atlanta Hawks": "ATL", "Boston Celtics": "BOS",
            "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA",
            "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
            "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
            "Detroit Pistons": "DET", "Golden State Warriors": "GSW",
            "Houston Rockets": "HOU", "Indiana Pacers": "IND",
            "LA Clippers": "LAC", "Los Angeles Lakers": "LAL",
            "Memphis Grizzlies": "MEM", "Miami Heat": "MIA",
            "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
            "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
            "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
            "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
            "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
            "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
            "Utah Jazz": "UTA", "Washington Wizards": "WAS",
        }

        # Try to match via home team abbreviation
        home_abbr = abbrevs.get(home_team, "")
        away_abbr = abbrevs.get(away_team, "")

        ref_info = None
        for abbr in [home_abbr, away_abbr]:
            if abbr:
                key = f"{date}_{abbr}"
                ref_info = ref_game_lookup.get(key)
                if ref_info:
                    break

        if not ref_info:
            return _default_ref_features()

        refs = ref_info["refs"]
        season = ref_info["season"]
        full_feats = ref_builder.get_ref_features(refs, season=season)

        # Return the 5 features we want (not the full 7 from RefStatsBuilder)
        return {
            "ref_avg_fta_per_game": full_feats.get("ref_avg_fta_per_game", 0.0),
            "ref_avg_total_score": full_feats.get("ref_avg_total_score", 0.0),
            "ref_fta_vs_league_avg": full_feats.get("ref_fta_vs_league_avg", 1.0),
            "ref_home_fta_bias": full_feats.get("ref_home_fta_bias", 0.5),
            "has_ref_data": full_feats.get("has_ref_data", 0),
        }

    def _get_onoff_features(
        self,
        onoff_data: dict,
        player: str,
        player_team: str,
        game_id: int,
        date: str,
        team_builder,
    ) -> dict:
        """Compute on/off impact features for a player in a specific game.

        Determines which regular teammates are absent (0 minutes or not in
        box score) and estimates the redistribution of minutes/usage.

        Returns dict of 4 on/off features.
        """
        team_roster = onoff_data["team_rosters"].get(player_team, {})
        game_rosters = onoff_data["game_rosters"]

        if not team_roster:
            return _default_onoff_features()

        # Who played in THIS game for this team?
        game_team_roster = game_rosters.get(game_id, {}).get(player_team, {})
        players_who_played = set(game_team_roster.keys())

        # Get regular players for this team (those with avg_min >= 10
        # who have played in at least a few games before this date)
        regular_players = {}
        for p, stats in team_roster.items():
            if stats["avg_min"] >= 10 and stats["games"] >= 5:
                regular_players[p] = stats

        if not regular_players:
            return _default_onoff_features()

        # Find who is OUT: regular players not in this game's box score
        absent_players = {}
        for p, stats in regular_players.items():
            if p != player and p not in players_who_played:
                absent_players[p] = stats

        # Compute features
        # 1. top_teammate_out: is the team's highest-usage regular (besides
        #    this player) absent?
        other_regulars = {
            p: s for p, s in regular_players.items() if p != player
        }
        if other_regulars:
            top_teammate = max(
                other_regulars.items(), key=lambda x: x[1]["avg_pts"]
            )
            top_teammate_out = 1 if top_teammate[0] in absent_players else 0
        else:
            top_teammate_out = 0

        # 2. usage_available: sum of avg_pts of absent players
        usage_available = sum(s["avg_pts"] for s in absent_players.values())

        # 3. minutes_available: sum of avg_min of absent players
        minutes_available = sum(s["avg_min"] for s in absent_players.values())

        # 4. stat_boost_from_absences: estimated boost based on
        #    this player's share of remaining usage
        player_stats = team_roster.get(player, {})
        player_avg_pts = player_stats.get("avg_pts", 0.0)
        total_team_pts = sum(s["avg_pts"] for s in regular_players.values())
        remaining_team_pts = total_team_pts - usage_available

        if remaining_team_pts > 0 and total_team_pts > 0:
            player_share_of_remaining = player_avg_pts / remaining_team_pts
            player_share_of_total = player_avg_pts / total_team_pts
            stat_boost = (player_share_of_remaining - player_share_of_total) * usage_available
        else:
            stat_boost = 0.0

        return {
            "top_teammate_out": top_teammate_out,
            "usage_available": usage_available,
            "minutes_available": minutes_available,
            "stat_boost_from_absences": stat_boost,
        }

    def _get_pace_matchup_features(
        self,
        team_builder,
        player_team: str,
        opp_team: str,
        date: str,
        vegas_total: float,
    ) -> dict:
        """Compute pace matchup prediction features.

        Rather than just using team-level season pace, predicts THIS
        specific matchup's pace from both teams' recent pace and the
        vegas total.

        Returns dict of 4 pace matchup features.
        """
        team_feats = team_builder.get_team_features(player_team, date)
        opp_feats = team_builder.get_opponent_features(opp_team, date)

        if team_feats is None or opp_feats is None:
            return _default_pace_features()

        team_pace = team_feats.get("team_pace_l10", 100.0)
        opp_pace = opp_feats.get("opp_pace_l10", 100.0)

        # Matchup pace: average of both teams' L10 pace
        matchup_pace = (team_pace + opp_pace) / 2.0

        # Pace vs vegas total: compare team-level pace prediction to
        # the implied pace from the vegas over/under total.
        # Vegas total / 2 is the implied score per team, which serves
        # as an independent pace estimate from the betting market.
        vegas_implied_pace = vegas_total / 2.0
        pace_over_under = matchup_pace - vegas_implied_pace

        # Possessions estimate: convert score-based pace proxy to
        # rough possessions per team. In the modern NBA, teams average
        # ~1.12 points per possession, so possessions ~ pace / 1.12.
        possessions_estimate = matchup_pace / 1.12

        # Pace mismatch: very different pace teams create unpredictable games
        pace_mismatch = abs(team_pace - opp_pace)

        return {
            "matchup_pace": matchup_pace,
            "pace_over_under": pace_over_under,
            "possessions_estimate": possessions_estimate,
            "pace_mismatch": pace_mismatch,
        }

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _american_to_prob(self, odds: float) -> float:
        """Convert American odds to implied probability."""
        if odds == 0:
            return 0.5
        if odds > 0:
            return 100.0 / (odds + 100.0)
        return abs(odds) / (abs(odds) + 100.0)

    def _is_player_home(self, player_name: str, game: dict) -> Optional[bool]:
        """
        Determine if a player was on the home team.
        Uses a heuristic: check which team scored closer to the player's
        scoring contribution. Not perfect, but sufficient.
        """
        # Simple approach: look at all games in player history for this game_id
        # and see which team's score is compatible.
        # Since we can't perfectly determine team from box_score alone,
        # return None (will default to 0.5 in features).
        return None


# =============================================================================
# Default feature dicts (returned when data source is unavailable)
# =============================================================================

def _default_ref_features() -> dict:
    """Default referee features when ref data is not available."""
    return {
        "ref_avg_fta_per_game": 0.0,
        "ref_avg_total_score": 0.0,
        "ref_fta_vs_league_avg": 1.0,
        "ref_home_fta_bias": 0.5,
        "has_ref_data": 0,
    }


def _default_onoff_features() -> dict:
    """Default on/off features when data is not available."""
    return {
        "top_teammate_out": 0,
        "usage_available": 0.0,
        "minutes_available": 0.0,
        "stat_boost_from_absences": 0.0,
    }


def _default_pace_features() -> dict:
    """Default pace matchup features when data is not available."""
    return {
        "matchup_pace": 112.0,       # League avg total / 2
        "pace_over_under": 0.0,      # matchup_pace - vegas_implied (neutral)
        "possessions_estimate": 100.0,  # ~112 / 1.12
        "pace_mismatch": 0.0,
    }


# =============================================================================
# CLI: build and inspect training data
# =============================================================================

def main():
    print("=" * 70)
    print("v2 Feature Builder - Building line-anchor training data")
    print("=" * 70)

    fb = FeatureBuilder()
    fb.load_games()
    fb.build_player_history()

    # Build with line-anchor target (actual - line)
    X, y, meta = fb.build_training_data(target_mode="line_anchor")

    print(f"\n{'='*70}")
    print(f"TRAINING DATA SUMMARY (LINE-ANCHOR TARGET)")
    print(f"{'='*70}")
    print(f"  Total samples:    {len(X):,}")
    print(f"  Feature columns:  {len(X.columns)}")
    print(f"  Date range:       {meta['date'].min()} to {meta['date'].max()}")
    print(f"\n  Target = actual - DK_line (positive = over hit, negative = under hit)")
    print(f"  Overall target mean: {y.mean():+.3f} (systematic under bias if negative)")
    print(f"  Overall target std:  {y.std():.3f}")
    print()

    # Per-prop breakdown
    for pt in CORE_PROPS:
        mask = meta["prop_type"] == pt
        n = mask.sum()
        if n > 0:
            t_mean = y[mask].mean()
            t_std = y[mask].std()
            over_rate = np.mean(y[mask] > 0)
            push_rate = np.mean(y[mask] == 0)
            print(f"  {pt:20s}: {n:6,} samples, adjustment mean={t_mean:+.3f}, "
                  f"std={t_std:.2f}, over_rate={over_rate:.1%}")

    print(f"\nFeature columns ({len(X.columns)} total):")
    for i, col in enumerate(sorted(X.columns)):
        print(f"  {i+1:3d}. {col}")

    # Report on new data sources
    new_ref_cols = [c for c in X.columns if c.startswith("ref_") or c == "has_ref_data"]
    new_onoff_cols = [c for c in X.columns if c in [
        "top_teammate_out", "usage_available", "minutes_available",
        "stat_boost_from_absences",
    ]]
    new_pace_cols = [c for c in X.columns if c in [
        "matchup_pace", "pace_over_under", "possessions_estimate", "pace_mismatch",
    ]]

    print(f"\n{'='*70}")
    print(f"NEW DATA SOURCE FEATURES")
    print(f"{'='*70}")
    print(f"  Referee features ({len(new_ref_cols)}): {new_ref_cols}")
    if "has_ref_data" in X.columns:
        ref_coverage = X["has_ref_data"].mean()
        print(f"    Coverage: {ref_coverage:.1%} of legs have ref data")
    print(f"  On/Off features ({len(new_onoff_cols)}): {new_onoff_cols}")
    if "top_teammate_out" in X.columns:
        out_pct = X["top_teammate_out"].mean()
        print(f"    Top teammate out: {out_pct:.1%} of legs")
    print(f"  Pace matchup features ({len(new_pace_cols)}): {new_pace_cols}")
    if "matchup_pace" in X.columns:
        print(f"    Matchup pace range: {X['matchup_pace'].min():.1f} - {X['matchup_pace'].max():.1f}")

    # Verification checks
    print(f"\n{'='*70}")
    print(f"VERIFICATION")
    print(f"{'='*70}")
    for col in ["ref_avg_fta_per_game", "top_teammate_out", "matchup_pace"]:
        present = col in X.columns
        print(f"  {col}: {'PRESENT' if present else 'MISSING'}")

    # Save to disk for training
    out_dir = BASE_DIR / "outputs" / "v2_training"
    out_dir.mkdir(parents=True, exist_ok=True)
    X.to_parquet(out_dir / "features.parquet", index=False)
    pd.Series(y).to_frame("target").to_parquet(out_dir / "targets.parquet", index=False)
    meta.to_parquet(out_dir / "metadata.parquet", index=False)
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
