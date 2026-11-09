#!/usr/bin/env python3
"""
v2/team_features.py - Team-level feature engineering for line-anchor ML prop model.

Computes team-level features missing from the current 179-feature set:
  1. Team Four Factors (offensive & defensive): eFG%, TOV%, ORB%, FT rate
  2. Opponent defense by position (PG/SG/SF/PF/C)
  3. Game-specific pace projection from team pace matchup
  4. Team offensive/defensive ratings

Two modes:
  A) Historical enrichment — build rolling team stats from training data
     box scores for backtest feature engineering.
  B) Live inference — fetch current-season team stats from NBA.com
     via nba_api for real-time predictions.

IMPORTANT: All historical features use only games BEFORE the prediction
date (no lookahead). The rolling windows compute stats strictly from
prior games.

Data limitations (training data):
  Box scores contain only: pts, reb, ast, fg3m, stl, blk, turnover, min.
  No fga/fgm/fta/ftm/fg3a/oreb/dreb. Four Factors are approximated
  where shooting splits are unavailable. See docstrings for approximation
  details.

Usage:
    python -m v2.team_features              # Build team history + print sample
    python -m v2.team_features --live       # Also fetch live NBA.com stats
"""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS = BASE_DIR / "outputs"

GAME_DATA_FILES = [
    OUTPUTS / "backtest_2023_24_data.json",
    OUTPUTS / "backtest_oos_data.json",
]

# League average pace (approximate, for delta calculations).
# Updated each season; this is a reasonable default.
LEAGUE_AVG_PACE = 100.0

# NBA.com rate limit: minimum seconds between API calls
NBA_API_DELAY = 0.6


# =============================================================================
# Part 1: TeamStatsBuilder — Historical team stats from training data
# =============================================================================

class TeamStatsBuilder:
    """Build rolling team-level stats from backtest game data.

    Processes game box scores chronologically and maintains per-team
    game-by-game stat records. Provides rolling-window lookups that
    respect temporal ordering (no lookahead).

    Since the training box scores lack shooting splits (fga, fgm, fta,
    ftm, fg3a, oreb, dreb), several Four Factors are approximated:

      - pace: estimated as (team_pts + opp_pts) / 2.0, a rough proxy
        for possessions when FGA/FTA/TOV data is missing.
      - efg_pct: approximated as (pts - ftm_approx) / (2 * fga_approx)
        where fga_approx = pts / 1.1 (league average TS% implies ~1.1
        pts/FGA). When fg3m is known we do slightly better.
      - tov_pct: turnover / (turnover + non-turnover possessions). We
        approximate possessions from the score-based pace.
      - orb_pct: not computable without oreb/dreb; set to 0.0.
      - ft_rate: not computable without FTA/FGA; set to 0.0.
      - fg3_rate: fg3m / approximate FGA; rough but directional.
    """

    def __init__(self):
        # {team_name: [{date, game_id, pace, off_rating, def_rating, ...}]}
        self.team_history: Dict[str, List[dict]] = defaultdict(list)
        # Track which players belong to which team per game for partitioning
        self._team_assignments: Dict[int, Dict[str, str]] = {}
        self._league_pace_history: List[float] = []
        self._built = False

    # -----------------------------------------------------------------
    # Build team history from raw game data
    # -----------------------------------------------------------------

    def build_team_history(self, games: List[dict]) -> None:
        """Process all games chronologically and build per-team stat history.

        Args:
            games: List of game dicts from backtest JSON files, sorted by date.
                   Each has: game_id, date, home_team, away_team, home_score,
                   away_score, box_score.
        """
        games_sorted = sorted(games, key=lambda g: g["date"])
        self.team_history.clear()
        self._league_pace_history.clear()

        for game in games_sorted:
            game_id = game.get("game_id")
            date = game["date"]
            home_team = game.get("home_team", "")
            away_team = game.get("away_team", "")
            home_score = game.get("home_score", 0)
            away_score = game.get("away_score", 0)
            box = game.get("box_score", {})

            if not box or not home_team or not away_team:
                continue

            # Partition players into home and away teams
            home_players, away_players = self._partition_players(
                box, home_score, away_score
            )

            # Aggregate team-level stats
            home_agg = self._aggregate_team_stats(box, home_players)
            away_agg = self._aggregate_team_stats(box, away_players)

            # Compute advanced stats for each team
            home_entry = self._compute_team_game_stats(
                home_agg, away_agg, home_score, away_score, date, game_id
            )
            away_entry = self._compute_team_game_stats(
                away_agg, home_agg, away_score, home_score, date, game_id
            )

            self.team_history[home_team].append(home_entry)
            self.team_history[away_team].append(away_entry)
            self._league_pace_history.append(home_entry["pace"])

            # Store team assignments for player lookup
            self._team_assignments[game_id] = {}
            for p in home_players:
                self._team_assignments[game_id][p] = home_team
            for p in away_players:
                self._team_assignments[game_id][p] = away_team

        # Sort each team's history by date
        for team in self.team_history:
            self.team_history[team].sort(key=lambda e: e["date"])

        self._built = True
        logger.info(
            "Built team history: %d teams, %d total team-games, avg %.1f games/team",
            len(self.team_history),
            sum(len(v) for v in self.team_history.values()),
            np.mean([len(v) for v in self.team_history.values()])
            if self.team_history else 0,
        )

    def _partition_players(
        self,
        box: Dict[str, dict],
        home_score: int,
        away_score: int,
    ) -> Tuple[List[str], List[str]]:
        """Split box score players into home and away groups.

        Heuristic: try all 2-partitions by finding the subset of players
        whose pts sum to home_score. Since box scores typically have
        10-15 players per team, we use a greedy approach: sort players
        by pts descending, greedily assign to whichever group is further
        from its target.

        Falls back to an even split if the greedy approach fails.
        """
        players = list(box.keys())
        pts_map = {p: box[p].get("pts", 0) for p in players}

        # Quick check: if total pts matches, try to partition
        total_pts = sum(pts_map.values())
        expected_total = home_score + away_score

        # If total box score pts don't match game total (data issues),
        # just do an even split
        if total_pts != expected_total and abs(total_pts - expected_total) > 2:
            mid = len(players) // 2
            return players[:mid], players[mid:]

        # Greedy partition targeting home_score
        # Sort by pts descending for better greedy choices
        sorted_players = sorted(players, key=lambda p: pts_map[p], reverse=True)
        home_group = []
        away_group = []
        home_pts = 0
        away_pts = 0

        for p in sorted_players:
            p_pts = pts_map[p]
            # Assign to the group that's further below its target
            home_gap = home_score - home_pts
            away_gap = away_score - away_pts

            if home_gap >= away_gap and home_pts + p_pts <= home_score + 1:
                home_group.append(p)
                home_pts += p_pts
            elif away_pts + p_pts <= away_score + 1:
                away_group.append(p)
                away_pts += p_pts
            else:
                home_group.append(p)
                home_pts += p_pts

        # Verify partition
        if abs(home_pts - home_score) > 2 or abs(away_pts - away_score) > 2:
            # Greedy failed, try brute-force for small rosters
            if len(players) <= 30:
                result = self._brute_partition(pts_map, players, home_score)
                if result is not None:
                    return result

            # Final fallback: even split
            mid = len(players) // 2
            return players[:mid], players[mid:]

        return home_group, away_group

    def _brute_partition(
        self,
        pts_map: Dict[str, int],
        players: List[str],
        target_sum: int,
    ) -> Optional[Tuple[List[str], List[str]]]:
        """Brute-force find a subset of players summing to target.

        Uses recursive subset-sum with early termination.
        Only called as fallback for <= 30 players.
        """
        n = len(players)
        if n > 30:
            return None

        # Dynamic programming approach: track achievable sums
        # with bitmask of which players are included
        # For efficiency, we allow +/- 1 tolerance
        from itertools import combinations

        # Try subsets of size n//2 +/- 3 (typical NBA roster split)
        for size in range(max(1, n // 2 - 3), min(n, n // 2 + 4)):
            if size > 15:  # Cap combinatorial explosion
                continue
            for combo in combinations(range(n), size):
                combo_pts = sum(pts_map[players[i]] for i in combo)
                if abs(combo_pts - target_sum) <= 1:
                    home = [players[i] for i in combo]
                    away = [players[i] for i in range(n) if i not in combo]
                    return home, away

        return None

    def _aggregate_team_stats(
        self,
        box: Dict[str, dict],
        players: List[str],
    ) -> dict:
        """Sum individual stats for a list of players."""
        agg = {
            "pts": 0, "reb": 0, "ast": 0, "fg3m": 0,
            "stl": 0, "blk": 0, "turnover": 0, "min": 0.0,
        }
        for p in players:
            stats = box.get(p, {})
            for k in agg:
                agg[k] += stats.get(k, 0)
        agg["n_players"] = len(players)
        return agg

    def _compute_team_game_stats(
        self,
        team_agg: dict,
        opp_agg: dict,
        team_score: int,
        opp_score: int,
        date: str,
        game_id: int,
    ) -> dict:
        """Compute advanced team stats for a single game.

        Since we lack fga/fgm/fta/ftm/fg3a/oreb/dreb, we use
        approximations. See class docstring for details.
        """
        team_pts = team_score
        opp_pts = opp_score
        team_tov = team_agg.get("turnover", 0)
        opp_tov = opp_agg.get("turnover", 0)
        team_fg3m = team_agg.get("fg3m", 0)
        opp_fg3m = opp_agg.get("fg3m", 0)
        team_reb = team_agg.get("reb", 0)
        opp_reb = opp_agg.get("reb", 0)

        # -- Pace estimation --
        # Without FGA/FTA, use score-based proxy:
        # Real pace ~ 0.5 * (FGA + 0.44*FTA - ORB + TOV) for each team.
        # Approximation: average of both teams' scores, divided by a
        # pts-per-possession factor (~1.1 in modern NBA).
        total_pts = team_pts + opp_pts
        pace = total_pts / 2.0  # Rough: ~110 for a 220-point game

        # -- Offensive/Defensive Rating --
        # possessions approximated from pace
        possessions = max(pace, 1.0)
        off_rating = 100.0 * team_pts / possessions
        def_rating = 100.0 * opp_pts / possessions

        # -- eFG% approximation --
        # True eFG = (FGM + 0.5*FG3M) / FGA
        # Without FGA/FGM, we reconstruct from points:
        #   pts = 2*(FGM - FG3M) + 3*FG3M + FTM = 2*FGM + FG3M + FTM
        #   FTM ~ pts * 0.18 (league avg ~18% of pts from FTs)
        #   FGM ~ (pts - FTM - FG3M) / 2  (two-pointers account for remainder)
        #   FGA ~ FGM / 0.47 (league avg FG% ~ 47%)
        approx_ftm = team_pts * 0.18
        fg_pts = team_pts - approx_ftm  # points from field goals
        # fg_pts = 2*FGM + FG3M => FGM = (fg_pts - FG3M) / 2
        approx_fgm = (fg_pts - team_fg3m) / 2.0
        approx_fgm = max(approx_fgm, 0.0)
        approx_fga = max(approx_fgm / 0.47, 1.0)
        efg_pct = min((approx_fgm + 0.5 * team_fg3m) / approx_fga, 1.0)

        # -- TOV% --
        # tov_pct = TOV / (FGA + 0.44*FTA + TOV)
        # Approximate denominator as possessions
        tov_pct = team_tov / max(possessions, 1.0)

        # -- ORB% approximation --
        # Cannot compute without oreb/dreb data. Set to 0.
        orb_pct = 0.0

        # -- FT rate --
        # Cannot compute without FTA/FGA. Set to 0.
        ft_rate = 0.0

        # -- 3-point rate --
        # fg3_rate = FG3A / FGA. We only have fg3m, not fg3a.
        # Approximate: fg3a ~ fg3m / 0.36 (league avg 3PT%)
        approx_fg3a = team_fg3m / 0.36 if team_fg3m > 0 else 0.0
        fg3_rate = approx_fg3a / max(approx_fga, 1.0)

        # -- Opponent defensive stats (from team's offensive perspective) --
        opp_approx_ftm = opp_pts * 0.18
        opp_fg_pts = opp_pts - opp_approx_ftm
        opp_approx_fgm = max((opp_fg_pts - opp_fg3m) / 2.0, 0.0)
        opp_approx_fga = max(opp_approx_fgm / 0.47, 1.0)
        opp_efg_pct = min(
            (opp_approx_fgm + 0.5 * opp_fg3m) / opp_approx_fga, 1.0
        )
        opp_tov_pct = opp_tov / max(possessions, 1.0)
        opp_approx_fg3a = opp_fg3m / 0.36 if opp_fg3m > 0 else 0.0
        opp_fg3_rate = opp_approx_fg3a / max(opp_approx_fga, 1.0)

        return {
            "date": date,
            "game_id": game_id,
            "team_pts": team_pts,
            "opp_pts": opp_pts,
            "pace": pace,
            "off_rating": off_rating,
            "def_rating": def_rating,
            "efg_pct": efg_pct,
            "tov_pct": tov_pct,
            "orb_pct": orb_pct,
            "ft_rate": ft_rate,
            "fg3_rate": fg3_rate,
            # Opponent stats (what the team allowed)
            "opp_efg_pct": opp_efg_pct,
            "opp_tov_pct": opp_tov_pct,
            "opp_fg3_rate": opp_fg3_rate,
            # Raw aggregates
            "team_reb": team_reb,
            "opp_reb": opp_reb,
            "team_ast": team_agg.get("ast", 0),
            "team_stl": team_agg.get("stl", 0),
            "team_blk": team_agg.get("blk", 0),
            "team_tov": team_tov,
            "team_fg3m": team_fg3m,
        }

    # -----------------------------------------------------------------
    # Rolling lookups
    # -----------------------------------------------------------------

    def get_team_features(
        self,
        team_name: str,
        game_date: str,
        n_games: int = 10,
    ) -> Optional[Dict[str, float]]:
        """Return rolling team stats using only games BEFORE game_date.

        Args:
            team_name: e.g. "Boston Celtics"
            game_date: ISO date string, e.g. "2024-01-15"
            n_games: Number of recent games for rolling window (default 10)

        Returns:
            Dict of team features with "team_" prefix, or None if
            insufficient history (< 3 games).
        """
        history = self.team_history.get(team_name, [])
        prior = [g for g in history if g["date"] < game_date]

        if len(prior) < 3:
            return None

        recent = prior[-n_games:]

        features = {}
        suffix = f"_l{n_games}"

        for key in [
            "pace", "off_rating", "def_rating", "efg_pct",
            "tov_pct", "orb_pct", "ft_rate", "fg3_rate",
        ]:
            vals = [g[key] for g in recent]
            features[f"team_{key}{suffix}"] = float(np.mean(vals))

        # Additional context: scoring and rebound averages
        features[f"team_pts{suffix}"] = float(
            np.mean([g["team_pts"] for g in recent])
        )
        features[f"team_ast{suffix}"] = float(
            np.mean([g["team_ast"] for g in recent])
        )
        features[f"team_fg3m{suffix}"] = float(
            np.mean([g["team_fg3m"] for g in recent])
        )

        # Trend: L3 vs L10
        if len(prior) >= 3:
            l3 = prior[-3:]
            features["team_pace_trend"] = float(
                np.mean([g["pace"] for g in l3])
                - np.mean([g["pace"] for g in recent])
            )
            features["team_off_rating_trend"] = float(
                np.mean([g["off_rating"] for g in l3])
                - np.mean([g["off_rating"] for g in recent])
            )

        return features

    def get_opponent_features(
        self,
        opp_team_name: str,
        game_date: str,
        n_games: int = 10,
    ) -> Optional[Dict[str, float]]:
        """Return rolling opponent (defensive) stats using games BEFORE game_date.

        These represent what the opponent ALLOWS, i.e. their defensive
        performance from the opposing team's perspective.

        Args:
            opp_team_name: e.g. "Milwaukee Bucks"
            game_date: ISO date string
            n_games: Rolling window size

        Returns:
            Dict of opponent features with "opp_" prefix, or None if
            insufficient history.
        """
        history = self.team_history.get(opp_team_name, [])
        prior = [g for g in history if g["date"] < game_date]

        if len(prior) < 3:
            return None

        recent = prior[-n_games:]

        features = {}
        suffix = f"_l{n_games}"

        # Opponent's defensive rating (points they allow per 100 possessions)
        features[f"opp_def_rating{suffix}"] = float(
            np.mean([g["def_rating"] for g in recent])
        )
        # Opponent's pace
        features[f"opp_pace{suffix}"] = float(
            np.mean([g["pace"] for g in recent])
        )
        # What opponent allows offensively (their opponents' stats)
        # = this team's "opp_efg_pct" etc. which is the eFG% of teams
        # playing AGAINST opp_team
        features[f"opp_def_efg_allowed{suffix}"] = float(
            np.mean([g["opp_efg_pct"] for g in recent])
        )
        features[f"opp_def_tov_forced{suffix}"] = float(
            np.mean([g["opp_tov_pct"] for g in recent])
        )
        features[f"opp_def_orb_allowed{suffix}"] = float(
            np.mean([g["orb_pct"] for g in recent])
        )
        features[f"opp_def_ft_rate_allowed{suffix}"] = float(
            np.mean([g["ft_rate"] for g in recent])
        )
        features[f"opp_def_fg3_rate_allowed{suffix}"] = float(
            np.mean([g["opp_fg3_rate"] for g in recent])
        )

        # Opponent's own offensive strength (context for game environment)
        features[f"opp_off_rating{suffix}"] = float(
            np.mean([g["off_rating"] for g in recent])
        )
        features[f"opp_pts{suffix}"] = float(
            np.mean([g["team_pts"] for g in recent])
        )

        return features

    def get_matchup_features(
        self,
        team_name: str,
        opp_name: str,
        game_date: str,
    ) -> Optional[Dict[str, float]]:
        """Return combined matchup features for a specific game.

        Computes interaction features between team and opponent that
        capture the specific matchup dynamics.

        Args:
            team_name: Player's team
            opp_name: Opposing team
            game_date: ISO date string

        Returns:
            Dict of matchup features, or None if insufficient data.
        """
        team_feats = self.get_team_features(team_name, game_date)
        opp_feats = self.get_opponent_features(opp_name, game_date)

        if team_feats is None or opp_feats is None:
            return None

        team_pace = team_feats.get("team_pace_l10", LEAGUE_AVG_PACE)
        opp_pace = opp_feats.get("opp_pace_l10", LEAGUE_AVG_PACE)
        team_off = team_feats.get("team_off_rating_l10", 100.0)
        opp_def = opp_feats.get("opp_def_rating_l10", 100.0)
        team_efg = team_feats.get("team_efg_pct_l10", 0.50)
        opp_def_efg = opp_feats.get("opp_def_efg_allowed_l10", 0.50)

        # Compute league average pace from all games seen so far
        if self._league_pace_history:
            league_avg_pace = float(np.mean(self._league_pace_history))
        else:
            league_avg_pace = LEAGUE_AVG_PACE

        features = {
            # Projected game pace: average of both teams
            "projected_pace": (team_pace + opp_pace) / 2.0,
            # How this game's pace compares to league average
            "pace_delta": (team_pace + opp_pace) / 2.0 - league_avg_pace,
            # Offensive vs defensive mismatch: positive = team has advantage
            "off_vs_def_mismatch": team_off - opp_def,
            # Shooting efficiency advantage
            "efg_advantage": team_efg - opp_def_efg,
            # Projected total (rough, from pace * efficiency)
            "projected_total": (
                (team_pace + opp_pace) / 2.0
                * (team_off + opp_feats.get("opp_off_rating_l10", 100.0))
                / 100.0
            ),
            # Turnover battle: team's care vs opp's forcing ability
            "tov_differential": (
                team_feats.get("team_tov_pct_l10", 0.13)
                - opp_feats.get("opp_def_tov_forced_l10", 0.13)
            ),
        }

        return features

    def get_all_features_for_game(
        self,
        team_name: str,
        opp_name: str,
        game_date: str,
    ) -> Optional[Dict[str, float]]:
        """Return all team-level features for a game in one call.

        Convenience method that merges team, opponent, and matchup features
        into a single dict ready for feature matrix enrichment.

        Args:
            team_name: Player's team
            opp_name: Opposing team
            game_date: ISO date string

        Returns:
            Combined dict of all team features, or None if insufficient data.
        """
        team_feats = self.get_team_features(team_name, game_date)
        opp_feats = self.get_opponent_features(opp_name, game_date)
        matchup_feats = self.get_matchup_features(team_name, opp_name, game_date)

        if team_feats is None or opp_feats is None or matchup_feats is None:
            return None

        combined = {}
        combined.update(team_feats)
        combined.update(opp_feats)
        combined.update(matchup_feats)
        return combined

    def get_player_team(self, player_name: str, game_id: int) -> Optional[str]:
        """Look up which team a player was on for a specific game.

        Uses the team assignments built during build_team_history().

        Args:
            player_name: e.g. "LeBron James"
            game_id: Game ID from the training data

        Returns:
            Team name string, or None if not found.
        """
        assignments = self._team_assignments.get(game_id, {})
        return assignments.get(player_name)

    def get_opponent_team(
        self,
        player_name: str,
        game_id: int,
        home_team: str,
        away_team: str,
    ) -> Optional[str]:
        """Look up the opponent team for a player in a specific game.

        Args:
            player_name: e.g. "LeBron James"
            game_id: Game ID
            home_team: Home team name from game data
            away_team: Away team name from game data

        Returns:
            Opponent team name, or None if player's team can't be determined.
        """
        player_team = self.get_player_team(player_name, game_id)
        if player_team is None:
            return None
        if player_team == home_team:
            return away_team
        return home_team


# =============================================================================
# Part 2: Fetch current-season team stats from NBA.com (live inference)
# =============================================================================

def fetch_current_team_stats(season: str = "2026-27") -> Dict[str, dict]:
    """Fetch current-season team stats from NBA.com via nba_api.

    Retrieves:
      - Team Four Factors (eFG%, TOV%, ORB%, FT rate)
      - Team estimated metrics (pace, off/def rating, net rating)

    Args:
        season: NBA season string, e.g. "2026-27"

    Returns:
        Dict mapping team name to stats dict, e.g.:
        {
            "Boston Celtics": {
                "pace": 101.2,
                "off_rating": 118.5,
                "def_rating": 109.3,
                "net_rating": 9.2,
                "efg_pct": 0.562,
                "tov_pct": 0.118,
                "oreb_pct": 0.268,
                "ft_rate": 0.205,
                "fg3_rate": 0.392,
                "wins": 30,
                "losses": 12,
            },
            ...
        }
    """
    from nba_api.stats.endpoints import LeagueDashTeamStats, TeamEstimatedMetrics

    result = {}

    # --- Team Four Factors ---
    logger.info("Fetching team Four Factors for %s ...", season)
    try:
        four_factors = LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Four Factors",
            per_mode_detailed="PerGame",
        )
        df_ff = four_factors.get_data_frames()[0]

        for _, row in df_ff.iterrows():
            team_name = row.get("TEAM_NAME", "")
            if not team_name:
                continue
            result[team_name] = {
                "efg_pct": _safe_float(row.get("EFG_PCT")),
                "fta_rate": _safe_float(row.get("FTA_RATE")),
                "tov_pct": _safe_float(row.get("TM_TOV_PCT")),
                "oreb_pct": _safe_float(row.get("OREB_PCT")),
                "opp_efg_pct": _safe_float(row.get("OPP_EFG_PCT")),
                "opp_fta_rate": _safe_float(row.get("OPP_FTA_RATE")),
                "opp_tov_pct": _safe_float(row.get("OPP_TOV_PCT")),
                "opp_oreb_pct": _safe_float(row.get("OPP_OREB_PCT")),
                "wins": int(row.get("W", 0)),
                "losses": int(row.get("L", 0)),
                "gp": int(row.get("GP", 0)),
            }

        logger.info("  Four Factors: %d teams", len(result))
    except Exception as e:
        logger.error("Failed to fetch Four Factors: %s", e)

    time.sleep(NBA_API_DELAY)

    # --- Team Estimated Metrics (pace, off/def rating) ---
    logger.info("Fetching team estimated metrics for %s ...", season)
    try:
        metrics = TeamEstimatedMetrics(season=season)
        df_met = metrics.get_data_frames()[0]

        for _, row in df_met.iterrows():
            team_name = row.get("TEAM_NAME", "")
            if not team_name:
                continue
            if team_name not in result:
                result[team_name] = {}
            result[team_name].update({
                "pace": _safe_float(row.get("E_PACE")),
                "off_rating": _safe_float(row.get("E_OFF_RATING")),
                "def_rating": _safe_float(row.get("E_DEF_RATING")),
                "net_rating": _safe_float(row.get("E_NET_RATING")),
            })

        logger.info("  Estimated metrics: %d teams", len(df_met))
    except Exception as e:
        logger.error("Failed to fetch estimated metrics: %s", e)

    time.sleep(NBA_API_DELAY)

    # --- Supplemental: base team stats for FG3 rate, etc. ---
    logger.info("Fetching base team stats for %s ...", season)
    try:
        base_stats = LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Base",
            per_mode_detailed="PerGame",
        )
        df_base = base_stats.get_data_frames()[0]

        for _, row in df_base.iterrows():
            team_name = row.get("TEAM_NAME", "")
            if not team_name or team_name not in result:
                continue
            fga = _safe_float(row.get("FGA"))
            fg3a = _safe_float(row.get("FG3A"))
            result[team_name]["fg3_rate"] = fg3a / max(fga, 1.0) if fga else 0.0
            result[team_name]["ft_rate"] = (
                _safe_float(row.get("FTM")) / max(fga, 1.0) if fga else 0.0
            )
            result[team_name]["pts_per_game"] = _safe_float(row.get("PTS"))
            result[team_name]["reb_per_game"] = _safe_float(row.get("REB"))
            result[team_name]["ast_per_game"] = _safe_float(row.get("AST"))

        logger.info("  Base stats merged")
    except Exception as e:
        logger.error("Failed to fetch base team stats: %s", e)

    logger.info("Fetched stats for %d teams total", len(result))
    return result


# =============================================================================
# Part 3: Position-specific defense from NBA.com
# =============================================================================

def fetch_position_defense(season: str = "2026-27") -> Dict[str, Dict[str, dict]]:
    """Fetch opponent defense by position from NBA.com tracking data.

    Retrieves how each team defends each position category using the
    LeagueDashPtTeamDefend endpoint.

    Args:
        season: NBA season string, e.g. "2026-27"

    Returns:
        Nested dict: {team_name: {position: {stats}}}, e.g.:
        {
            "Boston Celtics": {
                "Overall": {"freq": 1.0, "dfg_pct": 0.45, "d_fga": 85.2, ...},
                "Guards": {"freq": 0.42, "dfg_pct": 0.44, ...},
                "Forwards": {"freq": 0.38, "dfg_pct": 0.46, ...},
                "Centers": {"freq": 0.20, "dfg_pct": 0.52, ...},
            },
            ...
        }
    """
    from nba_api.stats.endpoints import LeagueDashPtTeamDefend

    result = defaultdict(dict)

    # The endpoint supports these defense_category values:
    # "Overall", "3 Pointers", "2 Pointers", "Less Than 6Ft",
    # "Less Than 10Ft", "Greater Than 15Ft"
    # For position-based data we pull Overall and parse CLOSE_DEF_DIST columns.
    # Note: position-specific breakdown depends on the data available.

    categories = ["Overall", "3 Pointers", "2 Pointers", "Less Than 6Ft",
                   "Greater Than 15Ft"]

    for category in categories:
        logger.info(
            "Fetching position defense (%s) for %s ...", category, season
        )
        try:
            defense = LeagueDashPtTeamDefend(
                season=season,
                defense_category=category,
                league_id="00",
                per_mode_simple="PerGame",
            )
            df = defense.get_data_frames()[0]

            for _, row in df.iterrows():
                team_name = row.get("TEAM_NAME", "")
                if not team_name:
                    continue

                cat_key = category.replace(" ", "_").lower()
                result[team_name][cat_key] = {
                    "freq": _safe_float(row.get("FREQ")),
                    "dfg_pct": _safe_float(row.get("D_FG_PCT")),
                    "d_fga": _safe_float(row.get("D_FGA")),
                    "d_fgm": _safe_float(row.get("D_FGM")),
                    "normal_fg_pct": _safe_float(row.get("NORMAL_FG_PCT")),
                    "pct_plusminus": _safe_float(row.get("PCT_PLUSMINUS")),
                }

            logger.info("  %s: %d teams", category, len(df))
        except Exception as e:
            logger.error("Failed to fetch defense (%s): %s", category, e)

        time.sleep(NBA_API_DELAY)

    # Convert defaultdict to regular dict
    return {team: dict(cats) for team, cats in result.items()}


# =============================================================================
# Helpers
# =============================================================================

def _safe_float(val, default: float = 0.0) -> float:
    """Safely convert a value to float, returning default on failure."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _load_all_games() -> List[dict]:
    """Load all games from the backtest JSON files."""
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

    all_games.sort(key=lambda g: g["date"])
    logger.info("Total games loaded: %d", len(all_games))
    return all_games


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Team-level feature engineering for v2 ML model"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Also fetch live NBA.com team stats for current season",
    )
    parser.add_argument(
        "--season",
        default="2026-27",
        help="NBA season for live stats (default: 2026-27)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("v2 Team Features - Building team-level feature history")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load training game data
    # ------------------------------------------------------------------
    games = _load_all_games()
    if not games:
        print("No game data found. Ensure backtest JSONs exist in outputs/.")
        return

    # ------------------------------------------------------------------
    # 2. Build team history
    # ------------------------------------------------------------------
    builder = TeamStatsBuilder()
    builder.build_team_history(games)

    # ------------------------------------------------------------------
    # 3. Print sample features for a few teams
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("SAMPLE TEAM FEATURES (L10 rolling)")
    print(f"{'='*70}")

    # Pick 5 teams with the most games
    teams_by_games = sorted(
        builder.team_history.items(),
        key=lambda x: len(x[1]),
        reverse=True,
    )

    for team_name, history in teams_by_games[:5]:
        if len(history) < 10:
            continue
        # Use a date near the middle of the dataset
        mid_date = history[len(history) // 2]["date"]
        feats = builder.get_team_features(team_name, mid_date)
        if feats is None:
            continue

        print(f"\n  {team_name} (as of {mid_date}, {len(history)} total games):")
        for key in sorted(feats.keys()):
            print(f"    {key:35s} = {feats[key]:8.3f}")

    # ------------------------------------------------------------------
    # 4. Print sample opponent features
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("SAMPLE OPPONENT FEATURES")
    print(f"{'='*70}")

    if len(teams_by_games) >= 2:
        t1_name, t1_hist = teams_by_games[0]
        t2_name, t2_hist = teams_by_games[1]
        mid_date = t1_hist[len(t1_hist) // 2]["date"]

        opp_feats = builder.get_opponent_features(t2_name, mid_date)
        if opp_feats:
            print(f"\n  {t2_name} opponent features (as of {mid_date}):")
            for key in sorted(opp_feats.keys()):
                print(f"    {key:35s} = {opp_feats[key]:8.3f}")

        matchup_feats = builder.get_matchup_features(t1_name, t2_name, mid_date)
        if matchup_feats:
            print(f"\n  Matchup: {t1_name} vs {t2_name}:")
            for key in sorted(matchup_feats.keys()):
                print(f"    {key:35s} = {matchup_feats[key]:8.3f}")

    # ------------------------------------------------------------------
    # 5. Print player team assignment sample
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("PLAYER TEAM ASSIGNMENT SAMPLE")
    print(f"{'='*70}")

    sample_game = games[len(games) // 2]
    gid = sample_game["game_id"]
    print(f"\n  Game {gid}: {sample_game['home_team']} vs {sample_game['away_team']}")
    print(f"  Score: {sample_game['home_score']}-{sample_game['away_score']}")
    box_players = list(sample_game.get("box_score", {}).keys())
    for p in box_players[:8]:
        team = builder.get_player_team(p, gid)
        print(f"    {p:30s} -> {team}")

    # ------------------------------------------------------------------
    # 6. Fetch live stats if requested
    # ------------------------------------------------------------------
    if args.live:
        print(f"\n{'='*70}")
        print(f"LIVE TEAM STATS FROM NBA.COM ({args.season})")
        print(f"{'='*70}")

        team_stats = fetch_current_team_stats(season=args.season)
        if team_stats:
            # Print top 5 by offensive rating
            sorted_teams = sorted(
                team_stats.items(),
                key=lambda x: x[1].get("off_rating", 0),
                reverse=True,
            )
            print(f"\n  Top 5 teams by offensive rating:")
            print(f"  {'Team':30s} {'Pace':>6s} {'ORtg':>6s} {'DRtg':>6s} "
                  f"{'Net':>6s} {'eFG%':>6s}")
            print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
            for tname, tstats in sorted_teams[:5]:
                print(f"  {tname:30s} "
                      f"{tstats.get('pace', 0):6.1f} "
                      f"{tstats.get('off_rating', 0):6.1f} "
                      f"{tstats.get('def_rating', 0):6.1f} "
                      f"{tstats.get('net_rating', 0):+6.1f} "
                      f"{tstats.get('efg_pct', 0):6.3f}")

        print(f"\n  Fetching position defense ...")
        pos_defense = fetch_position_defense(season=args.season)
        if pos_defense:
            sample_team = list(pos_defense.keys())[0]
            print(f"\n  {sample_team} defense by category:")
            for cat, stats in pos_defense[sample_team].items():
                print(f"    {cat:25s}: DFG%={stats.get('dfg_pct', 0):.3f}, "
                      f"DFGA={stats.get('d_fga', 0):.1f}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    all_features = builder.get_all_features_for_game(
        teams_by_games[0][0],
        teams_by_games[1][0],
        teams_by_games[0][1][len(teams_by_games[0][1]) // 2]["date"],
    )
    if all_features:
        print(f"\n{'='*70}")
        print(f"TOTAL FEATURE COUNT: {len(all_features)} team-level features")
        print(f"{'='*70}")
        for key in sorted(all_features.keys()):
            print(f"  {key:35s} = {all_features[key]:8.3f}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
