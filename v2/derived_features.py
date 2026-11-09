#!/usr/bin/env python3
"""
v2/derived_features.py - Derived player-level features for the line-anchor ML model.

Computes features that are derivable from existing box score data but are NOT
currently in v2/features.py. These supplement the existing 179 features with
~25-30 new signals covering:

  1. FTA rate features (free throw drawing tendencies)
  2. Home/away split features (with team inference heuristic)
  3. Scoring efficiency features (TS%, eFG%, scoring versatility)
  4. Injury cascade / usage redistribution features
  5. Opponent history features
  6. Streak / momentum features

All features use ONLY games BEFORE the prediction date (no lookahead).

Usage:
    python -m v2.derived_features   # Demo with sample players
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (same as features.py)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS = BASE_DIR / "outputs"

GAME_DATA_FILES = [
    OUTPUTS / "backtest_2023_24_data.json",
    OUTPUTS / "backtest_oos_data.json",
]

# ---------------------------------------------------------------------------
# Prop type -> box score stat key(s)  (mirrors features.py)
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


def get_stat_value(game: dict, prop_type: str) -> float:
    """Extract the stat value for a prop type from a game dict.

    Handles combined props (pts_rebs_asts) by summing component stats.

    Args:
        game: dict with keys like pts, reb, ast, fg3m, etc.
        prop_type: e.g. "points", "rebounds", "pts_rebs_asts"

    Returns:
        float: the stat value (sum of component stats for combined props)
    """
    keys = PROP_TO_STAT.get(prop_type)
    if keys is None:
        return 0.0
    return sum(game.get(k, 0) for k in keys)


# =============================================================================
# Team inference for home/away splits
# =============================================================================

def infer_player_team(player_name: str, game: dict, box_score: dict) -> Optional[str]:
    """Infer which team a player was on for a given game.

    Heuristic: partition all players in the box score into two groups by
    checking which subset's points sum to home_score vs away_score. The
    player's team is whichever group they belong to.

    If the heuristic fails (scores don't match), fall back to None.

    Args:
        player_name: the player to look up
        game: game dict with home_team, away_team, home_score, away_score
        box_score: {player_name: stats_dict} for this game

    Returns:
        team name string, or None if inference fails
    """
    if player_name not in box_score:
        return None

    home_team = game.get("home_team", "")
    away_team = game.get("away_team", "")
    home_score = game.get("home_score", 0)
    away_score = game.get("away_score", 0)

    if not home_team or not away_team:
        return None

    # Sort players by points descending for greedy assignment
    players_pts = []
    for name, stats in box_score.items():
        players_pts.append((name, stats.get("pts", 0)))
    players_pts.sort(key=lambda x: -x[1])

    # Greedy bin-packing: assign each player to the team whose remaining
    # score capacity is larger. This won't always work perfectly but is
    # a reasonable heuristic.
    remaining_home = home_score
    remaining_away = away_score
    team_assignment = {}

    for name, pts in players_pts:
        # Assign to whichever side has more points remaining
        if remaining_home >= remaining_away:
            team_assignment[name] = home_team
            remaining_home -= pts
        else:
            team_assignment[name] = away_team
            remaining_away -= pts

    return team_assignment.get(player_name)


def build_player_team_cache(games: list) -> Dict[str, Dict[str, str]]:
    """Build a cache of {game_id: {player_name: team_name}} for all games.

    Args:
        games: list of game dicts with box_score

    Returns:
        dict mapping game_id -> {player_name -> team_name}
    """
    cache = {}
    for game in games:
        game_id = game.get("game_id")
        if not game_id:
            continue
        box = game.get("box_score", {})
        if not box:
            continue

        assignments = {}
        for player_name in box:
            team = infer_player_team(player_name, game, box)
            if team:
                assignments[player_name] = team

        cache[game_id] = assignments

    return cache


# =============================================================================
# DerivedFeatures class
# =============================================================================

class DerivedFeatures:
    """Computes derived player-level features from box score history.

    These features supplement the base features in v2/features.py with
    signals that are derivable from existing data but not currently computed.
    """

    def __init__(self, player_team_cache: Optional[Dict[str, Dict[str, str]]] = None):
        """
        Args:
            player_team_cache: optional pre-built {game_id: {player: team}} mapping.
                If None, home/away features will use a simpler fallback.
        """
        self.player_team_cache = player_team_cache or {}

    def compute_all(
        self,
        prior_games: List[dict],
        prop_type: str,
        game_context: dict,
    ) -> dict:
        """Compute all derived features for a single prediction.

        Args:
            prior_games: list of game dicts, most recent first,
                         ONLY games before the target date.
            prop_type: e.g. "points", "rebounds", "pts_rebs_asts"
            game_context: dict with keys:
                - line (float): the DK prop line
                - is_home (bool or None): whether the player is home for target game
                - opponent (str): opponent team name for target game
                - player_name (str): player name (for team inference)

        Returns:
            dict of feature_name -> float (~25-30 features)
        """
        if len(prior_games) < 3:
            return {}

        features = {}

        line = game_context.get("line", 0)
        is_home = game_context.get("is_home", None)
        opponent = game_context.get("opponent", "")
        player_name = game_context.get("player_name", "")

        # 1. FTA rate features
        features.update(self._fta_features(prior_games))

        # 2. Home/away split features
        features.update(self._home_away_features(
            prior_games, prop_type, is_home, player_name
        ))

        # 3. Scoring efficiency features
        features.update(self._efficiency_features(prior_games))

        # 4. Injury cascade / usage redistribution features
        features.update(self._injury_cascade_features(prior_games))

        # 5. Opponent history features
        features.update(self._opponent_features(prior_games, opponent, prop_type))

        # 6. Streak / momentum features
        features.update(self._streak_features(prior_games, line, prop_type))

        return features

    # -----------------------------------------------------------------
    # 1. FTA Rate Features
    # -----------------------------------------------------------------

    def _fta_features(self, prior_games: List[dict]) -> dict:
        """Free throw rate features for scoring prop context.

        These capture how much a player relies on free throws, which
        affects points variance and scoring floor.

        NOTE: The backtest box score data may not include fta/ftm/fga fields.
        When fta/ftm are missing, we estimate FT points from available data:
            estimated_ft_pts = pts - (fg3m * 3) - ((fgm - fg3m) * 2)
        If fgm is also missing, we can still estimate a rough FT share using
        the relationship: pts = 2*fgm - fg3m + ftm, so ftm ~ pts - 2*fgm + fg3m.
        When we can't compute fgm, we fall back to a cruder proxy.
        """
        l3 = prior_games[:3]
        l10 = prior_games[:min(10, len(prior_games))]

        # Check if detailed shooting stats are available
        has_fta = any(g.get("fta", 0) > 0 for g in l10)
        has_fga = any(g.get("fga", 0) > 0 for g in l10)

        if has_fta and has_fga:
            # Full shooting data available -- use exact stats
            total_fta_l10 = sum(g.get("fta", 0) for g in l10)
            total_fga_l10 = sum(g.get("fga", 0) for g in l10)
            total_ftm_l10 = sum(g.get("ftm", 0) for g in l10)
            total_pts_l10 = sum(g.get("pts", 0) for g in l10)
            total_fta_l3 = sum(g.get("fta", 0) for g in l3)

            avg_fta_l3 = total_fta_l3 / len(l3)
            avg_fta_l10 = total_fta_l10 / len(l10)

            return {
                "fta_rate_l10": total_fta_l10 / max(total_fga_l10, 1),
                "fta_scoring_share_l10": total_ftm_l10 / max(total_pts_l10, 1),
                "fta_trend": avg_fta_l3 - avg_fta_l10,
                "and_one_proxy_l10": (
                    sum(g.get("fta", 0) - 2 * (g.get("fta", 0) // 2) for g in l10)
                    / len(l10)
                ),
                "ft_pct_l10": total_ftm_l10 / max(total_fta_l10, 1),
            }
        else:
            # Estimate FT contribution from pts and fg3m.
            # We know: pts = 2pt_fgm*2 + fg3m*3 + ftm
            # Without fgm, we can estimate the 3pt contribution: fg3m * 3
            # and approximate: non_3pt_pts = pts - fg3m * 3
            # Those non-3pt points come from 2pt FGs and FTs (both worth ~2 and ~1).
            # Crude proxy: ft_share ~ (pts - fg3m * 3) fraction that is "odd" points
            # Better: pts_per_min trend captures usage changes even without FTA.
            total_pts_l10 = sum(g.get("pts", 0) for g in l10)
            total_fg3m_l10 = sum(g.get("fg3m", 0) for g in l10)
            total_pts_l3 = sum(g.get("pts", 0) for g in l3)
            total_fg3m_l3 = sum(g.get("fg3m", 0) for g in l3)

            # Estimated points from 3s
            pts_from_3_l10 = total_fg3m_l10 * 3
            pts_from_3_l3 = total_fg3m_l3 * 3

            # Non-3pt points (mix of 2pt FGs and FTs)
            non3_pts_l10 = max(total_pts_l10 - pts_from_3_l10, 0)
            non3_pts_l3 = max(total_pts_l3 - pts_from_3_l3, 0)

            # 3pt scoring share as inverse proxy -- players who score heavily
            # from 3 have lower FT share
            three_pt_share_l10 = pts_from_3_l10 / max(total_pts_l10, 1)

            # Estimated FT share: assume ~40% of non-3pt points are from FTs
            # (league average is roughly 20-25% of total pts from FTs)
            est_ft_share_l10 = (1.0 - three_pt_share_l10) * 0.35  # rough proxy

            # Trend: non-3pt scoring rate change (captures FT trend indirectly)
            avg_non3_l3 = non3_pts_l3 / len(l3)
            avg_non3_l10 = non3_pts_l10 / len(l10)

            return {
                "fta_rate_l10": 0.0,  # can't compute without fta/fga
                "fta_scoring_share_l10": est_ft_share_l10,
                "fta_trend": avg_non3_l3 - avg_non3_l10,  # non-3pt trend as proxy
                "and_one_proxy_l10": 0.0,  # can't compute without fta
                "ft_pct_l10": 0.0,  # can't compute without fta/ftm
            }

    # -----------------------------------------------------------------
    # 2. Home/Away Split Features
    # -----------------------------------------------------------------

    def _home_away_features(
        self,
        prior_games: List[dict],
        prop_type: str,
        is_home: Optional[bool],
        player_name: str,
    ) -> dict:
        """Home/away context-aware split features.

        Uses the player_team_cache to infer which team the player was on
        in each historical game, then splits stats by home vs away.
        """
        home_values = []
        away_values = []

        for g in prior_games:
            game_id = g.get("game_id")
            home_team = g.get("home_team", "")
            away_team = g.get("away_team", "")
            val = get_stat_value(g, prop_type)

            # Determine if the player was home in this historical game
            player_was_home = None

            # Try cache first
            if game_id and game_id in self.player_team_cache:
                team = self.player_team_cache[game_id].get(player_name)
                if team == home_team:
                    player_was_home = True
                elif team == away_team:
                    player_was_home = False

            if player_was_home is True:
                home_values.append(val)
            elif player_was_home is False:
                away_values.append(val)
            # If None, skip this game for home/away split

        all_values = [get_stat_value(g, prop_type) for g in prior_games]
        overall_avg = np.mean(all_values) if all_values else 0.0

        home_avg = np.mean(home_values) if home_values else overall_avg
        away_avg = np.mean(away_values) if away_values else overall_avg

        # Context average: if we know the player is home/away in the target game
        if is_home is True:
            context_avg = home_avg
        elif is_home is False:
            context_avg = away_avg
        else:
            context_avg = overall_avg

        return {
            # How much better (or worse) at home vs away
            "home_away_avg_diff": home_avg - away_avg,
            # Average in home games (L10 window comes from prior_games[:10])
            "home_avg": home_avg,
            # Average in away games
            "away_avg": away_avg,
            # Average in the relevant context for this game
            "context_avg": context_avg,
            # Delta: context average vs overall
            "context_vs_overall": context_avg - overall_avg,
        }

    # -----------------------------------------------------------------
    # 3. Scoring Efficiency Features
    # -----------------------------------------------------------------

    def _efficiency_features(self, prior_games: List[dict]) -> dict:
        """Advanced shooting efficiency features.

        NOTE: The backtest box score data typically only has pts, fg3m, min
        (no fga, fgm, fta, ftm, fg3a). When full shooting stats are missing,
        we compute proxy efficiency features from available data:
          - pts_per_min: scoring rate (available always)
          - fg3m_share_of_pts: what fraction of points come from 3s
          - scoring_versatility: estimated from pts vs fg3m decomposition
          - pts_variance_ratio: consistency of scoring output

        When full shooting stats ARE present, we compute TS%, eFG%, etc.
        """
        l10 = prior_games[:min(10, len(prior_games))]

        total_pts = sum(g.get("pts", 0) for g in l10)
        total_fg3m = sum(g.get("fg3m", 0) for g in l10)
        total_mins = sum(g.get("min", 0) for g in l10)

        # Check if full shooting stats are available
        total_fga = sum(g.get("fga", 0) for g in l10)
        total_fgm = sum(g.get("fgm", 0) for g in l10)
        total_fta = sum(g.get("fta", 0) for g in l10)
        total_ftm = sum(g.get("ftm", 0) for g in l10)
        total_fg3a = sum(g.get("fg3a", 0) for g in l10)
        has_shooting = total_fga > 0

        if has_shooting:
            # Full shooting data: compute standard efficiency metrics
            ts_denom = 2.0 * (total_fga + 0.44 * total_fta)
            ts_pct = total_pts / max(ts_denom, 1.0)
            efg_pct = (total_fgm + 0.5 * total_fg3m) / max(total_fga, 1)
            pts_per_fga = total_pts / max(total_fga, 1)
            fg3_share = total_fg3a / max(total_fga, 1)

            two_pt_fga = total_fga - total_fg3a
            two_pt_fgm = total_fgm - total_fg3m
            two_pt_pct = two_pt_fgm / max(two_pt_fga, 1)

            pts_from_2 = (total_fgm - total_fg3m) * 2
            pts_from_3 = total_fg3m * 3
            pts_from_ft = total_ftm
        else:
            # Estimate from available data (pts, fg3m, min)
            # Points from 3s is known exactly: fg3m * 3
            pts_from_3 = total_fg3m * 3
            # Remaining points are from 2pt FGs and FTs
            non_3_pts = max(total_pts - pts_from_3, 0)

            # Proxy TS%: pts per minute normalized (captures overall efficiency)
            ts_pct = total_pts / max(2.0 * total_mins * 0.45, 1.0)  # ~0.45 FGA/min typical
            # Proxy eFG%: can't compute without FGA, use pts-per-min as proxy
            efg_pct = total_pts / max(total_mins, 1.0) / 0.9  # normalize to ~0.5 range
            # Proxy pts_per_fga: use pts_per_min / estimated_fga_per_min
            pts_per_fga = total_pts / max(total_mins * 0.45, 1.0)
            # 3pt scoring share as proxy for 3pt attempt rate
            fg3_share = pts_from_3 / max(total_pts, 1)
            # 2pt proxy: non-3pt, non-FT scoring
            two_pt_pct = non_3_pts / max(total_pts, 1)

            # Estimate FT points: assume ~25% of non-3pt points are from FTs
            pts_from_ft = non_3_pts * 0.25
            pts_from_2 = non_3_pts - pts_from_ft

        # Scoring versatility: how evenly distributed are points across
        # 2pt FGs, 3pt FGs, and FTs. Higher = more versatile scorer.
        total_pts_check = pts_from_2 + pts_from_3 + pts_from_ft
        if total_pts_check > 0:
            shares = np.array([
                pts_from_2 / total_pts_check,
                pts_from_3 / total_pts_check,
                pts_from_ft / total_pts_check,
            ])
            # 1.0 - std(shares): 1.0 = perfectly balanced, lower = concentrated
            scoring_versatility = 1.0 - float(np.std(shares))
        else:
            scoring_versatility = 0.0

        return {
            "ts_pct_l10": ts_pct,
            "efg_pct_l10": efg_pct,
            "pts_per_fga_l10": pts_per_fga,
            "fg3_share_l10": fg3_share,
            "two_pt_pct_l10": two_pt_pct,
            "scoring_versatility_l10": scoring_versatility,
        }

    # -----------------------------------------------------------------
    # 4. Injury Cascade / Usage Redistribution Features
    # -----------------------------------------------------------------

    def _injury_cascade_features(self, prior_games: List[dict]) -> dict:
        """Features detecting elevated role from teammate absences.

        When a high-usage teammate is injured, remaining players get more
        minutes and shot attempts. We detect this from historical data by
        looking at minutes/usage spikes.

        NOTE: When fga is not available, we use pts/min as a usage proxy
        instead of fga/min. Both capture the same signal: increased
        offensive involvement when teammates are absent.
        """
        all_mins = [g.get("min", 0) for g in prior_games]
        all_pts = [g.get("pts", 0) for g in prior_games]
        all_fga = [g.get("fga", 0) for g in prior_games]

        l3_mins = all_mins[:3]
        l5_mins = all_mins[:min(5, len(all_mins))]
        l10_mins = all_mins[:min(10, len(all_mins))]

        avg_mins_season = np.mean(all_mins) if all_mins else 0.0
        avg_mins_l3 = np.mean(l3_mins)

        total_mins_l5 = sum(l5_mins)
        total_mins_l10 = sum(l10_mins)

        # Check if fga data is available
        has_fga = any(g.get("fga", 0) > 0 for g in prior_games[:10])

        if has_fga:
            # Use FGA per minute as usage proxy
            l5_fga = all_fga[:min(5, len(all_fga))]
            l10_fga = all_fga[:min(10, len(all_fga))]
            total_fga_l5 = sum(l5_fga)
            total_fga_l10 = sum(l10_fga)
            usage_proxy_l10 = total_fga_l10 / max(total_mins_l10, 1.0)
            usage_proxy_l5 = total_fga_l5 / max(total_mins_l5, 1.0)
        else:
            # Use PTS per minute as usage proxy (always available)
            l5_pts = all_pts[:min(5, len(all_pts))]
            l10_pts = all_pts[:min(10, len(all_pts))]
            total_pts_l5 = sum(l5_pts)
            total_pts_l10 = sum(l10_pts)
            usage_proxy_l10 = total_pts_l10 / max(total_mins_l10, 1.0)
            usage_proxy_l5 = total_pts_l5 / max(total_mins_l5, 1.0)

        # Percentage of L10 games where minutes exceeded season avg + 3
        threshold = avg_mins_season + 3.0
        high_mins_count = sum(1 for m in l10_mins if m > threshold)
        high_mins_pct = high_mins_count / max(len(l10_mins), 1)

        return {
            # Positive = recently playing more minutes than season average
            "mins_above_avg_l3": avg_mins_l3 - avg_mins_season,
            # Volume per minute as usage proxy (FGA/min if available, else PTS/min)
            "usage_proxy_l10": usage_proxy_l10,
            # Usage trending up = potentially filling in for injured teammate
            "usage_trend": usage_proxy_l5 - usage_proxy_l10,
            # How often in L10 did this player play elevated minutes
            "high_mins_pct_l10": high_mins_pct,
            # Recent volume per minute
            "fga_per_min_l5": usage_proxy_l5,
        }

    # -----------------------------------------------------------------
    # 5. Opponent History Features
    # -----------------------------------------------------------------

    def _opponent_features(
        self,
        prior_games: List[dict],
        opponent: str,
        prop_type: str,
    ) -> dict:
        """Features about how a player performs vs a specific opponent.

        Looks at historical matchups against the same team.
        """
        if not opponent:
            return {
                "vs_opp_avg": 0.0,
                "vs_opp_games": 0,
                "vs_opp_delta": 0.0,
            }

        vs_opp = [
            g for g in prior_games
            if opponent in (g.get("home_team", ""), g.get("away_team", ""))
        ]

        all_values = [get_stat_value(g, prop_type) for g in prior_games]
        season_avg = np.mean(all_values) if all_values else 0.0

        if len(vs_opp) >= 2:
            opp_values = [get_stat_value(g, prop_type) for g in vs_opp]
            vs_opp_avg = np.mean(opp_values)
            return {
                "vs_opp_avg": vs_opp_avg,
                "vs_opp_games": len(vs_opp),
                "vs_opp_delta": vs_opp_avg - season_avg,
            }
        else:
            return {
                "vs_opp_avg": 0.0,
                "vs_opp_games": len(vs_opp),
                "vs_opp_delta": 0.0,
            }

    # -----------------------------------------------------------------
    # 6. Streak / Momentum Features
    # -----------------------------------------------------------------

    def _streak_features(
        self,
        prior_games: List[dict],
        line: float,
        prop_type: str,
    ) -> dict:
        """Streak and momentum features.

        Note: features.py already has over_streak/under_streak. These
        features add z-score-based hot/cold and sustained trend detection.
        """
        values = [get_stat_value(g, prop_type) for g in prior_games[:10]]
        if not values:
            return {
                "consecutive_overs": 0,
                "consecutive_unders": 0,
                "hot_cold_l3_vs_l10": 0.0,
                "improving": 0,
                "declining": 0,
            }

        # Consecutive overs/unders from most recent game
        consecutive_overs = 0
        for v in values:
            if v > line:
                consecutive_overs += 1
            else:
                break

        consecutive_unders = 0
        for v in values:
            if v <= line:
                consecutive_unders += 1
            else:
                break

        # Hot/cold: L3 performance relative to L10 as z-score
        l3 = values[:3]
        l5 = values[:min(5, len(values))]
        l10 = values[:min(10, len(values))]

        avg_l3 = np.mean(l3)
        avg_l5 = np.mean(l5)
        avg_l10 = np.mean(l10)
        std_l10 = np.std(l10) if len(l10) >= 3 else 1.0

        hot_cold = (avg_l3 - avg_l10) / max(std_l10, 0.1)

        # Sustained trend detection
        improving = 1 if (avg_l3 > avg_l5 > avg_l10 and len(values) >= 5) else 0
        declining = 1 if (avg_l3 < avg_l5 < avg_l10 and len(values) >= 5) else 0

        return {
            "consecutive_overs": consecutive_overs,
            "consecutive_unders": consecutive_unders,
            # Z-score: how hot/cold is the player's recent stretch
            "hot_cold_l3_vs_l10": hot_cold,
            # Binary flags for sustained trend direction
            "improving": improving,
            "declining": declining,
        }


# =============================================================================
# CLI: demo with sample players
# =============================================================================

def main():
    print("=" * 70)
    print("v2 Derived Features - Demo")
    print("=" * 70)

    # Load game data (same pattern as features.py)
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

    if not all_games:
        print("No game data found. Ensure outputs/backtest_*_data.json files exist.")
        return

    # Build player history (same pattern as FeatureBuilder.build_player_history)
    player_history = defaultdict(list)
    for game in all_games:
        date = game["date"]
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        game_id = game.get("game_id")
        home_score = game.get("home_score", 0)
        away_score = game.get("away_score", 0)
        box = game.get("box_score", {})

        for player_name, stats in box.items():
            entry = {
                "date": date,
                "game_id": game_id,
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score,
            }
            for k in ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover", "min"]:
                entry[k] = stats.get(k, 0)
            entry["fga"] = stats.get("fga", 0)
            entry["fgm"] = stats.get("fgm", 0)
            entry["fta"] = stats.get("fta", 0)
            entry["ftm"] = stats.get("ftm", 0)
            entry["fg3a"] = stats.get("fg3a", 0)

            player_history[player_name].append(entry)

    for player in player_history:
        player_history[player].sort(key=lambda g: g["date"])

    logger.info(
        "Built history for %d players (avg %.1f games/player)",
        len(player_history),
        np.mean([len(v) for v in player_history.values()]),
    )

    # Build team cache for home/away inference
    logger.info("Building player-team cache for home/away inference...")
    team_cache = build_player_team_cache(all_games)
    logger.info("  Cached %d games", len(team_cache))

    # Pick 3 sample players with most games
    top_players = sorted(
        player_history.keys(),
        key=lambda p: len(player_history[p]),
        reverse=True,
    )[:3]

    df = DerivedFeatures(player_team_cache=team_cache)

    for player_name in top_players:
        history = player_history[player_name]
        if len(history) < 15:
            continue

        # Use a game near the end of history as the "target" game
        target_idx = len(history) - 1
        target_game = history[target_idx]
        prior_games = list(reversed(history[:target_idx]))  # most recent first

        # Determine opponent from the target game using team cache
        target_game_id = target_game.get("game_id")
        home_team = target_game.get("home_team", "")
        away_team = target_game.get("away_team", "")
        player_team_in_game = team_cache.get(target_game_id, {}).get(player_name)

        if player_team_in_game == home_team:
            opponent = away_team
            is_home = True
        elif player_team_in_game == away_team:
            opponent = home_team
            is_home = False
        else:
            # Fallback: assume away_team is opponent
            opponent = away_team
            is_home = True

        game_context = {
            "line": get_stat_value(target_game, "points") - 1.5,  # fake line near actual
            "is_home": is_home,
            "opponent": opponent,
            "player_name": player_name,
        }

        print(f"\n{'='*70}")
        print(f"Player: {player_name} ({len(history)} total games)")
        print(f"Target game: {target_game['date']} "
              f"({target_game.get('home_team', '?')} vs {target_game.get('away_team', '?')})")
        print(f"Actual points: {target_game.get('pts', '?')}")
        print(f"Fake line: {game_context['line']:.1f}")
        print(f"Prior games used: {len(prior_games)}")
        print(f"{'='*70}")

        for prop_type in ["points", "rebounds", "pts_rebs_asts"]:
            features = df.compute_all(prior_games, prop_type, game_context)
            print(f"\n  Prop: {prop_type} ({len(features)} features)")
            print(f"  {'-'*50}")
            for name, val in sorted(features.items()):
                if isinstance(val, float):
                    print(f"    {name:30s} = {val:>10.4f}")
                else:
                    print(f"    {name:30s} = {val!s:>10}")


if __name__ == "__main__":
    main()
