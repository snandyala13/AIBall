#!/usr/bin/env python3
"""
v2/fetch_tracking.py - Fetch historical NBA tracking + PBP stats for ML training.

Fetches season-level tracking stats from NBA.com (via nba_api) and play-by-play
derived stats from PBPStats for both training seasons (2023-24 and 2025-26).

These stats (drives, catch-shoot, possessions, hustle, etc.) enrich the box-score-
only feature set in our 122K backtest legs with ~30 tracking + ~15 PBP features.

Usage:
    python -m v2.fetch_tracking              # Fetch both seasons
    python -m v2.fetch_tracking --season 2023-24  # Fetch one season
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS = BASE_DIR / "outputs"

SEASONS = ["2023-24", "2025-26"]

# ============================================================================
# NBA.com Tracking Stats (via nba_api)
# ============================================================================

def _safe_float(val) -> float:
    """Convert a value to float, returning 0.0 on failure."""
    try:
        if val is None:
            return 0.0
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def fetch_tracking_for_season(season_str: str) -> Dict[str, Dict]:
    """Fetch all tracking categories for a season from NBA.com.

    Collects: Drives, CatchShoot, PullUpShot, Passing, Rebounding,
    Possessions, Hustle stats, and Estimated Advanced Metrics.

    Args:
        season_str: NBA season string, e.g. '2023-24'

    Returns:
        dict mapping player name -> {stat_key: value, ...}
    """
    from nba_api.stats.endpoints import (
        leaguedashptstats,
        leaguehustlestatsplayer,
        playerestimatedmetrics,
    )

    players: Dict[str, Dict] = {}

    def _merge(name: str, stats: Dict):
        if name not in players:
            players[name] = {}
        players[name].update(stats)

    # ------------------------------------------------------------------
    # 1. Drives
    # ------------------------------------------------------------------
    logger.info("Fetching Drives for %s ...", season_str)
    time.sleep(1)
    try:
        resp = leaguedashptstats.LeagueDashPtStats(
            season=season_str,
            pt_measure_type="Drives",
            per_mode_simple="PerGame",
            player_or_team="Player",
        )
        df = resp.get_data_frames()[0]
        logger.info("  -> %d players", len(df))
        for _, row in df.iterrows():
            name = row.get("PLAYER_NAME", "")
            if not name:
                continue
            _merge(name, {
                "drives": _safe_float(row.get("DRIVES")),
                "drive_fgm": _safe_float(row.get("DRIVE_FGM")),
                "drive_fga": _safe_float(row.get("DRIVE_FGA")),
                "drive_fg_pct": _safe_float(row.get("DRIVE_FG_PCT")),
                "drive_ftm": _safe_float(row.get("DRIVE_FTM")),
                "drive_fta": _safe_float(row.get("DRIVE_FTA")),
                "drive_pts": _safe_float(row.get("DRIVE_PTS")),
                "drive_ast": _safe_float(row.get("DRIVE_AST")),
                "drive_tov": _safe_float(row.get("DRIVE_TOV")),
                "drive_pf": _safe_float(row.get("DRIVE_PF")),
            })
    except Exception as e:
        logger.error("  Drives failed: %s", e)

    # ------------------------------------------------------------------
    # 2. Catch & Shoot
    # ------------------------------------------------------------------
    logger.info("Fetching CatchShoot for %s ...", season_str)
    time.sleep(1)
    try:
        resp = leaguedashptstats.LeagueDashPtStats(
            season=season_str,
            pt_measure_type="CatchShoot",
            per_mode_simple="PerGame",
            player_or_team="Player",
        )
        df = resp.get_data_frames()[0]
        logger.info("  -> %d players", len(df))
        for _, row in df.iterrows():
            name = row.get("PLAYER_NAME", "")
            if not name:
                continue
            _merge(name, {
                "catch_shoot_fgm": _safe_float(row.get("CATCH_SHOOT_FGM")),
                "catch_shoot_fga": _safe_float(row.get("CATCH_SHOOT_FGA")),
                "catch_shoot_fg_pct": _safe_float(row.get("CATCH_SHOOT_FG_PCT")),
                "catch_shoot_fg3m": _safe_float(row.get("CATCH_SHOOT_FG3M")),
                "catch_shoot_fg3a": _safe_float(row.get("CATCH_SHOOT_FG3A")),
                "catch_shoot_fg3_pct": _safe_float(row.get("CATCH_SHOOT_FG3_PCT")),
                "catch_shoot_efg_pct": _safe_float(row.get("CATCH_SHOOT_EFG_PCT")),
            })
    except Exception as e:
        logger.error("  CatchShoot failed: %s", e)

    # ------------------------------------------------------------------
    # 3. Pull-Up Shooting
    # ------------------------------------------------------------------
    logger.info("Fetching PullUpShot for %s ...", season_str)
    time.sleep(1)
    try:
        resp = leaguedashptstats.LeagueDashPtStats(
            season=season_str,
            pt_measure_type="PullUpShot",
            per_mode_simple="PerGame",
            player_or_team="Player",
        )
        df = resp.get_data_frames()[0]
        logger.info("  -> %d players", len(df))
        for _, row in df.iterrows():
            name = row.get("PLAYER_NAME", "")
            if not name:
                continue
            _merge(name, {
                "pull_up_fgm": _safe_float(row.get("PULL_UP_FGM")),
                "pull_up_fga": _safe_float(row.get("PULL_UP_FGA")),
                "pull_up_fg_pct": _safe_float(row.get("PULL_UP_FG_PCT")),
                "pull_up_fg3m": _safe_float(row.get("PULL_UP_FG3M")),
                "pull_up_fg3a": _safe_float(row.get("PULL_UP_FG3A")),
                "pull_up_fg3_pct": _safe_float(row.get("PULL_UP_FG3_PCT")),
                "pull_up_efg_pct": _safe_float(row.get("PULL_UP_EFG_PCT")),
            })
    except Exception as e:
        logger.error("  PullUpShot failed: %s", e)

    # ------------------------------------------------------------------
    # 4. Passing
    # ------------------------------------------------------------------
    logger.info("Fetching Passing for %s ...", season_str)
    time.sleep(1)
    try:
        resp = leaguedashptstats.LeagueDashPtStats(
            season=season_str,
            pt_measure_type="Passing",
            per_mode_simple="PerGame",
            player_or_team="Player",
        )
        df = resp.get_data_frames()[0]
        logger.info("  -> %d players", len(df))
        for _, row in df.iterrows():
            name = row.get("PLAYER_NAME", "")
            if not name:
                continue
            _merge(name, {
                "passes_made": _safe_float(row.get("PASSES_MADE")),
                "passes_received": _safe_float(row.get("PASSES_RECEIVED")),
                "potential_ast": _safe_float(row.get("POTENTIAL_AST")),
                "ast_points_created": _safe_float(row.get("AST_POINTS_CREATED")),
                "ast_adj": _safe_float(row.get("AST_ADJ")),
                "ast_to_pass_pct": _safe_float(row.get("AST_TO_PASS_PCT")),
                "secondary_ast": _safe_float(row.get("SECONDARY_AST")),
                "ft_ast": _safe_float(row.get("FT_AST")),
            })
    except Exception as e:
        logger.error("  Passing failed: %s", e)

    # ------------------------------------------------------------------
    # 5. Rebounding
    # ------------------------------------------------------------------
    logger.info("Fetching Rebounding for %s ...", season_str)
    time.sleep(1)
    try:
        resp = leaguedashptstats.LeagueDashPtStats(
            season=season_str,
            pt_measure_type="Rebounding",
            per_mode_simple="PerGame",
            player_or_team="Player",
        )
        df = resp.get_data_frames()[0]
        logger.info("  -> %d players", len(df))
        for _, row in df.iterrows():
            name = row.get("PLAYER_NAME", "")
            if not name:
                continue
            _merge(name, {
                "oreb_chances": _safe_float(row.get("OREB_CHANCES")),
                "oreb_chance_pct": _safe_float(row.get("OREB_CHANCE_PCT")),
                "oreb_contest": _safe_float(row.get("OREB_CONTEST")),
                "oreb_uncontest": _safe_float(row.get("OREB_UNCONTEST")),
                "dreb_chances": _safe_float(row.get("DREB_CHANCES")),
                "dreb_chance_pct": _safe_float(row.get("DREB_CHANCE_PCT")),
                "dreb_contest": _safe_float(row.get("DREB_CONTEST")),
                "dreb_uncontest": _safe_float(row.get("DREB_UNCONTEST")),
                "reb_chances": _safe_float(row.get("REB_CHANCES")),
                "reb_chance_pct": _safe_float(row.get("REB_CHANCE_PCT")),
                "avg_oreb_dist": _safe_float(row.get("AVG_OREB_DIST")),
                "avg_dreb_dist": _safe_float(row.get("AVG_DREB_DIST")),
            })
    except Exception as e:
        logger.error("  Rebounding failed: %s", e)

    # ------------------------------------------------------------------
    # 6. Possessions / Touches
    # ------------------------------------------------------------------
    logger.info("Fetching Possessions for %s ...", season_str)
    time.sleep(1)
    try:
        resp = leaguedashptstats.LeagueDashPtStats(
            season=season_str,
            pt_measure_type="Possessions",
            per_mode_simple="PerGame",
            player_or_team="Player",
        )
        df = resp.get_data_frames()[0]
        logger.info("  -> %d players", len(df))
        for _, row in df.iterrows():
            name = row.get("PLAYER_NAME", "")
            if not name:
                continue
            _merge(name, {
                "touches": _safe_float(row.get("TOUCHES")),
                "front_ct_touches": _safe_float(row.get("FRONT_CT_TOUCHES")),
                "time_of_poss": _safe_float(row.get("TIME_OF_POSS")),
                "avg_sec_per_touch": _safe_float(row.get("AVG_SEC_PER_TOUCH")),
                "avg_drib_per_touch": _safe_float(row.get("AVG_DRIB_PER_TOUCH")),
                "pts_per_touch": _safe_float(row.get("PTS_PER_TOUCH")),
                "elbow_touches": _safe_float(row.get("ELBOW_TOUCHES")),
                "post_touches": _safe_float(row.get("POST_TOUCHES")),
                "paint_touches": _safe_float(row.get("PAINT_TOUCHES")),
            })
    except Exception as e:
        logger.error("  Possessions failed: %s", e)

    # ------------------------------------------------------------------
    # 7. Hustle Stats
    # ------------------------------------------------------------------
    logger.info("Fetching Hustle Stats for %s ...", season_str)
    time.sleep(1)
    try:
        resp = leaguehustlestatsplayer.LeagueHustleStatsPlayer(
            season=season_str,
            per_mode_time="PerGame",
        )
        df = resp.get_data_frames()[0]
        logger.info("  -> %d players", len(df))
        for _, row in df.iterrows():
            name = row.get("PLAYER_NAME", "")
            if not name:
                continue
            _merge(name, {
                "contested_shots": _safe_float(row.get("CONTESTED_SHOTS")),
                "contested_shots_2pt": _safe_float(row.get("CONTESTED_SHOTS_2PT")),
                "contested_shots_3pt": _safe_float(row.get("CONTESTED_SHOTS_3PT")),
                "deflections": _safe_float(row.get("DEFLECTIONS")),
                "charges_drawn": _safe_float(row.get("CHARGES_DRAWN")),
                "screen_assists": _safe_float(row.get("SCREEN_ASSISTS")),
                "loose_balls_recovered": _safe_float(row.get("LOOSE_BALLS_RECOVERED")),
                "off_loose_balls_recovered": _safe_float(row.get("OFF_LOOSE_BALLS_RECOVERED")),
                "def_loose_balls_recovered": _safe_float(row.get("DEF_LOOSE_BALLS_RECOVERED")),
                "off_boxouts": _safe_float(row.get("OFF_BOXOUTS")),
                "def_boxouts": _safe_float(row.get("DEF_BOXOUTS")),
            })
    except Exception as e:
        logger.error("  Hustle Stats failed: %s", e)

    # ------------------------------------------------------------------
    # 8. Estimated Advanced Metrics
    # ------------------------------------------------------------------
    logger.info("Fetching Estimated Metrics for %s ...", season_str)
    time.sleep(1)
    try:
        resp = playerestimatedmetrics.PlayerEstimatedMetrics(
            season=season_str,
        )
        df = resp.get_data_frames()[0]
        logger.info("  -> %d players", len(df))
        for _, row in df.iterrows():
            name = row.get("PLAYER_NAME", "")
            if not name:
                continue
            _merge(name, {
                "e_off_rating": _safe_float(row.get("E_OFF_RATING")),
                "e_def_rating": _safe_float(row.get("E_DEF_RATING")),
                "e_net_rating": _safe_float(row.get("E_NET_RATING")),
                "e_ast_ratio": _safe_float(row.get("E_AST_RATIO")),
                "e_oreb_pct": _safe_float(row.get("E_OREB_PCT")),
                "e_dreb_pct": _safe_float(row.get("E_DREB_PCT")),
                "e_reb_pct": _safe_float(row.get("E_REB_PCT")),
                "e_tov_pct": _safe_float(row.get("E_TOV_PCT")),
                "e_usg_pct": _safe_float(row.get("E_USG_PCT")),
                "e_pace": _safe_float(row.get("E_PACE")),
            })
    except Exception as e:
        logger.error("  Estimated Metrics failed: %s", e)

    return players


# ============================================================================
# PBP Stats (api.pbpstats.com)
# ============================================================================

def fetch_pbp_for_season(season_str: str) -> Dict[str, Dict]:
    """Fetch player totals from PBPStats for a season.

    PBPStats provides shot-quality, at-rim efficiency, turnover breakdowns, etc.

    Args:
        season_str: NBA season string, e.g. '2023-24'

    Returns:
        dict mapping player name -> {stat_key: value, ...}
    """
    BASE_URL = "https://api.pbpstats.com"
    players: Dict[str, Dict] = {}

    logger.info("Fetching PBP Stats for %s ...", season_str)

    try:
        url = f"{BASE_URL}/get-totals/nba"
        params = {
            "Season": season_str,
            "SeasonType": "Regular+Season",
            "Type": "Player",
        }
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            logger.error("  PBP Stats HTTP %d: %s", resp.status_code, resp.text[:200])
            return players

        data = resp.json()
        rows = data.get("multi_row_table_data", [])
        if not rows:
            # Try alternative keys
            rows = data.get("results", data.get("data", []))

        logger.info("  -> %d player records", len(rows))

        for row in rows:
            name = row.get("Name", "")
            if not name:
                continue

            gp = _safe_float(row.get("GamesPlayed", row.get("GP", 1)))
            if gp == 0:
                gp = 1

            # Per-game averages: divide totals by GP
            # Shot zone accuracy/frequency fields are already percentages
            players[name] = {
                # Shot zones (per game for volume, raw pct for accuracy)
                "pbp_at_rim_fga": _safe_float(row.get("AtRimFGA", 0)) / gp,
                "pbp_at_rim_fg_pct": _safe_float(row.get("AtRimAccuracy", 0)),
                "pbp_short_mid_fga": _safe_float(row.get("ShortMidRangeFGA", 0)) / gp,
                "pbp_short_mid_fg_pct": _safe_float(row.get("ShortMidRangeAccuracy", 0)),
                "pbp_long_mid_fga": _safe_float(row.get("LongMidRangeFGA", 0)) / gp,
                "pbp_long_mid_fg_pct": _safe_float(row.get("LongMidRangeAccuracy", 0)),
                "pbp_corner3_fga": _safe_float(row.get("Corner3FGA", 0)) / gp,
                "pbp_corner3_fg_pct": _safe_float(row.get("Corner3Accuracy", 0)),
                "pbp_arc3_fga": _safe_float(row.get("Arc3FGA", 0)) / gp,
                "pbp_arc3_fg_pct": _safe_float(row.get("Arc3Accuracy", 0)),
                # Assisted vs unassisted (percentage fields)
                "pbp_assisted_pct": _safe_float(row.get("Assisted2sPct", 0)),
                "pbp_unassisted_pct": 1.0 - _safe_float(row.get("Assisted2sPct", 0)),
                # Turnovers (totals -> per game)
                "pbp_bad_pass_tov": _safe_float(row.get("BadPassTurnovers", 0)) / gp,
                "pbp_lost_ball_tov": _safe_float(row.get("LostBallTurnovers", 0)) / gp,
                # Shooting fouls drawn (totals -> per game)
                "pbp_shooting_fouls_drawn": _safe_float(row.get("ShootingFouls", 0)) / gp,
                # Shot quality
                "pbp_shot_quality_avg": _safe_float(row.get("ShotQualityAvg", 0)),
                # Efficiency
                "pbp_ts_pct": _safe_float(row.get("TsPct", 0)),
                "pbp_efg_pct": _safe_float(row.get("EfgPct", 0)),
                "pbp_usage": _safe_float(row.get("Usage", 0)),
                # Self offensive rebounds
                "pbp_self_oreb": _safe_float(row.get("SelfOReb", 0)) / gp,
                # Putback points
                "pbp_pts_putbacks": _safe_float(row.get("PtsPutbacks", 0)) / gp,
                # Games played
                "pbp_gp": gp,
            }

    except requests.exceptions.Timeout:
        logger.error("  PBP Stats timed out (30s)")
    except Exception as e:
        logger.error("  PBP Stats failed: %s", e)

    return players


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fetch historical tracking stats")
    parser.add_argument(
        "--season",
        type=str,
        default=None,
        help="Fetch a single season (e.g. '2023-24'). Default: fetch all.",
    )
    parser.add_argument(
        "--skip-pbp",
        action="store_true",
        help="Skip PBP Stats fetch (only fetch NBA.com tracking).",
    )
    args = parser.parse_args()

    OUTPUTS.mkdir(parents=True, exist_ok=True)

    seasons = [args.season] if args.season else SEASONS

    for season in seasons:
        logger.info("=" * 60)
        logger.info("SEASON: %s", season)
        logger.info("=" * 60)

        # --- NBA.com Tracking ---
        tracking = fetch_tracking_for_season(season)
        season_tag = season.replace("-", "_")
        tracking_file = OUTPUTS / f"tracking_data_{season_tag}.json"
        with open(tracking_file, "w") as f:
            json.dump(
                {
                    "season": season,
                    "fetched_at": datetime.now().isoformat(),
                    "player_count": len(tracking),
                    "players": tracking,
                },
                f,
                indent=2,
            )
        logger.info("Saved %d players tracking -> %s", len(tracking), tracking_file.name)

        # --- PBP Stats ---
        if not args.skip_pbp:
            time.sleep(2)
            pbp = fetch_pbp_for_season(season)
            pbp_file = OUTPUTS / f"pbp_data_{season_tag}.json"
            with open(pbp_file, "w") as f:
                json.dump(
                    {
                        "season": season,
                        "fetched_at": datetime.now().isoformat(),
                        "player_count": len(pbp),
                        "players": pbp,
                    },
                    f,
                    indent=2,
                )
            logger.info("Saved %d players PBP -> %s", len(pbp), pbp_file.name)
        else:
            logger.info("Skipping PBP Stats (--skip-pbp)")

        logger.info("")

    logger.info("Done! All tracking data saved to %s", OUTPUTS)


if __name__ == "__main__":
    main()
