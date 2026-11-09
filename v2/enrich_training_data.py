#!/usr/bin/env python3
"""
v2/enrich_training_data.py - Merge tracking + PBP stats into the ML feature matrix.

Adds ~30 tracking features (drives, catch-shoot, possessions, hustle, etc.) and
~15 PBP features (shot zones, assist rates, turnover types) to each training leg.

Tracking stats are SEASON-LEVEL: one value per player per season.  For a leg from
Feb 2024 we use the full 2023-24 season stats.  This is slightly forward-looking
(the season totals include games after Feb) but it is the same data DraftKings and
sharp bettors have access to mid-season, so it is fair game for training.

Usage:
    python -m v2.enrich_training_data            # Enrich from cached parquets
    python -m v2.enrich_training_data --rebuild   # Rebuild features first
"""

import argparse
import json
import logging
import sys
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

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS = BASE_DIR / "outputs"
V2_TRAIN = OUTPUTS / "v2_training"

# Season date boundaries: map a game date -> season string
SEASON_RANGES = [
    # (start_date, end_date, season_str)
    ("2023-10-01", "2024-09-30", "2023-24"),
    ("2024-10-01", "2025-09-30", "2024-25"),
    ("2025-10-01", "2026-09-30", "2025-26"),
    ("2026-10-01", "2027-09-30", "2026-27"),
]

# All tracking feature keys we expect from the fetch script
TRACKING_FEATURES = [
    # Drives (10)
    "drives", "drive_fgm", "drive_fga", "drive_fg_pct",
    "drive_ftm", "drive_fta", "drive_pts", "drive_ast", "drive_tov", "drive_pf",
    # Catch & Shoot (7)
    "catch_shoot_fgm", "catch_shoot_fga", "catch_shoot_fg_pct",
    "catch_shoot_fg3m", "catch_shoot_fg3a", "catch_shoot_fg3_pct",
    "catch_shoot_efg_pct",
    # Pull-Up (7)
    "pull_up_fgm", "pull_up_fga", "pull_up_fg_pct",
    "pull_up_fg3m", "pull_up_fg3a", "pull_up_fg3_pct", "pull_up_efg_pct",
    # Passing (8)
    "passes_made", "passes_received", "potential_ast", "ast_points_created",
    "ast_adj", "ast_to_pass_pct", "secondary_ast", "ft_ast",
    # Rebounding (12)
    "oreb_chances", "oreb_chance_pct", "oreb_contest", "oreb_uncontest",
    "dreb_chances", "dreb_chance_pct", "dreb_contest", "dreb_uncontest",
    "reb_chances", "reb_chance_pct", "avg_oreb_dist", "avg_dreb_dist",
    # Possessions (9)
    "touches", "front_ct_touches", "time_of_poss", "avg_sec_per_touch",
    "avg_drib_per_touch", "pts_per_touch", "elbow_touches", "post_touches",
    "paint_touches",
    # Hustle (11)
    "contested_shots", "contested_shots_2pt", "contested_shots_3pt",
    "deflections", "charges_drawn", "screen_assists",
    "loose_balls_recovered", "off_loose_balls_recovered",
    "def_loose_balls_recovered", "off_boxouts", "def_boxouts",
    # Estimated Metrics (10)
    "e_off_rating", "e_def_rating", "e_net_rating", "e_ast_ratio",
    "e_oreb_pct", "e_dreb_pct", "e_reb_pct", "e_tov_pct", "e_usg_pct",
    "e_pace",
]

# PBP feature keys
PBP_FEATURES = [
    "pbp_at_rim_fga", "pbp_at_rim_fg_pct",
    "pbp_short_mid_fga", "pbp_short_mid_fg_pct",
    "pbp_long_mid_fga", "pbp_long_mid_fg_pct",
    "pbp_corner3_fga", "pbp_corner3_fg_pct",
    "pbp_arc3_fga", "pbp_arc3_fg_pct",
    "pbp_assisted_pct", "pbp_unassisted_pct",
    "pbp_bad_pass_tov", "pbp_lost_ball_tov",
    "pbp_shooting_fouls_drawn",
    "pbp_shot_quality_avg", "pbp_ts_pct", "pbp_efg_pct",
    "pbp_usage", "pbp_self_oreb", "pbp_pts_putbacks",
]


def date_to_season(date_str: str) -> Optional[str]:
    """Map a game date (YYYY-MM-DD) to an NBA season string.

    NBA seasons run Oct-June, so:
        2023-10-24 -> '2023-24'
        2024-03-15 -> '2023-24'
        2026-01-05 -> '2025-26'
    """
    for start, end, season in SEASON_RANGES:
        if start <= date_str <= end:
            return season
    # Fallback: derive from month
    parts = date_str.split("-")
    if len(parts) >= 2:
        year = int(parts[0])
        month = int(parts[1])
        if month >= 10:
            return f"{year}-{str(year + 1)[-2:]}"
        else:
            return f"{year - 1}-{str(year)[-2:]}"
    return None


def load_tracking_data() -> Dict[str, Dict[str, Dict]]:
    """Load all tracking data JSONs from outputs/.

    Returns:
        dict mapping season_str -> {player_name: {feature: value}}
    """
    tracking = {}
    for path in sorted(OUTPUTS.glob("tracking_data_*.json")):
        logger.info("Loading tracking: %s", path.name)
        with open(path) as f:
            data = json.load(f)
        season = data.get("season", "")
        players = data.get("players", {})
        tracking[season] = players
        logger.info("  -> %d players for %s", len(players), season)
    return tracking


def load_pbp_data() -> Dict[str, Dict[str, Dict]]:
    """Load all PBP data JSONs from outputs/.

    Returns:
        dict mapping season_str -> {player_name: {feature: value}}
    """
    pbp = {}
    for path in sorted(OUTPUTS.glob("pbp_data_*.json")):
        logger.info("Loading PBP: %s", path.name)
        with open(path) as f:
            data = json.load(f)
        season = data.get("season", "")
        players = data.get("players", {})
        pbp[season] = players
        logger.info("  -> %d players for %s", len(players), season)
    return pbp


def enrich_feature_matrix(
    X: pd.DataFrame,
    meta: pd.DataFrame,
    tracking: Dict[str, Dict[str, Dict]],
    pbp: Dict[str, Dict[str, Dict]],
) -> pd.DataFrame:
    """Add tracking + PBP features to the feature matrix.

    For each row, looks up the player's tracking stats for that season.
    Missing values are filled with 0 (no tracking data available for that
    player, which happens for players with minimal minutes).

    Args:
        X: existing feature DataFrame (N rows)
        meta: metadata DataFrame with 'player', 'date' columns
        tracking: {season: {player: {feature: value}}}
        pbp: {season: {player: {feature: value}}}

    Returns:
        Enriched DataFrame with additional columns.
    """
    n = len(X)
    assert len(meta) == n, f"X and meta must have same length ({n} vs {len(meta)})"

    # Pre-build new columns as arrays for speed
    new_cols = {feat: np.zeros(n) for feat in TRACKING_FEATURES}
    new_cols.update({feat: np.zeros(n) for feat in PBP_FEATURES})
    new_cols["has_tracking"] = np.zeros(n)
    new_cols["has_pbp"] = np.zeros(n)

    matched_tracking = 0
    matched_pbp = 0

    for i in range(n):
        player = meta.iloc[i]["player"]
        date = meta.iloc[i]["date"]
        season = date_to_season(date)

        if season is None:
            continue

        # Tracking lookup
        season_tracking = tracking.get(season, {})
        player_tracking = season_tracking.get(player)
        if player_tracking:
            new_cols["has_tracking"][i] = 1.0
            matched_tracking += 1
            for feat in TRACKING_FEATURES:
                val = player_tracking.get(feat, 0)
                try:
                    new_cols[feat][i] = float(val) if val is not None else 0.0
                except (ValueError, TypeError):
                    new_cols[feat][i] = 0.0

        # PBP lookup
        season_pbp = pbp.get(season, {})
        player_pbp = season_pbp.get(player)
        if not player_pbp:
            # Try case-insensitive match
            for pname, pdata in season_pbp.items():
                if pname.lower() == player.lower():
                    player_pbp = pdata
                    break
        if player_pbp:
            new_cols["has_pbp"][i] = 1.0
            matched_pbp += 1
            for feat in PBP_FEATURES:
                val = player_pbp.get(feat, 0)
                try:
                    new_cols[feat][i] = float(val) if val is not None else 0.0
                except (ValueError, TypeError):
                    new_cols[feat][i] = 0.0

    logger.info(
        "Enrichment: %d/%d (%.1f%%) legs matched tracking, %d/%d (%.1f%%) matched PBP",
        matched_tracking, n, 100 * matched_tracking / max(n, 1),
        matched_pbp, n, 100 * matched_pbp / max(n, 1),
    )

    # Build enriched DataFrame
    enriched = X.copy()
    for col_name, values in new_cols.items():
        enriched[col_name] = values

    return enriched


def main():
    parser = argparse.ArgumentParser(description="Enrich training data with tracking stats")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild base features from scratch before enriching.",
    )
    parser.add_argument(
        "--skip-pbp",
        action="store_true",
        help="Skip PBP feature enrichment.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load (or rebuild) the base feature matrix
    # ------------------------------------------------------------------
    features_path = V2_TRAIN / "features.parquet"
    targets_path = V2_TRAIN / "targets.parquet"
    meta_path = V2_TRAIN / "metadata.parquet"

    if args.rebuild or not features_path.exists():
        logger.info("Building base features from scratch ...")
        from v2.features import FeatureBuilder
        fb = FeatureBuilder()
        fb.load_games()
        fb.build_player_history()
        X, y, meta = fb.build_training_data(target_mode="line_anchor")
        V2_TRAIN.mkdir(parents=True, exist_ok=True)
        X.to_parquet(features_path, index=False)
        pd.Series(y).to_frame("target").to_parquet(targets_path, index=False)
        meta.to_parquet(meta_path, index=False)
        logger.info("Base features: %d samples x %d features", len(X), len(X.columns))
    else:
        logger.info("Loading cached base features from %s", V2_TRAIN)
        X = pd.read_parquet(features_path)
        y = pd.read_parquet(targets_path)["target"].values
        meta = pd.read_parquet(meta_path)
        logger.info("Base features: %d samples x %d features", len(X), len(X.columns))

    # ------------------------------------------------------------------
    # 2. Load tracking + PBP data
    # ------------------------------------------------------------------
    tracking = load_tracking_data()
    if not tracking:
        logger.warning(
            "No tracking data found in %s. Run `python -m v2.fetch_tracking` first.",
            OUTPUTS,
        )
    pbp = {} if args.skip_pbp else load_pbp_data()

    # ------------------------------------------------------------------
    # 3. Enrich
    # ------------------------------------------------------------------
    X_enriched = enrich_feature_matrix(X, meta, tracking, pbp)

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    new_feat_count = len(X_enriched.columns) - len(X.columns)
    logger.info(
        "Enriched: %d samples x %d features (+%d new)",
        len(X_enriched), len(X_enriched.columns), new_feat_count,
    )

    # Show coverage per season
    for season in sorted(tracking.keys()):
        season_mask = meta["date"].apply(lambda d: date_to_season(d) == season)
        n_season = season_mask.sum()
        if n_season > 0:
            n_has = int(X_enriched.loc[season_mask, "has_tracking"].sum())
            logger.info(
                "  Season %s: %d/%d legs have tracking (%.1f%%)",
                season, n_has, n_season, 100 * n_has / n_season,
            )

    # Print feature importance preview (non-zero variance new features)
    new_features = [c for c in X_enriched.columns if c not in X.columns]
    nz_features = [c for c in new_features if X_enriched[c].std() > 0]
    logger.info(
        "New features with non-zero variance: %d / %d", len(nz_features), len(new_features)
    )

    # ------------------------------------------------------------------
    # 5. Save enriched data
    # ------------------------------------------------------------------
    enriched_dir = OUTPUTS / "v2_training_enriched"
    enriched_dir.mkdir(parents=True, exist_ok=True)

    X_enriched.to_parquet(enriched_dir / "features.parquet", index=False)
    pd.Series(y).to_frame("target").to_parquet(enriched_dir / "targets.parquet", index=False)
    meta.to_parquet(enriched_dir / "metadata.parquet", index=False)

    logger.info("Saved enriched training data to %s/", enriched_dir)

    # Also save a human-readable summary
    summary = {
        "total_samples": len(X_enriched),
        "base_features": len(X.columns),
        "enriched_features": len(X_enriched.columns),
        "new_features_added": new_feat_count,
        "tracking_features": len(TRACKING_FEATURES),
        "pbp_features": len(PBP_FEATURES),
        "tracking_match_rate": float(X_enriched["has_tracking"].mean()),
        "pbp_match_rate": float(X_enriched["has_pbp"].mean()),
        "all_feature_names": list(X_enriched.columns),
        "new_feature_names": new_features,
    }
    with open(enriched_dir / "enrichment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to enrichment_summary.json")

    print(f"\n{'=' * 70}")
    print(f"ENRICHMENT COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Base features:     {len(X.columns)}")
    print(f"  New features:      {new_feat_count}")
    print(f"  Total features:    {len(X_enriched.columns)}")
    print(f"  Total samples:     {len(X_enriched):,}")
    print(f"  Tracking match:    {X_enriched['has_tracking'].mean():.1%}")
    print(f"  PBP match:         {X_enriched['has_pbp'].mean():.1%}")
    print(f"  Output:            {enriched_dir}/")
    print()


if __name__ == "__main__":
    main()
