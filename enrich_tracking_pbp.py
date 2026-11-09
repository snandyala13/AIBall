#!/usr/bin/env python3
"""
enrich_tracking_pbp.py - Add tracking and PBP features to existing records

Takes the recovered checkpoint and enriches current season records with
tracking and PBP data from the fixed API clients.
"""

import json
import logging
from collections import defaultdict

from data_pipeline import NBATrackingClient, PBPStatsClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("Enriching Current Season Records with Tracking/PBP")
    logger.info("=" * 60)

    # Load recovered checkpoint
    with open('outputs/checkpoints/recovered_checkpoint.json') as f:
        data = json.load(f)

    records = data.get('records', [])
    logger.info(f"Loaded {len(records)} total records")

    # Split by season
    historical = [r for r in records if r.get('date', '') < '2025-10-01']
    current = [r for r in records if r.get('date', '') >= '2025-10-01']
    logger.info(f"Historical: {len(historical)}, Current: {len(current)}")

    # Initialize clients
    logger.info("Initializing tracking and PBP clients...")
    tracking_client = NBATrackingClient(season='2025-26')
    pbp_client = PBPStatsClient(season='2025-26')

    # Build player caches (fetch once, use many times)
    logger.info("Building player tracking cache...")
    player_tracking = {}
    player_pbp = {}

    # Get unique players in current season
    unique_players = set()
    for r in current:
        unique_players.add(r.get('player', ''))
    logger.info(f"Found {len(unique_players)} unique players to enrich")

    # Fetch tracking stats for all players
    fetched = 0
    for player_name in unique_players:
        if not player_name:
            continue
        try:
            ts = tracking_client.get_player_tracking_stats(player_id=None, player_name=player_name)
            if ts:
                player_tracking[player_name] = {
                    "deflections": ts.deflections,
                    "contested_shots": ts.contested_shots,
                    "loose_balls_recovered": ts.loose_balls_recovered,
                    "potential_assists": ts.potential_assists,
                    "ast_to_pass_pct": ts.ast_to_pass_pct,
                    "passes_made": ts.passes_made,
                    "oreb_chance_pct": ts.oreb_chance_pct,
                    "dreb_chance_pct": ts.dreb_chance_pct,
                    "oreb_chances": ts.oreb_chances,
                    "dreb_chances": ts.dreb_chances,
                    "drives": ts.drives,
                    "drive_pts": ts.drive_pts,
                    "drive_fg_pct": ts.drive_fg_pct,
                    "catch_shoot_fg_pct": ts.catch_shoot_fg_pct,
                    "pull_up_fg_pct": ts.pull_up_fg_pct,
                    "catch_shoot_fga": ts.catch_shoot_fga,
                    "pull_up_fga": ts.pull_up_fga,
                    "touches": ts.touches,
                    "time_of_poss": ts.time_of_poss,
                    "pts_per_touch": ts.pts_per_touch,
                }
                fetched += 1
        except Exception as e:
            pass  # Player not found in tracking data

    logger.info(f"Got tracking data for {fetched}/{len(unique_players)} players")

    # Fetch PBP stats for all players
    fetched_pbp = 0
    for player_name in unique_players:
        if not player_name:
            continue
        try:
            ps = pbp_client.get_player_pbp_stats(player_name)
            if ps:
                player_pbp[player_name] = {
                    "at_rim_accuracy": ps.at_rim_accuracy,
                    "corner3_accuracy": ps.corner3_accuracy,
                    "arc3_accuracy": ps.arc3_accuracy,
                    "shot_quality_avg": ps.shot_quality_avg,
                    "assisted_2s_pct": ps.assisted_2s_pct,
                    "assisted_3s_pct": ps.assisted_3s_pct,
                    "bad_pass_turnovers": ps.bad_pass_turnovers,
                    "lost_ball_turnovers": ps.lost_ball_turnovers,
                    "ts_pct": ps.ts_pct,
                    "efg_pct": ps.efg_pct,
                    "usage": ps.usage,
                    "at_rim_assists": ps.at_rim_assists,
                    "three_pt_assists": ps.three_pt_assists,
                }
                fetched_pbp += 1
        except Exception as e:
            pass

    logger.info(f"Got PBP data for {fetched_pbp}/{len(unique_players)} players")

    # Now enrich each current season record
    logger.info("Enriching records...")
    enriched_count = 0
    tracking_added = 0
    pbp_added = 0

    for r in current:
        player_name = r.get('player', '')
        features = r.get('features', {})

        # Add tracking features
        if player_name in player_tracking:
            ts = player_tracking[player_name]
            features["track_deflections"] = ts["deflections"]
            features["track_contested_shots"] = ts["contested_shots"]
            features["track_loose_balls"] = ts["loose_balls_recovered"]
            features["track_potential_assists"] = ts["potential_assists"]
            features["track_ast_to_pass_pct"] = ts["ast_to_pass_pct"]
            features["track_passes_made"] = ts["passes_made"]
            features["track_oreb_chance_pct"] = ts["oreb_chance_pct"]
            features["track_dreb_chance_pct"] = ts["dreb_chance_pct"]
            features["track_reb_chances"] = ts["oreb_chances"] + ts["dreb_chances"]
            features["track_drives"] = ts["drives"]
            features["track_drive_pts"] = ts["drive_pts"]
            features["track_drive_fg_pct"] = ts["drive_fg_pct"]
            features["track_catch_shoot_fg_pct"] = ts["catch_shoot_fg_pct"]
            features["track_pull_up_fg_pct"] = ts["pull_up_fg_pct"]
            features["track_catch_shoot_fga"] = ts["catch_shoot_fga"]
            features["track_pull_up_fga"] = ts["pull_up_fga"]
            features["track_touches"] = ts["touches"]
            features["track_time_of_poss"] = ts["time_of_poss"]
            features["track_pts_per_touch"] = ts["pts_per_touch"]
            features["has_tracking"] = 1
            tracking_added += 1

        # Add PBP features
        if player_name in player_pbp:
            ps = player_pbp[player_name]
            features["pbp_at_rim_accuracy"] = ps["at_rim_accuracy"]
            features["pbp_corner3_accuracy"] = ps["corner3_accuracy"]
            features["pbp_arc3_accuracy"] = ps["arc3_accuracy"]
            features["pbp_shot_quality"] = ps["shot_quality_avg"]
            features["pbp_assisted_2s_pct"] = ps["assisted_2s_pct"]
            features["pbp_assisted_3s_pct"] = ps["assisted_3s_pct"]
            features["pbp_bad_pass_tov"] = ps["bad_pass_turnovers"]
            features["pbp_lost_ball_tov"] = ps["lost_ball_turnovers"]
            features["pbp_ts_pct"] = ps["ts_pct"]
            features["pbp_efg_pct"] = ps["efg_pct"]
            features["pbp_usage"] = ps["usage"]
            features["pbp_at_rim_assists"] = ps["at_rim_assists"]
            features["pbp_three_pt_assists"] = ps["three_pt_assists"]
            features["has_pbp"] = 1
            pbp_added += 1

        r['features'] = features
        enriched_count += 1

    logger.info(f"Enriched {enriched_count} records")
    logger.info(f"  Added tracking to {tracking_added} records")
    logger.info(f"  Added PBP to {pbp_added} records")

    # Combine back
    all_records = historical + current

    # Get all feature names
    all_feature_names = set()
    for r in all_records:
        all_feature_names.update(r.get('features', {}).keys())

    # Save enriched data
    output = {
        "generated": data.get("generated"),
        "last_date": data.get("last_date"),
        "total_records": len(all_records),
        "historical_records": len(historical),
        "current_season_records": len(current),
        "feature_names": sorted(list(all_feature_names)),
        "records": all_records
    }

    with open('outputs/full_training_data_enriched.json', 'w') as f:
        json.dump(output, f)

    logger.info(f"\nSaved enriched data to outputs/full_training_data_enriched.json")
    logger.info(f"Total: {len(all_records)} records with {len(all_feature_names)} features")

    # Verify
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION")
    logger.info("=" * 60)

    sample = current[-1]
    logger.info(f"Sample record from {sample.get('date')} ({sample.get('player')}):")
    f = sample.get('features', {})
    logger.info(f"  has_tracking: {f.get('has_tracking', 0)}")
    logger.info(f"  track_drives: {f.get('track_drives', 0)}")
    logger.info(f"  track_deflections: {f.get('track_deflections', 0)}")
    logger.info(f"  has_pbp: {f.get('has_pbp', 0)}")
    logger.info(f"  pbp_shot_quality: {f.get('pbp_shot_quality', 0)}")
    logger.info(f"  pbp_usage: {f.get('pbp_usage', 0)}")


if __name__ == "__main__":
    main()
