#!/usr/bin/env python3
"""
refetch_current_season.py - Re-fetch 2025-26 season with full 70 features

Takes existing checkpoint with historical data (36 features) and:
1. Keeps all records before Oct 1, 2025
2. Re-fetches Oct 1, 2025 - Jan 10, 2026 with Tracking + PBP features (70 total)
3. Merges into final training dataset
"""

import os
import json
import logging
import time
from datetime import datetime, timedelta

# Import from main build script
from build_full_training_data import (
    OddsAPIClient, BallDontLieClient, CTGClient,
    TrainingRecord, process_date, save_checkpoint_with_date,
    ODDS_API_KEY, BALLDONTLIE_API_KEY, HAS_TRACKING
)

if HAS_TRACKING:
    from data_pipeline import NBATrackingClient, PBPStatsClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Re-fetch current season with full features"""

    logger.info("="*60)
    logger.info("Re-fetching 2025-26 Season with Full 70 Features")
    logger.info("="*60)

    # Load existing checkpoint
    checkpoint_file = "outputs/checkpoints/latest_checkpoint.json"
    if not os.path.exists(checkpoint_file):
        logger.error("No checkpoint found! Run build_full_training_data.py first.")
        return

    with open(checkpoint_file) as f:
        checkpoint = json.load(f)

    all_records = [TrainingRecord(**r) if isinstance(r, dict) else r for r in checkpoint.get("records", [])]
    logger.info(f"Loaded checkpoint with {len(all_records)} total records")

    # Filter to keep only records BEFORE Oct 1, 2025 (historical data)
    historical_records = [r for r in all_records if r.date < "2025-10-01"]
    logger.info(f"Keeping {len(historical_records)} historical records (before Oct 2025)")

    # Initialize clients
    odds_client = OddsAPIClient(ODDS_API_KEY)
    bdl_client = BallDontLieClient(BALLDONTLIE_API_KEY)
    ctg = CTGClient()
    ctg.load_all()

    # Initialize tracking/PBP clients - REQUIRED for current season
    tracking_client = None
    pbp_client = None
    if HAS_TRACKING:
        try:
            tracking_client = NBATrackingClient(season='2025-26')
            pbp_client = PBPStatsClient(season='2025-26')
            logger.info("Tracking and PBP clients initialized for 2025-26 season")
        except Exception as e:
            logger.error(f"FAILED to initialize tracking/PBP clients: {e}")
            logger.error("Cannot proceed without tracking/PBP for 70-feature build!")
            return
    else:
        logger.error("HAS_TRACKING is False - cannot build 70-feature dataset!")
        return

    # Caches
    player_game_logs = {}
    player_tracking_cache = {}
    player_pbp_cache = {}

    # Fetch Oct 1, 2025 - Jan 10, 2026
    start_date = datetime(2025, 10, 1)
    end_date = datetime(2026, 1, 10)

    current_season_records = []
    current_date = start_date
    total_days = (end_date - start_date).days
    processed_days = 0
    start_time = time.time()

    logger.info(f"\nFetching {total_days} days: {start_date.date()} to {end_date.date()}")
    logger.info("="*60)

    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        processed_days += 1

        # Progress estimate
        elapsed = time.time() - start_time
        if processed_days > 1:
            avg_time = elapsed / processed_days
            remaining_days = total_days - processed_days
            eta_minutes = (remaining_days * avg_time) / 60
            progress_pct = processed_days / total_days * 100
            logger.info(f"[{progress_pct:.1f}%] Processing {date_str} (ETA: {eta_minutes:.0f} min)...")
        else:
            logger.info(f"Processing {date_str}...")

        try:
            records = process_date(
                date_str,
                odds_client,
                bdl_client,
                ctg,
                player_game_logs,
                tracking_client=tracking_client,
                pbp_client=pbp_client,
                player_tracking_cache=player_tracking_cache,
                player_pbp_cache=player_pbp_cache
            )

            current_season_records.extend(records)
            logger.info(f"  Got {len(records)} records, season total: {len(current_season_records)}")

            # Check API usage
            if odds_client.requests_remaining:
                remaining = int(odds_client.requests_remaining)
                if remaining < 500:
                    logger.warning(f"Low API quota ({remaining}), saving progress...")
                    break

            # Save checkpoint every 7 days
            if processed_days % 7 == 0:
                combined = historical_records + current_season_records
                save_checkpoint_with_date(combined, date_str)
                logger.info(f"  Checkpoint saved: {len(combined)} total records")

        except KeyboardInterrupt:
            logger.info("Interrupted - saving checkpoint...")
            combined = historical_records + current_season_records
            save_checkpoint_with_date(combined, date_str)
            return

        except Exception as e:
            logger.error(f"Error processing {date_str}: {e}")
            import traceback
            traceback.print_exc()

        current_date += timedelta(days=1)

    # Combine all records
    all_records = historical_records + current_season_records
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE!")
    logger.info(f"  Historical records (36 features): {len(historical_records)}")
    logger.info(f"  Current season records (70 features): {len(current_season_records)}")
    logger.info(f"  Total records: {len(all_records)}")
    logger.info(f"{'='*60}")

    # Check feature counts
    if current_season_records:
        sample = current_season_records[0]
        feature_count = len(sample.features) if hasattr(sample, 'features') else 0
        logger.info(f"Sample current season record has {feature_count} features")

    # Save final combined dataset
    save_checkpoint_with_date(all_records, end_date.strftime("%Y-%m-%d"))

    # Also save as the final training file
    os.makedirs("outputs", exist_ok=True)
    records_dict = [r.__dict__ if hasattr(r, '__dict__') else r for r in all_records]

    # Get all feature names
    all_feature_names = set()
    for r in records_dict:
        if isinstance(r, dict) and 'features' in r:
            all_feature_names.update(r['features'].keys())

    with open("outputs/full_training_data_70features.json", "w") as f:
        json.dump({
            "generated": datetime.now().isoformat(),
            "total_records": len(all_records),
            "historical_records": len(historical_records),
            "current_season_records": len(current_season_records),
            "feature_names": sorted(list(all_feature_names)),
            "records": records_dict
        }, f)

    logger.info(f"Saved to outputs/full_training_data_70features.json")
    logger.info(f"Feature names ({len(all_feature_names)}): {sorted(list(all_feature_names))[:10]}...")


if __name__ == "__main__":
    main()
