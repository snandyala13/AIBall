#!/usr/bin/env python3
"""
Fetch Historical NBA Player Props from The Odds API

This script fetches REAL betting lines from DraftKings/FanDuel for backtesting.
It uses The Odds API which charges credits per request.

IMPORTANT: This API is expensive! Each historical player props call costs:
    10 × (number of markets) × (number of regions) credits

With 6 markets × 1 region = 60 credits per game snapshot.

Usage:
    # Dry run - calculate costs without making API calls
    python fetch_historical_odds.py --dry-run --days 30

    # Actually fetch data (requires API key)
    python fetch_historical_odds.py --api-key YOUR_KEY --days 30

    # Check current database stats
    python fetch_historical_odds.py --stats
"""

import os
import sys
import argparse
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

sys.path.insert(0, os.path.dirname(__file__))
from cache_db import get_cache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "basketball_nba"

# Markets we want to fetch (6 markets)
PLAYER_PROP_MARKETS = [
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_threes",
    "player_points_rebounds_assists",  # PRA
    "player_double_double",
]

# Bookmakers to prioritize (in order of preference)
PREFERRED_BOOKMAKERS = ["draftkings", "fanduel", "betmgm", "caesars"]

# Region for US sportsbooks
REGION = "us"

# Credit costs
CREDITS_PER_HISTORICAL_EVENT = 1  # /historical/events endpoint
CREDITS_PER_HISTORICAL_ODDS = 10  # Multiplier for /historical/odds
MARKETS_COUNT = len(PLAYER_PROP_MARKETS)
REGIONS_COUNT = 1

# Cost per game = 10 × markets × regions
CREDITS_PER_GAME = CREDITS_PER_HISTORICAL_ODDS * MARKETS_COUNT * REGIONS_COUNT  # 60 credits

# Rate limiting
REQUESTS_PER_SECOND = 1  # Be conservative
REQUEST_DELAY = 1.0 / REQUESTS_PER_SECOND


# =============================================================================
# API FUNCTIONS
# =============================================================================

def get_historical_events(api_key: str, date: str, dry_run: bool = False) -> Tuple[List[Dict], int]:
    """
    Get all NBA events for a specific date.

    Cost: 1 credit (FREE if no results)

    Args:
        api_key: The Odds API key
        date: ISO8601 date string (e.g., "2025-01-01T00:00:00Z")
        dry_run: If True, don't make actual API call

    Returns:
        (list of events, credits used)
    """
    if dry_run:
        # Estimate ~12 games per day on average
        return [{"id": f"fake_{i}", "home_team": f"Home{i}", "away_team": f"Away{i}"}
                for i in range(12)], 1

    url = f"{BASE_URL}/historical/sports/{SPORT_KEY}/events"
    params = {
        "apiKey": api_key,
        "date": date,
    }

    response = requests.get(url, params=params)

    # Extract credits from headers
    credits_used = int(response.headers.get("x-requests-last", 1))
    credits_remaining = int(response.headers.get("x-requests-remaining", 0))

    logger.info(f"Historical events: {credits_used} credits used, {credits_remaining} remaining")

    if response.status_code == 200:
        data = response.json()
        return data.get("data", []), credits_used
    else:
        logger.error(f"Error fetching events: {response.status_code} - {response.text}")
        return [], credits_used


def get_historical_event_odds(api_key: str, event_id: str, snapshot_time: str,
                               dry_run: bool = False) -> Tuple[Dict, int]:
    """
    Get player props for a specific event at a specific time.

    Cost: 10 × markets × regions = 60 credits (for 6 markets, 1 region)

    Args:
        api_key: The Odds API key
        event_id: Event ID from the events endpoint
        snapshot_time: ISO8601 timestamp for the odds snapshot
        dry_run: If True, don't make actual API call

    Returns:
        (odds data dict, credits used)
    """
    if dry_run:
        # Return mock data structure
        return {
            "id": event_id,
            "bookmakers": [
                {
                    "key": "draftkings",
                    "markets": [
                        {
                            "key": "player_points",
                            "outcomes": [
                                {"name": "Over", "description": "Player Name", "price": -110, "point": 25.5},
                                {"name": "Under", "description": "Player Name", "price": -110, "point": 25.5},
                            ]
                        }
                    ]
                }
            ]
        }, CREDITS_PER_GAME

    url = f"{BASE_URL}/historical/sports/{SPORT_KEY}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": REGION,
        "markets": ",".join(PLAYER_PROP_MARKETS),
        "oddsFormat": "american",
        "date": snapshot_time,
    }

    response = requests.get(url, params=params)

    credits_used = int(response.headers.get("x-requests-last", CREDITS_PER_GAME))
    credits_remaining = int(response.headers.get("x-requests-remaining", 0))

    logger.debug(f"Event odds: {credits_used} credits used, {credits_remaining} remaining")

    if response.status_code == 200:
        data = response.json()
        return data.get("data", {}), credits_used
    else:
        logger.error(f"Error fetching odds for {event_id}: {response.status_code}")
        return {}, credits_used


# =============================================================================
# DATA PROCESSING
# =============================================================================

def parse_player_props(event_data: Dict, event_id: str, snapshot_time: str) -> List[Dict]:
    """
    Parse the odds response into individual player props for database storage.

    Returns list of dicts ready for save_historical_props_batch()
    """
    props = []

    bookmakers = event_data.get("bookmakers", [])

    for bookmaker in bookmakers:
        bookmaker_key = bookmaker.get("key", "")

        for market in bookmaker.get("markets", []):
            market_key = market.get("key", "")

            # Group outcomes by player (Over/Under pairs)
            outcomes = market.get("outcomes", [])
            player_lines = {}

            for outcome in outcomes:
                player_name = outcome.get("description", "")
                outcome_name = outcome.get("name", "")  # "Over" or "Under"
                price = outcome.get("price", 0)
                point = outcome.get("point", 0)

                if player_name not in player_lines:
                    player_lines[player_name] = {"line": point}

                if outcome_name == "Over":
                    player_lines[player_name]["over_odds"] = price
                elif outcome_name == "Under":
                    player_lines[player_name]["under_odds"] = price

            # Create prop records
            for player_name, line_data in player_lines.items():
                if "over_odds" in line_data and "under_odds" in line_data:
                    props.append({
                        "event_id": event_id,
                        "bookmaker": bookmaker_key,
                        "market": market_key,
                        "player_name": player_name,
                        "line": line_data["line"],
                        "over_odds": line_data["over_odds"],
                        "under_odds": line_data["under_odds"],
                        "snapshot_time": snapshot_time,
                    })

    return props


def calculate_snapshot_time(commence_time: str, minutes_before: int = 30) -> str:
    """
    Calculate the optimal snapshot time (X minutes before game start).

    Args:
        commence_time: Game start time in ISO8601 format
        minutes_before: How many minutes before game to snapshot

    Returns:
        ISO8601 timestamp for the snapshot
    """
    # Parse the commence time
    game_time = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))

    # Subtract minutes
    snapshot_time = game_time - timedelta(minutes=minutes_before)

    # Round to nearest 5-minute interval (API snapshots are every 5 min)
    minute = (snapshot_time.minute // 5) * 5
    snapshot_time = snapshot_time.replace(minute=minute, second=0, microsecond=0)

    return snapshot_time.strftime("%Y-%m-%dT%H:%M:%SZ")


# =============================================================================
# MAIN FETCH LOGIC
# =============================================================================

def fetch_historical_odds(api_key: str, start_date: str, days: int,
                          max_credits: int = 50000, dry_run: bool = False) -> Dict:
    """
    Main function to fetch historical odds for a date range.

    Args:
        api_key: The Odds API key
        start_date: Start date in YYYY-MM-DD format
        days: Number of days to fetch (going backwards from start_date)
        max_credits: Maximum credits to spend
        dry_run: If True, calculate costs without making API calls

    Returns:
        Summary dict with stats
    """
    cache = get_cache()

    total_credits_used = 0
    total_games = 0
    total_props = 0
    games_skipped = 0

    logger.info("=" * 70)
    logger.info(f"HISTORICAL ODDS FETCH {'(DRY RUN)' if dry_run else ''}")
    logger.info(f"Start date: {start_date}")
    logger.info(f"Days to fetch: {days}")
    logger.info(f"Max credits: {max_credits:,}")
    logger.info(f"Cost per game: {CREDITS_PER_GAME} credits")
    logger.info(f"Markets: {', '.join(PLAYER_PROP_MARKETS)}")
    logger.info("=" * 70)

    # Parse start date
    current_date = datetime.strptime(start_date, "%Y-%m-%d")

    for day_offset in range(days):
        date = current_date - timedelta(days=day_offset)
        date_str = date.strftime("%Y-%m-%d")

        # Check if we've already fetched this date
        existing_games = cache.get_historical_games_for_date(date_str)
        if existing_games and not dry_run:
            logger.info(f"[{date_str}] Already have {len(existing_games)} games, skipping")
            games_skipped += len(existing_games)
            continue

        # Check credit budget
        if total_credits_used >= max_credits:
            logger.warning(f"Credit limit reached ({total_credits_used:,}/{max_credits:,})")
            break

        logger.info(f"[{date_str}] Fetching events...")

        # Step 1: Get events for this date (FREE/1 credit)
        snapshot_date = f"{date_str}T12:00:00Z"  # Noon UTC as starting point
        events, credits = get_historical_events(api_key, snapshot_date, dry_run)
        total_credits_used += credits

        if not events:
            logger.info(f"[{date_str}] No events found")
            continue

        logger.info(f"[{date_str}] Found {len(events)} events")

        # Step 2: Fetch player props for each event
        for event in events:
            # Check credit budget
            if total_credits_used + CREDITS_PER_GAME > max_credits:
                logger.warning(f"Would exceed credit limit, stopping")
                break

            event_id = event.get("id", "")
            home_team = event.get("home_team", "")
            away_team = event.get("away_team", "")
            commence_time = event.get("commence_time", "")

            # Calculate snapshot time (10 min before game - closest to final lines)
            if commence_time:
                snapshot_time = calculate_snapshot_time(commence_time, minutes_before=10)
            else:
                snapshot_time = snapshot_date

            logger.info(f"  [{away_team} @ {home_team}] Fetching props...")

            # Fetch odds
            if not dry_run:
                time.sleep(REQUEST_DELAY)  # Rate limiting

            odds_data, credits = get_historical_event_odds(
                api_key, event_id, snapshot_time, dry_run
            )
            total_credits_used += credits

            # Parse and save
            if odds_data:
                props = parse_player_props(odds_data, event_id, snapshot_time)

                if props and not dry_run:
                    # Save game
                    cache.save_historical_game(
                        event_id=event_id,
                        sport_key=SPORT_KEY,
                        commence_time=commence_time,
                        home_team=home_team,
                        away_team=away_team,
                        snapshot_time=snapshot_time
                    )

                    # Save props
                    cache.save_historical_props_batch(props)

                    # Log API usage
                    cache.log_odds_api_usage(
                        endpoint=f"/historical/events/{event_id}/odds",
                        credits_used=credits,
                        credits_remaining=max_credits - total_credits_used,
                        notes=f"{away_team} @ {home_team}"
                    )

                total_props += len(props)
                logger.info(f"    -> {len(props)} player props")

            total_games += 1

        # Progress update
        logger.info(f"[{date_str}] Day complete. Total credits: {total_credits_used:,}")

    # Summary
    summary = {
        "dry_run": dry_run,
        "days_processed": days,
        "games_fetched": total_games,
        "games_skipped": games_skipped,
        "props_fetched": total_props,
        "credits_used": total_credits_used,
        "credits_remaining": max_credits - total_credits_used,
    }

    logger.info("")
    logger.info("=" * 70)
    logger.info("FETCH COMPLETE")
    logger.info("=" * 70)
    for key, value in summary.items():
        logger.info(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")
    logger.info("=" * 70)

    return summary


def estimate_costs(days: int) -> Dict:
    """
    Estimate API costs for fetching X days of data.
    """
    avg_games_per_day = 12  # NBA average
    games = days * avg_games_per_day

    # Events endpoint: 1 credit per day
    events_credits = days * 1

    # Props endpoint: 60 credits per game
    props_credits = games * CREDITS_PER_GAME

    total_credits = events_credits + props_credits

    return {
        "days": days,
        "estimated_games": games,
        "events_credits": events_credits,
        "props_credits": props_credits,
        "total_credits": total_credits,
        "cost_per_game": CREDITS_PER_GAME,
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fetch historical NBA player props from The Odds API")

    parser.add_argument("--api-key", type=str, help="The Odds API key")
    parser.add_argument("--days", type=int, default=30, help="Number of days to fetch (default: 30)")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Start date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--max-credits", type=int, default=50000,
                        help="Maximum credits to spend (default: 50000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Calculate costs without making API calls")
    parser.add_argument("--stats", action="store_true",
                        help="Show current database statistics")
    parser.add_argument("--estimate", action="store_true",
                        help="Estimate costs for given number of days")

    args = parser.parse_args()

    # Show stats
    if args.stats:
        cache = get_cache()
        stats = cache.get_historical_odds_stats()
        print("\n" + "=" * 50)
        print("HISTORICAL ODDS DATABASE STATS")
        print("=" * 50)
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print("=" * 50)
        return

    # Estimate costs
    if args.estimate:
        estimate = estimate_costs(args.days)
        print("\n" + "=" * 50)
        print(f"COST ESTIMATE FOR {args.days} DAYS")
        print("=" * 50)
        for key, value in estimate.items():
            print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")
        print("=" * 50)
        return

    # Validate API key for actual fetching
    if not args.dry_run and not args.api_key:
        print("ERROR: --api-key required for actual fetching (or use --dry-run)")
        sys.exit(1)

    # Default start date to yesterday
    if not args.start_date:
        args.start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Run fetch
    summary = fetch_historical_odds(
        api_key=args.api_key or "DRY_RUN_KEY",
        start_date=args.start_date,
        days=args.days,
        max_credits=args.max_credits,
        dry_run=args.dry_run
    )

    return summary


if __name__ == "__main__":
    main()
