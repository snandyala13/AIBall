#!/usr/bin/env python3
"""
v2/travel_features.py - Travel/fatigue and calendar features for prop prediction.

Computes features from travel distance, timezone changes, altitude, schedule
density, and calendar position. All derived from static arena data + schedule,
no external API calls needed.

Feature groups:
  Travel (11 features):
    travel_miles, timezone_delta, altitude_ft, altitude_delta, is_denver_altitude,
    road_trip_length, games_in_last_5_days, games_in_last_7_days, days_rest,
    is_b2b, is_3_in_4

  Calendar (8 features):
    month_of_season, season_game_number, is_preseason_month, is_pre_allstar,
    is_post_allstar, is_pre_trade_deadline, day_of_week, is_weekend

Usage:
    python -m v2.travel_features           # Demo with sample data
"""

import json
import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta
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


# =============================================================================
# NBA Arena Data (static — coordinates, timezone, altitude)
# =============================================================================

NBA_ARENAS = {
    "Atlanta Hawks": {"city": "Atlanta", "lat": 33.757, "lon": -84.396, "tz": "US/Eastern", "altitude_ft": 1050},
    "Boston Celtics": {"city": "Boston", "lat": 42.366, "lon": -71.062, "tz": "US/Eastern", "altitude_ft": 20},
    "Brooklyn Nets": {"city": "Brooklyn", "lat": 40.683, "lon": -73.975, "tz": "US/Eastern", "altitude_ft": 30},
    "Charlotte Hornets": {"city": "Charlotte", "lat": 35.225, "lon": -80.839, "tz": "US/Eastern", "altitude_ft": 751},
    "Chicago Bulls": {"city": "Chicago", "lat": 41.881, "lon": -87.674, "tz": "US/Central", "altitude_ft": 594},
    "Cleveland Cavaliers": {"city": "Cleveland", "lat": 41.496, "lon": -81.688, "tz": "US/Eastern", "altitude_ft": 653},
    "Dallas Mavericks": {"city": "Dallas", "lat": 32.790, "lon": -96.810, "tz": "US/Central", "altitude_ft": 430},
    "Denver Nuggets": {"city": "Denver", "lat": 39.749, "lon": -104.999, "tz": "US/Mountain", "altitude_ft": 5280},
    "Detroit Pistons": {"city": "Detroit", "lat": 42.341, "lon": -83.055, "tz": "US/Eastern", "altitude_ft": 600},
    "Golden State Warriors": {"city": "San Francisco", "lat": 37.768, "lon": -122.388, "tz": "US/Pacific", "altitude_ft": 10},
    "Houston Rockets": {"city": "Houston", "lat": 29.751, "lon": -95.362, "tz": "US/Central", "altitude_ft": 50},
    "Indiana Pacers": {"city": "Indianapolis", "lat": 39.764, "lon": -86.156, "tz": "US/Eastern", "altitude_ft": 717},
    "LA Clippers": {"city": "Inglewood", "lat": 33.958, "lon": -118.341, "tz": "US/Pacific", "altitude_ft": 60},
    "Los Angeles Lakers": {"city": "Los Angeles", "lat": 34.043, "lon": -118.267, "tz": "US/Pacific", "altitude_ft": 270},
    "Memphis Grizzlies": {"city": "Memphis", "lat": 35.138, "lon": -90.051, "tz": "US/Central", "altitude_ft": 337},
    "Miami Heat": {"city": "Miami", "lat": 25.781, "lon": -80.187, "tz": "US/Eastern", "altitude_ft": 6},
    "Milwaukee Bucks": {"city": "Milwaukee", "lat": 43.045, "lon": -87.917, "tz": "US/Central", "altitude_ft": 634},
    "Minnesota Timberwolves": {"city": "Minneapolis", "lat": 44.980, "lon": -93.276, "tz": "US/Central", "altitude_ft": 830},
    "New Orleans Pelicans": {"city": "New Orleans", "lat": 29.949, "lon": -90.082, "tz": "US/Central", "altitude_ft": 3},
    "New York Knicks": {"city": "New York", "lat": 40.751, "lon": -73.994, "tz": "US/Eastern", "altitude_ft": 33},
    "Oklahoma City Thunder": {"city": "Oklahoma City", "lat": 35.463, "lon": -97.515, "tz": "US/Central", "altitude_ft": 1201},
    "Orlando Magic": {"city": "Orlando", "lat": 28.539, "lon": -81.384, "tz": "US/Eastern", "altitude_ft": 82},
    "Philadelphia 76ers": {"city": "Philadelphia", "lat": 39.901, "lon": -75.172, "tz": "US/Eastern", "altitude_ft": 39},
    "Phoenix Suns": {"city": "Phoenix", "lat": 33.446, "lon": -112.071, "tz": "US/Mountain", "altitude_ft": 1086},
    "Portland Trail Blazers": {"city": "Portland", "lat": 45.532, "lon": -122.667, "tz": "US/Pacific", "altitude_ft": 50},
    "Sacramento Kings": {"city": "Sacramento", "lat": 38.580, "lon": -121.500, "tz": "US/Pacific", "altitude_ft": 30},
    "San Antonio Spurs": {"city": "San Antonio", "lat": 29.427, "lon": -98.438, "tz": "US/Central", "altitude_ft": 650},
    "Toronto Raptors": {"city": "Toronto", "lat": 43.643, "lon": -79.379, "tz": "US/Eastern", "altitude_ft": 249},
    "Utah Jazz": {"city": "Salt Lake City", "lat": 40.768, "lon": -111.901, "tz": "US/Mountain", "altitude_ft": 4226},
    "Washington Wizards": {"city": "Washington", "lat": 38.898, "lon": -77.021, "tz": "US/Eastern", "altitude_ft": 40},
}

# Timezone UTC offsets (standard time, not accounting for DST — close enough for features)
TZ_OFFSETS = {
    "US/Eastern": -5,
    "US/Central": -6,
    "US/Mountain": -7,
    "US/Pacific": -8,
}

# High altitude threshold (ft) — Denver and Utah have meaningful altitude effects
HIGH_ALTITUDE_THRESHOLD = 4000


# =============================================================================
# Haversine distance
# =============================================================================

def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great circle distance between two points in miles.

    Uses the haversine formula. Earth radius = 3959 miles.

    Args:
        lat1, lon1: coordinates of point 1 (degrees)
        lat2, lon2: coordinates of point 2 (degrees)

    Returns:
        distance in miles
    """
    R = 3959.0  # Earth radius in miles

    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def _get_arena(team_name: str) -> Optional[Dict]:
    """Look up arena info for a team, with fuzzy matching."""
    if team_name in NBA_ARENAS:
        return NBA_ARENAS[team_name]

    # Fuzzy: match by city or partial team name
    team_lower = team_name.lower()
    for full_name, info in NBA_ARENAS.items():
        if (
            full_name.lower() in team_lower
            or team_lower in full_name.lower()
            or info["city"].lower() in team_lower
        ):
            return info

    return None


def _get_tz_offset(team_name: str) -> int:
    """Get UTC offset for a team's home city."""
    arena = _get_arena(team_name)
    if arena:
        return TZ_OFFSETS.get(arena["tz"], -5)
    return -5  # Default to Eastern


# =============================================================================
# Travel features
# =============================================================================

def get_travel_features(
    team_name: str,
    opponent_name: str,
    game_date: str,
    is_home: bool,
    schedule: List[Dict],
) -> Dict[str, float]:
    """Compute travel and fatigue features for a team in a specific game.

    Args:
        team_name: full team name (e.g. "Boston Celtics")
        opponent_name: full opponent name
        game_date: ISO date string "YYYY-MM-DD"
        is_home: whether team_name is the home team
        schedule: team's schedule as list of dicts sorted by date:
            [{"date": "YYYY-MM-DD", "opponent": "team", "is_home": bool,
              "game_location_team": "team whose arena the game is at"}, ...]
            Only games BEFORE game_date are used (no lookahead).

    Returns:
        dict of 11 travel/fatigue features
    """
    game_dt = datetime.strptime(game_date, "%Y-%m-%d")

    # Current game venue
    if is_home:
        venue_team = team_name
    else:
        venue_team = opponent_name

    venue = _get_arena(venue_team)
    if venue is None:
        venue = _get_arena(team_name)  # Fallback

    # Filter schedule to games BEFORE this game
    prior_games = [
        g for g in schedule
        if g["date"] < game_date
    ]
    prior_games.sort(key=lambda g: g["date"])

    # -- Travel miles (from last game location to this game) --
    travel_miles = 0.0
    timezone_delta = 0
    altitude_delta = 0.0
    last_venue = None

    if prior_games:
        last_game = prior_games[-1]
        last_venue_team = last_game.get("game_location_team", "")
        if not last_venue_team:
            # Infer: if team was home, venue is team; if away, venue is opponent
            last_venue_team = team_name if last_game.get("is_home") else last_game.get("opponent", team_name)

        last_venue = _get_arena(last_venue_team)
        if last_venue and venue:
            travel_miles = haversine_miles(
                last_venue["lat"], last_venue["lon"],
                venue["lat"], venue["lon"],
            )
            last_tz = TZ_OFFSETS.get(last_venue["tz"], -5)
            curr_tz = TZ_OFFSETS.get(venue["tz"], -5)
            timezone_delta = abs(curr_tz - last_tz)
            altitude_delta = abs(venue["altitude_ft"] - last_venue["altitude_ft"])

    # -- Altitude --
    altitude_ft = venue["altitude_ft"] if venue else 0
    is_denver_altitude = 1 if altitude_ft >= HIGH_ALTITUDE_THRESHOLD else 0

    # -- Road trip length --
    road_trip_length = 0
    if not is_home:
        road_trip_length = 1
        # Count consecutive away games ending at this game
        for g in reversed(prior_games):
            if not g.get("is_home", True):
                road_trip_length += 1
            else:
                break

    # -- Schedule density --
    games_in_last_5 = 0
    games_in_last_7 = 0
    for g in prior_games:
        g_dt = datetime.strptime(g["date"], "%Y-%m-%d")
        diff = (game_dt - g_dt).days
        if diff <= 5:
            games_in_last_5 += 1
        if diff <= 7:
            games_in_last_7 += 1

    # -- Days rest --
    days_rest = 99  # Default if no prior games
    if prior_games:
        last_dt = datetime.strptime(prior_games[-1]["date"], "%Y-%m-%d")
        days_rest = (game_dt - last_dt).days

    is_b2b = 1 if days_rest <= 1 else 0

    # -- 3 in 4 detection --
    is_3_in_4 = 0
    # Count games in the 3-day window ending on game_date (inclusive of this game)
    games_in_window = 1  # This game counts
    for g in prior_games:
        g_dt = datetime.strptime(g["date"], "%Y-%m-%d")
        diff = (game_dt - g_dt).days
        if 0 < diff <= 3:
            games_in_window += 1
    if games_in_window >= 3:
        is_3_in_4 = 1

    return {
        "travel_miles": round(travel_miles, 1),
        "timezone_delta": timezone_delta,
        "altitude_ft": altitude_ft,
        "altitude_delta": round(altitude_delta, 0),
        "is_denver_altitude": is_denver_altitude,
        "road_trip_length": road_trip_length,
        "games_in_last_5_days": games_in_last_5,
        "games_in_last_7_days": games_in_last_7,
        "days_rest": min(days_rest, 10),  # Cap at 10 (all-star break, etc.)
        "is_b2b": is_b2b,
        "is_3_in_4": is_3_in_4,
    }


# =============================================================================
# Calendar features
# =============================================================================

# Approximate season milestones (dates shift slightly year to year)
ALL_STAR_BREAK_APPROX = {
    "2023-24": "2024-02-16",
    "2024-25": "2025-02-14",
    "2025-26": "2026-02-13",
    "2026-27": "2027-02-12",
}

TRADE_DEADLINE_APPROX = {
    "2023-24": "2024-02-08",
    "2024-25": "2025-02-06",
    "2025-26": "2026-02-05",
    "2026-27": "2027-02-04",
}

SEASON_START_APPROX = {
    "2023-24": "2023-10-24",
    "2024-25": "2024-10-22",
    "2025-26": "2025-10-21",
    "2026-27": "2026-10-20",
}


def _date_to_season(date_str: str) -> str:
    """Map game date to season string."""
    parts = date_str.split("-")
    year = int(parts[0])
    month = int(parts[1])
    if month >= 10:
        return f"{year}-{str(year + 1)[-2:]}"
    else:
        return f"{year - 1}-{str(year)[-2:]}"


def get_calendar_features(game_date: str) -> Dict[str, float]:
    """Compute calendar/seasonal features for a game date.

    Args:
        game_date: ISO date string "YYYY-MM-DD"

    Returns:
        dict of 8 calendar features
    """
    dt = datetime.strptime(game_date, "%Y-%m-%d")
    season = _date_to_season(game_date)

    # Month of season: Oct=1, Nov=2, Dec=3, Jan=4, Feb=5, Mar=6, Apr=7
    month = dt.month
    if month >= 10:
        month_of_season = month - 9  # Oct=1, Nov=2, Dec=3
    else:
        month_of_season = month + 3  # Jan=4, Feb=5, Mar=6, Apr=7

    # Season game number (approximate)
    season_start_str = SEASON_START_APPROX.get(season, f"{season[:4]}-10-22")
    season_start = datetime.strptime(season_start_str, "%Y-%m-%d")
    days_into_season = max((dt - season_start).days, 0)
    # ~82 games over ~170 days = ~0.48 games/day
    season_game_number = int(days_into_season * 0.48)
    season_game_number = min(max(season_game_number, 1), 82)

    # October = preseason-ish (first ~2 weeks are regular season, but sample sizes thin)
    is_preseason_month = 1 if dt.month == 10 else 0

    # All-Star break
    asg_str = ALL_STAR_BREAK_APPROX.get(season, f"{int(season[:4]) + 1}-02-15")
    asg_date = datetime.strptime(asg_str, "%Y-%m-%d")
    is_pre_allstar = 1 if dt < asg_date else 0
    is_post_allstar = 1 if dt >= asg_date else 0

    # Trade deadline
    td_str = TRADE_DEADLINE_APPROX.get(season, f"{int(season[:4]) + 1}-02-06")
    td_date = datetime.strptime(td_str, "%Y-%m-%d")
    is_pre_trade_deadline = 1 if dt < td_date else 0

    # Day of week (0=Monday, 6=Sunday)
    day_of_week = dt.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0  # Sat=5, Sun=6

    return {
        "month_of_season": month_of_season,
        "season_game_number": season_game_number,
        "is_preseason_month": is_preseason_month,
        "is_pre_allstar": is_pre_allstar,
        "is_post_allstar": is_post_allstar,
        "is_pre_trade_deadline": is_pre_trade_deadline,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
    }


# =============================================================================
# Build team schedules from training data
# =============================================================================

def build_team_schedules(
    games: List[Dict],
) -> Dict[str, List[Dict]]:
    """Reconstruct per-team schedules from the game list.

    Each game in the input has: date, home_team, away_team, game_id.
    For each team, builds a chronological list of games with location info.

    Args:
        games: list of game dicts from backtest JSON

    Returns:
        dict mapping team_name -> sorted list of schedule entries:
            [{"date", "opponent", "is_home", "game_location_team", "game_id"}, ...]
    """
    schedules = defaultdict(list)

    for game in games:
        date = game["date"]
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        game_id = game.get("game_id")

        if home:
            schedules[home].append({
                "date": date,
                "opponent": away,
                "is_home": True,
                "game_location_team": home,
                "game_id": game_id,
            })
        if away:
            schedules[away].append({
                "date": date,
                "opponent": home,
                "is_home": False,
                "game_location_team": home,  # Away team plays at home team's arena
                "game_id": game_id,
            })

    # Sort each team's schedule by date
    for team in schedules:
        schedules[team].sort(key=lambda g: g["date"])

    return dict(schedules)


# =============================================================================
# Feature names (for use by enrich_training_data and train modules)
# =============================================================================

TRAVEL_FEATURES = [
    "travel_miles",
    "timezone_delta",
    "altitude_ft",
    "altitude_delta",
    "is_denver_altitude",
    "road_trip_length",
    "games_in_last_5_days",
    "games_in_last_7_days",
    "days_rest",
    "is_b2b",
    "is_3_in_4",
]

CALENDAR_FEATURES = [
    "month_of_season",
    "season_game_number",
    "is_preseason_month",
    "is_pre_allstar",
    "is_post_allstar",
    "is_pre_trade_deadline",
    "day_of_week",
    "is_weekend",
]


# =============================================================================
# CLI: demo with training data
# =============================================================================

def main():
    print("=" * 70)
    print("Travel & Calendar Features - Demo")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load training data and build team schedules
    # ------------------------------------------------------------------
    game_data_files = [
        OUTPUTS / "backtest_2023_24_data.json",
        OUTPUTS / "backtest_oos_data.json",
    ]

    all_games = []
    for path in game_data_files:
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
    logger.info("Total games: %d", len(all_games))

    if not all_games:
        print("No game data found. Run backtest data collection first.")
        return

    schedules = build_team_schedules(all_games)
    logger.info("Built schedules for %d teams", len(schedules))

    # ------------------------------------------------------------------
    # 2. Sample travel features for a few games
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("SAMPLE TRAVEL FEATURES")
    print(f"{'='*70}")

    # Pick 5 games spread across the dataset
    sample_indices = np.linspace(0, len(all_games) - 1, 5, dtype=int)
    for idx in sample_indices:
        game = all_games[idx]
        home = game["home_team"]
        away = game["away_team"]
        date = game["date"]

        print(f"\n  {date}: {away} @ {home}")

        # Away team travel features (more interesting — they're traveling)
        away_schedule = schedules.get(away, [])
        if away_schedule:
            feats = get_travel_features(
                team_name=away,
                opponent_name=home,
                game_date=date,
                is_home=False,
                schedule=away_schedule,
            )
            print(f"    {away} (away):")
            for k, v in feats.items():
                print(f"      {k:25s} = {v}")

        # Home team features
        home_schedule = schedules.get(home, [])
        if home_schedule:
            feats = get_travel_features(
                team_name=home,
                opponent_name=away,
                game_date=date,
                is_home=True,
                schedule=home_schedule,
            )
            print(f"    {home} (home):")
            print(f"      travel_miles            = {feats['travel_miles']}")
            print(f"      days_rest               = {feats['days_rest']}")
            print(f"      is_b2b                  = {feats['is_b2b']}")

    # ------------------------------------------------------------------
    # 3. Calendar features for a range of dates
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("SAMPLE CALENDAR FEATURES")
    print(f"{'='*70}")

    sample_dates = [
        "2023-10-25",  # Early season
        "2023-12-25",  # Christmas
        "2024-01-15",  # Mid-season
        "2024-02-10",  # Near All-Star / trade deadline
        "2024-03-20",  # Late season
    ]

    for date in sample_dates:
        feats = get_calendar_features(date)
        print(f"\n  {date}:")
        for k, v in feats.items():
            print(f"    {k:25s} = {v}")

    # ------------------------------------------------------------------
    # 4. Haversine sanity checks
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("DISTANCE SANITY CHECKS")
    print(f"{'='*70}")

    test_routes = [
        ("Boston Celtics", "Brooklyn Nets"),
        ("Los Angeles Lakers", "Golden State Warriors"),
        ("Miami Heat", "Portland Trail Blazers"),
        ("Denver Nuggets", "Utah Jazz"),
        ("New York Knicks", "LA Clippers"),
    ]

    for team_a, team_b in test_routes:
        a = NBA_ARENAS[team_a]
        b = NBA_ARENAS[team_b]
        dist = haversine_miles(a["lat"], a["lon"], b["lat"], b["lon"])
        tz_delta = abs(TZ_OFFSETS[a["tz"]] - TZ_OFFSETS[b["tz"]])
        alt_delta = abs(a["altitude_ft"] - b["altitude_ft"])
        print(
            f"  {team_a:25s} -> {team_b:25s}: "
            f"{dist:7.0f} mi, {tz_delta}h TZ, {alt_delta:.0f}ft alt"
        )

    # ------------------------------------------------------------------
    # 5. Aggregate stats across all games
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("AGGREGATE TRAVEL STATS ACROSS ALL GAMES")
    print(f"{'='*70}")

    all_travel_miles = []
    all_days_rest = []
    b2b_count = 0
    three_in_four_count = 0
    total_team_games = 0

    for team, sched in schedules.items():
        for i, game_entry in enumerate(sched):
            if i == 0:
                continue  # Skip first game (no prior game)
            total_team_games += 1

            feats = get_travel_features(
                team_name=team,
                opponent_name=game_entry["opponent"],
                game_date=game_entry["date"],
                is_home=game_entry["is_home"],
                schedule=sched,
            )
            all_travel_miles.append(feats["travel_miles"])
            all_days_rest.append(feats["days_rest"])
            b2b_count += feats["is_b2b"]
            three_in_four_count += feats["is_3_in_4"]

    if all_travel_miles:
        print(f"  Total team-games analyzed: {total_team_games:,}")
        print(f"  Avg travel miles:     {np.mean(all_travel_miles):.0f}")
        print(f"  Median travel miles:  {np.median(all_travel_miles):.0f}")
        print(f"  Max travel miles:     {max(all_travel_miles):.0f}")
        print(f"  Avg days rest:        {np.mean(all_days_rest):.2f}")
        print(f"  Back-to-back games:   {b2b_count} ({100*b2b_count/total_team_games:.1f}%)")
        print(f"  3-in-4 games:         {three_in_four_count} ({100*three_in_four_count/total_team_games:.1f}%)")

    print()


if __name__ == "__main__":
    main()
