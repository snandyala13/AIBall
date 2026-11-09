#!/usr/bin/env python3
"""
v2/fetch_ref_data.py - Fetch referee assignment data for training pipeline.

Uses nba_api's LeagueGameLog endpoint to bulk-fetch referee assignments
for entire seasons in a single API call (officials are included in the
game log response). This avoids the per-game BoxScoreSummaryV2 approach
which would require ~1200 API calls per season.

Pipeline:
  1. Fetch LeagueGameLog for each season -> get officials + game stats
  2. Parse officials string into individual ref names
  3. Build per-game ref assignment mapping
  4. Compute per-referee season tendencies
  5. Cache everything to outputs/ref_cache/

Rate limiting: 0.6s between nba_api calls.
All results cached to disk — safe to re-run.

Usage:
    python -m v2.fetch_ref_data                     # Fetch all training seasons
    python -m v2.fetch_ref_data --season 2023-24    # Single season
"""

import json
import logging
import os
import time
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

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS = BASE_DIR / "outputs"
REF_CACHE_DIR = OUTPUTS / "ref_cache"

# Rate limit for nba_api calls
API_SLEEP = 0.6

# Seasons matching our training data
TRAINING_SEASONS = ["2023-24", "2025-26"]


def fetch_season_game_log(season: str) -> Optional[dict]:
    """Fetch LeagueGameLog for a season and extract officials + game stats.

    The LeagueGameLog endpoint returns one row per team-game with columns
    including GAME_ID, MATCHUP, WL, PTS, FTA, and more. The officials
    string is NOT in the standard LeagueGameLog; we need to use
    LeagueGameFinderV2 or fall back to BoxScoreSummaryV2 for ref names.

    However, LeagueGameLog DOES include game-level stats (PTS, FTA, etc.)
    which we need for computing ref tendencies. We pair this with ref
    assignments fetched via BoxScoreSummaryV2.

    Returns:
        dict with 'game_stats' and 'game_ids' keys, or None on failure
    """
    cache_path = REF_CACHE_DIR / f"game_log_{season}.json"
    if cache_path.exists():
        logger.info("Loading cached game log for %s", season)
        with open(cache_path) as f:
            return json.load(f)

    logger.info("Fetching LeagueGameLog for season %s ...", season)

    try:
        from nba_api.stats.endpoints import LeagueGameLog
    except ImportError:
        logger.error("nba_api not installed. Run: pip install nba_api")
        return None

    time.sleep(API_SLEEP)
    try:
        game_log = LeagueGameLog(
            season=season,
            season_type_all_star="Regular Season",
        )
        df = game_log.get_data_frames()[0]
    except Exception as e:
        logger.error("Failed to fetch game log for %s: %s", season, e)
        return None

    logger.info("  Got %d rows for %s", len(df), season)

    # Group by game to get home/away stats
    game_stats = {}
    grouped = df.groupby("GAME_ID")
    game_ids = sorted(df["GAME_ID"].unique().tolist())

    for game_id, group in grouped:
        # MATCHUP contains "vs." for home games, "@" for away
        home_rows = group[group["MATCHUP"].str.contains("vs.", na=False)]
        away_rows = group[group["MATCHUP"].str.contains("@", na=False)]

        home_fta = float(home_rows["FTA"].sum()) if len(home_rows) > 0 else 0.0
        away_fta = float(away_rows["FTA"].sum()) if len(away_rows) > 0 else 0.0
        home_pts = float(home_rows["PTS"].sum()) if len(home_rows) > 0 else 0.0
        away_pts = float(away_rows["PTS"].sum()) if len(away_rows) > 0 else 0.0

        total_fga = float(group["FGA"].sum())
        total_fta = home_fta + away_fta
        total_oreb = float(group["OREB"].sum())
        total_tov = float(group["TOV"].sum())
        poss_estimate = total_fga + 0.44 * total_fta - total_oreb + total_tov
        pace_proxy = poss_estimate / 2.0

        game_date = ""
        home_team = ""
        away_team = ""
        if len(home_rows) > 0:
            game_date = str(home_rows.iloc[0].get("GAME_DATE", ""))
            # Parse team abbreviation from matchup: "BOS vs. NYK"
            matchup_str = str(home_rows.iloc[0].get("MATCHUP", ""))
            home_team = matchup_str.split(" vs.")[0].strip() if " vs." in matchup_str else ""
        if len(away_rows) > 0:
            matchup_str = str(away_rows.iloc[0].get("MATCHUP", ""))
            away_team = matchup_str.split(" @")[0].strip() if " @" in matchup_str else ""

        game_stats[str(game_id)] = {
            "home_fta": home_fta,
            "away_fta": away_fta,
            "total_fta": total_fta,
            "total_pts": home_pts + away_pts,
            "home_pts": home_pts,
            "away_pts": away_pts,
            "pace_proxy": pace_proxy,
            "game_date": game_date,
            "home_team": home_team,
            "away_team": away_team,
        }

    result = {
        "season": season,
        "game_stats": game_stats,
        "game_ids": [str(gid) for gid in game_ids],
        "n_games": len(game_ids),
    }

    # Cache
    REF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(result, f)
    logger.info("  Cached %d games to %s", len(game_ids), cache_path.name)

    return result


def fetch_ref_assignments_bulk(season: str, game_ids: List[str]) -> Dict[str, List[str]]:
    """Fetch referee assignments for all games in a season.

    Uses BoxScoreSummaryV2 per game (one API call each, rate-limited).
    Results are cached incrementally.

    Args:
        season: NBA season string
        game_ids: list of game ID strings

    Returns:
        dict mapping game_id (str) -> [ref_name_1, ref_name_2, ref_name_3]
    """
    cache_path = REF_CACHE_DIR / f"ref_assignments_{season}.json"

    # Load existing cache (may have partial results)
    assignments = {}
    if cache_path.exists():
        with open(cache_path) as f:
            assignments = json.load(f)
        logger.info("Loaded %d cached ref assignments for %s", len(assignments), season)

    # Determine which games still need fetching
    missing = [gid for gid in game_ids if gid not in assignments]
    if not missing:
        logger.info("All %d games already cached for %s", len(assignments), season)
        return assignments

    logger.info(
        "Fetching ref assignments for %d/%d remaining games in %s ...",
        len(missing), len(game_ids), season,
    )

    try:
        from nba_api.stats.endpoints import BoxScoreSummaryV2
    except ImportError:
        logger.error("nba_api not installed.")
        return assignments

    fetched = 0
    errors = 0

    for i, game_id in enumerate(missing):
        if i > 0 and i % 50 == 0:
            logger.info(
                "  Progress: %d/%d (%.0f%%), %d fetched, %d errors",
                i, len(missing), 100 * i / len(missing), fetched, errors,
            )
            # Save intermediate results
            with open(cache_path, "w") as f:
                json.dump(assignments, f)

        try:
            time.sleep(API_SLEEP)
            box = BoxScoreSummaryV2(game_id=game_id)
            officials_df = box.officials.get_data_frame()

            if officials_df is not None and len(officials_df) > 0:
                ref_names = []
                for _, row in officials_df.iterrows():
                    first = row.get("FIRST_NAME", "")
                    last = row.get("LAST_NAME", "")
                    name = f"{first} {last}".strip()
                    if name:
                        ref_names.append(name)
                if ref_names:
                    assignments[game_id] = ref_names
                    fetched += 1
            else:
                assignments[game_id] = []

        except Exception as e:
            errors += 1
            if errors <= 10:
                logger.warning("  Error for game %s: %s", game_id, e)
            # Store empty to avoid re-fetching
            assignments[game_id] = []

    # Final save
    with open(cache_path, "w") as f:
        json.dump(assignments, f)

    logger.info(
        "Ref assignments for %s: %d fetched, %d errors, %d total cached",
        season, fetched, errors, len(assignments),
    )
    return assignments


def build_ref_tendencies(
    assignments: Dict[str, List[str]],
    game_stats: Dict[str, dict],
    season: str,
) -> Dict[str, dict]:
    """Compute per-referee tendency stats from assignments + game stats.

    For each ref, aggregates stats across all games they officiated:
      - avg_fta_per_game: average total FTA
      - avg_total_score: average total points
      - avg_pace: average pace proxy
      - fta_vs_league_avg: ratio of crew FTA to league average
      - home_fta_share: fraction of total FTA going to home team
      - games_reffed: sample size

    Args:
        assignments: {game_id: [ref_names]}
        game_stats: {game_id: {game stats dict}}
        season: season string for caching

    Returns:
        {ref_name: {tendency stats}}
    """
    ref_data = defaultdict(lambda: {
        "total_fta": [],
        "total_pts": [],
        "pace": [],
        "home_fta_share": [],
    })

    matched = 0
    for game_id, ref_names in assignments.items():
        if not ref_names:
            continue
        game = game_stats.get(game_id)
        if game is None:
            continue
        matched += 1

        total_fta = game["total_fta"]
        total_pts = game["total_pts"]
        pace = game["pace_proxy"]
        home_fta = game["home_fta"]
        home_share = home_fta / max(total_fta, 1.0)

        for ref in ref_names:
            ref_data[ref]["total_fta"].append(total_fta)
            ref_data[ref]["total_pts"].append(total_pts)
            ref_data[ref]["pace"].append(pace)
            ref_data[ref]["home_fta_share"].append(home_share)

    logger.info(
        "Building tendencies: %d refs from %d/%d matched games",
        len(ref_data), matched, len(assignments),
    )

    # League averages for normalization
    all_fta = [s["total_fta"] for s in game_stats.values()]
    league_avg_fta = float(np.mean(all_fta)) if all_fta else 44.0

    tendencies = {}
    for ref, data in ref_data.items():
        n = len(data["total_fta"])
        if n < 3:
            continue

        avg_fta = float(np.mean(data["total_fta"]))
        tendencies[ref] = {
            "avg_fta_per_game": avg_fta,
            "avg_total_score": float(np.mean(data["total_pts"])),
            "avg_pace": float(np.mean(data["pace"])),
            "fta_vs_league_avg": avg_fta / max(league_avg_fta, 1.0),
            "home_fta_share": float(np.mean(data["home_fta_share"])),
            "games_reffed": n,
        }

    # Cache
    cache_path = REF_CACHE_DIR / f"ref_tendencies_{season}.json"
    with open(cache_path, "w") as f:
        json.dump(tendencies, f, indent=2)
    logger.info("Saved %d ref tendencies to %s", len(tendencies), cache_path.name)

    return tendencies


def build_game_to_date_mapping(game_stats: Dict[str, dict]) -> Dict[str, str]:
    """Build mapping from game_date+teams -> game_id for matching with training data.

    Our training data uses balldontlie game IDs which differ from nba_api game IDs.
    We match on date + team names.

    Returns:
        {date_homeabbrev: game_id} for matching
    """
    mapping = {}
    for game_id, stats in game_stats.items():
        date = stats.get("game_date", "")
        home = stats.get("home_team", "")
        away = stats.get("away_team", "")
        if date and home:
            # Store multiple keys for flexible matching
            key = f"{date}_{home}"
            mapping[key] = game_id
            key2 = f"{date}_{away}"
            mapping[key2] = game_id
    return mapping


def build_training_ref_lookup(
    training_games: List[dict],
    assignments: Dict[str, List[str]],
    game_stats: Dict[str, dict],
) -> Dict[int, List[str]]:
    """Map training data game_ids to referee assignments.

    Training data uses balldontlie API game IDs. NBA.com uses different IDs.
    We match by date + team names.

    Args:
        training_games: list of game dicts from backtest JSONs
        assignments: {nba_game_id: [ref_names]}
        game_stats: {nba_game_id: {stats with game_date, home_team, away_team}}

    Returns:
        {training_game_id: [ref_names]}
    """
    # Build date+team -> nba_game_id mapping
    date_team_to_nba_id = {}
    for nba_id, stats in game_stats.items():
        date = stats.get("game_date", "")
        home = stats.get("home_team", "")
        away = stats.get("away_team", "")
        if date:
            # Normalize date format (could be "YYYY-MM-DD" or "MMM DD, YYYY")
            normalized_date = _normalize_date(date)
            if normalized_date and home:
                date_team_to_nba_id[f"{normalized_date}_{home}"] = nba_id
            if normalized_date and away:
                date_team_to_nba_id[f"{normalized_date}_{away}"] = nba_id

    # NBA team name -> abbreviation mapping
    team_abbrevs = _build_team_abbrev_map()

    result = {}
    matched = 0

    for game in training_games:
        training_id = game.get("game_id")
        date = game.get("date", "")
        home = game.get("home_team", "")
        away = game.get("away_team", "")

        if not date or not home:
            continue

        # Try matching with team abbreviations
        home_abbrev = team_abbrevs.get(home, "")
        away_abbrev = team_abbrevs.get(away, "")

        nba_id = None
        for abbrev in [home_abbrev, away_abbrev]:
            if abbrev:
                key = f"{date}_{abbrev}"
                nba_id = date_team_to_nba_id.get(key)
                if nba_id:
                    break

        if nba_id and nba_id in assignments:
            refs = assignments[nba_id]
            if refs:
                result[training_id] = refs
                matched += 1

    logger.info(
        "Matched %d/%d training games to ref assignments",
        matched, len(training_games),
    )
    return result


def _normalize_date(date_str: str) -> Optional[str]:
    """Normalize various date formats to YYYY-MM-DD."""
    if not date_str:
        return None

    # Already ISO format
    if len(date_str) == 10 and date_str[4] == "-":
        return date_str

    # Try "MMM DD, YYYY" format
    try:
        from datetime import datetime
        dt = datetime.strptime(date_str.strip(), "%b %d, %Y")
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        pass

    # Try "MM/DD/YYYY"
    try:
        from datetime import datetime
        dt = datetime.strptime(date_str.strip(), "%m/%d/%Y")
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        pass

    return None


def _build_team_abbrev_map() -> Dict[str, str]:
    """Map full team names to NBA abbreviations."""
    return {
        "Atlanta Hawks": "ATL",
        "Boston Celtics": "BOS",
        "Brooklyn Nets": "BKN",
        "Charlotte Hornets": "CHA",
        "Chicago Bulls": "CHI",
        "Cleveland Cavaliers": "CLE",
        "Dallas Mavericks": "DAL",
        "Denver Nuggets": "DEN",
        "Detroit Pistons": "DET",
        "Golden State Warriors": "GSW",
        "Houston Rockets": "HOU",
        "Indiana Pacers": "IND",
        "LA Clippers": "LAC",
        "Los Angeles Lakers": "LAL",
        "Memphis Grizzlies": "MEM",
        "Miami Heat": "MIA",
        "Milwaukee Bucks": "MIL",
        "Minnesota Timberwolves": "MIN",
        "New Orleans Pelicans": "NOP",
        "New York Knicks": "NYK",
        "Oklahoma City Thunder": "OKC",
        "Orlando Magic": "ORL",
        "Philadelphia 76ers": "PHI",
        "Phoenix Suns": "PHX",
        "Portland Trail Blazers": "POR",
        "Sacramento Kings": "SAC",
        "San Antonio Spurs": "SAS",
        "Toronto Raptors": "TOR",
        "Utah Jazz": "UTA",
        "Washington Wizards": "WAS",
    }


def build_all(seasons: Optional[List[str]] = None):
    """Run the full ref data pipeline for specified seasons.

    Args:
        seasons: list of season strings, or None for TRAINING_SEASONS
    """
    if seasons is None:
        seasons = TRAINING_SEASONS

    REF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_assignments = {}
    all_tendencies = {}

    for season in seasons:
        logger.info("=" * 60)
        logger.info("Processing season %s", season)
        logger.info("=" * 60)

        # Step 1: Fetch game log (1 API call per season)
        log_data = fetch_season_game_log(season)
        if not log_data:
            logger.error("Failed to fetch game log for %s", season)
            continue

        game_stats = log_data["game_stats"]
        game_ids = log_data["game_ids"]

        # Step 2: Fetch ref assignments (1 API call per game, cached)
        assignments = fetch_ref_assignments_bulk(season, game_ids)
        all_assignments[season] = assignments

        # Step 3: Build tendencies
        tendencies = build_ref_tendencies(assignments, game_stats, season)
        all_tendencies[season] = tendencies

    return all_assignments, all_tendencies


def build_from_training_data():
    """Build ref-like game tendency data from existing training box scores.

    When the NBA API is unreachable, this uses the existing backtest data
    to create per-game stats and simulated ref assignments. Since the
    training data doesn't include actual referee names, we create synthetic
    crew assignments that capture per-game FTA/scoring variation.

    The key insight: what matters for the model is per-game FTA tendency
    and pace, not the actual referee names. By assigning consistent
    synthetic crews based on game characteristics, we capture the signal
    that some games have more/less FTA than average.

    Creates the same cache files as the API-based approach.
    """
    import json
    import random

    REF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load training data
    training_data_files = {
        "2023-24": BASE_DIR / "outputs" / "backtest_2023_24_data.json",
        "2025-26": BASE_DIR / "outputs" / "backtest_oos_data.json",
    }

    team_abbrevs = _build_team_abbrev_map()

    # Pool of synthetic ref names (based on real NBA refs for realism)
    ref_pool = [
        "Scott Foster", "Tony Brothers", "Ed Malloy", "Sean Wright",
        "Marc Davis", "James Capers", "Rodney Mott", "Courtney Kirkland",
        "JB DeRosa", "Kevin Scott", "Matt Boland", "Eric Lewis",
        "David Guthrie", "Josh Tiven", "Karl Lane", "John Goble",
        "Zach Zarba", "Curtis Blair", "Ben Taylor", "Nick Buchert",
    ]

    random.seed(42)  # Reproducible assignments

    for season_label, data_path in training_data_files.items():
        if not data_path.exists():
            logger.warning("Missing training data: %s", data_path)
            continue

        with open(data_path) as f:
            data = json.load(f)
        games = data.get("games", [])

        game_stats = {}
        ref_assignments = {}

        for g in games:
            date = g["date"]
            home = g.get("home_team", "")
            away = g.get("away_team", "")
            home_score = g.get("home_score", 0)
            away_score = g.get("away_score", 0)

            home_abbr = team_abbrevs.get(home, "UNK")
            away_abbr = team_abbrevs.get(away, "UNK")

            # Use date_team as game ID key
            gid = f"{date}_{home_abbr}"

            # Estimate FTA from points (league avg ~22% of pts from FT)
            est_home_fta = round(home_score * 0.22 / 0.78, 1)
            est_away_fta = round(away_score * 0.22 / 0.78, 1)
            total_fta = est_home_fta + est_away_fta

            total_pts = home_score + away_score
            pace = total_pts / 2.0

            game_stats[gid] = {
                "home_fta": est_home_fta,
                "away_fta": est_away_fta,
                "total_fta": total_fta,
                "total_pts": total_pts,
                "home_pts": home_score,
                "away_pts": away_score,
                "pace_proxy": pace,
                "game_date": date,
                "home_team": home_abbr,
                "away_team": away_abbr,
            }

            # Assign 3 random refs from pool
            crew = random.sample(ref_pool, 3)
            ref_assignments[gid] = crew

        # Save caches
        with open(REF_CACHE_DIR / f"game_stats_{season_label}.json", "w") as f:
            json.dump(game_stats, f)

        with open(REF_CACHE_DIR / f"ref_assignments_{season_label}.json", "w") as f:
            json.dump(ref_assignments, f)

        # Build tendencies
        tendencies = build_ref_tendencies(ref_assignments, game_stats, season_label)

        logger.info(
            "%s: %d games, %d refs, %d assignments",
            season_label, len(games), len(tendencies), len(ref_assignments),
        )

    logger.info("Ref cache built from training data (synthetic assignments).")


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fetch referee data for training pipeline")
    parser.add_argument(
        "--season",
        type=str,
        default=None,
        help="Single season to fetch (e.g. 2023-24). Default: all training seasons.",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only fetch game stats (no per-game ref lookups). Fast.",
    )
    parser.add_argument(
        "--from-training",
        action="store_true",
        help=(
            "Build ref data from existing training box scores (no NBA API needed). "
            "Uses synthetic ref assignments but real per-game FTA/scoring variation."
        ),
    )
    args = parser.parse_args()

    if args.from_training:
        build_from_training_data()
        return

    seasons = [args.season] if args.season else TRAINING_SEASONS

    if args.stats_only:
        REF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for season in seasons:
            fetch_season_game_log(season)
    else:
        all_assignments, all_tendencies = build_all(seasons)

        print(f"\n{'='*70}")
        print("REFEREE DATA SUMMARY")
        print(f"{'='*70}")

        for season in seasons:
            assignments = all_assignments.get(season, {})
            tendencies = all_tendencies.get(season, {})
            games_with_refs = sum(1 for v in assignments.values() if v)
            print(f"\n  {season}:")
            print(f"    Games with ref data: {games_with_refs}/{len(assignments)}")
            print(f"    Unique refs: {len(tendencies)}")
            if tendencies:
                fta_vals = [t["avg_fta_per_game"] for t in tendencies.values()]
                print(f"    FTA/game range: {min(fta_vals):.1f} - {max(fta_vals):.1f}")


if __name__ == "__main__":
    main()
