#!/usr/bin/env python3
"""
fetch_backtest_data.py - Fetch Out-of-Sample NBA Data for Backtesting

Fetches all NBA game data from Jan 5 2026 through Jun 15 2026:
  1. Box scores from Balldontlie GOAT API (game results + player stats)
  2. Historical DraftKings prop lines from PropLine API

Outputs:
  outputs/backtest_oos_data.json   - Full game data with box scores + props
  outputs/backtest_oos_legs.json   - Flat leg format for backtest engine

Usage:
    python fetch_backtest_data.py                           # Full date range
    python fetch_backtest_data.py --start-date 2026-03-01   # Custom start
    python fetch_backtest_data.py --end-date 2026-04-01     # Custom end
    python fetch_backtest_data.py --resume                  # Resume from last saved date
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIG
# =============================================================================

BDL_API_KEY = "d2df7f14-636a-4ff2-8005-35e768d1745f"
BDL_BASE = "https://api.balldontlie.io/v1"
BDL_HEADERS = {"Authorization": BDL_API_KEY}

PROPLINE_API_KEY = os.environ.get("PROPLINE_API_KEY", "")
PROPLINE_BASE = "https://api.prop-line.com/v1/sports/basketball_nba"

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4/historical/sports/basketball_nba"

DEFAULT_START = "2026-01-05"
DEFAULT_END = "2026-06-15"

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
DATA_FILE = os.path.join(OUTPUT_DIR, "backtest_oos_data.json")
LEGS_FILE = os.path.join(OUTPUT_DIR, "backtest_oos_legs.json")

# Rate limits: sleep between API calls
BDL_DELAY = 0.5     # 2 req/sec for Balldontlie
PROPLINE_DELAY = 0.5  # 2 req/sec (conservative for free tier)
ODDS_API_DELAY = 0.3  # ~3 req/sec for The Odds API

MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # seconds, multiplied by attempt number

# PropLine market keys we want
PROPLINE_MARKETS = [
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_threes",
    "player_points_rebounds_assists",
    "player_steals",
    "player_blocks",
]

# Map PropLine market key -> our internal prop_type
MARKET_TO_PROP = {
    "player_points": "points",
    "player_rebounds": "rebounds",
    "player_assists": "assists",
    "player_threes": "threes",
    "player_points_rebounds_assists": "pts_rebs_asts",
    "player_steals": "steals",
    "player_blocks": "blocks",
}

# Map our prop_type -> stat keys to sum for combined props
PROP_STAT_KEYS = {
    "points": ["pts"],
    "rebounds": ["reb"],
    "assists": ["ast"],
    "threes": ["fg3m"],
    "steals": ["stl"],
    "blocks": ["blk"],
    "pts_rebs_asts": ["pts", "reb", "ast"],
}


# =============================================================================
# API HELPERS
# =============================================================================

def _request_with_retry(method, url, **kwargs):
    """Make an HTTP request with retries and exponential backoff."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.request(method, url, timeout=30, **kwargs)
            if resp.status_code == 429:
                wait = RETRY_BACKOFF * attempt * 2
                logger.warning("Rate limited (429). Waiting %.1fs...", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF * attempt
                logger.warning("HTTP %s (attempt %d/%d). Retrying in %.1fs...",
                               e.response.status_code if e.response else "?",
                               attempt, MAX_RETRIES, wait)
                time.sleep(wait)
            else:
                logger.error("Request failed after %d attempts: %s", MAX_RETRIES, e)
                return None
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF * attempt
                logger.warning("Request error (attempt %d/%d): %s. Retrying in %.1fs...",
                               attempt, MAX_RETRIES, e, wait)
                time.sleep(wait)
            else:
                logger.error("Request failed after %d attempts: %s", MAX_RETRIES, e)
                return None
    return None


# =============================================================================
# BALLDONTLIE API
# =============================================================================

def bdl_get_games_for_date(date_str: str) -> List[Dict]:
    """Fetch all games for a date from Balldontlie. Returns list of game dicts."""
    url = f"{BDL_BASE}/games"
    data = _request_with_retry("GET", url, headers=BDL_HEADERS,
                               params={"dates[]": date_str, "per_page": 50})
    time.sleep(BDL_DELAY)
    if data and data.get("data"):
        return data["data"]
    return []


def bdl_get_game_stats(game_id: int) -> List[Dict]:
    """Fetch player box score stats for a game. Returns list of player stat dicts."""
    url = f"{BDL_BASE}/stats"
    all_stats = []
    cursor = None

    while True:
        params = {"game_ids[]": game_id, "per_page": 100}
        if cursor:
            params["cursor"] = cursor

        data = _request_with_retry("GET", url, headers=BDL_HEADERS, params=params)
        time.sleep(BDL_DELAY)

        if not data or not data.get("data"):
            break

        all_stats.extend(data["data"])

        # Check for pagination
        meta = data.get("meta", {})
        cursor = meta.get("next_cursor")
        if not cursor:
            break

    return all_stats


def parse_minutes(min_str) -> float:
    """Parse minutes string to decimal. Handles 'MM:SS', 'MM', int, float, None."""
    if min_str is None or min_str == "" or min_str == "0:00" or min_str == "00:00":
        return 0.0
    if isinstance(min_str, (int, float)):
        return float(min_str)
    min_str = str(min_str).strip()
    if ":" in min_str:
        parts = min_str.split(":")
        try:
            minutes = int(parts[0])
            seconds = int(parts[1]) if len(parts) > 1 else 0
            return round(minutes + seconds / 60.0, 1)
        except (ValueError, IndexError):
            return 0.0
    try:
        return float(min_str)
    except ValueError:
        return 0.0


def build_box_score(stats: List[Dict]) -> Dict[str, Dict]:
    """Convert raw BDL stats list into {player_name: {stat_dict}}."""
    box = {}
    for s in stats:
        player = s.get("player", {})
        first = player.get("first_name", "")
        last = player.get("last_name", "")
        name = f"{first} {last}".strip()
        if not name:
            continue

        mins = parse_minutes(s.get("min"))
        # Skip players with 0 minutes (DNPs)
        if mins <= 0:
            continue

        box[name] = {
            "pts": s.get("pts", 0) or 0,
            "reb": s.get("reb", 0) or 0,
            "ast": s.get("ast", 0) or 0,
            "fg3m": s.get("fg3m", 0) or 0,
            "stl": s.get("stl", 0) or 0,
            "blk": s.get("blk", 0) or 0,
            "turnover": s.get("turnover", 0) or 0,
            "min": mins,
        }
    return box


def get_team_full_name(team_data: Dict) -> str:
    """Extract full team name from BDL team object."""
    return team_data.get("full_name", f"{team_data.get('city', '')} {team_data.get('name', '')}").strip()


# =============================================================================
# THE ODDS API (Historical) — PRIMARY SOURCE FOR BACKTEST PROPS
# =============================================================================

ODDS_MARKETS = "player_points,player_rebounds,player_assists,player_threes,player_points_rebounds_assists"

ODDS_MARKET_TO_PROP = {
    "player_points": "points",
    "player_rebounds": "rebounds",
    "player_assists": "assists",
    "player_threes": "threes",
    "player_points_rebounds_assists": "pts_rebs_asts",
}


def odds_api_get_events(date_str: str) -> List[Dict]:
    """Get historical NBA events for a date from The Odds API."""
    url = f"{ODDS_API_BASE}/events"
    params = {
        "apiKey": ODDS_API_KEY,
        "date": f"{date_str}T18:00:00Z",
    }
    data = _request_with_retry("GET", url, params=params)
    time.sleep(ODDS_API_DELAY)
    if data and isinstance(data, dict):
        return data.get("data", [])
    return []


def odds_api_get_props(event_id: str, date_str: str) -> Dict[str, Dict]:
    """
    Get historical DraftKings player props for a single event.

    Returns: {player_name: {prop_type: {line, over_odds, under_odds}, ...}, ...}
    """
    url = f"{ODDS_API_BASE}/events/{event_id}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": ODDS_MARKETS,
        "bookmakers": "draftkings",
        "date": f"{date_str}T18:00:00Z",
        "oddsFormat": "american",
    }
    data = _request_with_retry("GET", url, params=params)
    time.sleep(ODDS_API_DELAY)

    if not data or not isinstance(data, dict):
        return {}

    event_data = data.get("data", data)  # Sometimes nested, sometimes not
    bookmakers = event_data.get("bookmakers", [])

    player_props = {}
    for bk in bookmakers:
        if bk.get("key") != "draftkings":
            continue
        for market in bk.get("markets", []):
            mkt_key = market.get("key", "")
            prop_type = ODDS_MARKET_TO_PROP.get(mkt_key)
            if not prop_type:
                continue

            # Group Over/Under by player
            player_lines = {}
            for outcome in market.get("outcomes", []):
                pname = outcome.get("description", "")
                direction = outcome.get("name", "")
                price = outcome.get("price", -110)
                point = outcome.get("point", 0)
                if not pname:
                    continue

                if pname not in player_lines:
                    player_lines[pname] = {"line": point}
                if direction == "Over":
                    player_lines[pname]["over_odds"] = price
                elif direction == "Under":
                    player_lines[pname]["under_odds"] = price

            for pname, ld in player_lines.items():
                if "over_odds" in ld and "under_odds" in ld:
                    if pname not in player_props:
                        player_props[pname] = {}
                    player_props[pname][prop_type] = {
                        "line": float(ld["line"]),
                        "over_odds": ld["over_odds"],
                        "under_odds": ld["under_odds"],
                    }

    return player_props


def odds_api_get_all_props_for_date(date_str: str) -> Dict[str, Dict[str, Dict]]:
    """
    Fetch all DK player props for all games on a date via The Odds API.

    Returns: {"away|home": {player: {prop_type: {line, over_odds, under_odds}}}}
    """
    events = odds_api_get_events(date_str)
    if not events:
        return {}

    result = {}
    for ev in events:
        eid = ev.get("id")
        home = ev.get("home_team", "")
        away = ev.get("away_team", "")
        if not eid or not home or not away:
            continue

        props = odds_api_get_props(eid, date_str)
        if props:
            key = f"{away}|{home}"
            result[key] = props
            logger.debug(f"  Odds API: {away} @ {home}: {sum(len(v) for v in props.values())} props")

    return result


# =============================================================================
# PROPLINE API (fallback — only works for current/recent games)
# =============================================================================

def propline_get_odds_for_date(date_str: str) -> List[Dict]:
    """
    Fetch all NBA prop odds for a date from PropLine.

    Uses: GET /v1/sports/basketball_nba/odds?date=YYYY-MM-DD&markets=...&bookmakers=draftkings
    """
    url = f"{PROPLINE_BASE}/odds"
    params = {
        "apiKey": PROPLINE_API_KEY,
        "date": date_str,
        "markets": ",".join(PROPLINE_MARKETS),
        "bookmakers": "draftkings",
    }
    data = _request_with_retry("GET", url, params=params)
    time.sleep(PROPLINE_DELAY)
    if data and isinstance(data, list):
        return data
    return []


def parse_propline_events(events: List[Dict], target_date: str) -> Dict[str, Dict[str, Dict]]:
    """
    Parse PropLine odds response into a dict keyed by team names.

    Returns: {
        "away_team|home_team": {
            "player_name": {
                "prop_type": {"line": X, "over_odds": Y, "under_odds": Z},
                ...
            },
            ...
        },
        ...
    }
    """
    games_props = {}

    for event in events:
        # Filter to target date if commence_time available
        commence = event.get("commence_time", "")
        if commence:
            event_date = commence[:10]
            # Allow same-day or next-day (UTC offset)
            target_dt = datetime.strptime(target_date, "%Y-%m-%d")
            try:
                event_dt = datetime.strptime(event_date, "%Y-%m-%d")
                if abs((event_dt - target_dt).days) > 1:
                    continue
            except ValueError:
                pass

        home = event.get("home_team", "")
        away = event.get("away_team", "")
        if not home or not away:
            continue

        game_key = f"{away}|{home}"

        # Parse bookmaker data
        player_props = {}
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                market_key = market.get("key", "")
                prop_type = MARKET_TO_PROP.get(market_key)
                if not prop_type:
                    continue

                # Group by player
                player_lines = {}
                for outcome in market.get("outcomes", []):
                    pname = outcome.get("description", "")
                    direction = outcome.get("name", "")
                    odds = outcome.get("price", -110)
                    line = outcome.get("point", 0)

                    if pname not in player_lines:
                        player_lines[pname] = {"line": line}
                    if direction == "Over":
                        player_lines[pname]["over_odds"] = odds
                    elif direction == "Under":
                        player_lines[pname]["under_odds"] = odds

                # Store complete pairs
                for pname, ld in player_lines.items():
                    if "over_odds" in ld and "under_odds" in ld:
                        if pname not in player_props:
                            player_props[pname] = {}
                        player_props[pname][prop_type] = {
                            "line": float(ld["line"]),
                            "over_odds": ld["over_odds"],
                            "under_odds": ld["under_odds"],
                        }

        if player_props:
            games_props[game_key] = player_props

    return games_props


def match_bdl_game_to_propline(home_team: str, away_team: str,
                                propline_data: Dict[str, Dict]) -> Optional[Dict[str, Dict]]:
    """
    Match a BDL game to PropLine props by team name similarity.
    Case-insensitive, handles common abbreviation differences.
    """
    def normalize(name: str) -> str:
        return (name.lower()
                .replace("la clippers", "los angeles clippers")
                .strip())

    home_n = normalize(home_team)
    away_n = normalize(away_team)

    # Try direct match
    for key, props in propline_data.items():
        parts = key.split("|")
        if len(parts) != 2:
            continue
        key_away = normalize(parts[0])
        key_home = normalize(parts[1])

        # Match home/away in either direction
        home_match = (home_n in key_home or key_home in home_n)
        away_match = (away_n in key_away or key_away in away_n)
        if home_match and away_match:
            return props

        # Flipped?
        home_match2 = (home_n in key_away or key_away in home_n)
        away_match2 = (away_n in key_home or key_home in away_n)
        if home_match2 and away_match2:
            return props

    # Fallback: match by any shared team keyword
    home_words = set(home_team.lower().split()) - {"the", "of"}
    away_words = set(away_team.lower().split()) - {"the", "of"}

    for key, props in propline_data.items():
        key_lower = key.lower()
        key_words = set(key_lower.replace("|", " ").split()) - {"the", "of"}
        # Need at least one keyword from each team
        home_hit = bool(home_words & key_words)
        away_hit = bool(away_words & key_words)
        if home_hit and away_hit:
            return props

    return None


# =============================================================================
# LEGS GENERATION
# =============================================================================

def generate_legs(game_record: Dict) -> List[Dict]:
    """
    Generate flat leg records from a game's box_score + props.

    Each leg = one player prop line matched to actual box score result.
    No projection/edge -- just raw facts for the backtest engine.
    """
    legs = []
    box = game_record.get("box_score", {})
    props = game_record.get("props", {})
    game_id = game_record.get("game_id")
    date = game_record.get("date")
    home = game_record.get("home_team", "")
    away = game_record.get("away_team", "")
    matchup = f"{away} @ {home}"

    for player_name, player_props in props.items():
        # Find this player in box score (case-insensitive)
        box_stats = None
        for bname, bstats in box.items():
            if bname.lower() == player_name.lower():
                box_stats = bstats
                break

        # Fuzzy match: last name
        if box_stats is None:
            pl_parts = player_name.lower().split()
            if pl_parts:
                pl_last = pl_parts[-1]
                for bname, bstats in box.items():
                    if pl_last in bname.lower().split():
                        box_stats = bstats
                        break

        if box_stats is None:
            # Player had props but no box score (injury scratch, etc.)
            continue

        for prop_type, prop_data in player_props.items():
            line = prop_data.get("line", 0)
            over_odds = prop_data.get("over_odds", -110)
            under_odds = prop_data.get("under_odds", -110)

            # Calculate actual stat value
            stat_keys = PROP_STAT_KEYS.get(prop_type)
            if not stat_keys:
                continue

            actual = sum(box_stats.get(k, 0) or 0 for k in stat_keys)

            # Default pick to OVER. The backtest engine will override with
            # the model's predicted direction and recalculate hit.
            # Here we just record the raw outcome for each direction.
            pick = "OVER"
            if actual > line:
                hit = True    # Over would have hit
            elif actual < line:
                hit = False   # Over would NOT have hit (under would)
            else:
                hit = False   # Push -- neither side wins

            legs.append({
                "player": player_name,
                "prop_type": prop_type,
                "line": line,
                "over_odds": over_odds,
                "under_odds": under_odds,
                "actual": actual,
                "pick": pick,
                "hit": hit,
                "date": date,
                "game_id": game_id,
                "matchup": matchup,
            })

    return legs


# =============================================================================
# PROGRESS / SAVE
# =============================================================================

def load_existing_data(filepath: str) -> Dict:
    """Load existing output file, or return empty structure."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def get_completed_dates(data: Optional[Dict]) -> set:
    """Extract set of already-fetched dates from existing data."""
    if not data or "games" not in data:
        return set()
    return {g["date"] for g in data["games"] if "date" in g}


def save_data(data: Dict, filepath: str):
    """Atomically save JSON data (write to tmp then rename)."""
    tmp = filepath + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, filepath)


# =============================================================================
# MAIN FETCH LOOP
# =============================================================================

def date_range(start_str: str, end_str: str) -> List[str]:
    """Generate list of date strings from start to end (inclusive)."""
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


def fetch_all(start_date: str, end_date: str, resume: bool = False):
    """Main fetch loop. Process dates in weekly batches."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load existing data if resuming
    existing = load_existing_data(DATA_FILE) if resume else None
    completed_dates = get_completed_dates(existing) if resume else set()

    if existing and resume:
        all_games = existing.get("games", [])
        logger.info("Resuming: %d games already fetched, %d dates completed",
                     len(all_games), len(completed_dates))
    else:
        all_games = []

    dates = date_range(start_date, end_date)
    total_dates = len(dates)
    dates_to_fetch = [d for d in dates if d not in completed_dates]

    if not dates_to_fetch:
        logger.info("All dates already fetched. Nothing to do.")
        return

    logger.info("=" * 60)
    logger.info("BACKTEST DATA FETCH: %s to %s", start_date, end_date)
    logger.info("Total dates: %d | Already done: %d | Remaining: %d",
                total_dates, total_dates - len(dates_to_fetch), len(dates_to_fetch))
    logger.info("=" * 60)

    total_games_fetched = len(all_games)
    total_box_scores = sum(len(g.get("box_score", {})) for g in all_games)
    total_prop_lines = sum(
        sum(len(pp) for pp in g.get("props", {}).values())
        for g in all_games
    )
    games_with_no_data = 0
    # Rough estimate: ~5 games/day for progress reporting
    estimated_total_games = len(dates_to_fetch) * 5

    # Process in weekly batches
    batch_size = 7
    for batch_start in range(0, len(dates_to_fetch), batch_size):
        batch_dates = dates_to_fetch[batch_start:batch_start + batch_size]
        batch_label = f"{batch_dates[0]} to {batch_dates[-1]}"
        logger.info("--- Batch: %s (%d dates) ---", batch_label, len(batch_dates))

        for date_str in batch_dates:
            # 1. Fetch games from Balldontlie
            bdl_games = bdl_get_games_for_date(date_str)
            if not bdl_games:
                logger.debug("No games on %s", date_str)
                continue

            # 2. Fetch historical DK odds from The Odds API (primary)
            odds_api_data = odds_api_get_all_props_for_date(date_str)
            # Fall back to PropLine if Odds API has nothing
            if not odds_api_data:
                propline_events = propline_get_odds_for_date(date_str)
                propline_data = parse_propline_events(propline_events, date_str)
            else:
                propline_data = odds_api_data

            prop_matches = 0

            for game in bdl_games:
                game_id = game.get("id")
                home_team_data = game.get("home_team", {})
                away_team_data = game.get("visitor_team", {})
                home_team = get_team_full_name(home_team_data)
                away_team = get_team_full_name(away_team_data)

                # Check if game is final
                status = game.get("status", "")
                if status and status.lower() not in ("final", ""):
                    logger.debug("Skipping non-final game %d (%s)", game_id, status)
                    continue

                # 3. Fetch box score
                stats = bdl_get_game_stats(game_id)
                box_score = build_box_score(stats)

                if not box_score:
                    games_with_no_data += 1
                    continue

                # 4. Match to PropLine props
                game_props = match_bdl_game_to_propline(home_team, away_team, propline_data)
                if game_props:
                    prop_matches += 1

                # Build game record
                game_record = {
                    "game_id": game_id,
                    "date": date_str,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_score": game.get("home_team_score", 0),
                    "away_score": game.get("visitor_team_score", 0),
                    "box_score": box_score,
                    "props": game_props or {},
                }
                all_games.append(game_record)

                total_games_fetched += 1
                total_box_scores += len(box_score)
                if game_props:
                    total_prop_lines += sum(len(pp) for pp in game_props.values())

                # Progress report every 10 games
                if total_games_fetched % 10 == 0:
                    pct = 100.0 * total_games_fetched / max(1, estimated_total_games)
                    logger.info(
                        "Fetched %d/%d games (%.1f%%) | %d box scores | %d prop lines",
                        total_games_fetched, estimated_total_games, min(pct, 99.9),
                        total_box_scores,
                        total_prop_lines,
                    )

            if propline_data:
                logger.info("  %s: %d games, %d with props (PropLine had %d events)",
                            date_str, len(bdl_games), prop_matches, len(propline_data))
            else:
                logger.info("  %s: %d games (no PropLine data)", date_str, len(bdl_games))

        # Save after each weekly batch
        output = _build_output(all_games, start_date, end_date, total_box_scores, total_prop_lines)
        save_data(output, DATA_FILE)
        logger.info("  Saved checkpoint: %d games total", len(all_games))

    # Final save
    output = _build_output(all_games, start_date, end_date, total_box_scores, total_prop_lines)
    save_data(output, DATA_FILE)

    # Generate legs file
    all_legs = []
    for game in all_games:
        all_legs.extend(generate_legs(game))

    save_data(all_legs, LEGS_FILE)

    # Summary
    games_with_props = sum(1 for g in all_games if g.get("props"))
    logger.info("=" * 60)
    logger.info("FETCH COMPLETE")
    logger.info("  Games: %d", len(all_games))
    logger.info("  Games with props: %d (%.1f%%)",
                games_with_props,
                100.0 * games_with_props / max(1, len(all_games)))
    logger.info("  Box score entries: %d", total_box_scores)
    logger.info("  Prop lines: %d", total_prop_lines)
    logger.info("  Legs generated: %d", len(all_legs))
    logger.info("  Skipped (no box score): %d", games_with_no_data)
    logger.info("  Saved to: %s", DATA_FILE)
    logger.info("  Legs to:  %s", LEGS_FILE)
    logger.info("=" * 60)


def _build_output(all_games: List[Dict], start_date: str, end_date: str,
                  total_box_scores: int, total_prop_lines: int) -> Dict:
    """Build the full output dict with metadata."""
    return {
        "metadata": {
            "date_range": f"{start_date} to {end_date}",
            "total_games": len(all_games),
            "total_box_scores": total_box_scores,
            "total_prop_lines": total_prop_lines,
            "fetched_at": datetime.now().isoformat(),
        },
        "games": all_games,
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fetch NBA game data for out-of-sample backtesting"
    )
    parser.add_argument("--start-date", default=DEFAULT_START,
                        help=f"Start date YYYY-MM-DD (default: {DEFAULT_START})")
    parser.add_argument("--end-date", default=DEFAULT_END,
                        help=f"End date YYYY-MM-DD (default: {DEFAULT_END})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last saved progress (skip completed dates)")
    args = parser.parse_args()

    # Validate dates
    try:
        start = datetime.strptime(args.start_date, "%Y-%m-%d")
        end = datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError as e:
        logger.error("Invalid date format: %s", e)
        sys.exit(1)

    if end <= start:
        logger.error("End date must be after start date")
        sys.exit(1)

    days = (end - start).days
    logger.info("Date range: %s to %s (%d days)", args.start_date, args.end_date, days)
    logger.info("Estimated: ~%d-%d games, ~%d-%d minutes",
                days * 3, days * 8,
                days * 3 * 2 // 60, days * 8 * 2 // 60)

    fetch_all(args.start_date, args.end_date, resume=args.resume)


if __name__ == "__main__":
    main()
