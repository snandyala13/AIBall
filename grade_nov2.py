#!/usr/bin/env python3
"""
grade_nov2.py - Grade Nov 2, 2026 NBA games using historical DK lines.

Fetches historical DraftKings player prop lines from The Odds API,
fetches actual box scores from BDL API, and grades every prop.

Also backfills the v2_predictions JSONL so the standard grade_predictions.py
can track these games going forward.

Usage:
    python grade_nov2.py                    # Full run: fetch lines, grade, backfill
    python grade_nov2.py --fetch-only       # Just fetch and cache lines
    python grade_nov2.py --date 2026-11-01  # Grade a different date
"""

import argparse
import json
import os
import sys
import time
import requests
from collections import defaultdict
from datetime import datetime, timedelta

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---- Config ----------------------------------------------------------------

BDL_API_KEY = os.environ.get("BALLDONTLIE_API_KEY", "")
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
BDL_BASE = "https://api.balldontlie.io/v1"
ODDS_BASE = "https://api.the-odds-api.com/v4"

REQUEST_DELAY = 0.4  # seconds between API calls

PROP_MARKETS = [
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_threes",
    "player_points_rebounds_assists",
    "player_steals",
    "player_blocks",
    "player_turnovers",
]

MARKET_TO_PROP = {
    "player_points": "points",
    "player_rebounds": "rebounds",
    "player_assists": "assists",
    "player_threes": "threes",
    "player_points_rebounds_assists": "pts_rebs_asts",
    "player_steals": "steals",
    "player_blocks": "blocks",
    "player_turnovers": "turnovers",
}

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

_last_request = 0.0


# ---- Rate-limited requests -------------------------------------------------

def _rate_limited_get(url, params=None, headers=None):
    """GET with rate limiting and retry on 429."""
    global _last_request
    elapsed = time.time() - _last_request
    if elapsed < REQUEST_DELAY:
        time.sleep(REQUEST_DELAY - elapsed)
    _last_request = time.time()

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        if resp.status_code == 429:
            print("  Rate limited, waiting 5s...")
            time.sleep(5)
            return _rate_limited_get(url, params, headers)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print("  API error: {}".format(exc))
        return None


# ---- The Odds API: Historical Props ----------------------------------------

def get_historical_events(date_str):
    """Get Odds API event IDs for all NBA games on a date.

    Args:
        date_str: YYYY-MM-DD

    Returns:
        list of dicts with id, home_team, away_team, commence_time
    """
    # Use 8pm ET (00:00 UTC next day) as snapshot time to get evening games
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    # Games on Nov 2 have commence_time on Nov 3 UTC -- use midnight UTC of the game date
    snapshot = "{}T20:00:00Z".format(date_str)

    url = "{}/historical/sports/basketball_nba/events".format(ODDS_BASE)
    params = {"apiKey": ODDS_API_KEY, "date": snapshot}
    data = _rate_limited_get(url, params)

    if not data:
        print("ERROR: Could not fetch historical events")
        return []

    events = data.get("data", [])
    # Filter to events on this date (commence_time within 24h)
    target_start = dt
    target_end = dt + timedelta(days=1, hours=12)  # Games tip off evening, end early AM next day

    filtered = []
    for e in events:
        ct = e.get("commence_time", "")
        try:
            event_dt = datetime.fromisoformat(ct.replace("Z", "+00:00")).replace(tzinfo=None)
        except (ValueError, AttributeError):
            continue
        if target_start <= event_dt <= target_end:
            filtered.append(e)

    return filtered


def get_historical_props(event_id, date_str, bookmaker="draftkings"):
    """Get historical DK player props for a specific event.

    Args:
        event_id: The Odds API event ID
        date_str: YYYY-MM-DD (used for snapshot timestamp)
        bookmaker: bookmaker key

    Returns:
        list of dicts with player_name, prop_type, line, over_odds, under_odds
    """
    # Snapshot ~1h before typical tip-off to get closing lines
    snapshot = "{}T23:00:00Z".format(date_str)

    url = "{}/historical/sports/basketball_nba/events/{}/odds".format(ODDS_BASE, event_id)
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": ",".join(PROP_MARKETS),
        "bookmakers": bookmaker,
        "date": snapshot,
        "oddsFormat": "american",
    }

    data = _rate_limited_get(url, params)
    if not data:
        return []

    event_data = data.get("data", {})
    bookmakers = event_data.get("bookmakers", [])
    if not bookmakers:
        return []

    props = []
    for bk in bookmakers:
        for market in bk.get("markets", []):
            market_key = market.get("key", "")
            prop_type = MARKET_TO_PROP.get(market_key)
            if not prop_type:
                continue

            # Group outcomes by player
            player_lines = {}
            for outcome in market.get("outcomes", []):
                player = outcome.get("description", "")
                direction = outcome.get("name", "")
                odds = outcome.get("price", -110)
                line = outcome.get("point", 0)

                if player not in player_lines:
                    player_lines[player] = {"line": float(line), "player_name": player}

                if direction == "Over":
                    player_lines[player]["over_odds"] = odds
                elif direction == "Under":
                    player_lines[player]["under_odds"] = odds

            for player, pdata in player_lines.items():
                if "over_odds" in pdata and "under_odds" in pdata:
                    props.append({
                        "player_name": pdata["player_name"],
                        "prop_type": prop_type,
                        "line": pdata["line"],
                        "over_odds": pdata["over_odds"],
                        "under_odds": pdata["under_odds"],
                    })

    return props


# ---- BDL API: Games and Box Scores ----------------------------------------

def get_bdl_games(date_str):
    """Get BDL games for a date."""
    url = "{}/games".format(BDL_BASE)
    params = {"dates[]": date_str, "per_page": 25}
    headers = {"Authorization": BDL_API_KEY}
    data = _rate_limited_get(url, params, headers)
    return data.get("data", []) if data else []


def get_box_score(game_id):
    """Get player stats for a game from BDL."""
    url = "{}/stats".format(BDL_BASE)
    params = {"game_ids[]": game_id, "per_page": 50}
    headers = {"Authorization": BDL_API_KEY}
    data = _rate_limited_get(url, params, headers)
    return data.get("data", []) if data else []


def parse_minutes(min_str):
    """Parse minutes from BDL format."""
    if not min_str:
        return 0
    min_str = str(min_str)
    try:
        if ":" in min_str:
            return int(min_str.split(":")[0])
        return int(float(min_str))
    except (ValueError, TypeError):
        return 0


# ---- Team matching ---------------------------------------------------------

def normalize_team(name):
    """Normalize team names for matching between BDL and Odds API."""
    name = name.lower().strip()
    # The Odds API uses full city + team, BDL uses full names
    replacements = {
        "la clippers": "los angeles clippers",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name


def match_odds_to_bdl(odds_events, bdl_games):
    """Match Odds API events to BDL game IDs by team names.

    Returns:
        list of (odds_event, bdl_game) tuples
    """
    matched = []
    used_bdl = set()

    for event in odds_events:
        odds_home = normalize_team(event.get("home_team", ""))
        odds_away = normalize_team(event.get("away_team", ""))

        for game in bdl_games:
            gid = game.get("id")
            if gid in used_bdl:
                continue
            bdl_home = normalize_team(game.get("home_team", {}).get("full_name", ""))
            bdl_away = normalize_team(game.get("visitor_team", {}).get("full_name", ""))

            if (odds_home in bdl_home or bdl_home in odds_home) and \
               (odds_away in bdl_away or bdl_away in odds_away):
                matched.append((event, game))
                used_bdl.add(gid)
                break

    return matched


# ---- Grading ---------------------------------------------------------------

def american_to_decimal(odds):
    """Convert American odds to decimal odds."""
    if odds >= 100:
        return odds / 100.0
    else:
        return 100.0 / abs(odds)


def american_to_implied_prob(odds):
    """Convert American odds to implied probability."""
    if odds >= 100:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def grade_props(props, player_stats, side="both"):
    """Grade props against actual stats.

    For each prop, determines both over and under outcome.
    If side="both", generates two entries (over and under).
    """
    results = []
    for prop in props:
        player = prop["player_name"]
        prop_type = prop["prop_type"]
        line = prop["line"]
        over_odds = prop.get("over_odds", -110)
        under_odds = prop.get("under_odds", -110)

        # Find actual stats
        pstats = player_stats.get(player)
        if pstats is None:
            # Fuzzy match
            for name, stats in player_stats.items():
                if player.lower() == name.lower():
                    pstats = stats
                    break

        if pstats is None:
            continue

        stat_keys = PROP_TO_STAT.get(prop_type)
        if not stat_keys:
            continue
        actual = sum(pstats.get(k, 0) for k in stat_keys)

        push = (actual == line)
        over_hit = actual > line
        under_hit = actual < line

        # Over result
        if side in ("both", "over"):
            over_decimal = american_to_decimal(over_odds)
            over_profit = 0.0 if push else (over_decimal if over_hit else -1.0)
            results.append({
                "player": player,
                "prop_type": prop_type,
                "side": "over",
                "line": line,
                "actual": actual,
                "hit": over_hit,
                "push": push,
                "dk_odds": over_odds,
                "profit": round(over_profit, 3),
                "implied_prob": round(american_to_implied_prob(over_odds), 4),
            })

        # Under result
        if side in ("both", "under"):
            under_decimal = american_to_decimal(under_odds)
            under_profit = 0.0 if push else (under_decimal if under_hit else -1.0)
            results.append({
                "player": player,
                "prop_type": prop_type,
                "side": "under",
                "line": line,
                "actual": actual,
                "hit": under_hit,
                "push": push,
                "dk_odds": under_odds,
                "profit": round(under_profit, 3),
                "implied_prob": round(american_to_implied_prob(under_odds), 4),
            })

    return results


# ---- JSONL backfill --------------------------------------------------------

def backfill_predictions_jsonl(date_str, all_graded, game_map):
    """Write graded lines to the v2_predictions JSONL for tracking.

    Only writes props that aren't already in the file.
    """
    os.makedirs("outputs", exist_ok=True)

    now = datetime.now()
    season_year = now.year if now.month >= 10 else now.year - 1
    season = "{}-{}".format(season_year, str(season_year + 1)[-2:])
    log_file = "outputs/v2_predictions_{}.jsonl".format(season)

    # Load existing keys for dedup
    existing_keys = set()
    if os.path.exists(log_file):
        with open(log_file) as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    rec = json.loads(raw_line)
                    dk = "{}|{}|{}|{}|{}".format(
                        rec.get("game_id", ""),
                        rec.get("player", ""),
                        rec.get("prop_type", ""),
                        rec.get("side", ""),
                        rec.get("date", ""),
                    )
                    existing_keys.add(dk)
                except (json.JSONDecodeError, KeyError):
                    continue

    logged = 0
    with open(log_file, "a") as f:
        for entry in all_graded:
            game_id = entry.get("game_id", 0)
            matchup = entry.get("matchup", "")
            player = entry.get("player", "")
            prop_type = entry.get("prop_type", "")
            side = entry.get("side", "over")

            dk = "{}|{}|{}|{}|{}".format(game_id, player, prop_type, side, date_str)
            if dk in existing_keys:
                continue

            record = {
                "date": date_str,
                "game_id": game_id,
                "matchup": matchup,
                "player": player,
                "prop_type": prop_type,
                "side": side,
                "line": entry.get("line", 0),
                "dk_odds": entry.get("dk_odds", -110),
                "over_odds": entry.get("dk_odds", -110) if side == "over" else -110,
                "under_odds": entry.get("dk_odds", -110) if side == "under" else -110,
                "predicted_value": 0,  # No model prediction -- historical grade only
                "predicted_adjustment": 0,
                "edge_percent": 0,
                "p_over": entry.get("implied_prob", 0.5) if side == "over" else 1 - entry.get("implied_prob", 0.5),
                "p_under": entry.get("implied_prob", 0.5) if side == "under" else 1 - entry.get("implied_prob", 0.5),
                "confidence": 0,
                "confidence_stars": 0,
                "kelly_fraction": 0,
                "recommended_units": 0,
                "best_book": "draftkings",
                "best_odds": None,
                "odds_tier": "standard",
                "edge_at_dk": 0,
                "filter_tier": "historical_grade",
                "all_books": {},
                "line_shopping": [],
                "model": "historical_grade",
                "actual": entry.get("actual"),
                "hit": entry.get("hit"),
            }
            f.write(json.dumps(record) + "\n")
            existing_keys.add(dk)
            logged += 1

    return logged


# ---- Report ----------------------------------------------------------------

def print_report(all_graded, date_str, game_summaries):
    """Print grading report."""
    resolved = [r for r in all_graded if r.get("actual") is not None]

    if not resolved:
        print("\nNo resolved picks to grade.")
        return

    # Only grade UNDER side (our strategy) unless we want full market view
    # Grade both for comprehensive analysis
    over_picks = [r for r in resolved if r["side"] == "over"]
    under_picks = [r for r in resolved if r["side"] == "under"]

    print("")
    print("=" * 72)
    print("  HISTORICAL GRADING REPORT - {}".format(date_str))
    print("  Source: The Odds API (DraftKings lines)")
    print("=" * 72)
    print("")

    # Game summary
    print("-" * 72)
    print("  GAMES ({})".format(len(game_summaries)))
    print("-" * 72)
    for gs in game_summaries:
        print("  {} | Props: {} | Score: {}".format(
            gs["matchup"], gs["prop_count"], gs["score"]))
    print("")

    # Overall
    for label, picks in [("ALL", resolved), ("OVER", over_picks), ("UNDER", under_picks)]:
        if not picks:
            continue
        hits = sum(1 for r in picks if r["hit"])
        pushes = sum(1 for r in picks if r["push"])
        total = len(picks)
        hit_rate = hits / total * 100 if total else 0
        profit = sum(r["profit"] for r in picks)
        avg_profit = profit / total if total else 0
        print("  {:>6s}: {}/{} ({:.1f}%) | {} pushes | {:+.2f} units (avg {:+.3f}/pick)".format(
            label, hits, total, hit_rate, pushes, profit, avg_profit))
    print("")

    # By prop type
    print("-" * 72)
    print("  BY PROP TYPE (both sides)")
    print("-" * 72)
    print("  {:<20s} {:>5s} {:>6s} {:>6s}  {:>8s}  {:>8s}".format(
        "Prop", "Picks", "Overs", "Unders", "O Hit%", "U Hit%"))

    by_prop = defaultdict(list)
    for r in resolved:
        by_prop[r["prop_type"]].append(r)

    for prop in sorted(by_prop.keys()):
        picks = by_prop[prop]
        overs = [p for p in picks if p["side"] == "over"]
        unders = [p for p in picks if p["side"] == "under"]
        o_hits = sum(1 for p in overs if p["hit"])
        u_hits = sum(1 for p in unders if p["hit"])
        o_rate = o_hits / len(overs) * 100 if overs else 0
        u_rate = u_hits / len(unders) * 100 if unders else 0
        print("  {:<20s} {:>5d} {:>6d} {:>6d}  {:>7.1f}%  {:>7.1f}%".format(
            prop, len(picks), len(overs), len(unders), o_rate, u_rate))
    print("")

    # Market efficiency check: how often does the favorite side win?
    print("-" * 72)
    print("  MARKET EFFICIENCY CHECK")
    print("-" * 72)
    favorite_wins = 0
    total_checked = 0
    for r in resolved:
        if r["push"]:
            continue
        impl_prob = r.get("implied_prob", 0.5)
        is_favorite = impl_prob > 0.5
        if is_favorite and r["hit"]:
            favorite_wins += 1
        elif not is_favorite and not r["hit"]:
            favorite_wins += 1
        total_checked += 1

    if total_checked:
        print("  Favorite side won: {}/{} ({:.1f}%)".format(
            favorite_wins, total_checked,
            favorite_wins / total_checked * 100))
    print("")

    # Under bias analysis (key finding from backtesting)
    print("-" * 72)
    print("  UNDER BIAS ANALYSIS")
    print("-" * 72)
    if under_picks:
        u_hits = sum(1 for p in under_picks if p["hit"])
        u_total = len(under_picks)
        print("  Under hit rate: {}/{} ({:.1f}%)".format(u_hits, u_total, u_hits/u_total*100))
        # By prop type
        for prop in sorted(set(p["prop_type"] for p in under_picks)):
            pp = [p for p in under_picks if p["prop_type"] == prop]
            h = sum(1 for p in pp if p["hit"])
            n = len(pp)
            pf = sum(p["profit"] for p in pp)
            print("    {:<20s} {}/{} ({:.1f}%) | {:+.2f} units".format(
                prop, h, n, h/n*100, pf))
    print("")

    # Top misses (biggest margins)
    print("-" * 72)
    print("  BIGGEST BLOWOUTS (actual far from line)")
    print("-" * 72)
    by_margin = sorted(resolved, key=lambda x: abs(x["actual"] - x["line"]), reverse=True)
    for r in by_margin[:10]:
        margin = r["actual"] - r["line"]
        status = "HIT" if r["hit"] else ("PUSH" if r["push"] else "MISS")
        print("  {} {} {} {:.1f} | Actual: {:.0f} ({:+.1f}) | {} | {}".format(
            r["player"], r["prop_type"], r["side"].upper(), r["line"],
            r["actual"], margin, status, r.get("dk_odds", "")))
    print("")


# ---- Cache -----------------------------------------------------------------

def get_cache_path(date_str):
    """Return cache file path for a date's historical lines."""
    os.makedirs("outputs/historical_cache", exist_ok=True)
    return "outputs/historical_cache/odds_api_{}.json".format(date_str)


def save_cache(date_str, data):
    """Cache fetched data to avoid re-fetching."""
    path = get_cache_path(date_str)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print("  Cached {} lines to {}".format(len(data), path))


def load_cache(date_str):
    """Load cached data if available."""
    path = get_cache_path(date_str)
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        print("  Loaded {} cached lines from {}".format(len(data), path))
        return data
    return None


# ---- Main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Grade historical NBA games")
    parser.add_argument("--date", type=str, default="2026-11-02",
                        help="Game date to grade (YYYY-MM-DD, default: 2026-11-02)")
    parser.add_argument("--fetch-only", action="store_true",
                        help="Only fetch and cache lines, don't grade")
    parser.add_argument("--no-backfill", action="store_true",
                        help="Don't backfill the predictions JSONL")
    parser.add_argument("--use-cache", action="store_true",
                        help="Use cached lines if available")
    args = parser.parse_args()

    date_str = args.date

    if not BDL_API_KEY:
        print("ERROR: BALLDONTLIE_API_KEY not set")
        sys.exit(1)
    if not ODDS_API_KEY:
        print("ERROR: ODDS_API_KEY not set")
        sys.exit(1)

    print("=" * 72)
    print("  Grading NBA games for {}".format(date_str))
    print("=" * 72)
    print("")

    # ---- Step 1: Get BDL games and box scores ----
    print("[1/4] Fetching BDL games for {}...".format(date_str))
    bdl_games = get_bdl_games(date_str)
    final_games = [g for g in bdl_games if g.get("status") == "Final"]
    print("  Found {} games ({} Final)".format(len(bdl_games), len(final_games)))

    if not final_games:
        print("ERROR: No finalized games found for {}".format(date_str))
        sys.exit(1)

    print("\n[2/4] Fetching box scores...")
    all_actuals = {}  # game_id -> {player_name: {stat: val}}
    for game in final_games:
        gid = game["id"]
        home = game.get("home_team", {}).get("full_name", "?")
        away = game.get("visitor_team", {}).get("full_name", "?")
        h_score = game.get("home_team_score", 0)
        v_score = game.get("visitor_team_score", 0)

        stats = get_box_score(gid)
        player_stats = {}
        for stat in stats:
            player = stat.get("player", {})
            name = "{} {}".format(
                player.get("first_name", ""),
                player.get("last_name", ""),
            )
            mins = parse_minutes(stat.get("min", 0))
            if mins < 5:
                continue
            player_stats[name] = {
                "pts": stat.get("pts", 0) or 0,
                "reb": stat.get("reb", 0) or 0,
                "ast": stat.get("ast", 0) or 0,
                "fg3m": stat.get("fg3m", 0) or 0,
                "stl": stat.get("stl", 0) or 0,
                "blk": stat.get("blk", 0) or 0,
                "turnover": stat.get("turnover", 0) or 0,
                "min": mins,
            }
        all_actuals[gid] = {
            "players": player_stats,
            "home": home,
            "away": away,
            "score": "{}-{}".format(v_score, h_score),
        }
        print("  Game {} ({} @ {}): {} players with 5+ min".format(
            gid, away, home, len(player_stats)))

    # ---- Step 3: Get historical DK lines ----
    # Check cache first
    cached_lines = None
    if args.use_cache:
        cached_lines = load_cache(date_str)

    if cached_lines is not None:
        all_props_by_game = cached_lines
    else:
        print("\n[3/4] Fetching historical DK lines from The Odds API...")
        odds_events = get_historical_events(date_str)
        print("  Found {} Odds API events".format(len(odds_events)))

        # Match to BDL games
        matched = match_odds_to_bdl(odds_events, final_games)
        print("  Matched {} events to BDL games".format(len(matched)))

        all_props_by_game = {}
        for odds_event, bdl_game in matched:
            event_id = odds_event["id"]
            gid = bdl_game["id"]
            home = bdl_game.get("home_team", {}).get("full_name", "?")
            away = bdl_game.get("visitor_team", {}).get("full_name", "?")

            props = get_historical_props(event_id, date_str)
            all_props_by_game[str(gid)] = {
                "props": props,
                "matchup": "{} @ {}".format(away, home),
                "odds_event_id": event_id,
            }
            print("  {} @ {}: {} props".format(away, home, len(props)))

        # Cache the results
        save_cache(date_str, all_props_by_game)

    if args.fetch_only:
        total_props = sum(len(v["props"]) for v in all_props_by_game.values())
        print("\nFetch complete: {} props across {} games".format(
            total_props, len(all_props_by_game)))
        return

    # ---- Step 4: Grade ----
    print("\n[4/4] Grading...")
    all_graded = []
    game_summaries = []

    for gid_str, game_data in all_props_by_game.items():
        gid = int(gid_str)
        props = game_data["props"]
        matchup = game_data.get("matchup", "?")

        actuals = all_actuals.get(gid, {})
        player_stats = actuals.get("players", {})
        score = actuals.get("score", "?")

        if not player_stats:
            print("  WARNING: No box score for game {} ({})".format(gid, matchup))
            continue

        graded = grade_props(props, player_stats, side="both")

        # Tag with game info
        for entry in graded:
            entry["game_id"] = gid
            entry["matchup"] = matchup

        all_graded.extend(graded)
        game_summaries.append({
            "matchup": matchup,
            "prop_count": len(props),
            "score": score,
        })

    # Print report
    print_report(all_graded, date_str, game_summaries)

    # Backfill JSONL
    if not args.no_backfill:
        print("-" * 72)
        print("  BACKFILL")
        print("-" * 72)
        # Only backfill under picks (our strategy focus)
        under_graded = [r for r in all_graded if r["side"] == "under"]
        logged = backfill_predictions_jsonl(date_str, under_graded,
                                            {int(k): v for k, v in all_props_by_game.items()})
        print("  Backfilled {} under-side predictions to JSONL".format(logged))
        print("")

    # Summary stats file
    summary = {
        "date": date_str,
        "games": len(game_summaries),
        "total_props": len(all_graded),
        "over_hits": sum(1 for r in all_graded if r["side"] == "over" and r["hit"]),
        "over_total": sum(1 for r in all_graded if r["side"] == "over"),
        "under_hits": sum(1 for r in all_graded if r["side"] == "under" and r["hit"]),
        "under_total": sum(1 for r in all_graded if r["side"] == "under"),
        "total_profit_over": sum(r["profit"] for r in all_graded if r["side"] == "over"),
        "total_profit_under": sum(r["profit"] for r in all_graded if r["side"] == "under"),
    }
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/grade_summary_{}.json".format(date_str), "w") as f:
        json.dump(summary, f, indent=2)
    print("  Summary saved to outputs/grade_summary_{}.json".format(date_str))

    # Also save full graded results for detailed analysis
    with open("outputs/graded_results_{}.json".format(date_str), "w") as f:
        json.dump(all_graded, f, indent=2)
    print("  Full results saved to outputs/graded_results_{}.json".format(date_str))


if __name__ == "__main__":
    main()
