#!/usr/bin/env python3
"""
grade_predictions.py - Grade v2 model predictions against actual box score outcomes.

Reads outputs/v2_predictions_{season}.jsonl, fetches box scores from the
Balldontlie API, and calculates hit rate, ROI, and breakdowns by prop type,
side, and confidence level.

Usage:
    python grade_predictions.py                          # Grade all predictions
    python grade_predictions.py --date 2026-10-22        # Grade a specific date
    python grade_predictions.py --last 7                 # Grade last 7 days
    python grade_predictions.py --file outputs/v2_predictions_2026-27.jsonl
"""

import argparse
import json
import os
import sys
import time
import requests
from collections import defaultdict
from datetime import datetime, timedelta

# ---- Configuration ---------------------------------------------------------

API_KEY = os.environ.get("BALLDONTLIE_API_KEY", "")
BASE_URL = "https://api.balldontlie.io/v1"

# Map v2 prop_type strings to box score stat keys
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

REQUEST_DELAY = 0.35  # seconds between API calls (rate-limit safe)
_last_request = 0.0


# ---- API helpers -----------------------------------------------------------

def _rate_limited_get(url, params=None):
    """GET with rate limiting and retry on 429."""
    global _last_request
    elapsed = time.time() - _last_request
    if elapsed < REQUEST_DELAY:
        time.sleep(REQUEST_DELAY - elapsed)
    _last_request = time.time()

    headers = {"Authorization": API_KEY}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code == 429:
            print("  Rate limited, waiting 5s...")
            time.sleep(5)
            return _rate_limited_get(url, params)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print("  API error: {}".format(exc))
        return None


def get_games_for_date(date_str):
    """Return list of game dicts for a given date."""
    data = _rate_limited_get("{}/games".format(BASE_URL), {"dates[]": date_str})
    if data:
        return data.get("data", [])
    return []


def get_box_score_stats(game_id):
    """Return list of player stat lines for a game."""
    data = _rate_limited_get("{}/stats".format(BASE_URL), {
        "game_ids[]": game_id,
        "per_page": 50,
    })
    if data:
        return data.get("data", [])
    return []


def parse_minutes(min_str):
    """Parse minutes played from various BDL formats."""
    if not min_str:
        return 0
    min_str = str(min_str)
    try:
        if ":" in min_str:
            return int(min_str.split(":")[0])
        return int(float(min_str))
    except (ValueError, TypeError):
        return 0


# ---- Loading predictions ---------------------------------------------------

def get_default_log_file():
    """Return the default v2 predictions log file for the current season."""
    now = datetime.now()
    year = now.year if now.month >= 10 else now.year - 1
    season = "{}-{}".format(year, str(year + 1)[-2:])
    return "outputs/v2_predictions_{}.jsonl".format(season)


def load_predictions(log_file, date_filter=None):
    """Load predictions from JSONL, optionally filtered to a specific date.

    Args:
        log_file: path to the JSONL file
        date_filter: optional YYYY-MM-DD string (or list of strings) to filter

    Returns:
        dict mapping date -> list of prediction records
    """
    if isinstance(date_filter, str):
        date_filter = {date_filter}
    elif date_filter:
        date_filter = set(date_filter)

    by_date = defaultdict(list)
    if not os.path.exists(log_file):
        print("ERROR: log file not found: {}".format(log_file))
        sys.exit(1)

    with open(log_file) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print("  Skipping malformed line {}".format(lineno))
                continue
            d = rec.get("date", "")
            if date_filter and d not in date_filter:
                continue
            by_date[d].append(rec)

    return dict(by_date)


# ---- CLV (Closing Line Value) ----------------------------------------------

def get_default_snapshot_file():
    """Return the default line snapshots file for the current season."""
    now = datetime.now()
    year = now.year if now.month >= 10 else now.year - 1
    season = "{}-{}".format(year, str(year + 1)[-2:])
    return "outputs/line_snapshots_{}.jsonl".format(season)


def load_line_snapshots(snapshot_file=None):
    """Load line snapshots from JSONL.

    Builds an index for fast closing-line lookup:
        {game_id: {player|prop_type: [snapshots sorted by timestamp]}}

    Returns:
        dict: nested mapping described above, empty dict if file missing
    """
    if snapshot_file is None:
        snapshot_file = get_default_snapshot_file()

    if not os.path.exists(snapshot_file):
        return {}

    index = defaultdict(lambda: defaultdict(list))
    with open(snapshot_file) as f:
        for line_text in f:
            line_text = line_text.strip()
            if not line_text:
                continue
            try:
                rec = json.loads(line_text)
            except json.JSONDecodeError:
                continue
            gid = rec.get("game_id")
            player = rec.get("player", "")
            prop_type = rec.get("prop_type", "")
            if not gid or not player or not prop_type:
                continue
            key = "{}|{}".format(player, prop_type)
            index[gid][key].append(rec)

    # Sort each snapshot list by timestamp
    for gid in index:
        for key in index[gid]:
            index[gid][key].sort(key=lambda r: r.get("timestamp", ""))

    return dict(index)


def get_closing_line(snapshots_for_game, player, prop_type):
    """Get the closing line (last recorded snapshot) for a player/prop.

    Args:
        snapshots_for_game: {player|prop_type: [snapshots]} for one game
        player: player name
        prop_type: prop type string

    Returns:
        dict with closing line info, or None if no snapshot data
    """
    key = "{}|{}".format(player, prop_type)
    snaps = snapshots_for_game.get(key, [])
    if not snaps:
        return None

    # The last snapshot is the closest to tip-off (closing line)
    closing = snaps[-1]

    # Also find the analysis snapshot (first "analysis" type or the first one)
    analysis = None
    for s in snaps:
        if s.get("snapshot_type") == "analysis":
            analysis = s
            break
    if analysis is None:
        analysis = snaps[0]

    return {
        "closing_line": closing.get("line", 0),
        "closing_over_odds": closing.get("over_odds", -110),
        "closing_under_odds": closing.get("under_odds", -110),
        "closing_timestamp": closing.get("timestamp", ""),
        "analysis_line": analysis.get("line", 0),
        "analysis_timestamp": analysis.get("timestamp", ""),
        "num_snapshots": len(snaps),
    }


def calculate_clv(prediction, closing_info):
    """Calculate CLV for a single prediction.

    For overs: positive CLV means we got a lower line than the market settled on
        CLV = closing_line - our_analysis_line
    For unders: positive CLV means we got a higher line than the market settled on
        CLV = our_analysis_line - closing_line

    Returns:
        dict with clv, beat_close, etc. or None
    """
    if closing_info is None:
        return None

    our_line = prediction.get("line", 0)
    closing_line = closing_info.get("closing_line", 0)
    side = prediction.get("side", "over")

    if side == "over":
        clv = closing_line - our_line
    else:
        clv = our_line - closing_line

    return {
        "clv": round(clv, 2),
        "our_line": our_line,
        "closing_line": closing_line,
        "beat_close": clv > 0,
        "side": side,
        "num_snapshots": closing_info.get("num_snapshots", 0),
    }


# ---- Fetching actuals ------------------------------------------------------

def build_actual_stats_for_date(date_str, game_id_filter=None):
    """Fetch box scores for all games on a date and return actual stats.

    Returns:
        dict: {game_id: {player_name: {stat_key: value, ...}}}
    """
    games = get_games_for_date(date_str)
    if not games:
        return {}

    all_stats = {}
    for game in games:
        gid = game.get("id")
        status = game.get("status", "")
        if status != "Final":
            continue  # Skip games that haven't finished
        if game_id_filter and gid not in game_id_filter:
            continue

        stats = get_box_score_stats(gid)
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
        all_stats[gid] = player_stats

    return all_stats


def compute_actual_value(player_stats, prop_type):
    """Compute the actual stat value for a given prop type.

    Args:
        player_stats: dict of stat_key -> value for one player
        prop_type: string prop type from the prediction

    Returns:
        float or None if prop type not mappable
    """
    stat_keys = PROP_TO_STAT.get(prop_type)
    if not stat_keys:
        return None
    return sum(player_stats.get(k, 0) for k in stat_keys)


# ---- Grading ---------------------------------------------------------------

def american_to_decimal(odds):
    """Convert American odds to decimal odds."""
    if odds >= 100:
        return odds / 100.0
    else:
        return 100.0 / abs(odds)


def grade_predictions_for_date(predictions, actual_stats, snapshot_index=None):
    """Grade a list of predictions against actual stats.

    Args:
        predictions: list of prediction dicts (all from one date)
        actual_stats: {game_id: {player_name: {stat: val}}}
        snapshot_index: optional {game_id: {player|prop_type: [snapshots]}}

    Returns:
        list of graded result dicts
    """
    results = []
    for pred in predictions:
        game_id = pred.get("game_id")
        player = pred.get("player", "")
        prop_type = pred.get("prop_type", "")
        side = pred.get("side", "over")
        line = pred.get("line", 0)
        dk_odds = pred.get("dk_odds", -110)

        # Find actual stats
        game_stats = actual_stats.get(game_id, {})
        player_stat = game_stats.get(player)

        if player_stat is None:
            # Try fuzzy match (first-last vs last-first)
            for name, stats in game_stats.items():
                if player.lower() == name.lower():
                    player_stat = stats
                    break

        # CLV lookup (optional, graceful if no data)
        clv_data = None
        if snapshot_index and game_id:
            game_snaps = snapshot_index.get(game_id, {})
            if game_snaps:
                closing_info = get_closing_line(game_snaps, player, prop_type)
                clv_data = calculate_clv(pred, closing_info)

        if player_stat is None:
            entry = {
                "player": player,
                "prop_type": prop_type,
                "side": side,
                "line": line,
                "predicted_value": pred.get("predicted_value", 0),
                "actual": None,
                "hit": None,
                "push": False,
                "status": "DNP_OR_MISSING",
                "profit": 0.0,
                "edge_percent": pred.get("edge_percent", 0),
                "confidence": pred.get("confidence", 0),
                "confidence_stars": pred.get("confidence_stars", 1),
                "game_id": game_id,
                "matchup": pred.get("matchup", ""),
            }
            if clv_data:
                entry["clv"] = clv_data
            results.append(entry)
            continue

        actual = compute_actual_value(player_stat, prop_type)
        if actual is None:
            continue

        # Determine hit/push
        push = (actual == line)
        if side == "over":
            hit = actual > line
        else:
            hit = actual < line

        # Calculate profit per $100 wagered
        decimal_odds = american_to_decimal(dk_odds)
        if push:
            profit = 0.0
        elif hit:
            profit = decimal_odds  # net win per 1 unit
        else:
            profit = -1.0  # lost 1 unit

        entry = {
            "player": player,
            "prop_type": prop_type,
            "side": side,
            "line": line,
            "predicted_value": pred.get("predicted_value", 0),
            "actual": actual,
            "hit": hit,
            "push": push,
            "status": "PUSH" if push else ("HIT" if hit else "MISS"),
            "profit": round(profit, 3),
            "edge_percent": pred.get("edge_percent", 0),
            "confidence": pred.get("confidence", 0),
            "confidence_stars": pred.get("confidence_stars", 1),
            "game_id": game_id,
            "matchup": pred.get("matchup", ""),
        }
        if clv_data:
            entry["clv"] = clv_data
        results.append(entry)

    return results


# ---- Reporting -------------------------------------------------------------

def print_report(all_results, dates):
    """Print a formatted grading report."""
    # Filter to only resolved picks (actual is not None)
    resolved = [r for r in all_results if r["actual"] is not None]
    unresolved = [r for r in all_results if r["actual"] is None]

    if not resolved:
        print("\nNo resolved picks to grade.")
        if unresolved:
            print("  {} picks could not be matched to box scores.".format(len(unresolved)))
        return

    # ---- Overall summary ----
    hits = sum(1 for r in resolved if r["hit"])
    misses = sum(1 for r in resolved if not r["hit"] and not r["push"])
    pushes = sum(1 for r in resolved if r["push"])
    total = len(resolved)
    hit_rate = hits / total * 100 if total else 0
    total_profit = sum(r["profit"] for r in resolved)
    avg_profit = total_profit / total if total else 0

    print("")
    print("=" * 70)
    print("  V2 MODEL GRADING REPORT")
    print("  Dates: {}".format(", ".join(sorted(dates))))
    print("=" * 70)
    print("")
    print("  OVERALL: {}/{} ({:.1f}%) | {} pushes".format(hits, total, hit_rate, pushes))
    print("  PROFIT:  {:+.2f} units on {} picks (avg {:+.3f}/pick)".format(
        total_profit, total, avg_profit))
    if unresolved:
        print("  UNRESOLVED: {} picks (DNP / missing box scores)".format(len(unresolved)))
    print("")

    # ---- By prop type ----
    print("-" * 70)
    print("  BY PROP TYPE")
    print("-" * 70)
    by_prop = defaultdict(list)
    for r in resolved:
        by_prop[r["prop_type"]].append(r)

    print("  {:<18s} {:>5s} {:>6s}  {:>8s}  {:>8s}".format(
        "Prop", "Picks", "Hit%", "Profit", "Avg"))
    for prop in sorted(by_prop.keys()):
        picks = by_prop[prop]
        h = sum(1 for p in picks if p["hit"])
        n = len(picks)
        hr = h / n * 100 if n else 0
        pf = sum(p["profit"] for p in picks)
        avg = pf / n if n else 0
        print("  {:<18s} {:>5d} {:>5.1f}%  {:>+8.2f}  {:>+7.3f}".format(
            prop, n, hr, pf, avg))
    print("")

    # ---- By side ----
    print("-" * 70)
    print("  BY SIDE")
    print("-" * 70)
    for side in ["over", "under"]:
        picks = [r for r in resolved if r["side"] == side]
        if not picks:
            continue
        h = sum(1 for p in picks if p["hit"])
        n = len(picks)
        hr = h / n * 100 if n else 0
        pf = sum(p["profit"] for p in picks)
        avg = pf / n if n else 0
        print("  {:<8s} {:>5d} picks | {:.1f}% hit | {:+.2f} profit (avg {:+.3f})".format(
            side.upper(), n, hr, pf, avg))
    print("")

    # ---- By confidence stars ----
    print("-" * 70)
    print("  BY CONFIDENCE (stars)")
    print("-" * 70)
    by_stars = defaultdict(list)
    for r in resolved:
        by_stars[r["confidence_stars"]].append(r)

    print("  {:<10s} {:>5s} {:>6s}  {:>8s}  {:>8s}".format(
        "Stars", "Picks", "Hit%", "Profit", "Avg"))
    for stars in sorted(by_stars.keys()):
        picks = by_stars[stars]
        h = sum(1 for p in picks if p["hit"])
        n = len(picks)
        hr = h / n * 100 if n else 0
        pf = sum(p["profit"] for p in picks)
        avg = pf / n if n else 0
        label = "*" * stars
        print("  {:<10s} {:>5d} {:>5.1f}%  {:>+8.2f}  {:>+7.3f}".format(
            label, n, hr, pf, avg))
    print("")

    # ---- By edge bucket ----
    print("-" * 70)
    print("  BY EDGE BUCKET")
    print("-" * 70)
    edge_buckets = [
        ("0-3%",   0, 3),
        ("3-5%",   3, 5),
        ("5-10%",  5, 10),
        ("10-20%", 10, 20),
        ("20%+",   20, 100),
    ]
    print("  {:<10s} {:>5s} {:>6s}  {:>8s}  {:>8s}".format(
        "Edge", "Picks", "Hit%", "Profit", "Avg"))
    for label, lo, hi in edge_buckets:
        picks = [r for r in resolved if lo <= r["edge_percent"] < hi]
        if not picks:
            continue
        h = sum(1 for p in picks if p["hit"])
        n = len(picks)
        hr = h / n * 100 if n else 0
        pf = sum(p["profit"] for p in picks)
        avg = pf / n if n else 0
        print("  {:<10s} {:>5d} {:>5.1f}%  {:>+8.2f}  {:>+7.3f}".format(
            label, n, hr, pf, avg))
    print("")

    # ---- CLV (Closing Line Value) ----
    clv_results = [r for r in resolved if r.get("clv")]
    if clv_results:
        print("-" * 70)
        print("  CLOSING LINE VALUE (CLV)")
        print("-" * 70)
        clv_values = [r["clv"]["clv"] for r in clv_results]
        positive_clv = [c for c in clv_values if c > 0]
        avg_clv = sum(clv_values) / len(clv_values) if clv_values else 0
        pct_positive = len(positive_clv) / len(clv_values) * 100 if clv_values else 0

        print("  Picks with CLV data:  {}".format(len(clv_results)))
        print("  Average CLV:          {:+.2f} pts".format(avg_clv))
        print("  Beat closing line:    {}/{} ({:.1f}%)".format(
            len(positive_clv), len(clv_values), pct_positive))

        # CLV by side
        for side in ["over", "under"]:
            side_clv = [r["clv"]["clv"] for r in clv_results if r.get("side") == side]
            if side_clv:
                avg_s = sum(side_clv) / len(side_clv)
                pos_s = sum(1 for c in side_clv if c > 0)
                print("  {} CLV:  {:+.2f} avg | {}/{} beat close ({:.1f}%)".format(
                    side.upper().ljust(5), avg_s, pos_s, len(side_clv),
                    pos_s / len(side_clv) * 100))

        # CLV by hit/miss (does beating the close correlate with wins?)
        clv_hits = [r["clv"]["clv"] for r in clv_results if r.get("hit")]
        clv_misses = [r["clv"]["clv"] for r in clv_results if r.get("hit") is False]
        if clv_hits:
            print("  CLV on HITS:   {:+.2f} avg ({} picks)".format(
                sum(clv_hits) / len(clv_hits), len(clv_hits)))
        if clv_misses:
            print("  CLV on MISSES: {:+.2f} avg ({} picks)".format(
                sum(clv_misses) / len(clv_misses), len(clv_misses)))
        print("")
    else:
        print("-" * 70)
        print("  CLV: No line snapshot data available")
        print("  (Run the web server to collect line snapshots for CLV analysis)")
        print("-" * 70)
        print("")

    # ---- Per-date breakdown ----
    if len(dates) > 1:
        print("-" * 70)
        print("  DAILY BREAKDOWN")
        print("-" * 70)
        by_date = defaultdict(list)
        for r in resolved:
            # Find which date this belongs to based on game_id -> predictions mapping
            by_date[r.get("matchup", "?")[:10]].append(r)
        # Use the actual dates from predictions
        by_date2 = defaultdict(list)
        for r in all_results:
            d = None
            for pred_date in dates:
                by_date2[pred_date].append(r)  # simplified: just use the report dates
            # Actually better: tag results by date from the prediction
        # Rebuild by date from predictions
        by_date_clean = defaultdict(list)
        for r in resolved:
            # game_id is the key; walk predictions to find the date
            by_date_clean["all"].append(r)

        print("  (See per-date reports by running with --date YYYY-MM-DD)")
        print("")

    # ---- Individual picks (worst and best) ----
    print("-" * 70)
    print("  BIGGEST MISSES (top 5 by edge that missed)")
    print("-" * 70)
    misses_list = sorted(
        [r for r in resolved if not r["hit"] and not r["push"]],
        key=lambda x: x["edge_percent"],
        reverse=True,
    )
    for r in misses_list[:5]:
        print("  {} {} {} {:.1f} | Pred: {:.1f} Actual: {:.0f} | Edge: {:.1f}% | {}".format(
            r["player"], r["prop_type"], r["side"].upper(), r["line"],
            r["predicted_value"], r["actual"], r["edge_percent"],
            r["matchup"],
        ))
    print("")

    print("-" * 70)
    print("  BIGGEST HITS (top 5 by margin)")
    print("-" * 70)
    hits_list = sorted(
        [r for r in resolved if r["hit"]],
        key=lambda x: abs(x["actual"] - x["line"]) if x["actual"] is not None else 0,
        reverse=True,
    )
    for r in hits_list[:5]:
        margin = abs(r["actual"] - r["line"])
        print("  {} {} {} {:.1f} | Actual: {:.0f} (margin: {:.1f}) | Edge: {:.1f}% | {}".format(
            r["player"], r["prop_type"], r["side"].upper(), r["line"],
            r["actual"], margin, r["edge_percent"],
            r["matchup"],
        ))
    print("")


# ---- Main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Grade v2 model predictions")
    parser.add_argument("--date", type=str, help="Grade a specific date (YYYY-MM-DD)")
    parser.add_argument("--last", type=int, help="Grade the last N days")
    parser.add_argument("--file", type=str, help="Path to predictions JSONL file")
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: BALLDONTLIE_API_KEY not set in environment")
        sys.exit(1)

    log_file = args.file or get_default_log_file()

    # Determine date filter
    date_filter = None
    if args.date:
        date_filter = [args.date]
    elif args.last:
        today = datetime.now()
        date_filter = [
            (today - timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(args.last)
        ]

    print("Loading predictions from: {}".format(log_file))
    predictions_by_date = load_predictions(log_file, date_filter)

    if not predictions_by_date:
        print("No predictions found for the specified filter.")
        sys.exit(0)

    total_preds = sum(len(v) for v in predictions_by_date.values())
    dates = sorted(predictions_by_date.keys())
    print("Found {} predictions across {} date(s): {}".format(
        total_preds, len(dates), ", ".join(dates)))

    # Load line snapshots for CLV (optional, graceful if missing)
    snapshot_index = load_line_snapshots()
    if snapshot_index:
        snapshot_games = sum(len(v) for v in snapshot_index.values())
        print("Loaded line snapshots for {} games ({} total prop entries)".format(
            len(snapshot_index), snapshot_games))
    else:
        print("No line snapshot data found (CLV will be unavailable)")

    # Fetch actuals and grade each date
    all_results = []
    for date_str in dates:
        preds = predictions_by_date[date_str]
        game_ids = set(p.get("game_id") for p in preds if p.get("game_id"))

        print("\n  Fetching box scores for {} ({} games, {} predictions)...".format(
            date_str, len(game_ids), len(preds)))

        actual_stats = build_actual_stats_for_date(date_str, game_ids)
        if not actual_stats:
            print("  WARNING: No final box scores found for {}".format(date_str))
            # Mark all as unresolved
            for pred in preds:
                all_results.append({
                    "player": pred.get("player", ""),
                    "prop_type": pred.get("prop_type", ""),
                    "side": pred.get("side", "over"),
                    "line": pred.get("line", 0),
                    "predicted_value": pred.get("predicted_value", 0),
                    "actual": None,
                    "hit": None,
                    "push": False,
                    "status": "NO_BOX_SCORE",
                    "profit": 0.0,
                    "edge_percent": pred.get("edge_percent", 0),
                    "confidence": pred.get("confidence", 0),
                    "confidence_stars": pred.get("confidence_stars", 1),
                    "game_id": pred.get("game_id"),
                    "matchup": pred.get("matchup", ""),
                })
            continue

        graded = grade_predictions_for_date(preds, actual_stats, snapshot_index=snapshot_index)
        resolved = [r for r in graded if r["actual"] is not None]
        print("  Graded {}/{} predictions for {}".format(len(resolved), len(graded), date_str))
        all_results.extend(graded)

    # Print the full report
    print_report(all_results, dates)

    # Save detailed results to JSON
    out_file = "outputs/grading_report_{}.json".format(
        datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs("outputs", exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({
            "generated": datetime.now().isoformat(),
            "log_file": log_file,
            "dates": dates,
            "results": all_results,
        }, f, indent=2)
    print("Detailed results saved to: {}".format(out_file))

    # Update empirical filter with all graded data
    try:
        from v2.pick_filter import EmpiricalFilter
        ef = EmpiricalFilter()
        ef.update_from_grading(all_results, window_days=30)
        summary = ef.get_recommended_strategy()
        print("\n  EMPIRICAL FILTER UPDATE: {}".format(summary))
    except Exception as e:
        print("  (Filter update skipped: {})".format(e))


if __name__ == "__main__":
    main()
