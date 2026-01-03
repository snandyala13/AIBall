#!/usr/bin/env python3
"""
Analyze yesterday's games with the IMPROVED model and generate a hit/miss report.

This uses the NEW projection engine with:
- L3 weighted at 40%
- Minutes trend detection
- Streak detection (hot/cold)
- Opponent-specific history
- Position-based matchups
- Granular rest days

Usage:
    python analyze_yesterday.py                    # Analyze yesterday
    python analyze_yesterday.py --date 2026-01-02  # Analyze specific date
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import requests

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import DataPipeline
from model_engine import ModelEngine, PropType
from ml_filter import MLFilter
from cache_db import get_cache

API_KEY = "d2df7f14-636a-4ff2-8005-35e768d1745f"

# Prop type mapping
PROP_MAP = {
    "points": PropType.POINTS,
    "rebounds": PropType.REBOUNDS,
    "assists": PropType.ASSISTS,
    "threes": PropType.THREES,
    "pts_rebs_asts": PropType.PRA,
    "steals": PropType.STEALS,
    "blocks": PropType.BLOCKS,
}


def get_games_for_date(date_str: str):
    """Get all games for a specific date."""
    url = f"https://api.balldontlie.io/v1/games"
    params = {"dates[]": date_str}
    headers = {"Authorization": API_KEY}

    resp = requests.get(url, params=params, headers=headers)
    if resp.status_code == 200:
        return resp.json().get("data", [])
    return []


def get_game_stats(game_id: int):
    """Get player stats for a specific game."""
    url = f"https://api.balldontlie.io/v1/stats"
    params = {"game_ids[]": game_id, "per_page": 100}
    headers = {"Authorization": API_KEY}

    resp = requests.get(url, params=params, headers=headers)
    if resp.status_code == 200:
        return resp.json().get("data", [])
    return []


def normalize_market_name(market: str) -> str:
    """Convert cache market name to our prop type."""
    # Remove 'player_' prefix and handle special cases
    market = market.lower().replace('player_', '')
    mapping = {
        'points': 'points',
        'rebounds': 'rebounds',
        'assists': 'assists',
        'threes': 'threes',
        'three_pointers': 'threes',
        'steals': 'steals',
        'blocks': 'blocks',
        'points_rebounds_assists': 'pts_rebs_asts',
        'pts_rebs_asts': 'pts_rebs_asts',
    }
    return mapping.get(market, market)


def get_props_from_cache(cache, game_id: int, home_team: str, away_team: str, date_str: str):
    """Get historical props from cache, using team names to find event_id."""

    # Cache uses UTC dates - US evening games appear as next day in UTC
    # So check both the requested date and next day
    from datetime import datetime, timedelta
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    next_day = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d")

    # Try to find the event_id in historical_games (check both dates)
    historical_games = cache.get_historical_games_for_date(date_str)
    historical_games.extend(cache.get_historical_games_for_date(next_day))

    # Extract key words from team names for matching
    home_words = set(home_team.lower().split())
    away_words = set(away_team.lower().split())

    event_id = None
    for hg in historical_games:
        cache_home = hg.get('home_team', '').lower()
        cache_away = hg.get('away_team', '').lower()
        cache_home_words = set(cache_home.split())
        cache_away_words = set(cache_away.split())

        # Check if home team matches (at least 1 word in common)
        home_match = len(home_words & cache_home_words) > 0
        away_match = len(away_words & cache_away_words) > 0

        if home_match and away_match:
            event_id = hg.get('event_id')
            break

    if not event_id:
        # Fallback: match any word from either team
        for hg in historical_games:
            cache_home = hg.get('home_team', '').lower()
            cache_away = hg.get('away_team', '').lower()

            # Match any team word
            all_words = home_words | away_words
            if any(word in cache_home or word in cache_away for word in all_words if len(word) > 3):
                event_id = hg.get('event_id')
                break

    if not event_id:
        return {}

    # Get props for this event
    props = cache.get_historical_props_for_game(event_id, bookmaker="draftkings")
    if not props:
        props = cache.get_historical_props_for_game(event_id, bookmaker="fanduel")

    # Organize by player -> market -> line data (normalize market names)
    player_props = defaultdict(dict)
    for prop in props:
        player_name = prop["player_name"]
        market = normalize_market_name(prop["market"])
        player_props[player_name][market] = {
            "line": prop["line"],
            "over_odds": prop["over_odds"],
            "under_odds": prop["under_odds"]
        }

    return player_props


def calculate_actual_stat(player_stats: dict, prop_type: str) -> float:
    """Calculate actual stat value from box score."""
    pts = player_stats.get("pts", 0) or 0
    reb = player_stats.get("reb", 0) or 0
    ast = player_stats.get("ast", 0) or 0
    fg3m = player_stats.get("fg3m", 0) or 0
    stl = player_stats.get("stl", 0) or 0
    blk = player_stats.get("blk", 0) or 0

    if prop_type == "points":
        return pts
    elif prop_type == "rebounds":
        return reb
    elif prop_type == "assists":
        return ast
    elif prop_type == "threes":
        return fg3m
    elif prop_type == "pts_rebs_asts":
        return pts + reb + ast
    elif prop_type == "steals":
        return stl
    elif prop_type == "blocks":
        return blk
    return 0


def analyze_game(pipeline: DataPipeline, engine: ModelEngine, ml_filter: MLFilter,
                 cache, game: dict, game_date: str) -> list:
    """Analyze a single game using the IMPROVED model."""

    game_id = game.get("id")
    home_team = game.get("home_team", {}).get("full_name", "")
    away_team = game.get("visitor_team", {}).get("full_name", "")

    print(f"\n  {away_team} @ {home_team}...")

    # Get actual stats for this game
    stats_data = get_game_stats(game_id)
    if not stats_data:
        print(f"    No stats available")
        return []

    # Build player stats lookup
    player_stats = {}
    for stat_line in stats_data:
        player = stat_line.get("player", {})
        name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
        player_stats[name] = stat_line

    # Get props from cache
    player_props = get_props_from_cache(cache, game_id, home_team, away_team, game_date)

    if not player_props:
        print(f"    No historical props in cache")
        return []

    print(f"    Found props for {len(player_props)} players")

    # Build game context
    try:
        context = pipeline.build_game_context(home_team, away_team)
        if not context:
            print(f"    Could not build context")
            return []
    except Exception as e:
        print(f"    Error building context: {e}")
        return []

    results = []

    # Get list of players who have props
    player_names = list(player_props.keys())

    # Use engine.analyze_game to get projections for all players (like backtest does)
    try:
        analysis = engine.analyze_game(context, player_names, injured_player_names=[])
        projections = analysis.get("projections", [])
    except Exception as e:
        print(f"    Error in analyze_game: {e}")
        return []

    # Process each projection (one per player-prop combination)
    from model_engine import EdgeCalculator
    calc = EdgeCalculator()

    for proj in projections:
        player_name = proj.player_name

        # Skip if player didn't play
        if player_name not in player_stats:
            continue

        # Check minutes
        minutes = player_stats.get(player_name, {}).get("min", "0")
        if isinstance(minutes, str):
            try:
                if ":" in minutes:
                    parts = minutes.split(":")
                    minutes = int(parts[0]) + int(parts[1]) / 60
                else:
                    minutes = float(minutes) if minutes else 0
            except:
                minutes = 0

        if minutes < 5:
            continue

        # Get props for this player
        props = player_props.get(player_name, {})
        if not props:
            continue

        # Get prop type value from projection
        prop_type = proj.prop_type.value  # e.g., "points"

        if prop_type not in props:
            continue

        line_data = props[prop_type]
        line = line_data.get("line")
        over_odds = line_data.get("over_odds", -110)

        if line is None:
            continue

        # Get actual result
        actual = calculate_actual_stat(player_stats.get(player_name, {}), prop_type)

        # Get projection and std_dev from the Projection object
        projection_val = proj.projected_value
        std_dev = proj.std_dev

        # Recalculate edge with real line (like backtest does)
        over_prob = calc.calculate_over_probability(projection_val, line, std_dev, prop_type)
        under_prob = 1 - over_prob

        # Calculate implied probability from odds
        if over_odds < 0:
            implied_over = abs(over_odds) / (abs(over_odds) + 100)
        else:
            implied_over = 100 / (over_odds + 100)

        edge_over = (over_prob - implied_over) * 100
        edge_under = (under_prob - (1 - implied_over)) * 100

        # Pick direction with more edge
        if edge_over > edge_under:
            pick = "OVER"
            edge = edge_over
            probability = over_prob
        else:
            pick = "UNDER"
            edge = edge_under
            probability = under_prob

        # Skip low edge
        if edge < 3:
            continue

        # Apply ML filter
        try:
            ml_prob = ml_filter.get_ensemble_probability(
                probability=probability,
                edge=edge,
                confidence=proj.confidence,
                diff_pct=abs(projection_val - line) / line * 100 if line else 0,
                line=line,
                is_over=(pick == "OVER"),
                prop_type=prop_type
            )
        except Exception:
            ml_prob = None

        # Determine if hit
        if pick == "OVER":
            hit = actual > line
        else:
            hit = actual < line

        results.append({
            "player": player_name,
            "prop_type": prop_type,
            "line": line,
            "projection": round(projection_val, 1),
            "actual": actual,
            "pick": pick,
            "edge": round(edge, 1),
            "probability": round(probability * 100, 1),
            "ml_prob": round(ml_prob * 100, 1) if ml_prob else None,
            "hit": hit,
            "game": f"{away_team} @ {home_team}",
            "minutes": round(minutes, 1)
        })

    print(f"    Generated {len(results)} predictions")
    return results


def generate_report(results: list, date_str: str) -> str:
    """Generate a formatted report of predictions and results."""

    if not results:
        return f"No predictions generated for {date_str}"

    # Calculate overall stats
    total = len(results)
    hits = sum(1 for r in results if r["hit"])
    hit_rate = hits / total * 100 if total > 0 else 0

    # By direction
    overs = [r for r in results if r["pick"] == "OVER"]
    unders = [r for r in results if r["pick"] == "UNDER"]
    over_hits = sum(1 for r in overs if r["hit"])
    under_hits = sum(1 for r in unders if r["hit"])

    # By edge tier
    high_edge = [r for r in results if r["edge"] >= 10]
    mid_edge = [r for r in results if 5 <= r["edge"] < 10]
    low_edge = [r for r in results if 3 <= r["edge"] < 5]

    # By ML probability
    ml_60_plus = [r for r in results if r["ml_prob"] and r["ml_prob"] >= 60]
    ml_55_plus = [r for r in results if r["ml_prob"] and r["ml_prob"] >= 55]
    ml_50_plus = [r for r in results if r["ml_prob"] and r["ml_prob"] >= 50]

    # By prop type
    by_prop = defaultdict(list)
    for r in results:
        by_prop[r["prop_type"]].append(r)

    report = []
    report.append("=" * 70)
    report.append(f"  {date_str} ANALYSIS - IMPROVED MODEL")
    report.append("  (L3 weighting, streak detection, opponent history)")
    report.append("=" * 70)
    report.append("")

    # Overall summary
    report.append(f"OVERALL: {hits}/{total} = {hit_rate:.1f}%")
    report.append(f"  Breakeven at -110: 52.4%")
    status = "PROFITABLE" if hit_rate > 52.4 else "NOT PROFITABLE"
    report.append(f"  {status}")
    report.append("")

    # By direction
    report.append("BY DIRECTION:")
    if overs:
        over_rate = over_hits / len(overs) * 100
        report.append(f"  OVER:  {over_hits}/{len(overs)} = {over_rate:.1f}%")
    if unders:
        under_rate = under_hits / len(unders) * 100
        report.append(f"  UNDER: {under_hits}/{len(unders)} = {under_rate:.1f}%")
    report.append("")

    # By edge tier
    report.append("BY EDGE LEVEL:")
    if high_edge:
        high_hits = sum(1 for r in high_edge if r["hit"])
        report.append(f"  10%+:  {high_hits}/{len(high_edge)} = {high_hits/len(high_edge)*100:.1f}%")
    if mid_edge:
        mid_hits = sum(1 for r in mid_edge if r["hit"])
        report.append(f"  5-10%: {mid_hits}/{len(mid_edge)} = {mid_hits/len(mid_edge)*100:.1f}%")
    if low_edge:
        low_hits = sum(1 for r in low_edge if r["hit"])
        report.append(f"  3-5%:  {low_hits}/{len(low_edge)} = {low_hits/len(low_edge)*100:.1f}%")
    report.append("")

    # By ML probability (the key filter!)
    report.append("BY ML FILTER:")
    if ml_60_plus:
        ml60_hits = sum(1 for r in ml_60_plus if r["hit"])
        report.append(f"  ML >= 60%: {ml60_hits}/{len(ml_60_plus)} = {ml60_hits/len(ml_60_plus)*100:.1f}%")
    if ml_55_plus:
        ml55_hits = sum(1 for r in ml_55_plus if r["hit"])
        report.append(f"  ML >= 55%: {ml55_hits}/{len(ml_55_plus)} = {ml55_hits/len(ml_55_plus)*100:.1f}%")
    if ml_50_plus:
        ml50_hits = sum(1 for r in ml_50_plus if r["hit"])
        report.append(f"  ML >= 50%: {ml50_hits}/{len(ml_50_plus)} = {ml50_hits/len(ml_50_plus)*100:.1f}%")
    report.append("")

    # By prop type
    report.append("BY PROP TYPE:")
    for prop_type, prop_results in sorted(by_prop.items()):
        prop_hits = sum(1 for r in prop_results if r["hit"])
        prop_rate = prop_hits / len(prop_results) * 100
        report.append(f"  {prop_type.upper()}: {prop_hits}/{len(prop_results)} = {prop_rate:.1f}%")
    report.append("")

    # Best ML picks detail
    report.append("-" * 70)
    report.append("TOP ML PICKS (>= 55%)")
    report.append("-" * 70)

    top_picks = sorted([r for r in results if r["ml_prob"] and r["ml_prob"] >= 55],
                       key=lambda x: x["ml_prob"], reverse=True)

    for r in top_picks[:20]:
        status = "HIT" if r["hit"] else "MISS"
        emoji = "v" if r["hit"] else "x"
        diff = r["actual"] - r["line"]
        diff_str = f"+{diff:.1f}" if diff > 0 else f"{diff:.1f}"

        report.append(
            f"  [{emoji}] {r['player'][:20]:<20} {r['prop_type']:<12} "
            f"{r['pick']:<5} {r['line']:>5.1f} -> {r['actual']:>5.1f} ({diff_str}) "
            f"ML:{r['ml_prob']:.0f}% Edge:{r['edge']:.0f}%"
        )

    report.append("")

    # All picks by game
    report.append("-" * 70)
    report.append("ALL PICKS BY GAME")
    report.append("-" * 70)

    by_game = defaultdict(list)
    for r in results:
        by_game[r["game"]].append(r)

    for game, game_results in by_game.items():
        game_hits = sum(1 for r in game_results if r["hit"])
        report.append(f"\n{game} ({game_hits}/{len(game_results)} hit)")
        for r in sorted(game_results, key=lambda x: -(x["ml_prob"] or 0)):
            emoji = "v" if r["hit"] else "x"
            report.append(
                f"  [{emoji}] {r['player'][:16]:<16} {r['prop_type']:<10} "
                f"{r['pick']:<5} {r['line']:>5.1f} -> {r['actual']:>5.1f} "
                f"(proj:{r['projection']:>5.1f}) ML:{r['ml_prob'] or 0:.0f}%"
            )

    report.append("")
    report.append("=" * 70)

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Analyze games with improved model')
    parser.add_argument('--date', type=str, help='Date to analyze (YYYY-MM-DD)')
    args = parser.parse_args()

    # Determine date
    if args.date:
        date_str = args.date
    else:
        yesterday = datetime.now() - timedelta(days=1)
        date_str = yesterday.strftime("%Y-%m-%d")

    print(f"\nAnalyzing {date_str} with IMPROVED MODEL...")
    print("Using: L3 weighting, streak detection, opponent history")
    print("=" * 50)

    # Initialize
    pipeline = DataPipeline(API_KEY)
    engine = ModelEngine(pipeline)
    ml_filter = MLFilter()
    cache = get_cache()

    # Get games for the date
    games = get_games_for_date(date_str)

    if not games:
        print(f"No games found for {date_str}")
        return

    print(f"Found {len(games)} games")

    # Analyze each game
    all_results = []
    for game in games:
        results = analyze_game(pipeline, engine, ml_filter, cache, game, date_str)
        all_results.extend(results)

    # Generate report
    report = generate_report(all_results, date_str)
    print("\n" + report)

    # Save report
    os.makedirs("outputs", exist_ok=True)

    # Save text report
    report_file = f"outputs/improved_model_report_{date_str}.txt"
    with open(report_file, "w") as f:
        f.write(report)
    print(f"\nSaved report to: {report_file}")

    # Save JSON for further analysis
    json_file = f"outputs/improved_model_results_{date_str}.json"
    with open(json_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved results to: {json_file}")


if __name__ == "__main__":
    main()
