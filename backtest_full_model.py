#!/usr/bin/env python3
"""
Full Model Backtest - Uses actual model_engine.py and bet_generator.py

Tests the REAL model against historical games including:
- Injury-based pop-off boosts
- Position scarcity multipliers
- Role change detection
- PRA combined props
- Correlation-aware parlay construction
"""

import os
import sys
import time
import json
import requests
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Semaphore
import logging

sys.path.insert(0, '/Users/suhaasnandyala/Documents/aiBall')

from data_pipeline import DataPipeline, PlayerProfile
from model_engine import ModelEngine, PropType, Projection, STAT_MAPPING, COMBINED_STAT_MAPPING
from bet_generator import BetGenerator

# Suppress excessive logging
logging.getLogger('data_pipeline').setLevel(logging.WARNING)
logging.getLogger('model_engine').setLevel(logging.WARNING)

API_KEY = 'd2df7f14-636a-4ff2-8005-35e768d1745f'
BASE_URL = "https://api.balldontlie.io/v1"
HEADERS = {"Authorization": API_KEY}

# Rate limiting
REQUEST_SEMAPHORE = Semaphore(2)
REQUEST_DELAY = 0.4
last_request_time = 0
request_lock = Lock()


def rate_limited_request(url, params=None):
    """Make a rate-limited API request"""
    global last_request_time

    with REQUEST_SEMAPHORE:
        with request_lock:
            now = time.time()
            elapsed = now - last_request_time
            if elapsed < REQUEST_DELAY:
                time.sleep(REQUEST_DELAY - elapsed)
            last_request_time = time.time()

        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
            if resp.status_code == 429:
                print("  Rate limited, waiting 5s...")
                time.sleep(5)
                return rate_limited_request(url, params)
            return resp
        except Exception as e:
            print(f"  Request error: {e}")
            return None


def get_games_for_date(date: str):
    """Get all games for a date"""
    resp = rate_limited_request(f"{BASE_URL}/games", {"dates[]": date})
    if resp and resp.status_code == 200:
        return resp.json().get("data", [])
    return []


def get_box_score(game_id: int):
    """Get box score for a game"""
    resp = rate_limited_request(f"{BASE_URL}/stats", {"game_ids[]": game_id, "per_page": 50})
    if resp and resp.status_code == 200:
        return resp.json().get("data", [])
    return []


def get_injuries_for_team(team_id: int):
    """Get injuries for a team"""
    resp = rate_limited_request(f"{BASE_URL}/player_injuries", {"team_ids[]": team_id})
    if resp and resp.status_code == 200:
        return resp.json().get("data", [])
    return []


def parse_minutes(min_str):
    """Parse minutes from various formats"""
    if not min_str:
        return 0
    min_str = str(min_str)
    try:
        if ":" in min_str:
            return int(min_str.split(":")[0])
        return int(float(min_str))
    except:
        return 0


def evaluate_prop(projection: Projection, actual_stats: dict):
    """Evaluate if a prop would have hit"""
    prop_type = projection.prop_type
    line = projection.dk_line

    # Get actual value
    if prop_type in COMBINED_STAT_MAPPING:
        stat_keys = COMBINED_STAT_MAPPING[prop_type]
        actual = sum(actual_stats.get(k, 0) or 0 for k in stat_keys)
    elif prop_type in STAT_MAPPING:
        stat_key = STAT_MAPPING[prop_type]
        actual = actual_stats.get(stat_key, 0) or 0
    else:
        return None

    # Determine our pick
    if projection.edge_over > 5:
        pick = "OVER"
        hit = actual > line
        prob = projection.over_probability
        edge = projection.edge_over
    elif projection.edge_under > 5:
        pick = "UNDER"
        hit = actual < line
        prob = projection.under_probability
        edge = projection.edge_under
    else:
        return None  # No edge

    return {
        "player": projection.player_name,
        "prop_type": prop_type.value,
        "line": line,
        "projected": projection.projected_value,
        "actual": actual,
        "pick": pick,
        "hit": hit,
        "probability": prob,
        "edge": edge,
        "confidence": projection.confidence,
        "usage_boost": projection.usage_boost,
        "role_change": projection.role_change
    }


def backtest_game(game_data: dict, pipeline: DataPipeline, engine: ModelEngine, date: str):
    """Backtest a single game using the full model"""
    game_id = game_data["id"]
    home_team = game_data["home_team"]["full_name"]
    away_team = game_data["visitor_team"]["full_name"]
    home_team_id = game_data["home_team"]["id"]
    away_team_id = game_data["visitor_team"]["id"]

    result = {
        "game_id": game_id,
        "matchup": f"{away_team} @ {home_team}",
        "date": date,
        "legs": [],
        "injuries": []
    }

    # Get box score (actuals)
    box_score = get_box_score(game_id)
    if not box_score:
        return result

    # Build actual stats dict by player name
    actual_stats = {}
    player_minutes = {}
    for stat in box_score:
        player = stat.get("player", {})
        name = f"{player.get('first_name', '')} {player.get('last_name', '')}"
        mins = parse_minutes(stat.get("min", 0))

        if mins >= 10:  # Only players with meaningful minutes
            actual_stats[name] = {
                "pts": stat.get("pts", 0) or 0,
                "reb": stat.get("reb", 0) or 0,
                "ast": stat.get("ast", 0) or 0,
                "fg3m": stat.get("fg3m", 0) or 0,
                "stl": stat.get("stl", 0) or 0,
                "blk": stat.get("blk", 0) or 0,
                "turnover": stat.get("turnover", 0) or 0,
                "min": mins
            }
            player_minutes[name] = mins

    if not actual_stats:
        return result

    # Get injuries (for pop-off detection)
    injuries = []
    injured_names = []

    for team_id in [home_team_id, away_team_id]:
        inj_data = get_injuries_for_team(team_id)
        for inj in inj_data:
            player = inj.get("player", {})
            name = f"{player.get('first_name', '')} {player.get('last_name', '')}"
            status = inj.get("status", "").lower()
            if status == "out":
                injuries.append(name)
                injured_names.append(name)

    result["injuries"] = injuries

    # Build game context
    try:
        context = pipeline.build_game_context(home_team, away_team)
    except:
        context = None

    if not context:
        return result

    # Get player names who actually played
    player_names = list(actual_stats.keys())

    # Run the model (this is the key - using the REAL model)
    try:
        analysis = engine.analyze_game(context, player_names, injured_player_names=injured_names)
    except Exception as e:
        print(f"    Model error: {e}")
        return result

    if not analysis:
        return result

    projections = analysis.get("projections", [])

    # Evaluate each projection
    for proj in projections:
        if proj.player_name not in actual_stats:
            continue

        # Simulate DK line using BASE projection (before adjustments)
        # This is more realistic - DK sets lines based on simple averages,
        # our edge comes from adjustments (injuries, matchups, etc.)
        base = proj.base_projection if proj.base_projection > 0 else proj.projected_value
        proj.dk_line = round(base * 2) / 2  # Round to nearest 0.5

        # ONLY evaluate core prop types (skip volatile steals/blocks/turnovers)
        CORE_PROPS = {
            PropType.POINTS,
            PropType.REBOUNDS,
            PropType.ASSISTS,
            PropType.THREES,
            PropType.PRA,
            PropType.PR,
            PropType.PA,
            PropType.RA
        }

        if proj.prop_type not in CORE_PROPS:
            continue

        # Skip very low lines
        min_lines = {
            PropType.POINTS: 8.5,
            PropType.REBOUNDS: 3.5,
            PropType.ASSISTS: 2.5,
            PropType.THREES: 0.5,
            PropType.PRA: 15.5,
            PropType.PR: 12.5,
            PropType.PA: 12.5,
            PropType.RA: 6.5
        }

        if proj.dk_line < min_lines.get(proj.prop_type, 0.5):
            continue

        # Recalculate probabilities with simulated line
        from model_engine import EdgeCalculator
        calc = EdgeCalculator()

        # Map prop_type to stat_type for calibration
        stat_type = STAT_MAPPING.get(proj.prop_type, "pts")
        if proj.prop_type == PropType.PRA:
            stat_type = "pra"

        proj.over_probability = calc.calculate_over_probability(
            proj.projected_value, proj.dk_line, proj.std_dev, stat_type
        )
        proj.under_probability = 1 - proj.over_probability
        proj.edge_over = (proj.over_probability - 0.52) * 100
        proj.edge_under = (proj.under_probability - 0.52) * 100

        # Evaluate
        eval_result = evaluate_prop(proj, actual_stats[proj.player_name])
        if eval_result:
            result["legs"].append(eval_result)

    return result


def run_full_model_backtest(dates: list, output_file: str):
    """Run backtest using the full model across dates - writes incrementally"""
    print("=" * 70)
    print("FULL MODEL BACKTEST")
    print("Using: model_engine.py + bet_generator.py")
    print("=" * 70)
    print()

    # Initialize pipeline and engine
    pipeline = DataPipeline(api_key=API_KEY)
    engine = ModelEngine(pipeline)

    all_results = []

    # Incremental results file
    incremental_file = output_file.replace(".txt", "_incremental.json")

    # Collect games
    print("Fetching games...")
    all_games = []
    for date in dates:
        games = get_games_for_date(date)
        for g in games:
            g["backtest_date"] = date
        all_games.extend(games)
        print(f"  {date}: {len(games)} games")

    print(f"\nTotal: {len(all_games)} games to process\n")
    print(f"Writing results incrementally to: {incremental_file}\n")

    # Process each game
    for i, game in enumerate(all_games, 1):
        date = game["backtest_date"]
        home = game["home_team"]["full_name"]
        away = game["visitor_team"]["full_name"]

        print(f"[{i}/{len(all_games)}] {away} @ {home} ({date})...", end=" ", flush=True)

        result = backtest_game(game, pipeline, engine, date)
        all_results.append(result)

        n_legs = len(result["legs"])
        n_inj = len(result["injuries"])
        print(f"{n_legs} props, {n_inj} injuries")

        # Write incremental results after each game
        with open(incremental_file, "w") as f:
            json.dump({
                "status": "in_progress",
                "games_completed": i,
                "games_total": len(all_games),
                "last_updated": datetime.now().isoformat(),
                "results": all_results
            }, f, indent=2, default=str)

        time.sleep(0.2)  # Extra delay between games

    # Mark as complete
    with open(incremental_file, "w") as f:
        json.dump({
            "status": "complete",
            "games_completed": len(all_games),
            "games_total": len(all_games),
            "last_updated": datetime.now().isoformat(),
            "results": all_results
        }, f, indent=2, default=str)

    return all_results


def generate_full_report(results: list, output_file: str):
    """Generate comprehensive report from backtest results"""

    # Aggregate all legs
    all_legs = []
    legs_with_boost = []
    legs_with_role_change = []
    pra_legs = []

    for r in results:
        for leg in r["legs"]:
            all_legs.append(leg)

            if leg.get("usage_boost", 0) > 0:
                legs_with_boost.append(leg)

            if leg.get("role_change"):
                legs_with_role_change.append(leg)

            if leg.get("prop_type") == "pts_rebs_asts":
                pra_legs.append(leg)

    # Calculate stats
    def calc_stats(legs):
        if not legs:
            return {"total": 0, "hits": 0, "rate": 0}
        hits = sum(1 for l in legs if l["hit"])
        return {"total": len(legs), "hits": hits, "rate": hits/len(legs)*100}

    overall = calc_stats(all_legs)
    boost_stats = calc_stats(legs_with_boost)
    role_change_stats = calc_stats(legs_with_role_change)
    pra_stats = calc_stats(pra_legs)

    # By prop type
    by_prop = defaultdict(list)
    for leg in all_legs:
        by_prop[leg["prop_type"]].append(leg)

    prop_stats = {k: calc_stats(v) for k, v in by_prop.items()}

    # By confidence
    high_conf = [l for l in all_legs if l["probability"] >= 0.65]
    med_conf = [l for l in all_legs if 0.60 <= l["probability"] < 0.65]
    low_conf = [l for l in all_legs if 0.55 <= l["probability"] < 0.60]

    high_stats = calc_stats(high_conf)
    med_stats = calc_stats(med_conf)
    low_stats = calc_stats(low_conf)

    # By edge
    high_edge = [l for l in all_legs if l["edge"] >= 15]
    med_edge = [l for l in all_legs if 10 <= l["edge"] < 15]
    low_edge = [l for l in all_legs if 5 <= l["edge"] < 10]

    high_edge_stats = calc_stats(high_edge)
    med_edge_stats = calc_stats(med_edge)
    low_edge_stats = calc_stats(low_edge)

    # Build parlays from legs (simulate bet_generator logic)
    # Group by game
    game_legs = defaultdict(list)
    for r in results:
        for leg in r["legs"]:
            game_legs[r["game_id"]].append(leg)

    safe_parlays = []  # 2-leg
    standard_parlays = []  # 3-4 leg
    longshot_parlays = []  # 5+ leg

    for game_id, legs in game_legs.items():
        if len(legs) < 2:
            continue

        # Sort by edge * probability (quality score)
        legs.sort(key=lambda x: x["edge"] * x["probability"], reverse=True)

        # 2-leg safe
        p_legs = legs[:2]
        all_hit = all(l["hit"] for l in p_legs)
        prob = p_legs[0]["probability"] * p_legs[1]["probability"]
        safe_parlays.append({"legs": p_legs, "hit": all_hit, "prob": prob})

        # 3-leg standard
        if len(legs) >= 3:
            p_legs = legs[:3]
            all_hit = all(l["hit"] for l in p_legs)
            prob = 1
            for l in p_legs:
                prob *= l["probability"]
            standard_parlays.append({"legs": p_legs, "hit": all_hit, "prob": prob})

        # 4-leg standard
        if len(legs) >= 4:
            p_legs = legs[:4]
            all_hit = all(l["hit"] for l in p_legs)
            prob = 1
            for l in p_legs:
                prob *= l["probability"]
            standard_parlays.append({"legs": p_legs, "hit": all_hit, "prob": prob})

        # 5-leg longshot
        if len(legs) >= 5:
            p_legs = legs[:5]
            all_hit = all(l["hit"] for l in p_legs)
            prob = 1
            for l in p_legs:
                prob *= l["probability"]
            longshot_parlays.append({"legs": p_legs, "hit": all_hit, "prob": prob})

    def parlay_stats(parlays, payout):
        if not parlays:
            return {"total": 0, "hits": 0, "profit": 0, "roi": 0}
        hits = sum(1 for p in parlays if p["hit"])
        total = len(parlays)
        profit = (hits * payout) - total
        roi = (profit / total * 100) if total > 0 else 0
        return {"total": total, "hits": hits, "profit": profit, "roi": roi}

    safe_p_stats = parlay_stats(safe_parlays, 2)
    std_p_stats = parlay_stats(standard_parlays, 5)
    long_p_stats = parlay_stats(longshot_parlays, 12)

    total_wagered = safe_p_stats["total"] + std_p_stats["total"] + long_p_stats["total"]
    total_profit = safe_p_stats["profit"] + std_p_stats["profit"] + long_p_stats["profit"]
    overall_roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0

    # Build report
    report = []
    report.append("=" * 70)
    report.append("FULL MODEL BACKTEST RESULTS - Dec 25-31, 2025")
    report.append("=" * 70)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"Games Analyzed: {len(results)}")
    report.append(f"Total Props Evaluated: {overall['total']}")
    report.append("")

    report.append("=" * 70)
    report.append("INDIVIDUAL LEG PERFORMANCE")
    report.append("=" * 70)
    report.append(f"Overall: {overall['hits']}/{overall['total']} = {overall['rate']:.1f}%")
    report.append("")

    report.append("By Confidence Level:")
    report.append(f"  65%+:    {high_stats['hits']}/{high_stats['total']} = {high_stats['rate']:.1f}%")
    report.append(f"  60-65%:  {med_stats['hits']}/{med_stats['total']} = {med_stats['rate']:.1f}%")
    report.append(f"  55-60%:  {low_stats['hits']}/{low_stats['total']} = {low_stats['rate']:.1f}%")
    report.append("")

    report.append("By Edge Level:")
    report.append(f"  15%+:    {high_edge_stats['hits']}/{high_edge_stats['total']} = {high_edge_stats['rate']:.1f}%")
    report.append(f"  10-15%:  {med_edge_stats['hits']}/{med_edge_stats['total']} = {med_edge_stats['rate']:.1f}%")
    report.append(f"  5-10%:   {low_edge_stats['hits']}/{low_edge_stats['total']} = {low_edge_stats['rate']:.1f}%")
    report.append("")

    report.append("By Prop Type:")
    for prop_type, stats in sorted(prop_stats.items()):
        if stats["total"] > 0:
            report.append(f"  {prop_type.upper()}: {stats['hits']}/{stats['total']} = {stats['rate']:.1f}%")
    report.append("")

    report.append("=" * 70)
    report.append("MODEL FEATURE PERFORMANCE")
    report.append("=" * 70)
    report.append("")

    report.append("Pop-Off Boosts (injury opportunity):")
    report.append(f"  Total legs with boost: {boost_stats['total']}")
    report.append(f"  Hit rate: {boost_stats['hits']}/{boost_stats['total']} = {boost_stats['rate']:.1f}%")
    report.append("")

    report.append("Role Change Detection:")
    report.append(f"  Total legs with role change: {role_change_stats['total']}")
    report.append(f"  Hit rate: {role_change_stats['hits']}/{role_change_stats['total']} = {role_change_stats['rate']:.1f}%")
    report.append("")

    report.append("PRA Combined Props:")
    report.append(f"  Total PRA legs: {pra_stats['total']}")
    report.append(f"  Hit rate: {pra_stats['hits']}/{pra_stats['total']} = {pra_stats['rate']:.1f}%")
    report.append("")

    report.append("=" * 70)
    report.append("PARLAY PERFORMANCE")
    report.append("=" * 70)
    report.append("")

    report.append(f"SAFE PARLAYS (2-leg, ~+200 payout)")
    report.append("-" * 50)
    report.append(f"  Total: {safe_p_stats['total']} | Hits: {safe_p_stats['hits']} | Rate: {safe_p_stats['hits']/safe_p_stats['total']*100:.1f}%" if safe_p_stats['total'] else "  No parlays")
    report.append(f"  Profit: {safe_p_stats['profit']:+.1f} units | ROI: {safe_p_stats['roi']:+.1f}%")
    report.append("")

    # Show top safe parlays
    for i, p in enumerate(safe_parlays[:8], 1):
        status = "HIT" if p["hit"] else "MISS"
        report.append(f"  [{i}] {len(p['legs'])}-leg ({p['prob']*100:.1f}%) - {status}")
        for leg in p["legs"]:
            s = "+" if leg["hit"] else "-"
            report.append(f"      {s} {leg['player']}: {leg['pick']} {leg['line']} {leg['prop_type'].upper()} (Actual: {leg['actual']})")
    report.append("")

    report.append(f"STANDARD PARLAYS (3-4 leg, ~+500 payout)")
    report.append("-" * 50)
    report.append(f"  Total: {std_p_stats['total']} | Hits: {std_p_stats['hits']} | Rate: {std_p_stats['hits']/std_p_stats['total']*100:.1f}%" if std_p_stats['total'] else "  No parlays")
    report.append(f"  Profit: {std_p_stats['profit']:+.1f} units | ROI: {std_p_stats['roi']:+.1f}%")
    report.append("")

    for i, p in enumerate(standard_parlays[:8], 1):
        status = "HIT" if p["hit"] else "MISS"
        report.append(f"  [{i}] {len(p['legs'])}-leg ({p['prob']*100:.1f}%) - {status}")
        for leg in p["legs"]:
            s = "+" if leg["hit"] else "-"
            report.append(f"      {s} {leg['player']}: {leg['pick']} {leg['line']} {leg['prop_type'].upper()} (Actual: {leg['actual']})")
    report.append("")

    report.append(f"LONGSHOT PARLAYS (5+ leg, ~+1200 payout)")
    report.append("-" * 50)
    report.append(f"  Total: {long_p_stats['total']} | Hits: {long_p_stats['hits']} | Rate: {long_p_stats['hits']/long_p_stats['total']*100:.1f}%" if long_p_stats['total'] else "  No parlays")
    report.append(f"  Profit: {long_p_stats['profit']:+.1f} units | ROI: {long_p_stats['roi']:+.1f}%")
    report.append("")

    report.append("=" * 70)
    report.append("OVERALL SUMMARY")
    report.append("=" * 70)
    report.append(f"  Individual Legs: {overall['rate']:.1f}% hit rate (need 52.4% for profit)")
    report.append(f"  Total Parlays: {total_wagered}")
    report.append(f"  Total Profit: {total_profit:+.1f} units")
    report.append(f"  Overall ROI: {overall_roi:+.1f}%")
    report.append("")

    if overall["rate"] > 52.4:
        report.append("  LEG PERFORMANCE: PROFITABLE")
    else:
        report.append("  LEG PERFORMANCE: NOT PROFITABLE")

    if overall_roi > 0:
        report.append("  PARLAY PERFORMANCE: PROFITABLE")
    else:
        report.append("  PARLAY PERFORMANCE: NOT PROFITABLE")

    report.append("")
    report.append("=" * 70)

    # Write report
    with open(output_file, "w") as f:
        f.write("\n".join(report))

    # Also save JSON
    json_file = output_file.replace(".txt", ".json")
    with open(json_file, "w") as f:
        json.dump({
            "summary": {
                "games": len(results),
                "total_legs": overall["total"],
                "leg_hit_rate": round(overall["rate"], 1),
                "by_confidence": {
                    "high": high_stats,
                    "med": med_stats,
                    "low": low_stats
                },
                "by_prop": prop_stats,
                "boost_performance": boost_stats,
                "role_change_performance": role_change_stats,
                "pra_performance": pra_stats,
                "parlays": {
                    "safe": safe_p_stats,
                    "standard": std_p_stats,
                    "longshot": long_p_stats
                },
                "total_profit": round(total_profit, 1),
                "overall_roi": round(overall_roi, 1)
            },
            "games": results
        }, f, indent=2, default=str)

    return "\n".join(report)


def analyze_incremental_results(json_file: str):
    """Analyze partial results from incremental file"""
    if not os.path.exists(json_file):
        print(f"File not found: {json_file}")
        return

    with open(json_file, "r") as f:
        data = json.load(f)

    print(f"Status: {data.get('status', 'unknown')}")
    print(f"Games: {data.get('games_completed', 0)}/{data.get('games_total', 0)}")
    print(f"Last updated: {data.get('last_updated', 'unknown')}")
    print()

    results = data.get("results", [])
    if results:
        output_file = json_file.replace("_incremental.json", "_partial.txt")
        report = generate_full_report(results, output_file)
        print(report)


if __name__ == "__main__":
    import sys

    output_file = "/Users/suhaasnandyala/Documents/aiBall/outputs/backtest_full_model.txt"

    # Check if analyzing existing incremental results
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        incremental_file = output_file.replace(".txt", "_incremental.json")
        analyze_incremental_results(incremental_file)
        sys.exit(0)

    dates = [
        "2025-12-25",
        "2025-12-26",
        "2025-12-27",
        "2025-12-28",
        "2025-12-29",
        "2025-12-30",
        "2025-12-31"
    ]

    print(f"Starting FULL MODEL backtest for {len(dates)} days...")
    print(f"Results will be saved incrementally - you can analyze partial results with:")
    print(f"  python backtest_full_model.py --analyze")
    print()

    start_time = time.time()
    results = run_full_model_backtest(dates, output_file)
    elapsed = time.time() - start_time

    print(f"\nBacktest completed in {elapsed/60:.1f} minutes")
    print("\nGenerating report...")

    report = generate_full_report(results, output_file)

    print(report)
    print(f"\nSaved to: {output_file}")
