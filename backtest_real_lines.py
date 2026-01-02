#!/usr/bin/env python3
"""
Backtest with REAL Historical Lines from The Odds API

This backtest uses actual DraftKings/FanDuel lines stored in our database,
NOT simulated lines. This gives us a true measure of model performance.
"""

import os
import sys
import time
import json
import requests
from datetime import datetime, timedelta
from collections import defaultdict
from threading import Lock, Semaphore
import logging

sys.path.insert(0, '/Users/suhaasnandyala/Documents/aiBall')

from data_pipeline import DataPipeline
from model_engine import ModelEngine, PropType, STAT_MAPPING, COMBINED_STAT_MAPPING
from cache_db import get_cache

logging.getLogger('data_pipeline').setLevel(logging.WARNING)
logging.getLogger('model_engine').setLevel(logging.WARNING)

API_KEY = 'd2df7f14-636a-4ff2-8005-35e768d1745f'
BASE_URL = "https://api.balldontlie.io/v1"
HEADERS = {"Authorization": API_KEY}

REQUEST_SEMAPHORE = Semaphore(2)
REQUEST_DELAY = 0.4
last_request_time = 0
request_lock = Lock()

# Map PropType to The Odds API market names
PROP_TO_MARKET = {
    PropType.POINTS: "player_points",
    PropType.REBOUNDS: "player_rebounds",
    PropType.ASSISTS: "player_assists",
    PropType.THREES: "player_threes",
    PropType.PRA: "player_points_rebounds_assists",
}

# Reverse mapping for parsing
MARKET_TO_PROP = {v: k for k, v in PROP_TO_MARKET.items()}


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
    """Get all games for a date from balldontlie API"""
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


def match_game_to_historical(game_data: dict, cache, date: str) -> dict:
    """Match a balldontlie game to a historical game from The Odds API

    Handles timezone differences by checking ±1 day (UTC vs local time)
    """
    home_team = game_data["home_team"]["full_name"]
    away_team = game_data["visitor_team"]["full_name"]

    # Check date, date+1, and date-1 to handle UTC timezone differences
    from datetime import datetime, timedelta
    base_date = datetime.strptime(date, "%Y-%m-%d")
    dates_to_check = [
        date,
        (base_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        (base_date - timedelta(days=1)).strftime("%Y-%m-%d"),
    ]

    for check_date in dates_to_check:
        historical_games = cache.get_historical_games_for_date(check_date)

        for hg in historical_games:
            # Exact match
            if hg["home_team"] == home_team and hg["away_team"] == away_team:
                return hg

        # Try partial matching (mascot name)
        for hg in historical_games:
            home_match = home_team.split()[-1] in hg["home_team"]
            away_match = away_team.split()[-1] in hg["away_team"]
            if home_match and away_match:
                return hg

    return None


def get_real_props_for_game(cache, event_id: str) -> dict:
    """Get all real props for a game, organized by player and prop type"""
    props = cache.get_historical_props_for_game(event_id, bookmaker="draftkings")

    # If no DK props, try FanDuel
    if not props:
        props = cache.get_historical_props_for_game(event_id, bookmaker="fanduel")

    # Organize by player -> market -> line data
    player_props = defaultdict(dict)
    for prop in props:
        player_name = prop["player_name"]
        market = prop["market"]
        player_props[player_name][market] = {
            "line": prop["line"],
            "over_odds": prop["over_odds"],
            "under_odds": prop["under_odds"]
        }

    return player_props


def backtest_game_with_real_lines(game_data: dict, pipeline: DataPipeline,
                                   engine: ModelEngine, cache, date: str):
    """Backtest a single game using REAL historical lines"""
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
        "injuries": [],
        "props_matched": 0,
        "props_missing": 0
    }

    # Get historical game from database (handles timezone differences)
    historical_game = match_game_to_historical(game_data, cache, date)

    if not historical_game:
        result["error"] = "No historical game found"
        return result

    event_id = historical_game["event_id"]

    # Get REAL props from database
    real_props = get_real_props_for_game(cache, event_id)
    if not real_props:
        result["error"] = "No props in database"
        return result

    # Get box score (actuals)
    box_score = get_box_score(game_id)
    if not box_score:
        result["error"] = "No box score"
        return result

    # Build actual stats dict by player name
    actual_stats = {}
    for stat in box_score:
        player = stat.get("player", {})
        name = f"{player.get('first_name', '')} {player.get('last_name', '')}"
        mins = parse_minutes(stat.get("min", 0))

        if mins >= 10:
            actual_stats[name] = {
                "pts": stat.get("pts", 0) or 0,
                "reb": stat.get("reb", 0) or 0,
                "ast": stat.get("ast", 0) or 0,
                "fg3m": stat.get("fg3m", 0) or 0,
            }

    if not actual_stats:
        result["error"] = "No player stats"
        return result

    # Get injuries
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
        result["error"] = "No game context"
        return result

    # Get player names who actually played
    player_names = list(actual_stats.keys())

    # Run the model
    try:
        analysis = engine.analyze_game(context, player_names, injured_player_names=injured_names)
    except Exception as e:
        result["error"] = f"Model error: {e}"
        return result

    if not analysis:
        result["error"] = "No analysis"
        return result

    projections = analysis.get("projections", [])

    # Evaluate each projection using REAL lines
    for proj in projections:
        if proj.player_name not in actual_stats:
            continue

        # Get market name for this prop type
        market = PROP_TO_MARKET.get(proj.prop_type)
        if not market:
            continue

        # Get REAL line from database
        player_real_props = real_props.get(proj.player_name, {})
        real_line_data = player_real_props.get(market)

        if not real_line_data:
            result["props_missing"] += 1
            continue

        result["props_matched"] += 1

        # Use REAL DK line
        real_line = real_line_data["line"]
        real_over_odds = real_line_data["over_odds"]
        real_under_odds = real_line_data["under_odds"]

        # Recalculate probabilities with REAL line
        from model_engine import EdgeCalculator
        calc = EdgeCalculator()

        stat_type = STAT_MAPPING.get(proj.prop_type, "pts")
        if proj.prop_type == PropType.PRA:
            stat_type = "pra"

        over_prob = calc.calculate_over_probability(
            proj.projected_value, real_line, proj.std_dev, stat_type
        )
        under_prob = 1 - over_prob

        # Calculate edge vs implied odds
        # American odds to implied probability
        if real_over_odds < 0:
            implied_over = abs(real_over_odds) / (abs(real_over_odds) + 100)
        else:
            implied_over = 100 / (real_over_odds + 100)

        if real_under_odds < 0:
            implied_under = abs(real_under_odds) / (abs(real_under_odds) + 100)
        else:
            implied_under = 100 / (real_under_odds + 100)

        edge_over = (over_prob - implied_over) * 100
        edge_under = (under_prob - implied_under) * 100

        # Get actual value
        if proj.prop_type in COMBINED_STAT_MAPPING:
            stat_keys = COMBINED_STAT_MAPPING[proj.prop_type]
            actual = sum(actual_stats[proj.player_name].get(k, 0) or 0 for k in stat_keys)
        elif proj.prop_type in STAT_MAPPING:
            stat_key = STAT_MAPPING[proj.prop_type]
            actual = actual_stats[proj.player_name].get(stat_key, 0) or 0
        else:
            continue

        # Only evaluate if we have positive edge (5%+)
        if edge_over > 5:
            pick = "OVER"
            hit = actual > real_line
            prob = over_prob
            edge = edge_over
        elif edge_under > 5:
            pick = "UNDER"
            hit = actual < real_line
            prob = under_prob
            edge = edge_under
        else:
            continue  # No edge

        result["legs"].append({
            "player": proj.player_name,
            "prop_type": proj.prop_type.value,
            "line": real_line,
            "over_odds": real_over_odds,
            "under_odds": real_under_odds,
            "projected": round(proj.projected_value, 1),
            "actual": actual,
            "pick": pick,
            "hit": hit,
            "probability": round(prob, 3),
            "edge": round(edge, 1),
            "implied_prob": round(implied_over if pick == "OVER" else implied_under, 3),
            "usage_boost": proj.usage_boost,
            "role_change": proj.role_change,
            # Additional model details for analysis
            "std_dev": round(proj.std_dev, 2),
            "projected_minutes": round(proj.projected_minutes, 1) if proj.projected_minutes else None,
            "confidence": round(proj.confidence, 2) if proj.confidence else None,
            "diff_from_line": round(proj.projected_value - real_line, 1),
            "diff_pct": round((proj.projected_value - real_line) / real_line * 100, 1) if real_line > 0 else 0
        })

    return result


def run_backtest_real_lines(dates: list, output_file: str):
    """Run backtest using REAL historical lines"""
    print("=" * 70)
    print("BACKTEST WITH REAL DK/FANDUEL LINES")
    print("Using historical odds from The Odds API database")
    print("=" * 70)
    print()

    pipeline = DataPipeline(api_key=API_KEY)
    engine = ModelEngine(pipeline)
    cache = get_cache()

    all_results = []

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

    total_matched = 0
    total_missing = 0

    for i, game in enumerate(all_games, 1):
        date = game["backtest_date"]
        home = game["home_team"]["full_name"]
        away = game["visitor_team"]["full_name"]

        print(f"[{i}/{len(all_games)}] {away} @ {home} ({date})...", end=" ", flush=True)

        result = backtest_game_with_real_lines(game, pipeline, engine, cache, date)
        all_results.append(result)

        n_legs = len(result["legs"])
        matched = result.get("props_matched", 0)
        missing = result.get("props_missing", 0)
        total_matched += matched
        total_missing += missing

        if "error" in result:
            print(f"ERROR: {result['error']}")
        else:
            print(f"{n_legs} props (matched: {matched}, missing: {missing})")

        time.sleep(0.2)

    print(f"\nTotal props matched: {total_matched}")
    print(f"Total props missing: {total_missing}")

    return all_results


def generate_report(results: list, output_file: str):
    """Generate comprehensive report"""

    all_legs = []
    legs_with_boost = []
    legs_with_role_change = []

    for r in results:
        for leg in r["legs"]:
            all_legs.append(leg)
            # usage_boost is now a multiplier (1.0 = no boost, 1.10 = 10% boost)
            if leg.get("usage_boost", 1.0) > 1.0:
                legs_with_boost.append(leg)
            if leg.get("role_change"):
                legs_with_role_change.append(leg)

    def calc_stats(legs):
        if not legs:
            return {"total": 0, "hits": 0, "rate": 0}
        hits = sum(1 for l in legs if l["hit"])
        return {"total": len(legs), "hits": hits, "rate": hits/len(legs)*100}

    overall = calc_stats(all_legs)
    boost_stats = calc_stats(legs_with_boost)
    role_stats = calc_stats(legs_with_role_change)

    # By prop type
    by_prop = defaultdict(list)
    for leg in all_legs:
        by_prop[leg["prop_type"]].append(leg)
    prop_stats = {k: calc_stats(v) for k, v in by_prop.items()}

    # By edge level
    high_edge = [l for l in all_legs if l["edge"] >= 15]
    med_edge = [l for l in all_legs if 10 <= l["edge"] < 15]
    low_edge = [l for l in all_legs if 5 <= l["edge"] < 10]

    high_edge_stats = calc_stats(high_edge)
    med_edge_stats = calc_stats(med_edge)
    low_edge_stats = calc_stats(low_edge)

    # By pick direction
    overs = [l for l in all_legs if l["pick"] == "OVER"]
    unders = [l for l in all_legs if l["pick"] == "UNDER"]
    over_stats = calc_stats(overs)
    under_stats = calc_stats(unders)

    # Calculate expected profit
    # Standard -110 odds means we need 52.4% to break even
    # With edge, calculate expected value
    total_ev = 0
    for leg in all_legs:
        implied = leg["implied_prob"]
        our_prob = leg["probability"]
        # Assuming -110 odds, payout is 0.909 units
        ev = (our_prob * 0.909) - ((1 - our_prob) * 1.0)
        total_ev += ev

    # Build parlays
    game_legs = defaultdict(list)
    for r in results:
        for leg in r["legs"]:
            game_legs[r["game_id"]].append(leg)

    safe_parlays = []
    for game_id, legs in game_legs.items():
        if len(legs) >= 2:
            legs.sort(key=lambda x: x["edge"] * x["probability"], reverse=True)
            p_legs = legs[:2]
            all_hit = all(l["hit"] for l in p_legs)
            prob = p_legs[0]["probability"] * p_legs[1]["probability"]
            safe_parlays.append({"legs": p_legs, "hit": all_hit, "prob": prob})

    safe_hits = sum(1 for p in safe_parlays if p["hit"])
    safe_profit = (safe_hits * 2) - len(safe_parlays)  # 2:1 payout
    safe_roi = (safe_profit / len(safe_parlays) * 100) if safe_parlays else 0

    # Build report
    report = []
    report.append("=" * 70)
    report.append("BACKTEST RESULTS - REAL DK/FANDUEL LINES")
    report.append("=" * 70)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"Games Analyzed: {len(results)}")
    report.append(f"Total Props Evaluated: {overall['total']}")
    report.append("")

    report.append("=" * 70)
    report.append("INDIVIDUAL LEG PERFORMANCE")
    report.append("=" * 70)
    report.append(f"Overall: {overall['hits']}/{overall['total']} = {overall['rate']:.1f}%")
    report.append(f"Expected (breakeven): 52.4%")
    report.append(f"Expected Value (total): {total_ev:+.2f} units")
    report.append("")

    report.append("By Pick Direction:")
    report.append(f"  OVER:  {over_stats['hits']}/{over_stats['total']} = {over_stats['rate']:.1f}%")
    report.append(f"  UNDER: {under_stats['hits']}/{under_stats['total']} = {under_stats['rate']:.1f}%")
    report.append("")

    report.append("By Edge Level:")
    report.append(f"  15%+:   {high_edge_stats['hits']}/{high_edge_stats['total']} = {high_edge_stats['rate']:.1f}%")
    report.append(f"  10-15%: {med_edge_stats['hits']}/{med_edge_stats['total']} = {med_edge_stats['rate']:.1f}%")
    report.append(f"  5-10%:  {low_edge_stats['hits']}/{low_edge_stats['total']} = {low_edge_stats['rate']:.1f}%")
    report.append("")

    report.append("By Prop Type:")
    for prop_type, stats in sorted(prop_stats.items()):
        if stats["total"] > 0:
            report.append(f"  {prop_type.upper()}: {stats['hits']}/{stats['total']} = {stats['rate']:.1f}%")
    report.append("")

    report.append("=" * 70)
    report.append("MODEL FEATURES")
    report.append("=" * 70)
    report.append(f"Pop-Off Boosts: {boost_stats['hits']}/{boost_stats['total']} = {boost_stats['rate']:.1f}%")
    report.append(f"Role Changes: {role_stats['hits']}/{role_stats['total']} = {role_stats['rate']:.1f}%")
    report.append("")

    report.append("=" * 70)
    report.append("2-LEG PARLAYS (2:1 payout)")
    report.append("=" * 70)
    report.append(f"Total: {len(safe_parlays)} | Hits: {safe_hits} | Rate: {safe_hits/len(safe_parlays)*100:.1f}%" if safe_parlays else "No parlays")
    report.append(f"Profit: {safe_profit:+.1f} units | ROI: {safe_roi:+.1f}%")
    report.append("")

    # Show sample legs
    report.append("=" * 70)
    report.append("SAMPLE LEGS (first 20)")
    report.append("=" * 70)
    for leg in all_legs[:20]:
        status = "HIT" if leg["hit"] else "MISS"
        report.append(f"  {leg['player']}: {leg['pick']} {leg['line']} {leg['prop_type'].upper()}")
        report.append(f"    Projected: {leg['projected']} | Actual: {leg['actual']} | Edge: {leg['edge']:.1f}% | {status}")
    report.append("")

    report.append("=" * 70)
    report.append("SUMMARY")
    report.append("=" * 70)
    if overall["rate"] > 52.4:
        report.append(f"  RESULT: PROFITABLE ({overall['rate']:.1f}% > 52.4% breakeven)")
    else:
        report.append(f"  RESULT: NOT PROFITABLE ({overall['rate']:.1f}% < 52.4% breakeven)")
    report.append("=" * 70)

    report_text = "\n".join(report)

    with open(output_file, "w") as f:
        f.write(report_text)

    # Save JSON
    json_file = output_file.replace(".txt", ".json")
    with open(json_file, "w") as f:
        json.dump({
            "summary": {
                "games": len(results),
                "total_legs": overall["total"],
                "hit_rate": round(overall["rate"], 1),
                "expected_value": round(total_ev, 2),
                "by_edge": {
                    "high": high_edge_stats,
                    "med": med_edge_stats,
                    "low": low_edge_stats
                },
                "by_prop": prop_stats,
                "parlays": {
                    "safe": {"total": len(safe_parlays), "hits": safe_hits, "profit": safe_profit, "roi": round(safe_roi, 1)}
                }
            },
            "games": results
        }, f, indent=2, default=str)

    return report_text


def generate_parlay_analysis(results: list, output_file: str):
    """Generate comprehensive parlay analysis and save all bets"""

    all_legs = []
    games_data = []

    for r in results:
        game_legs = []
        for leg in r["legs"]:
            leg["game_id"] = r["game_id"]
            leg["matchup"] = r["matchup"]
            leg["date"] = r["date"]
            all_legs.append(leg)
            game_legs.append(leg)

        games_data.append({
            "game_id": r["game_id"],
            "matchup": r["matchup"],
            "date": r["date"],
            "injuries": r.get("injuries", []),
            "legs": game_legs,
            "props_matched": r.get("props_matched", 0),
            "props_missing": r.get("props_missing", 0),
            "error": r.get("error")
        })

    # Generate 2-leg parlays (same game - SGP)
    sgp_parlays = []
    from itertools import combinations

    for game in games_data:
        legs = game["legs"]
        if len(legs) >= 2:
            for leg1, leg2 in combinations(legs, 2):
                # Skip if same player
                if leg1["player"] == leg2["player"]:
                    continue

                parlay = {
                    "type": "SGP",
                    "game_id": game["game_id"],
                    "matchup": game["matchup"],
                    "date": game["date"],
                    "leg1": leg1,
                    "leg2": leg2,
                    "combined_prob": leg1["probability"] * leg2["probability"],
                    "combined_edge": (leg1["edge"] + leg2["edge"]) / 2,
                    "hit": leg1["hit"] and leg2["hit"]
                }
                sgp_parlays.append(parlay)

    # Generate cross-game parlays
    cross_game_parlays = []
    games_by_date = defaultdict(list)
    for game in games_data:
        games_by_date[game["date"]].append(game)

    for date, date_games in games_by_date.items():
        if len(date_games) >= 2:
            for g1, g2 in combinations(date_games, 2):
                if g1["legs"] and g2["legs"]:
                    # Pick best leg from each game
                    best1 = max(g1["legs"], key=lambda x: x["edge"] * x["probability"])
                    best2 = max(g2["legs"], key=lambda x: x["edge"] * x["probability"])

                    parlay = {
                        "type": "Cross-Game",
                        "date": date,
                        "game1": g1["matchup"],
                        "game2": g2["matchup"],
                        "leg1": best1,
                        "leg2": best2,
                        "combined_prob": best1["probability"] * best2["probability"],
                        "combined_edge": (best1["edge"] + best2["edge"]) / 2,
                        "hit": best1["hit"] and best2["hit"]
                    }
                    cross_game_parlays.append(parlay)

    # Calculate parlay stats
    def calc_parlay_stats(parlays):
        if not parlays:
            return {"total": 0, "hits": 0, "rate": 0, "profit": 0, "roi": 0}
        hits = sum(1 for p in parlays if p["hit"])
        profit = (hits * 2.5) - len(parlays)  # ~+250 odds for 2-leg
        roi = (profit / len(parlays)) * 100 if parlays else 0
        return {
            "total": len(parlays),
            "hits": hits,
            "rate": round(hits / len(parlays) * 100, 1),
            "profit": round(profit, 1),
            "roi": round(roi, 1)
        }

    sgp_stats = calc_parlay_stats(sgp_parlays)
    cross_stats = calc_parlay_stats(cross_game_parlays)

    # Filter parlays by different strategies for analysis
    high_prob_sgp = [p for p in sgp_parlays if p["combined_prob"] >= 0.35]
    low_edge_sgp = [p for p in sgp_parlays if p["leg1"]["edge"] < 10 and p["leg2"]["edge"] < 10]
    under_parlays = [p for p in sgp_parlays + cross_game_parlays
                     if p["leg1"]["pick"] == "UNDER" or p["leg2"]["pick"] == "UNDER"]
    both_under = [p for p in sgp_parlays + cross_game_parlays
                  if p["leg1"]["pick"] == "UNDER" and p["leg2"]["pick"] == "UNDER"]

    # Save comprehensive JSON output
    output_data = {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "total_games": len(games_data),
            "total_legs": len(all_legs),
            "date_range": {
                "start": min(g["date"] for g in games_data if g["date"]),
                "end": max(g["date"] for g in games_data if g["date"])
            }
        },
        "leg_summary": {
            "total": len(all_legs),
            "hits": sum(1 for l in all_legs if l["hit"]),
            "rate": round(sum(1 for l in all_legs if l["hit"]) / len(all_legs) * 100, 1) if all_legs else 0
        },
        "parlay_summary": {
            "sgp": sgp_stats,
            "cross_game": cross_stats,
            "high_prob_sgp": calc_parlay_stats(high_prob_sgp),
            "low_edge_sgp": calc_parlay_stats(low_edge_sgp),
            "at_least_one_under": calc_parlay_stats(under_parlays),
            "both_under": calc_parlay_stats(both_under)
        },
        "games": games_data,
        "legs": all_legs,
        "sgp_parlays": sgp_parlays[:500],  # Limit for file size
        "cross_game_parlays": cross_game_parlays[:500]
    }

    json_output = output_file.replace(".txt", "_full.json")
    with open(json_output, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nFull data saved to: {json_output}")

    # Print parlay summary
    print("\n" + "=" * 70)
    print("PARLAY ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"\nSGP (Same Game Parlays): {sgp_stats['total']} total")
    print(f"  Hit Rate: {sgp_stats['hits']}/{sgp_stats['total']} = {sgp_stats['rate']}%")
    print(f"  Profit: {sgp_stats['profit']:+.1f} units | ROI: {sgp_stats['roi']:+.1f}%")

    print(f"\nCross-Game Parlays: {cross_stats['total']} total")
    print(f"  Hit Rate: {cross_stats['hits']}/{cross_stats['total']} = {cross_stats['rate']}%")
    print(f"  Profit: {cross_stats['profit']:+.1f} units | ROI: {cross_stats['roi']:+.1f}%")

    print(f"\nStrategy Breakdown:")
    for name, stats in [
        ("High Prob SGP (>35%)", calc_parlay_stats(high_prob_sgp)),
        ("Low Edge SGP (<10%)", calc_parlay_stats(low_edge_sgp)),
        ("At Least One UNDER", calc_parlay_stats(under_parlays)),
        ("Both UNDER", calc_parlay_stats(both_under))
    ]:
        if stats["total"] > 0:
            print(f"  {name}: {stats['hits']}/{stats['total']} = {stats['rate']}% (ROI: {stats['roi']:+.1f}%)")

    return output_data


if __name__ == "__main__":
    output_file = "/Users/suhaasnandyala/Documents/aiBall/outputs/backtest_real_lines.txt"

    # Generate all available historical data (full season)
    # 2024-25 season: Oct 22, 2024 - Apr 13, 2025 (includes backfilled data)
    # 2025-26 season: Oct 24, 2025 - Jan 2, 2026
    from datetime import datetime, timedelta

    # Two date ranges - now includes backfilled early season data
    ranges = [
        (datetime(2024, 10, 22), datetime(2025, 4, 13)),  # 2024-25 season (full)
        (datetime(2025, 10, 24), datetime(2026, 1, 2)),   # 2025-26 season
    ]

    dates = []
    for start_date, end_date in ranges:
        current = start_date
        while current <= end_date:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)

    # Remove duplicates and sort
    dates = sorted(set(dates))

    # Create timestamped output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "/Users/suhaasnandyala/Documents/aiBall/outputs"

    output_file = f"{output_dir}/backtest_real_lines.txt"
    detailed_report = f"{output_dir}/backtest_detailed_{timestamp}.txt"
    predictions_file = f"{output_dir}/backtest_predictions_{timestamp}.json"

    print(f"Starting backtest with REAL lines for {len(dates)} days...")
    print(f"Date range: {dates[0]} to {dates[-1]}")
    print(f"Model changes: usage_boost cap=1.10, UNDER pref=1.15, threes penalty=0.75")
    start_time = time.time()

    results = run_backtest_real_lines(dates, output_file)

    elapsed = time.time() - start_time
    print(f"\nBacktest completed in {elapsed/60:.1f} minutes")
    print("\nGenerating report...")

    report = generate_report(results, output_file)
    print(report)
    print(f"\nSaved to: {output_file}")

    # Generate comprehensive parlay analysis
    print("\nGenerating parlay analysis...")
    parlay_data = generate_parlay_analysis(results, output_file)

    # Save all predictions to JSON for detailed analysis
    print(f"\nSaving all predictions to: {predictions_file}")
    all_predictions = []
    for r in results:
        for leg in r.get("legs", []):
            leg["game_date"] = r.get("date")
            leg["matchup"] = r.get("matchup")
            all_predictions.append(leg)

    with open(predictions_file, "w") as f:
        json.dump({
            "metadata": {
                "generated": datetime.now().isoformat(),
                "model_version": "v2_conservative",
                "changes": [
                    "usage_boost capped at 1.10 (was 1.15)",
                    "UNDER preference multiplier 1.15",
                    "threes deprioritized (0.75 penalty)"
                ],
                "total_predictions": len(all_predictions),
                "date_range": f"{dates[0]} to {dates[-1]}"
            },
            "predictions": all_predictions
        }, f, indent=2, default=str)

    # Generate detailed text report
    print(f"\nGenerating detailed report: {detailed_report}")
    with open(detailed_report, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("DETAILED BACKTEST REPORT - MODEL V2 (Conservative Changes)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Date range: {dates[0]} to {dates[-1]}\n")
        f.write(f"Total games: {len(results)}\n")
        f.write(f"Total predictions: {len(all_predictions)}\n\n")

        f.write("MODEL CHANGES:\n")
        f.write("  - Usage boost capped at 10% (was 15%)\n")
        f.write("  - UNDER picks get 15% score boost\n")
        f.write("  - Threes props get 25% score penalty\n\n")

        f.write("=" * 80 + "\n")
        f.write("PERFORMANCE SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(report)
        f.write("\n\n")

        # Detailed breakdown by category
        f.write("=" * 80 + "\n")
        f.write("PREDICTION ACCURACY BY CATEGORY\n")
        f.write("=" * 80 + "\n\n")

        # By prop type
        from collections import defaultdict
        by_prop = defaultdict(lambda: {"total": 0, "hits": 0})
        by_direction = defaultdict(lambda: {"total": 0, "hits": 0})
        by_edge = defaultdict(lambda: {"total": 0, "hits": 0})
        by_boost = defaultdict(lambda: {"total": 0, "hits": 0})

        for p in all_predictions:
            prop = p.get("prop_type", "unknown")
            by_prop[prop]["total"] += 1
            if p.get("hit"):
                by_prop[prop]["hits"] += 1

            direction = p.get("pick", "OVER")
            by_direction[direction]["total"] += 1
            if p.get("hit"):
                by_direction[direction]["hits"] += 1

            edge = p.get("edge", 0)
            if edge < 8:
                bucket = "5-8%"
            elif edge < 12:
                bucket = "8-12%"
            else:
                bucket = "12%+"
            by_edge[bucket]["total"] += 1
            if p.get("hit"):
                by_edge[bucket]["hits"] += 1

            boost = p.get("usage_boost", 1.0)
            if boost <= 1.01:
                b_bucket = "No boost"
            elif boost <= 1.05:
                b_bucket = "Small (1-5%)"
            else:
                b_bucket = "Medium+ (5%+)"
            by_boost[b_bucket]["total"] += 1
            if p.get("hit"):
                by_boost[b_bucket]["hits"] += 1

        f.write("By Prop Type:\n")
        for prop, stats in sorted(by_prop.items(), key=lambda x: -x[1]["hits"]/x[1]["total"] if x[1]["total"] > 0 else 0):
            rate = 100 * stats["hits"] / stats["total"] if stats["total"] > 0 else 0
            f.write(f"  {prop}: {stats['hits']}/{stats['total']} = {rate:.1f}%\n")

        f.write("\nBy Direction:\n")
        for d, stats in by_direction.items():
            rate = 100 * stats["hits"] / stats["total"] if stats["total"] > 0 else 0
            f.write(f"  {d}: {stats['hits']}/{stats['total']} = {rate:.1f}%\n")

        f.write("\nBy Edge Level:\n")
        for bucket in ["5-8%", "8-12%", "12%+"]:
            if bucket in by_edge:
                stats = by_edge[bucket]
                rate = 100 * stats["hits"] / stats["total"] if stats["total"] > 0 else 0
                f.write(f"  {bucket}: {stats['hits']}/{stats['total']} = {rate:.1f}%\n")

        f.write("\nBy Usage Boost:\n")
        for bucket in ["No boost", "Small (1-5%)", "Medium+ (5%+)"]:
            if bucket in by_boost:
                stats = by_boost[bucket]
                rate = 100 * stats["hits"] / stats["total"] if stats["total"] > 0 else 0
                f.write(f"  {bucket}: {stats['hits']}/{stats['total']} = {rate:.1f}%\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("SAMPLE PREDICTIONS (first 50)\n")
        f.write("=" * 80 + "\n\n")

        for p in all_predictions[:50]:
            status = "✓ HIT" if p.get("hit") else "✗ MISS"
            f.write(f"{p.get('game_date')} | {p.get('player')}\n")
            f.write(f"  {p.get('pick')} {p.get('line')} {p.get('prop_type')}\n")
            f.write(f"  Projected: {p.get('projected')} | Actual: {p.get('actual')} | {status}\n")
            f.write(f"  Edge: {p.get('edge')}% | Boost: {p.get('usage_boost')}\n\n")

    print(f"\nAll outputs saved:")
    print(f"  - Summary: {output_file}")
    print(f"  - Detailed report: {detailed_report}")
    print(f"  - All predictions JSON: {predictions_file}")
