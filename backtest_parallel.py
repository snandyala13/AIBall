#!/usr/bin/env python3
"""
Parallelized backtest for NBA prop model.
Respects API rate limits while maximizing throughput.
"""

import os
import sys
import time
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Semaphore
from collections import defaultdict
import json

sys.path.insert(0, '/Users/suhaasnandyala/Documents/aiBall')

from scipy.stats import norm
from data_pipeline import DataPipeline

API_KEY = os.getenv('BALLDONTLIE_API_KEY', 'd2df7f14-636a-4ff2-8005-35e768d1745f')
BASE_URL = "https://api.balldontlie.io/v1"
HEADERS = {"Authorization": API_KEY}

# Rate limiting: 60 requests per minute = 1 per second
REQUEST_SEMAPHORE = Semaphore(3)  # Max 3 concurrent requests
REQUEST_DELAY = 0.35  # 350ms between requests
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
            resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
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


def process_game(game_data, pipeline, date):
    """Process a single game and return betting results"""
    game_id = game_data["id"]
    home = game_data["home_team"]["full_name"]
    away = game_data["visitor_team"]["full_name"]
    matchup = f"{away} @ {home}"

    results = {
        "game_id": game_id,
        "matchup": matchup,
        "date": date,
        "legs": [],
        "parlays": {"safe": [], "standard": [], "longshot": []}
    }

    # Get box score
    box_score = get_box_score(game_id)
    if not box_score:
        return results

    # Process each player
    for player_stats in box_score:
        player = player_stats.get("player", {})
        player_name = f"{player.get('first_name', '')} {player.get('last_name', '')}"

        # Skip low-minute players
        min_str = player_stats.get("min", "0") or "0"
        try:
            mins = int(min_str.split(":")[0]) if ":" in str(min_str) else int(min_str)
        except:
            mins = 0

        if mins < 15:
            continue

        # Get projections from pipeline
        profile = pipeline.build_player_profile(player_name)
        if not profile:
            continue

        for stat in ["pts", "reb", "ast", "fg3m"]:
            actual = player_stats.get(stat, 0) or 0

            # Get weighted projection
            values, weights = [], []
            if stat in profile.last_5_avg and profile.last_5_avg[stat] > 0:
                values.append(profile.last_5_avg[stat])
                weights.append(0.45)
            if stat in profile.last_10_avg and profile.last_10_avg[stat] > 0:
                values.append(profile.last_10_avg[stat])
                weights.append(0.35)
            if stat in profile.season_avg and profile.season_avg[stat] > 0:
                values.append(profile.season_avg[stat])
                weights.append(0.20)

            if not values:
                continue

            proj = sum(v * w for v, w in zip(values, weights)) / sum(weights)
            std = profile.std_dev.get(stat, proj * 0.3)
            std = max(std, 1)

            # Simulate DK line
            line = round(proj - 0.5) + 0.5

            # Min lines
            min_lines = {"pts": 8.5, "reb": 3.5, "ast": 2.5, "fg3m": 0.5}
            if line < min_lines.get(stat, 0):
                continue

            # Calculate probabilities
            z = (line - proj) / std if std > 0 else 0
            over_prob = 1 - norm.cdf(z)
            under_prob = norm.cdf(z)

            is_star = mins >= 30
            volume_score = proj / 10

            if over_prob > 0.55:
                hit = actual > line
                quality = over_prob * (1 + volume_score) * (1.2 if is_star else 1.0)
                results["legs"].append({
                    "player": player_name,
                    "stat": stat.upper(),
                    "side": "OVER",
                    "line": line,
                    "proj": round(proj, 1),
                    "prob": round(over_prob, 3),
                    "quality": round(quality, 3),
                    "is_star": is_star,
                    "actual": actual,
                    "hit": hit
                })
            elif under_prob > 0.55:
                hit = actual < line
                quality = under_prob * (1 + volume_score) * (1.2 if is_star else 1.0)
                results["legs"].append({
                    "player": player_name,
                    "stat": stat.upper(),
                    "side": "UNDER",
                    "line": line,
                    "proj": round(proj, 1),
                    "prob": round(under_prob, 3),
                    "quality": round(quality, 3),
                    "is_star": is_star,
                    "actual": actual,
                    "hit": hit
                })

    # Build parlays from legs
    if len(results["legs"]) >= 2:
        legs = sorted(results["legs"], key=lambda x: x["quality"], reverse=True)

        # 2-leg safe parlay
        p_legs = legs[:2]
        all_hit = all(l["hit"] for l in p_legs)
        prob = p_legs[0]["prob"] * p_legs[1]["prob"]
        results["parlays"]["safe"].append({
            "legs": p_legs,
            "hit": all_hit,
            "prob": round(prob, 3)
        })

        # 3-leg standard
        if len(legs) >= 3:
            p_legs = legs[:3]
            all_hit = all(l["hit"] for l in p_legs)
            prob = 1
            for l in p_legs:
                prob *= l["prob"]
            results["parlays"]["standard"].append({
                "legs": p_legs,
                "hit": all_hit,
                "prob": round(prob, 3)
            })

        # 4-leg standard
        if len(legs) >= 4:
            p_legs = legs[:4]
            all_hit = all(l["hit"] for l in p_legs)
            prob = 1
            for l in p_legs:
                prob *= l["prob"]
            results["parlays"]["standard"].append({
                "legs": p_legs,
                "hit": all_hit,
                "prob": round(prob, 3)
            })

        # 5-leg longshot
        if len(legs) >= 5:
            p_legs = legs[:5]
            all_hit = all(l["hit"] for l in p_legs)
            prob = 1
            for l in p_legs:
                prob *= l["prob"]
            results["parlays"]["longshot"].append({
                "legs": p_legs,
                "hit": all_hit,
                "prob": round(prob, 3)
            })

    return results


def run_parallel_backtest(dates):
    """Run backtest across dates with parallelization"""
    print("=" * 70)
    print(f"PARALLEL BACKTEST: {dates[0]} to {dates[-1]}")
    print("=" * 70)

    pipeline = DataPipeline(api_key=API_KEY)
    all_results = []

    # Collect all games first
    print("\nFetching games for all dates...")
    all_games = []
    for date in dates:
        games = get_games_for_date(date)
        for g in games:
            g["backtest_date"] = date
        all_games.extend(games)
        print(f"  {date}: {len(games)} games")

    print(f"\nTotal games to process: {len(all_games)}")

    # Process games with thread pool (limited concurrency)
    print("\nProcessing games...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(process_game, game, pipeline, game["backtest_date"]): game
            for game in all_games
        }

        for i, future in enumerate(as_completed(futures), 1):
            game = futures[future]
            try:
                result = future.result()
                all_results.append(result)
                matchup = result["matchup"]
                n_legs = len(result["legs"])
                print(f"  [{i}/{len(all_games)}] {matchup}: {n_legs} legs")
            except Exception as e:
                print(f"  Error processing game: {e}")

    return all_results


def generate_report(results, output_file):
    """Generate comprehensive backtest report"""

    # Aggregate stats
    all_legs = []
    safe_parlays = []
    standard_parlays = []
    longshot_parlays = []

    for r in results:
        all_legs.extend(r["legs"])
        safe_parlays.extend(r["parlays"]["safe"])
        standard_parlays.extend(r["parlays"]["standard"])
        longshot_parlays.extend(r["parlays"]["longshot"])

    # Calculate metrics
    leg_hits = sum(1 for l in all_legs if l["hit"])
    leg_total = len(all_legs)

    # Parlay metrics
    def parlay_stats(parlays, payout):
        if not parlays:
            return {"total": 0, "hits": 0, "profit": 0, "roi": 0}
        hits = sum(1 for p in parlays if p["hit"])
        total = len(parlays)
        profit = (hits * payout) - total
        roi = (profit / total * 100) if total > 0 else 0
        return {"total": total, "hits": hits, "profit": profit, "roi": roi}

    safe_stats = parlay_stats(safe_parlays, 2)
    std_stats = parlay_stats(standard_parlays, 5)
    long_stats = parlay_stats(longshot_parlays, 12)

    total_wagered = safe_stats["total"] + std_stats["total"] + long_stats["total"]
    total_profit = safe_stats["profit"] + std_stats["profit"] + long_stats["profit"]
    overall_roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0

    # By confidence breakdown
    high_conf = [l for l in all_legs if l["prob"] >= 0.65]
    med_conf = [l for l in all_legs if 0.60 <= l["prob"] < 0.65]
    low_conf = [l for l in all_legs if 0.55 <= l["prob"] < 0.60]

    # By stat breakdown
    stats_breakdown = defaultdict(lambda: {"hits": 0, "total": 0})
    for l in all_legs:
        stats_breakdown[l["stat"]]["total"] += 1
        if l["hit"]:
            stats_breakdown[l["stat"]]["hits"] += 1

    # Build report
    report = []
    report.append("=" * 70)
    report.append("BACKTEST RESULTS - WEEK OF DEC 25-31, 2025")
    report.append("=" * 70)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"Games Analyzed: {len(results)}")
    report.append("")

    report.append("=" * 70)
    report.append("INDIVIDUAL LEG PERFORMANCE")
    report.append("=" * 70)
    report.append(f"Total Legs: {leg_total}")
    report.append(f"Hits: {leg_hits} ({leg_hits/leg_total*100:.1f}%)" if leg_total > 0 else "N/A")
    report.append("")

    report.append("By Confidence Level:")
    if high_conf:
        h = sum(1 for l in high_conf if l["hit"])
        report.append(f"  65%+:    {h}/{len(high_conf)} = {h/len(high_conf)*100:.1f}%")
    if med_conf:
        h = sum(1 for l in med_conf if l["hit"])
        report.append(f"  60-65%:  {h}/{len(med_conf)} = {h/len(med_conf)*100:.1f}%")
    if low_conf:
        h = sum(1 for l in low_conf if l["hit"])
        report.append(f"  55-60%:  {h}/{len(low_conf)} = {h/len(low_conf)*100:.1f}%")
    report.append("")

    report.append("By Stat Type:")
    for stat, data in sorted(stats_breakdown.items()):
        rate = data["hits"] / data["total"] * 100 if data["total"] > 0 else 0
        report.append(f"  {stat}: {data['hits']}/{data['total']} = {rate:.1f}%")
    report.append("")

    report.append("=" * 70)
    report.append("PARLAY PERFORMANCE")
    report.append("=" * 70)
    report.append("")

    report.append(f"SAFE PARLAYS (2-leg, ~+200 payout)")
    report.append("-" * 50)
    report.append(f"  Total: {safe_stats['total']} | Hits: {safe_stats['hits']} | " +
                  f"Rate: {safe_stats['hits']/safe_stats['total']*100:.1f}%" if safe_stats['total'] > 0 else "  No parlays")
    report.append(f"  Profit: {safe_stats['profit']:+.1f} units | ROI: {safe_stats['roi']:+.1f}%")
    report.append("")

    # Show safe parlay details
    for i, p in enumerate(safe_parlays[:10], 1):
        status = "HIT" if p["hit"] else "MISS"
        report.append(f"  [{i}] {len(p['legs'])}-leg ({p['prob']*100:.1f}%) - {status}")
        for leg in p["legs"]:
            s = "+" if leg["hit"] else "-"
            report.append(f"      {s} {leg['player']}: {leg['side']} {leg['line']} {leg['stat']} (Actual: {leg['actual']})")
    report.append("")

    report.append(f"STANDARD PARLAYS (3-4 leg, ~+500 payout)")
    report.append("-" * 50)
    report.append(f"  Total: {std_stats['total']} | Hits: {std_stats['hits']} | " +
                  f"Rate: {std_stats['hits']/std_stats['total']*100:.1f}%" if std_stats['total'] > 0 else "  No parlays")
    report.append(f"  Profit: {std_stats['profit']:+.1f} units | ROI: {std_stats['roi']:+.1f}%")
    report.append("")

    for i, p in enumerate(standard_parlays[:10], 1):
        status = "HIT" if p["hit"] else "MISS"
        report.append(f"  [{i}] {len(p['legs'])}-leg ({p['prob']*100:.1f}%) - {status}")
        for leg in p["legs"]:
            s = "+" if leg["hit"] else "-"
            report.append(f"      {s} {leg['player']}: {leg['side']} {leg['line']} {leg['stat']} (Actual: {leg['actual']})")
    report.append("")

    report.append(f"LONGSHOT PARLAYS (5+ leg, ~+1200 payout)")
    report.append("-" * 50)
    report.append(f"  Total: {long_stats['total']} | Hits: {long_stats['hits']} | " +
                  f"Rate: {long_stats['hits']/long_stats['total']*100:.1f}%" if long_stats['total'] > 0 else "  No parlays")
    report.append(f"  Profit: {long_stats['profit']:+.1f} units | ROI: {long_stats['roi']:+.1f}%")
    report.append("")

    report.append("=" * 70)
    report.append("OVERALL SUMMARY")
    report.append("=" * 70)
    report.append(f"  Total Parlays Wagered: {total_wagered}")
    report.append(f"  Total Profit: {total_profit:+.1f} units")
    report.append(f"  Overall ROI: {overall_roi:+.1f}%")
    report.append("")

    if overall_roi > 0:
        report.append("  STATUS: PROFITABLE")
    else:
        report.append("  STATUS: NOT PROFITABLE")

    report.append("")
    report.append("=" * 70)

    # Write to file
    with open(output_file, "w") as f:
        f.write("\n".join(report))

    # Also save raw JSON
    json_file = output_file.replace(".txt", ".json")
    with open(json_file, "w") as f:
        json.dump({
            "summary": {
                "games": len(results),
                "total_legs": leg_total,
                "leg_hits": leg_hits,
                "leg_rate": round(leg_hits/leg_total*100, 1) if leg_total > 0 else 0,
                "safe_parlays": safe_stats,
                "standard_parlays": std_stats,
                "longshot_parlays": long_stats,
                "total_profit": round(total_profit, 1),
                "overall_roi": round(overall_roi, 1)
            },
            "games": results
        }, f, indent=2)

    return "\n".join(report)


if __name__ == "__main__":
    dates = [
        "2025-12-25",
        "2025-12-26",
        "2025-12-27",
        "2025-12-28",
        "2025-12-29",
        "2025-12-30",
        "2025-12-31"
    ]

    print(f"Starting parallel backtest for {len(dates)} days...")
    print(f"Rate limiting: {REQUEST_DELAY}s between requests, max 3 concurrent")
    print()

    start_time = time.time()
    results = run_parallel_backtest(dates)
    elapsed = time.time() - start_time

    print(f"\nBacktest completed in {elapsed:.1f}s")
    print("\nGenerating report...")

    output_file = "/Users/suhaasnandyala/Documents/aiBall/outputs/backtest_week_parallel.txt"
    report = generate_report(results, output_file)

    print(report)
    print(f"\nSaved to: {output_file}")
