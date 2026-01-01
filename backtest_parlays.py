#!/usr/bin/env python3
"""
Backtest parlays against yesterday's games (Dec 31, 2025)
Uses actual box scores and simulates what the model would have predicted
"""

import os
import sys
sys.path.insert(0, '/Users/suhaasnandyala/Documents/aiBall')

import requests
from scipy.stats import norm
from data_pipeline import DataPipeline

API_KEY = os.getenv('BALLDONTLIE_API_KEY', '')
BASE_URL = "https://api.balldontlie.io/v1"

# Yesterday's games
GAMES = {
    18447276: "Warriors @ Hornets (132-125)",
    18447277: "Timberwolves @ Hawks (102-126)",
    18447278: "Magic @ Pacers (112-110)",
    18447279: "Suns @ Cavaliers (113-129)",
    18447280: "Pelicans @ Bulls (118-134)",
    18447281: "Knicks @ Spurs (132-134)",
    18447282: "Nuggets @ Raptors (106-103)",
    18447283: "Wizards @ Bucks (114-113)",
    18447284: "Blazers @ Thunder (95-124)",
}

def get_box_scores():
    """Get box scores for all yesterday's games"""
    all_stats = {}
    headers = {"Authorization": API_KEY}

    for game_id, matchup in GAMES.items():
        print(f"  Fetching {matchup}...")
        resp = requests.get(f"{BASE_URL}/stats", params={"game_ids[]": game_id}, headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            game_stats = {}
            for stat in data.get("data", []):
                player = stat.get("player", {})
                name = f"{player.get('first_name', '')} {player.get('last_name', '')}"
                mins_str = stat.get("min", "0") or "0"
                try:
                    mins = int(mins_str.split(":")[0]) if ":" in str(mins_str) else int(mins_str)
                except:
                    mins = 0

                if mins >= 15:  # Only players with 15+ minutes
                    game_stats[name] = {
                        "pts": stat.get("pts", 0) or 0,
                        "reb": stat.get("reb", 0) or 0,
                        "ast": stat.get("ast", 0) or 0,
                        "fg3m": stat.get("fg3m", 0) or 0,
                        "stl": stat.get("stl", 0) or 0,
                        "blk": stat.get("blk", 0) or 0,
                        "min": mins,
                    }
            all_stats[game_id] = {"stats": game_stats, "matchup": matchup}

    return all_stats

def get_weighted_projection(pipeline, player_name, stat_key):
    """Get weighted projection for a player stat"""
    profile = pipeline.build_player_profile(player_name)
    if not profile:
        return None, None

    # Calculate weighted average
    values = []
    weights = []

    if stat_key in profile.last_5_avg and profile.last_5_avg[stat_key] > 0:
        values.append(profile.last_5_avg[stat_key])
        weights.append(0.45)

    if stat_key in profile.last_10_avg and profile.last_10_avg[stat_key] > 0:
        values.append(profile.last_10_avg[stat_key])
        weights.append(0.35)

    if stat_key in profile.season_avg and profile.season_avg[stat_key] > 0:
        values.append(profile.season_avg[stat_key])
        weights.append(0.20)

    if not values:
        return None, None

    proj = sum(v * w for v, w in zip(values, weights)) / sum(weights)

    # Get std dev
    std = profile.std_dev.get(stat_key, proj * 0.3)
    return proj, max(std, 1)

def evaluate_leg(proj, std, line, actual, side):
    """Check if a bet would have hit"""
    if side == "OVER":
        return actual > line
    else:
        return actual < line

def main():
    print("=" * 70)
    print("PARLAY BACKTEST - December 31, 2025")
    print("=" * 70)
    print()

    print("Fetching box scores...")
    box_scores = get_box_scores()
    print(f"\nGot stats for {len(box_scores)} games\n")

    print("Building projections and evaluating parlays...")
    pipeline = DataPipeline(api_key=API_KEY)

    all_legs = []  # All individual legs
    game_legs = {}  # Legs by game

    for game_id, game_data in box_scores.items():
        matchup = game_data["matchup"]
        stats = game_data["stats"]
        print(f"\n  Processing {matchup}...")

        game_legs[game_id] = []

        for player_name, actual_stats in stats.items():
            for stat in ["pts", "reb", "ast", "fg3m"]:
                proj, std = get_weighted_projection(pipeline, player_name, stat)
                if proj is None or proj < 1:
                    continue

                actual = actual_stats.get(stat, 0)

                # Simulate DK line (around projection)
                line = round(proj - 0.5) + 0.5

                # Calculate probabilities
                z = (line - proj) / std if std > 0 else 0
                over_prob = 1 - norm.cdf(z)
                under_prob = norm.cdf(z)

                # FILTER: Skip low-volume props (too much variance)
                min_lines = {"pts": 8.5, "reb": 3.5, "ast": 2.5, "fg3m": 0.5}
                if line < min_lines.get(stat, 0):
                    continue

                # Calculate a quality score: prefer higher volume + edge
                volume_score = proj / 10  # Higher projections = more volume
                is_star = actual_stats.get("min", 0) >= 30  # Stars play 30+ mins

                # Pick the side with edge
                if over_prob > 0.55:
                    hit = actual > line
                    # Quality score combines probability + volume + star bonus
                    quality = over_prob * (1 + volume_score) * (1.2 if is_star else 1.0)
                    leg = {
                        "game": matchup,
                        "player": player_name,
                        "stat": stat.upper(),
                        "side": "OVER",
                        "line": line,
                        "proj": proj,
                        "prob": over_prob,
                        "quality": quality,
                        "is_star": is_star,
                        "actual": actual,
                        "hit": hit
                    }
                    all_legs.append(leg)
                    game_legs[game_id].append(leg)
                elif under_prob > 0.55:
                    hit = actual < line
                    quality = under_prob * (1 + volume_score) * (1.2 if is_star else 1.0)
                    leg = {
                        "game": matchup,
                        "player": player_name,
                        "stat": stat.upper(),
                        "side": "UNDER",
                        "line": line,
                        "proj": proj,
                        "prob": under_prob,
                        "quality": quality,
                        "is_star": is_star,
                        "actual": actual,
                        "hit": hit
                    }
                    all_legs.append(leg)
                    game_legs[game_id].append(leg)

    # Sort legs by QUALITY (not just probability)
    all_legs.sort(key=lambda x: x["quality"], reverse=True)

    print("\n" + "=" * 70)
    print("INDIVIDUAL LEG PERFORMANCE")
    print("=" * 70)

    hits = sum(1 for l in all_legs if l["hit"])
    total = len(all_legs)
    print(f"\nTotal Legs Evaluated: {total}")
    print(f"Hits: {hits} ({hits/total*100:.1f}%)" if total > 0 else "N/A")

    # Break down by confidence
    high_conf = [l for l in all_legs if l["prob"] >= 0.65]
    med_conf = [l for l in all_legs if 0.60 <= l["prob"] < 0.65]
    low_conf = [l for l in all_legs if 0.55 <= l["prob"] < 0.60]

    print(f"\nBy Confidence:")
    if high_conf:
        h = sum(1 for l in high_conf if l["hit"])
        print(f"  65%+:    {h}/{len(high_conf)} = {h/len(high_conf)*100:.1f}%")
    if med_conf:
        h = sum(1 for l in med_conf if l["hit"])
        print(f"  60-65%:  {h}/{len(med_conf)} = {h/len(med_conf)*100:.1f}%")
    if low_conf:
        h = sum(1 for l in low_conf if l["hit"])
        print(f"  55-60%:  {h}/{len(low_conf)} = {h/len(low_conf)*100:.1f}%")

    # Now simulate parlays
    print("\n" + "=" * 70)
    print("PARLAY PERFORMANCE")
    print("=" * 70)

    # Build parlays from top legs per game
    safe_parlays = []     # 2-leg
    standard_parlays = [] # 3-4 leg
    longshot_parlays = [] # 5-6 leg

    for game_id, legs in game_legs.items():
        if len(legs) < 2:
            continue

        # Sort by QUALITY (combines probability + volume + star status)
        legs.sort(key=lambda x: x["quality"], reverse=True)

        # 2-leg parlays (take top 2)
        if len(legs) >= 2:
            p_legs = legs[:2]
            all_hit = all(l["hit"] for l in p_legs)
            prob = p_legs[0]["prob"] * p_legs[1]["prob"]
            safe_parlays.append({"legs": p_legs, "hit": all_hit, "prob": prob, "game": legs[0]["game"]})

        # 3-leg parlay
        if len(legs) >= 3:
            p_legs = legs[:3]
            all_hit = all(l["hit"] for l in p_legs)
            prob = 1
            for l in p_legs:
                prob *= l["prob"]
            standard_parlays.append({"legs": p_legs, "hit": all_hit, "prob": prob, "game": legs[0]["game"]})

        # 4-leg parlay
        if len(legs) >= 4:
            p_legs = legs[:4]
            all_hit = all(l["hit"] for l in p_legs)
            prob = 1
            for l in p_legs:
                prob *= l["prob"]
            standard_parlays.append({"legs": p_legs, "hit": all_hit, "prob": prob, "game": legs[0]["game"]})

        # 5-leg parlay
        if len(legs) >= 5:
            p_legs = legs[:5]
            all_hit = all(l["hit"] for l in p_legs)
            prob = 1
            for l in p_legs:
                prob *= l["prob"]
            longshot_parlays.append({"legs": p_legs, "hit": all_hit, "prob": prob, "game": legs[0]["game"]})

        # 6-leg parlay
        if len(legs) >= 6:
            p_legs = legs[:6]
            all_hit = all(l["hit"] for l in p_legs)
            prob = 1
            for l in p_legs:
                prob *= l["prob"]
            longshot_parlays.append({"legs": p_legs, "hit": all_hit, "prob": prob, "game": legs[0]["game"]})

    # Results
    def show_tier(name, parlays, payout_mult):
        if not parlays:
            print(f"\n{name}: No parlays")
            return 0

        hits = sum(1 for p in parlays if p["hit"])
        total = len(parlays)
        profit = (hits * payout_mult) - total

        print(f"\n{name}")
        print("-" * 50)
        print(f"  Total: {total} | Hits: {hits} | Hit Rate: {hits/total*100:.1f}%")
        print(f"  Profit: {profit:+.1f} units (at avg {payout_mult}x payout)")

        # Show details
        print(f"\n  Details:")
        for i, p in enumerate(parlays[:8], 1):
            status = "✓ HIT" if p["hit"] else "✗ MISS"
            num_legs = len(p["legs"])
            print(f"    [{i}] {num_legs}-leg ({p['prob']*100:.1f}%) - {status}")
            for leg in p["legs"]:
                leg_status = "✓" if leg["hit"] else "✗"
                print(f"        {leg_status} {leg['player']}: {leg['side']} {leg['line']} {leg['stat']} (Actual: {leg['actual']})")

        return profit

    safe_profit = show_tier("SAFE PARLAYS (2-leg, ~+200)", safe_parlays, 2)
    std_profit = show_tier("STANDARD PARLAYS (3-4 leg, ~+500)", standard_parlays, 5)
    long_profit = show_tier("LONGSHOT PARLAYS (5-6 leg, ~+1200)", longshot_parlays, 12)

    # Overall
    total_wagered = len(safe_parlays) + len(standard_parlays) + len(longshot_parlays)
    total_profit = safe_profit + std_profit + long_profit

    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"  Total Parlays: {total_wagered}")
    print(f"  Total Profit: {total_profit:+.1f} units")
    if total_wagered > 0:
        print(f"  ROI: {total_profit/total_wagered*100:+.1f}%")

if __name__ == "__main__":
    main()
