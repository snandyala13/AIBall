#!/usr/bin/env python3
"""
Backtest a single game - Knicks @ Spurs Dec 31, 2025
"""

import os
import sys
sys.path.insert(0, '/Users/suhaasnandyala/Documents/aiBall')

from scipy.stats import norm
from data_pipeline import DataPipeline

# Game info
GAME_DATE = "2025-12-31"
GAME_ID = 18447281

# Actual results from the game
ACTUAL_RESULTS = {
    # Spurs
    "Julian Champagnie": {"pts": 36, "reb": 6, "ast": 0, "fg3m": 11, "pra": 42},
    "Victor Wembanyama": {"pts": 31, "reb": 13, "ast": 1, "fg3m": 2, "pra": 45},
    "De'Aaron Fox": {"pts": 26, "reb": 1, "ast": 7, "fg3m": 1, "pra": 34},
    "Harrison Barnes": {"pts": 7, "reb": 6, "ast": 4, "fg3m": 0, "pra": 17},
    "Stephon Castle": {"pts": 5, "reb": 2, "ast": 7, "fg3m": 0, "pra": 14},
    "Keldon Johnson": {"pts": 19, "reb": 8, "ast": 0, "fg3m": 0, "pra": 27},
    # Knicks
    "Jalen Brunson": {"pts": 29, "reb": 4, "ast": 8, "fg3m": 5, "pra": 41},
    "Karl-Anthony Towns": {"pts": 25, "reb": 10, "ast": 4, "fg3m": 3, "pra": 39},
    "OG Anunoby": {"pts": 25, "reb": 7, "ast": 3, "fg3m": 5, "pra": 35},
    "Mikal Bridges": {"pts": 18, "reb": 3, "ast": 2, "fg3m": 3, "pra": 23},
    "Josh Hart": {"pts": 14, "reb": 9, "ast": 2, "fg3m": 2, "pra": 25},
}

# Weights for averaging periods (favor recent form)
WEIGHTS = {
    "last_5": 0.55,
    "last_10": 0.30,
    "season": 0.15
}

# Weights when role change detected
ROLE_CHANGE_WEIGHTS = {
    "last_5": 0.70,
    "last_10": 0.20,
    "season": 0.10
}

def detect_role_change(profile):
    """Detect if player's role has recently changed based on minutes or scoring trend"""
    l5_min = profile.last_5_avg.get("min", 0)
    l10_min = profile.last_10_avg.get("min", 0)
    l5_pts = profile.last_5_avg.get("pts", 0)
    l10_pts = profile.last_10_avg.get("pts", 0)

    min_change = 0.0
    pts_change = 0.0

    if l10_min > 0:
        min_change = (l5_min - l10_min) / l10_min

    if l10_pts > 0:
        pts_change = (l5_pts - l10_pts) / l10_pts

    # Expanded: 15%+ minutes increase OR 10%+ scoring increase
    if min_change >= 0.15 or pts_change >= 0.10:
        return True, "expanded"

    # Reduced: 15%+ minutes decrease OR 10%+ scoring decrease
    if min_change <= -0.15 or pts_change <= -0.10:
        return True, "reduced"

    return False, ""

def get_weighted_avg(profile, stat_key):
    """Calculate weighted average for a stat, adjusting for role changes"""
    values = []
    weights = []

    # Detect role change and select appropriate weights
    role_changed, _ = detect_role_change(profile)
    weight_set = ROLE_CHANGE_WEIGHTS if role_changed else WEIGHTS

    if stat_key in profile.last_5_avg:
        values.append(profile.last_5_avg[stat_key])
        weights.append(weight_set["last_5"])

    if stat_key in profile.last_10_avg:
        values.append(profile.last_10_avg[stat_key])
        weights.append(weight_set["last_10"])

    if stat_key in profile.season_avg:
        values.append(profile.season_avg[stat_key])
        weights.append(weight_set["season"])

    if not values:
        return 0

    return sum(v * w for v, w in zip(values, weights)) / sum(weights)

def get_std_dev(profile, stat_key):
    """Get standard deviation for a stat"""
    if stat_key in profile.std_dev:
        return profile.std_dev[stat_key]
    # Default std dev
    avg = get_weighted_avg(profile, stat_key)
    return max(1, avg * 0.3)

def calculate_over_prob(projected, line, std_dev):
    """Calculate probability of going over the line"""
    if std_dev <= 0:
        std_dev = 1
    z_score = (line - projected) / std_dev
    return 1 - norm.cdf(z_score)

def run_backtest():
    """Run the model on the game and compare against actual results"""
    pipeline = DataPipeline(api_key=os.getenv('BALLDONTLIE_API_KEY', ''))

    # Key players to analyze
    players = [
        "Victor Wembanyama",
        "De'Aaron Fox",
        "Julian Champagnie",
        "Keldon Johnson",
        "Jalen Brunson",
        "Karl-Anthony Towns",
        "OG Anunoby",
        "Mikal Bridges",
        "Josh Hart",
    ]

    prop_types = ["pts", "reb", "ast", "fg3m", "pra"]

    print("=" * 80)
    print(f"BACKTEST: Knicks @ Spurs - Dec 31, 2025")
    print(f"Final Score: Spurs 134 - Knicks 132")
    print("=" * 80)
    print()

    results = []

    for player_name in players:
        print(f"\n--- {player_name} ---")

        # Get player profile
        profile = pipeline.build_player_profile(player_name)
        if not profile:
            print(f"  Could not find player profile")
            continue

        actual = ACTUAL_RESULTS.get(player_name, {})
        if not actual:
            print(f"  No actual results found")
            continue

        for prop in prop_types:
            # Map prop type to profile key
            stat_key = prop
            if prop == "pra":
                # Calculate PRA from components
                proj_pts = get_weighted_avg(profile, "pts")
                proj_reb = get_weighted_avg(profile, "reb")
                proj_ast = get_weighted_avg(profile, "ast")
                projected = proj_pts + proj_reb + proj_ast
                # Estimate std dev for PRA (sum of variances)
                std_dev = (get_std_dev(profile, "pts")**2 +
                          get_std_dev(profile, "reb")**2 +
                          get_std_dev(profile, "ast")**2)**0.5
            else:
                projected = get_weighted_avg(profile, stat_key)
                std_dev = get_std_dev(profile, stat_key)

            if projected <= 0:
                continue

            actual_val = actual.get(prop, 0)

            # Create a synthetic DraftKings line
            # DK typically sets lines 0.5-1.5 below projections for overs
            line = round(projected - 0.5) + 0.5 if projected > 1 else projected - 0.5

            # Calculate probabilities
            over_prob = calculate_over_prob(projected, line, std_dev)
            under_prob = 1 - over_prob

            bet_direction = "OVER" if over_prob > under_prob else "UNDER"
            bet_prob = max(over_prob, under_prob)

            # Only consider bets with edge (prob > 55%)
            if bet_prob < 0.55:
                continue

            # Did it hit?
            if bet_direction == "OVER":
                hit = actual_val > line
            else:
                hit = actual_val < line

            result = {
                "player": player_name,
                "prop": prop,
                "projected": projected,
                "line": line,
                "actual": actual_val,
                "bet": bet_direction,
                "probability": bet_prob,
                "std_dev": std_dev,
                "hit": hit
            }
            results.append(result)

            hit_str = "✓ HIT" if hit else "✗ MISS"
            print(f"  {prop.upper():6} | Proj: {projected:5.1f} | Line: {line:5.1f} | Actual: {actual_val:5.0f} | {bet_direction:5} ({bet_prob:.1%}) | {hit_str}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total = len(results)
    hits = sum(1 for r in results if r["hit"])
    misses = total - hits

    print(f"\nTotal Picks: {total}")
    print(f"Hits: {hits}")
    print(f"Misses: {misses}")
    if total > 0:
        print(f"Hit Rate: {hits/total:.1%}")

    # Break down by confidence tier
    tier1 = [r for r in results if r["probability"] >= 0.65]
    tier2 = [r for r in results if 0.60 <= r["probability"] < 0.65]
    tier3 = [r for r in results if 0.55 <= r["probability"] < 0.60]

    print(f"\nBy Confidence Tier:")
    if tier1:
        t1_hits = sum(1 for r in tier1 if r["hit"])
        print(f"  Tier 1 (65%+):   {t1_hits}/{len(tier1)} = {t1_hits/len(tier1):.1%}")
    if tier2:
        t2_hits = sum(1 for r in tier2 if r["hit"])
        print(f"  Tier 2 (60-65%): {t2_hits}/{len(tier2)} = {t2_hits/len(tier2):.1%}")
    if tier3:
        t3_hits = sum(1 for r in tier3 if r["hit"])
        print(f"  Tier 3 (55-60%): {t3_hits}/{len(tier3)} = {t3_hits/len(tier3):.1%}")

    # Detailed results
    print(f"\n\nDetailed Results:")
    print("-" * 80)
    for r in results:
        hit_str = "✓" if r["hit"] else "✗"
        print(f"{hit_str} {r['player']:20} | {r['prop'].upper():6} | {r['bet']:5} {r['line']:.1f} @ {r['probability']:.1%} | Actual: {r['actual']}")

    return results

if __name__ == "__main__":
    run_backtest()
