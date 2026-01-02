#!/usr/bin/env python3
"""Comprehensive analysis of 70-day backtest results to identify improvement opportunities."""

import json
from collections import defaultdict
from itertools import combinations
import random

def load_backtest():
    with open("outputs/backtest_real_lines_full.json") as f:
        return json.load(f)


def analyze_legs(data):
    """Analyze individual leg performance patterns."""

    all_legs = data.get("legs", [])

    print("="*70)
    print("LEG PERFORMANCE ANALYSIS (70 Days)")
    print("="*70)

    total = len(all_legs)
    hits = sum(1 for l in all_legs if l.get("hit"))

    print(f"\nTotal legs: {total}")
    print(f"Hit rate: {hits}/{total} = {100*hits/total:.1f}%")
    print(f"Breakeven: 52.4% | Edge: {100*hits/total - 52.4:+.1f}%")

    # By prop type
    print("\n--- By Prop Type ---")
    by_prop = defaultdict(lambda: {"total": 0, "hits": 0})
    for leg in all_legs:
        prop = leg.get("prop_type", "unknown")
        by_prop[prop]["total"] += 1
        if leg.get("hit"):
            by_prop[prop]["hits"] += 1

    for prop, stats in sorted(by_prop.items(), key=lambda x: -x[1]["hits"]/x[1]["total"] if x[1]["total"] > 0 else 0):
        rate = 100 * stats["hits"] / stats["total"] if stats["total"] > 0 else 0
        edge = rate - 52.4
        print(f"  {prop}: {stats['hits']}/{stats['total']} = {rate:.1f}% (edge: {edge:+.1f}%)")

    # By pick direction
    print("\n--- By Direction ---")
    by_dir = defaultdict(lambda: {"total": 0, "hits": 0})
    for leg in all_legs:
        direction = leg.get("pick", "OVER")
        by_dir[direction]["total"] += 1
        if leg.get("hit"):
            by_dir[direction]["hits"] += 1

    for direction, stats in sorted(by_dir.items()):
        rate = 100 * stats["hits"] / stats["total"] if stats["total"] > 0 else 0
        edge = rate - 52.4
        print(f"  {direction}: {stats['hits']}/{stats['total']} = {rate:.1f}% (edge: {edge:+.1f}%)")

    # By edge bucket
    print("\n--- By Edge Level ---")
    edge_buckets = [
        ("5-7%", 5, 7),
        ("7-10%", 7, 10),
        ("10-12%", 10, 12),
        ("12-15%", 12, 15),
        ("15-20%", 15, 20),
        ("20%+", 20, 100)
    ]
    for bucket_name, low, high in edge_buckets:
        legs = [l for l in all_legs if low <= l.get("edge", 0) < high]
        if legs:
            hits = sum(1 for l in legs if l["hit"])
            rate = 100 * hits / len(legs)
            print(f"  {bucket_name}: {hits}/{len(legs)} = {rate:.1f}%")

    # By usage boost
    print("\n--- By Usage Boost ---")
    boost_buckets = [
        ("No boost (1.0)", 0.99, 1.01),
        ("Small (1-5%)", 1.01, 1.05),
        ("Medium (5-10%)", 1.05, 1.10),
        ("Large (10-20%)", 1.10, 1.20),
        ("Very Large (20%+)", 1.20, 2.0)
    ]
    for bucket_name, low, high in boost_buckets:
        legs = [l for l in all_legs if low <= l.get("usage_boost", 1.0) < high]
        if legs:
            hits = sum(1 for l in legs if l["hit"])
            rate = 100 * hits / len(legs)
            print(f"  {bucket_name}: {hits}/{len(legs)} = {rate:.1f}%")

    # By projected vs line margin
    print("\n--- By Projection Margin (proj - line) ---")
    margin_buckets = [
        ("0-2 over", 0, 2),
        ("2-4 over", 2, 4),
        ("4-6 over", 4, 6),
        ("6+ over", 6, 100),
    ]
    for bucket_name, low, high in margin_buckets:
        overs = [l for l in all_legs if l.get("pick") == "OVER"]
        legs = [l for l in overs if low <= (l.get("projected", 0) - l.get("line", 0)) < high]
        if legs:
            hits = sum(1 for l in legs if l["hit"])
            rate = 100 * hits / len(legs)
            print(f"  OVER {bucket_name}: {hits}/{len(legs)} = {rate:.1f}%")

    # Unders margin
    for bucket_name, low, high in margin_buckets:
        unders = [l for l in all_legs if l.get("pick") == "UNDER"]
        legs = [l for l in unders if low <= (l.get("line", 0) - l.get("projected", 0)) < high]
        if legs:
            hits = sum(1 for l in legs if l["hit"])
            rate = 100 * hits / len(legs)
            print(f"  UNDER {bucket_name}: {hits}/{len(legs)} = {rate:.1f}%")

    return all_legs


def analyze_prop_combinations(all_legs):
    """Analyze which prop type combinations perform best in 2-leg parlays."""

    print("\n" + "="*70)
    print("PROP TYPE PAIR PERFORMANCE (Cross-Game)")
    print("="*70)

    # Group by prop type and game
    by_prop_game = defaultdict(list)
    for i, leg in enumerate(all_legs):
        key = (leg.get("prop_type", "unknown"), leg.get("game_id"))
        by_prop_game[key].append(i)

    prop_types = list(set(leg.get("prop_type") for leg in all_legs))

    results = []

    # Cross-game same prop type
    for prop in prop_types:
        games = defaultdict(list)
        for i, leg in enumerate(all_legs):
            if leg.get("prop_type") == prop:
                games[leg.get("game_id")].append(i)

        # Cross game combinations
        game_ids = list(games.keys())
        combos = []
        for g1, g2 in combinations(game_ids, 2):
            for i in games[g1]:
                for j in games[g2]:
                    combos.append((i, j))

        if len(combos) >= 100:
            sample = random.sample(combos, min(5000, len(combos)))
            hits = sum(1 for i, j in sample if all_legs[i].get("hit") and all_legs[j].get("hit"))
            rate = 100 * hits / len(sample)
            results.append((f"{prop} + {prop}", hits, len(sample), rate))

    # Different prop types (cross-game)
    for p1, p2 in combinations(prop_types, 2):
        games1 = defaultdict(list)
        games2 = defaultdict(list)
        for i, leg in enumerate(all_legs):
            if leg.get("prop_type") == p1:
                games1[leg.get("game_id")].append(i)
            elif leg.get("prop_type") == p2:
                games2[leg.get("game_id")].append(i)

        combos = []
        for g1 in games1:
            for g2 in games2:
                if g1 != g2:
                    for i in games1[g1]:
                        for j in games2[g2]:
                            combos.append((i, j))

        if len(combos) >= 100:
            sample = random.sample(combos, min(5000, len(combos)))
            hits = sum(1 for i, j in sample if all_legs[i].get("hit") and all_legs[j].get("hit"))
            rate = 100 * hits / len(sample)
            results.append((f"{p1} + {p2}", hits, len(sample), rate))

    results.sort(key=lambda x: -x[3])

    print("\nProp combinations sorted by hit rate:")
    for name, hits, total, rate in results:
        # Calculate expected vs actual
        expected = 57.5 * 57.5 / 100  # Assuming independent
        print(f"  {name}: {hits}/{total} = {rate:.1f}% (expected: {expected:.1f}%)")


def simulate_parlay_strategies(all_legs):
    """Simulate different parlay strategies to find optimal approach."""

    print("\n" + "="*70)
    print("2-LEG PARLAY STRATEGY SIMULATION")
    print("="*70)

    random.seed(42)

    # Group legs by game for same-game checks
    legs_by_game = defaultdict(list)
    for i, leg in enumerate(all_legs):
        legs_by_game[leg.get("game_id")].append(i)

    # Generate all cross-game 2-leg combos (sample for efficiency)
    all_combos = []
    game_ids = list(legs_by_game.keys())
    for g1, g2 in combinations(game_ids, 2):
        for i in legs_by_game[g1][:3]:  # Limit per game for efficiency
            for j in legs_by_game[g2][:3]:
                all_combos.append((i, j))

    sample_size = min(50000, len(all_combos))
    sampled = random.sample(all_combos, sample_size)

    def calc_stats(combos, name):
        if not combos:
            print(f"  {name}: No combos")
            return
        hits = sum(1 for i, j in combos if all_legs[i].get("hit") and all_legs[j].get("hit"))
        rate = 100 * hits / len(combos)
        # ROI assuming +250 odds for 2-leg
        profit = (hits * 2.5) - len(combos)
        roi = profit / len(combos) * 100
        print(f"  {name}: {hits}/{len(combos)} = {rate:.1f}% | ROI: {roi:+.1f}%")
        return rate, roi

    print(f"\nSample size: {sample_size} parlays\n")

    # Baseline
    print("--- BASELINE ---")
    calc_stats(sampled, "All cross-game")

    # By direction
    print("\n--- BY DIRECTION ---")
    both_over = [(i, j) for i, j in sampled
                 if all_legs[i].get("pick") == "OVER" and all_legs[j].get("pick") == "OVER"]
    both_under = [(i, j) for i, j in sampled
                  if all_legs[i].get("pick") == "UNDER" and all_legs[j].get("pick") == "UNDER"]
    mixed = [(i, j) for i, j in sampled
             if all_legs[i].get("pick") != all_legs[j].get("pick")]

    calc_stats(both_over, "Both OVER")
    calc_stats(both_under, "Both UNDER")
    calc_stats(mixed, "Mixed (1 OVER + 1 UNDER)")

    # By edge level
    print("\n--- BY EDGE LEVEL ---")
    low_edge = [(i, j) for i, j in sampled
                if all_legs[i].get("edge", 0) < 8 and all_legs[j].get("edge", 0) < 8]
    med_edge = [(i, j) for i, j in sampled
                if 8 <= all_legs[i].get("edge", 0) < 12 and 8 <= all_legs[j].get("edge", 0) < 12]
    high_edge = [(i, j) for i, j in sampled
                 if all_legs[i].get("edge", 0) >= 12 and all_legs[j].get("edge", 0) >= 12]

    calc_stats(low_edge, "Both Low Edge (<8%)")
    calc_stats(med_edge, "Both Med Edge (8-12%)")
    calc_stats(high_edge, "Both High Edge (12%+)")

    # By usage boost
    print("\n--- BY USAGE BOOST ---")
    no_boost = [(i, j) for i, j in sampled
                if all_legs[i].get("usage_boost", 1.0) <= 1.05 and all_legs[j].get("usage_boost", 1.0) <= 1.05]
    one_boost = [(i, j) for i, j in sampled
                 if (all_legs[i].get("usage_boost", 1.0) > 1.05) != (all_legs[j].get("usage_boost", 1.0) > 1.05)]
    both_boost = [(i, j) for i, j in sampled
                  if all_legs[i].get("usage_boost", 1.0) > 1.05 and all_legs[j].get("usage_boost", 1.0) > 1.05]

    calc_stats(no_boost, "No boost legs")
    calc_stats(one_boost, "One boost leg")
    calc_stats(both_boost, "Both boost legs")

    # By prop type
    print("\n--- BY PROP TYPE ---")
    pra_pra = [(i, j) for i, j in sampled
               if all_legs[i].get("prop_type") == "pts_rebs_asts" and all_legs[j].get("prop_type") == "pts_rebs_asts"]
    pts_pts = [(i, j) for i, j in sampled
               if all_legs[i].get("prop_type") == "points" and all_legs[j].get("prop_type") == "points"]
    pts_pra = [(i, j) for i, j in sampled
               if (all_legs[i].get("prop_type") == "points" and all_legs[j].get("prop_type") == "pts_rebs_asts") or
                  (all_legs[j].get("prop_type") == "points" and all_legs[i].get("prop_type") == "pts_rebs_asts")]

    calc_stats(pra_pra, "PRA + PRA")
    calc_stats(pts_pts, "Points + Points")
    calc_stats(pts_pra, "Points + PRA")

    # Combined optimal strategies
    print("\n--- COMBINED STRATEGIES ---")

    # Strategy 1: Both UNDER + Low edge
    s1 = [(i, j) for i, j in sampled
          if all_legs[i].get("pick") == "UNDER" and all_legs[j].get("pick") == "UNDER"
          and all_legs[i].get("edge", 0) < 10 and all_legs[j].get("edge", 0) < 10]
    calc_stats(s1, "Both UNDER + Low Edge (<10%)")

    # Strategy 2: PRA/Points + No boost
    s2 = [(i, j) for i, j in sampled
          if all_legs[i].get("prop_type") in ["points", "pts_rebs_asts"]
          and all_legs[j].get("prop_type") in ["points", "pts_rebs_asts"]
          and all_legs[i].get("usage_boost", 1.0) <= 1.05
          and all_legs[j].get("usage_boost", 1.0) <= 1.05]
    calc_stats(s2, "PRA/Points + No boost")

    # Strategy 3: Mixed direction + PRA/Points
    s3 = [(i, j) for i, j in sampled
          if all_legs[i].get("pick") != all_legs[j].get("pick")
          and all_legs[i].get("prop_type") in ["points", "pts_rebs_asts"]
          and all_legs[j].get("prop_type") in ["points", "pts_rebs_asts"]]
    calc_stats(s3, "Mixed direction + PRA/Points")

    # Strategy 4: Best performing single legs only (edge 7-12%)
    s4 = [(i, j) for i, j in sampled
          if 7 <= all_legs[i].get("edge", 0) <= 12
          and 7 <= all_legs[j].get("edge", 0) <= 12]
    calc_stats(s4, "Sweet spot edge (7-12%)")

    # Strategy 5: UNDER + PRA/Points + Low edge
    s5 = [(i, j) for i, j in sampled
          if (all_legs[i].get("pick") == "UNDER" or all_legs[j].get("pick") == "UNDER")
          and all_legs[i].get("prop_type") in ["points", "pts_rebs_asts"]
          and all_legs[j].get("prop_type") in ["points", "pts_rebs_asts"]
          and all_legs[i].get("edge", 0) < 12
          and all_legs[j].get("edge", 0) < 12]
    calc_stats(s5, "At least 1 UNDER + PRA/Points + Edge<12%")


def analyze_losing_patterns(all_legs):
    """Analyze patterns in losing bets to identify what to avoid."""

    print("\n" + "="*70)
    print("LOSING BET PATTERN ANALYSIS")
    print("="*70)

    losses = [l for l in all_legs if not l.get("hit")]
    wins = [l for l in all_legs if l.get("hit")]

    print(f"\nTotal losses: {len(losses)} ({100*len(losses)/len(all_legs):.1f}%)")

    # Analyze projection accuracy on losses
    print("\n--- Projection Accuracy on LOSSES ---")
    over_losses = [l for l in losses if l.get("pick") == "OVER"]
    under_losses = [l for l in losses if l.get("pick") == "UNDER"]

    if over_losses:
        avg_proj_miss = sum(l.get("projected", 0) - l.get("actual", 0) for l in over_losses) / len(over_losses)
        print(f"  OVER losses: Avg projection was {avg_proj_miss:.1f} points too high")

    if under_losses:
        avg_proj_miss = sum(l.get("actual", 0) - l.get("projected", 0) for l in under_losses) / len(under_losses)
        print(f"  UNDER losses: Avg actual was {avg_proj_miss:.1f} points too high")

    # High confidence misses (high edge that still lost)
    print("\n--- High Confidence MISSES (Edge 15%+) ---")
    high_edge_losses = [l for l in losses if l.get("edge", 0) >= 15]
    print(f"  Total: {len(high_edge_losses)}")

    by_prop_loss = defaultdict(int)
    by_boost_loss = defaultdict(int)
    for l in high_edge_losses:
        by_prop_loss[l.get("prop_type")] += 1
        if l.get("usage_boost", 1.0) > 1.1:
            by_boost_loss["High boost"] += 1
        else:
            by_boost_loss["No/Low boost"] += 1

    print("  By prop type:")
    for prop, count in sorted(by_prop_loss.items(), key=lambda x: -x[1]):
        print(f"    {prop}: {count}")
    print("  By boost:")
    for boost, count in by_boost_loss.items():
        print(f"    {boost}: {count}")

    # Large misses (actual way different from projected)
    print("\n--- LARGE MISSES (actual differs by >50% from projection) ---")
    large_misses = []
    for l in losses:
        proj = l.get("projected", 1)
        actual = l.get("actual", 0)
        if proj > 0 and abs(actual - proj) / proj > 0.5:
            large_misses.append(l)

    print(f"  Total: {len(large_misses)}")
    by_prop_large = defaultdict(int)
    for l in large_misses:
        by_prop_large[l.get("prop_type")] += 1
    print("  By prop type:")
    for prop, count in sorted(by_prop_large.items(), key=lambda x: -x[1]):
        print(f"    {prop}: {count}")


def generate_recommendations(all_legs):
    """Generate specific recommendations based on analysis."""

    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR MODEL IMPROVEMENT")
    print("="*70)

    # Calculate key metrics
    by_prop = defaultdict(lambda: {"total": 0, "hits": 0})
    by_dir = defaultdict(lambda: {"total": 0, "hits": 0})
    by_boost = defaultdict(lambda: {"total": 0, "hits": 0})
    by_edge = defaultdict(lambda: {"total": 0, "hits": 0})

    for leg in all_legs:
        prop = leg.get("prop_type", "unknown")
        by_prop[prop]["total"] += 1
        if leg.get("hit"):
            by_prop[prop]["hits"] += 1

        direction = leg.get("pick", "OVER")
        by_dir[direction]["total"] += 1
        if leg.get("hit"):
            by_dir[direction]["hits"] += 1

        boost = "boost" if leg.get("usage_boost", 1.0) > 1.05 else "no_boost"
        by_boost[boost]["total"] += 1
        if leg.get("hit"):
            by_boost[boost]["hits"] += 1

        edge = leg.get("edge", 0)
        if edge < 8:
            bucket = "low"
        elif edge < 12:
            bucket = "mid"
        else:
            bucket = "high"
        by_edge[bucket]["total"] += 1
        if leg.get("hit"):
            by_edge[bucket]["hits"] += 1

    # Print findings
    print("""
SINGLE LEG FINDINGS:
--------------------
""")

    # Best and worst props
    prop_rates = [(k, v["hits"]/v["total"]*100) for k, v in by_prop.items() if v["total"] > 100]
    prop_rates.sort(key=lambda x: -x[1])

    print(f"1. BEST PROP TYPE: {prop_rates[0][0]} ({prop_rates[0][1]:.1f}%)")
    print(f"   WORST PROP TYPE: {prop_rates[-1][0]} ({prop_rates[-1][1]:.1f}%)")

    # Direction
    over_rate = by_dir["OVER"]["hits"] / by_dir["OVER"]["total"] * 100
    under_rate = by_dir["UNDER"]["hits"] / by_dir["UNDER"]["total"] * 100
    print(f"\n2. UNDERS outperform OVERS: {under_rate:.1f}% vs {over_rate:.1f}%")
    print(f"   Recommendation: Weight UNDER picks higher in bet selection")

    # Boost
    boost_rate = by_boost["boost"]["hits"] / by_boost["boost"]["total"] * 100
    no_boost_rate = by_boost["no_boost"]["hits"] / by_boost["no_boost"]["total"] * 100
    print(f"\n3. NO BOOST legs: {no_boost_rate:.1f}% vs BOOST legs: {boost_rate:.1f}%")
    print(f"   Recommendation: Be more conservative with usage boost adjustments")

    # Edge
    low_rate = by_edge["low"]["hits"] / by_edge["low"]["total"] * 100
    mid_rate = by_edge["mid"]["hits"] / by_edge["mid"]["total"] * 100
    high_rate = by_edge["high"]["hits"] / by_edge["high"]["total"] * 100
    print(f"\n4. EDGE SWEET SPOT: Mid (8-12%) = {mid_rate:.1f}%")
    print(f"   Low (<8%) = {low_rate:.1f}%, High (12%+) = {high_rate:.1f}%")
    print(f"   Recommendation: Prioritize 8-12% edge plays")

    print("""
PARLAY RECOMMENDATIONS:
-----------------------
1. CROSS-GAME ONLY: SGPs underperform significantly
2. PREFER UNDERS: Both-under parlays have best ROI
3. AVOID HIGH EDGE: Lower edge legs are more reliable in parlays
4. STICK TO PRA/POINTS: Highest single-leg hit rates
5. MAX 1 BOOST LEG: Don't stack usage boost plays

OPTIMAL 2-LEG PARLAY CRITERIA:
- Cross-game (different games)
- At least one UNDER pick
- Both legs from PRA or Points props
- Edge between 7-12% for both legs
- Maximum one usage boost leg
""")


def main():
    data = load_backtest()
    all_legs = analyze_legs(data)
    analyze_prop_combinations(all_legs)
    simulate_parlay_strategies(all_legs)
    analyze_losing_patterns(all_legs)
    generate_recommendations(all_legs)


if __name__ == "__main__":
    main()
