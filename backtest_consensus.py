#!/usr/bin/env python3
"""
backtest_consensus.py — Backtest cross-book consensus deviation strategies

Tests whether betting against DK when it deviates from the multi-book
consensus line is profitable. Uses enriched legs with per-book odds data.

Usage:
    python backtest_consensus.py --season 2024-25
    python backtest_consensus.py --season 2024-25 --strategy consensus-under --threshold 1.0
    python backtest_consensus.py --season 2024-25 --strategy all
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"

SEASONS = {
    "2023-24": "backtest_2023_24_multibook.json",
    "2024-25": "backtest_2024_25_multibook.json",
    "2025-26": "backtest_oos_multibook.json",
}

# ---------------------------------------------------------------------------
# Devig helpers (inline fallbacks if consensus_engine not ready yet)
# ---------------------------------------------------------------------------

def _american_to_prob(odds: int) -> float:
    if odds >= 100:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def _devig_power(over_odds: int, under_odds: int) -> Tuple[float, float]:
    p_over_raw = _american_to_prob(over_odds)
    p_under_raw = _american_to_prob(under_odds)
    total = p_over_raw + p_under_raw
    if total <= 0:
        return 0.5, 0.5
    return p_over_raw / total, p_under_raw / total


def _compute_consensus_prob(all_books: Dict) -> Tuple[float, float]:
    if not all_books:
        return 0.5, 0.5
    p_overs, p_unders = [], []
    for bk, data in all_books.items():
        over_o = data.get("over_odds", -110)
        under_o = data.get("under_odds", -110)
        if over_o and under_o:
            po, pu = _devig_power(over_o, under_o)
            p_overs.append(po)
            p_unders.append(pu)
    if not p_overs:
        return 0.5, 0.5
    avg_o = sum(p_overs) / len(p_overs)
    avg_u = sum(p_unders) / len(p_unders)
    total = avg_o + avg_u
    return avg_o / total, avg_u / total


try:
    from consensus_engine import american_to_prob, devig_power, compute_consensus_prob
except ImportError:
    american_to_prob = _american_to_prob
    devig_power = _devig_power
    compute_consensus_prob = _compute_consensus_prob


# ---------------------------------------------------------------------------
# Payout / ROI helpers
# ---------------------------------------------------------------------------

def american_to_decimal(odds: int) -> float:
    if odds >= 100:
        return 1.0 + odds / 100.0
    else:
        return 1.0 + 100.0 / abs(odds)


def compute_payout(odds: int) -> float:
    """Net profit on a $100 bet if it wins."""
    if odds >= 100:
        return float(odds)
    else:
        return 10000.0 / abs(odds)


def bet_result(leg: Dict, pick: str) -> Tuple[bool, float]:
    """
    Returns (hit, profit) for a $100 bet on the given side.
    Push (actual == line) returns (False, 0).
    """
    actual = leg.get("actual", 0)
    line = leg.get("line", 0)
    over_odds = leg.get("over_odds", -110)
    under_odds = leg.get("under_odds", -110)

    if actual == line:
        return False, 0.0

    if pick == "OVER":
        hit = actual > line
        profit = compute_payout(over_odds) if hit else -100.0
    else:
        hit = actual < line
        profit = compute_payout(under_odds) if hit else -100.0

    return hit, profit


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_legs(season: str) -> List[Dict]:
    fname = SEASONS.get(season)
    if not fname:
        print(f"Unknown season: {season}. Options: {list(SEASONS.keys())}")
        sys.exit(1)

    fpath = OUTPUT_DIR / fname
    if not fpath.exists():
        print(f"Enriched legs not found: {fpath}")
        print("Run: python fetch_multibook_odds.py --export <season>")
        sys.exit(1)

    with open(fpath) as f:
        legs = json.load(f)

    multi = sum(1 for l in legs if l.get("num_books", 0) > 1)
    print(f"Loaded {len(legs):,} legs, {multi:,} with multi-book data ({100*multi/max(1,len(legs)):.1f}%)")
    return legs


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

class StrategyResult:
    def __init__(self, name: str):
        self.name = name
        self.bets: List[Dict] = []

    def add(self, leg: Dict, pick: str, meta: Optional[Dict] = None):
        hit, profit = bet_result(leg, pick)
        entry = {
            "player": leg.get("player"),
            "prop_type": leg.get("prop_type"),
            "date": leg.get("date"),
            "line": leg.get("line"),
            "actual": leg.get("actual"),
            "pick": pick,
            "hit": hit,
            "profit": profit,
            "over_odds": leg.get("over_odds", -110),
            "under_odds": leg.get("under_odds", -110),
            "num_books": leg.get("num_books", 0),
            "dk_vs_consensus": leg.get("dk_vs_consensus", 0),
        }
        if meta:
            entry.update(meta)
        self.bets.append(entry)

    def summary(self) -> Dict:
        if not self.bets:
            return {"name": self.name, "bets": 0, "win_rate": 0, "roi": 0,
                    "profit": 0, "monthly": {}, "by_prop": {}, "quarterly_stable": False}

        total = len(self.bets)
        wins = sum(1 for b in self.bets if b["hit"])
        total_profit = sum(b["profit"] for b in self.bets)
        total_wagered = total * 100.0
        roi = 100.0 * total_profit / total_wagered if total_wagered else 0

        monthly = defaultdict(lambda: {"bets": 0, "wins": 0, "profit": 0.0})
        for b in self.bets:
            m = b["date"][:7]
            monthly[m]["bets"] += 1
            monthly[m]["wins"] += int(b["hit"])
            monthly[m]["profit"] += b["profit"]

        monthly_out = {}
        for m, v in sorted(monthly.items()):
            w = 100.0 * v["profit"] / (v["bets"] * 100) if v["bets"] else 0
            monthly_out[m] = {
                "bets": v["bets"],
                "wins": v["wins"],
                "win_rate": round(100 * v["wins"] / v["bets"], 1) if v["bets"] else 0,
                "roi": round(w, 2),
                "profit": round(v["profit"], 2),
            }

        by_prop = defaultdict(lambda: {"bets": 0, "wins": 0, "profit": 0.0})
        for b in self.bets:
            pt = b["prop_type"]
            by_prop[pt]["bets"] += 1
            by_prop[pt]["wins"] += int(b["hit"])
            by_prop[pt]["profit"] += b["profit"]

        by_prop_out = {}
        for pt, v in sorted(by_prop.items()):
            w = 100.0 * v["profit"] / (v["bets"] * 100) if v["bets"] else 0
            by_prop_out[pt] = {
                "bets": v["bets"],
                "wins": v["wins"],
                "win_rate": round(100 * v["wins"] / v["bets"], 1) if v["bets"] else 0,
                "roi": round(w, 2),
            }

        # Walk-forward: quarterly stability
        quarterly = defaultdict(float)
        quarterly_wager = defaultdict(float)
        for b in self.bets:
            d = b["date"]
            try:
                dt = datetime.strptime(d, "%Y-%m-%d")
                q = f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
            except ValueError:
                q = "unknown"
            quarterly[q] += b["profit"]
            quarterly_wager[q] += 100.0

        quarters_profitable = sum(1 for q in quarterly if quarterly[q] > 0)
        total_quarters = len(quarterly)
        stable = quarters_profitable >= (total_quarters * 0.6) if total_quarters >= 2 else False

        return {
            "name": self.name,
            "bets": total,
            "wins": wins,
            "win_rate": round(100 * wins / total, 2),
            "roi": round(roi, 2),
            "profit": round(total_profit, 2),
            "quarters_profitable": f"{quarters_profitable}/{total_quarters}",
            "quarterly_stable": stable,
            "monthly": monthly_out,
            "by_prop": by_prop_out,
            "quarterly": {q: round(100 * quarterly[q] / quarterly_wager[q], 2)
                          for q in sorted(quarterly)},
        }


def strategy_consensus_deviation(legs: List[Dict], threshold: float,
                                  under_only: bool = False) -> StrategyResult:
    name = f"consensus_{'under' if under_only else 'both'}_{threshold}"
    result = StrategyResult(name)

    for leg in legs:
        dev = leg.get("dk_vs_consensus", 0)
        if leg.get("num_books", 0) < 2:
            continue

        if dev > threshold:
            # DK line is higher than consensus → bet UNDER
            result.add(leg, "UNDER", {"deviation": dev})
        elif not under_only and dev < -threshold:
            # DK line is lower than consensus → bet OVER
            result.add(leg, "OVER", {"deviation": dev})

    return result


def strategy_devigged_ev(legs: List[Dict], ev_threshold: float) -> StrategyResult:
    name = f"devigged_ev_{ev_threshold}"
    result = StrategyResult(name)

    for leg in legs:
        books = leg.get("all_books", {})
        if len(books) < 2:
            continue

        p_over_true, p_under_true = compute_consensus_prob(books)

        over_odds = leg.get("over_odds", -110)
        under_odds = leg.get("under_odds", -110)

        dec_over = american_to_decimal(over_odds)
        dec_under = american_to_decimal(under_odds)

        ev_over = p_over_true * (dec_over - 1) - (1 - p_over_true)
        ev_under = p_under_true * (dec_under - 1) - (1 - p_under_true)

        if ev_under > ev_threshold / 100.0:
            result.add(leg, "UNDER", {"ev": round(ev_under, 4)})
        elif ev_over > ev_threshold / 100.0:
            result.add(leg, "OVER", {"ev": round(ev_over, 4)})

    return result


def strategy_devigged_ev_under_only(legs: List[Dict], ev_threshold: float) -> StrategyResult:
    name = f"devigged_ev_under_{ev_threshold}"
    result = StrategyResult(name)

    for leg in legs:
        books = leg.get("all_books", {})
        if len(books) < 2:
            continue

        _, p_under_true = compute_consensus_prob(books)
        under_odds = leg.get("under_odds", -110)
        dec_under = american_to_decimal(under_odds)
        ev_under = p_under_true * (dec_under - 1) - (1 - p_under_true)

        if ev_under > ev_threshold / 100.0:
            result.add(leg, "UNDER", {"ev": round(ev_under, 4)})

    return result


def strategy_book_count_filter(legs: List[Dict], min_books: int,
                                threshold: float) -> StrategyResult:
    name = f"books_{min_books}+_dev_{threshold}"
    result = StrategyResult(name)

    for leg in legs:
        if leg.get("num_books", 0) < min_books:
            continue
        dev = leg.get("dk_vs_consensus", 0)
        if dev > threshold:
            result.add(leg, "UNDER", {"deviation": dev, "num_books": leg["num_books"]})

    return result


def strategy_pinnacle_anchor(legs: List[Dict], threshold: float) -> StrategyResult:
    name = f"pinnacle_anchor_{threshold}"
    result = StrategyResult(name)

    for leg in legs:
        books = leg.get("all_books", {})
        pin = books.get("pinnacle") or books.get("pinnaclesports")
        dk = books.get("draftkings")
        if not pin or not dk:
            continue

        pin_line = pin.get("line", 0)
        dk_line = dk.get("line", 0)
        dev = dk_line - pin_line

        if dev > threshold:
            result.add(leg, "UNDER", {"dk_line": dk_line, "pin_line": pin_line, "deviation": dev})
        elif dev < -threshold:
            result.add(leg, "OVER", {"dk_line": dk_line, "pin_line": pin_line, "deviation": dev})

    return result


def strategy_pinnacle_anchor_under_only(legs: List[Dict], threshold: float) -> StrategyResult:
    name = f"pinnacle_under_{threshold}"
    result = StrategyResult(name)

    for leg in legs:
        books = leg.get("all_books", {})
        pin = books.get("pinnacle") or books.get("pinnaclesports")
        dk = books.get("draftkings")
        if not pin or not dk:
            continue

        dev = dk.get("line", 0) - pin.get("line", 0)
        if dev > threshold:
            result.add(leg, "UNDER", {"deviation": dev})

    return result


def strategy_pinnacle_devigged_ev(legs: List[Dict], ev_threshold: float) -> StrategyResult:
    name = f"pinnacle_ev_{ev_threshold}"
    result = StrategyResult(name)

    for leg in legs:
        books = leg.get("all_books", {})
        pin = books.get("pinnacle") or books.get("pinnaclesports")
        if not pin:
            continue

        pin_over = pin.get("over_odds", -110)
        pin_under = pin.get("under_odds", -110)
        if not pin_over or not pin_under:
            continue

        p_over_true, p_under_true = devig_power(pin_over, pin_under)

        dk_over = leg.get("over_odds", -110)
        dk_under = leg.get("under_odds", -110)
        dec_over = american_to_decimal(dk_over)
        dec_under = american_to_decimal(dk_under)

        ev_over = p_over_true * (dec_over - 1) - (1 - p_over_true)
        ev_under = p_under_true * (dec_under - 1) - (1 - p_under_true)

        if ev_under > ev_threshold / 100.0:
            result.add(leg, "UNDER", {"ev": round(ev_under, 4)})
        elif ev_over > ev_threshold / 100.0:
            result.add(leg, "OVER", {"ev": round(ev_over, 4)})

    return result


# ---------------------------------------------------------------------------
# Run all strategies
# ---------------------------------------------------------------------------

ALL_STRATEGY_NAMES = [
    "consensus-both",
    "consensus-under",
    "devigged-ev",
    "devigged-ev-under",
    "book-count",
    "pinnacle-anchor",
    "pinnacle-under",
    "pinnacle-ev",
]


def run_all(legs: List[Dict]) -> List[Dict]:
    results = []

    for t in [0.5, 1.0, 1.5, 2.0]:
        results.append(strategy_consensus_deviation(legs, t, under_only=False).summary())
        results.append(strategy_consensus_deviation(legs, t, under_only=True).summary())

    for ev in [2.0, 3.0, 5.0]:
        results.append(strategy_devigged_ev(legs, ev).summary())
        results.append(strategy_devigged_ev_under_only(legs, ev).summary())

    for books in [3, 5, 7]:
        for t in [0.5, 1.0]:
            results.append(strategy_book_count_filter(legs, books, t).summary())

    for t in [0.5, 1.0, 1.5]:
        results.append(strategy_pinnacle_anchor(legs, t).summary())
        results.append(strategy_pinnacle_anchor_under_only(legs, t).summary())

    for ev in [2.0, 3.0, 5.0]:
        results.append(strategy_pinnacle_devigged_ev(legs, ev).summary())

    return results


def run_single(legs: List[Dict], strategy: str, threshold: float) -> List[Dict]:
    results = []

    if strategy == "consensus-both":
        results.append(strategy_consensus_deviation(legs, threshold, under_only=False).summary())
    elif strategy == "consensus-under":
        results.append(strategy_consensus_deviation(legs, threshold, under_only=True).summary())
    elif strategy == "devigged-ev":
        results.append(strategy_devigged_ev(legs, threshold).summary())
    elif strategy == "devigged-ev-under":
        results.append(strategy_devigged_ev_under_only(legs, threshold).summary())
    elif strategy == "book-count":
        results.append(strategy_book_count_filter(legs, 5, threshold).summary())
    elif strategy == "pinnacle-anchor":
        results.append(strategy_pinnacle_anchor(legs, threshold).summary())
    elif strategy == "pinnacle-under":
        results.append(strategy_pinnacle_anchor_under_only(legs, threshold).summary())
    elif strategy == "pinnacle-ev":
        results.append(strategy_pinnacle_devigged_ev(legs, threshold).summary())
    else:
        print(f"Unknown strategy: {strategy}")
        print(f"Options: {', '.join(ALL_STRATEGY_NAMES)}, all")
        sys.exit(1)

    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_results(results: List[Dict]):
    if not results:
        print("No results.")
        return

    # Sort by ROI descending
    results = sorted(results, key=lambda r: r.get("roi", 0), reverse=True)

    print()
    print(f"{'Strategy':<35} {'Bets':>6} {'Win%':>6} {'ROI%':>7} {'Profit':>10} {'Qtrs':>6} {'Stable':>6}")
    print("-" * 82)
    for r in results:
        if r["bets"] == 0:
            continue
        stable_mark = "YES" if r.get("quarterly_stable") else "no"
        print(f"{r['name']:<35} {r['bets']:>6} {r['win_rate']:>5.1f}% {r['roi']:>6.2f}% {r['profit']:>9.0f} {r.get('quarters_profitable', '?'):>6} {stable_mark:>6}")

    # Show top strategies with detail
    top = [r for r in results if r["bets"] >= 50 and r["roi"] > 0]
    if top:
        print(f"\n{'='*82}")
        print("TOP STRATEGIES (positive ROI, 50+ bets)")
        print(f"{'='*82}")
        for r in top[:5]:
            print(f"\n--- {r['name']} ---")
            print(f"  Bets: {r['bets']}, Win: {r['win_rate']}%, ROI: {r['roi']}%, Profit: ${r['profit']:.0f}")
            print(f"  Quarterly: {r.get('quarters_profitable', '?')} profitable, Stable: {r.get('quarterly_stable')}")

            if r.get("by_prop"):
                print("  By prop:")
                for pt, pv in sorted(r["by_prop"].items(), key=lambda x: x[1]["roi"], reverse=True):
                    print(f"    {pt:<20} {pv['bets']:>5} bets  {pv['win_rate']:>5.1f}% win  {pv['roi']:>6.2f}% ROI")

            if r.get("quarterly"):
                print("  By quarter:")
                for q, roi in sorted(r["quarterly"].items()):
                    marker = "+" if roi > 0 else " "
                    print(f"    {q}: {marker}{roi:.2f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Backtest consensus deviation strategies")
    parser.add_argument("--season", required=True, choices=list(SEASONS.keys()))
    parser.add_argument("--strategy", default="all",
                        choices=ALL_STRATEGY_NAMES + ["all"])
    parser.add_argument("--threshold", type=float, default=1.0)
    args = parser.parse_args()

    legs = load_legs(args.season)

    if args.strategy == "all":
        results = run_all(legs)
    else:
        results = run_single(legs, args.strategy, args.threshold)

    print_results(results)

    out_file = OUTPUT_DIR / "consensus_backtest_results.json"
    # Strip individual bets from monthly/by_prop for compact output
    save_results = []
    for r in results:
        save_results.append({k: v for k, v in r.items()})
    with open(out_file, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nDetailed results saved to: {out_file}")


if __name__ == "__main__":
    main()
