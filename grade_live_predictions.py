#!/usr/bin/env python3
"""
grade_live_predictions.py — Grade in-game prediction accuracy.

Compares live projections (logged per quarter) against actual final stats.
Answers: "When the system said 'Proj: 14.8' at halftime, how close was it?"

Usage:
    python grade_live_predictions.py
    python grade_live_predictions.py --date 2026-10-28
"""

import argparse
import json
import os
import sys
import requests
import time
from collections import defaultdict
from datetime import datetime

API_KEY = os.environ.get("BALLDONTLIE_API_KEY", "")
BASE_URL = "https://api.balldontlie.io/v1"

PROP_TO_STAT = {
    "points": ["pts"], "rebounds": ["reb"], "assists": ["ast"],
    "threes": ["fg3m"], "pts_rebs_asts": ["pts", "reb", "ast"],
}


def load_live_predictions(log_file, date_filter=None):
    preds = []
    if not os.path.exists(log_file):
        print(f"No live predictions file: {log_file}")
        return preds
    with open(log_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if date_filter:
                    ts = rec.get("timestamp", "")[:10]
                    if ts != date_filter:
                        continue
                preds.append(rec)
            except json.JSONDecodeError:
                continue
    return preds


def get_box_scores(game_ids):
    """Fetch final box scores for games."""
    all_stats = {}
    headers = {"Authorization": API_KEY}
    for gid in game_ids:
        resp = requests.get(f"{BASE_URL}/stats", params={"game_ids[]": gid, "per_page": 50},
                           headers=headers, timeout=15)
        if resp.status_code != 200:
            continue
        for s in resp.json().get("data", []):
            pname = f"{s['player']['first_name']} {s['player']['last_name']}"
            mins = s.get("min", "0")
            if isinstance(mins, str) and ":" in mins:
                mins = int(mins.split(":")[0])
            else:
                mins = int(float(mins or 0))
            all_stats[f"{gid}|{pname}"] = {
                "pts": s.get("pts", 0), "reb": s.get("reb", 0),
                "ast": s.get("ast", 0), "fg3m": s.get("fg3m", 0),
                "min": mins,
            }
        time.sleep(0.35)
    return all_stats


def grade(preds, actuals):
    results = []
    for p in preds:
        gid = p.get("game_id")
        player = p.get("player", "")
        prop_type = p.get("prop_type", "")
        stat_keys = PROP_TO_STAT.get(prop_type)
        if not stat_keys:
            continue

        box_key = f"{gid}|{player}"
        box = actuals.get(box_key)
        if not box:
            continue

        actual_final = sum(box.get(k, 0) for k in stat_keys)
        projected = p.get("projected_final", 0)
        current = p.get("current_stat", 0)
        line = p.get("line", 0)
        period = p.get("period", 0)
        side = p.get("side", "")

        # Projection accuracy
        proj_error = abs(projected - actual_final)

        # Did the live projection get the direction right?
        if side == "over":
            proj_says_hit = projected > line
            actually_hit = actual_final > line
        else:
            proj_says_hit = projected < line
            actually_hit = actual_final < line

        results.append({
            "player": player, "prop_type": prop_type, "period": period,
            "line": line, "side": side,
            "current_stat": current, "projected_final": round(projected, 1),
            "actual_final": actual_final,
            "proj_error": round(proj_error, 1),
            "direction_correct": proj_says_hit == actually_hit,
            "live_p": p.get("live_p_over" if side == "over" else "live_p_under", 0.5),
        })
    return results


def print_report(results):
    if not results:
        print("No graded live predictions.")
        return

    print(f"\n{'='*70}")
    print(f"  LIVE IN-GAME PREDICTION ACCURACY")
    print(f"  {len(results)} predictions graded")
    print(f"{'='*70}")

    # By quarter
    by_period = defaultdict(list)
    for r in results:
        by_period[r["period"]].append(r)

    print(f"\n  BY QUARTER:")
    print(f"  {'Quarter':<10} {'Count':>6} {'Avg Error':>10} {'Dir Acc':>8}")
    print(f"  {'-'*40}")
    for period in sorted(by_period):
        pr = by_period[period]
        avg_err = sum(r["proj_error"] for r in pr) / len(pr)
        dir_acc = sum(1 for r in pr if r["direction_correct"]) / len(pr)
        label = f"Q{period}" if period <= 4 else f"OT{period-4}"
        print(f"  {label:<10} {len(pr):>6} {avg_err:>10.1f} {dir_acc:>7.1%}")

    # Overall
    avg_err = sum(r["proj_error"] for r in results) / len(results)
    dir_acc = sum(1 for r in results if r["direction_correct"]) / len(results)
    print(f"\n  OVERALL: avg error = {avg_err:.1f}, direction accuracy = {dir_acc:.1%}")

    # By prop type
    print(f"\n  BY PROP TYPE:")
    by_prop = defaultdict(list)
    for r in results:
        by_prop[r["prop_type"]].append(r)
    for prop in sorted(by_prop):
        pr = by_prop[prop]
        avg_err = sum(r["proj_error"] for r in pr) / len(pr)
        dir_acc = sum(1 for r in pr if r["direction_correct"]) / len(pr)
        print(f"  {prop:<18} {len(pr):>5} preds | err={avg_err:.1f} | dir_acc={dir_acc:.1%}")

    # Key question: does accuracy improve later in the game?
    print(f"\n  ACCURACY BY GAME PROGRESS:")
    for period in sorted(by_period):
        if period < 2:
            continue
        pr = by_period[period]
        dir_acc = sum(1 for r in pr if r["direction_correct"]) / len(pr)
        label = f"Q{period}" if period <= 4 else f"OT{period-4}"
        bar = "#" * int(dir_acc * 40)
        print(f"  {label}: {dir_acc:.1%} {bar}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Grade specific date (YYYY-MM-DD)")
    parser.add_argument("--file", help="Live predictions JSONL file")
    args = parser.parse_args()

    now = datetime.now()
    season_year = now.year if now.month >= 10 else now.year - 1
    season = f"{season_year}-{str(season_year + 1)[-2:]}"

    log_file = args.file or f"outputs/live_predictions_{season}.jsonl"
    preds = load_live_predictions(log_file, date_filter=args.date)
    print(f"Loaded {len(preds)} live predictions from {log_file}")

    if not preds:
        return

    game_ids = list(set(p.get("game_id") for p in preds if p.get("game_id")))
    print(f"Fetching box scores for {len(game_ids)} games...")
    actuals = get_box_scores(game_ids)
    print(f"Got stats for {len(actuals)} player lines")

    results = grade(preds, actuals)
    print_report(results)

    # Save
    out_file = f"outputs/live_grading_{args.date or 'all'}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
