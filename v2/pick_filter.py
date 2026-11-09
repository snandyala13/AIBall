"""
pick_filter.py - Empirical pick filter based on rolling hit rates from grading data.

NOT an ML model. Just a lookup table that auto-updates nightly from
grade_predictions.py output. Checks category-level profitability before
surfacing picks.

Categories tracked:
- side (over/under)
- prop_type (points, rebounds, assists, threes, pts_rebs_asts)
- edge_bucket (0-3, 3-5, 5-10, 10-15, 15-20, 20+)
- side x prop_type (e.g., under_threes, over_points)
- side x edge_bucket (e.g., under_3-5, over_10-15)
- side x prop_type x edge_bucket (finest grain)

Each category stores: {picks, hits, profit, hit_rate, roi, last_updated}
"""

import json
import os
from datetime import datetime, timedelta


# Edge bucket definitions: (label, low_inclusive, high_exclusive)
EDGE_BUCKETS = [
    ("0-3", 0, 3),
    ("3-5", 3, 5),
    ("5-10", 5, 10),
    ("10-15", 10, 15),
    ("15-20", 15, 20),
    ("20+", 20, 100),
]


def _edge_bucket_label(edge_pct):
    """Return the bucket label for a given edge percent."""
    for label, lo, hi in EDGE_BUCKETS:
        if lo <= edge_pct < hi:
            return label
    # Anything >= 100 is extreme outlier
    return "20+"


class EmpiricalFilter:
    """
    Empirical pick filter based on rolling hit rates from grading data.

    Updates nightly from grade_predictions.py output.
    Checks category-level profitability before surfacing picks.
    """

    def __init__(self, filter_file="outputs/empirical_filter.json", min_sample=30):
        self.filter_file = filter_file
        self.min_sample = min_sample
        self.categories = {}  # {category_key: {picks, hits, profit, hit_rate, roi, last_updated}}
        self.metadata = {}

        # Load existing filter data if available
        if os.path.exists(self.filter_file):
            try:
                with open(self.filter_file) as f:
                    data = json.load(f)
                self.categories = data.get("categories", {})
                self.metadata = data.get("metadata", {})
            except (json.JSONDecodeError, IOError):
                # Corrupt or unreadable file - start fresh
                self.categories = {}
                self.metadata = {}

    def update_from_grading(self, graded_picks, window_days=30):
        """
        Rebuild filter categories from graded pick results.

        Args:
            graded_picks: list of graded pick dicts with keys:
                player, prop_type, side, line, actual, hit, profit,
                edge_percent, confidence, etc.
            window_days: only consider picks from the last N days
        """
        cutoff = datetime.now() - timedelta(days=window_days)
        cutoff_str = cutoff.strftime("%Y-%m-%d")

        # Filter to resolved picks within the window
        recent = []
        for pick in graded_picks:
            if pick.get("actual") is None:
                continue
            if pick.get("hit") is None:
                continue
            # If there's a date field, filter by window
            pick_date = pick.get("date", "")
            if pick_date and pick_date < cutoff_str:
                continue
            recent.append(pick)

        if not recent:
            return

        # Rebuild all categories from scratch
        categories = {}

        for pick in recent:
            side = pick.get("side", "over")
            prop_type = pick.get("prop_type", "")
            edge_pct = pick.get("edge_percent", 0)
            hit = pick.get("hit", False)
            profit = pick.get("profit", 0.0)
            bucket = _edge_bucket_label(abs(edge_pct))

            # Build all category keys for this pick
            keys = [
                "side:{}".format(side),
                "prop:{}".format(prop_type),
                "edge:{}".format(bucket),
                "side_prop:{}_{}".format(side, prop_type),
                "side_edge:{}_{}".format(side, bucket),
                "side_prop_edge:{}_{}_{}".format(side, prop_type, bucket),
            ]

            for key in keys:
                if key not in categories:
                    categories[key] = {
                        "picks": 0,
                        "hits": 0,
                        "profit": 0.0,
                    }
                cat = categories[key]
                cat["picks"] += 1
                if hit:
                    cat["hits"] += 1
                cat["profit"] += profit

        # Compute derived stats
        now_str = datetime.now().isoformat()
        for key, cat in categories.items():
            n = cat["picks"]
            cat["hit_rate"] = round(cat["hits"] / n * 100, 2) if n > 0 else 0.0
            cat["roi"] = round(cat["profit"] / n * 100, 2) if n > 0 else 0.0
            cat["profit"] = round(cat["profit"], 3)
            cat["last_updated"] = now_str

        self.categories = categories
        self.metadata = {
            "generated_at": now_str,
            "total_graded_picks": len(recent),
            "window_days": window_days,
            "cutoff_date": cutoff_str,
        }

        # Save to JSON (human-readable)
        self._save()

        # Append daily snapshot to history
        self._append_history()

    def _save(self):
        """Save filter data to JSON file."""
        os.makedirs(os.path.dirname(self.filter_file) or ".", exist_ok=True)
        data = {
            "metadata": self.metadata,
            "categories": self.categories,
        }
        with open(self.filter_file, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)

    def _append_history(self):
        """Append a daily snapshot to the history JSONL file."""
        history_file = self.filter_file.replace(".json", "_history.jsonl")
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "metadata": self.metadata,
            "categories": self.categories,
        }
        os.makedirs(os.path.dirname(history_file) or ".", exist_ok=True)
        with open(history_file, "a") as f:
            f.write(json.dumps(snapshot) + "\n")

    def should_show(self, side, prop_type, edge_percent):
        """
        Decide whether to show a pick based on empirical hit rates.

        Args:
            side: "over" or "under"
            prop_type: e.g. "points", "threes", "rebounds"
            edge_percent: model's estimated edge (0-100 scale)

        Returns:
            dict: {"show": bool, "tier": str, "reason": str}
        """
        # Hard rule: edge > 20% is always bogus data
        if edge_percent > 20:
            return {
                "show": False,
                "tier": "hidden",
                "reason": "Edge >20% indicates unreliable data",
            }

        # If no filter data exists, show everything
        if not self.categories:
            return {
                "show": True,
                "tier": "marginal",
                "reason": "No grading data yet - showing all picks",
            }

        bucket = _edge_bucket_label(abs(edge_percent))

        # Check from finest grain to coarsest:
        # 1. side x prop x edge (finest)
        # 2. side x prop
        # 3. side x edge
        # 4. prop
        # 5. side (coarsest)
        lookup_chain = [
            ("side_prop_edge:{}_{}_{}".format(side, prop_type, bucket),
             "{} {} edge {}%".format(side, prop_type, bucket)),
            ("side_prop:{}_{}".format(side, prop_type),
             "{} {}".format(side, prop_type)),
            ("side_edge:{}_{}".format(side, bucket),
             "{} edge {}%".format(side, bucket)),
            ("prop:{}".format(prop_type),
             prop_type),
            ("side:{}".format(side),
             "{}s".format(side)),
        ]

        # Find the finest-grain category with enough data
        best_key = None
        best_label = None
        best_cat = None
        for key, label in lookup_chain:
            cat = self.categories.get(key)
            if cat and cat.get("picks", 0) >= self.min_sample:
                best_key = key
                best_label = label
                best_cat = cat
                break

        # No category has enough data - show the pick
        if best_cat is None:
            return {
                "show": True,
                "tier": "marginal",
                "reason": "Insufficient sample size (< {} picks in category)".format(
                    self.min_sample),
            }

        hit_rate = best_cat.get("hit_rate", 50.0)
        n = best_cat.get("picks", 0)

        # Override: if side x prop combo has very poor hit rate with large sample,
        # always hide regardless of edge bucket
        side_prop_key = "side_prop:{}_{}".format(side, prop_type)
        side_prop_cat = self.categories.get(side_prop_key)
        if (side_prop_cat and
                side_prop_cat.get("picks", 0) >= 50 and
                side_prop_cat.get("hit_rate", 50.0) < 45.0):
            return {
                "show": False,
                "tier": "hidden",
                "reason": "{} {} hit rate {:.1f}% ({} picks) - consistently losing".format(
                    side, prop_type,
                    side_prop_cat["hit_rate"],
                    side_prop_cat["picks"]),
            }

        # Tier assignment based on hit rate
        if hit_rate >= 52.0:
            return {
                "show": True,
                "tier": "strong",
                "reason": "{} hit rate {:.1f}% ({} picks)".format(
                    best_label, hit_rate, n),
            }
        elif hit_rate >= 48.0:
            return {
                "show": True,
                "tier": "marginal",
                "reason": "{} hit rate {:.1f}% ({} picks)".format(
                    best_label, hit_rate, n),
            }
        else:
            return {
                "show": False,
                "tier": "hidden",
                "reason": "{} hit rate {:.1f}% ({} picks) - below threshold".format(
                    best_label, hit_rate, n),
            }

    def get_filter_summary(self):
        """
        Return a dict of all categories with their current hit rates.
        Suitable for display in the UI.
        """
        summary = {}
        for key, cat in sorted(self.categories.items()):
            n = cat.get("picks", 0)
            if n < 5:
                continue
            summary[key] = {
                "picks": n,
                "hits": cat.get("hits", 0),
                "hit_rate": cat.get("hit_rate", 0.0),
                "roi": cat.get("roi", 0.0),
                "profit": cat.get("profit", 0.0),
            }
        return {
            "metadata": self.metadata,
            "categories": summary,
        }

    def get_recommended_strategy(self):
        """
        Return a plain-English summary of what categories to focus on / avoid.
        """
        focus = []
        avoid = []

        # Look at side x prop combos (the most actionable level)
        for key, cat in self.categories.items():
            if not key.startswith("side_prop:"):
                continue
            n = cat.get("picks", 0)
            if n < self.min_sample:
                continue
            hr = cat.get("hit_rate", 50.0)
            label = key.replace("side_prop:", "").replace("_", " ")
            entry = "{} ({:.0f}% hit, {} picks)".format(label, hr, n)
            if hr >= 52.0:
                focus.append((hr, entry))
            elif hr < 48.0:
                avoid.append((hr, entry))

        # Also check pure prop types
        for key, cat in self.categories.items():
            if not key.startswith("prop:"):
                continue
            n = cat.get("picks", 0)
            if n < self.min_sample:
                continue
            hr = cat.get("hit_rate", 50.0)
            label = key.replace("prop:", "")
            entry = "{} ({:.0f}% hit, {} picks)".format(label, hr, n)
            if hr < 45.0:
                avoid.append((hr, entry))

        # Sort focus by hit rate desc, avoid by hit rate asc
        focus.sort(key=lambda x: -x[0])
        avoid.sort(key=lambda x: x[0])

        parts = []
        if focus:
            focus_str = ", ".join(e for _, e in focus[:5])
            parts.append("Focus on: {}".format(focus_str))
        if avoid:
            avoid_str = ", ".join(e for _, e in avoid[:5])
            parts.append("Avoid: {}".format(avoid_str))

        if not parts:
            total = self.metadata.get("total_graded_picks", 0)
            if total > 0:
                return "No strong patterns yet ({} picks graded)".format(total)
            return "No grading data available yet"

        return ". ".join(parts)
