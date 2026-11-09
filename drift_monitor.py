#!/usr/bin/env python3
"""
drift_monitor.py - Concept Drift Detection for NBA Prop Betting Models

Tracks model performance over time by matching predictions to actual game
outcomes, detecting calibration drift, and recommending when retraining
is needed.

Usage:
    from drift_monitor import DriftMonitor

    monitor = DriftMonitor()
    report = monitor.generate_report(days_back=30)
    print(report)

    drift = monitor.detect_drift(df)
    if drift['recommendation'] == 'retrain':
        print("Models need retraining!")
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mapping from prop_type names in predictions to box score stat keys.
# These mirror model_engine.py STAT_MAPPING / COMBINED_STAT_MAPPING.
PROP_TO_STAT = {
    "points": ["pts"],
    "rebounds": ["reb"],
    "assists": ["ast"],
    "threes": ["fg3m"],
    "steals": ["stl"],
    "blocks": ["blk"],
    "turnovers": ["turnover"],
    "pts_rebs_asts": ["pts", "reb", "ast"],
    "pts_rebs": ["pts", "reb"],
    "pts_asts": ["pts", "ast"],
    "rebs_asts": ["reb", "ast"],
    "stls_blks": ["stl", "blk"],
}


class DriftMonitor:
    """Tracks model performance and detects when retraining is needed."""

    def __init__(self, predictions_dir: str = "outputs"):
        self.predictions_dir = Path(predictions_dir)
        self._api_client = None
        self._propline_client = None

    # ------------------------------------------------------------------
    # API clients (lazy-loaded -- only needed when resolving outcomes)
    # ------------------------------------------------------------------

    def _get_client(self):
        """Lazy-load BallDontLieClient for fetching box scores."""
        if self._api_client is not None:
            return self._api_client
        from data_pipeline import BallDontLieClient
        api_key = os.getenv("BALLDONTLIE_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "BALLDONTLIE_API_KEY environment variable must be set "
                "to fetch game results for drift monitoring."
            )
        self._api_client = BallDontLieClient(api_key)
        return self._api_client

    def _get_propline_client(self):
        """Lazy-load PropLineClient for graded results (optional, faster)."""
        if self._propline_client is not None:
            return self._propline_client
        try:
            from propline_client import PropLineClient
            client = PropLineClient()
            if client.api_key:
                self._propline_client = client
                return client
        except ImportError:
            pass
        return None

    # ------------------------------------------------------------------
    # Load & match predictions with outcomes
    # ------------------------------------------------------------------

    def _find_prediction_files(self) -> List[Path]:
        """Find all predictions JSONL files."""
        files = sorted(self.predictions_dir.glob("predictions_*.jsonl"))
        return files

    def load_predictions(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load raw predictions from JSONL files.

        Args:
            start_date: Optional YYYY-MM-DD lower bound (inclusive).
            end_date:   Optional YYYY-MM-DD upper bound (inclusive).

        Returns:
            DataFrame with one row per prediction.
        """
        files = self._find_prediction_files()
        if not files:
            logger.warning("No prediction JSONL files found in %s", self.predictions_dir)
            return pd.DataFrame()

        rows: List[dict] = []
        for f in files:
            with open(f, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])

        if start_date:
            df = df[df["date"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["date"] <= pd.Timestamp(end_date)]

        return df.reset_index(drop=True)

    @staticmethod
    def _extract_stat_from_log(log: dict, stat_keys: List[str]) -> Optional[float]:
        """Sum the requested stat keys from a single game log dict."""
        total = 0.0
        for key in stat_keys:
            val = log.get(key)
            if val is None:
                return None
            total += float(val)
        return total

    def _resolve_outcome_for_row(
        self,
        player_name: str,
        date_str: str,
        prop_type: str,
        client,
    ) -> Optional[float]:
        """Fetch the actual stat value for a single prediction.

        Uses /v1/stats to find the player's box score on the given date.
        """
        stat_keys = PROP_TO_STAT.get(prop_type)
        if stat_keys is None:
            return None

        # Search for the player
        player_data = client.search_player(player_name)
        if not player_data:
            return None
        player_id = player_data["id"]

        # Determine season year (Oct+ belongs to that year's season)
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        season = dt.year if dt.month >= 10 else dt.year - 1

        # Fetch game logs for the player around that date
        data = client._make_request("stats", {
            "player_ids[]": player_id,
            "seasons[]": season,
            "start_date": date_str,
            "end_date": date_str,
        })
        if not data or not data.get("data"):
            return None

        # There should be exactly one game on the target date
        for log in data["data"]:
            game_date = log.get("game", {}).get("date", "")[:10]
            if game_date == date_str:
                return self._extract_stat_from_log(log, stat_keys)

        return None

    def _resolve_via_propline(
        self,
        df: pd.DataFrame,
        cache: Dict[str, float],
    ) -> Tuple[List[Optional[float]], int]:
        """Try to resolve outcomes using PropLine graded results.

        PropLine returns actual values and hit/loss/push per event, which is
        much faster than fetching individual box scores from Balldontlie.

        Returns (actuals_list, new_cache_count) -- entries are None when
        PropLine could not resolve a particular row.
        """
        propline = self._get_propline_client()
        if propline is None:
            return [None] * len(df), 0

        # Group predictions by date so we can batch-fetch events
        actuals = [None] * len(df)  # type: List[Optional[float]]
        new_entries = 0

        # Get PropLine events for each unique date in predictions
        dates_seen = set()  # type: set
        # {date_str: {(player_lower, prop_type): actual_value}}
        propline_results_by_date = {}  # type: Dict[str, Dict[Tuple[str, str], float]]

        for idx, row in df.iterrows():
            date_str = row["date"].strftime("%Y-%m-%d")
            cache_key = "{0}|{1}|{2}".format(date_str, row["player"], row["prop_type"])

            if cache_key in cache:
                actuals[idx] = cache[cache_key]
                continue

            if pd.Timestamp(date_str) >= pd.Timestamp(datetime.now().strftime("%Y-%m-%d")):
                continue

            # Fetch PropLine graded results for this date's events (once per date)
            if date_str not in dates_seen:
                dates_seen.add(date_str)
                events = propline.get_events(date_str=date_str)
                date_results = {}  # type: Dict[Tuple[str, str], float]
                for event in events:
                    event_id = str(event.get("id", ""))
                    graded = propline.get_graded_results(event_id)
                    for g in graded:
                        player = g.get("player", "").lower().strip()
                        prop_type = g.get("prop_type", "")
                        actual_val = g.get("actual_value")
                        if player and prop_type and actual_val is not None:
                            date_results[(player, prop_type)] = actual_val
                propline_results_by_date[date_str] = date_results

            # Look up the result
            date_results = propline_results_by_date.get(date_str, {})
            player_key = row["player"].lower().strip()
            lookup_key = (player_key, row["prop_type"])
            if lookup_key in date_results:
                actual = date_results[lookup_key]
                actuals[idx] = actual
                cache[cache_key] = actual
                new_entries += 1

        resolved = sum(1 for a in actuals if a is not None)
        logger.info("PropLine resolved %d/%d outcomes", resolved, len(df))
        return actuals, new_entries

    def load_predictions_with_results(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        cache_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load predictions and match with actual game results.

        For each prediction, fetches the actual box score stat from the API
        and determines whether the bet hit (actual > line for overs, actual
        < line for unders).

        Results are cached to disk so subsequent calls avoid redundant API
        requests.

        When PropLine is available, graded results are fetched first (fast
        bulk resolution). Any remaining unresolved rows fall back to
        Balldontlie box score lookups.

        Args:
            start_date: YYYY-MM-DD lower bound (inclusive).
            end_date:   YYYY-MM-DD upper bound (inclusive).
            cache_path: Path to cache file. Defaults to
                        ``outputs/drift_results_cache.json``.

        Returns:
            DataFrame with columns from predictions plus:
            ``actual``, ``side``, ``hit`` (bool).
        """
        df = self.load_predictions(start_date, end_date)
        if df.empty:
            return df

        # Load cache
        if cache_path is None:
            cache_path = str(self.predictions_dir / "drift_results_cache.json")

        cache: Dict[str, float] = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                cache = {}

        # Phase 1: Try PropLine graded results first (fast bulk resolution)
        actuals, propline_new = self._resolve_via_propline(df, cache)
        new_cache_entries = propline_new

        # Phase 2: Fall back to Balldontlie for any unresolved rows
        client = self._get_client()
        for idx, row in df.iterrows():
            if actuals[idx] is not None:
                continue  # Already resolved by PropLine or cache

            date_str = row["date"].strftime("%Y-%m-%d")
            cache_key = "{0}|{1}|{2}".format(date_str, row["player"], row["prop_type"])

            if cache_key in cache:
                actuals[idx] = cache[cache_key]
                continue

            # Only resolve past games (not today or future)
            if pd.Timestamp(date_str) >= pd.Timestamp(datetime.now().strftime("%Y-%m-%d")):
                continue

            actual = self._resolve_outcome_for_row(
                row["player"], date_str, row["prop_type"], client
            )
            actuals[idx] = actual

            if actual is not None:
                cache[cache_key] = actual
                new_cache_entries += 1

        df["actual"] = actuals

        # Persist cache
        if new_cache_entries > 0:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(cache, f)
            logger.info("Cached %d new outcome lookups (%d total)", new_cache_entries, len(cache))

        # Drop rows where we could not determine the actual
        df = df.dropna(subset=["actual"]).copy()

        if df.empty:
            return df

        # Determine side and hit
        df["side"] = np.where(df["over_prob"] >= 0.5, "over", "under")
        df["hit"] = np.where(
            df["side"] == "over",
            df["actual"] > df["line"],
            df["actual"] < df["line"],
        )
        df["hit"] = df["hit"].astype(bool)

        # Expected probability for the chosen side
        df["expected_prob"] = np.where(
            df["side"] == "over",
            df["over_prob"],
            1.0 - df["over_prob"],
        )

        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Calibration metrics
    # ------------------------------------------------------------------

    def calculate_calibration(self, df: pd.DataFrame) -> dict:
        """Calculate calibration metrics.

        Args:
            df: DataFrame from ``load_predictions_with_results``.

        Returns:
            Dictionary with:
            - ``buckets``: list of {range, count, expected_hr, actual_hr}
            - ``brier_score``: float
            - ``ece``: expected calibration error
            - ``per_prop``: {prop_type: {count, expected_hr, actual_hr, drift}}
        """
        if df.empty:
            return {"buckets": [], "brier_score": None, "ece": None, "per_prop": {}}

        # --- Probability buckets ---
        bucket_edges = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.01]
        buckets = []
        for i in range(len(bucket_edges) - 1):
            lo, hi = bucket_edges[i], bucket_edges[i + 1]
            mask = (df["expected_prob"] >= lo) & (df["expected_prob"] < hi)
            subset = df[mask]
            if len(subset) == 0:
                continue
            buckets.append({
                "range": f"{lo*100:.0f}-{hi*100:.0f}%",
                "count": int(len(subset)),
                "expected_hr": float(subset["expected_prob"].mean()),
                "actual_hr": float(subset["hit"].mean()),
            })

        # --- Brier score ---
        brier = float(((df["expected_prob"] - df["hit"].astype(float)) ** 2).mean())

        # --- Expected Calibration Error ---
        ece = 0.0
        n_total = len(df)
        for b in buckets:
            weight = b["count"] / n_total
            ece += weight * abs(b["actual_hr"] - b["expected_hr"])

        # --- Per-prop breakdown ---
        per_prop: Dict[str, dict] = {}
        for prop_type, group in df.groupby("prop_type"):
            expected_hr = float(group["expected_prob"].mean())
            actual_hr = float(group["hit"].mean())
            per_prop[str(prop_type)] = {
                "count": int(len(group)),
                "expected_hr": round(expected_hr, 4),
                "actual_hr": round(actual_hr, 4),
                "drift": round(actual_hr - expected_hr, 4),
            }

        return {
            "buckets": buckets,
            "brier_score": round(brier, 6),
            "ece": round(ece, 6),
            "per_prop": per_prop,
        }

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def detect_drift(self, df: pd.DataFrame, threshold: float = 0.03) -> dict:
        """Detect if model performance has degraded.

        Args:
            df: DataFrame from ``load_predictions_with_results``.
            threshold: Minimum negative drift to trigger retrain (default 3%).

        Returns:
            Dictionary with drift assessment:
            - ``drift_detected``: bool
            - ``overall_drift``: actual_hr - expected_hr
            - ``worst_prop``: prop type with worst drift
            - ``worst_prop_drift``: drift value for worst prop
            - ``rolling_7d_alert``: bool, True if last 7 days HR is 10%+ below expected
            - ``recommendation``: 'retrain' | 'monitor' | 'ok'
            - ``reason``: human-readable explanation
        """
        result = {
            "drift_detected": False,
            "overall_drift": 0.0,
            "worst_prop": None,
            "worst_prop_drift": 0.0,
            "rolling_7d_alert": False,
            "recommendation": "ok",
            "reason": "Insufficient data",
            "sample_size": 0,
        }

        if df.empty or len(df) < 30:
            result["reason"] = f"Only {len(df)} resolved predictions; need >= 30 for drift analysis."
            return result

        result["sample_size"] = len(df)

        # Overall drift
        overall_expected = float(df["expected_prob"].mean())
        overall_actual = float(df["hit"].mean())
        overall_drift = overall_actual - overall_expected
        result["overall_drift"] = round(overall_drift, 4)

        # Per-prop drift
        worst_prop = None
        worst_drift = 0.0
        for prop_type, group in df.groupby("prop_type"):
            if len(group) < 10:
                continue
            prop_drift = float(group["hit"].mean()) - float(group["expected_prob"].mean())
            if prop_drift < worst_drift:
                worst_drift = prop_drift
                worst_prop = str(prop_type)

        result["worst_prop"] = worst_prop
        result["worst_prop_drift"] = round(worst_drift, 4)

        # Rolling 7-day check
        now = pd.Timestamp(datetime.now().strftime("%Y-%m-%d"))
        seven_days_ago = now - pd.Timedelta(days=7)
        recent = df[df["date"] >= seven_days_ago]
        rolling_7d_alert = False
        if len(recent) >= 10:
            recent_expected = float(recent["expected_prob"].mean())
            recent_actual = float(recent["hit"].mean())
            if (recent_actual - recent_expected) < -0.10:
                rolling_7d_alert = True
        result["rolling_7d_alert"] = rolling_7d_alert

        # Decision logic
        if rolling_7d_alert:
            result["drift_detected"] = True
            result["recommendation"] = "retrain"
            result["reason"] = (
                f"URGENT: Rolling 7-day hit rate is 10%+ below expected "
                f"({recent_actual*100:.1f}% actual vs {recent_expected*100:.1f}% expected)."
            )
        elif overall_drift < -threshold:
            result["drift_detected"] = True
            result["recommendation"] = "retrain"
            result["reason"] = (
                f"Overall drift of {overall_drift*100:+.1f}% exceeds "
                f"-{threshold*100:.0f}% threshold "
                f"({overall_actual*100:.1f}% actual vs {overall_expected*100:.1f}% expected)."
            )
        elif worst_drift < -0.05:
            result["drift_detected"] = True
            result["recommendation"] = "retrain"
            result["reason"] = (
                f"Prop '{worst_prop}' has drift of {worst_drift*100:+.1f}%, "
                f"exceeding -5% single-prop threshold."
            )
        elif overall_drift < -(threshold / 2):
            result["drift_detected"] = False
            result["recommendation"] = "monitor"
            result["reason"] = (
                f"Mild overall drift of {overall_drift*100:+.1f}%. "
                f"Not yet at retrain threshold but worth watching."
            )
        else:
            result["drift_detected"] = False
            result["recommendation"] = "ok"
            result["reason"] = (
                f"Model performing within tolerance. "
                f"Overall drift: {overall_drift*100:+.1f}%."
            )

        return result

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self, days_back: int = 30) -> str:
        """Generate a human-readable performance report.

        Args:
            days_back: Number of days to look back (default 30).

        Returns:
            Formatted multi-line string with performance summary.
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        df = self.load_predictions_with_results(start_date=start_date, end_date=end_date)

        lines: List[str] = [
            "",
            "=" * 70,
            "  MODEL DRIFT REPORT",
            f"  Period: {start_date} to {end_date} ({days_back} days)",
            f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 70,
            "",
        ]

        if df.empty:
            lines.append("  No resolved predictions found for this period.")
            lines.append("  Ensure predictions are logged to outputs/predictions_*.jsonl")
            lines.append("  and that games have been played (outcomes are available).")
            return "\n".join(lines)

        # Overall stats
        overall_expected = df["expected_prob"].mean()
        overall_actual = df["hit"].mean()
        total_preds = len(df)

        lines.append(f"  Total resolved predictions: {total_preds}")
        lines.append(f"  Overall expected hit rate:  {overall_expected*100:.1f}%")
        lines.append(f"  Overall actual hit rate:    {overall_actual*100:.1f}%")
        lines.append(f"  Overall drift:              {(overall_actual - overall_expected)*100:+.1f}%")
        lines.append("")

        # Calibration
        cal = self.calculate_calibration(df)
        lines.append("  --- CALIBRATION ---")
        lines.append(f"  Brier Score:  {cal['brier_score']:.4f}  (lower is better; 0.25 = random)")
        lines.append(f"  ECE:          {cal['ece']:.4f}  (lower is better)")
        lines.append("")

        lines.append("  Probability Buckets:")
        lines.append(f"  {'Bucket':<12} {'Count':>6} {'Expected':>10} {'Actual':>10} {'Diff':>8}")
        lines.append("  " + "-" * 48)
        for b in cal["buckets"]:
            diff = b["actual_hr"] - b["expected_hr"]
            lines.append(
                f"  {b['range']:<12} {b['count']:>6} "
                f"{b['expected_hr']*100:>9.1f}% {b['actual_hr']*100:>9.1f}% "
                f"{diff*100:>+7.1f}%"
            )
        lines.append("")

        # Per-prop breakdown
        lines.append("  --- PER-PROP PERFORMANCE ---")
        lines.append(f"  {'Prop Type':<18} {'Count':>6} {'Expected':>10} {'Actual':>10} {'Drift':>8}")
        lines.append("  " + "-" * 54)
        for prop_type, stats in sorted(cal["per_prop"].items()):
            flag = " ***" if stats["drift"] < -0.05 else ""
            lines.append(
                f"  {prop_type:<18} {stats['count']:>6} "
                f"{stats['expected_hr']*100:>9.1f}% {stats['actual_hr']*100:>9.1f}% "
                f"{stats['drift']*100:>+7.1f}%{flag}"
            )
        lines.append("")

        # Weekly trend
        lines.append("  --- WEEKLY TREND ---")
        df["week"] = df["date"].dt.isocalendar().week.astype(int)
        df["year"] = df["date"].dt.isocalendar().year.astype(int)
        weekly = df.groupby(["year", "week"]).agg(
            count=("hit", "count"),
            actual_hr=("hit", "mean"),
            expected_hr=("expected_prob", "mean"),
        ).reset_index()

        lines.append(f"  {'Week':>6} {'Count':>6} {'Expected':>10} {'Actual':>10} {'Drift':>8}")
        lines.append("  " + "-" * 42)
        for _, w in weekly.iterrows():
            drift = w["actual_hr"] - w["expected_hr"]
            lines.append(
                f"  {int(w['year'])}-W{int(w['week']):02d} {int(w['count']):>6} "
                f"{w['expected_hr']*100:>9.1f}% {w['actual_hr']*100:>9.1f}% "
                f"{drift*100:>+7.1f}%"
            )
        lines.append("")

        # Drift verdict
        drift_result = self.detect_drift(df)
        lines.append("  --- DRIFT ASSESSMENT ---")
        lines.append(f"  Recommendation: {drift_result['recommendation'].upper()}")
        lines.append(f"  Reason: {drift_result['reason']}")
        if drift_result["worst_prop"]:
            lines.append(
                f"  Worst prop: {drift_result['worst_prop']} "
                f"(drift: {drift_result['worst_prop_drift']*100:+.1f}%)"
            )
        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)


if __name__ == "__main__":
    monitor = DriftMonitor()
    report = monitor.generate_report(days_back=30)
    print(report)
