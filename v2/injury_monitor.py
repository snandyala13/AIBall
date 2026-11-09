"""
injury_monitor.py - Real-Time Injury Cascade Monitor

Detects injury report changes and immediately flags mispriced teammate props
using pre-computed cascade tables.

Architecture:
    1. Poll BDL /v1/player_injuries on a configurable interval
    2. Compare current report to last-known state
    3. On status change -> look up cascade table -> emit alerts
    4. Web layer consumes alerts via InjuryCascadeMonitor.get_alerts()

Usage (standalone):
    python v2/injury_monitor.py              # One-shot scan
    python v2/injury_monitor.py --watch 60   # Poll every 60 seconds
"""

import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from v2.injury_cascade import (
    InjuryCascadeLookup,
    CASCADE_TABLE_PATH,
    FULL_TO_ABBREV,
    TEAM_ABBREVS,
    normalize_team_name,
)

# -------------------------------------------------------------------------
# Injury Status Classification
# -------------------------------------------------------------------------

# Status levels from most to least severe
STATUS_SEVERITY = {
    "out": 4,
    "doubtful": 3,
    "questionable": 2,
    "probable": 1,
    "day-to-day": 1,
    "available": 0,
    "active": 0,
    "": 0,
}

# Status changes that trigger cascade alerts (escalations only)
ALERT_TRIGGERS = {
    # (old_status, new_status) -> should alert
    # Alert when a player moves to OUT or DOUBTFUL
}


def classify_status(raw_status: str) -> str:
    """Normalize an injury status string to a canonical form."""
    s = raw_status.strip().lower()
    if "out" in s:
        return "out"
    if "doubtful" in s:
        return "doubtful"
    if "questionable" in s or "gtd" in s or "game time" in s:
        return "questionable"
    if "probable" in s or "likely" in s:
        return "probable"
    if "day" in s:
        return "day-to-day"
    if "available" in s or "active" in s:
        return "available"
    return s


def is_escalation(old_status: str, new_status: str) -> bool:
    """Return True if the status change represents an injury escalation."""
    old_sev = STATUS_SEVERITY.get(classify_status(old_status), 0)
    new_sev = STATUS_SEVERITY.get(classify_status(new_status), 0)
    return new_sev > old_sev


def should_trigger_cascade(old_status: str, new_status: str) -> bool:
    """
    Return True if this status change should trigger cascade alerts.

    We trigger on:
    - Any change TO "out" (the definitive signal)
    - Any change TO "doubtful" (likely to sit)
    - New injury (previously not on report -> now out/doubtful)
    """
    new_class = classify_status(new_status)
    return new_class in ("out", "doubtful")


# -------------------------------------------------------------------------
# Injury State Tracker
# -------------------------------------------------------------------------

class InjuryState:
    """
    Tracks the last-known injury report and detects changes.

    State format: {player_name: {"status": str, "team": str, "team_id": int,
                                  "comment": str, "updated_at": str}}
    """

    def __init__(self):
        self._state = {}  # type: Dict[str, dict]
        self._last_poll = None  # type: Optional[datetime]

    @property
    def last_poll(self) -> Optional[datetime]:
        return self._last_poll

    def update(self, injuries: List[dict]) -> List[dict]:
        """
        Update state with fresh injury data and return list of changes.

        Args:
            injuries: Raw injury data from BDL API. Each entry has:
                      {player: {first_name, last_name, id, ...},
                       status: "Out", comment: "Knee", ...}

        Returns:
            List of change dicts:
            [{"player": str, "team": str, "old_status": str, "new_status": str,
              "comment": str, "is_escalation": bool, "timestamp": str}]
        """
        now = datetime.now()
        self._last_poll = now
        timestamp = now.isoformat()

        changes = []
        current_names = set()

        for inj in injuries:
            player_data = inj.get("player", {})
            first = player_data.get("first_name", "")
            last = player_data.get("last_name", "")
            name = f"{first} {last}".strip()
            if not name:
                continue

            team_data = player_data.get("team", {})
            team_name = team_data.get("full_name", "")
            team_id = team_data.get("id")
            status = inj.get("status", "")
            comment = inj.get("comment", "")

            current_names.add(name)

            old = self._state.get(name)
            if old is None:
                # New injury (player wasn't on report before)
                changes.append({
                    "player": name,
                    "team": team_name,
                    "team_id": team_id,
                    "old_status": "available",
                    "new_status": status,
                    "comment": comment,
                    "is_escalation": True,
                    "is_new": True,
                    "timestamp": timestamp,
                })
            elif classify_status(old["status"]) != classify_status(status):
                # Status changed
                changes.append({
                    "player": name,
                    "team": team_name,
                    "team_id": team_id,
                    "old_status": old["status"],
                    "new_status": status,
                    "comment": comment,
                    "is_escalation": is_escalation(old["status"], status),
                    "is_new": False,
                    "timestamp": timestamp,
                })

            # Update state
            self._state[name] = {
                "status": status,
                "team": team_name,
                "team_id": team_id,
                "comment": comment,
                "updated_at": timestamp,
            }

        # Detect players removed from injury report (recovered)
        for name in list(self._state.keys()):
            if name not in current_names:
                old = self._state[name]
                changes.append({
                    "player": name,
                    "team": old.get("team", ""),
                    "team_id": old.get("team_id"),
                    "old_status": old["status"],
                    "new_status": "available",
                    "comment": "Removed from injury report",
                    "is_escalation": False,
                    "is_new": False,
                    "timestamp": timestamp,
                })
                del self._state[name]

        return changes

    def get_current_outs(self) -> List[dict]:
        """Return all players currently marked as OUT or DOUBTFUL."""
        outs = []
        for name, data in self._state.items():
            status = classify_status(data["status"])
            if status in ("out", "doubtful"):
                outs.append({
                    "player": name,
                    "team": data.get("team", ""),
                    "status": data["status"],
                    "comment": data.get("comment", ""),
                })
        return outs


# -------------------------------------------------------------------------
# Cascade Alert Generator
# -------------------------------------------------------------------------

class CascadeAlert:
    """Represents an actionable injury cascade alert."""

    def __init__(
        self,
        absent_player: str,
        team: str,
        status_change: str,
        comment: str,
        cascade_impacts: List[dict],
        recommended_bets: List[dict],
        timestamp: str,
    ):
        self.absent_player = absent_player
        self.team = team
        self.status_change = status_change
        self.comment = comment
        self.cascade_impacts = cascade_impacts
        self.recommended_bets = recommended_bets
        self.timestamp = timestamp
        self.created_at = datetime.now()

    def to_dict(self) -> dict:
        team_abbrev = FULL_TO_ABBREV.get(self.team, "???")
        return {
            "absent_player": self.absent_player,
            "team": self.team,
            "team_abbrev": team_abbrev,
            "status_change": self.status_change,
            "comment": self.comment,
            "timestamp": self.timestamp,
            "age_minutes": round(
                (datetime.now() - self.created_at).total_seconds() / 60, 1
            ),
            "cascade_impacts": self.cascade_impacts,
            "recommended_bets": self.recommended_bets,
            "impact_count": len(self.cascade_impacts),
            "bet_count": len(self.recommended_bets),
        }


# -------------------------------------------------------------------------
# Main Monitor
# -------------------------------------------------------------------------

class InjuryCascadeMonitor:
    """
    Real-time injury cascade monitor.

    Integrates with:
    - BDL API injury endpoint (via DataPipeline or direct)
    - Pre-computed cascade tables
    - Web layer for alert delivery
    """

    def __init__(
        self,
        cascade_path: str = CASCADE_TABLE_PATH,
        min_boost_pct: float = 5.0,
        min_confidence: str = "low",
    ):
        self.lookup = InjuryCascadeLookup(cascade_path)
        self.state = InjuryState()
        self.min_boost_pct = min_boost_pct
        self.min_confidence = min_confidence

        # Active alerts (recent cascade alerts, expire after 4 hours)
        self._alerts = []  # type: List[CascadeAlert]
        self._alert_ttl_hours = 4

        # Pre-game scan results
        self._pregame_scan = {}  # type: Dict[str, dict]

        if not self.lookup.is_loaded:
            logger.warning(
                "Cascade tables not loaded. Run: python v2/injury_cascade.py"
            )

    @property
    def is_ready(self) -> bool:
        return self.lookup.is_loaded

    def _prune_old_alerts(self):
        """Remove alerts older than TTL."""
        now = datetime.now()
        self._alerts = [
            a for a in self._alerts
            if (now - a.created_at).total_seconds() < self._alert_ttl_hours * 3600
        ]

    def process_injury_update(self, injuries: List[dict]) -> List[CascadeAlert]:
        """
        Process a fresh injury report and return any new cascade alerts.

        Args:
            injuries: Raw injury data from BDL API

        Returns:
            List of new CascadeAlert objects
        """
        changes = self.state.update(injuries)
        new_alerts = []

        for change in changes:
            if not should_trigger_cascade(
                change["old_status"], change["new_status"]
            ):
                continue

            player = change["player"]
            team = change["team"]
            status_str = f"{change['old_status']} -> {change['new_status']}"

            # Look up cascade impacts
            impacts = self.lookup.get_impacts(
                player, min_confidence=self.min_confidence
            )

            if not impacts:
                logger.info(
                    "No cascade data for %s (%s)", player, team
                )
                continue

            # Filter to meaningful impacts
            meaningful = [
                i for i in impacts
                if abs(i["boost_pct"]) >= self.min_boost_pct
            ]

            # Build cascade impact list for the alert
            cascade_impacts = []
            for imp in meaningful[:15]:  # Top 15 impacts
                cascade_impacts.append({
                    "player": imp["player"],
                    "prop": imp["prop"],
                    "boost_pct": imp["boost_pct"],
                    "avg_with": imp["avg_with"],
                    "avg_without": imp["avg_without"],
                    "confidence": imp["confidence"],
                    "sample": imp["sample_without"],
                })

            # Build bet recommendations
            bets = self.lookup.suggest_bets(
                player,
                min_boost_pct=self.min_boost_pct,
                min_confidence=self.min_confidence,
            )

            recommended = []
            for bet in bets[:8]:  # Top 8 recommendations
                recommended.append({
                    "player": bet["player"],
                    "prop": bet["prop"],
                    "side": bet["side"],
                    "boost_pct": bet["boost_pct"],
                    "confidence": bet["confidence"],
                    "reason": bet["reason"],
                })

            alert = CascadeAlert(
                absent_player=player,
                team=team,
                status_change=status_str,
                comment=change.get("comment", ""),
                cascade_impacts=cascade_impacts,
                recommended_bets=recommended,
                timestamp=change["timestamp"],
            )

            new_alerts.append(alert)
            self._alerts.append(alert)

            logger.info(
                "CASCADE ALERT: %s (%s) %s -> %d impacts, %d bets",
                player, team, status_str,
                len(cascade_impacts), len(recommended),
            )

        self._prune_old_alerts()
        return new_alerts

    def get_alerts(self, max_age_minutes: int = 240) -> List[dict]:
        """
        Get active cascade alerts as dicts (for web API).

        Args:
            max_age_minutes: Only return alerts younger than this
        """
        self._prune_old_alerts()
        now = datetime.now()
        result = []
        for alert in self._alerts:
            age_mins = (now - alert.created_at).total_seconds() / 60
            if age_mins <= max_age_minutes:
                result.append(alert.to_dict())
        return result

    def scan_pregame(
        self,
        injuries: Optional[List[dict]] = None,
        pipeline=None,
    ) -> List[dict]:
        """
        Pre-game cascade scan: check current injury report and flag
        any OUT/DOUBTFUL players with cascade impacts.

        This is used BEFORE games start to identify mispriced props.

        Args:
            injuries: Pre-fetched injury list, or None to use pipeline
            pipeline: DataPipeline instance for fetching injuries

        Returns:
            List of pre-game cascade alerts
        """
        if injuries is None and pipeline is not None:
            # Fetch all injuries
            injuries = []
            for team_name, team_id in _get_all_team_ids().items():
                try:
                    team_injuries = pipeline.client.get_injuries(team_id)
                    injuries.extend(team_injuries)
                except Exception as e:
                    logger.debug("Failed to fetch injuries for %s: %s", team_name, e)

        if not injuries:
            return []

        # Initialize state if first scan
        if self.state.last_poll is None:
            self.state.update(injuries)

        # Find all OUT/DOUBTFUL players
        outs = self.state.get_current_outs()
        if not outs:
            # Even if state is empty, check the raw data
            for inj in injuries:
                status = classify_status(inj.get("status", ""))
                if status in ("out", "doubtful"):
                    player_data = inj.get("player", {})
                    name = f"{player_data.get('first_name', '')} {player_data.get('last_name', '')}".strip()
                    team_data = player_data.get("team", {})
                    outs.append({
                        "player": name,
                        "team": team_data.get("full_name", ""),
                        "status": inj.get("status", ""),
                        "comment": inj.get("comment", ""),
                    })

        results = []
        for out in outs:
            player = out["player"]
            impacts = self.lookup.get_impacts(
                player, min_confidence=self.min_confidence
            )
            meaningful = [
                i for i in impacts
                if abs(i["boost_pct"]) >= self.min_boost_pct
            ]

            if not meaningful:
                continue

            bets = self.lookup.suggest_bets(
                player,
                min_boost_pct=self.min_boost_pct,
                min_confidence=self.min_confidence,
            )

            results.append({
                "absent_player": player,
                "team": out.get("team", ""),
                "team_abbrev": FULL_TO_ABBREV.get(out.get("team", ""), "???"),
                "status": out.get("status", "Out"),
                "comment": out.get("comment", ""),
                "cascade_impacts": [
                    {
                        "player": i["player"],
                        "prop": i["prop"],
                        "boost_pct": i["boost_pct"],
                        "avg_with": i["avg_with"],
                        "avg_without": i["avg_without"],
                        "confidence": i["confidence"],
                        "sample": i["sample_without"],
                    }
                    for i in meaningful[:10]
                ],
                "recommended_bets": [
                    {
                        "player": b["player"],
                        "prop": b["prop"],
                        "side": b["side"],
                        "boost_pct": b["boost_pct"],
                        "confidence": b["confidence"],
                        "reason": b["reason"],
                    }
                    for b in bets[:6]
                ],
            })

        return results

    def lookup_player(self, player_name: str) -> dict:
        """
        Look up cascade impacts if a specific player were to sit out.

        Used by the /api/injury-cascade/lookup endpoint.
        """
        impacts = self.lookup.get_impacts(player_name, min_confidence="low")
        bets = self.lookup.suggest_bets(
            player_name, min_boost_pct=3.0, min_confidence="low"
        )
        team = self.lookup.get_team_for_player(player_name)

        return {
            "player": player_name,
            "team": team or "Unknown",
            "team_abbrev": FULL_TO_ABBREV.get(team, "???") if team else "???",
            "cascade_impacts": [
                {
                    "player": i["player"],
                    "prop": i["prop"],
                    "boost_pct": i["boost_pct"],
                    "avg_with": i["avg_with"],
                    "avg_without": i["avg_without"],
                    "confidence": i["confidence"],
                    "sample": i["sample_without"],
                    "minutes_change_pct": i.get("minutes_change_pct", 0),
                }
                for i in impacts[:20]
            ],
            "recommended_bets": [
                {
                    "player": b["player"],
                    "prop": b["prop"],
                    "side": b["side"],
                    "boost_pct": b["boost_pct"],
                    "confidence": b["confidence"],
                    "reason": b["reason"],
                }
                for b in bets[:10]
            ],
            "impact_count": len(impacts),
        }


def _get_all_team_ids() -> Dict[str, int]:
    """Get team name -> BDL team ID mapping."""
    return {
        "Atlanta Hawks": 1, "Boston Celtics": 2, "Brooklyn Nets": 3,
        "Charlotte Hornets": 4, "Chicago Bulls": 5, "Cleveland Cavaliers": 6,
        "Dallas Mavericks": 7, "Denver Nuggets": 8, "Detroit Pistons": 9,
        "Golden State Warriors": 10, "Houston Rockets": 11, "Indiana Pacers": 12,
        "LA Clippers": 13, "Los Angeles Lakers": 14, "Memphis Grizzlies": 15,
        "Miami Heat": 16, "Milwaukee Bucks": 17, "Minnesota Timberwolves": 18,
        "New Orleans Pelicans": 19, "New York Knicks": 20,
        "Oklahoma City Thunder": 21, "Orlando Magic": 22,
        "Philadelphia 76ers": 23, "Phoenix Suns": 24,
        "Portland Trail Blazers": 25, "Sacramento Kings": 26,
        "San Antonio Spurs": 27, "Toronto Raptors": 28,
        "Utah Jazz": 29, "Washington Wizards": 30,
    }


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Injury cascade monitor")
    parser.add_argument(
        "--watch", type=int, default=0,
        help="Poll interval in seconds (0 = one-shot scan)",
    )
    parser.add_argument(
        "--lookup", type=str, default=None,
        help="Look up cascade impacts for a specific player",
    )
    args = parser.parse_args()

    monitor = InjuryCascadeMonitor()

    if not monitor.is_ready:
        print("ERROR: Cascade tables not found. Build them first:")
        print("  python v2/injury_cascade.py")
        sys.exit(1)

    if args.lookup:
        result = monitor.lookup_player(args.lookup)
        print(f"\n=== Cascade Lookup: {result['player']} ({result['team']}) ===")
        print(f"Total impacts: {result['impact_count']}")
        for imp in result["cascade_impacts"][:10]:
            sign = "+" if imp["boost_pct"] > 0 else ""
            print(
                f"  {imp['player']:25s} {imp['prop']:12s} "
                f"{sign}{imp['boost_pct']:.1f}% "
                f"({imp['avg_with']:.1f} -> {imp['avg_without']:.1f}) "
                f"[{imp['sample']}g, {imp['confidence']}]"
            )
        if result["recommended_bets"]:
            print(f"\nRecommended bets:")
            for bet in result["recommended_bets"][:5]:
                print(f"  {bet['side']:5s} {bet['player']:25s} {bet['prop']:12s} - {bet['reason']}")
        return

    # Need API key for live monitoring
    api_key = os.environ.get("BALLDONTLIE_API_KEY")
    if not api_key:
        print("ERROR: BALLDONTLIE_API_KEY not set. Required for injury monitoring.")
        print("Set it in .env or environment.")
        sys.exit(1)

    from data_pipeline import BallDontLieClient
    client = BallDontLieClient(api_key)

    print("Fetching injury reports...")
    all_injuries = []
    for team_name, team_id in _get_all_team_ids().items():
        try:
            injuries = client.get_injuries(team_id)
            all_injuries.extend(injuries)
        except Exception as e:
            logger.debug("Failed for %s: %s", team_name, e)

    print(f"Found {len(all_injuries)} injury entries")

    # Initial scan
    results = monitor.scan_pregame(injuries=all_injuries)
    if results:
        print(f"\n=== Pre-Game Cascade Scan: {len(results)} active cascades ===")
        for r in results:
            print(f"\n  {r['absent_player']} ({r['team_abbrev']}) - {r['status']}")
            if r.get("comment"):
                print(f"    Injury: {r['comment']}")
            for imp in r["cascade_impacts"][:5]:
                sign = "+" if imp["boost_pct"] > 0 else ""
                print(
                    f"    {imp['player']:25s} {imp['prop']:10s} "
                    f"{sign}{imp['boost_pct']:.1f}% [{imp['confidence']}]"
                )
            if r["recommended_bets"]:
                print(f"    Bets:")
                for bet in r["recommended_bets"][:3]:
                    print(f"      {bet['side']:5s} {bet['player']} {bet['prop']} - {bet['reason']}")
    else:
        print("No active cascades found.")

    if args.watch > 0:
        print(f"\nWatching for injury changes every {args.watch}s... (Ctrl+C to stop)")
        while True:
            time.sleep(args.watch)
            all_injuries = []
            for team_name, team_id in _get_all_team_ids().items():
                try:
                    injuries = client.get_injuries(team_id)
                    all_injuries.extend(injuries)
                except Exception:
                    pass
            alerts = monitor.process_injury_update(all_injuries)
            if alerts:
                for alert in alerts:
                    d = alert.to_dict()
                    print(f"\n{'='*60}")
                    print(f"CASCADE ALERT: {d['absent_player']} ({d['team_abbrev']}) "
                          f"{d['status_change']}")
                    for imp in d["cascade_impacts"][:5]:
                        sign = "+" if imp["boost_pct"] > 0 else ""
                        print(f"  {imp['player']:25s} {imp['prop']:10s} "
                              f"{sign}{imp['boost_pct']:.1f}% [{imp['confidence']}]")


if __name__ == "__main__":
    main()
