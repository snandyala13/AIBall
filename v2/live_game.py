#!/usr/bin/env python3
"""
v2/live_game.py - Live in-game prediction engine for aiBall.

Tracks live NBA box scores, detects game events (injuries, foul trouble,
blowouts, overtime), and updates prop projections based on actual game pace.

Integrates with the pre-game v2 engine: takes pre-game predictions and
continuously refines them as the game progresses.

Usage:
    from v2.live_game import LiveGameTracker

    tracker = LiveGameTracker()
    predictions = tracker.get_live_predictions(game_id, pre_game_preds, sigma_map)

Standalone:
    python -m v2.live_game          # Show live games and sample projections
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# PROP -> STAT KEY MAPPING
# =============================================================================

PROP_STAT_MAP = {
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


def get_stat_for_prop(stats: Dict, prop_type: str) -> float:
    """Sum the relevant box-score stat keys for a given prop type."""
    keys = PROP_STAT_MAP.get(prop_type, [])
    return sum(stats.get(k, 0) for k in keys)


# =============================================================================
# GAME STATUS PARSING
# =============================================================================

# BDL returns various status strings. Map them to structured data.
_STATUS_PERIOD_MAP = {
    "1st qtr": 1,
    "2nd qtr": 2,
    "halftime": 2,
    "3rd qtr": 3,
    "4th qtr": 4,
    "ot":      5,
    "ot1":     5,
    "ot2":     6,
    "ot3":     7,
}


def parse_game_status(raw_status: str, game_data: Dict = None) -> Dict:
    """
    Parse BDL game status into structured format.

    Returns:
        {
            "raw":            original status string,
            "period":         int (1-4 regular, 5+ OT),
            "is_halftime":    bool,
            "is_final":       bool,
            "is_in_progress": bool,
            "home_score":     int,
            "away_score":     int,
            "home_team":      str,
            "away_team":      str,
            "time_remaining": str (clock, if available),
        }
    """
    status_lower = (raw_status or "").strip().lower()
    game_data = game_data or {}

    home_score = game_data.get("home_team_score", 0) or 0
    away_score = game_data.get("visitor_team_score", 0) or 0
    home_team = game_data.get("home_team", {}).get("full_name", "")
    away_team = game_data.get("visitor_team", {}).get("full_name", "")
    game_time = game_data.get("time", "")

    is_final = status_lower in ("final", "final/ot", "final/2ot", "final/3ot")
    is_halftime = status_lower == "halftime"

    # Determine period
    period = _STATUS_PERIOD_MAP.get(status_lower, 0)
    if not period and "ot" in status_lower and not is_final:
        # Generic OT handling
        period = 5

    # BDL sometimes uses "In Progress" without a quarter label.
    # A game is NOT in progress if the status is a datetime string (scheduled start time)
    # or if both scores are 0 with no period.
    looks_like_datetime = "t" in status_lower and ("z" in status_lower or "+" in status_lower or "-" in status_lower[5:])
    is_in_progress = (
        not is_final
        and not looks_like_datetime
        and status_lower not in ("", "scheduled", "not started", "postponed", "cancelled")
        and not (home_score == 0 and away_score == 0 and period == 0)
    )

    # If the period field is on the game object, prefer it
    if game_data.get("period"):
        period = game_data["period"]

    return {
        "raw": raw_status,
        "period": period,
        "is_halftime": is_halftime,
        "is_final": is_final,
        "is_in_progress": is_in_progress,
        "home_score": home_score,
        "away_score": away_score,
        "home_team": home_team,
        "away_team": away_team,
        "time_remaining": game_time or "",
    }


# =============================================================================
# MINUTES PARSING
# =============================================================================

def parse_minutes(min_value) -> float:
    """
    Parse BDL minutes value into a float.

    BDL returns minutes as "MM:SS" strings or occasionally as ints/floats.
    """
    if min_value is None:
        return 0.0
    if isinstance(min_value, (int, float)):
        return float(min_value)
    min_str = str(min_value).strip()
    if not min_str:
        return 0.0
    try:
        if ":" in min_str:
            parts = min_str.split(":")
            return int(parts[0]) + int(parts[1]) / 60
        return float(min_str)
    except (ValueError, TypeError, IndexError):
        return 0.0


# =============================================================================
# LIVE GAME TRACKER
# =============================================================================

class LiveGameTracker:
    """Track live NBA games and update predictions based on real-time box scores."""

    # Minimum delay between API calls (seconds)
    RATE_LIMIT_DELAY = 0.5

    # Default projected minutes for a starter
    DEFAULT_STARTER_MINUTES = 36.0

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("BALLDONTLIE_API_KEY", "")
        self._game_states: Dict[int, Dict] = {}       # {game_id: game status dict}
        self._last_box_scores: Dict[int, Dict] = {}   # {game_id: {player_name: stats}}
        self._stale_timestamps: Dict[str, datetime] = {}  # {game_id|player: first_seen_stale}
        self._last_api_call: float = 0.0

    # -----------------------------------------------------------------
    # Internal: BDL API helpers
    # -----------------------------------------------------------------

    def _get_headers(self) -> Dict[str, str]:
        return {"Authorization": self.api_key}

    def _rate_limit(self):
        """Enforce minimum delay between API calls."""
        elapsed = time.time() - self._last_api_call
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self._last_api_call = time.time()

    def _bdl_get(self, path: str, params: Dict = None) -> Optional[Dict]:
        """Make an authenticated GET to BDL v1 API."""
        import requests

        self._rate_limit()
        url = f"https://api.balldontlie.io/v1/{path}"
        try:
            resp = requests.get(url, headers=self._get_headers(), params=params or {}, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.error("BDL API error (%s): %s", path, exc)
            return None

    # -----------------------------------------------------------------
    # Public: data fetching
    # -----------------------------------------------------------------

    def fetch_live_box_score(self, game_id: int) -> Dict[str, Dict]:
        """
        Fetch the current box score for a game.

        Calls BDL ``/v1/stats?game_ids[]={game_id}`` which returns per-player
        stat lines for the game.

        Returns:
            {player_name: {pts, reb, ast, fg3m, stl, blk, turnover, min,
                           fga, fgm, fta, ftm, pf, team, player_id}}
        """
        data = self._bdl_get("stats", {"game_ids[]": game_id, "per_page": 100})
        if not data or not data.get("data"):
            return {}

        box: Dict[str, Dict] = {}
        for row in data["data"]:
            player_info = row.get("player", {})
            first = player_info.get("first_name", "")
            last = player_info.get("last_name", "")
            name = f"{first} {last}".strip()
            if not name:
                continue

            team_info = row.get("team", {})
            team_name = team_info.get("full_name", team_info.get("abbreviation", ""))

            box[name] = {
                "pts": row.get("pts", 0) or 0,
                "reb": row.get("reb", 0) or 0,
                "ast": row.get("ast", 0) or 0,
                "fg3m": row.get("fg3m", 0) or 0,
                "stl": row.get("stl", 0) or 0,
                "blk": row.get("blk", 0) or 0,
                "turnover": row.get("turnover", 0) or 0,
                "min": parse_minutes(row.get("min")),
                "fga": row.get("fga", 0) or 0,
                "fgm": row.get("fgm", 0) or 0,
                "fta": row.get("fta", 0) or 0,
                "ftm": row.get("ftm", 0) or 0,
                "pf": row.get("pf", 0) or 0,
                "team": team_name,
                "player_id": player_info.get("id"),
            }

        # Cache for event detection (compare against previous snapshot)
        self._last_box_scores[game_id] = box
        return box

    def get_game_status(self, game_id: int) -> Dict:
        """
        Fetch and parse the current status of a game.

        Calls BDL ``/v1/games/{game_id}``.

        Returns structured dict (see ``parse_game_status``).
        """
        data = self._bdl_get(f"games/{game_id}")
        if not data:
            return parse_game_status("", {})

        game_data = data.get("data", data)  # BDL may wrap in {data: ...}
        raw_status = game_data.get("status", "")
        return parse_game_status(raw_status, game_data)

    # -----------------------------------------------------------------
    # Event detection
    # -----------------------------------------------------------------

    def detect_events(self, game_id: int) -> List[Dict]:
        """
        Compare the current box score to the previous snapshot and detect
        noteworthy in-game events.

        Event types:
            - possible_injury: player minutes stale for 8+ real minutes
            - foul_trouble:    5+ personal fouls
            - blowout:         large margin in late quarters
            - overtime_likely:  within 3 points with <2 min in Q4
            - hot_streak:      current pace 2+ sigma above pre-game
            - cold_streak:     current pace 2+ sigma below pre-game
        """
        current_box = self._last_box_scores.get(game_id, {})
        game_status = self._game_states.get(game_id) or self.get_game_status(game_id)
        self._game_states[game_id] = game_status

        events: List[Dict] = []

        period = game_status.get("period", 0)
        home_score = game_status.get("home_score", 0)
        away_score = game_status.get("away_score", 0)
        score_diff = abs(home_score - away_score)
        home_team = game_status.get("home_team", "")
        away_team = game_status.get("away_team", "")

        now = datetime.now()

        # --- Per-player events ---
        for player, stats in current_box.items():
            mins = stats.get("min", 0)
            pf = stats.get("pf", 0)
            stale_key = f"{game_id}|{player}"

            # 1. Possible injury / DNP: minutes haven't increased
            #    Only flag players who had 10+ minutes (not bench guys between
            #    rotations) AND haven't logged minutes for 12+ real minutes.
            if mins >= 10:
                if stale_key in self._stale_timestamps:
                    first_stale = self._stale_timestamps[stale_key]
                    stale_duration = (now - first_stale).total_seconds() / 60
                    if stale_duration >= 12:
                        events.append({
                            "type": "possible_injury",
                            "player": player,
                            "last_active_min": round(mins, 1),
                            "stale_real_minutes": round(stale_duration, 1),
                        })
                else:
                    self._stale_timestamps[stale_key] = now

            # 2. Foul trouble (5+ PF)
            if pf >= 5:
                events.append({
                    "type": "foul_trouble",
                    "player": player,
                    "fouls": pf,
                })

        # Clear stale markers for players whose minutes *did* increase since
        # the last poll (done after the loop so we don't interfere mid-iteration).
        prev_box = self._last_box_scores.get(f"_prev_{game_id}", {})
        for player, stats in current_box.items():
            stale_key = f"{game_id}|{player}"
            prev_mins = prev_box.get(player, {}).get("min", 0)
            curr_mins = stats.get("min", 0)
            if curr_mins > prev_mins:
                self._stale_timestamps.pop(stale_key, None)

        # Save current as previous for next comparison
        self._last_box_scores[f"_prev_{game_id}"] = {
            name: dict(s) for name, s in current_box.items()
        }

        # --- Game-level events ---

        # 3. Blowout detection
        if (period >= 4 and score_diff > 25) or (period >= 3 and score_diff > 30):
            leading = home_team if home_score > away_score else away_team
            events.append({
                "type": "blowout",
                "leading_team": leading,
                "margin": score_diff,
                "quarter": period,
            })

        # 4. Overtime likely: within 3 points, <2 min in Q4
        time_remaining_str = game_status.get("time_remaining", "")
        time_remaining_mins = _parse_clock_to_minutes(time_remaining_str)
        if period == 4 and time_remaining_mins is not None and time_remaining_mins < 2 and score_diff <= 3:
            events.append({
                "type": "overtime_likely",
                "margin": score_diff,
            })

        return events

    # -----------------------------------------------------------------
    # Projection: remaining stat output
    # -----------------------------------------------------------------

    def project_remaining(
        self,
        pre_game_prediction: float,
        current_stat: float,
        minutes_played: float,
        total_projected_minutes: float,
        prop_type: str,
        game_events: Optional[List[Dict]] = None,
        player_name: Optional[str] = None,
    ) -> float:
        """
        Blend pre-game model prediction with live pace to project the final stat.

        As more of the game is played the live pace receives increasing weight:
            - 0 min played  -> 100 % pre-game
            - ~24 min (half) -> ~60 % live
            - ~36 min (end)  -> ~85 % live (capped at 0.85)
        """
        if minutes_played < 1:
            return pre_game_prediction  # too early -- trust pre-game

        minutes_remaining = max(total_projected_minutes - minutes_played, 0)

        # Per-minute pace
        current_pace = current_stat / minutes_played
        pre_game_pace = pre_game_prediction / total_projected_minutes if total_projected_minutes > 0 else 0

        # Blend weight ramps linearly, capped at 0.85
        blend_weight = min(minutes_played / 30.0, 0.85)
        blended_pace = blend_weight * current_pace + (1 - blend_weight) * pre_game_pace

        projected_final = current_stat + blended_pace * minutes_remaining

        # Apply event-based adjustments
        if game_events:
            for event in game_events:
                etype = event.get("type")

                if etype == "blowout":
                    # Starters will be pulled; most remaining time is garbage
                    minutes_remaining *= 0.3
                    projected_final = current_stat + blended_pace * minutes_remaining

                elif etype == "foul_trouble":
                    # Only adjust if this event is about the player we are projecting
                    event_player = event.get("player", "")
                    if player_name and event_player == player_name:
                        minutes_remaining *= 0.65
                        projected_final = current_stat + blended_pace * minutes_remaining

                elif etype == "overtime_likely":
                    minutes_remaining += 5  # one OT period
                    projected_final = current_stat + blended_pace * minutes_remaining

        # Floor: can never go below what they already have
        return max(projected_final, current_stat)

    # -----------------------------------------------------------------
    # Main integration point
    # -----------------------------------------------------------------

    def get_live_predictions(
        self,
        game_id: int,
        pre_game_predictions: Dict[str, Dict],
        pre_game_sigma_map: Dict[str, float],
    ) -> List[Dict]:
        """
        Update predictions for an in-progress game.

        Args:
            game_id:              BDL game ID.
            pre_game_predictions: ``{player|prop_type: {predicted_value, line, p_over, ...}}``.
            pre_game_sigma_map:   ``{player|prop_type: sigma}`` from q10/q90.

        Returns:
            List of dicts, one per prop, with live-updated projections, probabilities,
            and event flags.
        """
        from scipy.stats import norm

        box_score = self.fetch_live_box_score(game_id)
        game_status = self.get_game_status(game_id)
        events = self.detect_events(game_id)

        results: List[Dict] = []

        for cache_key, pre_game in pre_game_predictions.items():
            parts = cache_key.split("|", 1)
            if len(parts) != 2:
                logger.warning("Skipping malformed cache key: %s", cache_key)
                continue
            player, prop_type = parts

            # Current box-score stats for this player
            player_stats = box_score.get(player, {})
            current_stat = get_stat_for_prop(player_stats, prop_type)
            minutes_played = player_stats.get("min", 0.0)

            total_projected_minutes = self.DEFAULT_STARTER_MINUTES

            # Collect events relevant to this player
            player_events = [
                e for e in events
                if e.get("player") == player or e["type"] in ("blowout", "overtime_likely")
            ]

            # Project final stat
            projected_final = self.project_remaining(
                pre_game_prediction=pre_game.get("predicted_value", 0),
                current_stat=current_stat,
                minutes_played=minutes_played,
                total_projected_minutes=total_projected_minutes,
                prop_type=prop_type,
                game_events=player_events,
                player_name=player,
            )

            line = pre_game.get("line", 0)
            minutes_remaining = max(total_projected_minutes - minutes_played, 0)

            # Uncertainty shrinks as the game progresses (sqrt decay)
            pre_sigma = pre_game_sigma_map.get(cache_key, 3.0)
            remaining_frac = max(minutes_remaining / total_projected_minutes, 0.05)
            live_sigma = pre_sigma * (remaining_frac ** 0.5)

            # P(over) using the live projection
            stat_remaining = projected_final - current_stat
            needed_remaining = max(line - current_stat, 0)

            if minutes_remaining > 0:
                z = (needed_remaining - stat_remaining) / max(live_sigma, 0.1)
                p_over = 1 - norm.cdf(z)
            else:
                # Game is over -- binary outcome
                p_over = 1.0 if current_stat > line else 0.0

            p_under = 1.0 - p_over

            # Progress toward the line
            progress_pct = min(current_stat / max(line, 0.1), 2.0)

            results.append({
                "player": player,
                "prop_type": prop_type,
                "line": line,
                "current_stat": current_stat,
                "minutes_played": round(minutes_played, 1),
                "minutes_remaining": round(minutes_remaining, 1),
                "projected_final": round(projected_final, 1),
                "pre_game_projection": round(pre_game.get("predicted_value", 0), 1),
                "live_p_over": round(p_over, 4),
                "live_p_under": round(p_under, 4),
                "pre_game_p_over": pre_game.get("p_over", 0.5),
                "live_sigma": round(live_sigma, 2),
                "progress_pct": round(progress_pct * 100, 1),
                "on_pace": current_stat >= (line * minutes_played / total_projected_minutes) if minutes_played > 0 else None,
                "events": player_events,
                "game_status": game_status,
                "source": "v2_live_game",
            })

        return results


# =============================================================================
# HELPERS
# =============================================================================

def _parse_clock_to_minutes(clock_str: str) -> Optional[float]:
    """
    Parse a game-clock string like '4:32' or '11:00' into decimal minutes.
    Returns None if the string is empty or unparseable.
    """
    if not clock_str or not clock_str.strip():
        return None
    try:
        parts = clock_str.strip().split(":")
        if len(parts) == 2:
            return int(parts[0]) + int(parts[1]) / 60
        return float(clock_str)
    except (ValueError, TypeError, IndexError):
        return None


# =============================================================================
# STANDALONE ENTRY POINT
# =============================================================================

def _format_event(event: Dict) -> str:
    """Human-readable one-liner for a detected event."""
    etype = event.get("type", "")
    if etype == "possible_injury":
        return f"  INJURY? {event['player']} -- no minutes added for {event.get('stale_real_minutes', '?')} min (last at {event.get('last_active_min')} min)"
    if etype == "foul_trouble":
        return f"  FOUL TROUBLE: {event['player']} has {event['fouls']} PF"
    if etype == "blowout":
        return f"  BLOWOUT: {event['leading_team']} up {event['margin']} in Q{event['quarter']}"
    if etype == "overtime_likely":
        return f"  OT LIKELY: margin is {event['margin']}"
    return f"  {etype}: {event}"


def main():
    """Fetch today's in-progress games, print box scores and sample projections."""
    import requests

    api_key = os.environ.get("BALLDONTLIE_API_KEY", "")
    if not api_key:
        print("Set BALLDONTLIE_API_KEY to use the live tracker.")
        return

    tracker = LiveGameTracker(api_key=api_key)

    # Fetch today's games
    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    tracker._rate_limit()
    resp = requests.get(
        "https://api.balldontlie.io/v1/games",
        headers={"Authorization": api_key},
        params={"start_date": today, "end_date": tomorrow},
        timeout=30,
    )
    if resp.status_code != 200:
        print(f"Failed to fetch games: {resp.status_code}")
        return

    games = resp.json().get("data", [])
    if not games:
        print("No games found for today.")
        return

    in_progress = [g for g in games if g.get("status", "") not in ("", "Final", "Scheduled", "Not Started")]
    target_games = in_progress if in_progress else games  # fall back to all games

    print(f"\n{'=' * 60}")
    print(f"  aiBall Live Tracker -- {today}")
    print(f"  {len(games)} games today, {len(in_progress)} in progress")
    print(f"{'=' * 60}\n")

    for game in target_games:
        game_id = game.get("id")
        home = game.get("home_team", {}).get("full_name", "?")
        away = game.get("visitor_team", {}).get("full_name", "?")
        status_raw = game.get("status", "")
        h_score = game.get("home_team_score", 0) or 0
        a_score = game.get("visitor_team_score", 0) or 0

        print(f"--- {away} @ {home} ---")
        print(f"    Status: {status_raw}   Score: {a_score} - {h_score}")

        # Fetch box score
        box = tracker.fetch_live_box_score(game_id)
        if not box:
            print("    (no box score data yet)\n")
            continue

        # Detect events
        events = tracker.detect_events(game_id)
        if events:
            print("    Events:")
            for ev in events:
                print(_format_event(ev))

        # Print top stat lines (sorted by points descending)
        sorted_players = sorted(box.items(), key=lambda kv: kv[1].get("pts", 0), reverse=True)
        print(f"    {'Player':<25} MIN   PTS  REB  AST  3PM  STL  BLK  TO   PF")
        for name, s in sorted_players[:10]:
            mins = s.get("min", 0)
            if mins == 0:
                continue
            print(
                f"    {name:<25} {mins:4.1f}  "
                f"{s.get('pts',0):3d}  {s.get('reb',0):3d}  {s.get('ast',0):3d}  "
                f"{s.get('fg3m',0):3d}  {s.get('stl',0):3d}  {s.get('blk',0):3d}  "
                f"{s.get('turnover',0):3d}  {s.get('pf',0):3d}"
            )

        # Demo projection for top scorer
        if sorted_players:
            top_name, top_stats = sorted_players[0]
            demo_line = top_stats.get("pts", 0) + 5  # fake line for demo
            pre_game_pred = demo_line + 2  # pretend pre-game predicted slightly over

            proj = tracker.project_remaining(
                pre_game_prediction=pre_game_pred,
                current_stat=top_stats.get("pts", 0),
                minutes_played=top_stats.get("min", 0),
                total_projected_minutes=36.0,
                prop_type="points",
                game_events=events,
                player_name=top_name,
            )
            print(f"\n    Sample projection for {top_name}:")
            print(f"      Current: {top_stats.get('pts', 0)} pts in {top_stats.get('min', 0):.1f} min")
            print(f"      Pre-game pred: {pre_game_pred:.1f}  |  Live projection: {proj:.1f}")

        print()


if __name__ == "__main__":
    main()
