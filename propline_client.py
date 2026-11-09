"""
propline_client.py - Fast PropLine API Client for NBA Player Props

PropLine fetches ALL props for ALL games in ONE bulk call, replacing
the per-game approach used by Balldontlie (3.3s/game) and Odds API.

Endpoints:
  GET /v1/sports/basketball_nba/events                    - Today's games
  GET /v1/sports/basketball_nba/odds?markets=X&bookmakers=draftkings  - BULK props
  GET /v1/sports/basketball_nba/events/{id}/odds?markets=X  - Single game props
  GET /v1/sports/basketball_nba/events/{id}/results       - Graded results

Auth: apiKey query param or X-API-Key header.
"""

import os
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


class PropLineClient:
    """Fast PropLine API client for player props and graded results."""

    BASE_URL = "https://api.prop-line.com/v1"
    SPORT = "basketball_nba"

    # Market key mapping to internal prop types (mirrors OddsAPIClient.MARKET_MAP)
    MARKET_MAP = {
        "player_points": "points",
        "player_rebounds": "rebounds",
        "player_assists": "assists",
        "player_threes": "threes",
        "player_steals": "steals",
        "player_blocks": "blocks",
        "player_points_rebounds_assists": "pts_rebs_asts",
    }

    ALL_MARKETS = ",".join(MARKET_MAP.keys())

    def __init__(self, api_key=None):
        # type: (Optional[str]) -> None
        self.api_key = api_key or os.environ.get("PROPLINE_API_KEY")
        self._cache = {}  # type: Dict[str, Tuple[float, object]]
        self._cache_ttl = 300  # 5 min cache

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_request(self, path, params=None, timeout=30):
        # type: (str, Optional[Dict], int) -> Optional[object]
        """Make authenticated GET request to PropLine API."""
        if not self.api_key:
            logger.warning("PropLine API key not set")
            return None

        url = "{}/sports/{}/{}".format(self.BASE_URL, self.SPORT, path)
        if params is None:
            params = {}
        params["apiKey"] = self.api_key

        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error("PropLine request failed for %s: %s", path, e)
            return None

    def _get_cached(self, key):
        # type: (str) -> Optional[object]
        """Return cached value if still fresh, else None."""
        if key in self._cache:
            cached_time, cached_data = self._cache[key]
            if time.time() - cached_time < self._cache_ttl:
                return cached_data
        return None

    def _set_cached(self, key, data):
        # type: (str, object) -> None
        self._cache[key] = (time.time(), data)

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def get_events(self, date_str=None):
        # type: (Optional[str]) -> List[Dict]
        """
        Get today's NBA games from PropLine.

        Args:
            date_str: Optional YYYY-MM-DD to filter events (defaults to today).

        Returns list of event dicts with keys:
            id, home_team, away_team, commence_time, live
        """
        cache_key = "events:{}".format(date_str or "today")
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        data = self._make_request("events")
        if data is None:
            return []

        # PropLine may return events across multiple days; filter to target date
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        events = []
        for event in data:
            commence = event.get("commence_time", "")
            # commence_time is ISO 8601, e.g. "2026-10-20T23:00:00Z"
            event_date = commence[:10] if commence else ""
            if event_date == date_str:
                events.append(event)

        self._set_cached(cache_key, events)
        return events

    # ------------------------------------------------------------------
    # Bulk props (the main performance win)
    # ------------------------------------------------------------------

    def get_all_props(self, bookmaker="draftkings", date_str=None):
        # type: (str, Optional[str]) -> Dict[str, List[Dict]]
        """
        Fetch ALL prop lines for ALL today's games in ONE call.

        This replaces N separate Balldontlie/OddsAPI calls with 1 PropLine call.

        Args:
            bookmaker: Bookmaker key (default "draftkings").
            date_str: Optional YYYY-MM-DD to filter (defaults to today).

        Returns:
            {event_id: [prop_dict, ...]} where each prop_dict has:
                player_name, prop_type, line, over_odds, under_odds
        """
        cache_key = "all_props:{}:{}".format(bookmaker, date_str or "today")
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        data = self._make_request("odds", {
            "markets": self.ALL_MARKETS,
            "bookmakers": bookmaker,
        })
        if data is None:
            return {}

        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        result = {}  # type: Dict[str, List[Dict]]

        for event in data:
            commence = event.get("commence_time", "")
            event_date = commence[:10] if commence else ""
            if event_date != date_str:
                continue

            event_id = str(event.get("id", ""))
            props = self._parse_event_props(event)
            if props:
                result[event_id] = props

        self._set_cached(cache_key, result)
        total_props = sum(len(v) for v in result.values())
        logger.info(
            "PropLine bulk fetch: %d props across %d games",
            total_props, len(result),
        )
        return result

    def get_best_lines_all_books(self, date_str=None):
        # type: (Optional[str]) -> Dict[str, List[Dict]]
        """
        Fetch props from ALL bookmakers and return the best line per prop.

        For overs: best = lowest line (easier to go over) or best odds at same line
        For unders: best = highest line (easier to stay under) or best odds at same line

        Returns: {event_id: [prop_dict with best_book, all_books fields, ...]}
        """
        cache_key = "best_lines:{}".format(date_str or "today")
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Fetch WITHOUT bookmaker filter → all books
        data = self._make_request("odds", {"markets": self.ALL_MARKETS})
        if data is None:
            return {}

        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        result = {}  # type: Dict[str, List[Dict]]

        for event in data:
            commence = event.get("commence_time", "")
            event_date = commence[:10] if commence else ""
            if event_date != date_str:
                continue

            event_id = str(event.get("id", ""))

            # Collect all books' lines per player|prop
            # {player|prop: {book: {line, over_odds, under_odds}}}
            book_data = {}  # type: Dict[str, Dict[str, Dict]]

            for bk in event.get("bookmakers", []):
                book_key = bk.get("key", "")
                for market in bk.get("markets", []):
                    mkt_key = market.get("key", "")
                    prop_type = self.MARKET_MAP.get(mkt_key)
                    if not prop_type:
                        continue

                    player_lines = {}  # type: Dict[str, Dict]
                    for outcome in market.get("outcomes", []):
                        pname = outcome.get("description", "")
                        direction = outcome.get("name", "")
                        price = outcome.get("price", -110)
                        point = outcome.get("point", 0)
                        if not pname:
                            continue
                        if pname not in player_lines:
                            player_lines[pname] = {"line": point}
                        if direction == "Over":
                            player_lines[pname]["over_odds"] = price
                        elif direction == "Under":
                            player_lines[pname]["under_odds"] = price

                    for pname, ld in player_lines.items():
                        if "over_odds" in ld and "under_odds" in ld and ld.get("line") is not None:
                            key = "{}|{}".format(pname, prop_type)
                            if key not in book_data:
                                book_data[key] = {}
                            try:
                                book_data[key][book_key] = {
                                    "line": float(ld["line"]),
                                    "over_odds": int(ld["over_odds"]) if ld["over_odds"] is not None else -110,
                                    "under_odds": int(ld["under_odds"]) if ld["under_odds"] is not None else -110,
                                }
                            except (TypeError, ValueError):
                                pass  # Skip malformed data

            # For each player|prop, find best over and under lines
            best_props = []
            for key, books in book_data.items():
                parts = key.split("|")
                if len(parts) != 2:
                    continue
                pname, prop_type = parts

                try:
                    # Group books by line value (within 0.5 = same bet)
                    line_groups = {}  # {line: {book: data}}
                    for book, bd in books.items():
                        bk_line = bd.get("line")
                        if bk_line is None:
                            continue
                        bk_line = float(bk_line)
                        group_key = round(bk_line * 2) / 2
                        if group_key not in line_groups:
                            line_groups[group_key] = {}
                        line_groups[group_key][book] = bd

                    if not line_groups:
                        continue

                    # Pick the CONSENSUS line — the line most books agree on.
                    # Alt lines (Kalshi milestones, etc.) have 1-2 books.
                    # The real market line has 5+ books.
                    consensus_line = max(line_groups.keys(),
                                        key=lambda k: len(line_groups[k]))
                    primary_line = consensus_line
                    primary_books = line_groups[consensus_line]

                    # Track best over/under lines for reference
                    sorted_lines = sorted(line_groups.keys())
                    best_over_line_val = sorted_lines[0]
                    best_under_line_val = sorted_lines[-1]

                    # Find best odds within the primary group
                    best_over_odds_val = -9999
                    best_over_book_val = None
                    best_under_odds_val = -9999
                    best_under_book_val = None
                    for bk_name, bk_data in primary_books.items():
                        bo = bk_data.get("over_odds") or -9999
                        bu = bk_data.get("under_odds") or -9999
                        if bo > best_over_odds_val:
                            best_over_odds_val = bo
                            best_over_book_val = bk_name
                        if bu > best_under_odds_val:
                            best_under_odds_val = bu
                            best_under_book_val = bk_name

                    if best_over_odds_val == -9999:
                        best_over_odds_val = -110
                    if best_under_odds_val == -9999:
                        best_under_odds_val = -110

                    # DK info (for reference, not as default)
                    dk = books.get("draftkings", {})
                    dk_line = float(dk.get("line", 0) or 0)
                    dk_over = int(dk.get("over_odds", -110) or -110)
                    dk_under = int(dk.get("under_odds", -110) or -110)

                    best_props.append({
                        "player_name": pname,
                        "prop_type": prop_type,
                        # Primary = best edge line, not DK
                        "line": primary_line,
                        "over_odds": best_over_odds_val,
                        "under_odds": best_under_odds_val,
                        "best_over_book": best_over_book_val or "unknown",
                        "best_under_book": best_under_book_val or "unknown",
                        "best_over_line": best_over_line_val,
                        "best_under_line": best_under_line_val,
                        # DK for reference
                        "dk_line": dk_line,
                        "dk_over_odds": dk_over,
                        "dk_under_odds": dk_under,
                        # All books at this line, ranked by odds
                        "all_books": primary_books,
                        # All available line groups for context
                        "all_line_groups": {
                            str(k): {bk: {"over_odds": bd.get("over_odds"), "under_odds": bd.get("under_odds")}
                                     for bk, bd in v.items()}
                            for k, v in line_groups.items()
                        },
                    })
                except (TypeError, ValueError) as e:
                    logger.debug(f"Skipping prop {key}: {e}")

            if best_props:
                result[event_id] = best_props

        self._set_cached(cache_key, result)
        return result

    def get_game_props(self, event_id, bookmaker="draftkings"):
        # type: (str, str) -> List[Dict]
        """
        Get props for a single game.

        Returns list of prop dicts with:
            player_name, prop_type, line, over_odds, under_odds
        """
        cache_key = "game_props:{}:{}".format(event_id, bookmaker)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        data = self._make_request(
            "events/{}/odds".format(event_id),
            {"markets": self.ALL_MARKETS, "bookmakers": bookmaker},
        )
        if data is None:
            return []

        props = self._parse_event_props(data)
        self._set_cached(cache_key, props)
        return props

    # ------------------------------------------------------------------
    # Graded results
    # ------------------------------------------------------------------

    def get_graded_results(self, event_id):
        # type: (str) -> List[Dict]
        """
        Get graded results (hit/loss/push) for a completed game.

        Returns list of dicts with:
            player, prop_type, line, actual_value, resolution
        """
        cache_key = "results:{}".format(event_id)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        data = self._make_request("events/{}/results".format(event_id))
        if data is None:
            return []

        results = []
        # data may be a list or a dict with results nested
        items = data if isinstance(data, list) else data.get("results", [])

        for item in items:
            market_key = item.get("market", item.get("market_key", ""))
            prop_type = self.MARKET_MAP.get(market_key)
            if not prop_type:
                continue

            results.append({
                "player": item.get("description", item.get("player", "")),
                "prop_type": prop_type,
                "line": float(item.get("point", item.get("line", 0))),
                "actual_value": float(item.get("actual", item.get("actual_value", 0))),
                "resolution": item.get("resolution", item.get("result", "")),
            })

        self._set_cached(cache_key, results)
        return results

    # ------------------------------------------------------------------
    # Active players (derived from props)
    # ------------------------------------------------------------------

    def get_active_players_for_game(self, event_id, bookmaker="draftkings"):
        # type: (str, str) -> List[Dict]
        """
        Get unique player names from props — if DK has props on you, you're playing.

        Returns: [{"player_name": "Jalen Brunson"}, ...]
        """
        props = self.get_game_props(event_id, bookmaker)
        seen = set()  # type: set
        players = []
        for prop in props:
            name = prop.get("player_name", "")
            if name and name not in seen:
                seen.add(name)
                players.append({"player_name": name})
        return players

    # ------------------------------------------------------------------
    # Team name matching (PropLine uses full names, e.g. "New York Knicks")
    # ------------------------------------------------------------------

    def find_event_id(self, home_team, away_team, date_str=None):
        # type: (str, str, Optional[str]) -> Optional[str]
        """
        Find PropLine event ID for a game by team names.

        Team name matching is case-insensitive and handles common variants
        (e.g. "LA Clippers" vs "Los Angeles Clippers").
        """
        events = self.get_events(date_str)

        def normalize(name):
            # type: (str) -> str
            return name.lower().replace("la clippers", "los angeles clippers")

        home_norm = normalize(home_team)
        away_norm = normalize(away_team)

        for event in events:
            event_home = normalize(event.get("home_team", ""))
            event_away = normalize(event.get("away_team", ""))

            # Check both directions (API might swap home/away)
            if (home_norm in event_home or event_home in home_norm) and \
               (away_norm in event_away or event_away in away_norm):
                return str(event.get("id", ""))
            if (away_norm in event_home or event_home in away_norm) and \
               (home_norm in event_away or event_away in home_norm):
                return str(event.get("id", ""))

        return None

    def get_event_team_map(self, date_str=None):
        # type: (Optional[str]) -> Dict[str, Dict]
        """
        Build lookup from team name to event info for quick matching.

        Returns: {"new york knicks": {"event_id": "129635", "home_team": ..., "away_team": ...}, ...}
        """
        events = self.get_events(date_str)
        team_map = {}
        for event in events:
            event_id = str(event.get("id", ""))
            info = {
                "event_id": event_id,
                "home_team": event.get("home_team", ""),
                "away_team": event.get("away_team", ""),
                "commence_time": event.get("commence_time", ""),
            }
            home_key = event.get("home_team", "").lower()
            away_key = event.get("away_team", "").lower()
            team_map[home_key] = info
            team_map[away_key] = info
        return team_map

    # ------------------------------------------------------------------
    # Props parsing (shared between bulk and single-game)
    # ------------------------------------------------------------------

    def _parse_event_props(self, event_data):
        # type: (Dict) -> List[Dict]
        """
        Parse a single event's bookmaker/market data into prop dicts.

        Works for both the bulk response (list of events) and single-event response.
        """
        props = []

        bookmakers = event_data.get("bookmakers", [])
        if not bookmakers:
            return props

        for bookmaker in bookmakers:
            for market in bookmaker.get("markets", []):
                market_key = market.get("key", "")
                prop_type = self.MARKET_MAP.get(market_key)
                if not prop_type:
                    continue

                # Group outcomes by player (Over/Under pairs)
                player_lines = {}  # type: Dict[str, Dict]
                for outcome in market.get("outcomes", []):
                    player_name = outcome.get("description", "")
                    direction = outcome.get("name", "")  # "Over" or "Under"
                    odds = outcome.get("price", -110)
                    line = outcome.get("point", 0)

                    if player_name not in player_lines:
                        player_lines[player_name] = {
                            "line": line,
                            "player_name": player_name,
                        }

                    if direction == "Over":
                        player_lines[player_name]["over_odds"] = odds
                    elif direction == "Under":
                        player_lines[player_name]["under_odds"] = odds

                # Only include complete pairs (both over and under)
                for player_name, line_data in player_lines.items():
                    if "over_odds" in line_data and "under_odds" in line_data:
                        props.append({
                            "player_name": player_name,
                            "prop_type": prop_type,
                            "line": float(line_data["line"]),
                            "over_odds": line_data["over_odds"],
                            "under_odds": line_data["under_odds"],
                        })

        return props

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self):
        # type: () -> None
        """Clear all cached data."""
        self._cache.clear()


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    client = PropLineClient()
    if not client.api_key:
        print("Error: PROPLINE_API_KEY not set in environment")
        exit(1)

    print("\n=== PropLine Events ===")
    events = client.get_events()
    for e in events:
        print("  {} vs {} (id={}, {})".format(
            e.get("away_team"), e.get("home_team"),
            e.get("id"), e.get("commence_time"),
        ))

    print("\n=== Bulk Props Fetch ===")
    t0 = time.time()
    all_props = client.get_all_props()
    elapsed = time.time() - t0
    total = sum(len(v) for v in all_props.values())
    print("  Fetched {} props across {} games in {:.2f}s".format(
        total, len(all_props), elapsed,
    ))

    for event_id, props in all_props.items():
        players = set(p["player_name"] for p in props)
        print("  Event {}: {} players, {} props".format(
            event_id, len(players), len(props),
        ))
        for p in props[:3]:
            print("    {} {} O/U {:.1f} ({}/{})".format(
                p["player_name"], p["prop_type"], p["line"],
                p["over_odds"], p["under_odds"],
            ))
