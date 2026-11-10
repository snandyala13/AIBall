"""
engine.py - Analysis service singleton

Creates and holds the expensive pipeline/engine/generator instances.
Mirrors the flows in ball.py for game and player analysis.
"""

import os
import sys
import logging
from typing import Dict, List, Optional
from datetime import datetime

# Add parent directory to path so we can import the core modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_pipeline import DataPipeline
from model_engine import ModelEngine
from bet_generator import BetGenerator
from propline_client import PropLineClient

try:
    from line_edge import LineEdgeDetector
    LINE_EDGE_AVAILABLE = True
except ImportError:
    LINE_EDGE_AVAILABLE = False

from web.clv_tracker import get_clv_tracker
from consensus_engine import ConsensusEngine as _ConsensusEngine, SOFT_BOOKS

# v8 calibrated classifier (primary model)
try:
    from v2.v8_predictor import V8Predictor
    V8_AVAILABLE = True
except ImportError:
    V8_AVAILABLE = False

# Portfolio management
try:
    from portfolio_manager import PortfolioManager
    PORTFOLIO_AVAILABLE = True
except ImportError:
    PORTFOLIO_AVAILABLE = False

# Push alerts
try:
    from push_alerts import get_alert_manager
    ALERTS_AVAILABLE = True
except ImportError:
    ALERTS_AVAILABLE = False

# Account health monitoring
try:
    from account_health import AccountHealthMonitor
    ACCOUNT_HEALTH_AVAILABLE = True
except ImportError:
    ACCOUNT_HEALTH_AVAILABLE = False

# v2 ML engine (line-anchor LightGBM models)
try:
    from v2.engine import V2AnalysisEngine
    V2_AVAILABLE = True
except ImportError:
    V2_AVAILABLE = False

# Injury cascade engine
try:
    from v2.injury_monitor import InjuryCascadeMonitor
    CASCADE_AVAILABLE = True
except ImportError:
    CASCADE_AVAILABLE = False

try:
    from v2.stale_lines import StaleLineDetector
    STALE_LINES_AVAILABLE = True
except ImportError:
    STALE_LINES_AVAILABLE = False

logger = logging.getLogger(__name__)


class AnalysisEngine:
    """
    Singleton analysis engine that holds the expensive pipeline instances.
    Caches full analysis results per game_id to avoid re-running on refresh.
    """

    _instance = None  # type: Optional[AnalysisEngine]

    def __init__(self):
        api_key = os.getenv("BALLDONTLIE_API_KEY", "")
        if not api_key:
            logger.warning("BALLDONTLIE_API_KEY not set - API calls will fail")

        logger.info("Initializing DataPipeline...")
        self.pipeline = DataPipeline(api_key)

        logger.info("Initializing ModelEngine (loading ML models)...")
        self.engine = ModelEngine(self.pipeline)

        # PropLine client for fast bulk props
        self.propline = PropLineClient()
        self.use_propline = self.propline.api_key is not None
        if self.use_propline:
            logger.info("PropLine enabled - bulk props available")
        else:
            logger.info("PropLine not configured - using standard prop sources")

        logger.info("Initializing BetGenerator...")
        self.generator = BetGenerator(
            self.engine,
            propline_client=self.propline if self.use_propline else None,
        )

        # v2 ML engine (line-anchor LightGBM models)
        self.v2_engine = None
        self.use_v2 = False
        if V2_AVAILABLE:
            try:
                self.v2_engine = V2AnalysisEngine()
                self.use_v2 = self.v2_engine.is_ready()
                if self.use_v2:
                    logger.info("V2 ML engine loaded - using line-anchor LightGBM models")
                else:
                    logger.info("V2 engine initialized but no models found - falling back to v1")
            except Exception as e:
                logger.warning("V2 engine init failed: %s - falling back to v1", e)
        else:
            logger.info("V2 engine not available - using v1 rules-based engine")

        # Live game tracker for in-progress games
        self.live_tracker = None
        try:
            from v2.live_game import LiveGameTracker
            self.live_tracker = LiveGameTracker()
            logger.info("LiveGameTracker available for in-game predictions")
        except Exception as e:
            logger.debug("LiveGameTracker not available: %s", e)

        # Line edge detector (for direct access from engine)
        self.line_edge = None
        if LINE_EDGE_AVAILABLE:
            try:
                self.line_edge = LineEdgeDetector(
                    propline_client=self.propline if self.use_propline else None
                )
                logger.info("LineEdgeDetector available for web UI")
            except Exception as e:
                logger.debug("LineEdgeDetector init failed: %s", e)

        # Empirical pick filter (lookup table from grading data)
        self.pick_filter = None
        try:
            from v2.pick_filter import EmpiricalFilter
            self.pick_filter = EmpiricalFilter()
            if self.pick_filter.categories:
                logger.info(
                    "Empirical pick filter loaded (%d categories, %d graded picks)",
                    len(self.pick_filter.categories),
                    self.pick_filter.metadata.get("total_graded_picks", 0),
                )
            else:
                logger.info("Empirical pick filter loaded (no grading data yet)")
        except Exception as e:
            logger.debug("Empirical pick filter not available: %s", e)

        # Stale line detector
        self.stale_detector = None
        if STALE_LINES_AVAILABLE:
            try:
                fb = self.v2_engine.feature_builder if self.v2_engine else None
                self.stale_detector = StaleLineDetector(feature_builder=fb)
                logger.info("StaleLineDetector initialized")
            except Exception as e:
                logger.debug("StaleLineDetector init failed: %s", e)

        # Injury cascade monitor
        self.cascade_monitor = None
        if CASCADE_AVAILABLE:
            try:
                self.cascade_monitor = InjuryCascadeMonitor()
                if self.cascade_monitor.is_ready:
                    logger.info("InjuryCascadeMonitor loaded (pre-computed cascade tables)")
                else:
                    logger.info("InjuryCascadeMonitor: no cascade tables found, run v2/injury_cascade.py")
                    self.cascade_monitor = None
            except Exception as e:
                logger.debug("InjuryCascadeMonitor init failed: %s", e)

        # Cached PropLine bulk data: {event_id: [props]}
        self._propline_bulk_cache = None  # type: Optional[Dict]

        # Analysis result cache: game_id -> (timestamp, serialized result)
        self._analysis_cache = {}  # type: Dict[int, tuple]
        self._cache_ttl = 120  # 2 minutes (reduced from 5 for fresher data)

        # Prediction cache: stores MODEL OUTPUTS per prop for fast edge recalculation.
        # The model adjustment is fixed — only odds change — so we cache it and
        # recalculate edge every poll cycle (30s) with fresh odds.
        # Key: game_id -> {player|prop_type: {adjustment, q10, q90, predicted_value, ...}}
        self._prediction_cache = {}  # type: Dict[int, Dict[str, Dict]]

        # Live line tracking: {player_name|prop_type: {line, over_odds, under_odds, timestamp}}
        self._previous_lines = {}  # type: Dict[str, Dict]

        # Line snapshot tracking: stores last-logged snapshot per key for dedup
        self._snapshot_previous = {}  # type: Dict[str, Dict]

        # CLV tracker (SQLite-backed)
        self.clv_tracker = get_clv_tracker()
        logger.info("CLV tracker initialized (%s)", self.clv_tracker.db_path)

        # Consensus engine for sharp-weighted devigged probabilities
        self.consensus = _ConsensusEngine()
        logger.info("ConsensusEngine initialized for live consensus analysis")

        # v8 calibrated classifier (replaces v7 regression with classification + consensus blend)
        self.v8_predictor = None
        self.use_v8 = False
        if V8_AVAILABLE:
            try:
                self.v8_predictor = V8Predictor()
                self.use_v8 = self.v8_predictor.is_ready()
                if self.use_v8:
                    logger.info("V8 calibrated classifier loaded — consensus-blended predictions active")
                else:
                    logger.info("V8 predictor initialized but no models found — using v2 fallback")
            except Exception as e:
                logger.warning("V8 predictor init failed: %s — using v2 fallback", e)

        # Portfolio exposure manager
        self.portfolio = None
        if PORTFOLIO_AVAILABLE:
            try:
                bankroll = float(os.getenv("AIBALL_BANKROLL", "1000"))
                self.portfolio = PortfolioManager(bankroll=bankroll)
                logger.info("PortfolioManager initialized (bankroll=$%.0f)", bankroll)
            except Exception as e:
                logger.warning("PortfolioManager init failed: %s", e)

        # Push alert manager
        self.alert_manager = None
        if ALERTS_AVAILABLE:
            try:
                self.alert_manager = get_alert_manager()
                logger.info("PushAlertManager initialized")
            except Exception as e:
                logger.warning("PushAlertManager init failed: %s", e)

        # Account health monitor
        self.account_health = None
        if ACCOUNT_HEALTH_AVAILABLE:
            try:
                self.account_health = AccountHealthMonitor()
                logger.info("AccountHealthMonitor initialized")
            except Exception as e:
                logger.warning("AccountHealthMonitor init failed: %s", e)

        # Track which games have had closing lines captured
        self._closing_lines_captured = set()  # type: set

        logger.info("AnalysisEngine ready")

    @classmethod
    def get_instance(cls):
        # type: () -> AnalysisEngine
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_propline_bulk(self, date_str=None):
        # type: (Optional[str]) -> None
        """Load PropLine bulk props into cache if not already loaded."""
        if self.use_propline and self._propline_bulk_cache is None:
            self._propline_bulk_cache = self.propline.get_all_props(date_str=date_str)

    def get_games_for_date(self, date_str=None):
        # type: (Optional[str]) -> List[Dict]
        """
        Get games for a specific date with DK odds.

        Args:
            date_str: YYYY-MM-DD format, defaults to today
        """
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        # Warm the PropLine bulk cache so analyze_game() can reuse it
        self._ensure_propline_bulk(date_str)

        # Fetch games for the specific date only
        data = self.pipeline.client._make_request("games", {
            "dates[]": date_str
        })
        games = data.get("data", []) if data else []
        dk_odds = self.pipeline.client.get_draftkings_game_odds()

        # Build a PropLine team map for enriching game data
        propline_team_map = {}
        if self.use_propline:
            propline_team_map = self.propline.get_event_team_map(date_str)

        result = []
        for game in games:
            home = game.get("home_team", {})
            away = game.get("visitor_team", {})
            game_id = game.get("id")
            odds = dk_odds.get(game_id, {})

            # Count injuries for each team
            home_id = home.get("id")
            away_id = away.get("id")
            home_injuries = []
            away_injuries = []

            if home_id:
                for inj in self.pipeline.client.get_injuries(home_id):
                    status = inj.get("status", "").lower()
                    player = inj.get("player", {})
                    name = "{} {}".format(
                        player.get("first_name", ""),
                        player.get("last_name", "")
                    )
                    home_injuries.append({
                        "name": name,
                        "status": status,
                    })

            if away_id:
                for inj in self.pipeline.client.get_injuries(away_id):
                    status = inj.get("status", "").lower()
                    player = inj.get("player", {})
                    name = "{} {}".format(
                        player.get("first_name", ""),
                        player.get("last_name", "")
                    )
                    away_injuries.append({
                        "name": name,
                        "status": status,
                    })

            # Look up PropLine event ID for this game (for cross-referencing)
            propline_event_id = None
            home_key = home.get("full_name", "").lower()
            if home_key in propline_team_map:
                propline_event_id = propline_team_map[home_key].get("event_id")

            # Scores from BDL API (available for live/final games)
            home_score = game.get("home_team_score") or 0
            away_score = game.get("visitor_team_score") or 0

            # Determine game status category
            # BDL returns: "Final", "1st Qtr", "2nd Qtr", "3rd Qtr", "4th Qtr",
            # "Halftime", "OT1", etc. for live, or a datetime string for upcoming
            raw_status = game.get("status", "")
            period = game.get("period", 0)
            if raw_status == "Final":
                status_category = "final"
            elif period > 0 or raw_status in ("Halftime",) or "Qtr" in raw_status or "OT" in raw_status:
                status_category = "live"
            else:
                status_category = "upcoming"

            # Get PropLine live status if available
            propline_live = False
            if self.use_propline and propline_event_id:
                for team_key, info in propline_team_map.items():
                    if info.get("event_id") == propline_event_id:
                        propline_live = info.get("live", False)
                        break

            game_entry = {
                "game_id": game_id,
                "home_team": home.get("full_name", ""),
                "home_abbreviation": home.get("abbreviation", ""),
                "home_team_id": home_id,
                "away_team": away.get("full_name", ""),
                "away_abbreviation": away.get("abbreviation", ""),
                "away_team_id": away_id,
                "status": raw_status,
                "status_category": status_category,
                "home_score": home_score,
                "away_score": away_score,
                "date": game.get("date", "")[:10],
                "spread": odds.get("spread", 0),
                "total": odds.get("total", 0),
                "home_injuries": home_injuries,
                "away_injuries": away_injuries,
                "injury_count": len([
                    i for i in home_injuries + away_injuries
                    if i["status"] == "out"
                ]),
            }
            if propline_event_id:
                game_entry["propline_event_id"] = propline_event_id

            result.append(game_entry)

        return result

    def get_best_lines_for_game(self, home_team, away_team, date_str=None):
        # type: (str, str, Optional[str]) -> Dict[str, Dict]
        """
        For each player prop in this game, find the best line across all books.

        Returns: {
            "Jayson Tatum|points|over": {
                "best_book": "fanduel",
                "best_line": 24.5,
                "best_odds": -105,
                "dk_line": 25.5,
                "dk_odds": -110,
                "edge_vs_dk": 1.0,
                "all_books": {
                    "draftkings": {"line": 25.5, "odds": -110},
                    "fanduel": {"line": 24.5, "odds": -105},
                }
            }
        }
        """
        if not self.use_propline:
            return {}

        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        cache_key = "best_lines:{}:{}:{}".format(
            home_team.lower(), away_team.lower(), date_str
        )
        cached = self.propline._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            all_books_data = self.propline._make_request("odds", {
                "markets": self.propline.ALL_MARKETS,
            })
            if not all_books_data:
                return {}

            # Find the event matching this game
            target_events = []
            home_norm = home_team.lower()
            away_norm = away_team.lower()
            # PropLine uses UTC dates, so evening ET games show as next day.
            # Accept both the requested date and the next day.
            from datetime import timedelta
            try:
                base = datetime.strptime(date_str, "%Y-%m-%d")
                next_day = (base + timedelta(days=1)).strftime("%Y-%m-%d")
            except ValueError:
                next_day = date_str
            valid_dates = {date_str, next_day}

            for event in all_books_data:
                commence = event.get("commence_time", "")
                event_date = commence[:10] if commence else ""
                if event_date not in valid_dates:
                    continue
                event_home = event.get("home_team", "").lower()
                event_away = event.get("away_team", "").lower()
                if ((home_norm in event_home or event_home in home_norm) and
                    (away_norm in event_away or event_away in away_norm)):
                    target_events.append(event)
                elif ((away_norm in event_home or event_home in away_norm) and
                      (home_norm in event_away or event_away in home_norm)):
                    target_events.append(event)

            if not target_events:
                return {}

            # Collect lines: {player|prop|side: {book: {line, odds}}}
            prop_books = {}  # type: Dict[str, Dict[str, Dict]]

            # Exclude DFS/fantasy pick'em platforms from betting recommendations
            # but keep prediction markets (kalshi, polymarket) and exchanges (novig) for logging/tracking
            EXCLUDED_BOOKS_BETTING = {"pick6", "prizepicks", "sleeper", "underdog"}
            TRACK_ONLY_BOOKS = {"kalshi", "polymarket"}  # Log but don't use for recommendations
            for event in target_events:
                for bookmaker in event.get("bookmakers", []):
                    book_key = bookmaker.get("key", "unknown")
                    if book_key.lower() in EXCLUDED_BOOKS_BETTING:
                        continue
                    for market in bookmaker.get("markets", []):
                        market_key = market.get("key", "")
                        prop_type = self.propline.MARKET_MAP.get(market_key)
                        if not prop_type:
                            continue

                        # Group outcomes by player
                        player_lines = {}  # type: Dict[str, Dict]
                        for outcome in market.get("outcomes", []):
                            player_name = outcome.get("description", "")
                            direction = outcome.get("name", "")
                            odds = outcome.get("price", -110)
                            line = outcome.get("point", 0)

                            if line is None:
                                continue
                            if player_name not in player_lines:
                                player_lines[player_name] = {"line": float(line)}
                            if direction == "Over":
                                player_lines[player_name]["over_odds"] = odds
                            elif direction == "Under":
                                player_lines[player_name]["under_odds"] = odds

                        for player_name, pdata in player_lines.items():
                            line_val = pdata["line"]

                            # Over side
                            if "over_odds" in pdata:
                                over_key = "{}|{}|over".format(player_name, prop_type)
                                if over_key not in prop_books:
                                    prop_books[over_key] = {}
                                prop_books[over_key][book_key] = {
                                    "line": line_val,
                                    "odds": pdata["over_odds"],
                                }

                            # Under side
                            if "under_odds" in pdata:
                                under_key = "{}|{}|under".format(player_name, prop_type)
                                if under_key not in prop_books:
                                    prop_books[under_key] = {}
                                prop_books[under_key][book_key] = {
                                    "line": line_val,
                                    "odds": pdata["under_odds"],
                                }

            # Build best-line results
            best_lines = {}  # type: Dict[str, Dict]
            for lookup_key, books in prop_books.items():
                if len(books) < 1:
                    continue
                parts = lookup_key.split("|", 2)
                if len(parts) != 3:
                    continue
                player, prop_type, side = parts

                dk = books.get("draftkings")
                dk_line = dk["line"] if dk else None
                dk_odds = dk["odds"] if dk else None

                # Find best: for over, best = lowest line or best odds at same line
                # For under, best = highest line or best odds at same line
                best_book = None
                best_line = None
                best_odds = None

                # Exclude DFS/prediction markets from best-book selection
                _NON_BETTING_BOOKS = {"pick6", "prizepicks", "sleeper", "underdog", "kalshi", "polymarket"}
                for bk, bdata in books.items():
                    if bk.lower() in _NON_BETTING_BOOKS:
                        continue
                    bline = bdata["line"]
                    bodds = bdata["odds"]
                    if best_book is None:
                        best_book = bk
                        best_line = bline
                        best_odds = bodds
                        continue

                    if side == "over":
                        # Lower line is better for over, then better odds
                        if bline < best_line or (bline == best_line and bodds > best_odds):
                            best_book = bk
                            best_line = bline
                            best_odds = bodds
                    else:
                        # Higher line is better for under, then better odds
                        if bline > best_line or (bline == best_line and bodds > best_odds):
                            best_book = bk
                            best_line = bline
                            best_odds = bodds

                edge_vs_dk = 0.0
                if dk_line is not None and best_line is not None:
                    if side == "over":
                        edge_vs_dk = round(dk_line - best_line, 1)
                    else:
                        edge_vs_dk = round(best_line - dk_line, 1)

                best_lines[lookup_key] = {
                    "best_book": best_book,
                    "best_line": best_line,
                    "best_odds": best_odds,
                    "dk_line": dk_line,
                    "dk_odds": dk_odds,
                    "edge_vs_dk": edge_vs_dk,
                    "all_books": {bk: {"line": bd["line"], "odds": bd["odds"]}
                                  for bk, bd in books.items()},
                }

            self.propline._set_cached(cache_key, best_lines)
            logger.info(
                "Best lines: %d props across %d books for %s vs %s",
                len(best_lines),
                len(set(bk for bl in best_lines.values() for bk in bl.get("all_books", {}))),
                home_team, away_team,
            )
            return best_lines

        except Exception as exc:
            logger.error("Error getting best lines: %s", exc)
            return {}

    def analyze_game(self, game_id):
        # type: (int) -> Optional[Dict]
        """
        Full analysis for a single game (the heavy call).

        Mirrors ball.py run_game_analysis flow:
        1. Find the game by ID
        2. Build game context
        3. Get active players from DK props
        4. Collect injuries for pop-off detection
        5. Run engine.analyze_game
        6. Generate recommendations

        Returns serialized GameRecommendations or None.
        """
        from web.serializers import serialize_game_recommendations, serialize_projection

        # Return cached result if fresh (TTL-based, avoids stale data)
        import time as _time
        if game_id in self._analysis_cache:
            cached_at, cached_result = self._analysis_cache[game_id]
            age = _time.time() - cached_at
            if age < self._cache_ttl:
                logger.info(f"Returning cached analysis for game {game_id} ({age:.0f}s old)")
                return cached_result
            else:
                logger.info(f"Cache expired for game {game_id} ({age:.0f}s old), re-analyzing")

        # Find the game — search yesterday, today, and tomorrow
        # (UTC vs ET offset means evening ET games show as yesterday in UTC after midnight)
        from datetime import timedelta
        now = datetime.now()
        yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        today = now.strftime("%Y-%m-%d")
        tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")
        target_game = None
        for date_str in [today, yesterday, tomorrow]:
            data = self.pipeline.client._make_request("games", {"dates[]": date_str})
            for game in (data.get("data", []) if data else []):
                if game.get("id") == game_id:
                    target_game = game
                    break
            if target_game:
                break

        if not target_game:
            return None

        home_team = target_game.get("home_team", {}).get("full_name", "")
        away_team = target_game.get("visitor_team", {}).get("full_name", "")
        home_id = target_game.get("home_team", {}).get("id")
        away_id = target_game.get("visitor_team", {}).get("id")

        # Build game context
        context = self.pipeline.build_game_context(home_team, away_team)
        if not context:
            return None

        # Get injured player names from context (already fetched by build_game_context)
        injured_names = []
        for inj in getattr(context, 'injuries', []):
            if isinstance(inj, dict):
                player = inj.get("player", {})
                name = "{} {}".format(player.get("first_name", ""), player.get("last_name", ""))
                if inj.get("status", "").lower() == "out":
                    injured_names.append(name)
            elif isinstance(inj, str):
                injured_names.append(inj)

        # Get active players from props (PropLine fast path or Balldontlie fallback)
        active_players = self.pipeline.get_active_players_for_game(
            game_id, home_team=home_team, away_team=away_team
        )
        if not active_players:
            return {"error": "No DraftKings props available yet for this game"}

        player_names = [p["player_name"] for p in active_players]

        game_date = target_game.get("date", "")[:10] or today
        best_lines = self.get_best_lines_for_game(home_team, away_team, game_date)
        matchup = "{} @ {}".format(away_team, home_team)

        # ------------------------------------------------------------------
        # V2 is the PRIMARY model. V1 only runs as fallback.
        # ------------------------------------------------------------------
        result = {
            "matchup": matchup,
            "game_id": game_id,
            "timestamp": datetime.now().isoformat(),
            "spread": getattr(context, 'spread', 0),
            "total": getattr(context, 'total', 220),
        }

        if best_lines:
            result["best_lines"] = best_lines

        v2_success = False
        if self.use_v2 and self.v2_engine:
            try:
                v2_predictions = self._run_v2_analysis(
                    context, active_players, player_names,
                    game_date, best_lines or {}, game_id=game_id,
                )
                if v2_predictions:
                    result["v2_predictions"] = v2_predictions
                    # Tag model version from the engine's predictor
                    if (self.v2_engine and hasattr(self.v2_engine, 'predictor')
                            and self.v2_engine.predictor.model_type == "v6_dual"):
                        v_meta = getattr(self.v2_engine.predictor, 'v6_metadata', {})
                        if v_meta.get("version", "").startswith("v7"):
                            result["model_version"] = "v7"
                        else:
                            result["model_version"] = "v6"
                    else:
                        result["model_version"] = "v3"
                    v2_success = True
                    logger.info(
                        "V2 produced %d predictions for game %d",
                        len(v2_predictions), game_id,
                    )

                    # Log v2 predictions
                    self._log_v2_predictions(v2_predictions, game_id, matchup, game_date)

                    # Build v2 parlays
                    v2_parlays = self.v2_engine.build_parlays(v2_predictions)
                    if v2_parlays:
                        result["v2_parlays"] = v2_parlays
            except Exception as e:
                import traceback
                logger.error("V2 failed for game %d: %s\n%s", game_id, e, traceback.format_exc())

        # V1 fallback disabled — v6/v3 ML model is the only model
        if not v2_success:
            logger.warning("ML model produced no predictions for game %d — no fallback", game_id)
            result["model_version"] = "v6_no_predictions"

        # Stale line detection
        if self.stale_detector and result.get("v2_predictions"):
            try:
                # Build props list from v2 predictions for the detector
                stale_props = []
                pred_lookup = {}
                for pred in result["v2_predictions"]:
                    pname = pred.get("player", "")
                    ptype = pred.get("prop_type", "")
                    stale_props.append({
                        "player_name": pname,
                        "prop_type": ptype,
                        "line": pred.get("line", 0),
                        "all_books": pred.get("all_books", {}),
                    })
                    pred_lookup["{}|{}".format(pname, ptype)] = pred

                # Collect injuries for cascade signal
                game_injuries = []
                for inj_list in [result.get("home_injuries", []), result.get("away_injuries", [])]:
                    if isinstance(inj_list, list):
                        game_injuries.extend(inj_list)
                # Also try from the data pipeline injuries
                if hasattr(context, 'injuries'):
                    for inj in context.injuries:
                        if isinstance(inj, dict):
                            player = inj.get("player", {})
                            name = "{} {}".format(player.get("first_name", ""), player.get("last_name", ""))
                            status = inj.get("status", "")
                            game_injuries.append({"name": name.strip(), "status": status.lower()})

                stale_results = self.stale_detector.detect_for_game(
                    stale_props, game_date, game_context=context,
                    injuries=game_injuries, predictions=pred_lookup,
                )
                result["stale_lines"] = stale_results

                # Annotate individual predictions with staleness info
                stale_lookup = {}
                for sr in stale_results.get("all_results", []):
                    key = "{}|{}".format(sr["player"], sr["prop_type"])
                    stale_lookup[key] = sr

                for pred in result["v2_predictions"]:
                    pkey = "{}|{}".format(pred.get("player", ""), pred.get("prop_type", ""))
                    sl = stale_lookup.get(pkey)
                    if sl:
                        pred["staleness_score"] = sl["staleness_score"]
                        pred["is_stale"] = sl["is_stale"]
                        pred["stale_reason"] = sl["top_reason"]
                    else:
                        pred["staleness_score"] = 0
                        pred["is_stale"] = False
                        pred["stale_reason"] = ""

                stale_count = stale_results.get("stale_count", 0)
                if stale_count:
                    logger.info("Stale lines detected: %d/%d for game %d",
                                stale_count, stale_results.get("total_props", 0), game_id)
            except Exception as e:
                logger.warning("Stale line detection failed for game %d: %s", game_id, e)

        # Injury cascade enrichment
        if self.cascade_monitor and result.get("v2_predictions"):
            try:
                # Build injury lists from context
                home_injs = []
                away_injs = []
                for inj in getattr(context, 'injuries', []):
                    if isinstance(inj, dict):
                        player = inj.get("player", {})
                        name = "{} {}".format(
                            player.get("first_name", ""),
                            player.get("last_name", ""),
                        )
                        status = inj.get("status", "")
                        team_data = player.get("team", {})
                        entry = {"name": name, "status": status}
                        if team_data.get("full_name") == home_team:
                            home_injs.append(entry)
                        else:
                            away_injs.append(entry)

                cascades = self.get_cascade_for_game(
                    game_id,
                    home_injuries=home_injs,
                    away_injuries=away_injs,
                )
                if cascades:
                    result["injury_cascades"] = cascades

                    # Tag individual predictions with cascade info
                    cascade_lookup = {}  # {player|prop: boost_pct}
                    for c in cascades:
                        for imp in c.get("impacts", []):
                            key = "{}|{}".format(imp["player"], imp["prop"])
                            cascade_lookup[key] = {
                                "boost_pct": imp["boost_pct"],
                                "absent_player": c["absent_player"],
                                "confidence": imp["confidence"],
                            }

                    for pred in result["v2_predictions"]:
                        pname = pred.get("player", "")
                        ptype = pred.get("prop_type", "")
                        ckey = "{}|{}".format(pname, ptype)
                        if ckey in cascade_lookup:
                            pred["cascade"] = cascade_lookup[ckey]

                    tagged = sum(
                        1 for p in result["v2_predictions"] if "cascade" in p
                    )
                    if tagged:
                        logger.info(
                            "Cascade: tagged %d predictions for game %d", tagged, game_id
                        )
            except Exception as e:
                logger.debug("Cascade enrichment failed for game %d: %s", game_id, e)

        # Cache the result with timestamp
        import time as _time
        self._analysis_cache[game_id] = (_time.time(), result)
        logger.info(f"Cached analysis for game {game_id}")

        return result

    def _log_v2_predictions(self, predictions, game_id, matchup, game_date):
        # type: (List[Dict], int, str, str) -> None
        """Log v2 predictions to JSONL for tracking and grading.

        Writes to outputs/v2_predictions_{season}.jsonl.
        Uses an absolute path anchored to the project root so it works
        regardless of CWD (uvicorn, Docker, EFS symlink scenarios).

        Only skips predictions with truly broken data (NaN, extreme outliers).
        """
        import json
        import os
        import traceback

        # Resolve output dir relative to THIS file's location (web/engine.py -> project root)
        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _outputs_dir = os.path.join(_project_root, "outputs")
        os.makedirs(_outputs_dir, exist_ok=True)

        # Compute season string (NBA season spans Oct-June)
        now = datetime.now()
        season_year = now.year if now.month >= 10 else now.year - 1
        season = "{}-{}".format(season_year, str(season_year + 1)[-2:])
        log_file = os.path.join(_outputs_dir, "v2_predictions_{}.jsonl".format(season))

        logged = 0
        skipped = 0

        # Dedup: track what we've already logged so re-analysis doesn't create duplicates.
        # On first call, scan existing JSONL to seed the set (survives server restarts).
        if not hasattr(self, '_logged_predictions'):
            self._logged_predictions = set()
            try:
                if os.path.exists(log_file):
                    with open(log_file) as rf:
                        for raw_line in rf:
                            raw_line = raw_line.strip()
                            if not raw_line:
                                continue
                            try:
                                rec = json.loads(raw_line)
                                dk = "{}|{}|{}|{}|{}".format(
                                    rec.get("game_id", ""),
                                    rec.get("player", ""),
                                    rec.get("prop_type", ""),
                                    rec.get("side", ""),
                                    rec.get("date", ""),
                                )
                                self._logged_predictions.add(dk)
                            except (json.JSONDecodeError, KeyError):
                                continue
                    logger.info("Loaded %d existing prediction keys for dedup from %s", len(self._logged_predictions), log_file)
            except Exception as e:
                logger.error("Could not load existing predictions for dedup: %s", e)

        try:
            with open(log_file, "a") as f:
                for pred in predictions:
                    # Skip duplicates (same player/prop/side/game/date)
                    dedup_key = "{}|{}|{}|{}|{}".format(
                        game_id, pred.get("player", ""),
                        pred.get("prop_type", ""),
                        pred.get("side", "over"),
                        game_date,
                    )
                    if dedup_key in self._logged_predictions:
                        continue
                    edge = pred.get("edge_percent", 0)
                    predicted_value = pred.get("predicted_value", 0)
                    confidence = pred.get("confidence", 0)

                    # Minimal sanity checks -- log everything the model produces
                    # so we can grade it later. Only skip truly broken data.
                    import math
                    if isinstance(edge, float) and (math.isnan(edge) or math.isinf(edge)):
                        skipped += 1
                        continue
                    if not (-100 <= edge <= 100):
                        skipped += 1
                        continue

                    # Detect model version from predictor
                    _log_model = "v6_dual"
                    if self.v2_engine and hasattr(self.v2_engine, 'predictor'):
                        _log_model = self.v2_engine.predictor.model_type or "v6_dual"

                    record = {
                        "date": game_date,
                        "game_id": game_id,
                        "matchup": matchup,
                        "player": pred.get("player", ""),
                        "prop_type": pred.get("prop_type", ""),
                        "side": pred.get("side", "over"),
                        "line": pred.get("line", 0),
                        "dk_odds": pred.get("dk_odds", -110),
                        "over_odds": pred.get("over_odds", -110),
                        "under_odds": pred.get("under_odds", -110),
                        "predicted_value": pred.get("predicted_value", 0),
                        "predicted_adjustment": pred.get("predicted_adjustment", 0),
                        "edge_percent": edge,
                        "p_over": pred.get("p_over", 0.5),
                        "p_under": pred.get("p_under", 0.5),
                        "confidence": confidence,
                        "confidence_stars": pred.get("confidence_stars", 1),
                        "kelly_fraction": pred.get("kelly_fraction", 0),
                        "recommended_units": pred.get("recommended_units", 0),
                        "best_book": pred.get("best_book", "draftkings"),
                        "best_odds": pred.get("best_odds"),
                        "odds_tier": pred.get("odds_tier", "standard"),
                        "edge_at_dk": pred.get("edge_at_dk", edge),
                        "filter_tier": pred.get("filter_tier", ""),
                        "all_books": pred.get("all_books", {}),
                        "line_shopping": pred.get("line_shopping", []),
                        "model": _log_model,
                    }
                    f.write(json.dumps(record) + "\n")
                    self._logged_predictions.add(dedup_key)
                    logged += 1

                # Force flush to disk -- do not rely on Python buffering
                f.flush()
                os.fsync(f.fileno())

            if logged > 0:
                logger.info(
                    "V2 logged %d predictions to %s (skipped %d)",
                    logged, log_file, skipped,
                )
            else:
                logger.info(
                    "V2 logging: 0 new predictions (all %d were duplicates or skipped %d) -> %s",
                    len(predictions), skipped, log_file,
                )
        except Exception as e:
            # CRITICAL: Do NOT silently swallow write failures.
            # Log at ERROR level with full traceback so we can diagnose.
            logger.error(
                "FAILED to write predictions to %s: %s\n%s",
                log_file, e, traceback.format_exc(),
            )
            # Fallback: try writing to /tmp so data is not lost entirely
            try:
                fallback_file = "/tmp/v2_predictions_fallback_{}.jsonl".format(season)
                with open(fallback_file, "a") as fb:
                    for pred in predictions:
                        record = {
                            "date": game_date, "game_id": game_id, "matchup": matchup,
                            "player": pred.get("player", ""), "prop_type": pred.get("prop_type", ""),
                            "side": pred.get("side", "over"), "line": pred.get("line", 0),
                            "edge_percent": pred.get("edge_percent", 0),
                            "predicted_value": pred.get("predicted_value", 0),
                            "confidence": pred.get("confidence", 0),
                            "model": "v3_classifier",
                        }
                        fb.write(json.dumps(record) + "\n")
                    fb.flush()
                    os.fsync(fb.fileno())
                logger.warning("Wrote %d predictions to FALLBACK %s", len(predictions), fallback_file)
            except Exception as e2:
                logger.error("FALLBACK write also failed: %s", e2)

    @staticmethod
    def _get_season_string():
        # type: () -> str
        """Return the NBA season string (e.g. '2026-27'). Oct+ belongs to current year."""
        now = datetime.now()
        season_year = now.year if now.month >= 10 else now.year - 1
        return "{}-{}".format(season_year, str(season_year + 1)[-2:])

    def _log_line_snapshot(self, game_id, fresh_props_by_key, snapshot_type, best_lines=None):
        # type: (int, Dict[str, Dict], str, Optional[Dict]) -> None
        """
        Persist a line snapshot to JSONL for CLV calculation later.

        Args:
            game_id: BDL game ID
            fresh_props_by_key: {"player|prop_type": {line, over_odds, under_odds, ...}}
            snapshot_type: "analysis" | "poll" | "closing"
            best_lines: optional cross-book data for best_book / best_odds
        """
        import json

        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _outputs_dir = os.path.join(_project_root, "outputs")
        os.makedirs(_outputs_dir, exist_ok=True)
        season = self._get_season_string()
        log_file = os.path.join(_outputs_dir, "line_snapshots_{}.jsonl".format(season))
        now_iso = datetime.now().isoformat()
        logged = 0

        try:
            with open(log_file, "a") as f:
                for pkey, pdata in fresh_props_by_key.items():
                    parts = pkey.split("|", 1)
                    if len(parts) != 2:
                        continue
                    player, prop_type = parts

                    line = pdata.get("line", 0)
                    over_odds = pdata.get("over_odds", -110)
                    under_odds = pdata.get("under_odds", -110)

                    # For "poll" snapshots, only log if something changed
                    if snapshot_type == "poll":
                        prev = self._snapshot_previous.get(pkey)
                        if prev and prev.get("line") == line and prev.get("over_odds") == over_odds and prev.get("under_odds") == under_odds:
                            continue

                    # Resolve best book from best_lines or all_books
                    best_book = None
                    best_odds = None
                    if best_lines:
                        bl = best_lines.get("{}|over".format(pkey)) or best_lines.get("{}|under".format(pkey))
                        if bl:
                            best_book = bl.get("best_book")
                            best_odds = bl.get("best_odds")
                    if not best_book:
                        all_books = pdata.get("all_books", {})
                        if all_books:
                            for bk in all_books:
                                if bk != "draftkings":
                                    best_book = bk
                                    break

                    record = {
                        "timestamp": now_iso,
                        "game_id": game_id,
                        "player": player,
                        "prop_type": prop_type,
                        "line": line,
                        "over_odds": over_odds,
                        "under_odds": under_odds,
                        "best_book": best_book,
                        "best_odds": best_odds,
                        "snapshot_type": snapshot_type,
                    }
                    f.write(json.dumps(record) + "\n")
                    logged += 1

                    # Update previous snapshot for dedup
                    self._snapshot_previous[pkey] = {
                        "line": line,
                        "over_odds": over_odds,
                        "under_odds": under_odds,
                    }

            if logged > 0:
                logger.info(
                    "Line snapshot (%s): logged %d props for game %d -> %s",
                    snapshot_type, logged, game_id, log_file,
                )
        except Exception as e:
            logger.error("Failed to log line snapshot: %s", e)

    def _log_live_predictions(self, game_id, predictions, game_status):
        # type: (int, list, dict) -> None
        """Log in-game predictions for accuracy tracking.

        Writes to outputs/live_predictions_{season}.jsonl.
        Only logs once per game per quarter (deduped by game_id + period).
        Each record captures the live projection at that point in the game.
        """
        import json

        period = game_status.get("period", 0)
        raw_status = game_status.get("raw", "")

        # Dedup: only log once per game per quarter
        if not hasattr(self, "_live_log_dedup"):
            self._live_log_dedup = set()
        dedup_key = "{}|{}".format(game_id, period)
        if dedup_key in self._live_log_dedup:
            return
        self._live_log_dedup.add(dedup_key)

        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _outputs_dir = os.path.join(_project_root, "outputs")
        os.makedirs(_outputs_dir, exist_ok=True)
        season = self._get_season_string()
        log_file = os.path.join(_outputs_dir, "live_predictions_{}.jsonl".format(season))
        now_iso = datetime.now().isoformat()

        logged = 0
        try:
            with open(log_file, "a") as f:
                for pred in predictions:
                    lg = pred.get("live_game")
                    if not lg:
                        continue

                    record = {
                        "timestamp": now_iso,
                        "game_id": game_id,
                        "period": period,
                        "game_status": raw_status,
                        "home_score": game_status.get("home_score", 0),
                        "away_score": game_status.get("away_score", 0),
                        "player": pred.get("player", ""),
                        "prop_type": pred.get("prop_type", ""),
                        "side": pred.get("side", ""),
                        "line": pred.get("line", 0),
                        "current_stat": lg.get("current_stat", 0),
                        "minutes_played": lg.get("minutes_played", 0),
                        "projected_final": lg.get("projected_final", 0),
                        "live_p_over": lg.get("live_p_over", 0.5),
                        "live_p_under": lg.get("live_p_under", 0.5),
                        "on_pace": lg.get("on_pace", False),
                        "pre_game_edge": pred.get("edge_percent", 0),
                        "pre_game_side": pred.get("side", ""),
                        "dk_odds": pred.get("dk_odds", -110),
                    }
                    f.write(json.dumps(record) + "\n")
                    logged += 1

            if logged > 0:
                logger.info(
                    "Live predictions logged: %d props for game %d (Q%d)",
                    logged, game_id, period,
                )
        except Exception as e:
            logger.error("Failed to log live predictions: %s", e)

    def _run_v2_analysis(self, context, active_players, player_names,
                          game_date, best_lines, game_id=None):
        # type: (...) -> List[Dict]
        """
        Run v2 line-anchor ML analysis on all active players' props.

        Gets props from the data pipeline (DK/PropLine) and passes them
        through the V2AnalysisEngine.

        Also populates _prediction_cache with model outputs for fast
        edge recalculation when odds change (Phase 1 real-time overhaul).

        Returns:
            list of formatted v2 prediction dicts for the web UI
        """
        from web.serializers import serialize_v2_prediction

        # Build props list from the GameContext props (PlayerProp objects)
        # These are the actual DK/PropLine lines matched to players
        props = []
        seen = set()

        for pp in getattr(context, 'props', []):
            pname = pp.player_name
            prop_type = pp.prop_type
            if not pname or not prop_type:
                continue
            # Only process props for active players
            if pname not in set(player_names):
                continue
            # Only prop types the v3 classifier handles
            if prop_type not in ('points', 'rebounds', 'assists', 'threes', 'pts_rebs_asts', 'steals', 'blocks'):
                continue

            pkey = "{}|{}".format(pname, prop_type)
            if pkey in seen:
                continue
            seen.add(pkey)

            all_books = getattr(pp, 'all_books', {})

            props.append({
                    "player_name": pname,
                    "prop_type": prop_type,
                    "line": pp.line,
                    "over_odds": pp.over_odds,
                    "under_odds": pp.under_odds,
                    "all_books": all_books,
                })

        # If no props from active_players (PropLine path returns no nested props),
        # build props from the best_lines data which has all player/prop combos
        if not props and best_lines:
            for lookup_key, bl_data in best_lines.items():
                parts = lookup_key.split("|", 2)
                if len(parts) != 3:
                    continue
                pname, prop_type, side = parts
                if side != "over":
                    continue  # Only process once per player|prop (skip 'under' dups)

                if pname not in set(player_names):
                    continue

                pkey = "{}|{}".format(pname, prop_type)
                if pkey in seen:
                    continue
                seen.add(pkey)

                dk_data = bl_data.get("all_books", {}).get("draftkings", {})
                line = dk_data.get("line", bl_data.get("dk_line", 0))
                over_odds = dk_data.get("over_odds", dk_data.get("odds", -110))

                # Get under odds from the under key
                under_key = "{}|{}|under".format(pname, prop_type)
                under_bl = best_lines.get(under_key, {})
                under_dk = under_bl.get("all_books", {}).get("draftkings", {})
                under_odds = under_dk.get("under_odds", under_dk.get("odds", -110))

                props.append({
                    "player_name": pname,
                    "prop_type": prop_type,
                    "line": float(line) if line else 0,
                    "over_odds": int(over_odds) if over_odds else -110,
                    "under_odds": int(under_odds) if under_odds else -110,
                    "all_books": bl_data.get("all_books", {}),
                })

        if not props:
            logger.info("V2: no props found for analysis")
            return []

        logger.info("V2: analyzing %d props across %d players", len(props), len(player_names))

        # Build lightweight player profile list (v2 just needs team info)
        profiles = []
        for pname in player_names:
            profiles.append({"name": pname, "team": ""})

        # Run v2 engine
        raw_predictions = self.v2_engine.analyze_game(
            game_context=context,
            player_profiles=profiles,
            props=props,
            game_date=game_date,
        )

        # Format for web UI and enrich with cross-book line shopping + arb/middle detection
        formatted = []
        for pred in raw_predictions:
            web_pred = self.v2_engine.format_prediction_for_web(pred)

            # Compute EV at every available book, detect arbs and middles
            self._compute_line_shopping(web_pred, best_lines)

            formatted.append(serialize_v2_prediction(web_pred))

        # Populate prediction cache for fast edge recalculation
        if game_id and formatted:
            import time as _time
            cache_entry = {}
            for pred in formatted:
                pname = pred.get("player", "")
                ptype = pred.get("prop_type", "")
                if not pname or not ptype:
                    continue
                # Sanity check: skip obviously broken predictions
                adj = pred.get("predicted_adjustment", 0)
                line = pred.get("line", 0)
                if line > 0 and abs(adj) > line * 2:
                    logger.warning(
                        "Skipping suspicious prediction: %s %s adj=%+.1f on line %.1f",
                        pname, ptype, adj, line,
                    )
                    continue
                cache_key = "{}|{}".format(pname, ptype)
                cache_entry[cache_key] = {
                    "adjustment": pred.get("predicted_adjustment", 0),
                    "predicted_value": pred.get("predicted_value", 0),
                    "p10": pred.get("p10", 0),
                    "p90": pred.get("p90", 0),
                    "sigma": pred.get("sigma", 3.0),
                    "confidence": pred.get("confidence", 0),
                    "confidence_stars": pred.get("confidence_stars", 1),
                    "line_at_analysis": pred.get("line", 0),
                    "over_odds_at_analysis": pred.get("over_odds", -110),
                    "under_odds_at_analysis": pred.get("under_odds", -110),
                    "all_books": pred.get("all_books", {}),
                    "cached_at": _time.time(),
                }
            self._prediction_cache[game_id] = cache_entry
            logger.info("Cached %d model outputs for game %d (fast edge recalc)",
                        len(cache_entry), game_id)

            # Log initial line snapshot at analysis time (for CLV baseline)
            analysis_props = {}
            for pred in formatted:
                pname = pred.get("player", "")
                ptype = pred.get("prop_type", "")
                if pname and ptype:
                    pk = "{}|{}".format(pname, ptype)
                    analysis_props[pk] = {
                        "line": pred.get("line", 0),
                        "over_odds": pred.get("over_odds", -110),
                        "under_odds": pred.get("under_odds", -110),
                        "all_books": pred.get("all_books", {}),
                    }
            if analysis_props:
                self._log_line_snapshot(
                    game_id, analysis_props, "analysis", best_lines=best_lines,
                )

            # Log to CLV tracker: snapshots + recommendations
            self._log_clv_data(game_id, formatted, game_date, best_lines)

        # Enrich with v8 calibrated classifier predictions
        if self.use_v8 and self.v8_predictor:
            self.v8_predictor.ensure_data_loaded()
            v8_count = 0
            for pred in formatted:
                pname = pred.get("player", "")
                ptype = pred.get("prop_type", "")
                line = pred.get("line", 0)
                over_odds = pred.get("over_odds", -110)
                under_odds = pred.get("under_odds", -110)
                all_books = pred.get("all_books", {})
                if not pname or not ptype or not line:
                    continue
                try:
                    v8_result = self.v8_predictor.predict(
                        pname, ptype, game_date,
                        {"line": line, "over_odds": over_odds,
                         "under_odds": under_odds, "all_books": all_books}
                    )
                    if v8_result:
                        pred["v8_p_over"] = v8_result["p_over"]
                        pred["v8_p_under"] = v8_result["p_under"]
                        pred["v8_model_prob"] = v8_result["model_prob"]
                        pred["v8_consensus_prob"] = v8_result["consensus_prob"]
                        pred["v8_blended_prob"] = v8_result["blended_prob"]
                        pred["v8_edge"] = v8_result["edge"]
                        pred["v8_side"] = v8_result["side"]
                        pred["v8_kelly"] = v8_result["kelly_fraction"]
                        pred["v8_recommended_bet_pct"] = v8_result["recommended_bet_pct"]
                        pred["v8_confidence"] = v8_result["confidence"]
                        # Override main edge/confidence with v8 values for ranking
                        pred["edge_percent"] = round(v8_result["edge"] * 100, 1)
                        pred["confidence"] = v8_result["confidence"]
                        pred["side"] = v8_result["side"]
                        v8_count += 1
                except Exception as e:
                    logger.debug("v8 prediction failed for %s %s: %s", pname, ptype, e)
            if v8_count:
                logger.info("V8 enriched %d/%d predictions with calibrated probabilities",
                           v8_count, len(formatted))

        # Send push alerts if configured
        if self.alert_manager and formatted:
            try:
                alerts = self.alert_manager.check_and_alert(formatted)
                if alerts:
                    logger.info("Sent %d edge alerts", len(alerts))
            except Exception as e:
                logger.debug("Push alert error: %s", e)

        # Tag predictions with empirical filter tier
        if self.pick_filter:
            for pred in formatted:
                side = pred.get("side", "over")
                ptype = pred.get("prop_type", "")
                edge = pred.get("edge_percent", 0)
                try:
                    filt = self.pick_filter.should_show(side, ptype, abs(edge))
                    pred["filter_tier"] = filt["tier"]
                    pred["filter_reason"] = filt["reason"]
                except Exception:
                    pred["filter_tier"] = "marginal"
                    pred["filter_reason"] = ""

        # Apply balanced ranking: interleave picks across prop types
        formatted = self._apply_balanced_ranking(formatted)

        return formatted

    def _log_clv_data(self, game_id, formatted_preds, game_date, best_lines=None):
        # type: (int, List[Dict], str, Optional[Dict]) -> None
        """
        Log line snapshots and bet recommendations to the CLV tracker.

        Called after v2 analysis produces formatted predictions.
        """
        try:
            # Determine model version
            model_version = "v6"
            if self.v2_engine and hasattr(self.v2_engine, 'predictor'):
                model_version = self.v2_engine.predictor.model_type or "v6_dual"

            # Build snapshot and recommendation lists
            snapshots = []
            recommendations = []

            for pred in formatted_preds:
                pname = pred.get("player", "")
                ptype = pred.get("prop_type", "")
                side = pred.get("side", "over")
                if not pname or not ptype or side == "pass":
                    continue

                line = pred.get("line", 0)
                over_odds = pred.get("over_odds", -110)
                under_odds = pred.get("under_odds", -110)

                # Snapshot: initial line observation
                snapshots.append({
                    "game_id": game_id,
                    "game_date": game_date,
                    "player": pname,
                    "prop_type": ptype,
                    "line": line,
                    "over_odds": over_odds,
                    "under_odds": under_odds,
                    "snapshot_type": "initial",
                    "best_book": pred.get("best_book"),
                    "best_odds": pred.get("best_odds"),
                })

                # Recommendation: only log picks with positive edge
                edge = pred.get("edge_percent", 0)
                if edge > 0:
                    dk_odds = over_odds if side == "over" else under_odds
                    recommendations.append({
                        "game_id": game_id,
                        "game_date": game_date,
                        "player": pname,
                        "prop_type": ptype,
                        "side": side,
                        "our_probability": pred.get("our_probability", pred.get("p_over", 0.5)),
                        "edge": edge / 100.0,  # Store as decimal
                        "line": line,
                        "odds": dk_odds,
                        "over_odds": over_odds,
                        "under_odds": under_odds,
                        "predicted_value": pred.get("predicted_value"),
                        "confidence": pred.get("confidence"),
                        "kelly_fraction": pred.get("kelly_fraction"),
                        "recommended_units": pred.get("recommended_units"),
                        "best_book": pred.get("best_book"),
                        "best_odds": pred.get("best_odds"),
                        "model_version": model_version,
                    })

            # Bulk log to SQLite
            if snapshots:
                self.clv_tracker.log_snapshots_bulk(snapshots)
            if recommendations:
                logged = self.clv_tracker.log_recommendations_bulk(recommendations)
                if logged:
                    logger.info("CLV: logged %d recommendations for game %d", logged, game_id)

        except Exception as e:
            logger.error("CLV logging failed for game %d: %s", game_id, e)

    def capture_closing_lines(self, game_id, fresh_props_by_key, game_date=None):
        # type: (int, Dict[str, Dict], Optional[str]) -> None
        """
        Capture closing lines for a game (called near tipoff).

        This is the critical CLV data point -- the final line before the game starts.
        """
        if game_id in self._closing_lines_captured:
            return

        try:
            self.clv_tracker.log_closing_lines(game_id, fresh_props_by_key, game_date)
            self._closing_lines_captured.add(game_id)
            logger.info("CLV: captured closing lines for game %d", game_id)
        except Exception as e:
            logger.error("CLV: failed to capture closing lines for game %d: %s", game_id, e)

    def compute_clv_for_game(self, game_id, actual_results=None):
        # type: (int, Optional[Dict]) -> Optional[List[Dict]]
        """
        Compute CLV results for a completed game.

        Args:
            game_id: BDL game ID
            actual_results: {"player|prop_type": {"actual": X, "hit": bool}}
        """
        try:
            return self.clv_tracker.compute_clv_for_game(game_id, actual_results)
        except Exception as e:
            logger.error("CLV computation failed for game %d: %s", game_id, e)
            return None

    # Excluded non-betting platforms (DFS, prediction markets)
    EXCLUDED_BOOKS_SHOPPING = frozenset({
        "pick6", "prizepicks", "sleeper", "underdog", "kalshi", "polymarket",
    })

    @staticmethod
    def _american_to_implied(odds):
        # type: (int) -> float
        """Convert American odds to implied probability."""
        if odds > 0:
            return 100.0 / (odds + 100)
        return abs(odds) / (abs(odds) + 100)

    @staticmethod
    def _american_to_decimal(odds):
        # type: (int) -> float
        """Convert American odds to decimal odds."""
        if odds > 0:
            return 1.0 + odds / 100.0
        return 1.0 + 100.0 / abs(odds)

    def _compute_line_shopping(self, web_pred, best_lines):
        # type: (dict, dict) -> dict
        """
        Compute EV at each available book, detect arb/middle opportunities.

        Enriches web_pred in-place with:
          line_shopping, edge_at_dk, edge_at_best, odds_improvement,
          odds_tier, best_book, best_odds, arbitrage, middles
        """
        pname = web_pred.get("player", "")
        ptype = web_pred.get("prop_type", "")
        side = web_pred.get("side", "over")
        prop_line = web_pred.get("line", 0)

        # Model probabilities
        p_over = web_pred.get("p_over", 0.5)
        p_under = web_pred.get("p_under", 0.5)
        model_prob = p_over if side == "over" else p_under

        # Gather all book data for this player|prop from both sides
        all_books_raw = dict(web_pred.get("all_books") or {})
        over_key = "{}|{}|over".format(pname, ptype)
        under_key = "{}|{}|under".format(pname, ptype)
        bl_over = best_lines.get(over_key, {}) if best_lines else {}
        bl_under = best_lines.get(under_key, {}) if best_lines else {}

        # Merge books from best_lines if web_pred has sparse data
        if best_lines and len(all_books_raw) < 2:
            bl = bl_over or bl_under
            for bk, bd in bl.get("all_books", {}).items():
                if bk.lower() in self.EXCLUDED_BOOKS_SHOPPING:
                    continue
                if bk not in all_books_raw:
                    all_books_raw[bk] = bd

        # Build a unified per-book view: {book: {line, over_odds, under_odds}}
        # best_lines stores side-specific: over entry has "odds" = over_odds,
        # under entry has "odds" = under_odds
        unified_books = {}  # type: dict
        for bk, bd in all_books_raw.items():
            if bk.lower() in self.EXCLUDED_BOOKS_SHOPPING:
                continue
            entry = dict(bd)
            unified_books[bk] = entry

        # Supplement with over/under odds from the opposing side in best_lines
        for bk, bd in bl_over.get("all_books", {}).items():
            if bk.lower() in self.EXCLUDED_BOOKS_SHOPPING:
                continue
            if bk not in unified_books:
                unified_books[bk] = {}
            if "over_odds" not in unified_books[bk] and "odds" in bd:
                unified_books[bk]["over_odds"] = bd["odds"]
            if "line" not in unified_books[bk] and "line" in bd:
                unified_books[bk]["line"] = bd["line"]
        for bk, bd in bl_under.get("all_books", {}).items():
            if bk.lower() in self.EXCLUDED_BOOKS_SHOPPING:
                continue
            if bk not in unified_books:
                unified_books[bk] = {}
            if "under_odds" not in unified_books[bk] and "odds" in bd:
                unified_books[bk]["under_odds"] = bd["odds"]
            if "line" not in unified_books[bk] and "line" in bd:
                unified_books[bk]["line"] = bd["line"]

        # --- Line Shopping EV ---
        line_shopping = []
        dk_odds_val = web_pred.get("dk_odds", -110)
        if side == "over":
            dk_odds_val = web_pred.get("over_odds", dk_odds_val)
        else:
            dk_odds_val = web_pred.get("under_odds", dk_odds_val)

        for bk, bd in unified_books.items():
            bk_line = bd.get("line", 0)
            # Determine the relevant odds for the recommended side
            if side == "over":
                bk_odds = bd.get("over_odds", bd.get("odds", 0))
            else:
                bk_odds = bd.get("under_odds", bd.get("odds", 0))

            if not bk_odds or bk_odds == 0:
                continue
            # Same-line filter (within 0.5 pts)
            if bk_line and prop_line and abs(bk_line - prop_line) > 0.5:
                continue
            # Outlier odds filter
            if abs(bk_odds) > 500:
                continue

            decimal_odds = self._american_to_decimal(bk_odds)
            ev = model_prob * (decimal_odds - 1) - (1 - model_prob)
            line_shopping.append({
                "book": bk,
                "line": bk_line or prop_line,
                "odds": bk_odds,
                "ev": round(ev, 4),
                "is_soft_book": bk.lower() in SOFT_BOOKS,
            })

        line_shopping.sort(key=lambda x: x["ev"], reverse=True)

        # Best book / best odds
        best_book = "draftkings"
        best_odds = dk_odds_val
        if line_shopping:
            best_book = line_shopping[0]["book"]
            best_odds = line_shopping[0]["odds"]

        soft_book_has_best = best_book.lower() in SOFT_BOOKS if best_book else False

        # Edge at DK vs edge at best
        dk_implied = self._american_to_implied(dk_odds_val)
        best_implied = self._american_to_implied(best_odds) if best_odds else dk_implied
        edge_at_dk = round((model_prob - dk_implied) * 100, 2)
        edge_at_best = round((model_prob - best_implied) * 100, 2)
        odds_improvement = round(edge_at_best - edge_at_dk, 2) if best_odds != dk_odds_val else 0.0

        # Odds tier
        if best_odds and best_odds >= 110:
            odds_tier = "premium"
        elif best_odds and best_odds >= 100:
            odds_tier = "plus_money"
        elif best_odds and best_odds >= -105:
            odds_tier = "reduced_vig"
        else:
            odds_tier = "standard"

        # --- Arbitrage Detection ---
        # Check every pair of books for arb: over_implied + under_implied < 1.0
        arbitrage = []
        book_names = list(unified_books.keys())
        for i in range(len(book_names)):
            for j in range(i + 1, len(book_names)):
                bk_a, bk_b = book_names[i], book_names[j]
                bd_a, bd_b = unified_books[bk_a], unified_books[bk_b]

                # Case 1: A over, B under
                a_over = bd_a.get("over_odds", bd_a.get("odds", 0)) if side == "over" else 0
                b_under = bd_b.get("under_odds", bd_b.get("odds", 0)) if side == "under" else 0
                # Need both sides from the opposing direction too
                a_over_odds = bd_a.get("over_odds", bd_a.get("odds" if side == "over" else "", 0))
                b_under_odds = bd_b.get("under_odds", bd_b.get("odds" if side == "under" else "", 0))
                a_under_odds = bd_a.get("under_odds", 0)
                b_over_odds = bd_b.get("over_odds", 0)

                # Check A-over + B-under
                if a_over_odds and b_under_odds and a_over_odds != 0 and b_under_odds != 0:
                    imp_a_over = self._american_to_implied(int(a_over_odds))
                    imp_b_under = self._american_to_implied(int(b_under_odds))
                    total = imp_a_over + imp_b_under
                    if total < 1.0:
                        arbitrage.append({
                            "book_a": bk_a, "book_a_side": "over",
                            "book_b": bk_b, "book_b_side": "under",
                            "total_implied": round(total, 4),
                            "arb_pct": round((1.0 - total) * 100, 2),
                        })

                # Check B-over + A-under
                if b_over_odds and a_under_odds and b_over_odds != 0 and a_under_odds != 0:
                    imp_b_over = self._american_to_implied(int(b_over_odds))
                    imp_a_under = self._american_to_implied(int(a_under_odds))
                    total = imp_b_over + imp_a_under
                    if total < 1.0:
                        arbitrage.append({
                            "book_a": bk_b, "book_a_side": "over",
                            "book_b": bk_a, "book_b_side": "under",
                            "total_implied": round(total, 4),
                            "arb_pct": round((1.0 - total) * 100, 2),
                        })

        # --- Middle Detection ---
        # Find books with lines that differ by >= 1.0 from the consensus/prop_line
        middles = []
        for bk, bd in unified_books.items():
            bk_line = bd.get("line", 0)
            if not bk_line or not prop_line:
                continue
            diff = bk_line - prop_line
            if abs(diff) < 1.0:
                continue
            # Compound with model: model says OVER and book has LOWER line,
            # or model says UNDER and book has HIGHER line
            compound = (side == "over" and diff < 0) or (side == "under" and diff > 0)
            middles.append({
                "book": bk,
                "line": bk_line,
                "consensus_line": prop_line,
                "diff": round(diff, 1),
                "compound_with_model": compound,
            })
        middles.sort(key=lambda x: abs(x["diff"]), reverse=True)

        # Consensus signal from all available books
        try:
            csig = self.consensus.analyze_from_books_data(
                pname, ptype, unified_books,
                dk_line=prop_line,
                dk_over_odds=int(web_pred.get("over_odds", -110)),
                dk_under_odds=int(web_pred.get("under_odds", -110)),
            )
            if csig:
                web_pred["consensus_over_prob"] = csig.consensus_over_prob
                web_pred["consensus_under_prob"] = csig.consensus_under_prob
                web_pred["consensus_line"] = csig.consensus_line
                web_pred["dk_vs_consensus"] = round(prop_line - csig.consensus_line, 2)
                web_pred["line_range"] = csig.line_range
                web_pred["soft_book_score"] = csig.soft_book_score
                web_pred["consensus_num_books"] = csig.num_books
                web_pred["odds_spread_width"] = csig.odds_spread_width
                # Consensus edge for the recommended side
                if side == "under":
                    web_pred["consensus_edge"] = round(
                        (csig.consensus_under_prob - self._american_to_implied(
                            int(web_pred.get("under_odds", -110)))) * 100, 2)
                else:
                    web_pred["consensus_edge"] = round(
                        (csig.consensus_over_prob - self._american_to_implied(
                            int(web_pred.get("over_odds", -110)))) * 100, 2)
        except Exception as e:
            logger.debug("Consensus in line shopping failed for %s|%s: %s", pname, ptype, e)

        # Star player heuristic
        is_star = (
            (ptype == "points" and prop_line >= 20.5) or
            (ptype == "assists" and prop_line >= 6.5) or
            (ptype == "rebounds" and prop_line >= 8.5) or
            len(unified_books) >= 12
        )
        web_pred["is_star_player"] = is_star
        web_pred["soft_book_has_best"] = soft_book_has_best

        # Enrich the prediction dict
        web_pred["line_shopping"] = line_shopping[:5]
        web_pred["edge_at_dk"] = edge_at_dk
        web_pred["edge_at_best"] = edge_at_best
        web_pred["odds_improvement"] = odds_improvement
        web_pred["odds_tier"] = odds_tier
        web_pred["best_book"] = best_book
        web_pred["best_odds"] = best_odds
        web_pred["arbitrage"] = arbitrage
        web_pred["middles"] = middles

        # Update all_books to the unified filtered set
        web_pred["all_books"] = {
            bk: bd for bk, bd in unified_books.items()
            if bk in {s["book"] for s in line_shopping} or bk == "draftkings"
        }

        return web_pred

    def _detect_arbitrage(self, books):
        # type: (dict) -> list
        """Detect arb opportunities across books. Returns list of arb dicts."""
        arbs = []
        book_names = list(books.keys())
        for i in range(len(book_names)):
            for j in range(i + 1, len(book_names)):
                bk_a, bk_b = book_names[i], book_names[j]
                bd_a, bd_b = books[bk_a], books[bk_b]

                a_over = bd_a.get("over_odds", 0)
                b_under = bd_b.get("under_odds", 0)
                if a_over and b_under and a_over != 0 and b_under != 0:
                    imp = self._american_to_implied(int(a_over)) + self._american_to_implied(int(b_under))
                    if imp < 1.0:
                        arbs.append({
                            "book_a": bk_a, "book_a_side": "over",
                            "book_b": bk_b, "book_b_side": "under",
                            "total_implied": round(imp, 4),
                            "arb_pct": round((1.0 - imp) * 100, 2),
                        })

                b_over = bd_b.get("over_odds", 0)
                a_under = bd_a.get("under_odds", 0)
                if b_over and a_under and b_over != 0 and a_under != 0:
                    imp = self._american_to_implied(int(b_over)) + self._american_to_implied(int(a_under))
                    if imp < 1.0:
                        arbs.append({
                            "book_a": bk_b, "book_a_side": "over",
                            "book_b": bk_a, "book_b_side": "under",
                            "total_implied": round(imp, 4),
                            "arb_pct": round((1.0 - imp) * 100, 2),
                        })
        return arbs

    @staticmethod
    def _detect_middles(books, consensus_line, side):
        # type: (dict, float, str) -> list
        """Detect middle opportunities where books have different lines."""
        middles = []
        if not consensus_line:
            return middles
        for bk, bd in books.items():
            bk_line = bd.get("line", 0)
            if not bk_line:
                continue
            diff = bk_line - consensus_line
            if abs(diff) < 1.0:
                continue
            compound = (side == "over" and diff < 0) or (side == "under" and diff > 0)
            middles.append({
                "book": bk,
                "line": bk_line,
                "consensus_line": consensus_line,
                "diff": round(diff, 1),
                "compound_with_model": compound,
            })
        middles.sort(key=lambda x: abs(x["diff"]), reverse=True)
        return middles

    @staticmethod
    def _compute_pick_score(pred, portfolio=None):
        # type: (dict, object) -> float
        """Composite score incorporating backtest research findings.

        Weights: base edge + soft book bonus + star UNDER bonus +
        odds spread bonus + consensus agreement bonus - portfolio exposure penalty.
        """
        base_edge = abs(pred.get("edge_percent", 0))

        # Soft book bonus: historically +6-7% ROI when soft book has best odds
        soft_bonus = 2.0 if pred.get("soft_book_has_best") else 0

        # Star player UNDER bonus: stars hit unders at +2.3% ROI historically
        star_bonus = 1.5 if pred.get("is_star_player") and pred.get("side") == "under" else 0

        # Odds spread bonus: wider spread = more likely mispricing
        line_range = pred.get("line_range", 0)
        spread_bonus = min(line_range * 0.5, 2.0)

        # Consensus agreement: model + consensus both favor the same side
        consensus_agrees = False
        side = pred.get("side", "")
        if side == "under" and pred.get("consensus_under_prob", 0) > 0.55:
            consensus_agrees = True
        elif side == "over" and pred.get("consensus_over_prob", 0) > 0.55:
            consensus_agrees = True
        consensus_bonus = 1.0 if consensus_agrees else 0

        # v8 calibrated probability bonus: if v8 blended prob is strong, boost
        v8_bonus = 0.0
        v8_conf = pred.get("v8_confidence", 0)
        if v8_conf >= 0.58:
            v8_bonus = 1.5
        elif v8_conf >= 0.55:
            v8_bonus = 0.75

        # Portfolio exposure penalty: reduce score if overexposed to player/team
        exposure_penalty = 0.0
        if portfolio:
            try:
                player = pred.get("player", "")
                team = pred.get("team", "")
                bankroll = portfolio.bankroll or 1000
                if player:
                    player_pct = portfolio._player_exposure(player) / bankroll * 100
                    if player_pct > 2.0:
                        exposure_penalty += min((player_pct - 2.0) * 1.5, 3.0)
                if team:
                    team_pct = portfolio._team_exposure(team) / bankroll * 100
                    if team_pct > 5.0:
                        exposure_penalty += min((team_pct - 5.0) * 0.5, 2.0)
            except Exception:
                pass

        return (base_edge + soft_bonus + star_bonus + spread_bonus
                + consensus_bonus + v8_bonus - exposure_penalty)

    @staticmethod
    def _apply_balanced_ranking(predictions):
        # type: (list) -> list
        """Rank predictions using Strategy P filter + composite score interleaving.

        Strategy P (backtest-validated): edge 5-15%, confidence >= 0.55.
        Caps displayed edge at 15% (higher values are data artifacts).
        Filters out thin-history players (< 10 games).
        """
        STRATEGY_PROPS = {"threes", "points", "pts_rebs_asts", "steals", "blocks", "assists", "rebounds"}
        EDGE_CAP = 15.0
        MIN_CONFIDENCE = 0.55
        MIN_EDGE = 5.0
        MAX_EDGE = 15.0

        strategy = []
        other = []
        for p in predictions:
            edge = abs(p.get("edge_percent", 0))
            conf = p.get("confidence", 0)
            ptype = p.get("prop_type", "")

            # Cap displayed edge at 15% — higher values are noise
            if edge > EDGE_CAP:
                p["edge_percent"] = EDGE_CAP if p.get("edge_percent", 0) > 0 else -EDGE_CAP
                p["edge_capped"] = True

            # Strategy P: edge 5-15%, confidence >= 0.55, core props
            passes_filter = (
                MIN_EDGE <= edge <= MAX_EDGE
                and conf >= MIN_CONFIDENCE
                and ptype in STRATEGY_PROPS
            )

            if passes_filter:
                p["strategy_p"] = True
                p["composite_score"] = AnalysisEngine._compute_pick_score(p)
                strategy.append(p)
            else:
                p["strategy_p"] = False
                other.append(p)

        if not strategy:
            return predictions

        # Group by prop type, sort each by composite score
        by_prop = {}  # type: dict
        for p in strategy:
            pt = p.get("prop_type", "")
            if pt not in by_prop:
                by_prop[pt] = []
            by_prop[pt].append(p)
        for pt in by_prop:
            by_prop[pt].sort(key=lambda x: x.get("composite_score", 0), reverse=True)

        # Interleave: rotate through prop types
        prop_order = ["threes", "points", "assists", "rebounds", "steals", "blocks", "pts_rebs_asts"]
        available = [pt for pt in prop_order if pt in by_prop]
        idx = {pt: 0 for pt in available}
        balanced = []

        while True:
            added = False
            for pt in available:
                if idx[pt] < len(by_prop[pt]):
                    balanced.append(by_prop[pt][idx[pt]])
                    idx[pt] += 1
                    added = True
            if not added:
                break

        # Tag with balanced rank
        for i, p in enumerate(balanced):
            p["balanced_rank"] = i + 1

        return balanced + other

    def get_player_profile(self, player_name):
        # type: (str) -> Optional[Dict]
        """
        Get a player's full profile.

        Returns serialized PlayerProfile or None.
        """
        from web.serializers import serialize_player_profile

        profile = self.pipeline.build_player_profile(player_name)
        if not profile:
            return None

        return serialize_player_profile(profile)

    def get_game_results(self, game_id):
        # type: (int) -> Optional[Dict]
        """
        Get graded results for a finished game using PropLine.

        Finds the PropLine event_id for this BDL game, calls
        get_graded_results(), and matches results to our predictions.

        Returns:
            {
                "results": [
                    {
                        "player": str,
                        "prop_type": str,
                        "line": float,
                        "actual": float,
                        "hit": bool,
                        "resolution": str,
                        "profit": float  (per 1 unit at -110)
                    },
                    ...
                ],
                "summary": {
                    "total": int,
                    "hits": int,
                    "misses": int,
                    "pushes": int,
                    "hit_rate": float,
                    "total_profit": float,
                }
            }
        """
        if not self.use_propline:
            return None

        # Find the game to get team names
        from datetime import timedelta
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        target_game = None
        for date_str in [today, yesterday]:
            data = self.pipeline.client._make_request("games", {"dates[]": date_str})
            for game in (data.get("data", []) if data else []):
                if game.get("id") == game_id:
                    target_game = game
                    break
            if target_game:
                break

        if not target_game:
            return None

        home_team = target_game.get("home_team", {}).get("full_name", "")
        away_team = target_game.get("visitor_team", {}).get("full_name", "")

        # Find PropLine event_id
        game_date = target_game.get("date", "")[:10]
        event_id = self.propline.find_event_id(home_team, away_team, game_date)
        if not event_id:
            return None

        # Get graded results
        graded = self.propline.get_graded_results(event_id)
        if not graded:
            return None

        # Get our cached analysis to match predictions
        cached_analysis = self._analysis_cache.get(game_id)

        results = []
        for item in graded:
            resolution = item.get("resolution", "").lower()
            hit = resolution in ("win", "hit", "over_hit", "under_hit", "won")
            push = resolution in ("push", "tie")

            # Calculate profit per unit (assuming -110 standard odds)
            if push:
                profit = 0.0
            elif hit:
                profit = 0.91  # win at -110
            else:
                profit = -1.0

            results.append({
                "player": item.get("player", ""),
                "prop_type": item.get("prop_type", ""),
                "line": item.get("line", 0),
                "actual": item.get("actual_value", 0),
                "hit": hit,
                "push": push,
                "resolution": item.get("resolution", ""),
                "profit": round(profit, 2),
            })

        hits = sum(1 for r in results if r["hit"])
        pushes = sum(1 for r in results if r["push"])
        misses = len(results) - hits - pushes
        total_profit = sum(r["profit"] for r in results)
        hit_rate = (hits / len(results) * 100) if results else 0

        # Build CLV stats from line snapshots
        clv_stats = self._compute_clv_stats_for_game(game_id, results)

        response = {
            "results": results,
            "summary": {
                "total": len(results),
                "hits": hits,
                "misses": misses,
                "pushes": pushes,
                "hit_rate": round(hit_rate, 1),
                "total_profit": round(total_profit, 2),
            }
        }
        if clv_stats:
            response["clv_stats"] = clv_stats

        return response

    def _compute_clv_stats_for_game(self, game_id, results):
        # type: (int, List[Dict]) -> Optional[Dict]
        """
        Compute CLV stats for a game's results using line snapshot data.

        Reads line_snapshots_{season}.jsonl and finds the closing line
        (last snapshot) for each prop. Compares to the analysis-time line.

        Returns:
            CLV stats dict or None if no snapshot data
        """
        import json

        season = self._get_season_string()
        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        snapshot_file = os.path.join(_project_root, "outputs", "line_snapshots_{}.jsonl".format(season))

        if not os.path.exists(snapshot_file):
            return None

        # Load snapshots for this game
        game_snapshots = {}  # type: Dict[str, List[Dict]]
        try:
            with open(snapshot_file) as f:
                for line_text in f:
                    line_text = line_text.strip()
                    if not line_text:
                        continue
                    try:
                        rec = json.loads(line_text)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("game_id") != game_id:
                        continue
                    pkey = "{}|{}".format(rec.get("player", ""), rec.get("prop_type", ""))
                    if pkey not in game_snapshots:
                        game_snapshots[pkey] = []
                    game_snapshots[pkey].append(rec)
        except Exception as e:
            logger.debug("Could not read line snapshots: %s", e)
            return None

        if not game_snapshots:
            return None

        # Sort each key's snapshots by timestamp
        for pkey in game_snapshots:
            game_snapshots[pkey].sort(key=lambda r: r.get("timestamp", ""))

        # Compute CLV for each result that we have snapshot data for
        clv_values = []
        for result in results:
            player = result.get("player", "")
            prop_type = result.get("prop_type", "")
            our_line = result.get("line", 0)
            pkey = "{}|{}".format(player, prop_type)

            snaps = game_snapshots.get(pkey)
            if not snaps:
                continue

            # Closing line = last snapshot, analysis line = first "analysis" or first snapshot
            closing_line = snaps[-1].get("line", 0)
            analysis_snap = None
            for s in snaps:
                if s.get("snapshot_type") == "analysis":
                    analysis_snap = s
                    break
            analysis_line = analysis_snap.get("line", our_line) if analysis_snap else our_line

            # Determine side from prediction cache if available
            cached = self._prediction_cache.get(game_id, {})
            cached_pred = cached.get(pkey, {})
            # We don't know the side from PropLine results, so use the analysis data
            # For now, compute unsigned CLV = |closing - analysis| direction
            # Use the prediction cache to find the recommended side
            # If we don't have the side, compute raw line movement
            clv = closing_line - analysis_line
            clv_values.append(clv)

        if not clv_values:
            return None

        avg_clv = sum(clv_values) / len(clv_values)
        positive_clv = [c for c in clv_values if c > 0]
        pct_positive = len(positive_clv) / len(clv_values) if clv_values else 0

        return {
            "available": True,
            "avg_clv": round(avg_clv, 2),
            "pct_positive_clv": round(pct_positive, 2),
            "total_picks_with_clv": len(clv_values),
            "total_picks": len(results),
        }

    def recalculate_edges_for_game(self, game_id, fresh_props_by_key, best_lines=None):
        # type: (int, Dict[str, Dict], Optional[Dict]) -> List[Dict]
        """
        Fast edge recalculation using cached model outputs + fresh odds.

        The model's adjustment is FIXED for a given player/game context.
        Only odds change. This recalculates edge/kelly/side in <1ms per prop
        instead of re-running the full model (300ms/prop).

        Args:
            game_id: BDL game ID
            fresh_props_by_key: {"player|prop_type": {line, over_odds, under_odds, all_books}}
            best_lines: cross-book best lines data (optional)

        Returns:
            List of updated prediction dicts with fresh edge values
        """
        from v2.edge import EdgeCalculator
        from web.serializers import serialize_v2_prediction

        cached = self._prediction_cache.get(game_id)
        if not cached:
            return []

        calc = EdgeCalculator()
        updated = []

        for cache_key, model_out in cached.items():
            fresh = fresh_props_by_key.get(cache_key)
            if not fresh:
                continue

            # Original model outputs (FIXED)
            orig_adj = model_out["adjustment"]
            orig_q10 = model_out.get("p10", orig_adj - 3)
            orig_q90 = model_out.get("p90", orig_adj + 3)
            orig_line = model_out["line_at_analysis"]

            # Sanity check: if fresh line is wildly different from the original
            # AND from what other books show, it's likely a stale/bogus feed.
            # Use the original analysis line instead.
            fresh_line_raw = fresh.get("line", orig_line)
            other_books = fresh.get("all_books", {})
            other_lines = [
                bd.get("line", 0) for bk, bd in other_books.items()
                if bk != "draftkings" and bd.get("line", 0) > 0
            ]
            if other_lines and abs(fresh_line_raw - orig_line) > 5:
                # Check if other books agree with the original more than DK
                avg_other = sum(other_lines) / len(other_lines)
                if abs(avg_other - orig_line) < abs(fresh_line_raw - orig_line):
                    # Other books are closer to original — DK line is bogus
                    fresh["line"] = orig_line
                    logger.debug(
                        "Suspicious DK line for %s: %.1f (orig %.1f, other books avg %.1f) — using original",
                        cache_key, fresh_line_raw, orig_line, avg_other,
                    )

            # Fresh market data
            fresh_line = fresh.get("line", orig_line)
            fresh_over = fresh.get("over_odds", -110)
            fresh_under = fresh.get("under_odds", -110)
            fresh_all_books = fresh.get("all_books", model_out.get("all_books", {}))

            # Adjust for line movement: the live line is what matters —
            # that's what you'd actually bet on right now.
            line_delta = fresh_line - orig_line
            effective_adj = orig_adj - line_delta
            effective_q10 = orig_q10 - line_delta
            effective_q90 = orig_q90 - line_delta

            # Consensus line from cross-book data
            consensus_line = None
            line_range = None
            if best_lines:
                pname, ptype = cache_key.split("|", 1)
                bl_over = best_lines.get("{}|{}|over".format(pname, ptype), {})
                bl_under = best_lines.get("{}|{}|under".format(pname, ptype), {})
                bl = bl_over or bl_under
                if bl:
                    consensus_line = bl.get("consensus_line")
                    best_over_line = bl.get("best_over_line_val")
                    best_under_line = bl.get("best_under_line_val")
                    if best_over_line and best_under_line:
                        line_range = abs(best_under_line - best_over_line)

            # Recalculate edge with fresh odds
            edge_result = calc.from_adjustment(
                adjustment=effective_adj,
                q10=effective_q10,
                q90=effective_q90,
                dk_line=fresh_line,
                over_odds=int(fresh_over),
                under_odds=int(fresh_under),
                consensus_line=consensus_line,
                line_range=line_range,
            )

            pname, ptype = cache_key.split("|", 1)
            side = edge_result.get("recommended_side", "PASS").lower()

            # Compute EV at each available book
            # ONLY include books at the SAME line (±0.5) as the reference line.
            # Books with different lines are pricing a different market.
            # Exclude DFS platforms and prediction markets from line shopping recommendations
            line_shopping = []
            model_prob = edge_result.get("p_over", 0.5) if side == "over" else edge_result.get("p_under", 0.5)

            for book_key, book_data in fresh_all_books.items():
                if book_key.lower() in self.EXCLUDED_BOOKS_SHOPPING:
                    continue
                book_line = book_data.get("line", 0)
                book_odds = book_data.get("over_odds" if side == "over" else "under_odds", 0)

                # Skip books with no odds
                if not book_odds or book_odds == 0:
                    continue

                # Skip books at a different line (different market entirely)
                if book_line and fresh_line and abs(book_line - fresh_line) > 0.5:
                    continue

                # Skip extreme outlier odds
                if abs(book_odds) > 500:
                    continue

                if book_odds > 0:
                    decimal_odds = 1 + book_odds / 100
                else:
                    decimal_odds = 1 + 100 / abs(book_odds)
                ev = model_prob * (decimal_odds - 1) - (1 - model_prob)
                line_shopping.append({
                    "book": book_key,
                    "line": book_line or fresh_line,
                    "odds": book_odds,
                    "ev": round(ev, 4),
                })
            line_shopping.sort(key=lambda x: x["ev"], reverse=True)

            # Build filtered all_books dict from same books that passed line_shopping filters
            # This ensures display matches exactly what line_shopping shows
            valid_book_keys = {s["book"] for s in line_shopping}
            valid_book_keys.add("draftkings")  # always include DK
            line_shopping_books = {
                bk: bd for bk, bd in fresh_all_books.items()
                if bk in valid_book_keys
            }

            # Determine best available odds and compute edge AT those odds
            dk_odds = int(fresh_over) if side == "over" else int(fresh_under)
            best_book = "draftkings"
            best_odds = dk_odds
            if line_shopping:
                best_book = line_shopping[0]["book"]
                best_odds = line_shopping[0]["odds"]

            # Compute edge at best available odds (not just DK)
            dk_implied = self._american_to_implied(dk_odds)
            best_implied = self._american_to_implied(best_odds) if best_odds else dk_implied
            edge_at_dk = model_prob - dk_implied
            edge_at_best = model_prob - best_implied
            odds_improvement = round((edge_at_best - edge_at_dk) * 100, 2) if best_odds != dk_odds else 0

            # Odds tier: how good is the best available price?
            if best_odds and best_odds >= 110:
                odds_tier = "premium"      # Plus money — strong value
            elif best_odds and best_odds >= 100:
                odds_tier = "plus_money"   # Even money or better
            elif best_odds and best_odds >= -105:
                odds_tier = "reduced_vig"  # Better than standard -110
            else:
                odds_tier = "standard"     # Standard -110 or worse

            # Recalculate kelly and units at best odds
            if best_odds and best_odds > 0:
                best_decimal = 1 + best_odds / 100
            elif best_odds and best_odds < 0:
                best_decimal = 1 + 100 / abs(best_odds)
            else:
                best_decimal = 1.909  # -110 default
            b = best_decimal - 1
            q = 1 - model_prob
            # Standard kelly: 0.25 * (bp - q) / b
            kelly_standard = max(0, 0.25 * (b * model_prob - q) / b) if b > 0 else 0
            # Edge-based kelly: size proportional to edge when standard kelly is too conservative
            # This handles the classifier's tight probability range (50-55%)
            edge_pct = abs(edge_at_best)
            kelly_edge = 0.25 * edge_pct / 100 * 2  # 5% edge → 2.5% of bankroll
            kelly_at_best = max(kelly_standard, kelly_edge) if edge_pct > 0.02 else kelly_standard
            units_at_best = round(min(kelly_at_best * 100, 5.0), 1)

            # Compute consensus signal from all available books
            consensus_data = {}
            try:
                csig = self.consensus.analyze_from_books_data(
                    pname, ptype, fresh_all_books,
                    dk_line=fresh_line,
                    dk_over_odds=int(fresh_over),
                    dk_under_odds=int(fresh_under),
                )
                if csig:
                    consensus_data = {
                        "consensus_over_prob": csig.consensus_over_prob,
                        "consensus_under_prob": csig.consensus_under_prob,
                        "consensus_line": csig.consensus_line,
                        "consensus_edge": round(
                            (csig.consensus_under_prob - self._american_to_implied(int(fresh_under))) * 100
                            if side == "under" else
                            (csig.consensus_over_prob - self._american_to_implied(int(fresh_over))) * 100,
                            2,
                        ),
                        "dk_vs_consensus": round(fresh_line - csig.consensus_line, 2),
                        "line_range": csig.line_range,
                        "soft_book_score": csig.soft_book_score,
                        "soft_book_has_best": best_book.lower() in SOFT_BOOKS if best_book else False,
                        "consensus_num_books": csig.num_books,
                        "odds_spread_width": csig.odds_spread_width,
                    }
            except Exception as e:
                logger.debug("Consensus computation failed for %s: %s", cache_key, e)

            # Star player heuristic: high line or appears in many books
            is_star = (
                (ptype == "points" and fresh_line >= 20.5) or
                (ptype == "assists" and fresh_line >= 6.5) or
                (ptype == "rebounds" and fresh_line >= 8.5) or
                len(fresh_all_books) >= 12
            )

            pred = {
                "player": pname,
                "prop_type": ptype,
                "side": side,
                "line": fresh_line,
                "line_at_analysis": orig_line,
                "line_moved": round(line_delta, 1) if line_delta != 0 else 0,
                "dk_odds": dk_odds,
                "over_odds": int(fresh_over),
                "under_odds": int(fresh_under),
                "predicted_value": round(fresh_line + effective_adj, 2),
                "predicted_adjustment": round(effective_adj, 3),
                "p_over": round(edge_result.get("p_over", 0.5), 4),
                "p_under": round(edge_result.get("p_under", 0.5), 4),
                "edge_percent": round(edge_at_best * 100, 2),  # Edge at BEST odds
                "edge_at_dk": round(edge_at_dk * 100, 2),
                "odds_improvement": odds_improvement,
                "our_probability": round(model_prob, 4),
                "implied_probability": round(best_implied, 4),
                "confidence": model_out.get("confidence", 0),
                "confidence_stars": model_out.get("confidence_stars", 1),
                "kelly_fraction": round(kelly_at_best, 5),
                "recommended_units": units_at_best,
                "sigma": round(edge_result.get("sigma", 3.0), 3),
                "all_books": {bk: bd for bk, bd in line_shopping_books.items()},
                "best_book": best_book,
                "best_odds": best_odds,
                "odds_tier": odds_tier,
                "line_shopping": line_shopping[:5],
                "cross_book_signal": edge_result.get("cross_book_signal", "NEUTRAL"),
                "arbitrage": self._detect_arbitrage(line_shopping_books),
                "middles": self._detect_middles(line_shopping_books, fresh_line, side),
                "is_star_player": is_star,
                "source": "v2_realtime_recalc",
            }
            pred.update(consensus_data)

            # Overlay v8 calibrated classifier probabilities
            if self.use_v8 and self.v8_predictor:
                try:
                    game_date = datetime.now().strftime("%Y-%m-%d")
                    v8_result = self.v8_predictor.predict(
                        pname, ptype, game_date,
                        {"line": fresh_line, "over_odds": int(fresh_over),
                         "under_odds": int(fresh_under), "all_books": fresh_all_books}
                    )
                    if v8_result:
                        pred["v8_p_over"] = v8_result["p_over"]
                        pred["v8_p_under"] = v8_result["p_under"]
                        pred["v8_model_prob"] = v8_result["model_prob"]
                        pred["v8_consensus_prob"] = v8_result["consensus_prob"]
                        pred["v8_blended_prob"] = v8_result["blended_prob"]
                        pred["v8_edge"] = v8_result["edge"]
                        pred["v8_side"] = v8_result["side"]
                        pred["v8_kelly"] = v8_result["kelly_fraction"]
                        pred["v8_recommended_bet_pct"] = v8_result["recommended_bet_pct"]
                        pred["v8_confidence"] = v8_result["confidence"]
                        pred["edge_percent"] = round(v8_result["edge"] * 100, 1)
                        pred["confidence"] = v8_result["confidence"]
                        pred["side"] = v8_result["side"]
                        pred["p_over"] = v8_result["p_over"]
                        pred["p_under"] = v8_result["p_under"]
                        pred["our_probability"] = round(
                            max(v8_result["p_over"], v8_result["p_under"]), 4)
                except Exception as e:
                    logger.debug("v8 recalc failed for %s: %s", cache_key, e)

            # Add account health recommendation per book
            if self.account_health and pred.get("line_shopping"):
                try:
                    for shop_entry in pred["line_shopping"]:
                        book = shop_entry.get("book", "")
                        health = self.account_health.get_book_health(book)
                        if health and health.get("status") != "unknown":
                            shop_entry["health_risk"] = health.get("limiting_risk_score", 0)
                            shop_entry["health_status"] = health.get("status", "healthy")
                except Exception:
                    pass

            # Cap displayed edge at 15% — higher values are data artifacts
            raw_edge = pred["edge_percent"]
            if abs(raw_edge) > 15.0:
                pred["edge_percent"] = 15.0 if raw_edge > 0 else -15.0
                pred["edge_capped"] = True

            # Strategy P tag: edge 5-15%, confidence >= 0.55
            edge_abs = abs(pred["edge_percent"])
            pred["strategy_p"] = (
                5.0 <= edge_abs <= 15.0
                and pred.get("confidence", 0) >= 0.55
                and ptype in {"threes", "points", "pts_rebs_asts", "steals",
                              "blocks", "assists", "rebounds"}
            )

            # Tag with empirical filter tier
            if self.pick_filter:
                try:
                    filt = self.pick_filter.should_show(
                        side, ptype, abs(pred["edge_percent"]))
                    pred["filter_tier"] = filt["tier"]
                    pred["filter_reason"] = filt["reason"]
                except Exception:
                    pred["filter_tier"] = "marginal"
                    pred["filter_reason"] = ""

            updated.append(pred)

        return updated

    def get_live_predictions(self, date_str=None):
        # type: (Optional[str]) -> Dict
        """
        Real-time predictions: recalculate edge for all cached predictions
        using fresh odds from PropLine. Called every 30s from frontend.

        Unlike get_live_lines() which returns raw odds, this returns
        fully updated predictions with current edge, kelly, and line shopping.
        """
        import time as _time

        if not self.use_propline:
            return {"status": "propline_unavailable"}

        if not self._prediction_cache:
            return {"status": "not_analyzed", "message": "No games analyzed yet"}

        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        t0 = _time.time()

        # Clear PropLine cache for fresh data
        for key in list(self.propline._cache.keys()):
            if key.startswith("all_props:"):
                del self.propline._cache[key]

        # Fetch fresh DK props (single bulk call)
        current_props = self.propline.get_all_props(
            bookmaker="draftkings", date_str=date_str,
        )

        # Detect line movements vs previous scan
        movements = self._detect_line_movements(current_props)

        # Fetch all-books data for cross-book comparison
        best_lines = None
        try:
            best_lines_data = self.propline.get_best_lines_all_books(date_str=date_str)
            if best_lines_data:
                best_lines = {}
                for event_id, event_props in best_lines_data.items():
                    for prop in event_props:
                        pname = prop.get("player_name", "")
                        ptype = prop.get("prop_type", "")
                        for side in ["over", "under"]:
                            lk = "{}|{}|{}".format(pname, ptype, side)
                            best_lines[lk] = prop
        except Exception as e:
            logger.debug("Could not fetch all-books data: %s", e)

        # Build fresh props lookup from DK data
        # Key: "player|prop_type" -> {line, over_odds, under_odds, all_books}
        fresh_by_key = {}
        for event_id, props in current_props.items():
            for prop in props:
                pname = prop.get("player_name", "")
                ptype = prop.get("prop_type", "")
                if not pname or not ptype:
                    continue
                pkey = "{}|{}".format(pname, ptype)

                # Merge all-books data if available
                all_books = {"draftkings": {
                    "line": prop.get("line"),
                    "over_odds": prop.get("over_odds"),
                    "under_odds": prop.get("under_odds"),
                }}
                if best_lines:
                    bl = best_lines.get("{}|{}|over".format(pname, ptype), {})
                    if bl.get("all_books"):
                        all_books = bl["all_books"]

                fresh_by_key[pkey] = {
                    "line": prop.get("line", 0),
                    "over_odds": prop.get("over_odds", -110),
                    "under_odds": prop.get("under_odds", -110),
                    "all_books": all_books,
                }

        # Log line snapshots for props that changed (poll type, deduped)
        for game_id in self._prediction_cache:
            # Build subset of fresh props relevant to this game's cached predictions
            cached_keys = set(self._prediction_cache[game_id].keys())
            game_fresh = {k: v for k, v in fresh_by_key.items() if k in cached_keys}
            if game_fresh:
                self._log_line_snapshot(
                    game_id, game_fresh, "poll", best_lines=best_lines,
                )
                # Also log to SQLite CLV tracker (for movement snapshots)
                try:
                    today = datetime.now().strftime("%Y-%m-%d")
                    clv_snaps = []
                    for pkey, pdata in game_fresh.items():
                        parts = pkey.split("|", 1)
                        if len(parts) != 2:
                            continue
                        clv_snaps.append({
                            "game_id": game_id,
                            "game_date": today,
                            "player": parts[0],
                            "prop_type": parts[1],
                            "line": pdata.get("line", 0),
                            "over_odds": pdata.get("over_odds", -110),
                            "under_odds": pdata.get("under_odds", -110),
                            "snapshot_type": "movement",
                        })
                    if clv_snaps:
                        self.clv_tracker.log_snapshots_bulk(clv_snaps)
                except Exception as e:
                    logger.debug("CLV poll snapshot failed: %s", e)

        # Recalculate edges for each cached game
        games_result = {}
        total_preds = 0
        live_game_count = 0

        for game_id, cached_preds in self._prediction_cache.items():
            updated = self.recalculate_edges_for_game(
                game_id, fresh_by_key, best_lines,
            )
            if not updated:
                continue

            game_entry = {
                "predictions": updated,
                "count": len(updated),
            }

            # Check if game is in progress and add live projections
            if self.live_tracker:
                try:
                    game_status = self.live_tracker.get_game_status(game_id)
                    is_live = game_status and (
                        game_status.get("is_in_progress", False) or
                        game_status.get("raw", "") in (
                            "1st Qtr", "2nd Qtr", "3rd Qtr", "4th Qtr",
                            "Halftime", "OT", "In Progress",
                        )
                    )
                    if is_live:
                        # Capture closing lines when game goes live
                        # (the last pre-tipoff lines are the closing lines)
                        game_fresh = {k: v for k, v in fresh_by_key.items()
                                      if k in set(cached_preds.keys())}
                        if game_fresh:
                            self.capture_closing_lines(game_id, game_fresh)

                        # Build sigma map from prediction cache
                        sigma_map = {}
                        for ck, cv in cached_preds.items():
                            sigma_map[ck] = cv.get("sigma", 3.0)

                        live_preds = self.live_tracker.get_live_predictions(
                            game_id, cached_preds, sigma_map,
                        )
                        if live_preds:
                            # Merge live data into the recalculated predictions
                            live_lookup = {}
                            for lp in live_preds:
                                lk = "{}|{}".format(lp["player"], lp["prop_type"])
                                live_lookup[lk] = lp

                            for pred in game_entry["predictions"]:
                                lk = "{}|{}".format(pred["player"], pred["prop_type"])
                                live = live_lookup.get(lk)
                                if live:
                                    pred["live_game"] = {
                                        "current_stat": live["current_stat"],
                                        "minutes_played": live["minutes_played"],
                                        "projected_final": live["projected_final"],
                                        "live_p_over": live["live_p_over"],
                                        "live_p_under": live["live_p_under"],
                                        "progress_pct": live["progress_pct"],
                                        "on_pace": live["on_pace"],
                                        "events": live.get("events", []),
                                        "game_status": game_status,
                                    }

                                    # Override pre-game probabilities with live ones
                                    # The live tracker uses actual game pace — much
                                    # more accurate than the pre-game classifier once
                                    # the game is underway.
                                    live_p_over = live["live_p_over"]
                                    live_p_under = live["live_p_under"]
                                    pred["p_over"] = round(live_p_over, 4)
                                    pred["p_under"] = round(live_p_under, 4)
                                    pred["our_probability"] = round(
                                        max(live_p_over, live_p_under), 4
                                    )
                                    pred["confidence"] = round(
                                        max(live_p_over, live_p_under), 3
                                    )

                                    # Flip side if live data contradicts pre-game
                                    if live_p_over > live_p_under and pred["side"] == "under":
                                        pred["side"] = "over"
                                    elif live_p_under > live_p_over and pred["side"] == "over":
                                        pred["side"] = "under"

                                    # Recalculate edge with live probability
                                    line = pred.get("line", 0)
                                    dk_odds = pred.get("dk_odds", -110)
                                    side = pred["side"]
                                    model_prob = live_p_over if side == "over" else live_p_under
                                    if dk_odds > 0:
                                        implied = 100.0 / (dk_odds + 100)
                                    else:
                                        implied = abs(dk_odds) / (abs(dk_odds) + 100)
                                    pred["edge_percent"] = round(
                                        (model_prob - implied) * 100, 2
                                    )
                                    pred["source"] = "v3_live_override"

                            game_entry["game_status"] = "in_progress"
                            game_entry["game_info"] = game_status
                            live_game_count += 1

                            # Log live predictions for accuracy tracking
                            self._log_live_predictions(
                                game_id, game_entry["predictions"], game_status,
                            )
                except Exception as e:
                    logger.debug("Live game tracking failed for %s: %s", game_id, e)

            games_result[str(game_id)] = game_entry
            total_preds += game_entry["count"]

        # Send push alerts for high-edge predictions across all games
        if self.alert_manager and games_result:
            try:
                all_preds = []
                for ge in games_result.values():
                    all_preds.extend(ge.get("predictions", []))
                if all_preds:
                    alerts = self.alert_manager.check_and_alert(all_preds)
                    if alerts:
                        logger.info("Sent %d real-time edge alerts", len(alerts))
            except Exception as e:
                logger.debug("Real-time push alert error: %s", e)

        elapsed = _time.time() - t0

        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "games": games_result,
            "movements": movements,
            "total_predictions": total_preds,
            "live_games": live_game_count,
            "recalc_time_ms": round(elapsed * 1000, 1),
            "source": "v2_realtime_recalc",
        }

    def get_live_lines(self, date_str=None):
        # type: (Optional[str]) -> Dict
        """
        Fast endpoint data: fetch current lines for all games, detect movements,
        and find cross-book edges. Uses a single PropLine bulk call (<1s).

        Returns dict with keys: timestamp, games, movements, cross_book_edges
        """
        if not self.use_propline:
            return {"error": "PropLine not configured"}

        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        # Bypass PropLine internal cache so we always get fresh data.
        # Remove the cached entry for today's bulk props so the client re-fetches.
        for key in list(self.propline._cache.keys()):
            if key.startswith("all_props:"):
                del self.propline._cache[key]

        # Single bulk call -- all games, all props, DraftKings
        current_props = self.propline.get_all_props(
            bookmaker="draftkings", date_str=date_str,
        )

        # Detect line movements vs previous scan
        movements = self._detect_line_movements(current_props)

        # Find cross-book edges (requires a second bulk call with all books)
        cross_book_edges = self._find_cross_book_edges(date_str)

        # Format games for the response
        games_data = self._format_live_lines(current_props)

        return {
            "timestamp": datetime.now().isoformat(),
            "games": games_data,
            "movements": movements,
            "cross_book_edges": cross_book_edges,
        }

    def _detect_line_movements(self, current_props):
        # type: (Dict[str, List[Dict]]) -> List[Dict]
        """
        Compare current lines to previous scan.
        Updates _previous_lines in place; returns list of movements.
        """
        movements = []
        now = datetime.now().isoformat()

        for event_id, props in current_props.items():
            for prop in props:
                player = prop.get("player_name", "")
                prop_type = prop.get("prop_type", "")
                line = prop.get("line", 0)
                over_odds = prop.get("over_odds")
                under_odds = prop.get("under_odds")

                key = "{}|{}".format(player, prop_type)
                prev = self._previous_lines.get(key)

                if prev and prev["line"] != line:
                    movements.append({
                        "player": player,
                        "prop_type": prop_type,
                        "old_line": prev["line"],
                        "new_line": line,
                        "direction": "up" if line > prev["line"] else "down",
                        "magnitude": round(abs(line - prev["line"]), 1),
                        "event_id": event_id,
                    })
                elif prev and over_odds is not None and prev.get("over_odds") != over_odds:
                    # Odds moved even if line didn't
                    movements.append({
                        "player": player,
                        "prop_type": prop_type,
                        "old_line": prev["line"],
                        "new_line": line,
                        "old_over_odds": prev.get("over_odds"),
                        "new_over_odds": over_odds,
                        "direction": "odds_shift",
                        "magnitude": 0,
                        "event_id": event_id,
                    })

                # Always update stored lines
                self._previous_lines[key] = {
                    "line": line,
                    "over_odds": over_odds,
                    "under_odds": under_odds,
                    "timestamp": now,
                }

        # Sort by magnitude descending so biggest moves come first
        movements.sort(key=lambda m: m.get("magnitude", 0), reverse=True)
        return movements

    def _find_cross_book_edges(self, date_str=None):
        # type: (Optional[str]) -> List[Dict]
        """
        Fetch lines from ALL bookmakers and find significant discrepancies
        where one book's line differs meaningfully from the consensus.
        """
        if not self.use_propline:
            return []

        try:
            # Clear the all-books cache to get fresh data
            for key in list(self.propline._cache.keys()):
                if "all_books" in key:
                    del self.propline._cache[key]

            # Fetch with bookmaker=None gets all available books
            all_books_data = self.propline._make_request("odds", {
                "markets": self.propline.ALL_MARKETS,
            })
            if not all_books_data:
                return []

            if date_str is None:
                date_str = datetime.now().strftime("%Y-%m-%d")

            # Collect lines per player|prop across books
            # Structure: {player|prop: {bookmaker: {line, over_odds, under_odds}}}
            book_lines = {}  # type: Dict[str, Dict[str, Dict]]

            for event in all_books_data:
                commence = event.get("commence_time", "")
                event_date = commence[:10] if commence else ""
                if event_date != date_str:
                    continue

                for bookmaker in event.get("bookmakers", []):
                    book_key = bookmaker.get("key", "unknown")
                    for market in bookmaker.get("markets", []):
                        market_key = market.get("key", "")
                        prop_type = self.propline.MARKET_MAP.get(market_key)
                        if not prop_type:
                            continue

                        player_lines = {}  # type: Dict[str, Dict]
                        for outcome in market.get("outcomes", []):
                            player_name = outcome.get("description", "")
                            direction = outcome.get("name", "")
                            odds = outcome.get("price", -110)
                            line = outcome.get("point", 0)

                            if player_name not in player_lines:
                                player_lines[player_name] = {"line": line}
                            if direction == "Over":
                                player_lines[player_name]["over_odds"] = odds
                            elif direction == "Under":
                                player_lines[player_name]["under_odds"] = odds

                        for player_name, pdata in player_lines.items():
                            if "over_odds" not in pdata or "under_odds" not in pdata:
                                continue
                            lookup_key = "{}|{}".format(player_name, prop_type)
                            if lookup_key not in book_lines:
                                book_lines[lookup_key] = {}
                            book_lines[lookup_key][book_key] = {
                                "line": float(pdata["line"]),
                                "over_odds": pdata["over_odds"],
                                "under_odds": pdata["under_odds"],
                            }

            # Find edges: where DraftKings line differs from best available
            edges = []
            for lookup_key, books in book_lines.items():
                if len(books) < 2:
                    continue
                parts = lookup_key.split("|", 1)
                if len(parts) != 2:
                    continue
                player, prop_type = parts

                dk = books.get("draftkings")
                if not dk:
                    continue

                # Compare DK line to other books
                for other_book, other_data in books.items():
                    if other_book == "draftkings":
                        continue
                    line_diff = dk["line"] - other_data["line"]
                    if abs(line_diff) >= 0.5:
                        edges.append({
                            "player": player,
                            "prop_type": prop_type,
                            "dk_line": dk["line"],
                            "dk_over_odds": dk["over_odds"],
                            "other_book": other_book,
                            "other_line": other_data["line"],
                            "line_diff": round(line_diff, 1),
                            "direction": "over" if line_diff > 0 else "under",
                        })

            # Sort by absolute line difference descending
            edges.sort(key=lambda e: abs(e["line_diff"]), reverse=True)
            # Deduplicate: keep only the largest diff per player|prop
            seen = set()  # type: set
            unique_edges = []
            for e in edges:
                key = "{}|{}".format(e["player"], e["prop_type"])
                if key not in seen:
                    seen.add(key)
                    unique_edges.append(e)

            return unique_edges[:20]  # Cap at 20 to keep response small

        except Exception as exc:
            logger.error("Error finding cross-book edges: %s", exc)
            return []

    def _format_live_lines(self, props_by_event):
        # type: (Dict[str, List[Dict]]) -> List[Dict]
        """Format PropLine bulk data into a list of game objects for the API."""
        games = []
        for event_id, props in props_by_event.items():
            # Group props by player
            player_props = {}  # type: Dict[str, List[Dict]]
            for prop in props:
                pname = prop.get("player_name", "")
                if pname not in player_props:
                    player_props[pname] = []

                # Attach stored movement info if available
                key = "{}|{}".format(pname, prop.get("prop_type", ""))
                prev = self._previous_lines.get(key, {})
                prop_entry = {
                    "prop_type": prop.get("prop_type", ""),
                    "line": prop.get("line", 0),
                    "over_odds": prop.get("over_odds"),
                    "under_odds": prop.get("under_odds"),
                    "last_updated": prev.get("timestamp", ""),
                }
                player_props[pname].append(prop_entry)

            games.append({
                "event_id": event_id,
                "player_count": len(player_props),
                "prop_count": len(props),
                "players": player_props,
            })

        return games

    # ------------------------------------------------------------------
    # Line Shopping endpoints
    # ------------------------------------------------------------------

    def get_best_lines_for_date(self, date_str=None):
        # type: (Optional[str]) -> Dict
        """
        Get best available lines across all books for all games on a date.

        Uses PropLine's get_best_lines_all_books() bulk endpoint and formats
        the result for the /api/best-lines endpoint.
        """
        if not self.use_propline:
            return {"error": "PropLine not configured"}

        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        raw = self.propline.get_best_lines_all_books(date_str=date_str)
        if not raw:
            return {
                "date": date_str,
                "total_props": 0,
                "total_books": 0,
                "books": [],
                "games": {},
            }

        all_book_names = set()  # type: set
        games = {}  # type: Dict[str, Dict]
        total_props = 0

        for event_id, props_list in raw.items():
            game_props = []
            for prop in props_list:
                all_books_dict = prop.get("all_books", {})
                book_names = list(all_books_dict.keys())
                all_book_names.update(book_names)

                dk = {
                    "line": prop.get("dk_line", 0),
                    "over_odds": prop.get("dk_over_odds", -110),
                    "under_odds": prop.get("dk_under_odds", -110),
                }

                best_over = {
                    "book": prop.get("best_over_book", "unknown"),
                    "odds": prop.get("over_odds", -110),
                    "line": prop.get("best_over_line", prop.get("line", 0)),
                }
                best_under = {
                    "book": prop.get("best_under_book", "unknown"),
                    "odds": prop.get("under_odds", -110),
                    "line": prop.get("best_under_line", prop.get("line", 0)),
                }

                game_props.append({
                    "player_name": prop.get("player_name", ""),
                    "prop_type": prop.get("prop_type", ""),
                    "consensus_line": prop.get("line", 0),
                    "best_over": best_over,
                    "best_under": best_under,
                    "dk": dk,
                    "book_count": len(book_names),
                })
                total_props += 1

            games[event_id] = {"props": game_props}

        book_list = sorted(all_book_names)

        return {
            "date": date_str,
            "total_props": total_props,
            "total_books": len(book_list),
            "books": book_list,
            "games": games,
        }

    def get_shopping_summary(self):
        # type: () -> Dict
        """
        Aggregate line shopping statistics from current session's predictions.

        Scans all analyzed games' cached results for line_shopping data and
        computes summary stats including EV improvement, book frequency,
        arbitrage/middle detection.
        """
        games_analyzed = 0
        total_picks = 0
        picks_with_better_odds = 0
        ev_improvements = []  # type: List[float]
        best_shopping_pick = None  # type: Optional[Dict]
        best_ev_advantage = 0.0
        book_freq = {}  # type: Dict[str, int]
        arbitrage_count = 0
        middle_count = 0

        # Scan prediction cache (populated by real-time recalc or initial analysis)
        for game_id, cached_preds in self._prediction_cache.items():
            games_analyzed += 1

            # Also look into analysis cache for the full v2_predictions with line_shopping
            analysis = self._analysis_cache.get(game_id)
            v2_preds = []
            if analysis and isinstance(analysis, tuple) and len(analysis) == 2:
                v2_preds = analysis[1].get("v2_predictions", [])
            elif analysis and isinstance(analysis, dict):
                v2_preds = analysis.get("v2_predictions", [])

            for pred in v2_preds:
                total_picks += 1
                player = pred.get("player", "")
                prop_type = pred.get("prop_type", "")
                side = pred.get("side", "over")
                line = pred.get("line", 0)

                # Get line shopping entries from the prediction
                shopping = pred.get("line_shopping", [])
                all_books = pred.get("all_books", {})

                # Determine model probability
                model_prob = pred.get("our_probability", 0.5)
                dk_odds = pred.get("dk_odds", -110)

                # Compute DK EV
                dk_implied = self._american_to_implied(dk_odds)
                dk_decimal = (1 / dk_implied) if dk_implied > 0 else 1.909
                dk_ev = model_prob * (dk_decimal - 1) - (1 - model_prob)

                # Find best book EV from shopping data or all_books
                best_book = pred.get("best_book", "draftkings")
                best_odds = pred.get("best_odds", dk_odds)

                if best_odds and best_odds != dk_odds:
                    best_implied = self._american_to_implied(best_odds)
                    best_decimal = (1 / best_implied) if best_implied > 0 else 1.909
                    best_ev = model_prob * (best_decimal - 1) - (1 - model_prob)
                    ev_advantage = best_ev - dk_ev

                    if ev_advantage > 0.001:
                        picks_with_better_odds += 1
                        ev_improvements.append(ev_advantage * 100)

                        if ev_advantage > best_ev_advantage:
                            best_ev_advantage = ev_advantage
                            best_shopping_pick = {
                                "player": player,
                                "prop_type": prop_type,
                                "dk_ev": round(dk_ev, 4),
                                "best_ev": round(best_ev, 4),
                                "best_book": best_book,
                                "ev_advantage": round(ev_advantage, 4),
                            }

                # Track book frequency from best_book
                if best_book:
                    book_freq[best_book] = book_freq.get(best_book, 0) + 1

                # Arbitrage detection: check if over implied + under implied < 1.0
                # across different books for the same prop
                if len(all_books) >= 2:
                    best_over_implied = 1.0
                    best_under_implied = 1.0
                    min_over_line = None
                    max_under_line = None

                    for bk, bd in all_books.items():
                        bk_over = bd.get("over_odds")
                        bk_under = bd.get("under_odds")
                        bk_line = bd.get("line", line)

                        if bk_over and bk_over != 0:
                            imp = self._american_to_implied(int(bk_over))
                            if imp < best_over_implied:
                                best_over_implied = imp
                        if bk_under and bk_under != 0:
                            imp = self._american_to_implied(int(bk_under))
                            if imp < best_under_implied:
                                best_under_implied = imp

                        # Track line differences for middle detection
                        if bk_line:
                            if min_over_line is None or bk_line < min_over_line:
                                min_over_line = bk_line
                            if max_under_line is None or bk_line > max_under_line:
                                max_under_line = bk_line

                    # Arbitrage: total implied < 100%
                    if best_over_implied + best_under_implied < 1.0:
                        arbitrage_count += 1

                    # Middle: different lines across books (over at low, under at high)
                    if (min_over_line is not None and max_under_line is not None
                            and max_under_line - min_over_line >= 1.0):
                        middle_count += 1

        avg_ev_improvement = (
            round(sum(ev_improvements) / len(ev_improvements), 2)
            if ev_improvements else 0.0
        )

        return {
            "status": "ok",
            "games_analyzed": games_analyzed,
            "total_picks": total_picks,
            "picks_with_better_odds": picks_with_better_odds,
            "avg_ev_improvement_pct": avg_ev_improvement,
            "best_shopping_pick": best_shopping_pick,
            "book_frequency": book_freq,
            "arbitrage_count": arbitrage_count,
            "middle_count": middle_count,
        }

    # -----------------------------------------------------------------
    # Injury Cascade
    # -----------------------------------------------------------------

    def get_injury_cascade_alerts(self):
        # type: () -> Dict
        """
        Get current injury cascade alerts.

        Fetches fresh injury data, runs cascade analysis, returns alerts.
        Used by GET /api/injury-cascade.
        """
        if not self.cascade_monitor:
            return {"alerts": [], "status": "unavailable", "message": "Cascade engine not loaded"}

        try:
            # Fetch all injuries via BDL API
            all_injuries = []
            team_ids = {
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

            for team_name, team_id in team_ids.items():
                try:
                    injuries = self.pipeline.client.get_injuries(team_id)
                    all_injuries.extend(injuries)
                except Exception:
                    pass

            # Process through cascade monitor
            alerts = self.cascade_monitor.process_injury_update(all_injuries)
            active_alerts = self.cascade_monitor.get_alerts()

            # Also run pre-game scan
            pregame = self.cascade_monitor.scan_pregame(injuries=all_injuries)

            return {
                "alerts": active_alerts,
                "pregame_cascades": pregame,
                "total_injuries": len(all_injuries),
                "status": "ok",
            }
        except Exception as e:
            logger.error("Cascade alert fetch failed: %s", e)
            return {"alerts": [], "pregame_cascades": [], "status": "error", "message": str(e)}

    def lookup_injury_cascade(self, player_name):
        # type: (str) -> Dict
        """
        Look up cascade impacts if a specific player sits out.
        Used by GET /api/injury-cascade/lookup?player=NAME.
        """
        if not self.cascade_monitor:
            return {"error": "Cascade engine not loaded"}
        return self.cascade_monitor.lookup_player(player_name)

    def get_cascade_for_game(self, game_id, home_injuries=None, away_injuries=None):
        # type: (int, Optional[List], Optional[List]) -> List[Dict]
        """
        Get cascade impacts relevant to a specific game.

        Called during analyze_game to enrich predictions with cascade data.
        Returns list of cascade impacts for OUT players in this game.
        """
        if not self.cascade_monitor:
            return []

        cascades = []
        all_injuries = (home_injuries or []) + (away_injuries or [])

        for inj in all_injuries:
            name = inj.get("name", "")
            status = inj.get("status", "")
            if not name or status.lower() not in ("out", "doubtful"):
                continue

            impacts = self.cascade_monitor.lookup.get_impacts(name, min_confidence="low")
            if impacts:
                cascades.append({
                    "absent_player": name,
                    "status": status,
                    "impacts": [
                        {
                            "player": i["player"],
                            "prop": i["prop"],
                            "boost_pct": i["boost_pct"],
                            "confidence": i["confidence"],
                            "sample": i["sample_without"],
                        }
                        for i in impacts[:10]
                        if abs(i["boost_pct"]) >= 5.0
                    ],
                })

        return cascades

    def clear_caches(self):
        # type: () -> None
        """Clear API, analysis, and internal caches."""
        self._analysis_cache = {}
        self._propline_bulk_cache = None
        self._previous_lines = {}
        self._snapshot_previous = {}
        if hasattr(self.pipeline.client, '_cache'):
            self.pipeline.client._cache = {}
        if hasattr(self.pipeline, 'odds_client') and hasattr(self.pipeline.odds_client, '_cache'):
            self.pipeline.odds_client._cache = {}
        if hasattr(self.pipeline, '_player_cache'):
            self.pipeline._player_cache = {}
        if self.use_propline:
            self.propline.clear_cache()
        if self.line_edge:
            self.line_edge.clear_cache()
        logger.info("All caches cleared")
