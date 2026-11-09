"""
market_scanner.py - Morning Line Scanner & Market Edge Detector

The PRIMARY edge source: market timing and line inefficiency.
Professional bettors make money by out-TIMING the market, not out-projecting it.

Key insight: Lines are LEAST efficient when first posted (~10am ET).
Sharp money corrects mispricing throughout the day.
Betting early on mispriced lines = Closing Line Value = real edge.

Combines:
  - Model edge (projection vs line)
  - Line movement tracking (which direction are sharps pushing?)
  - Cross-book comparison (when available via PropLine)
  - Timing signal (fresher lines = more opportunity)

Usage:
    from market_scanner import MarketScanner
    scanner = MarketScanner(api_key="...")
    opportunities = scanner.generate_daily_card()
"""

import os
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum

from data_pipeline import DataPipeline, BallDontLieClient, PlayerProp, GameContext
from model_engine import ModelEngine, ProjectionEngine, EdgeCalculator, PropType, STAT_MAPPING
from line_tracker import LineTracker, get_line_tracker

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

class Urgency(Enum):
    """How urgently should you bet this?"""
    BET_NOW = "BET_NOW"          # Line likely to move soon, model + sharp agree
    STRONG = "STRONG"            # Clear edge, bet within the hour
    MODERATE = "MODERATE"        # Edge exists, can wait a bit
    MONITOR = "MONITOR"          # Interesting but not actionable yet


class MovementDirection(Enum):
    """Which way is the line moving?"""
    SHARP_OVER = "sharp_over"    # Line moving up (sharps on OVER)
    SHARP_UNDER = "sharp_under"  # Line moving down (sharps on UNDER)
    STABLE = "stable"            # No significant movement
    UNKNOWN = "unknown"          # No movement data yet


@dataclass
class MarketOpportunity:
    """A single market opportunity identified by the scanner."""
    # Player & prop
    player_name: str
    prop_type: str
    game_id: int
    matchup: str

    # Line data
    current_line: float
    current_over_odds: int
    current_under_odds: int

    # Recommended side
    side: str  # "OVER" or "UNDER"

    # Model edge
    projected_value: float
    model_probability: float
    implied_probability: float
    model_edge_pct: float       # Our probability - implied probability (%)

    # Market signals
    movement_direction: str     # MovementDirection value
    line_movement: float        # How much the line has moved (points)
    opening_line: float         # What was the opening line?
    line_age_minutes: int       # How long has this line been posted?

    # Cross-book (populated when PropLine available)
    best_book: str              # Which book has the best line
    consensus_line: float       # Average line across books
    cross_book_edge: float      # Points better than consensus

    # Scoring
    combined_score: float       # Overall opportunity score
    urgency: str                # Urgency enum value
    model_agrees_with_sharp: bool  # Model and sharp money on same side?

    # Metadata
    confidence: float           # Model confidence (0-1)
    scan_timestamp: str         # When this opportunity was identified
    reasoning: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


# =============================================================================
# MARKET SCANNER
# =============================================================================

class MarketScanner:
    """
    Scans lines and identifies mispriced props using market signals.

    The scanner combines:
    1. Model projections (sanity check, not primary edge)
    2. Line movement tracking (sharp money direction)
    3. Cross-book comparison (best available number)
    4. Timing (freshness of lines)

    Into a single COMBINED SCORE that ranks opportunities.
    """

    # Scoring weights for combined score
    WEIGHT_MODEL_EDGE = 0.30       # Model projection edge
    WEIGHT_LINE_MOVEMENT = 0.25    # Sharp money signal
    WEIGHT_CROSS_BOOK = 0.25       # Cross-book edge (points better than consensus)
    WEIGHT_TIMING = 0.20           # Line freshness bonus

    # Thresholds
    MIN_MODEL_EDGE = 2.0           # Minimum model edge to consider (%)
    MIN_COMBINED_SCORE = 3.0       # Minimum combined score to report
    SIGNIFICANT_MOVEMENT = 0.5     # Line moved 0.5+ points = significant
    FRESH_LINE_MINUTES = 120       # Lines posted < 2 hours ago = fresh

    # Cross-book thresholds
    CROSS_BOOK_SIGNIFICANT = 0.5   # 0.5 points better than consensus = value
    CROSS_BOOK_MAJOR = 1.0         # 1.0 points better = major value

    def __init__(
        self,
        api_key: str = None,
        pipeline: DataPipeline = None,
        model_engine: ModelEngine = None,
        line_tracker: LineTracker = None
    ):
        """
        Initialize the scanner.

        Args:
            api_key: BallDontLie API key. If not provided, reads from env.
            pipeline: Existing DataPipeline (reuse to avoid re-init).
            model_engine: Existing ModelEngine (reuse).
            line_tracker: Existing LineTracker (reuse).
        """
        if api_key is None:
            api_key = os.getenv("BALLDONTLIE_API_KEY", "")

        if pipeline is not None:
            self.pipeline = pipeline
        else:
            self.pipeline = DataPipeline(api_key) if api_key else None

        if model_engine is not None:
            self.model_engine = model_engine
        elif self.pipeline is not None:
            self.model_engine = ModelEngine(self.pipeline)
        else:
            self.model_engine = None

        self.line_tracker = line_tracker or get_line_tracker()
        self.edge_calc = EdgeCalculator()

        # Cross-book data source (None until PropLine is integrated)
        self._propline_client = None

        # Scan history for CLV tracking
        self._scan_history: List[Dict] = []

    # -------------------------------------------------------------------------
    # Core scanning
    # -------------------------------------------------------------------------

    def scan_all_games(self) -> List[MarketOpportunity]:
        """
        Scan all today's props and return opportunities ranked by score.

        For each prop:
        1. Get current DraftKings line
        2. Run projection model for sanity check
        3. Track line movement (opening vs current)
        4. Check cross-book consensus (when available)
        5. Score the opportunity

        Returns: list of MarketOpportunity sorted by combined_score (desc)
        """
        if self.pipeline is None:
            logger.error("No pipeline configured. Set BALLDONTLIE_API_KEY.")
            return []

        opportunities = []

        # Get today's games
        games = self.pipeline.client.get_todays_games()
        if not games:
            logger.info("No games found for today.")
            return []

        logger.info(f"Scanning {len(games)} games...")

        # Get DK odds for spreads/totals
        dk_odds = self.pipeline.client.get_draftkings_game_odds()

        for game in games:
            home = game.get("home_team", {})
            away = game.get("visitor_team", {})
            home_name = home.get("full_name", "")
            away_name = away.get("full_name", "")
            game_id = game.get("id")

            if not game_id or not home_name or not away_name:
                continue

            matchup = f"{away_name} @ {home_name}"
            logger.info(f"  Scanning: {matchup}")

            try:
                game_opps = self._scan_single_game(
                    game_id, home_name, away_name, matchup, dk_odds
                )
                opportunities.extend(game_opps)
            except Exception as e:
                logger.error(f"  Error scanning {matchup}: {e}")
                continue

        # Sort by combined score (best first)
        opportunities.sort(key=lambda o: o.combined_score, reverse=True)

        # Filter by minimum score
        opportunities = [o for o in opportunities if o.combined_score >= self.MIN_COMBINED_SCORE]

        logger.info(f"Found {len(opportunities)} opportunities across all games")
        return opportunities

    def _scan_single_game(
        self,
        game_id: int,
        home_name: str,
        away_name: str,
        matchup: str,
        dk_odds: Dict
    ) -> List[MarketOpportunity]:
        """Scan a single game for opportunities."""
        opportunities = []

        # Get DK props for this game
        raw_props = self.pipeline.client.get_player_props(game_id)
        if not raw_props:
            return []

        # Build game context for projections
        context = self.pipeline.build_game_context(home_name, away_name)
        if not context:
            return []

        # Group props by player for efficient processing
        player_props = self._group_props_by_player(raw_props)

        for player_id, props in player_props.items():
            player_name = self._resolve_player_name(player_id)
            if not player_name:
                continue

            # Build player profile for projections
            try:
                profile = self.pipeline.build_player_profile(player_name)
            except Exception:
                profile = None

            for prop in props:
                prop_type = prop.get("prop_type", "")
                line = prop.get("line", 0)
                over_odds = prop.get("over_odds", -110)
                under_odds = prop.get("under_odds", -110)

                if line <= 0:
                    continue

                # 1. Track the line
                movement_info = self.line_tracker.record_line(
                    game_id=game_id,
                    player_name=player_name,
                    prop_type=prop_type,
                    line=line,
                    over_odds=over_odds,
                    under_odds=under_odds,
                    game_date=datetime.now().strftime("%Y-%m-%d")
                )

                # 2. Run projection model (if profile available)
                projection_data = self._get_projection(
                    profile, prop_type, context
                )

                # 3. Get line movement signal
                movement_signal = self._analyze_movement(
                    game_id, player_name, prop_type, movement_info
                )

                # 4. Get cross-book data (stub for now)
                cross_book = self._get_cross_book_edge(
                    player_name, prop_type, line
                )

                # 5. Score both sides and pick the better one
                opp = self._score_opportunity(
                    player_name=player_name,
                    prop_type=prop_type,
                    game_id=game_id,
                    matchup=matchup,
                    line=line,
                    over_odds=over_odds,
                    under_odds=under_odds,
                    projection_data=projection_data,
                    movement_signal=movement_signal,
                    cross_book=cross_book,
                    movement_info=movement_info
                )

                if opp is not None:
                    opportunities.append(opp)

        return opportunities

    # -------------------------------------------------------------------------
    # Projection integration
    # -------------------------------------------------------------------------

    def _get_projection(
        self,
        profile,
        prop_type_str: str,
        context: GameContext
    ) -> Optional[Dict]:
        """
        Run the projection model for a specific prop.

        Returns dict with: projected_value, probability, edge, confidence, std_dev
        """
        if profile is None or self.model_engine is None:
            return None

        # Map string prop type to PropType enum
        prop_type = self._str_to_prop_type(prop_type_str)
        if prop_type is None:
            return None

        try:
            projection = self.model_engine.create_projection(
                profile, prop_type, context
            )
            return {
                "projected_value": projection.projected_value,
                "confidence": projection.confidence,
                "std_dev": projection.std_dev,
                "projected_minutes": projection.projected_minutes,
            }
        except Exception as e:
            logger.debug(f"Projection failed for {profile.name} {prop_type_str}: {e}")
            return None

    # -------------------------------------------------------------------------
    # Line movement analysis
    # -------------------------------------------------------------------------

    def _analyze_movement(
        self,
        game_id: int,
        player_name: str,
        prop_type: str,
        movement_info: Dict
    ) -> Dict:
        """
        Analyze line movement to determine sharp money direction.

        Returns dict with: direction, magnitude, is_sharp, line_age_minutes
        """
        # Get full movement data
        full_movement = self.line_tracker.get_line_movement(
            game_id, player_name, prop_type
        )

        result = {
            "direction": MovementDirection.UNKNOWN.value,
            "magnitude": 0.0,
            "is_sharp": False,
            "opening_line": movement_info.get("opening_line", 0),
            "line_age_minutes": 0,
        }

        if full_movement is None:
            return result

        movement = full_movement.get("movement", 0)
        result["magnitude"] = abs(movement)
        result["opening_line"] = full_movement.get("opening_line", 0)

        # Determine direction
        if movement > self.SIGNIFICANT_MOVEMENT:
            result["direction"] = MovementDirection.SHARP_OVER.value
            result["is_sharp"] = True
        elif movement < -self.SIGNIFICANT_MOVEMENT:
            result["direction"] = MovementDirection.SHARP_UNDER.value
            result["is_sharp"] = True
        else:
            result["direction"] = MovementDirection.STABLE.value

        # Calculate line age
        first_seen_str = full_movement.get("first_seen", "")
        if first_seen_str:
            try:
                first_seen = datetime.fromisoformat(first_seen_str)
                age = datetime.now() - first_seen
                result["line_age_minutes"] = int(age.total_seconds() / 60)
            except (ValueError, TypeError):
                pass

        return result

    def detect_line_movements(self) -> List[Dict]:
        """
        Track line movements across all today's games.

        Returns list of significant movements with direction and magnitude.
        """
        if self.pipeline is None:
            return []

        movements = []
        games = self.pipeline.client.get_todays_games()

        for game in games:
            game_id = game.get("id")
            if not game_id:
                continue

            home = game.get("home_team", {}).get("full_name", "")
            away = game.get("visitor_team", {}).get("full_name", "")
            matchup = f"{away} @ {home}"

            game_movements = self.line_tracker.get_all_movements_for_game(game_id)
            for m in game_movements:
                if abs(m.get("movement", 0)) >= self.SIGNIFICANT_MOVEMENT:
                    m["matchup"] = matchup
                    m["game_id"] = game_id
                    movements.append(m)

        # Sort by absolute movement (biggest moves first)
        movements.sort(key=lambda m: abs(m.get("movement", 0)), reverse=True)
        return movements

    # -------------------------------------------------------------------------
    # Cross-book comparison
    # -------------------------------------------------------------------------

    def _get_cross_book_edge(
        self,
        player_name: str,
        prop_type: str,
        dk_line: float
    ) -> Dict:
        """
        Get cross-book comparison data.

        When PropLine is integrated, this will compare across
        DraftKings, FanDuel, BetMGM, Caesars, etc.

        For now, returns default values (DK only).
        """
        result = {
            "best_book": "draftkings",
            "consensus_line": dk_line,
            "cross_book_edge": 0.0,
            "books_available": ["draftkings"],
        }

        # PropLine integration point
        if self._propline_client is not None:
            try:
                result = self._fetch_cross_book_data(
                    player_name, prop_type, dk_line
                )
            except Exception as e:
                logger.debug(f"Cross-book fetch failed: {e}")

        return result

    def _fetch_cross_book_data(
        self,
        player_name: str,
        prop_type: str,
        dk_line: float
    ) -> Dict:
        """
        Fetch cross-book data from PropLine API.

        This is the integration point for when PropLine is set up.
        PropLine's bulk endpoint returns odds from multiple books in one call:
            props = propline.get_all_props(bookmaker=None)
        """
        # TODO: Integrate PropLine client when available
        # Example of what this would return:
        # {
        #     "best_book": "fanduel",
        #     "consensus_line": 25.0,
        #     "cross_book_edge": 0.5,  # DK line is 0.5 points better
        #     "books_available": ["draftkings", "fanduel", "betmgm", "caesars"],
        #     "lines_by_book": {
        #         "draftkings": {"line": 24.5, "over_odds": -110},
        #         "fanduel": {"line": 25.5, "over_odds": -112},
        #         "betmgm": {"line": 25.0, "over_odds": -108},
        #     }
        # }
        return {
            "best_book": "draftkings",
            "consensus_line": dk_line,
            "cross_book_edge": 0.0,
            "books_available": ["draftkings"],
        }

    def set_propline_client(self, client):
        """
        Attach a PropLine client for cross-book comparison.

        Call this when PropLine is integrated:
            from propline_client import PropLineClient
            scanner.set_propline_client(PropLineClient(api_key="..."))
        """
        self._propline_client = client
        logger.info("PropLine client attached for cross-book scanning")

    # -------------------------------------------------------------------------
    # Opening line value detection
    # -------------------------------------------------------------------------

    def find_opening_line_value(self) -> List[MarketOpportunity]:
        """
        Compare current lines to what they'll likely close at.

        Opening lines have ~2-3% more vig built in as insurance.
        As the day progresses, sharp money corrects mispricing.
        If our model agrees with the likely correction direction, bet early.

        Returns: opportunities where model + likely sharp direction agree
        """
        all_opps = self.scan_all_games()

        # Filter for opportunities where:
        # 1. Line is relatively fresh (< 2 hours since first seen)
        # 2. Model agrees with sharp direction OR sharp hasn't moved yet
        opening_value = []
        for opp in all_opps:
            is_fresh = opp.line_age_minutes < self.FRESH_LINE_MINUTES
            model_and_sharp_agree = opp.model_agrees_with_sharp
            no_sharp_yet = opp.movement_direction == MovementDirection.STABLE.value

            if is_fresh and (model_and_sharp_agree or no_sharp_yet):
                opp.reasoning.append("OPENING LINE VALUE: Fresh line, model has edge")
                if no_sharp_yet:
                    opp.reasoning.append("Sharp money hasn't moved this yet")
                opening_value.append(opp)

        return opening_value

    # -------------------------------------------------------------------------
    # Scoring engine
    # -------------------------------------------------------------------------

    def _score_opportunity(
        self,
        player_name: str,
        prop_type: str,
        game_id: int,
        matchup: str,
        line: float,
        over_odds: int,
        under_odds: int,
        projection_data: Optional[Dict],
        movement_signal: Dict,
        cross_book: Dict,
        movement_info: Dict
    ) -> Optional[MarketOpportunity]:
        """
        Score a prop opportunity combining all signals.

        Returns a MarketOpportunity if it clears the minimum threshold,
        or None if it doesn't.
        """
        now = datetime.now()
        reasoning = []

        # --- Model edge (for both sides) ---
        model_edge_over = 0.0
        model_edge_under = 0.0
        proj_value = 0.0
        prob_over = 0.5
        prob_under = 0.5
        confidence = 0.5

        if projection_data is not None:
            proj_value = projection_data["projected_value"]
            std_dev = projection_data["std_dev"]
            confidence = projection_data["confidence"]

            stat_type = self._prop_type_to_stat_key(prop_type)

            edge_over, edge_under, prob_over, prob_under = self.edge_calc.calculate_edge(
                proj_value, line, std_dev, over_odds, under_odds, stat_type
            )
            model_edge_over = edge_over
            model_edge_under = edge_under
        else:
            # No projection available -- use implied probability only
            implied_over = self.edge_calc.american_to_prob(over_odds)
            implied_under = self.edge_calc.american_to_prob(under_odds)
            prob_over = implied_over
            prob_under = implied_under

        # --- Determine best side ---
        # Consider model edge, movement direction, and cross-book
        over_score = 0.0
        under_score = 0.0

        # Model edge component
        over_score += max(0, model_edge_over) * self.WEIGHT_MODEL_EDGE
        under_score += max(0, model_edge_under) * self.WEIGHT_MODEL_EDGE

        # Movement component
        movement_dir = movement_signal.get("direction", MovementDirection.UNKNOWN.value)
        movement_mag = movement_signal.get("magnitude", 0)

        if movement_dir == MovementDirection.SHARP_OVER.value:
            over_score += movement_mag * 2.0 * self.WEIGHT_LINE_MOVEMENT
            reasoning.append(f"Sharp money on OVER (line moved +{movement_mag:.1f})")
        elif movement_dir == MovementDirection.SHARP_UNDER.value:
            under_score += movement_mag * 2.0 * self.WEIGHT_LINE_MOVEMENT
            reasoning.append(f"Sharp money on UNDER (line moved -{movement_mag:.1f})")

        # Cross-book component
        cb_edge = cross_book.get("cross_book_edge", 0)
        if cb_edge > 0:
            # Positive cross-book edge means DK line is better than consensus
            # This applies to the OVER side (lower line = better over)
            over_score += cb_edge * 2.0 * self.WEIGHT_CROSS_BOOK
            reasoning.append(f"DK line {cb_edge:.1f} pts better than consensus")
        elif cb_edge < 0:
            under_score += abs(cb_edge) * 2.0 * self.WEIGHT_CROSS_BOOK

        # Timing component (fresh lines get bonus)
        line_age = movement_signal.get("line_age_minutes", 0)
        if line_age > 0 and line_age < self.FRESH_LINE_MINUTES:
            freshness_bonus = (1.0 - line_age / self.FRESH_LINE_MINUTES) * 2.0
            over_score += freshness_bonus * self.WEIGHT_TIMING
            under_score += freshness_bonus * self.WEIGHT_TIMING
            reasoning.append(f"Fresh line ({line_age} min old)")

        # Pick the better side
        if over_score >= under_score:
            side = "OVER"
            combined_score = over_score
            model_edge = model_edge_over
            model_prob = prob_over
            implied = self.edge_calc.american_to_prob(over_odds)
        else:
            side = "UNDER"
            combined_score = under_score
            model_edge = model_edge_under
            model_prob = prob_under
            implied = self.edge_calc.american_to_prob(under_odds)

        # Minimum check -- need at least some model edge
        if model_edge < self.MIN_MODEL_EDGE and combined_score < self.MIN_COMBINED_SCORE:
            return None

        # Model agreement with sharp direction
        model_agrees = False
        if side == "OVER" and movement_dir == MovementDirection.SHARP_OVER.value:
            model_agrees = True
            reasoning.append("MODEL + SHARP AGREE on OVER")
        elif side == "UNDER" and movement_dir == MovementDirection.SHARP_UNDER.value:
            model_agrees = True
            reasoning.append("MODEL + SHARP AGREE on UNDER")

        # Add model reasoning
        if projection_data is not None:
            diff = proj_value - line
            if side == "OVER" and diff > 0:
                reasoning.append(f"Model projects {proj_value:.1f} vs line {line} (+{diff:.1f})")
            elif side == "UNDER" and diff < 0:
                reasoning.append(f"Model projects {proj_value:.1f} vs line {line} ({diff:.1f})")

        # Determine urgency
        urgency = self._determine_urgency(
            combined_score, model_agrees, line_age, model_edge
        )

        # Opening line data
        opening_line = movement_signal.get("opening_line", line)

        return MarketOpportunity(
            player_name=player_name,
            prop_type=prop_type,
            game_id=game_id,
            matchup=matchup,
            current_line=line,
            current_over_odds=over_odds,
            current_under_odds=under_odds,
            side=side,
            projected_value=round(proj_value, 1),
            model_probability=round(model_prob, 3),
            implied_probability=round(implied, 3),
            model_edge_pct=round(model_edge, 2),
            movement_direction=movement_dir,
            line_movement=round(line - opening_line, 1),
            opening_line=opening_line,
            line_age_minutes=line_age,
            best_book=cross_book.get("best_book", "draftkings"),
            consensus_line=cross_book.get("consensus_line", line),
            cross_book_edge=round(cb_edge, 2),
            combined_score=round(combined_score, 2),
            urgency=urgency.value,
            model_agrees_with_sharp=model_agrees,
            confidence=round(confidence, 2),
            scan_timestamp=now.isoformat(),
            reasoning=reasoning,
        )

    def _determine_urgency(
        self,
        combined_score: float,
        model_agrees_with_sharp: bool,
        line_age_minutes: int,
        model_edge: float
    ) -> Urgency:
        """Determine urgency level based on all signals."""
        # BET NOW: High score + model agrees with sharp + fresh line
        if (combined_score >= 8.0 and model_agrees_with_sharp
                and line_age_minutes < self.FRESH_LINE_MINUTES):
            return Urgency.BET_NOW

        # STRONG: High combined score or model + sharp agree
        if combined_score >= 6.0 or (model_agrees_with_sharp and model_edge >= 4.0):
            return Urgency.STRONG

        # MODERATE: Decent edge
        if combined_score >= 4.0 or model_edge >= 3.0:
            return Urgency.MODERATE

        # MONITOR: Interesting but not enough conviction
        return Urgency.MONITOR

    # -------------------------------------------------------------------------
    # Daily card generation
    # -------------------------------------------------------------------------

    def generate_daily_card(self) -> List[MarketOpportunity]:
        """
        The main output: a ranked list of today's best opportunities.

        Combines:
        - Model edge (projection vs line)
        - Line movement signal (which direction is the line going?)
        - Cross-book edge (best available line)
        - Timing (how fresh is the line?)

        Returns: list of MarketOpportunity sorted by combined score
        """
        opportunities = self.scan_all_games()

        # Log summary
        if opportunities:
            bet_now = sum(1 for o in opportunities if o.urgency == Urgency.BET_NOW.value)
            strong = sum(1 for o in opportunities if o.urgency == Urgency.STRONG.value)
            moderate = sum(1 for o in opportunities if o.urgency == Urgency.MODERATE.value)
            monitor = sum(1 for o in opportunities if o.urgency == Urgency.MONITOR.value)

            logger.info(
                f"Daily card: {len(opportunities)} opportunities "
                f"(BET_NOW: {bet_now}, STRONG: {strong}, "
                f"MODERATE: {moderate}, MONITOR: {monitor})"
            )

        # Save scan to history
        self._scan_history.append({
            "timestamp": datetime.now().isoformat(),
            "count": len(opportunities),
            "top_score": opportunities[0].combined_score if opportunities else 0,
        })

        return opportunities

    # -------------------------------------------------------------------------
    # Formatting
    # -------------------------------------------------------------------------

    def format_opportunities(self, opportunities: List[MarketOpportunity]) -> str:
        """Format opportunities for CLI output."""
        if not opportunities:
            return "No market opportunities found.\n"

        lines = []
        lines.append("")
        lines.append("=" * 74)
        lines.append("  MARKET SCANNER - LIVE OPPORTUNITIES")
        lines.append(f"  Scanned at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}")
        lines.append("=" * 74)
        lines.append("")

        # Summary counts
        bet_now = [o for o in opportunities if o.urgency == Urgency.BET_NOW.value]
        strong = [o for o in opportunities if o.urgency == Urgency.STRONG.value]
        moderate = [o for o in opportunities if o.urgency == Urgency.MODERATE.value]
        monitor = [o for o in opportunities if o.urgency == Urgency.MONITOR.value]

        lines.append(f"  Total: {len(opportunities)} opportunities")
        if bet_now:
            lines.append(f"  >>> BET NOW: {len(bet_now)}")
        if strong:
            lines.append(f"      STRONG:  {len(strong)}")
        if moderate:
            lines.append(f"      MODERATE: {len(moderate)}")
        if monitor:
            lines.append(f"      MONITOR: {len(monitor)}")
        lines.append("")

        # Group by urgency for display
        urgency_groups = [
            (">>> BET NOW <<<", bet_now),
            ("STRONG", strong),
            ("MODERATE", moderate),
            ("MONITOR", monitor),
        ]

        for label, group in urgency_groups:
            if not group:
                continue

            lines.append("-" * 74)
            lines.append(f"  [{label}]")
            lines.append("-" * 74)

            for opp in group:
                lines.append("")
                lines.append(
                    f"  {opp.player_name} {opp.side} {opp.current_line} "
                    f"{opp.prop_type.upper()}"
                )
                lines.append(
                    f"    Score: {opp.combined_score:.1f} | "
                    f"Model Edge: {opp.model_edge_pct:+.1f}% | "
                    f"Prob: {opp.model_probability*100:.0f}% vs "
                    f"Implied: {opp.implied_probability*100:.0f}%"
                )

                if opp.projected_value > 0:
                    lines.append(
                        f"    Projection: {opp.projected_value:.1f} vs Line: "
                        f"{opp.current_line}"
                    )

                if opp.line_movement != 0:
                    lines.append(
                        f"    Line Movement: {opp.line_movement:+.1f} "
                        f"(opened at {opp.opening_line})"
                    )

                if opp.cross_book_edge != 0:
                    lines.append(
                        f"    Cross-Book: {opp.cross_book_edge:+.1f} pts "
                        f"better than consensus ({opp.best_book})"
                    )

                if opp.reasoning:
                    lines.append(f"    Why: {' | '.join(opp.reasoning[:3])}")

                lines.append(
                    f"    Game: {opp.matchup} | Confidence: "
                    f"{opp.confidence:.0%}"
                )

            lines.append("")

        lines.append("=" * 74)
        lines.append(
            "  Tip: BET NOW = line likely to move, get in early. "
            "STRONG = clear value."
        )
        lines.append(
            "  MODERATE = solid but can wait. MONITOR = watch for "
            "line movement."
        )
        lines.append("=" * 74)
        lines.append("")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Serialization for web UI
    # -------------------------------------------------------------------------

    def opportunities_to_json(self, opportunities: List[MarketOpportunity]) -> str:
        """Serialize opportunities to JSON for the web dashboard."""
        data = {
            "scan_timestamp": datetime.now().isoformat(),
            "count": len(opportunities),
            "opportunities": [opp.to_dict() for opp in opportunities],
        }
        return json.dumps(data, indent=2, default=str)

    def save_opportunities(
        self,
        opportunities: List[MarketOpportunity],
        filepath: str = None
    ):
        """Save opportunities to a JSON file."""
        if filepath is None:
            os.makedirs("outputs", exist_ok=True)
            filepath = "outputs/live_opportunities.json"

        with open(filepath, "w") as f:
            f.write(self.opportunities_to_json(opportunities))

        logger.info(f"Saved {len(opportunities)} opportunities to {filepath}")

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def _group_props_by_player(self, raw_props: List[Dict]) -> Dict[int, List[Dict]]:
        """Group raw props by player_id."""
        grouped = {}
        for prop in raw_props:
            pid = prop.get("player_id")
            if pid not in grouped:
                grouped[pid] = []
            grouped[pid].append(prop)
        return grouped

    def _resolve_player_name(self, player_id: int) -> Optional[str]:
        """Resolve a player_id to a name."""
        try:
            player_data = self.pipeline.client.get_player_by_id(player_id)
            if player_data:
                return f"{player_data.get('first_name', '')} {player_data.get('last_name', '')}"
        except Exception:
            pass
        return None

    def _str_to_prop_type(self, prop_type_str: str) -> Optional[PropType]:
        """Map a string prop type to PropType enum."""
        mapping = {
            "points": PropType.POINTS,
            "rebounds": PropType.REBOUNDS,
            "assists": PropType.ASSISTS,
            "threes": PropType.THREES,
            "three_pointers": PropType.THREES,
            "3pt": PropType.THREES,
            "steals": PropType.STEALS,
            "blocks": PropType.BLOCKS,
            "turnovers": PropType.TURNOVERS,
            "pts_rebs_asts": PropType.PRA,
            "pra": PropType.PRA,
            "pts_rebs": PropType.PR,
            "pts_asts": PropType.PA,
            "rebs_asts": PropType.RA,
            "stls_blks": PropType.STEALS_BLOCKS,
            "double_double": PropType.DOUBLE_DOUBLE,
            "triple_double": PropType.TRIPLE_DOUBLE,
        }
        return mapping.get(prop_type_str.lower())

    def _prop_type_to_stat_key(self, prop_type_str: str) -> str:
        """Map a prop type string to the stat key used by EdgeCalculator."""
        mapping = {
            "points": "pts",
            "rebounds": "reb",
            "assists": "ast",
            "threes": "fg3m",
            "three_pointers": "fg3m",
            "steals": "stl",
            "blocks": "blk",
            "turnovers": "turnover",
            "pts_rebs_asts": "pra",
            "pra": "pra",
        }
        return mapping.get(prop_type_str.lower(), "pts")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def run_scan():
    """Run a market scan from the command line."""
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("BALLDONTLIE_API_KEY")
    if not api_key:
        print("Error: BALLDONTLIE_API_KEY not set")
        print("Set it with: export BALLDONTLIE_API_KEY='your_key'")
        return

    scanner = MarketScanner(api_key=api_key)
    opportunities = scanner.generate_daily_card()

    # Print to console
    print(scanner.format_opportunities(opportunities))

    # Save to file
    scanner.save_opportunities(opportunities)
    print(f"Saved to outputs/live_opportunities.json")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    run_scan()
