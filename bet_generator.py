"""
bet_generator.py - Bet Selection & Position Sizing

Generates parlay-focused recommendations per game:
- Best Singles: Top 1-2 highest edge single bets (optional)
- Safe Parlays: 2-3 leg parlays (lower risk, ~+200 to +400)
- Standard Parlays: 3-4 leg parlays (medium risk, ~+400 to +800)
- Longshot Parlays: 4-6 leg parlays (higher risk, ~+800 to +2000+)

Uses Fractional Kelly Criterion (0.25) for position sizing.
"""

import math
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum

from model_engine import (
    ModelEngine, Projection, CorrelatedPair, PropType,
    EdgeCalculator, CorrelationEngine
)
from data_pipeline import DataPipeline, GameContext, PlayerProp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

class BetTier(Enum):
    STEADY_SINGLE = "tier_1_steady"
    POPOFF_SPECIAL = "tier_2_popoff"
    CORRELATED_PARLAY = "tier_3_parlay"


class PlayerTier(Enum):
    """Player classification for parlay construction"""
    STAR = "star"           # 30+ min, high usage, top 2-3 on team
    ROTATION = "rotation"   # 20-30 min, consistent role player
    BENCH = "bench"         # <20 min or inconsistent minutes


@dataclass
class Bet:
    """A single bet recommendation"""
    tier: BetTier
    description: str
    player: str
    prop_type: str
    side: str  # "over" or "under"
    line: float
    dk_odds: int
    fair_odds: int
    edge_percent: float
    confidence: float
    recommended_units: float
    ev_per_unit: float
    reasoning: str

    # Projection breakdown
    projected_value: float = 0.0
    our_probability: float = 0.0
    implied_probability: float = 0.0

    # Player classification
    player_tier: PlayerTier = PlayerTier.ROTATION
    minutes_avg: float = 0.0

    # Detailed reasoning components
    reasoning_factors: List[str] = None

    def __post_init__(self):
        if self.reasoning_factors is None:
            self.reasoning_factors = []


@dataclass
class ParlayBet:
    """A parlay (SGP) bet recommendation"""
    tier: BetTier
    legs: List[Bet]
    combined_dk_odds: int
    true_fair_odds: int
    naive_fair_odds: int
    edge_percent: float
    recommended_units: float
    ev_per_unit: float
    correlation: float
    true_probability: float
    naive_probability: float
    reasoning: str


@dataclass
class GameRecommendations:
    """All recommendations for a single game"""
    matchup: str
    game_id: int
    timestamp: datetime
    spread: float
    total: float

    # Singles (reduced - only best plays)
    best_singles: List[Bet]

    # Parlays by risk level
    safe_parlays: List[ParlayBet]      # 2-3 legs
    standard_parlays: List[ParlayBet]  # 3-4 legs
    longshot_parlays: List[ParlayBet]  # 4-6 legs

    total_recommended_units: float
    bankroll_allocation: Dict[str, float]

    # Legacy compatibility
    @property
    def tier_1_singles(self) -> List[Bet]:
        return self.best_singles

    @property
    def tier_2_popoffs(self) -> List[Bet]:
        return []

    @property
    def tier_3_parlays(self) -> List[ParlayBet]:
        return self.safe_parlays + self.standard_parlays + self.longshot_parlays


# =============================================================================
# KELLY CRITERION
# =============================================================================

class KellyCriterion:
    """
    Fractional Kelly Criterion for position sizing.

    Full Kelly: f* = (bp - q) / b
    We use 0.25 Kelly to reduce variance.
    """

    KELLY_FRACTION = 0.25
    MAX_SINGLE_BET = 0.05  # 5% of bankroll max
    MAX_PARLAY_BET = 0.02  # 2% of bankroll max
    MIN_EDGE_SINGLE = 0.02  # 2% minimum edge
    MIN_EDGE_PARLAY = 0.03  # 3% minimum edge for parlays

    @classmethod
    def american_to_decimal(cls, odds: int) -> float:
        """Convert American odds to decimal"""
        if odds > 0:
            return 1 + (odds / 100)
        else:
            return 1 + (100 / abs(odds))

    @classmethod
    def calculate_kelly_fraction(
        cls,
        win_probability: float,
        american_odds: int
    ) -> float:
        """Calculate optimal bet size using Fractional Kelly"""
        decimal_odds = cls.american_to_decimal(american_odds)
        b = decimal_odds - 1  # Net payout per dollar
        p = win_probability
        q = 1 - p

        # Full Kelly
        full_kelly = (b * p - q) / b

        # Fractional Kelly
        fractional = full_kelly * cls.KELLY_FRACTION

        return max(0, fractional)

    @classmethod
    def calculate_units(
        cls,
        win_probability: float,
        american_odds: int,
        is_parlay: bool = False
    ) -> float:
        """Calculate recommended units (1 unit = 1% of bankroll)"""
        kelly_fraction = cls.calculate_kelly_fraction(win_probability, american_odds)
        units = kelly_fraction * 100

        max_units = (cls.MAX_PARLAY_BET if is_parlay else cls.MAX_SINGLE_BET) * 100
        return round(min(units, max_units), 2)

    @classmethod
    def calculate_ev(
        cls,
        win_probability: float,
        american_odds: int
    ) -> float:
        """Calculate expected value per unit"""
        decimal_odds = cls.american_to_decimal(american_odds)
        payout = decimal_odds - 1

        ev = (win_probability * payout) - ((1 - win_probability) * 1)
        return round(ev, 4)


# =============================================================================
# SGP JUICE CALCULATOR
# =============================================================================

class SGPJuiceCalculator:
    """
    Analyzes DraftKings SGP pricing vs true fair value.

    DK manually adjusts SGP odds because they know correlated legs
    are more likely to hit together than independent probability suggests.
    """

    # Estimated DK juice by leg count
    DK_SGP_JUICE = {
        2: 0.08,   # ~8% juice on 2-leg
        3: 0.12,   # ~12% on 3-leg
        4: 0.18,   # ~18% on 4+ legs
    }

    @classmethod
    def estimate_dk_juice(cls, num_legs: int) -> float:
        """Estimate DK's SGP juice percentage"""
        return cls.DK_SGP_JUICE.get(min(num_legs, 4), 0.20)

    @classmethod
    def calculate_true_parlay_probability(
        cls,
        leg_probabilities: List[float],
        correlations: List[float]
    ) -> float:
        """
        Calculate true parlay probability with correlation adjustment.

        P(A ∩ B) = P(A) × P(B) + ρ × σ_A × σ_B
        """
        if len(leg_probabilities) < 2:
            return leg_probabilities[0] if leg_probabilities else 0

        cumulative = leg_probabilities[0]

        for i in range(1, len(leg_probabilities)):
            p_current = leg_probabilities[i]
            corr = correlations[i-1] if i-1 < len(correlations) else 0

            sigma_prev = math.sqrt(cumulative * (1 - cumulative))
            sigma_current = math.sqrt(p_current * (1 - p_current))

            cumulative = cumulative * p_current + corr * sigma_prev * sigma_current
            cumulative = max(0.001, min(0.999, cumulative))

        return cumulative

    @classmethod
    def calculate_naive_probability(cls, leg_probabilities: List[float]) -> float:
        """Calculate parlay probability assuming independence"""
        prob = 1.0
        for p in leg_probabilities:
            prob *= p
        return prob

    @classmethod
    def probability_to_american(cls, probability: float) -> int:
        """Convert probability to American odds"""
        if probability >= 0.5:
            return int(-100 * probability / (1 - probability))
        else:
            return int(100 * (1 - probability) / probability)


# =============================================================================
# BET GENERATOR
# =============================================================================

class BetGenerator:
    """
    Main bet generator that produces parlay-focused recommendations.

    Focuses on parlays with varying risk levels:
    - Safe (2-3 legs): Lower risk, consistent value
    - Standard (3-4 legs): Medium risk, good payout
    - Longshot (4-6 legs): Higher risk, big payout potential
    """

    # Edge thresholds
    SINGLES_MIN_EDGE = 4.0      # 4% minimum edge for singles (higher bar)
    SINGLES_MAX_CV = 0.50       # Max coefficient of variation (50%)
    PARLAY_MIN_EDGE = 2.5       # 2.5% for parlays
    PARLAY_MIN_CORRELATION = 0.20

    # Props to EXCLUDE from recommendations (poor backtest performance)
    # Assists: 42.8% hit rate in backtest - terrible!
    # Threes: 49.0% - below breakeven
    EXCLUDED_PROPS = {"assists"}  # Add "threes" if you want to exclude them too

    # Bet limits
    MAX_SINGLES = 2             # Only top 2 singles
    MAX_SAFE_PARLAYS = 4        # 2-3 leg parlays
    MAX_STANDARD_PARLAYS = 4    # 3-4 leg parlays
    MAX_LONGSHOT_PARLAYS = 4    # 4-6 leg parlays

    # Legacy compatibility
    TIER_1_MIN_EDGE = SINGLES_MIN_EDGE
    TIER_1_MAX_CV = SINGLES_MAX_CV
    TIER_2_MIN_EDGE = 3.0
    TIER_3_MIN_EDGE = PARLAY_MIN_EDGE
    TIER_3_MIN_CORRELATION = PARLAY_MIN_CORRELATION
    MAX_TIER_1 = MAX_SINGLES
    MAX_TIER_2 = 0
    MAX_TIER_3 = MAX_SAFE_PARLAYS

    # Display names for props
    PROP_NAMES = {
        "points": "Points",
        "rebounds": "Rebounds",
        "assists": "Assists",
        "threes": "3-Pointers",
        "steals": "Steals",
        "blocks": "Blocks",
        "turnovers": "Turnovers",
        "pts_rebs_asts": "Pts+Rebs+Asts",
        "pts_rebs": "Pts+Rebs",
        "pts_asts": "Pts+Asts",
        "rebs_asts": "Rebs+Asts",
        "stls_blks": "Steals+Blocks",
        "double_double": "Double-Double",
        "triple_double": "Triple-Double",
    }

    def __init__(self, model_engine: ModelEngine):
        self.engine = model_engine
        self.kelly = KellyCriterion()
        self.sgp = SGPJuiceCalculator()
        self.edge_calc = EdgeCalculator()

    def _get_prop_name(self, prop_type: PropType) -> str:
        """Get display name for prop type"""
        return self.PROP_NAMES.get(prop_type.value, prop_type.value)

    def _classify_player_tier(self, proj: Projection) -> PlayerTier:
        """
        Classify player into Star/Rotation/Bench tier.

        Star: High minutes + high scoring (true stars)
        Rotation: 20-30 minutes OR high minutes but low scoring
        Bench: <20 minutes or inconsistent
        """
        minutes = proj.projected_minutes

        # For points props, use the projection directly
        # For other props, estimate scoring from line (rough proxy)
        if proj.prop_type.value == "points":
            scoring = proj.projected_value
        else:
            # Can't determine scoring from non-points props
            # Use minutes as primary indicator
            scoring = proj.dk_line * 2 if proj.prop_type.value in ["rebounds", "assists"] else 15

        # TRUE STAR: 30+ minutes AND 20+ PPG projection
        # This filters out role players who just happen to play heavy minutes
        if minutes >= 30 and scoring >= 20:
            return PlayerTier.STAR
        # HIGH MINUTES but lower scoring = rotation player getting opportunity
        elif minutes >= 28 and scoring >= 15:
            return PlayerTier.STAR
        elif minutes >= 20:
            return PlayerTier.ROTATION
        else:
            return PlayerTier.BENCH

    def _generate_reasoning(self, proj: Projection, side: str) -> tuple:
        """
        Generate detailed reasoning for a pick.

        Returns: (summary_string, list_of_factors)
        """
        factors = []

        # Player tier context
        tier = self._classify_player_tier(proj)
        if tier == PlayerTier.STAR:
            factors.append(f"Star player ({proj.projected_minutes:.0f} min)")
        elif tier == PlayerTier.ROTATION:
            factors.append(f"Rotation player ({proj.projected_minutes:.0f} min)")
        else:
            factors.append(f"Bench player ({proj.projected_minutes:.0f} min)")

        # Role change detection
        if proj.role_change == "expanded":
            factors.append("📈 Expanded role (min trending up)")
        elif proj.role_change == "reduced":
            factors.append("📉 Reduced role (min trending down)")

        # Projection vs line
        diff = proj.projected_value - proj.dk_line
        if side == "over":
            if diff > 2:
                factors.append(f"Projects {diff:.1f} above line")
            elif diff > 0:
                factors.append(f"Slight edge over line (+{diff:.1f})")
        else:
            if diff < -2:
                factors.append(f"Projects {abs(diff):.1f} below line")
            elif diff < 0:
                factors.append(f"Slight edge under line ({diff:.1f})")

        # Adjustments that moved the projection
        if abs(proj.pace_adjustment) > 0.5:
            direction = "Pace-up" if proj.pace_adjustment > 0 else "Pace-down"
            factors.append(f"{direction} game ({proj.pace_adjustment:+.1f})")

        if abs(proj.defense_adjustment) > 0.5:
            matchup = "Weak defense" if proj.defense_adjustment > 0 else "Tough defense"
            factors.append(f"{matchup} ({proj.defense_adjustment:+.1f})")

        if abs(proj.fatigue_adjustment) > 0.3:
            if proj.fatigue_adjustment < 0:
                factors.append(f"B2B fatigue ({proj.fatigue_adjustment:.1f} min)")
            else:
                factors.append(f"Extra rest (+{proj.fatigue_adjustment:.1f} min)")

        if abs(proj.blowout_adjustment) > 0.5:
            factors.append(f"Blowout risk ({proj.blowout_adjustment:.1f} min)")

        if proj.usage_boost > 0:
            factors.append(f"Injury opportunity (+{proj.usage_boost:.1f})")

        # Sample size warning
        if proj.games_sample < 5:
            factors.append(f"⚠️ Low sample ({proj.games_sample} games)")
        elif proj.games_sample < 10:
            factors.append(f"Limited data ({proj.games_sample} games)")

        # Consistency
        if proj.confidence >= 0.8:
            factors.append("Very consistent performer")
        elif proj.confidence >= 0.6:
            factors.append("Consistent performer")
        elif proj.confidence < 0.4:
            factors.append("High variance player")

        # Build summary
        summary = " | ".join(factors[:4])  # Top 4 factors for summary

        return summary, factors

    def _calculate_quality_score(self, proj: Projection, edge: float) -> float:
        """
        Calculate quality score for parlay leg selection.

        Balances: edge + volume + player tier + confidence
        """
        tier = self._classify_player_tier(proj)

        # Base score from edge
        score = edge

        # Volume multiplier (prefer higher-volume props)
        volume_mult = min(1.5, 1 + (proj.dk_line / 20))
        score *= volume_mult

        # Tier multiplier
        tier_mult = {
            PlayerTier.STAR: 1.3,
            PlayerTier.ROTATION: 1.0,
            PlayerTier.BENCH: 0.7
        }
        score *= tier_mult[tier]

        # Confidence bonus
        score *= (0.8 + proj.confidence * 0.4)

        # Penalize very low lines (too much variance)
        if proj.dk_line < 5:
            score *= 0.7
        elif proj.dk_line < 10:
            score *= 0.85

        # Penalize low sample size (less reliable projections)
        if proj.games_sample < 5:
            score *= 0.6  # Significant penalty for very low sample
        elif proj.games_sample < 10:
            score *= 0.85  # Moderate penalty

        return score

    def generate_tier_1_singles(
        self,
        projections: List[Projection]
    ) -> List[Bet]:
        """
        Generate Tier 1: Steady Singles

        Criteria:
        - Edge >= 3%
        - Low variance (consistent performer)
        - Has DK market data
        """
        bets = []

        for proj in projections:
            # Skip if no market data
            if proj.dk_line == 0:
                continue

            # Skip excluded prop types (poor backtest performance)
            if proj.prop_type.value in self.EXCLUDED_PROPS:
                continue

            # Skip pop-off plays (they go to tier 2)
            if proj.source_layer == "popoff":
                continue

            # Determine best side
            edge = max(proj.edge_over, proj.edge_under)
            side = "over" if proj.edge_over > proj.edge_under else "under"

            if edge < self.TIER_1_MIN_EDGE:
                continue

            # Check variance (coefficient of variation)
            # Use a more lenient CV filter - exclude only truly erratic players
            # CV = std_dev / mean; 50% means 1 std dev is half the average
            # Most NBA stars have CV around 30-45% for points
            if proj.projected_value > 0 and proj.std_dev > 0:
                cv = proj.std_dev / proj.projected_value
                # Skip only very high variance players (CV > 55%)
                if cv > 0.55:
                    continue

            # Calculate Kelly
            prob = proj.over_probability if side == "over" else proj.under_probability
            dk_odds = proj.dk_over_odds if side == "over" else proj.dk_under_odds
            units = self.kelly.calculate_units(prob, dk_odds)
            ev = self.kelly.calculate_ev(prob, dk_odds)
            fair_odds = self.sgp.probability_to_american(prob)

            prop_name = self._get_prop_name(proj.prop_type)

            # Get player tier and detailed reasoning
            player_tier = self._classify_player_tier(proj)
            reasoning_summary, reasoning_factors = self._generate_reasoning(proj, side)

            bet = Bet(
                tier=BetTier.STEADY_SINGLE,
                description=f"{proj.player_name} {side.upper()} {proj.dk_line} {prop_name}",
                player=proj.player_name,
                prop_type=proj.prop_type.value,
                side=side,
                line=proj.dk_line,
                dk_odds=dk_odds,
                fair_odds=fair_odds,
                edge_percent=round(edge, 2),
                confidence=proj.confidence,
                recommended_units=units,
                ev_per_unit=ev,
                reasoning=reasoning_summary,
                projected_value=proj.projected_value,
                our_probability=round(prob, 3),
                implied_probability=round(proj.implied_over_prob if side == "over" else proj.implied_under_prob, 3),
                player_tier=player_tier,
                minutes_avg=proj.projected_minutes,
                reasoning_factors=reasoning_factors
            )
            bets.append(bet)

        # Sort by edge
        bets.sort(key=lambda x: x.edge_percent, reverse=True)
        return bets

    def generate_tier_2_popoffs(
        self,
        projections: List[Projection]
    ) -> List[Bet]:
        """
        Generate Tier 2: Pop-Off Specials

        Criteria:
        - Has usage boost from injuries
        - Edge >= 3%
        """
        bets = []

        for proj in projections:
            # Only pop-off plays
            if proj.usage_boost <= 0:
                continue

            # Must have market data
            if proj.dk_line == 0:
                continue

            # Skip excluded prop types (poor backtest performance)
            if proj.prop_type.value in self.EXCLUDED_PROPS:
                continue

            # Pop-offs are always overs
            edge = proj.edge_over
            if edge < self.TIER_2_MIN_EDGE:
                continue

            # Calculate Kelly (reduced for higher variance)
            prob = proj.over_probability
            dk_odds = proj.dk_over_odds
            units = self.kelly.calculate_units(prob, dk_odds) * 0.7  # 30% reduction
            ev = self.kelly.calculate_ev(prob, dk_odds)
            fair_odds = self.sgp.probability_to_american(prob)

            prop_name = self._get_prop_name(proj.prop_type)
            bet = Bet(
                tier=BetTier.POPOFF_SPECIAL,
                description=f"POP-OFF: {proj.player_name} OVER {proj.dk_line} {prop_name}",
                player=proj.player_name,
                prop_type=proj.prop_type.value,
                side="over",
                line=proj.dk_line,
                dk_odds=dk_odds,
                fair_odds=fair_odds,
                edge_percent=round(edge, 2),
                confidence=proj.confidence * 0.8,  # Reduced confidence for pop-offs
                recommended_units=round(units, 2),
                ev_per_unit=ev,
                reasoning=f"Usage boost: +{proj.usage_boost:.1f} from injuries. "
                         f"Projection: {proj.projected_value:.1f} vs line {proj.dk_line}",
                projected_value=proj.projected_value,
                our_probability=round(prob, 3),
                implied_probability=round(proj.implied_over_prob, 3)
            )
            bets.append(bet)

        bets.sort(key=lambda x: x.edge_percent, reverse=True)
        return bets

    def generate_tier_3_parlays(
        self,
        projections: List[Projection],
        correlations: List[CorrelatedPair]
    ) -> List[ParlayBet]:
        """
        Generate Tier 3: Correlated Parlays (SGP)

        Criteria:
        - Correlation >= 0.25
        - Combined edge >= 2%
        """
        parlays = []

        for pair in correlations:
            if pair.correlation < self.TIER_3_MIN_CORRELATION:
                continue

            # Find matching projections
            proj_a = next((p for p in projections
                          if p.player_name == pair.player_a and p.prop_type == pair.prop_a), None)
            proj_b = next((p for p in projections
                          if p.player_name == pair.player_b and p.prop_type == pair.prop_b), None)

            if not proj_a or not proj_b:
                continue

            # Need market data
            if proj_a.dk_line == 0 or proj_b.dk_line == 0:
                continue

            # Determine sides
            side_a = "over" if proj_a.edge_over > proj_a.edge_under else "under"
            side_b = "over" if proj_b.edge_over > proj_b.edge_under else "under"

            prob_a = proj_a.over_probability if side_a == "over" else proj_a.under_probability
            prob_b = proj_b.over_probability if side_b == "over" else proj_b.under_probability

            # Skip if individual legs are bad (calibrated threshold)
            if prob_a < 0.55 or prob_b < 0.55:
                continue

            # SAME-TEAM SCORING CONSTRAINT
            # Don't pair two OVER scoring props from the same team
            SCORING_PROPS = {"points", "pts_rebs_asts"}
            both_over_scoring = (
                side_a == "over" and side_b == "over" and
                proj_a.prop_type.value in SCORING_PROPS and
                proj_b.prop_type.value in SCORING_PROPS and
                proj_a.team and proj_b.team and
                proj_a.team == proj_b.team
            )
            if both_over_scoring:
                continue  # Skip - unrealistic to bet OVER points on two same-team players

            # Calculate parlay probabilities
            naive_prob = self.sgp.calculate_naive_probability([prob_a, prob_b])
            true_prob = self.sgp.calculate_true_parlay_probability(
                [prob_a, prob_b],
                [pair.correlation]
            )

            # Calculate combined odds
            odds_a = proj_a.dk_over_odds if side_a == "over" else proj_a.dk_under_odds
            odds_b = proj_b.dk_over_odds if side_b == "over" else proj_b.dk_under_odds

            decimal_a = self.kelly.american_to_decimal(odds_a)
            decimal_b = self.kelly.american_to_decimal(odds_b)
            combined_decimal = decimal_a * decimal_b

            # Apply DK juice estimate
            juice = self.sgp.estimate_dk_juice(2)
            combined_decimal *= (1 - juice)

            if combined_decimal >= 2:
                combined_dk_odds = int((combined_decimal - 1) * 100)
            else:
                combined_dk_odds = int(-100 / (combined_decimal - 1))

            # Calculate edge
            dk_implied = 1 / combined_decimal
            edge = (true_prob - dk_implied) * 100

            if edge < self.TIER_3_MIN_EDGE:
                continue

            # Calculate Kelly
            units = self.kelly.calculate_units(true_prob, combined_dk_odds, is_parlay=True)
            ev = self.kelly.calculate_ev(true_prob, combined_dk_odds)

            true_fair_odds = self.sgp.probability_to_american(true_prob)
            naive_fair_odds = self.sgp.probability_to_american(naive_prob)

            # Create leg bets with tier and reasoning
            prop_name_a = self._get_prop_name(proj_a.prop_type)
            prop_name_b = self._get_prop_name(proj_b.prop_type)

            # Get tier and reasoning for each leg
            tier_a = self._classify_player_tier(proj_a)
            tier_b = self._classify_player_tier(proj_b)
            reasoning_a, factors_a = self._generate_reasoning(proj_a, side_a)
            reasoning_b, factors_b = self._generate_reasoning(proj_b, side_b)

            leg_a = Bet(
                tier=BetTier.CORRELATED_PARLAY,
                description=f"{proj_a.player_name} {side_a.upper()} {proj_a.dk_line} {prop_name_a}",
                player=proj_a.player_name,
                prop_type=proj_a.prop_type.value,
                side=side_a,
                line=proj_a.dk_line,
                dk_odds=odds_a,
                fair_odds=self.sgp.probability_to_american(prob_a),
                edge_percent=round(max(proj_a.edge_over, proj_a.edge_under), 2),
                confidence=proj_a.confidence,
                recommended_units=0,
                ev_per_unit=0,
                reasoning=reasoning_a,
                projected_value=proj_a.projected_value,
                our_probability=round(prob_a, 3),
                implied_probability=0,
                player_tier=tier_a,
                minutes_avg=proj_a.projected_minutes,
                reasoning_factors=factors_a
            )

            leg_b = Bet(
                tier=BetTier.CORRELATED_PARLAY,
                description=f"{proj_b.player_name} {side_b.upper()} {proj_b.dk_line} {prop_name_b}",
                player=proj_b.player_name,
                prop_type=proj_b.prop_type.value,
                side=side_b,
                line=proj_b.dk_line,
                dk_odds=odds_b,
                fair_odds=self.sgp.probability_to_american(prob_b),
                edge_percent=round(max(proj_b.edge_over, proj_b.edge_under), 2),
                confidence=proj_b.confidence,
                recommended_units=0,
                ev_per_unit=0,
                reasoning=reasoning_b,
                projected_value=proj_b.projected_value,
                our_probability=round(prob_b, 3),
                implied_probability=0,
                player_tier=tier_b,
                minutes_avg=proj_b.projected_minutes,
                reasoning_factors=factors_b
            )

            parlay = ParlayBet(
                tier=BetTier.CORRELATED_PARLAY,
                legs=[leg_a, leg_b],
                combined_dk_odds=combined_dk_odds,
                true_fair_odds=true_fair_odds,
                naive_fair_odds=naive_fair_odds,
                edge_percent=round(edge, 2),
                recommended_units=units,
                ev_per_unit=ev,
                correlation=pair.correlation,
                true_probability=round(true_prob, 3),
                naive_probability=round(naive_prob, 3),
                reasoning=f"Correlation: {pair.correlation:.2f}. "
                         f"True prob {true_prob*100:.1f}% vs naive {naive_prob*100:.1f}%. "
                         f"Correlation adds +{(true_prob-naive_prob)*100:.1f}% to hit rate."
            )
            parlays.append(parlay)

        parlays.sort(key=lambda x: x.edge_percent, reverse=True)
        return parlays

    def generate_multi_leg_parlays(
        self,
        projections: List[Projection],
        correlations: List[CorrelatedPair],
        min_legs: int = 2,
        max_legs: int = 6
    ) -> List[ParlayBet]:
        """
        Generate multi-leg parlays (2-6 legs) ranked by risk.

        Strategy:
        1. Get all legs with positive edge
        2. Build parlays of varying sizes (2-3 safer, 4-6 riskier)
        3. Calculate combined probability and compare to DK implied
        4. Rank by risk (probability of hitting)
        """
        multi_parlays = []

        # Get all viable legs (positive edge, has DK line)
        viable_legs = []
        for proj in projections:
            if proj.dk_line == 0:
                continue

            # Skip excluded prop types (poor backtest performance)
            if proj.prop_type.value in self.EXCLUDED_PROPS:
                continue

            best_edge = max(proj.edge_over, proj.edge_under)
            if best_edge < 3.0:  # At least 3% edge for parlay legs
                continue

            side = "over" if proj.edge_over > proj.edge_under else "under"
            prob = proj.over_probability if side == "over" else proj.under_probability

            if prob < 0.55:  # Need >55% probability for inclusion (calibrated threshold)
                continue

            # Calculate quality score and player tier
            quality_score = self._calculate_quality_score(proj, best_edge)
            player_tier = self._classify_player_tier(proj)
            reasoning_summary, reasoning_factors = self._generate_reasoning(proj, side)

            viable_legs.append({
                "proj": proj,
                "side": side,
                "prob": min(0.68, prob),  # Cap at 68% (calibrated max)
                "edge": best_edge,
                "odds": proj.dk_over_odds if side == "over" else proj.dk_under_odds,
                "quality": quality_score,
                "tier": player_tier,
                "reasoning": reasoning_summary,
                "reasoning_factors": reasoning_factors
            })

        # Sort by QUALITY SCORE (balances edge, volume, and player tier)
        viable_legs.sort(key=lambda x: x["quality"], reverse=True)

        if len(viable_legs) < min_legs:
            return []

        # Build correlation lookup
        corr_lookup = {}
        for pair in correlations:
            key = (pair.player_a, pair.prop_a, pair.player_b, pair.prop_b)
            corr_lookup[key] = pair.correlation
            key_rev = (pair.player_b, pair.prop_b, pair.player_a, pair.prop_a)
            corr_lookup[key_rev] = pair.correlation

        # Generate parlays with DIVERSITY - avoid same picks in every parlay
        # Strategy: Track usage, create themed parlays, limit repetition
        parlay_sizes = [2, 3, 4, 5, 6] if len(viable_legs) >= 6 else list(range(min_legs, min(max_legs + 1, len(viable_legs) + 1)))

        # Track how many times each leg is used across ALL parlays
        leg_usage_count = {i: 0 for i in range(len(viable_legs))}
        MAX_LEG_USAGE = 3  # Each leg can appear in max 3 parlays

        # Group legs by stat type for themed parlays
        pts_legs = [i for i, l in enumerate(viable_legs) if l["proj"].prop_type.value in ["points", "pra"]]
        reb_legs = [i for i, l in enumerate(viable_legs) if l["proj"].prop_type.value in ["rebounds", "pra"]]
        ast_legs = [i for i, l in enumerate(viable_legs) if l["proj"].prop_type.value in ["assists", "pra"]]
        threes_legs = [i for i, l in enumerate(viable_legs) if l["proj"].prop_type.value == "threes"]
        other_legs = [i for i, l in enumerate(viable_legs) if l["proj"].prop_type.value not in ["points", "rebounds", "assists", "pra", "threes"]]

        def get_available_legs(preferred_indices=None, exclude_players=set()):
            """Get legs sorted by quality, preferring less-used and preferred types"""
            available = []
            for i, leg in enumerate(viable_legs):
                if leg["proj"].player_name in exclude_players:
                    continue
                if leg_usage_count[i] >= MAX_LEG_USAGE:
                    continue
                # Score: quality bonus - usage penalty + preference bonus
                usage_penalty = leg_usage_count[i] * 0.3
                pref_bonus = 0.2 if preferred_indices and i in preferred_indices else 0
                score = leg["quality"] - usage_penalty + pref_bonus
                available.append((i, leg, score))
            available.sort(key=lambda x: x[2], reverse=True)
            return available

        def build_parlay_with_diversity(num_legs, theme=None, exclude_combos=set()):
            """Build a single parlay with diversity preferences

            KEY CONSTRAINT: Avoid same-team scoring stacking
            - Only 1 OVER points/PRA pick per team
            - Prevents unrealistic parlays like "4 Kings players all go OVER on points"
            - Cross-stat (rebounds, assists) or cross-team is fine
            """
            preferred = None
            if theme == "points":
                preferred = set(pts_legs)
            elif theme == "rebounds":
                preferred = set(reb_legs)
            elif theme == "assists":
                preferred = set(ast_legs)
            elif theme == "threes":
                preferred = set(threes_legs)

            available = get_available_legs(preferred)
            if len(available) < num_legs:
                return None

            parlay_legs = []
            used_players = set()
            used_indices = []
            has_star = False
            all_bench = True

            # Track teams with OVER scoring picks (points, PRA)
            # Only allow 1 OVER scoring pick per team
            teams_with_over_scoring = set()
            SCORING_PROPS = {"points", "pts_rebs_asts"}

            for idx, leg, score in available:
                if len(parlay_legs) >= num_legs:
                    break

                player = leg["proj"].player_name
                if player in used_players:
                    continue

                # Check same-team scoring constraint
                prop_type = leg["proj"].prop_type.value
                is_over = leg["side"] == "over"
                is_scoring_prop = prop_type in SCORING_PROPS
                team = leg["proj"].team

                if is_over and is_scoring_prop and team:
                    if team in teams_with_over_scoring:
                        # Skip - already have OVER scoring pick from this team
                        continue
                    teams_with_over_scoring.add(team)

                parlay_legs.append(leg)
                used_players.add(player)
                used_indices.append(idx)

                if leg["tier"] == PlayerTier.STAR:
                    has_star = True
                    all_bench = False
                elif leg["tier"] == PlayerTier.ROTATION:
                    all_bench = False

            if len(parlay_legs) < num_legs:
                return None

            # Skip all-bench parlays for 4+ legs (too risky)
            if all_bench and num_legs >= 4:
                return None

            # For 5+ leg parlays, require at least 1 star or rotation player
            if num_legs >= 5 and all_bench:
                return None

            # Create a signature to detect duplicate parlays
            sig = frozenset((l["proj"].player_name, l["proj"].prop_type.value, l["side"]) for l in parlay_legs)
            if sig in exclude_combos:
                return None

            # Mark these legs as used
            for idx in used_indices:
                leg_usage_count[idx] += 1

            return parlay_legs, sig

        # Build parlays with variety
        parlay_signatures = set()

        for num_legs in parlay_sizes:
            if len(viable_legs) < num_legs:
                continue

            # For each size, try different themes to get variety
            # Note: "assists" removed from themes due to poor backtest performance (42.8%)
            themes = [None, "points", "rebounds", "threes", "pra"] if num_legs <= 3 else [None, "points", "rebounds", "threes"]
            parlays_this_size = 0
            max_per_size = 4  # Max 4 parlays per size

            for theme in themes:
                if parlays_this_size >= max_per_size:
                    break

                # Try to build 2 parlays per theme for more variety
                for attempt in range(2):
                    if parlays_this_size >= max_per_size:
                        break

                    result = build_parlay_with_diversity(num_legs, theme, parlay_signatures)
                    if result is None:
                        break  # No more valid parlays for this theme

                    parlay_legs, sig = result
                    parlay_signatures.add(sig)
                    parlays_this_size += 1

                    # Track tier mix
                    has_star = any(l["tier"] == PlayerTier.STAR for l in parlay_legs)
                    all_bench = all(l["tier"] == PlayerTier.BENCH for l in parlay_legs)

                    # Calculate OUR probability (conservative - multiply and reduce slightly)
                    probs = [leg["prob"] for leg in parlay_legs]
                    naive_prob = 1.0
                    for p in probs:
                        naive_prob *= p

                    # Small correlation adjustment (only ~1% boost per correlated pair)
                    avg_corr = 0.0
                    corr_count = 0
                    for i in range(len(parlay_legs)):
                        for j in range(i + 1, len(parlay_legs)):
                            key = (
                                parlay_legs[i]["proj"].player_name,
                                parlay_legs[i]["proj"].prop_type,
                                parlay_legs[j]["proj"].player_name,
                                parlay_legs[j]["proj"].prop_type
                            )
                            if key in corr_lookup:
                                avg_corr += abs(corr_lookup[key])
                                corr_count += 1
                    if corr_count > 0:
                        avg_corr /= corr_count

                    # Conservative correlation boost (max 3% total)
                    corr_boost = min(0.03, avg_corr * 0.01 * num_legs)
                    true_prob = naive_prob + corr_boost

                    # Calculate DK parlay odds
                    combined_decimal = 1.0
                    for leg in parlay_legs:
                        decimal_odds = self.kelly.american_to_decimal(leg["odds"])
                        combined_decimal *= decimal_odds

                    # DK adds juice to parlays
                    juice = self.sgp.estimate_dk_juice(num_legs)
                    combined_decimal *= (1 - juice)

                    if combined_decimal >= 2:
                        combined_dk_odds = int((combined_decimal - 1) * 100)
                    else:
                        combined_dk_odds = int(-100 / (combined_decimal - 1)) if combined_decimal > 1 else -200

                    # Calculate DK implied probability
                    dk_implied = 1 / combined_decimal if combined_decimal > 0 else 1

                    # Edge = our probability minus DK implied
                    edge = (true_prob - dk_implied) * 100

                    # Require meaningful edge (at least 3% for parlays)
                    if edge < 3.0:
                        continue

                    # Risk level based on TRUE hit probability
                    if true_prob >= 0.40:
                        risk_level = "LOW"
                    elif true_prob >= 0.25:
                        risk_level = "MEDIUM"
                    else:
                        risk_level = "HIGH"

                    # Calculate Kelly (capped for parlays)
                    units = self.kelly.calculate_units(true_prob, combined_dk_odds, is_parlay=True)
                    ev = self.kelly.calculate_ev(true_prob, combined_dk_odds)

                    # Build leg Bet objects with tier and reasoning
                    bet_legs = []
                    for leg in parlay_legs:
                        proj = leg["proj"]
                        prop_name = self._get_prop_name(proj.prop_type)
                        bet_legs.append(Bet(
                            tier=BetTier.CORRELATED_PARLAY,
                            description=f"{proj.player_name} {leg['side'].upper()} {proj.dk_line} {prop_name}",
                            player=proj.player_name,
                            prop_type=proj.prop_type.value,
                            side=leg["side"],
                            line=proj.dk_line,
                            dk_odds=leg["odds"],
                            fair_odds=self.sgp.probability_to_american(leg["prob"]),
                            edge_percent=round(leg["edge"], 2),
                            confidence=proj.confidence,
                            recommended_units=0,
                            ev_per_unit=0,
                            reasoning=leg["reasoning"],
                            projected_value=proj.projected_value,
                            our_probability=round(leg["prob"], 3),
                            implied_probability=0,
                            player_tier=leg["tier"],
                            minutes_avg=proj.projected_minutes,
                            reasoning_factors=leg["reasoning_factors"]
                        ))

                    parlay = ParlayBet(
                        tier=BetTier.CORRELATED_PARLAY,
                        legs=bet_legs,
                        combined_dk_odds=combined_dk_odds,
                        true_fair_odds=self.sgp.probability_to_american(true_prob),
                        naive_fair_odds=self.sgp.probability_to_american(naive_prob),
                        edge_percent=round(edge, 2),
                        recommended_units=units,
                        ev_per_unit=ev,
                        correlation=avg_corr,
                        true_probability=round(true_prob, 3),
                        naive_probability=round(naive_prob, 3),
                        reasoning=f"{num_legs}-leg parlay | Risk: {risk_level} | "
                                 f"True prob {true_prob*100:.1f}% vs implied {dk_implied*100:.1f}%"
                    )
                    multi_parlays.append(parlay)

        # Remove duplicates (same legs in different order)
        seen = set()
        unique_parlays = []
        for p in multi_parlays:
            leg_key = tuple(sorted([leg.description for leg in p.legs]))
            if leg_key not in seen:
                seen.add(leg_key)
                unique_parlays.append(p)

        # Sort by risk: highest probability first (lowest risk), then by number of legs
        unique_parlays.sort(key=lambda x: (-x.true_probability, len(x.legs)))

        return unique_parlays[:12]  # Top 12 multi-leg parlays (mix of sizes)


    def generate_recommendations(
        self,
        analysis_results: Dict
    ) -> GameRecommendations:
        """
        Generate parlay-focused recommendations for a game.

        Structure:
        - Best Singles: Top 1-2 highest edge singles (optional)
        - Safe Parlays: 2-3 legs (~+200 to +400)
        - Standard Parlays: 3-4 legs (~+400 to +800)
        - Longshot Parlays: 4-6 legs (~+800 to +2000+)
        """
        projections = analysis_results.get("projections", [])
        correlations = analysis_results.get("correlations", [])

        # Generate singles (limited)
        all_singles = self.generate_tier_1_singles(projections)
        best_singles = all_singles[:self.MAX_SINGLES]

        # Generate all parlays
        two_leg_parlays = self.generate_tier_3_parlays(projections, correlations)
        multi_leg_parlays = self.generate_multi_leg_parlays(projections, correlations, min_legs=3, max_legs=6)

        # Categorize parlays by leg count
        # Safe: 2-3 legs
        safe_parlays = [p for p in two_leg_parlays if len(p.legs) == 2]
        safe_parlays.extend([p for p in multi_leg_parlays if len(p.legs) == 3])
        safe_parlays.sort(key=lambda x: x.true_probability, reverse=True)
        safe_parlays = safe_parlays[:self.MAX_SAFE_PARLAYS]

        # Standard: 3-4 legs (but prioritize 4-leg for this tier)
        standard_parlays = [p for p in multi_leg_parlays if len(p.legs) == 4]
        # Add some 3-leg if not enough 4-leg
        if len(standard_parlays) < self.MAX_STANDARD_PARLAYS:
            three_leg = [p for p in multi_leg_parlays if len(p.legs) == 3 and p not in safe_parlays]
            standard_parlays.extend(three_leg)
        standard_parlays.sort(key=lambda x: (-len(x.legs), -x.true_probability))
        standard_parlays = standard_parlays[:self.MAX_STANDARD_PARLAYS]

        # Longshot: 5-6 legs
        longshot_parlays = [p for p in multi_leg_parlays if len(p.legs) >= 5]
        longshot_parlays.sort(key=lambda x: x.edge_percent, reverse=True)
        longshot_parlays = longshot_parlays[:self.MAX_LONGSHOT_PARLAYS]

        # Calculate totals
        all_parlays = safe_parlays + standard_parlays + longshot_parlays
        total_units = (
            sum(b.recommended_units for b in best_singles) +
            sum(p.recommended_units for p in all_parlays)
        )

        allocation = {
            "best_singles": sum(b.recommended_units for b in best_singles),
            "safe_parlays": sum(p.recommended_units for p in safe_parlays),
            "standard_parlays": sum(p.recommended_units for p in standard_parlays),
            "longshot_parlays": sum(p.recommended_units for p in longshot_parlays),
        }

        return GameRecommendations(
            matchup=analysis_results.get("matchup", "Unknown"),
            game_id=analysis_results.get("game_id", 0),
            timestamp=datetime.now(),
            spread=analysis_results.get("spread", 0),
            total=analysis_results.get("total", 220),
            best_singles=best_singles,
            safe_parlays=safe_parlays,
            standard_parlays=standard_parlays,
            longshot_parlays=longshot_parlays,
            total_recommended_units=round(total_units, 2),
            bankroll_allocation=allocation
        )

    def _get_tier_label(self, bet: Bet) -> str:
        """Get a short label for player tier"""
        tier_labels = {
            PlayerTier.STAR: "[STAR]",
            PlayerTier.ROTATION: "[ROT]",
            PlayerTier.BENCH: "[BENCH]"
        }
        return tier_labels.get(bet.player_tier, "")

    def format_recommendations(self, recs: GameRecommendations) -> str:
        """Format recommendations with detailed reasoning for each pick"""
        total_parlays = len(recs.safe_parlays) + len(recs.standard_parlays) + len(recs.longshot_parlays)

        lines = [
            "",
            "=" * 70,
            f"  GAME: {recs.matchup}",
            f"  Spread: {recs.spread:+.1f} | Total: {recs.total}",
            f"  Generated: {recs.timestamp.strftime('%Y-%m-%d %H:%M')}",
            "=" * 70,
            "",
        ]

        # Best Singles with detailed reasoning
        if recs.best_singles:
            lines.append(f"TOP SINGLE BETS ({len(recs.best_singles)} picks)")
            lines.append("-" * 60)
            for bet in recs.best_singles:
                tier_label = self._get_tier_label(bet)
                lines.extend([
                    f"  {tier_label} {bet.description}",
                    f"    Line: {bet.line} | Proj: {bet.projected_value:.1f} | Edge: {bet.edge_percent:.1f}%",
                    f"    Why: {bet.reasoning}",
                    ""
                ])

        # Safe Parlays (2-3 legs) with reasoning
        lines.append(f"SAFE PARLAYS ({len(recs.safe_parlays)} picks)")
        lines.append("-" * 60)
        if recs.safe_parlays:
            for i, parlay in enumerate(recs.safe_parlays[:3], 1):  # Limit to 3
                num_legs = len(parlay.legs)
                lines.append(f"  PARLAY {i}: {num_legs}-Leg @ {parlay.combined_dk_odds:+d}")
                lines.append(f"  Win Prob: {parlay.true_probability*100:.1f}% | Edge: {parlay.edge_percent:.1f}%")
                lines.append("")
                for j, leg in enumerate(parlay.legs, 1):
                    tier_label = self._get_tier_label(leg)
                    lines.append(f"    Leg {j}: {tier_label} {leg.description}")
                    lines.append(f"           Proj: {leg.projected_value:.1f} vs Line: {leg.line}")
                    lines.append(f"           {leg.reasoning}")
                    lines.append("")
                lines.append("")
        else:
            lines.append("  No safe parlays found\n")

        # Standard Parlays (3-4 legs) with reasoning
        lines.append(f"STANDARD PARLAYS ({len(recs.standard_parlays)} picks)")
        lines.append("-" * 60)
        if recs.standard_parlays:
            for i, parlay in enumerate(recs.standard_parlays[:2], 1):  # Limit to 2
                num_legs = len(parlay.legs)
                lines.append(f"  PARLAY {i}: {num_legs}-Leg @ {parlay.combined_dk_odds:+d}")
                lines.append(f"  Win Prob: {parlay.true_probability*100:.1f}% | Edge: {parlay.edge_percent:.1f}%")
                lines.append("")
                for j, leg in enumerate(parlay.legs, 1):
                    tier_label = self._get_tier_label(leg)
                    lines.append(f"    Leg {j}: {tier_label} {leg.description}")
                    lines.append(f"           Proj: {leg.projected_value:.1f} vs Line: {leg.line}")
                    lines.append(f"           {leg.reasoning}")
                    lines.append("")
                lines.append("")
        else:
            lines.append("  No standard parlays found\n")

        # Longshot Parlays (5-6 legs) with reasoning
        lines.append(f"LONGSHOT PARLAYS ({len(recs.longshot_parlays)} picks)")
        lines.append("-" * 60)
        if recs.longshot_parlays:
            for i, parlay in enumerate(recs.longshot_parlays[:2], 1):  # Limit to 2
                num_legs = len(parlay.legs)
                lines.append(f"  PARLAY {i}: {num_legs}-Leg @ {parlay.combined_dk_odds:+d}")
                lines.append(f"  Win Prob: {parlay.true_probability*100:.1f}% | Edge: {parlay.edge_percent:.1f}%")
                lines.append("")
                for j, leg in enumerate(parlay.legs, 1):
                    tier_label = self._get_tier_label(leg)
                    lines.append(f"    Leg {j}: {tier_label} {leg.description}")
                    lines.append(f"           Proj: {leg.projected_value:.1f} vs Line: {leg.line}")
                    lines.append(f"           {leg.reasoning}")
                    lines.append("")
                lines.append("")
        else:
            lines.append("  No longshot parlays found\n")

        # Summary
        lines.extend([
            "=" * 70,
            ""
        ])

        return "\n".join(lines)

    def export_to_json(self, recs: GameRecommendations) -> str:
        """Export recommendations to JSON"""
        def convert(obj):
            if isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, PropType):
                return obj.value
            elif hasattr(obj, '__dict__'):
                return {k: convert(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj

        return json.dumps(convert(recs), indent=2)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NBA BETTING SYSTEM - DEMONSTRATION")
    print("=" * 60)

    # Kelly Criterion Demo
    print("\n--- FRACTIONAL KELLY CRITERION ---\n")

    kelly = KellyCriterion()

    win_prob = 0.55
    odds = -110

    full_kelly = kelly.calculate_kelly_fraction(win_prob, odds) / kelly.KELLY_FRACTION
    fractional = kelly.calculate_kelly_fraction(win_prob, odds)
    units = kelly.calculate_units(win_prob, odds)
    ev = kelly.calculate_ev(win_prob, odds)

    print(f"Win Probability: {win_prob*100:.0f}%")
    print(f"American Odds: {odds}")
    print(f"Full Kelly: {full_kelly*100:.2f}% of bankroll")
    print(f"Fractional Kelly (0.25): {fractional*100:.2f}% of bankroll")
    print(f"Recommended Units: {units}")
    print(f"Expected Value: ${ev:.4f} per unit")

    # SGP Juice Demo
    print("\n--- SGP CORRELATION VALUE ---\n")

    sgp = SGPJuiceCalculator()

    prob_a = 0.55
    prob_b = 0.52
    correlation = 0.40

    naive = sgp.calculate_naive_probability([prob_a, prob_b])
    true = sgp.calculate_true_parlay_probability([prob_a, prob_b], [correlation])

    naive_odds = sgp.probability_to_american(naive)
    true_odds = sgp.probability_to_american(true)

    print(f"Leg A: {prob_a*100:.0f}% | Leg B: {prob_b*100:.0f}%")
    print(f"Correlation: {correlation}")
    print(f"\nNaive parlay prob: {naive*100:.1f}% (fair odds: {naive_odds:+d})")
    print(f"True parlay prob: {true*100:.1f}% (fair odds: {true_odds:+d})")
    print(f"\nCorrelation adds: +{(true-naive)*100:.1f} percentage points")
    print(f"This is why correlated SGPs can have hidden value!")
