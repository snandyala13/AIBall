"""
minutes_model.py - ML-ready Minutes Prediction Model

Predicts expected minutes and full minutes distributions for NBA players
in specific game contexts. Minutes are the single most important variable
in prop betting -- every stat projection flows through projected minutes.

Key insights this model captures:
- Minutes are NOT normally distributed. They are bimodal:
  * Full game: 30-38 min (starter plays normally)
  * Reduced: 20-28 min (blowout, foul trouble, load management)
  * DNP: 0 min (scratch, injury)
- The spread IS the game script prediction -- a -12 favorite's starters sit Q4
- Coach rotation patterns are embedded in the player's minutes variance
- B2B fatigue is non-linear and age/workload dependent
- Position depth matters: if the backup C is out, the starter C plays more

This is a structured rule-based model designed for easy upgrade to a trained
ML model once training data is accumulated.
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Historical blowout probabilities by spread magnitude
# Source: ~10 years of NBA data, blowout defined as 20+ point margin
_SPREAD_TO_BLOWOUT_PROB = [
    (0, 3, 0.08),
    (3, 5, 0.12),
    (5, 7, 0.15),
    (7, 8, 0.20),
    (8, 10, 0.25),
    (10, 12, 0.33),
    (12, 15, 0.40),
    (15, 99, 0.55),
]

# Overtime probability by spread (close games go to OT ~6% of the time)
_SPREAD_TO_OT_PROB = [
    (0, 2, 0.06),
    (2, 4, 0.04),
    (4, 6, 0.03),
    (6, 8, 0.02),
    (8, 10, 0.01),
    (10, 99, 0.005),
]

# Player role classification thresholds (based on L5 avg minutes)
ROLE_STARTER = 28.0
ROLE_ROTATION = 18.0
ROLE_BENCH = 10.0
# Below ROLE_BENCH = deep bench

# League average minutes for starters in a normal game
LEAGUE_AVG_STARTER_MINUTES = 33.5


@dataclass
class MinutesPrediction:
    """Full output from the minutes model."""
    expected_minutes: float          # Point estimate
    minutes_std: float               # Standard deviation of prediction
    is_high_variance: bool           # Flag for unusually uncertain prediction
    base_minutes: float              # Raw base before adjustments
    role: str                        # "starter", "rotation", "bench", "deep_bench"

    # Adjustment breakdown (for transparency / debugging)
    spread_adj: float = 0.0          # Minutes change from game script
    b2b_adj: float = 0.0             # Minutes change from fatigue
    rest_adj: float = 0.0            # Minutes change from extra rest
    trend_adj: float = 0.0           # Minutes change from recent trend
    depth_adj: float = 0.0           # Minutes change from teammate injuries

    @property
    def total_adjustment(self) -> float:
        return self.spread_adj + self.b2b_adj + self.rest_adj + self.trend_adj + self.depth_adj


@dataclass
class MinutesScenario:
    """A single scenario in the minutes distribution."""
    label: str
    minutes: float
    probability: float
    description: str = ""


class MinutesPredictor:
    """
    Predicts expected minutes and minutes distribution for a player
    in a specific game context.

    The core insight is that minutes are driven by game script, which
    the spread predicts. A -12 favorite's starters will sit in Q4 of
    a blowout, but an underdog's starters play heavy minutes trying
    to come back.

    Usage:
        predictor = MinutesPredictor()
        prediction = predictor.predict_minutes(profile, context)
        scenarios = predictor.predict_minutes_distribution(profile, context)
    """

    def predict_minutes(
        self,
        profile,       # PlayerProfile
        context,       # GameContext
        is_starter: Optional[bool] = None,
        injured_teammates: Optional[List] = None
    ) -> Tuple[float, float, float]:
        """
        Predict expected minutes for a player in this game.

        Args:
            profile: PlayerProfile with game_logs, rolling averages, etc.
            context: GameContext with spread, teams, schedule info
            is_starter: Override starter detection (None = auto-detect)
            injured_teammates: List of injured PlayerProfiles on same team
                              (for position depth adjustment)

        Returns:
            (expected_minutes, minutes_std, is_high_variance)
        """
        # Step 1: Classify role
        role = self._classify_player_role(profile, is_starter)

        # Step 2: Compute base minutes from recent history
        base_minutes = self._compute_base_minutes(profile, role)

        # Step 3: Detect minutes trend
        trend_adj = self._compute_minutes_trend(profile, base_minutes)

        # Step 4: Determine team context
        is_home = profile.team == context.home_team.name
        team = context.home_team if is_home else context.away_team
        spread = context.spread if is_home else -context.spread
        is_b2b = team.is_back_to_back
        days_rest = team.days_rest

        # Step 5: Spread-based game script adjustment
        spread_adj = self._compute_spread_adjustment(
            base_minutes, spread, role
        )

        # Step 6: B2B fatigue (non-linear, age/workload aware)
        b2b_adj = self._calculate_b2b_fatigue(
            days_rest, base_minutes, profile
        )

        # Step 7: Rest bonus (diminishing returns)
        rest_adj = self._compute_rest_bonus(days_rest, base_minutes)

        # Step 8: Position depth (injured backup = more starter minutes)
        depth_adj = self._compute_depth_adjustment(
            profile, injured_teammates, role
        )

        # Combine all adjustments
        projected = base_minutes + trend_adj + spread_adj + b2b_adj + rest_adj + depth_adj

        # Floor: no one plays negative minutes, and active starters play at least some
        min_floor = self._get_minutes_floor(role)
        projected = max(projected, min_floor)

        # Ceiling: NBA regulation is 48 min, OT can push higher but cap at 46
        projected = min(projected, 46.0)

        # Compute minutes standard deviation from game logs
        minutes_std = self._compute_minutes_std(profile, projected)

        # High variance flag
        is_high_variance = self._is_high_variance(
            profile, minutes_std, projected, spread, role
        )

        prediction = MinutesPrediction(
            expected_minutes=round(projected, 1),
            minutes_std=round(minutes_std, 1),
            is_high_variance=is_high_variance,
            base_minutes=round(base_minutes, 1),
            role=role,
            spread_adj=round(spread_adj, 1),
            b2b_adj=round(b2b_adj, 1),
            rest_adj=round(rest_adj, 1),
            trend_adj=round(trend_adj, 1),
            depth_adj=round(depth_adj, 1),
        )

        logger.debug(
            f"Minutes prediction for {profile.name}: "
            f"base={base_minutes:.1f}, projected={projected:.1f}, "
            f"role={role}, spread_adj={spread_adj:+.1f}, "
            f"b2b_adj={b2b_adj:+.1f}, rest_adj={rest_adj:+.1f}, "
            f"trend_adj={trend_adj:+.1f}, depth_adj={depth_adj:+.1f}, "
            f"std={minutes_std:.1f}, high_var={is_high_variance}"
        )

        return prediction.expected_minutes, prediction.minutes_std, prediction.is_high_variance

    def predict_minutes_full(
        self,
        profile,
        context,
        is_starter: Optional[bool] = None,
        injured_teammates: Optional[List] = None
    ) -> MinutesPrediction:
        """
        Same as predict_minutes but returns the full MinutesPrediction object
        with adjustment breakdown.
        """
        # Reuse the internal logic -- predict_minutes already builds the
        # MinutesPrediction internally, but we need to expose it.
        role = self._classify_player_role(profile, is_starter)
        base_minutes = self._compute_base_minutes(profile, role)
        trend_adj = self._compute_minutes_trend(profile, base_minutes)

        is_home = profile.team == context.home_team.name
        team = context.home_team if is_home else context.away_team
        spread = context.spread if is_home else -context.spread
        is_b2b = team.is_back_to_back
        days_rest = team.days_rest

        spread_adj = self._compute_spread_adjustment(base_minutes, spread, role)
        b2b_adj = self._calculate_b2b_fatigue(days_rest, base_minutes, profile)
        rest_adj = self._compute_rest_bonus(days_rest, base_minutes)
        depth_adj = self._compute_depth_adjustment(profile, injured_teammates, role)

        projected = base_minutes + trend_adj + spread_adj + b2b_adj + rest_adj + depth_adj
        min_floor = self._get_minutes_floor(role)
        projected = max(projected, min_floor)
        projected = min(projected, 46.0)

        minutes_std = self._compute_minutes_std(profile, projected)
        is_high_variance = self._is_high_variance(
            profile, minutes_std, projected, spread, role
        )

        return MinutesPrediction(
            expected_minutes=round(projected, 1),
            minutes_std=round(minutes_std, 1),
            is_high_variance=is_high_variance,
            base_minutes=round(base_minutes, 1),
            role=role,
            spread_adj=round(spread_adj, 1),
            b2b_adj=round(b2b_adj, 1),
            rest_adj=round(rest_adj, 1),
            trend_adj=round(trend_adj, 1),
            depth_adj=round(depth_adj, 1),
        )

    def predict_minutes_distribution(
        self,
        profile,
        context,
        is_starter: Optional[bool] = None,
        injured_teammates: Optional[List] = None
    ) -> List[MinutesScenario]:
        """
        Return a full minutes distribution (not just a point estimate).

        The point estimate from predict_minutes is an expected value across
        scenarios. This method returns the individual scenarios with their
        probabilities so that downstream stat projections can be computed
        as weighted averages across game scripts.

        This is more accurate than a single minutes value because it lets
        you compute E[f(minutes)] instead of f(E[minutes]) -- important
        when the stat-to-minutes relationship is non-linear (e.g., crunch
        time minutes are worth more for assists than garbage time minutes).

        Returns:
            List of MinutesScenario (label, minutes, probability)
            Probabilities sum to ~1.0.
        """
        role = self._classify_player_role(profile, is_starter)
        base_minutes = self._compute_base_minutes(profile, role)

        is_home = profile.team == context.home_team.name
        team = context.home_team if is_home else context.away_team
        spread = context.spread if is_home else -context.spread
        days_rest = team.days_rest
        is_b2b = team.is_back_to_back

        abs_spread = abs(spread)
        blowout_prob = self._spread_to_blowout_probability(abs_spread)
        ot_prob = self._spread_to_ot_probability(abs_spread)

        # B2B and rest adjustments apply to ALL scenarios
        b2b_adj = self._calculate_b2b_fatigue(days_rest, base_minutes, profile)
        rest_adj = self._compute_rest_bonus(days_rest, base_minutes)
        trend_adj = self._compute_minutes_trend(profile, base_minutes)
        depth_adj = self._compute_depth_adjustment(profile, injured_teammates, role)
        universal_adj = b2b_adj + rest_adj + trend_adj + depth_adj

        # Is this team the favorite or underdog?
        is_favorite = spread < 0  # Negative spread = favored

        scenarios = []

        if role in ("starter", "rotation"):
            # -- Normal game scenario --
            normal_min = base_minutes + universal_adj
            # Close game: starters play their full allotment
            normal_prob = 1.0 - blowout_prob - ot_prob

            scenarios.append(MinutesScenario(
                label="normal",
                minutes=round(max(normal_min, self._get_minutes_floor(role)), 1),
                probability=round(normal_prob, 3),
                description="Competitive game, normal rotation"
            ))

            # -- Blowout scenario --
            if is_favorite:
                # Favorite in a blowout: starters sit early in Q4
                # Reduction depends on spread magnitude
                blowout_reduction = self._blowout_minutes_reduction(
                    abs_spread, role, is_favorite=True
                )
                blowout_min = base_minutes * (1.0 - blowout_reduction) + universal_adj
            else:
                # Underdog in a blowout: starters may actually play MORE
                # trying to mount a comeback, but get pulled if truly hopeless
                if abs_spread >= 12:
                    # Big underdog getting blown out -- starters pulled late Q3/early Q4
                    blowout_reduction = self._blowout_minutes_reduction(
                        abs_spread, role, is_favorite=False
                    )
                    blowout_min = base_minutes * (1.0 - blowout_reduction) + universal_adj
                else:
                    # Moderate underdog: starters play heavy trying to compete
                    blowout_min = base_minutes * 1.02 + universal_adj

            scenarios.append(MinutesScenario(
                label="blowout",
                minutes=round(max(blowout_min, self._get_minutes_floor(role)), 1),
                probability=round(blowout_prob, 3),
                description=(
                    "Blowout win (starters rest)" if is_favorite
                    else "Blowout loss (starters may be pulled)"
                )
            ))

            # -- Overtime scenario --
            ot_min = min(base_minutes * 1.10 + universal_adj, 46.0)
            scenarios.append(MinutesScenario(
                label="overtime",
                minutes=round(max(ot_min, self._get_minutes_floor(role)), 1),
                probability=round(ot_prob, 3),
                description="Close game goes to overtime"
            ))

        else:
            # Bench / deep bench players
            # Blowouts HELP bench players (garbage time)
            normal_min = base_minutes + universal_adj
            normal_prob = 1.0 - blowout_prob - ot_prob

            scenarios.append(MinutesScenario(
                label="normal",
                minutes=round(max(normal_min, 0.0), 1),
                probability=round(normal_prob, 3),
                description="Normal rotation"
            ))

            # Blowout: bench players get MORE minutes
            garbage_time_boost = 1.20 if is_favorite else 1.10
            blowout_min = base_minutes * garbage_time_boost + universal_adj
            scenarios.append(MinutesScenario(
                label="blowout",
                minutes=round(max(blowout_min, 0.0), 1),
                probability=round(blowout_prob, 3),
                description="Blowout garbage time minutes"
            ))

            # OT: bench players usually don't play in OT
            ot_min = base_minutes * 0.90 + universal_adj
            scenarios.append(MinutesScenario(
                label="overtime",
                minutes=round(max(ot_min, 0.0), 1),
                probability=round(ot_prob, 3),
                description="OT -- bench usually sits"
            ))

        # Normalize probabilities to sum to 1.0
        total_prob = sum(s.probability for s in scenarios)
        if total_prob > 0 and abs(total_prob - 1.0) > 0.001:
            for s in scenarios:
                s.probability = round(s.probability / total_prob, 3)

        return scenarios

    # =========================================================================
    # INTERNAL: Role Classification
    # =========================================================================

    def _classify_player_role(
        self,
        profile,
        is_starter: Optional[bool] = None
    ) -> str:
        """
        Classify player into: starter, rotation, bench, deep_bench.

        Uses L5 minutes (not season average) because roles change mid-season.
        A player who averaged 22 min/game on the season but 32 min/game over
        the last 5 is now a starter -- the season average is misleading.
        """
        if is_starter is not None:
            return "starter" if is_starter else "rotation"

        # Prefer L5, fall back to L3, then L10, then season
        l5_min = profile.last_5_avg.get("min", 0)
        l3_min = profile.last_3_avg.get("min", 0)
        l10_min = profile.last_10_avg.get("min", 0)
        season_min = profile.season_avg.get("min", 0)

        # Use L5 as primary, but if only 1-2 games played, use what we have
        if l5_min > 0:
            ref_min = l5_min
        elif l3_min > 0:
            ref_min = l3_min
        elif l10_min > 0:
            ref_min = l10_min
        elif season_min > 0:
            ref_min = season_min
        else:
            # No data at all -- assume rotation player
            return "rotation"

        if ref_min >= ROLE_STARTER:
            return "starter"
        elif ref_min >= ROLE_ROTATION:
            return "rotation"
        elif ref_min >= ROLE_BENCH:
            return "bench"
        else:
            return "deep_bench"

    # =========================================================================
    # INTERNAL: Base Minutes
    # =========================================================================

    def _compute_base_minutes(self, profile, role: str) -> float:
        """
        Compute base minutes from recent history.

        Weighting:
        - L3: 45% (most recent signal, captures role changes)
        - L5: 30% (smooths single-game noise)
        - L10: 15% (longer baseline)
        - Season: 10% (stabilizer)

        For role-change detection: if L3 and L10 diverge significantly,
        we weight L3 even heavier (60%) to capture the new role faster.
        """
        l3 = profile.last_3_avg.get("min", 0)
        l5 = profile.last_5_avg.get("min", 0)
        l10 = profile.last_10_avg.get("min", 0)
        season = profile.season_avg.get("min", 0)

        # Detect role change: L3 vs L10 divergence
        role_changed = False
        if l3 > 0 and l10 > 0:
            pct_change = abs(l3 - l10) / l10
            if pct_change >= 0.15:  # 15% change = likely new role
                role_changed = True

        values = []
        weights = []

        if role_changed:
            # Heavy recency bias for role changes
            if l3 > 0:
                values.append(l3)
                weights.append(0.60)
            if l5 > 0:
                values.append(l5)
                weights.append(0.25)
            if l10 > 0:
                values.append(l10)
                weights.append(0.10)
            if season > 0:
                values.append(season)
                weights.append(0.05)
        else:
            # Standard weighting
            if l3 > 0:
                values.append(l3)
                weights.append(0.45)
            if l5 > 0:
                values.append(l5)
                weights.append(0.30)
            if l10 > 0:
                values.append(l10)
                weights.append(0.15)
            if season > 0:
                values.append(season)
                weights.append(0.10)

        if not values:
            # No data -- use role-based defaults
            defaults = {
                "starter": LEAGUE_AVG_STARTER_MINUTES,
                "rotation": 23.0,
                "bench": 14.0,
                "deep_bench": 6.0,
            }
            return defaults.get(role, 20.0)

        total_weight = sum(weights)
        return sum(v * w / total_weight for v, w in zip(values, weights))

    # =========================================================================
    # INTERNAL: Minutes Trend
    # =========================================================================

    def _compute_minutes_trend(self, profile, base_minutes: float) -> float:
        """
        Detect if player's minutes are trending up or down.

        Compares L3 vs L10 minutes. A player whose L3 is 35 min but L10
        is 30 min is clearly getting an expanded role -- we should project
        closer to 35, not the blended average.

        Returns: adjustment in raw minutes (positive = trending up)
        """
        l3 = profile.last_3_avg.get("min", 0)
        l10 = profile.last_10_avg.get("min", 0)

        if l10 <= 0 or l3 <= 0:
            return 0.0

        pct_change = (l3 - l10) / l10

        # Only apply trend if meaningful (>5% change)
        if abs(pct_change) < 0.05:
            return 0.0

        # Scale: apply 30% of the observed trend as forward-looking adjustment
        # Cap at +/- 3 minutes to avoid overreaction
        raw_adj = base_minutes * pct_change * 0.30
        return max(-3.0, min(3.0, raw_adj))

    # =========================================================================
    # INTERNAL: Spread / Game Script
    # =========================================================================

    def _compute_spread_adjustment(
        self,
        base_minutes: float,
        spread: float,
        role: str
    ) -> float:
        """
        Adjust minutes based on point spread (game script prediction).

        The spread tells us the most likely game script:
        - Close game (spread 0-5): normal minutes for everyone
        - Moderate favorite (5-10): slight reduction for starters
        - Big favorite (10-15): meaningful reduction (sit early Q4)
        - Massive favorite (15+): significant reduction (sit mid-Q3)
        - Underdogs in close games get INCREASED minutes
        - Bench players BENEFIT from blowouts (garbage time)

        Args:
            base_minutes: Player's baseline minutes
            spread: From player's team perspective (negative = favored)
            role: "starter", "rotation", "bench", "deep_bench"
        """
        abs_spread = abs(spread)
        is_favorite = spread < 0

        if role in ("starter", "rotation"):
            if is_favorite:
                # Favorites: starters play fewer minutes in expected blowouts
                # This is an EXPECTED VALUE adjustment, not a "will they sit" binary
                # Spread 5 -> tiny reduction, spread 15 -> large reduction
                reduction_pct = self._favorite_starter_reduction(abs_spread)
                return -base_minutes * reduction_pct
            else:
                # Underdogs: starters play slightly MORE in close games
                # But in blowout losses, they also get pulled
                # Net effect: small boost for moderate underdogs, small reduction for big ones
                if abs_spread <= 6:
                    # Close underdog: coach plays starters heavy
                    return base_minutes * 0.01  # +1%
                elif abs_spread <= 10:
                    # Moderate underdog: roughly neutral
                    return 0.0
                else:
                    # Big underdog: risk of getting blown out, starters pulled
                    reduction_pct = self._underdog_starter_reduction(abs_spread)
                    return -base_minutes * reduction_pct
        else:
            # Bench / deep bench: blowouts = more garbage time
            if abs_spread >= 10:
                boost = base_minutes * 0.08  # Up to 8% more minutes
                return boost if is_favorite else boost * 0.5
            elif abs_spread >= 7:
                return base_minutes * 0.03
            return 0.0

    def _favorite_starter_reduction(self, abs_spread: float) -> float:
        """
        Expected minutes reduction for a starter on the favored team.

        Uses a smooth curve rather than hard thresholds. The blowout
        probability increases with spread, and in a blowout starters
        lose ~8-12 minutes. So the expected reduction is:
        E[reduction] = P(blowout) * minutes_lost_in_blowout

        But we also account for the non-blowout scenarios where minutes
        are normal, making this an expected-value adjustment.
        """
        if abs_spread < 3:
            return 0.0

        # Smooth logistic-style curve
        # spread 5 -> ~1% reduction
        # spread 8 -> ~3% reduction
        # spread 10 -> ~5% reduction
        # spread 12 -> ~8% reduction
        # spread 15 -> ~12% reduction
        # spread 18+ -> ~15% (cap)
        raw = 0.002 * (abs_spread - 3) ** 1.5
        return min(raw, 0.15)

    def _underdog_starter_reduction(self, abs_spread: float) -> float:
        """
        Expected minutes reduction for a starter on a big underdog.
        Only kicks in for 10+ point underdogs.
        """
        if abs_spread < 10:
            return 0.0
        # Gentler than favorite reduction -- underdogs keep starters in longer
        # hoping for a comeback
        raw = 0.003 * (abs_spread - 10) ** 1.3
        return min(raw, 0.10)

    # =========================================================================
    # INTERNAL: Back-to-Back Fatigue
    # =========================================================================

    def _calculate_b2b_fatigue(
        self,
        days_rest: int,
        base_minutes: float,
        profile=None
    ) -> float:
        """
        More nuanced B2B model than a flat -8% reduction.

        Factors:
        1. Days rest (0=B2B is the big one)
        2. Player age: older players (30+) lose more on B2B
        3. Minutes load: high-minutes players (35+) lose more on B2B
        4. Recent heavy workload: if L3 avg > 36 min, extra fatigue

        Historical data shows:
        - Young players (22-25) lose ~5% on B2B
        - Prime players (26-29) lose ~7% on B2B
        - Veterans (30-33) lose ~10% on B2B
        - Old veterans (34+) lose ~12% on B2B
        - High-minutes players lose an additional ~2%

        Returns: adjustment in raw minutes (always negative or zero)
        """
        if days_rest > 1:
            # Not a B2B -- no fatigue penalty
            return 0.0

        if days_rest == 1:
            # One day rest is standard, no penalty
            return 0.0

        # B2B (days_rest == 0)
        # Base fatigue rate by estimated age bracket
        age_fatigue = self._estimate_age_fatigue(profile)

        # High-minutes workload penalty
        workload_penalty = 0.0
        if base_minutes >= 36:
            workload_penalty = 0.025  # Extra 2.5% for ironman players
        elif base_minutes >= 33:
            workload_penalty = 0.015  # Extra 1.5% for heavy-minutes starters

        # Check if they've had a heavy recent workload (L3 avg minutes)
        recent_load_penalty = 0.0
        if profile is not None:
            l3_min = profile.last_3_avg.get("min", 0)
            if l3_min >= 38:
                recent_load_penalty = 0.02  # Extra 2% if grinding recently
            elif l3_min >= 35:
                recent_load_penalty = 0.01

        total_fatigue_rate = age_fatigue + workload_penalty + recent_load_penalty

        # Cap at 15% reduction
        total_fatigue_rate = min(total_fatigue_rate, 0.15)

        return -base_minutes * total_fatigue_rate

    def _estimate_age_fatigue(self, profile) -> float:
        """
        Estimate age-based fatigue rate for B2B games.

        Since we don't have birth date on PlayerProfile, we use a proxy:
        players with high season averages and low L3 variance are typically
        veterans. But the safest default is the league-average fatigue rate.

        If we later add age to PlayerProfile, this becomes a simple lookup.
        """
        # Default league-average B2B fatigue: ~7%
        # This is conservative and correct on average
        base_rate = 0.07

        if profile is None:
            return base_rate

        # Heuristic: very high-usage players who are also high-minutes
        # tend to be established veterans (likely older)
        usage = getattr(profile, 'usage_pct', 0)
        season_min = profile.season_avg.get("min", 0)

        if usage >= 28 and season_min >= 34:
            # Star player with heavy usage -- likely prime or older
            return 0.09
        elif usage >= 22 and season_min >= 30:
            # Solid starter
            return 0.07
        elif season_min < 20:
            # Role player / young player -- less affected
            return 0.05

        return base_rate

    # =========================================================================
    # INTERNAL: Rest Bonus
    # =========================================================================

    def _compute_rest_bonus(self, days_rest: int, base_minutes: float) -> float:
        """
        Extra rest bonus with diminishing returns.

        - 2 days rest: +1% (marginal benefit over standard 1-day)
        - 3 days rest: +2% (e.g., before/after All-Star break)
        - 4+ days rest: +2.5% (cap -- more rest doesn't keep helping)

        Note: this is separate from B2B. A standard 1-day rest gets no
        bonus and no penalty. Only extra rest (2+) helps, and B2B (0) hurts.
        """
        if days_rest <= 1:
            return 0.0
        elif days_rest == 2:
            return base_minutes * 0.01
        elif days_rest == 3:
            return base_minutes * 0.02
        else:
            # 4+ days: diminishing returns, cap at 2.5%
            return base_minutes * 0.025

    # =========================================================================
    # INTERNAL: Position Depth / Injury Impact on Minutes
    # =========================================================================

    def _compute_depth_adjustment(
        self,
        profile,
        injured_teammates: Optional[List],
        role: str
    ) -> float:
        """
        If a teammate at the same position is injured, the remaining players
        at that position get more minutes.

        Logic:
        - If the backup C is out, the starter C gets +2-3 min
        - If the starter is out, the backup gets +8-12 min (but that's
          handled by the usage boost system, not here)
        - Cross-position impact is minimal (a PG injury doesn't help the C)

        This only handles MINUTES impact. The stat boost from increased
        usage/role is handled separately by the injury pop-off system.
        """
        if not injured_teammates:
            return 0.0

        player_pos = (getattr(profile, 'position', '') or '').upper()
        if not player_pos:
            return 0.0

        # Define position groups (positions that share minutes)
        position_groups = {
            "PG": {"PG", "SG"},
            "SG": {"PG", "SG"},
            "SF": {"SF", "PF", "SG"},
            "PF": {"PF", "SF", "C"},
            "C": {"C", "PF"},
        }

        related_positions = position_groups.get(player_pos, {player_pos})

        # Count injured players at related positions
        injured_same_pos = 0
        total_injured_minutes = 0.0
        for teammate in injured_teammates:
            tm_pos = (getattr(teammate, 'position', '') or '').upper()
            tm_status = getattr(teammate, 'status', 'active')
            if tm_pos in related_positions and tm_status in ('out', 'doubtful'):
                injured_same_pos += 1
                # How many minutes does the injured player usually get?
                tm_min = (
                    teammate.last_5_avg.get("min", 0)
                    or teammate.season_avg.get("min", 0)
                )
                total_injured_minutes += tm_min

        if injured_same_pos == 0:
            return 0.0

        # The injured player's minutes get redistributed among remaining players
        # at that position. Assume 2-3 players share the position group.
        # A starter absorbs ~30-40% of the lost minutes if the backup is out.
        # A backup absorbs ~60-70% of the lost minutes if the starter is out.
        if role == "starter":
            # Starter picks up some of backup's minutes
            absorption_rate = 0.30
        elif role == "rotation":
            # Rotation player picks up more (they're the next in line)
            absorption_rate = 0.45
        else:
            # Bench player may get called up
            absorption_rate = 0.50

        minutes_gained = total_injured_minutes * absorption_rate

        # Cap at 5 minutes gained (no one gets a 10-minute boost)
        minutes_gained = min(minutes_gained, 5.0)

        logger.debug(
            f"{profile.name}: {injured_same_pos} injured at related positions, "
            f"total_injured_min={total_injured_minutes:.1f}, "
            f"gained={minutes_gained:.1f}"
        )

        return minutes_gained

    # =========================================================================
    # INTERNAL: Blowout Probability
    # =========================================================================

    def _spread_to_blowout_probability(self, abs_spread: float) -> float:
        """
        Convert absolute point spread to probability of a blowout
        (20+ point margin at some point, leading to starters sitting).

        Based on historical NBA data:
        - Spread 0-3: ~8% blowout probability
        - Spread 5-7: ~15%
        - Spread 8-10: ~25%
        - Spread 12-15: ~40%
        - Spread 15+: ~55%

        Uses linear interpolation between the lookup points.
        """
        for low, high, prob in _SPREAD_TO_BLOWOUT_PROB:
            if low <= abs_spread < high:
                # Interpolate within the bucket
                frac = (abs_spread - low) / max(high - low, 1)
                # Find next bucket's prob for interpolation
                next_prob = prob
                for l2, h2, p2 in _SPREAD_TO_BLOWOUT_PROB:
                    if l2 == high:
                        next_prob = p2
                        break
                return prob + frac * (next_prob - prob)

        # Beyond all buckets
        return 0.55

    def _spread_to_ot_probability(self, abs_spread: float) -> float:
        """
        Convert absolute point spread to overtime probability.
        Close games go to OT ~6% of the time. Big spreads almost never do.
        """
        for low, high, prob in _SPREAD_TO_OT_PROB:
            if low <= abs_spread < high:
                return prob
        return 0.005

    def _blowout_minutes_reduction(
        self,
        abs_spread: float,
        role: str,
        is_favorite: bool
    ) -> float:
        """
        In a blowout scenario, how much do minutes drop?

        This is NOT the expected-value reduction (that's _compute_spread_adjustment).
        This is the conditional reduction: "given that a blowout happens,
        how many fewer minutes does the starter play?"

        Favorites in blowout wins: starters sit entire Q4 (-8 to -12 min)
        Underdogs in blowout losses: starters pulled late Q3 (-6 to -10 min)
        """
        if role == "starter":
            if is_favorite:
                # Starters sit Q4 in a blowout win
                # Bigger spread = more likely to sit earlier
                if abs_spread >= 15:
                    return 0.28  # ~10 min reduction on a 35-min player
                elif abs_spread >= 12:
                    return 0.23
                elif abs_spread >= 10:
                    return 0.20
                else:
                    return 0.15
            else:
                # Underdog starters stay in longer hoping for comeback
                if abs_spread >= 15:
                    return 0.20  # Still pulled eventually
                elif abs_spread >= 12:
                    return 0.15
                else:
                    return 0.10
        elif role == "rotation":
            # Rotation players: similar pattern but less extreme
            if is_favorite:
                return min(0.15, 0.01 * abs_spread)
            else:
                return min(0.10, 0.008 * abs_spread)
        else:
            # Bench players actually gain minutes in blowouts
            return 0.0  # Handled separately (garbage time boost)

    # =========================================================================
    # INTERNAL: Minutes Variance / Standard Deviation
    # =========================================================================

    def _compute_minutes_std(self, profile, projected: float) -> float:
        """
        Compute standard deviation of minutes prediction.

        Uses actual game-log minutes variance when available.
        Falls back to role-based defaults.
        """
        # Try to compute from game logs
        played_logs = [g for g in profile.game_logs if g.minutes > 0]

        if len(played_logs) >= 5:
            minutes_values = [g.minutes for g in played_logs[:15]]
            mean = sum(minutes_values) / len(minutes_values)
            variance = sum((m - mean) ** 2 for m in minutes_values) / len(minutes_values)
            empirical_std = math.sqrt(variance)

            # Blend empirical with a prior based on role
            # This prevents wild std estimates from small samples
            role_std = self._role_based_std(projected)
            n = len(minutes_values)
            # More data = more weight on empirical
            empirical_weight = min(n / 15.0, 0.85)
            return empirical_weight * empirical_std + (1 - empirical_weight) * role_std

        elif len(played_logs) >= 3:
            # Small sample -- blend more with prior
            minutes_values = [g.minutes for g in played_logs]
            mean = sum(minutes_values) / len(minutes_values)
            variance = sum((m - mean) ** 2 for m in minutes_values) / len(minutes_values)
            empirical_std = math.sqrt(variance)
            role_std = self._role_based_std(projected)
            return 0.4 * empirical_std + 0.6 * role_std

        else:
            return self._role_based_std(projected)

    def _role_based_std(self, projected_minutes: float) -> float:
        """
        Default minutes standard deviation by projected minutes level.

        Higher-minutes players have lower relative variance (coaches
        are more consistent with their stars). Bench players have
        high relative variance (game-to-game swings).
        """
        if projected_minutes >= 32:
            return 4.0   # Starters: ~4 min std
        elif projected_minutes >= 25:
            return 4.5   # Rotation+: slightly more variable
        elif projected_minutes >= 18:
            return 5.0   # Rotation: moderate variance
        elif projected_minutes >= 10:
            return 5.5   # Bench: high variance
        else:
            return 4.0   # Deep bench: actually lower absolute variance

    # =========================================================================
    # INTERNAL: High Variance Detection
    # =========================================================================

    def _is_high_variance(
        self,
        profile,
        minutes_std: float,
        projected: float,
        spread: float,
        role: str
    ) -> bool:
        """
        Flag predictions with unusually high uncertainty.

        High variance means the minutes estimate is unreliable and
        downstream stat projections should be treated with caution.
        Bets on high-variance-minutes players are riskier.

        Triggers:
        1. Minutes std > 30% of projected (very inconsistent player)
        2. Big spread (15+) for a starter (blowout risk dominates)
        3. Player has recent DNPs mixed with full games
        4. Very small game log sample
        """
        # Check 1: High relative variance
        if projected > 0 and (minutes_std / projected) > 0.30:
            return True

        # Check 2: Big spread for a starter
        if abs(spread) >= 15 and role == "starter":
            return True

        # Check 3: Recent DNPs mixed with full games (load management pattern)
        played_logs = [g for g in profile.game_logs if g.minutes > 0]
        all_logs = profile.game_logs[:10]
        if len(all_logs) >= 5:
            dnp_count = sum(1 for g in all_logs if g.minutes == 0)
            if dnp_count >= 2:
                # 2+ DNPs in last 10 games = load management risk
                return True

        # Check 4: Very small sample
        if len(played_logs) < 3:
            return True

        return False

    # =========================================================================
    # INTERNAL: Helpers
    # =========================================================================

    def _get_minutes_floor(self, role: str) -> float:
        """Minimum projected minutes by role (prevents nonsensical projections)."""
        floors = {
            "starter": 20.0,    # Starter almost always plays 20+
            "rotation": 12.0,   # Rotation player gets at least 12
            "bench": 4.0,       # Bench player might only get a few
            "deep_bench": 0.0,  # Deep bench can DNP
        }
        return floors.get(role, 0.0)


# =============================================================================
# CONVENIENCE: Expected minutes from scenario distribution
# =============================================================================

def expected_minutes_from_scenarios(scenarios: List[MinutesScenario]) -> float:
    """
    Compute expected minutes as weighted average across scenarios.
    This should match predict_minutes() closely but may differ slightly
    due to rounding.
    """
    return sum(s.minutes * s.probability for s in scenarios)
