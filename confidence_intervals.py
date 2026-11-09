"""
confidence_intervals.py - Prediction Uncertainty Estimation

Estimates uncertainty around probability predictions to adjust Kelly sizing.
A confident 62% warrants a bigger bet than an uncertain 62%.

Sources of uncertainty:
1. Sample size (fewer games = wider intervals)
2. Player consistency (high std_dev = wider)
3. Model disagreement (adjustments pulling in opposite directions)
4. Projection-line distance (close to line = more uncertain)

Integration:
    In bet_generator.py, after computing Kelly fraction:

        from confidence_intervals import PredictionUncertainty
        uncertainty_model = PredictionUncertainty()
        result = uncertainty_model.estimate_uncertainty(
            projection, line, std_dev, num_games, adjustments_dict
        )
        adjusted_kelly = kelly_fraction * result['kelly_adjustment']
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats


# =============================================================================
# CONSTANTS
# =============================================================================

# Confidence interval level (90% CI -> z = 1.645)
CI_Z = 1.645

# Kelly adjustment by uncertainty tier
KELLY_ADJUSTMENTS = {
    "low": 1.0,
    "medium": 0.7,
    "high": 0.4,
    "very_high": 0.2,
}

# Thresholds for uncertainty classification (on probability CI width)
# These are calibrated to be conservative -- wider = higher uncertainty.
# NBA props inherently have high variance, so "low" is still fairly wide.
UNCERTAINTY_THRESHOLDS = {
    "low": 0.16,       # CI width < 16% -> low uncertainty (best realistic case)
    "medium": 0.25,    # CI width < 25% -> medium
    "high": 0.35,      # CI width < 35% -> high
    # anything above 35% -> very_high
}

# Minimum games for reliable estimates
MIN_GAMES_RELIABLE = 15
MIN_GAMES_USABLE = 5

# Std dev calibration (matches EdgeCalculator in model_engine.py)
STD_DEV_CALIBRATION = 2.0

MIN_STD_DEV = {
    "pts": 6.0,
    "reb": 3.5,
    "ast": 3.0,
    "fg3m": 1.5,
    "stl": 1.0,
    "blk": 1.0,
    "turnover": 1.2,
    "pra": 9.0,
}

# Prop type -> std_dev key
PROP_TO_STD_KEY = {
    "points": "pts",
    "rebounds": "reb",
    "assists": "ast",
    "threes": "fg3m",
    "steals": "stl",
    "blocks": "blk",
    "turnovers": "turnover",
    "pts_rebs_asts": "pra",
    "pts_rebs": "pra",
    "pts_asts": "pra",
    "rebs_asts": "pra",
    "stls_blks": "pra",
}


class PredictionUncertainty:
    """
    Estimates uncertainty around probability predictions based on:
    1. Sample size (fewer games = wider intervals)
    2. Player consistency (high std_dev = wider)
    3. Model disagreement (when multiple features point different directions)
    4. Projection-line distance (close to line = more uncertain)
    """

    def estimate_uncertainty(
        self,
        projection: float,
        line: float,
        std_dev: float,
        num_games: int,
        adjustments_dict: Dict[str, float],
        prop_type: str = "points",
        over_odds: int = -110,
    ) -> Dict:
        """
        Estimate uncertainty around a probability prediction.

        Args:
            projection: Model's projected value
            line: DraftKings line
            std_dev: Raw standard deviation of the stat
            num_games: Number of game logs used to build projection
            adjustments_dict: Dict of adjustments applied, e.g.
                {"base": 22.5, "defense": -1.2, "pace": 0.8, ...}
            prop_type: e.g. "points", "rebounds"
            over_odds: American odds for over (for implied prob)

        Returns: dict with keys:
            probability: float, point estimate P(over)
            ci_lower: float, lower bound of 90% CI
            ci_upper: float, upper bound of 90% CI
            ci_width: float, ci_upper - ci_lower
            uncertainty: str, "low"/"medium"/"high"/"very_high"
            kelly_adjustment: float, multiplier for Kelly fraction
            components: dict, breakdown of uncertainty sources
        """
        # Compute the point estimate probability (same as EdgeCalculator)
        calibrated_std = self._get_calibrated_std(std_dev, prop_type)
        probability = self._compute_over_prob(projection, line, calibrated_std)

        # 1. Base standard error from sample size
        # Use raw std (not calibrated) for SE of the mean -- calibration
        # is for the individual outcome variance, not the mean uncertainty
        raw_std_for_se = max(std_dev, 1.0)
        se_base = self._sample_size_uncertainty(raw_std_for_se, num_games)

        # 2. Consistency penalty (high CV = wider)
        consistency_mult = self._consistency_multiplier(std_dev, projection, prop_type)

        # 3. Proximity penalty (close to line = uncertain)
        proximity_mult = self._proximity_multiplier(projection, line, calibrated_std)

        # 4. Adjustment disagreement penalty
        disagreement = self.calculate_adjustment_disagreement(adjustments_dict)
        disagreement_mult = 1.0 + disagreement * 0.5  # scale: 0-1 -> 1.0-1.5x

        # Combined standard error of the projection
        combined_se = se_base * consistency_mult * proximity_mult * disagreement_mult

        # Convert projection SE to probability CI using the delta method.
        # dP/dx = -phi(z) / sigma, where z = (line - proj) / sigma
        # SE(P) = |dP/dx| * SE(x)
        z = (line - projection) / calibrated_std if calibrated_std > 0 else 0
        dphi = scipy_stats.norm.pdf(z)  # density at z
        if calibrated_std > 0 and dphi > 0.001:
            prob_se = (dphi / calibrated_std) * combined_se
        else:
            # Fallback: use a simpler heuristic
            prob_se = combined_se / (calibrated_std * 4) if calibrated_std > 0 else 0.10

        # Ensure minimum uncertainty floor (we are never perfectly certain)
        prob_se = max(prob_se, 0.02)

        # Build 90% CI
        ci_lower = max(0.01, probability - CI_Z * prob_se)
        ci_upper = min(0.99, probability + CI_Z * prob_se)
        ci_width = ci_upper - ci_lower

        # Apply damping: same caps as EdgeCalculator (32%-68%)
        # The CI is on the raw probability; clamp to the model's operating range
        ci_lower = max(0.30, ci_lower)
        ci_upper = min(0.70, ci_upper)
        ci_width = ci_upper - ci_lower

        # Classify uncertainty
        uncertainty_level = self._classify_uncertainty(ci_width)
        kelly_adj = self.get_kelly_adjustment(uncertainty_level)

        return {
            "probability": round(probability, 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "ci_width": round(ci_width, 4),
            "uncertainty": uncertainty_level,
            "kelly_adjustment": kelly_adj,
            "components": {
                "se_base": round(se_base, 4),
                "consistency_mult": round(consistency_mult, 3),
                "proximity_mult": round(proximity_mult, 3),
                "disagreement_mult": round(disagreement_mult, 3),
                "disagreement_score": round(disagreement, 3),
                "prob_se": round(prob_se, 4),
                "num_games": num_games,
            },
        }

    @staticmethod
    def get_kelly_adjustment(uncertainty_level: str) -> float:
        """
        Scale Kelly bet size by confidence.

        Low uncertainty (tight CI):  1.0x Kelly (full size)
        Medium uncertainty:          0.7x Kelly
        High uncertainty:            0.4x Kelly
        Very high:                   0.2x Kelly (near-minimum bet)
        """
        return KELLY_ADJUSTMENTS.get(uncertainty_level, 0.4)

    @staticmethod
    def calculate_adjustment_disagreement(adjustments: Dict[str, float]) -> float:
        """
        Measure how much the adjustments agree or disagree.

        If pace says +5% but defense says -8%, that's disagreement.
        If pace says +3% and defense says +2%, that's agreement.

        Returns: float in [0, 1] where 0 = perfect agreement, 1 = max disagreement.
        """
        # Filter to directional adjustments (exclude "base" which is the starting point)
        directional_keys = [
            "defense", "pace", "home_away", "streak", "opponent",
            "tracking", "pbp", "ctg", "lineup", "minutes",
        ]

        adj_values = []
        for key in directional_keys:
            val = adjustments.get(key, 0.0)
            if val != 0.0:
                adj_values.append(val)

        if len(adj_values) < 2:
            return 0.0  # Not enough adjustments to disagree

        # Count positive and negative adjustments
        positives = [v for v in adj_values if v > 0]
        negatives = [v for v in adj_values if v < 0]

        if not positives or not negatives:
            return 0.0  # All same direction = agreement

        # Disagreement = fraction of total absolute adjustment that is "opposing"
        total_positive = sum(positives)
        total_negative = sum(abs(v) for v in negatives)
        total_abs = total_positive + total_negative

        if total_abs < 0.01:
            return 0.0

        # Min(positive, negative) / total gives disagreement fraction
        # When equal positive and negative, this is 0.5 (max disagreement)
        # When one dominates, this approaches 0
        minority = min(total_positive, total_negative)
        disagreement = (2 * minority) / total_abs  # normalized to [0, 1]

        # Weight by magnitude: large opposing adjustments are worse than small ones
        magnitude_factor = min(total_abs / 5.0, 1.0)  # saturates at 5 units

        return min(1.0, disagreement * magnitude_factor)

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _sample_size_uncertainty(self, calibrated_std: float, num_games: int) -> float:
        """
        Standard error of the mean: sigma / sqrt(n).

        With small samples, the SE is large. With 20+ games, it's small.
        Conservative: use a minimum effective sample of 3 to avoid division
        by very small numbers, and a maximum of 20 (beyond that, diminishing
        returns on uncertainty reduction).
        """
        effective_n = max(3, min(num_games, 20))
        se = calibrated_std / math.sqrt(effective_n)

        # Extra penalty for very small samples (< 5 games)
        if num_games < MIN_GAMES_USABLE:
            se *= 1.5  # 50% wider for tiny samples
        elif num_games < MIN_GAMES_RELIABLE:
            # Linear interpolation between 1.0 and 1.3 penalty
            penalty = 1.0 + 0.3 * (MIN_GAMES_RELIABLE - num_games) / (MIN_GAMES_RELIABLE - MIN_GAMES_USABLE)
            se *= penalty

        return se

    def _consistency_multiplier(self, raw_std: float, projection: float, prop_type: str) -> float:
        """
        High coefficient of variation = less consistent player = wider CI.

        A player averaging 25 pts with std_dev 4 (CV=0.16) is more predictable
        than one averaging 10 pts with std_dev 4 (CV=0.40).
        """
        if projection <= 0:
            return 1.5  # Unknown projection -> conservative

        cv = raw_std / projection

        # Typical CVs by prop type (from NBA stats):
        # Points: ~0.25-0.35, Rebounds: ~0.30-0.45, Assists: ~0.35-0.50
        # Steals/blocks: ~0.60-0.80 (very volatile)
        volatile_props = {"steals", "blocks", "stls_blks", "turnovers"}

        if prop_type in volatile_props:
            # Higher baseline CV expected -- don't penalize as much
            if cv > 0.60:
                return 1.2
            return 1.0
        else:
            if cv > 0.40:
                return 1.3
            elif cv > 0.30:
                return 1.1
            return 1.0

    def _proximity_multiplier(self, projection: float, line: float, calibrated_std: float) -> float:
        """
        When projection is very close to the line, small errors in the
        projection have large effects on the probability.

        The probability function is steepest near 50% -- so a projection
        that is within 0.5 sigma of the line is much more uncertain.
        """
        if calibrated_std <= 0:
            return 1.5

        z_dist = abs(projection - line) / calibrated_std

        if z_dist < 0.25:
            return 1.5  # Very close to line: high uncertainty
        elif z_dist < 0.50:
            return 1.3
        elif z_dist < 0.75:
            return 1.1
        else:
            return 1.0  # Far from line: projection direction is clear

    def _get_calibrated_std(self, raw_std: float, prop_type: str) -> float:
        """Apply calibration and floor, matching EdgeCalculator logic."""
        calibrated = raw_std * STD_DEV_CALIBRATION
        std_key = PROP_TO_STD_KEY.get(prop_type, "pts")
        min_floor = MIN_STD_DEV.get(std_key, 3.0)
        return max(calibrated, min_floor)

    def _compute_over_prob(self, projection: float, line: float, calibrated_std: float) -> float:
        """Compute P(over) matching EdgeCalculator's logic."""
        if calibrated_std <= 0:
            return 0.55 if projection > line else 0.45

        z = (line - projection) / calibrated_std
        over_prob = 1.0 - scipy_stats.norm.cdf(z)

        # Same damping as EdgeCalculator
        if over_prob > 0.60:
            over_prob = 0.60 + (over_prob - 0.60) * 0.30
        elif over_prob < 0.40:
            over_prob = 0.40 - (0.40 - over_prob) * 0.30

        return max(0.32, min(0.68, over_prob))

    def _classify_uncertainty(self, ci_width: float) -> str:
        """Classify uncertainty level based on CI width."""
        if ci_width <= UNCERTAINTY_THRESHOLDS["low"]:
            return "low"
        elif ci_width <= UNCERTAINTY_THRESHOLDS["medium"]:
            return "medium"
        elif ci_width <= UNCERTAINTY_THRESHOLDS["high"]:
            return "high"
        else:
            return "very_high"


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def get_uncertainty_model() -> PredictionUncertainty:
    """Factory function for easy import."""
    return PredictionUncertainty()


# =============================================================================
# DEMO / SELF-TEST
# =============================================================================

if __name__ == "__main__":
    model = PredictionUncertainty()

    print("=" * 70)
    print("  PREDICTION UNCERTAINTY DEMO")
    print("=" * 70)
    print()

    # Scenario 1: Confident pick (lots of data, consistent player, far from line)
    result = model.estimate_uncertainty(
        projection=28.5, line=24.5, std_dev=4.0,
        num_games=20,
        adjustments_dict={"base": 28.5, "defense": -0.5, "pace": 0.3, "home_away": 0.2},
        prop_type="points",
    )
    print("Scenario 1: Confident (20 games, consistent, far from line)")
    print(f"  Projection: 28.5 | Line: 24.5 | Std Dev: 4.0")
    print(f"  P(over) = {result['probability']:.1%} "
          f"  CI: [{result['ci_lower']:.1%}, {result['ci_upper']:.1%}] "
          f"  Width: {result['ci_width']:.1%}")
    print(f"  Uncertainty: {result['uncertainty']} | Kelly adj: {result['kelly_adjustment']:.2f}")
    print()

    # Scenario 2: Uncertain pick (few games, volatile, close to line, disagreement)
    result2 = model.estimate_uncertainty(
        projection=10.2, line=9.5, std_dev=3.5,
        num_games=7,
        adjustments_dict={"base": 10.0, "defense": -1.5, "pace": 1.2, "tracking": -0.8, "streak": 0.5},
        prop_type="rebounds",
    )
    print("Scenario 2: Uncertain (7 games, volatile, close to line, disagreeing adjustments)")
    print(f"  Projection: 10.2 | Line: 9.5 | Std Dev: 3.5")
    print(f"  P(over) = {result2['probability']:.1%} "
          f"  CI: [{result2['ci_lower']:.1%}, {result2['ci_upper']:.1%}] "
          f"  Width: {result2['ci_width']:.1%}")
    print(f"  Uncertainty: {result2['uncertainty']} | Kelly adj: {result2['kelly_adjustment']:.2f}")
    print(f"  Disagreement: {result2['components']['disagreement_score']:.3f}")
    print()

    # Scenario 3: Steals prop (inherently volatile, small numbers)
    result3 = model.estimate_uncertainty(
        projection=1.8, line=1.5, std_dev=1.1,
        num_games=15,
        adjustments_dict={"base": 1.8, "defense": 0.1, "tracking": 0.1},
        prop_type="steals",
    )
    print("Scenario 3: Steals prop (volatile stat, small numbers)")
    print(f"  Projection: 1.8 | Line: 1.5 | Std Dev: 1.1")
    print(f"  P(over) = {result3['probability']:.1%} "
          f"  CI: [{result3['ci_lower']:.1%}, {result3['ci_upper']:.1%}] "
          f"  Width: {result3['ci_width']:.1%}")
    print(f"  Uncertainty: {result3['uncertainty']} | Kelly adj: {result3['kelly_adjustment']:.2f}")
    print()

    # Scenario 4: Strong pick but tiny sample
    result4 = model.estimate_uncertainty(
        projection=22.0, line=18.5, std_dev=5.0,
        num_games=3,
        adjustments_dict={"base": 22.0, "defense": -0.5, "pace": 0.5},
        prop_type="points",
    )
    print("Scenario 4: Strong projection but only 3 games of data")
    print(f"  Projection: 22.0 | Line: 18.5 | Std Dev: 5.0")
    print(f"  P(over) = {result4['probability']:.1%} "
          f"  CI: [{result4['ci_lower']:.1%}, {result4['ci_upper']:.1%}] "
          f"  Width: {result4['ci_width']:.1%}")
    print(f"  Uncertainty: {result4['uncertainty']} | Kelly adj: {result4['kelly_adjustment']:.2f}")
    print()

    # Kelly adjustment comparison table
    print("=" * 70)
    print("  KELLY ADJUSTMENT TABLE")
    print("=" * 70)
    print()
    print(f"  {'Uncertainty':15s} {'Kelly Mult':>12s} {'Effective Kelly':>16s}")
    print(f"  {'-'*15} {'-'*12} {'-'*16}")
    for level in ["low", "medium", "high", "very_high"]:
        adj = KELLY_ADJUSTMENTS[level]
        effective = 0.25 * adj  # Quarter Kelly * adjustment
        print(f"  {level:15s} {adj:>11.2f}x {effective:>15.3f}")
    print()
    print("  Example: 0.25 (quarter Kelly) x 0.7 (medium uncertainty) = 0.175 effective Kelly")
