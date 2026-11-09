#!/usr/bin/env python3
"""
v2/edge.py - Convert line-anchor model output to betting decisions.

Architecture:
  predicted_adjustment = model.predict(features)  # e.g., +1.5 or -2.3
  final_prediction = DK_line + predicted_adjustment

  If adjustment > 0: model thinks over will hit
  If adjustment < 0: model thinks under will hit
  Magnitude = confidence

  P(over) from quantile models:
    sigma = (q90_pred - q10_pred) / 2.56
    p_over = 1 - norm.cdf(-predicted_adjustment / sigma)

  Or equivalently:
    p_over = norm.cdf(predicted_adjustment / sigma)

Usage:
    from v2.edge import EdgeCalculator
    calc = EdgeCalculator()
    result = calc.from_adjustment(
        adjustment=1.5, q10=-3.2, q90=5.1,
        dk_line=24.5, over_odds=-110, under_odds=-110,
    )
"""

import logging
from typing import Dict, Optional

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


class EdgeCalculator:
    """
    Convert line-anchor model predictions to edge and sizing decisions.

    The model predicts an adjustment to the DK line:
      adjustment = actual - DK_line (predicted)

    A positive adjustment means the model expects the actual to exceed the line.
    We convert this to P(over) using the quantile regression uncertainty.
    """

    def __init__(self, kelly_fraction: float = 0.25, max_single_bet: float = 0.05):
        """
        Args:
            kelly_fraction: Fraction of full Kelly to use (0.25 = quarter Kelly)
            max_single_bet: Maximum bet size as fraction of bankroll
        """
        self.kelly_fraction = kelly_fraction
        self.max_single_bet = max_single_bet

    def from_adjustment(
        self,
        adjustment: float,
        q10: float,
        q90: float,
        dk_line: float,
        over_odds: int,
        under_odds: int,
        consensus_line: Optional[float] = None,
        line_range: Optional[float] = None,
    ) -> dict:
        """
        Convert a model adjustment prediction to a betting decision.

        Args:
            adjustment: Model's predicted adjustment (actual - line)
            q10: 10th percentile of adjustment distribution
            q90: 90th percentile of adjustment distribution
            dk_line: DraftKings prop line
            over_odds: DK over odds (American)
            under_odds: DK under odds (American)
            consensus_line: Median line across books (optional, for cross-book edge)
            line_range: Max - min line across books (optional, for disagreement signal)

        Returns:
            dict with edge analysis, recommended side, and sizing
        """
        # Estimate sigma from the 80% interval
        # 80% interval spans 2.56 sigma in a normal distribution
        sigma = max((q90 - q10) / 2.56, 0.5)

        # P(over) = P(actual > line) = P(adjustment > 0)
        # Under the model: adjustment ~ N(predicted_adjustment, sigma)
        # P(adjustment > 0) = 1 - Phi(-adjustment/sigma) = Phi(adjustment/sigma)
        z = adjustment / sigma
        p_over = float(scipy_stats.norm.cdf(z))
        p_under = 1.0 - p_over

        # Implied probabilities from American odds
        implied_over = self._american_to_prob(over_odds)
        implied_under = self._american_to_prob(under_odds)

        # Edge = our probability - market implied probability
        edge_over = p_over - implied_over
        edge_under = p_under - implied_under

        # Determine best side
        if edge_over > edge_under and edge_over > 0:
            side = "OVER"
            edge = edge_over
            fair_prob = p_over
            odds = over_odds
        elif edge_under > 0:
            side = "UNDER"
            edge = edge_under
            fair_prob = p_under
            odds = under_odds
        else:
            side = "PASS"
            edge = max(edge_over, edge_under)
            fair_prob = max(p_over, p_under)
            odds = over_odds if edge_over >= edge_under else under_odds

        # Kelly criterion (fractional Kelly for safety)
        decimal_odds = self._american_to_decimal(odds)
        b = decimal_odds - 1.0  # net payout per dollar
        p = fair_prob
        q = 1.0 - p
        if b > 0 and p > 0:
            full_kelly = (b * p - q) / b
            kelly = self.kelly_fraction * max(full_kelly, 0.0)
        else:
            kelly = 0.0

        if side == "PASS":
            kelly = 0.0
        kelly = min(kelly, self.max_single_bet)

        # Confidence rating (0-5 scale)
        confidence_raw = abs(adjustment) / sigma
        if confidence_raw >= 1.5:
            confidence_stars = 5
        elif confidence_raw >= 1.0:
            confidence_stars = 4
        elif confidence_raw >= 0.6:
            confidence_stars = 3
        elif confidence_raw >= 0.3:
            confidence_stars = 2
        else:
            confidence_stars = 1

        result = {
            # Model output
            "adjustment": round(adjustment, 3),
            "final_prediction": round(dk_line + adjustment, 2),
            "sigma": round(sigma, 3),
            "z_score": round(z, 3),

            # Probabilities
            "p_over": round(p_over, 4),
            "p_under": round(p_under, 4),
            "implied_over": round(implied_over, 4),
            "implied_under": round(implied_under, 4),

            # Edge
            "edge_over": round(edge_over, 4),
            "edge_under": round(edge_under, 4),
            "recommended_side": side,
            "edge": round(edge, 4),

            # Sizing
            "kelly_fraction": round(kelly, 5),

            # Confidence
            "confidence_stars": confidence_stars,
            "confidence_raw": round(confidence_raw, 3),

            # Interval
            "p10_adjustment": round(q10, 2),
            "p90_adjustment": round(q90, 2),
            "p10_prediction": round(dk_line + q10, 2),
            "p90_prediction": round(dk_line + q90, 2),
        }

        # Add cross-book signal if available
        if consensus_line is not None:
            dk_vs_consensus = dk_line - consensus_line
            result["consensus_line"] = round(consensus_line, 2)
            result["dk_vs_consensus"] = round(dk_vs_consensus, 2)
            # If DK's line is higher than consensus and model says under, double signal
            if dk_vs_consensus > 0.5 and side == "UNDER":
                result["cross_book_signal"] = "STRONG_UNDER"
            elif dk_vs_consensus < -0.5 and side == "OVER":
                result["cross_book_signal"] = "STRONG_OVER"
            else:
                result["cross_book_signal"] = "NEUTRAL"

        if line_range is not None:
            result["line_range"] = round(line_range, 2)
            # Wide range = market disagreement = potential opportunity
            result["market_disagreement"] = "HIGH" if line_range >= 2.0 else (
                "MEDIUM" if line_range >= 1.0 else "LOW"
            )

        return result

    def batch_analyze(
        self,
        predictions: list,
        min_edge: float = 0.03,
        max_picks: int = 10,
    ) -> list:
        """
        Analyze a batch of predictions and return ranked picks.

        Args:
            predictions: list of dicts, each with keys:
                player, prop_type, adjustment, q10, q90,
                dk_line, over_odds, under_odds
            min_edge: Minimum edge to include in picks
            max_picks: Maximum number of picks to return

        Returns:
            list of analyzed picks, sorted by edge descending
        """
        analyzed = []
        for pred in predictions:
            result = self.from_adjustment(
                adjustment=pred["adjustment"],
                q10=pred["q10"],
                q90=pred["q90"],
                dk_line=pred["dk_line"],
                over_odds=pred["over_odds"],
                under_odds=pred["under_odds"],
                consensus_line=pred.get("consensus_line"),
                line_range=pred.get("line_range"),
            )
            result["player"] = pred.get("player", "")
            result["prop_type"] = pred.get("prop_type", "")
            result["dk_line"] = pred["dk_line"]

            if result["recommended_side"] != "PASS" and result["edge"] >= min_edge:
                analyzed.append(result)

        # Sort by edge descending
        analyzed.sort(key=lambda x: x["edge"], reverse=True)
        return analyzed[:max_picks]

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _american_to_prob(odds: int) -> float:
        if odds == 0:
            return 0.5
        if odds > 0:
            return 100.0 / (odds + 100.0)
        return abs(odds) / (abs(odds) + 100.0)

    @staticmethod
    def _american_to_decimal(odds: int) -> float:
        if odds == 0:
            return 2.0
        if odds > 0:
            return 1.0 + odds / 100.0
        return 1.0 + 100.0 / abs(odds)


# =============================================================================
# CLI demo
# =============================================================================

def main():
    """Demo the edge calculator with sample data."""
    print("=" * 70)
    print("v2 Edge Calculator - Demo")
    print("=" * 70)

    calc = EdgeCalculator()

    # Sample scenarios
    scenarios = [
        {
            "desc": "Model sees points trending up, line hasn't adjusted",
            "adjustment": 2.5, "q10": -4.0, "q90": 8.0,
            "dk_line": 24.5, "over_odds": -110, "under_odds": -110,
            "consensus_line": 24.0,
        },
        {
            "desc": "Model agrees with line (no edge)",
            "adjustment": 0.1, "q10": -5.0, "q90": 5.0,
            "dk_line": 24.5, "over_odds": -110, "under_odds": -110,
        },
        {
            "desc": "Strong under signal - line too high",
            "adjustment": -3.0, "q10": -8.0, "q90": 1.0,
            "dk_line": 24.5, "over_odds": -110, "under_odds": -110,
            "consensus_line": 23.0,
        },
        {
            "desc": "Moderate over with consensus disagreement (DK outlier low)",
            "adjustment": 1.5, "q10": -3.0, "q90": 6.0,
            "dk_line": 6.5, "over_odds": -125, "under_odds": 100,
            "consensus_line": 7.5,
        },
    ]

    for s in scenarios:
        print(f"\n  Scenario: {s['desc']}")
        result = calc.from_adjustment(
            adjustment=s["adjustment"],
            q10=s["q10"],
            q90=s["q90"],
            dk_line=s["dk_line"],
            over_odds=s["over_odds"],
            under_odds=s["under_odds"],
            consensus_line=s.get("consensus_line"),
        )
        print(f"    Line: {s['dk_line']}, Adjustment: {s['adjustment']:+.1f}")
        print(f"    Final prediction: {result['final_prediction']}")
        print(f"    P(over): {result['p_over']:.1%}, P(under): {result['p_under']:.1%}")
        print(f"    Edge: {result['edge']:+.1%} {result['recommended_side']}")
        print(f"    Kelly: {result['kelly_fraction']:.3%}")
        print(f"    Confidence: {'*' * result['confidence_stars']} ({result['confidence_raw']:.2f})")
        if "cross_book_signal" in result:
            print(f"    Cross-book: {result['cross_book_signal']} "
                  f"(DK vs consensus: {result['dk_vs_consensus']:+.1f})")

    print()


if __name__ == "__main__":
    main()
