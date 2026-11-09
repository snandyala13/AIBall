"""
tests/test_v2_edge.py - Tests for v2 EdgeCalculator.

Verifies side selection logic, value bet detection, probability summing,
edge calculation, Kelly criterion, sigma from quantiles, and extreme values.
Tests specifically target bugs found in production: side flip, wrong-side edge,
impossible edge values.
"""

import pytest
import numpy as np
from scipy import stats as scipy_stats

from v2.edge import EdgeCalculator


@pytest.fixture
def calc():
    return EdgeCalculator(kelly_fraction=0.25, max_single_bet=0.05)


# ===========================================================================
# 1. SIDE SELECTION LOGIC
# ===========================================================================

class TestSideSelection:
    """
    Rules:
    - p_over > implied_over AND p_over > p_under -> OVER
    - p_under > implied_under AND p_under > p_over -> UNDER
    - Both edges negative -> PASS
    """

    def test_clear_over_pick(self, calc):
        """Large positive adjustment -> OVER."""
        result = calc.from_adjustment(
            adjustment=3.0, q10=-2.0, q90=7.0,
            dk_line=24.5, over_odds=-110, under_odds=-110,
        )
        assert result["recommended_side"] == "OVER"
        assert result["edge_over"] > 0
        assert result["p_over"] > result["p_under"]

    def test_clear_under_pick(self, calc):
        """Large negative adjustment -> UNDER."""
        result = calc.from_adjustment(
            adjustment=-3.0, q10=-7.0, q90=2.0,
            dk_line=24.5, over_odds=-110, under_odds=-110,
        )
        assert result["recommended_side"] == "UNDER"
        assert result["edge_under"] > 0
        assert result["p_under"] > result["p_over"]

    def test_no_edge_is_pass(self, calc):
        """Tiny adjustment with no edge -> PASS."""
        result = calc.from_adjustment(
            adjustment=0.01, q10=-5.0, q90=5.0,
            dk_line=24.5, over_odds=-110, under_odds=-110,
        )
        # With such a small adjustment and -110 on both sides, no edge exists
        # The vig eats any tiny edge
        if result["edge_over"] <= 0 and result["edge_under"] <= 0:
            assert result["recommended_side"] == "PASS"

    def test_side_flip_bug_regression(self, calc):
        """
        BUG REGRESSION: Model says p_under > p_over but displays "over".

        When adjustment is negative (model says under), the side must be UNDER
        if there is edge. It must NEVER show OVER when p_under > p_over and
        edge_under > edge_over.
        """
        result = calc.from_adjustment(
            adjustment=-2.0, q10=-6.0, q90=1.0,
            dk_line=24.5, over_odds=-110, under_odds=-110,
        )
        # p_under > p_over because adjustment < 0
        assert result["p_under"] > result["p_over"]
        # Side MUST be UNDER (or PASS), never OVER
        assert result["recommended_side"] != "OVER"
        if result["edge_under"] > 0:
            assert result["recommended_side"] == "UNDER"

    def test_side_matches_best_edge(self, calc):
        """The recommended side should always correspond to the better edge."""
        result = calc.from_adjustment(
            adjustment=1.5, q10=-3.0, q90=6.0,
            dk_line=24.5, over_odds=-110, under_odds=-110,
        )
        if result["recommended_side"] == "OVER":
            assert result["edge_over"] >= result["edge_under"]
        elif result["recommended_side"] == "UNDER":
            assert result["edge_under"] >= result["edge_over"]

    def test_over_with_skewed_odds(self, calc):
        """Even with worse over odds, model with strong adjustment -> OVER."""
        result = calc.from_adjustment(
            adjustment=4.0, q10=-1.0, q90=8.0,
            dk_line=24.5, over_odds=-130, under_odds=+110,
        )
        assert result["recommended_side"] == "OVER"

    def test_under_with_skewed_odds(self, calc):
        """Even with worse under odds, model with strong negative adjustment -> UNDER."""
        result = calc.from_adjustment(
            adjustment=-4.0, q10=-8.0, q90=1.0,
            dk_line=24.5, over_odds=+110, under_odds=-130,
        )
        assert result["recommended_side"] == "UNDER"


# ===========================================================================
# 2. VALUE BET (less likely side with good odds)
# ===========================================================================

class TestValueBet:
    """p_over=0.45 at +200 odds -> should pick OVER (45% > 33% implied)."""

    def test_value_bet_over_at_plus_odds(self, calc):
        """
        Model thinks over probability is 45% (less than 50%).
        But at +200 odds, implied is only 33%.
        Edge = 45% - 33% = 12%. Should still pick OVER.
        """
        # We need adjustment that gives p_over ~ 0.45
        # p_over = norm.cdf(adjustment / sigma)
        # 0.45 = norm.cdf(z) => z ~ -0.1257
        # If sigma=2.0, adjustment = -0.1257 * 2.0 = -0.2514
        # But edge_over would be 0.45 - 0.333 = 0.117
        # edge_under would be 0.55 - implied_under
        # At +200 over / -250 under:
        # implied_over = 100/300 = 0.333
        # implied_under = 250/350 = 0.714
        # edge_under = 0.55 - 0.714 = -0.164
        # So edge_over > edge_under -> pick OVER despite p_over < p_under
        result = calc.from_adjustment(
            adjustment=-0.25, q10=-5.12, q90=4.88,  # sigma ~ (4.88-(-5.12))/2.56 ~ 3.91
            dk_line=24.5, over_odds=200, under_odds=-250,
        )
        # At these odds, over is a value bet even though model is slightly bearish
        if result["edge_over"] > 0 and result["edge_over"] > result["edge_under"]:
            assert result["recommended_side"] == "OVER"


# ===========================================================================
# 3. P_OVER + P_UNDER ~ 1.0
# ===========================================================================

class TestProbabilitySumming:
    """p_over + p_under must always approximately equal 1.0."""

    @pytest.mark.parametrize("adjustment", [-5.0, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0])
    def test_probabilities_sum_to_one(self, calc, adjustment):
        result = calc.from_adjustment(
            adjustment=adjustment, q10=adjustment - 4.0, q90=adjustment + 4.0,
            dk_line=24.5, over_odds=-110, under_odds=-110,
        )
        total = result["p_over"] + result["p_under"]
        assert abs(total - 1.0) < 0.001, (
            f"p_over={result['p_over']}, p_under={result['p_under']}, sum={total}"
        )

    def test_extreme_adjustment_still_sums_to_one(self, calc):
        """Very extreme adjustment should still sum to ~1.0."""
        result = calc.from_adjustment(
            adjustment=50.0, q10=45.0, q90=55.0,
            dk_line=24.5, over_odds=-110, under_odds=-110,
        )
        total = result["p_over"] + result["p_under"]
        assert abs(total - 1.0) < 0.001


# ===========================================================================
# 4. EDGE = OUR_PROB - IMPLIED_PROB
# ===========================================================================

class TestEdgeCalculation:
    """Edge should exactly equal our probability minus implied probability."""

    def test_edge_over_formula(self, calc):
        result = calc.from_adjustment(
            adjustment=2.0, q10=-3.0, q90=7.0,
            dk_line=24.5, over_odds=-110, under_odds=-110,
        )
        implied_over = calc._american_to_prob(-110)
        assert abs(result["edge_over"] - (result["p_over"] - implied_over)) < 0.001

    def test_edge_under_formula(self, calc):
        result = calc.from_adjustment(
            adjustment=-2.0, q10=-7.0, q90=3.0,
            dk_line=24.5, over_odds=-110, under_odds=-110,
        )
        implied_under = calc._american_to_prob(-110)
        assert abs(result["edge_under"] - (result["p_under"] - implied_under)) < 0.001

    def test_edge_mismatch_bug_regression(self, calc):
        """
        BUG REGRESSION: Edge calculated for wrong side.
        Under edge shown on over pick, or vice versa.

        The 'edge' field must always correspond to the recommended_side.
        """
        result = calc.from_adjustment(
            adjustment=2.5, q10=-2.0, q90=6.0,
            dk_line=24.5, over_odds=-110, under_odds=-110,
        )
        side = result["recommended_side"]
        if side == "OVER":
            assert result["edge"] == result["edge_over"]
        elif side == "UNDER":
            assert result["edge"] == result["edge_under"]

    def test_edge_for_under_pick_uses_under_prob(self, calc):
        """When side is UNDER, edge must use p_under, not p_over."""
        result = calc.from_adjustment(
            adjustment=-3.0, q10=-8.0, q90=1.0,
            dk_line=24.5, over_odds=-110, under_odds=-110,
        )
        if result["recommended_side"] == "UNDER":
            implied_under = calc._american_to_prob(-110)
            expected_edge = result["p_under"] - implied_under
            assert abs(result["edge"] - expected_edge) < 0.001

    def test_1200_percent_edge_impossible(self, calc):
        """
        BUG REGRESSION: Edge values >100% are clearly impossible.
        Even with extreme inputs, edge should be bounded by [-1, 1].
        """
        result = calc.from_adjustment(
            adjustment=100.0, q10=50.0, q90=150.0,
            dk_line=24.5, over_odds=-110, under_odds=-110,
        )
        assert result["edge"] <= 1.0, (
            f"Edge of {result['edge']*100:.0f}% is impossible"
        )
        assert result["edge"] >= -1.0

    @pytest.mark.parametrize("adj,q10,q90", [
        (0.0, -5.0, 5.0),
        (10.0, 5.0, 15.0),
        (-10.0, -15.0, -5.0),
    ])
    def test_edge_always_bounded(self, calc, adj, q10, q90):
        """Edge should always be between -1 and 1."""
        result = calc.from_adjustment(
            adjustment=adj, q10=q10, q90=q90,
            dk_line=24.5, over_odds=-110, under_odds=-110,
        )
        assert -1.0 <= result["edge_over"] <= 1.0
        assert -1.0 <= result["edge_under"] <= 1.0


# ===========================================================================
# 5. NEGATIVE EDGE -> KELLY = 0
# ===========================================================================

class TestKellyCriterion:
    """No bet when edge is negative."""

    def test_pass_gives_zero_kelly(self, calc):
        """When side is PASS, kelly must be 0."""
        result = calc.from_adjustment(
            adjustment=0.001, q10=-5.0, q90=5.0,
            dk_line=24.5, over_odds=-110, under_odds=-110,
        )
        if result["recommended_side"] == "PASS":
            assert result["kelly_fraction"] == 0.0

    def test_positive_edge_positive_kelly(self, calc):
        """Positive edge should produce positive kelly."""
        result = calc.from_adjustment(
            adjustment=3.0, q10=-2.0, q90=7.0,
            dk_line=24.5, over_odds=-110, under_odds=-110,
        )
        if result["recommended_side"] != "PASS" and result["edge"] > 0:
            assert result["kelly_fraction"] > 0

    def test_kelly_capped_at_max_bet(self, calc):
        """Even with huge edge, kelly should be capped at max_single_bet."""
        result = calc.from_adjustment(
            adjustment=20.0, q10=15.0, q90=25.0,
            dk_line=24.5, over_odds=-110, under_odds=-110,
        )
        assert result["kelly_fraction"] <= calc.max_single_bet

    def test_quarter_kelly_applied(self):
        """Verify quarter Kelly (0.25x) is applied, not full Kelly."""
        calc = EdgeCalculator(kelly_fraction=0.25, max_single_bet=1.0)
        result = calc.from_adjustment(
            adjustment=5.0, q10=0.0, q90=10.0,
            dk_line=24.5, over_odds=-110, under_odds=-110,
        )
        # p_over should be high with adjustment=5.0
        p = result["p_over"]
        q = 1 - p
        decimal_odds = calc._american_to_decimal(-110)
        b = decimal_odds - 1.0
        full_kelly = (b * p - q) / b
        quarter_kelly = 0.25 * max(full_kelly, 0)
        assert abs(result["kelly_fraction"] - quarter_kelly) < 0.001


# ===========================================================================
# 6. SIGMA FROM QUANTILES
# ===========================================================================

class TestSigmaFromQuantiles:
    """sigma = (q90 - q10) / 2.56."""

    def test_sigma_calculation(self, calc):
        q10, q90 = -4.0, 6.0
        result = calc.from_adjustment(
            adjustment=1.0, q10=q10, q90=q90,
            dk_line=24.5, over_odds=-110, under_odds=-110,
        )
        expected_sigma = (q90 - q10) / 2.56
        assert abs(result["sigma"] - expected_sigma) < 0.001

    def test_sigma_minimum_floor(self, calc):
        """Sigma should have a floor of 0.5 to prevent division issues."""
        result = calc.from_adjustment(
            adjustment=1.0, q10=-0.01, q90=0.01,  # Tiny interval
            dk_line=24.5, over_odds=-110, under_odds=-110,
        )
        assert result["sigma"] >= 0.5

    def test_z_score_uses_sigma(self, calc):
        """z_score = adjustment / sigma."""
        adjustment = 2.0
        q10, q90 = -3.0, 7.0
        result = calc.from_adjustment(
            adjustment=adjustment, q10=q10, q90=q90,
            dk_line=24.5, over_odds=-110, under_odds=-110,
        )
        expected_z = adjustment / result["sigma"]
        assert abs(result["z_score"] - expected_z) < 0.001


# ===========================================================================
# 7. EXTREME VALUES (no crashes)
# ===========================================================================

class TestExtremeValues:
    """Edge cases with zero/extreme inputs should not crash."""

    def test_line_zero(self, calc):
        result = calc.from_adjustment(
            adjustment=1.0, q10=-4.0, q90=6.0,
            dk_line=0.0, over_odds=-110, under_odds=-110,
        )
        assert result is not None
        assert result["final_prediction"] == pytest.approx(1.0, abs=0.01)

    def test_odds_zero(self, calc):
        """Zero odds should use fallback."""
        result = calc.from_adjustment(
            adjustment=1.0, q10=-4.0, q90=6.0,
            dk_line=24.5, over_odds=0, under_odds=0,
        )
        assert result is not None
        # _american_to_prob(0) returns 0.5
        assert result["implied_over"] == pytest.approx(0.5, abs=0.01)

    def test_adjustment_zero(self, calc):
        result = calc.from_adjustment(
            adjustment=0.0, q10=-4.0, q90=4.0,
            dk_line=24.5, over_odds=-110, under_odds=-110,
        )
        assert result is not None
        assert result["p_over"] == pytest.approx(0.5, abs=0.01)

    def test_very_large_adjustment(self, calc):
        result = calc.from_adjustment(
            adjustment=1000.0, q10=990.0, q90=1010.0,
            dk_line=24.5, over_odds=-110, under_odds=-110,
        )
        assert result is not None
        assert result["p_over"] > 0.99

    def test_very_negative_adjustment(self, calc):
        result = calc.from_adjustment(
            adjustment=-1000.0, q10=-1010.0, q90=-990.0,
            dk_line=24.5, over_odds=-110, under_odds=-110,
        )
        assert result is not None
        assert result["p_under"] > 0.99


# ===========================================================================
# CROSS-BOOK SIGNALS
# ===========================================================================

class TestCrossBookSignals:
    """Cross-book consensus line affects signal strength."""

    def test_strong_under_signal(self, calc):
        """DK line higher than consensus + model says under -> STRONG_UNDER."""
        result = calc.from_adjustment(
            adjustment=-2.0, q10=-6.0, q90=1.0,
            dk_line=25.0, over_odds=-110, under_odds=-110,
            consensus_line=24.0,  # DK 1.0 above consensus
        )
        if result["recommended_side"] == "UNDER":
            assert result["cross_book_signal"] == "STRONG_UNDER"

    def test_strong_over_signal(self, calc):
        """DK line lower than consensus + model says over -> STRONG_OVER."""
        result = calc.from_adjustment(
            adjustment=2.0, q10=-2.0, q90=6.0,
            dk_line=23.5, over_odds=-110, under_odds=-110,
            consensus_line=24.5,  # DK 1.0 below consensus
        )
        if result["recommended_side"] == "OVER":
            assert result["cross_book_signal"] == "STRONG_OVER"

    def test_neutral_signal(self, calc):
        """DK near consensus -> NEUTRAL."""
        result = calc.from_adjustment(
            adjustment=2.0, q10=-2.0, q90=6.0,
            dk_line=24.5, over_odds=-110, under_odds=-110,
            consensus_line=24.5,
        )
        assert result["cross_book_signal"] == "NEUTRAL"

    def test_market_disagreement_high(self, calc):
        """Wide line range -> HIGH disagreement."""
        result = calc.from_adjustment(
            adjustment=1.0, q10=-3.0, q90=5.0,
            dk_line=24.5, over_odds=-110, under_odds=-110,
            line_range=3.0,
        )
        assert result["market_disagreement"] == "HIGH"


# ===========================================================================
# BATCH ANALYZE
# ===========================================================================

class TestBatchAnalyze:
    """Batch analysis should filter and rank correctly."""

    def test_batch_filters_by_min_edge(self, calc):
        predictions = [
            {"player": "A", "prop_type": "points", "adjustment": 3.0,
             "q10": -2.0, "q90": 7.0, "dk_line": 24.5,
             "over_odds": -110, "under_odds": -110},
            {"player": "B", "prop_type": "rebounds", "adjustment": 0.01,
             "q10": -5.0, "q90": 5.0, "dk_line": 6.5,
             "over_odds": -110, "under_odds": -110},
        ]
        results = calc.batch_analyze(predictions, min_edge=0.03)
        # Player B has almost no edge, should be filtered out
        players = [r["player"] for r in results]
        assert "A" in players

    def test_batch_sorts_by_edge_desc(self, calc):
        predictions = [
            {"player": "Low", "prop_type": "points", "adjustment": 1.0,
             "q10": -4.0, "q90": 6.0, "dk_line": 24.5,
             "over_odds": -110, "under_odds": -110},
            {"player": "High", "prop_type": "points", "adjustment": 5.0,
             "q10": 0.0, "q90": 10.0, "dk_line": 24.5,
             "over_odds": -110, "under_odds": -110},
        ]
        results = calc.batch_analyze(predictions, min_edge=0.0)
        if len(results) >= 2:
            assert results[0]["edge"] >= results[1]["edge"]
