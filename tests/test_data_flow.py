"""
tests/test_data_flow.py - Data flow integrity tests.

Traces data through the full pipeline and verifies nothing gets lost
or corrupted as it flows from raw input to serialized API output.
"""

import os
import sys
import math
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# Path setup is handled by pyproject.toml [tool.pytest.ini_options] pythonpath
# and by conftest.py which ensures the main repo root is on sys.path.

from v2.edge import EdgeCalculator
from web.serializers import serialize_v2_prediction


# =========================================================================
# 1. all_books survives the full pipeline
# =========================================================================

class TestAllBooksSurvivesPipeline:
    """all_books from PropLine must survive through every layer to the API."""

    def test_all_books_survives_format_prediction(self, mock_all_books):
        """all_books in raw prediction -> format_prediction_for_web -> output."""
        from v2.engine import V2AnalysisEngine

        raw_pred = {
            "player": "Jayson Tatum",
            "prop_type": "points",
            "line": 24.5,
            "predicted": 26.0,
            "adjustment": 1.5,
            "confidence": 0.7,
            "confidence_stars": 3,
            "p_over": 0.61,
            "p_under": 0.39,
            "edge_over": 0.08,
            "edge_under": -0.13,
            "implied_over": 0.5238,
            "implied_under": 0.5238,
            "recommended_side": "OVER",
            "edge": 0.08,
            "kelly_fraction": 0.01,
            "over_odds": -110,
            "under_odds": -110,
            "all_books": mock_all_books,
            "sigma": 2.4,
            "z_score": 0.6,
        }

        # Simulate the engine formatting step
        engine = MagicMock(spec=V2AnalysisEngine)
        engine.format_prediction_for_web = V2AnalysisEngine.format_prediction_for_web

        formatted = engine.format_prediction_for_web(engine, raw_pred)
        assert "all_books" in formatted
        assert len(formatted["all_books"]) == len(mock_all_books)
        assert "draftkings" in formatted["all_books"]

    def test_all_books_survives_serialization(self, mock_all_books):
        """all_books in formatted prediction -> serialize_v2_prediction -> output."""
        formatted = {
            "player": "Jayson Tatum",
            "prop_type": "points",
            "side": "over",
            "line": 24.5,
            "dk_odds": -110,
            "over_odds": -110,
            "under_odds": -110,
            "predicted_value": 26.0,
            "predicted_adjustment": 1.5,
            "p10": 20.0,
            "p90": 32.0,
            "uncertainty": 12.0,
            "confidence": 0.7,
            "confidence_stars": 3,
            "edge_percent": 8.0,
            "edge": 0.08,
            "our_probability": 0.61,
            "implied_probability": 0.52,
            "p_over": 0.61,
            "p_under": 0.39,
            "kelly_fraction": 0.01,
            "recommended_units": 1.0,
            "all_books": mock_all_books,
            "sigma": 2.4,
            "z_score": 0.6,
            "cross_book_signal": "NEUTRAL",
            "source": "v2_line_anchor",
        }

        serialized = serialize_v2_prediction(formatted)
        assert "all_books" in serialized
        assert len(serialized["all_books"]) == len(mock_all_books)
        for book_name in mock_all_books:
            assert book_name in serialized["all_books"]


# =========================================================================
# 2. Edge calculation consistency
# =========================================================================

class TestEdgeCalculationConsistency:
    """Edge values must be mathematically consistent."""

    def test_edge_matches_manual_calculation(self, edge_calculator):
        """Calculate edge by hand and verify it matches the calculator output."""
        from scipy.stats import norm

        adjustment = 2.5
        q10 = -2.0
        q90 = 7.0
        dk_line = 24.5
        over_odds = -110
        under_odds = -110

        result = edge_calculator.from_adjustment(
            adjustment=adjustment, q10=q10, q90=q90,
            dk_line=dk_line, over_odds=over_odds, under_odds=under_odds,
        )

        # Manual calculation
        sigma = max((q90 - q10) / 2.56, 0.5)
        z = adjustment / sigma
        expected_p_over = float(norm.cdf(z))
        expected_p_under = 1.0 - expected_p_over

        # Implied probabilities
        expected_implied_over = 110 / (110 + 100)  # -110 -> 0.5238
        expected_implied_under = expected_implied_over

        expected_edge_over = expected_p_over - expected_implied_over
        expected_edge_under = expected_p_under - expected_implied_under

        assert abs(result["p_over"] - expected_p_over) < 0.001
        assert abs(result["p_under"] - expected_p_under) < 0.001
        assert abs(result["edge_over"] - expected_edge_over) < 0.001
        assert abs(result["edge_under"] - expected_edge_under) < 0.001

    def test_edge_sign_matches_side(self, edge_calculator):
        """If recommended_side is OVER, edge_over should be the reported edge."""
        result = edge_calculator.from_adjustment(
            adjustment=3.0, q10=-1.0, q90=7.0,
            dk_line=24.5, over_odds=-110, under_odds=-110,
        )
        side = result["recommended_side"]
        if side == "OVER":
            assert result["edge"] == result["edge_over"]
            assert result["edge_over"] > 0
        elif side == "UNDER":
            assert result["edge"] == result["edge_under"]
            assert result["edge_under"] > 0

    def test_p_over_and_p_under_sum_to_one(self, edge_calculator):
        """Probabilities must sum to 1.0."""
        result = edge_calculator.from_adjustment(
            adjustment=1.5, q10=-3.0, q90=6.0,
            dk_line=24.5, over_odds=-110, under_odds=-110,
        )
        total = result["p_over"] + result["p_under"]
        assert abs(total - 1.0) < 0.0001

    def test_final_prediction_equals_line_plus_adjustment(self, edge_calculator):
        """final_prediction must equal dk_line + adjustment."""
        for adj in [-5.0, -1.0, 0.0, 1.0, 5.0]:
            result = edge_calculator.from_adjustment(
                adjustment=adj, q10=adj - 3.0, q90=adj + 3.0,
                dk_line=24.5, over_odds=-110, under_odds=-110,
            )
            expected = 24.5 + adj
            assert abs(result["final_prediction"] - expected) < 0.01


# =========================================================================
# 3. Implied probability math
# =========================================================================

class TestImpliedProbabilityMath:
    """Verify implied probability conversions are correct."""

    @pytest.mark.parametrize("odds,expected_prob", [
        (-110, 110 / 210),  # 0.5238
        (-150, 150 / 250),  # 0.60
        (100, 100 / 200),   # 0.50
        (150, 100 / 250),   # 0.40
        (-200, 200 / 300),  # 0.6667
        (200, 100 / 300),   # 0.3333
    ])
    def test_american_to_prob(self, odds, expected_prob):
        prob = EdgeCalculator._american_to_prob(odds)
        assert abs(prob - expected_prob) < 0.001

    @pytest.mark.parametrize("odds,expected_decimal", [
        (-110, 1 + 100 / 110),  # 1.909
        (100, 2.0),
        (200, 3.0),
        (-200, 1.5),
    ])
    def test_american_to_decimal(self, odds, expected_decimal):
        dec = EdgeCalculator._american_to_decimal(odds)
        assert abs(dec - expected_decimal) < 0.01


# =========================================================================
# 4. Kelly sizing sanity
# =========================================================================

class TestKellySizing:
    """Kelly fraction must be bounded and correct."""

    def test_kelly_never_negative(self, edge_calculator):
        """Kelly should never be negative."""
        for adj in [-10, -5, -1, 0, 1, 5, 10]:
            result = edge_calculator.from_adjustment(
                adjustment=float(adj), q10=float(adj - 3), q90=float(adj + 3),
                dk_line=24.5, over_odds=-110, under_odds=-110,
            )
            assert result["kelly_fraction"] >= 0

    def test_kelly_capped_at_max(self, edge_calculator):
        """Kelly should not exceed max_single_bet (default 5%)."""
        result = edge_calculator.from_adjustment(
            adjustment=20.0, q10=15.0, q90=25.0,
            dk_line=24.5, over_odds=-110, under_odds=-110,
        )
        assert result["kelly_fraction"] <= 0.05

    def test_kelly_zero_on_pass(self, edge_calculator):
        """When side is PASS, Kelly should be 0."""
        result = edge_calculator.from_adjustment(
            adjustment=0.01, q10=-5.0, q90=5.0,
            dk_line=24.5, over_odds=-110, under_odds=-110,
        )
        if result["recommended_side"] == "PASS":
            assert result["kelly_fraction"] == 0.0


# =========================================================================
# 5. Confidence stars mapping
# =========================================================================

class TestConfidenceStars:
    """Confidence stars should be 1-5 based on |adjustment|/sigma."""

    def test_confidence_stars_range(self, edge_calculator):
        """Stars should always be 1-5."""
        for adj in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
            result = edge_calculator.from_adjustment(
                adjustment=adj, q10=adj - 3.0, q90=adj + 3.0,
                dk_line=24.5, over_odds=-110, under_odds=-110,
            )
            assert 1 <= result["confidence_stars"] <= 5

    def test_higher_adjustment_more_stars(self, edge_calculator):
        """Larger |adjustment| relative to sigma should mean more stars."""
        small = edge_calculator.from_adjustment(
            adjustment=0.1, q10=-3.0, q90=3.0,
            dk_line=24.5, over_odds=-110, under_odds=-110,
        )
        large = edge_calculator.from_adjustment(
            adjustment=5.0, q10=2.0, q90=8.0,
            dk_line=24.5, over_odds=-110, under_odds=-110,
        )
        assert large["confidence_stars"] >= small["confidence_stars"]


# =========================================================================
# 6. Cross-book signal logic
# =========================================================================

class TestCrossBookSignal:
    """Cross-book signals require consensus_line and matching side."""

    def test_strong_under_when_dk_higher_and_model_under(self, edge_calculator):
        """DK line above consensus + model says under = STRONG_UNDER."""
        result = edge_calculator.from_adjustment(
            adjustment=-3.0, q10=-7.0, q90=1.0,
            dk_line=25.5, over_odds=-110, under_odds=-110,
            consensus_line=24.0,
        )
        assert result["cross_book_signal"] == "STRONG_UNDER"

    def test_strong_over_when_dk_lower_and_model_over(self, edge_calculator):
        """DK line below consensus + model says over = STRONG_OVER."""
        result = edge_calculator.from_adjustment(
            adjustment=3.0, q10=-1.0, q90=7.0,
            dk_line=23.5, over_odds=-110, under_odds=-110,
            consensus_line=25.0,
        )
        assert result["cross_book_signal"] == "STRONG_OVER"

    def test_neutral_when_no_consensus(self, edge_calculator):
        """Without consensus_line, no cross-book signal."""
        result = edge_calculator.from_adjustment(
            adjustment=3.0, q10=-1.0, q90=7.0,
            dk_line=24.5, over_odds=-110, under_odds=-110,
        )
        assert "cross_book_signal" not in result

    def test_market_disagreement_from_line_range(self, edge_calculator):
        """Wide line range signals market disagreement."""
        result = edge_calculator.from_adjustment(
            adjustment=1.0, q10=-3.0, q90=5.0,
            dk_line=24.5, over_odds=-110, under_odds=-110,
            consensus_line=24.5,
            line_range=3.0,
        )
        assert result["market_disagreement"] == "HIGH"


# =========================================================================
# 7. Feature builder uses prior games only (no lookahead)
# =========================================================================

class TestNoLookahead:
    """Feature builder must only use games BEFORE the target date."""

    def test_features_exclude_future_games(self, mock_player_history):
        from v2.features import FeatureBuilder
        fb = FeatureBuilder()
        fb.player_history["Test Player"] = mock_player_history

        # Target date is before all games (none should be used)
        game_date = "2025-01-01"
        result = fb.get_features_for_leg(
            "Test Player", "points", game_date,
            {"line": 20.0, "over_odds": -110, "under_odds": -110},
        )
        # No prior games -> should return None
        assert result is None

    def test_features_use_only_prior_games(self, mock_player_history):
        from v2.features import FeatureBuilder
        fb = FeatureBuilder()
        fb.player_history["Test Player"] = mock_player_history

        # Pick a date after all games
        game_date = "2026-12-01"
        result = fb.get_features_for_leg(
            "Test Player", "points", game_date,
            {"line": 20.0, "over_odds": -110, "under_odds": -110},
        )
        # Should have features from all 20 games
        assert result is not None


# =========================================================================
# 8. Prediction dict key consistency
# =========================================================================

class TestPredictionKeyConsistency:
    """Verify the key names match at each pipeline stage."""

    def test_format_prediction_has_predicted_value(self):
        """format_prediction_for_web must output 'predicted_value'."""
        from v2.engine import V2AnalysisEngine

        raw = {
            "player": "Test", "prop_type": "points", "line": 24.5,
            "predicted": 26.0, "adjustment": 1.5,
            "confidence": 0.7, "confidence_stars": 3,
            "p_over": 0.6, "p_under": 0.4,
            "edge_over": 0.08, "edge_under": -0.13,
            "implied_over": 0.52, "implied_under": 0.52,
            "recommended_side": "OVER", "edge": 0.08,
            "kelly_fraction": 0.01,
            "over_odds": -110, "under_odds": -110,
            "all_books": {},
            "sigma": 2.4, "z_score": 0.6,
        }

        engine = MagicMock(spec=V2AnalysisEngine)
        engine.format_prediction_for_web = V2AnalysisEngine.format_prediction_for_web
        formatted = engine.format_prediction_for_web(engine, raw)

        # Must have both predicted_value and projected_value (alias)
        assert "predicted_value" in formatted
        assert "projected_value" in formatted
        assert formatted["predicted_value"] == formatted["projected_value"]

    def test_serialized_prediction_has_predicted_value(self, mock_v2_formatted_prediction):
        """serialize_v2_prediction must output 'predicted_value'."""
        serialized = serialize_v2_prediction(mock_v2_formatted_prediction)
        assert "predicted_value" in serialized
        assert serialized["predicted_value"] == mock_v2_formatted_prediction["predicted_value"]
