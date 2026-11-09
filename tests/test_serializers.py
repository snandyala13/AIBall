"""
tests/test_serializers.py - Serialization layer tests.

Verifies that v2 predictions are serialized with all required fields,
correct types, and sensible value ranges for the frontend JS renderer.
"""

import os
import sys

import pytest

# Path setup is handled by pyproject.toml [tool.pytest.ini_options] pythonpath
# and by conftest.py which ensures the main repo root is on sys.path.

from web.serializers import serialize_v2_prediction


# =========================================================================
# Required fields
# =========================================================================

REQUIRED_V2_FIELDS = [
    "player", "prop_type", "side", "line", "dk_odds",
    "predicted_value", "edge_percent", "p_over", "p_under",
    "over_odds", "under_odds", "confidence",
]


class TestSerializeV2PredictionFields:
    """serialize_v2_prediction must include every field the JS renderer uses."""

    def test_all_required_fields_present(self, mock_v2_formatted_prediction):
        result = serialize_v2_prediction(mock_v2_formatted_prediction)
        for field in REQUIRED_V2_FIELDS:
            assert field in result, f"Missing required field: {field}"

    def test_predicted_value_uses_correct_key(self, mock_v2_formatted_prediction):
        """Regression: the key must be 'predicted_value', not 'projected_value'."""
        result = serialize_v2_prediction(mock_v2_formatted_prediction)
        assert "predicted_value" in result
        assert result["predicted_value"] == mock_v2_formatted_prediction["predicted_value"]

    def test_over_under_odds_present(self, mock_v2_formatted_prediction):
        """Both over_odds and under_odds must always be present -- not just dk_odds."""
        result = serialize_v2_prediction(mock_v2_formatted_prediction)
        assert "over_odds" in result
        assert "under_odds" in result
        assert result["over_odds"] is not None
        assert result["under_odds"] is not None

    def test_source_is_v2(self, mock_v2_formatted_prediction):
        result = serialize_v2_prediction(mock_v2_formatted_prediction)
        assert result["source"] == "v2_line_anchor"


# =========================================================================
# Type correctness
# =========================================================================

class TestSerializeV2PredictionTypes:
    """Verify serialized values have the correct types for JSON transport."""

    def test_over_odds_is_int(self, mock_v2_formatted_prediction):
        result = serialize_v2_prediction(mock_v2_formatted_prediction)
        assert isinstance(result["over_odds"], int)

    def test_under_odds_is_int(self, mock_v2_formatted_prediction):
        result = serialize_v2_prediction(mock_v2_formatted_prediction)
        assert isinstance(result["under_odds"], int)

    def test_predicted_value_is_float(self, mock_v2_formatted_prediction):
        result = serialize_v2_prediction(mock_v2_formatted_prediction)
        assert isinstance(result["predicted_value"], (int, float))

    def test_edge_percent_is_float(self, mock_v2_formatted_prediction):
        result = serialize_v2_prediction(mock_v2_formatted_prediction)
        assert isinstance(result["edge_percent"], (int, float))

    def test_confidence_is_float(self, mock_v2_formatted_prediction):
        result = serialize_v2_prediction(mock_v2_formatted_prediction)
        assert isinstance(result["confidence"], (int, float))

    def test_line_is_numeric(self, mock_v2_formatted_prediction):
        result = serialize_v2_prediction(mock_v2_formatted_prediction)
        assert isinstance(result["line"], (int, float))

    def test_all_books_is_dict(self, mock_v2_formatted_prediction):
        result = serialize_v2_prediction(mock_v2_formatted_prediction)
        assert isinstance(result["all_books"], dict)


# =========================================================================
# Value range validation
# =========================================================================

class TestSerializeV2PredictionRanges:
    """Serialized values must be in sensible ranges."""

    def test_edge_percent_reasonable(self, mock_v2_formatted_prediction):
        """Edge should be between -50 and +50. Anything beyond is a bug."""
        result = serialize_v2_prediction(mock_v2_formatted_prediction)
        assert -50 <= result["edge_percent"] <= 50

    def test_confidence_between_0_and_1(self, mock_v2_formatted_prediction):
        result = serialize_v2_prediction(mock_v2_formatted_prediction)
        assert 0 <= result["confidence"] <= 1

    def test_p_over_between_0_and_1(self, mock_v2_formatted_prediction):
        result = serialize_v2_prediction(mock_v2_formatted_prediction)
        assert 0 <= result["p_over"] <= 1

    def test_p_under_between_0_and_1(self, mock_v2_formatted_prediction):
        result = serialize_v2_prediction(mock_v2_formatted_prediction)
        assert 0 <= result["p_under"] <= 1

    def test_p_over_plus_p_under_near_1(self, mock_v2_formatted_prediction):
        """p_over + p_under should sum to ~1.0 (within rounding tolerance)."""
        result = serialize_v2_prediction(mock_v2_formatted_prediction)
        total = result["p_over"] + result["p_under"]
        assert 0.99 <= total <= 1.01, f"p_over + p_under = {total}"

    def test_predicted_value_not_zero_when_model_has_data(self, mock_v2_formatted_prediction):
        """If the model ran, predicted_value should not be 0."""
        result = serialize_v2_prediction(mock_v2_formatted_prediction)
        assert result["predicted_value"] != 0

    def test_kelly_fraction_non_negative(self, mock_v2_formatted_prediction):
        result = serialize_v2_prediction(mock_v2_formatted_prediction)
        assert result["kelly_fraction"] >= 0


# =========================================================================
# all_books sub-structure
# =========================================================================

class TestSerializeV2AllBooks:
    """Verify all_books values have over_odds and under_odds."""

    def test_all_books_values_have_odds(self, mock_v2_formatted_prediction, mock_all_books):
        pred = dict(mock_v2_formatted_prediction)
        pred["all_books"] = mock_all_books
        result = serialize_v2_prediction(pred)

        for book_name, book_data in result["all_books"].items():
            assert "over_odds" in book_data, f"{book_name} missing over_odds"
            assert "under_odds" in book_data, f"{book_name} missing under_odds"

    def test_all_books_values_have_line(self, mock_v2_formatted_prediction, mock_all_books):
        pred = dict(mock_v2_formatted_prediction)
        pred["all_books"] = mock_all_books
        result = serialize_v2_prediction(pred)

        for book_name, book_data in result["all_books"].items():
            assert "line" in book_data, f"{book_name} missing line"

    def test_empty_all_books_is_ok(self, mock_v2_formatted_prediction):
        """Empty all_books should still produce a valid prediction."""
        pred = dict(mock_v2_formatted_prediction)
        pred["all_books"] = {}
        result = serialize_v2_prediction(pred)
        assert result["all_books"] == {}
        assert result["player"] != ""  # rest of prediction still valid


# =========================================================================
# Edge case inputs
# =========================================================================

class TestSerializeV2EdgeCases:
    """Serializer must handle degenerate inputs gracefully."""

    def test_empty_prediction_dict(self):
        """An empty dict should not crash the serializer."""
        result = serialize_v2_prediction({})
        assert result["player"] == ""
        assert result["predicted_value"] == 0

    def test_none_values_handled(self):
        """
        None values for numeric fields: serialize_v2_prediction calls
        round() on .get() defaults. If None is explicitly set in the dict,
        round(None) will raise. This test verifies that the serializer's
        .get(key, 0) defaults prevent crashes when the key is MISSING.

        NOTE: If the key is present with an explicit None value,
        the current serializer will raise TypeError. This documents
        the expected behavior.
        """
        # Keys MISSING (not None) -- should use defaults
        pred = {
            "player": "Test Player",
            "prop_type": "points",
            # predicted_value, confidence, edge_percent are all missing
        }
        result = serialize_v2_prediction(pred)
        assert isinstance(result["predicted_value"], (int, float))
        assert isinstance(result["confidence"], (int, float))

    def test_side_preserved(self, mock_v2_formatted_prediction):
        """Side should be passed through exactly as given."""
        for side in ["over", "under"]:
            pred = dict(mock_v2_formatted_prediction)
            pred["side"] = side
            result = serialize_v2_prediction(pred)
            assert result["side"] == side
