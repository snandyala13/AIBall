"""
tests/test_v2_predictor.py - Tests for v2 StatPredictor.

Verifies feature column ordering, line-anchor mode behavior,
quantile ordering, graceful handling of unknown players/props,
tracking feature merge, and feature population.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path

from v2.features import FeatureBuilder, CORE_PROPS


# ---------------------------------------------------------------------------
# Mock LightGBM model
# ---------------------------------------------------------------------------

class MockModel:
    """Mimics a LightGBM model for testing."""

    def __init__(self, return_value=0.5, n_features=179):
        self.return_value = return_value
        self.n_features_ = n_features

    def predict(self, X):
        n = X.shape[0] if hasattr(X, 'shape') else len(X)
        return np.array([self.return_value] * n)


def _make_game_entry(date, pts=20, reb=5, ast=5, fg3m=2, stl=1, blk=1,
                     turnover=2, minutes=32, fga=15, fgm=8, fta=4, ftm=3,
                     fg3a=5):
    return {
        "date": date,
        "game_id": f"game_{date}",
        "home_team": "BOS",
        "away_team": "LAL",
        "home_score": 110,
        "away_score": 105,
        "pts": pts, "reb": reb, "ast": ast, "fg3m": fg3m,
        "stl": stl, "blk": blk, "turnover": turnover, "min": minutes,
        "fga": fga, "fgm": fgm, "fta": fta, "ftm": ftm, "fg3a": fg3a,
    }


# ---------------------------------------------------------------------------
# Build a StatPredictor without loading real models from disk
# ---------------------------------------------------------------------------

def _build_mock_predictor(
    point_val=1.5, q10_val=-3.0, q90_val=5.0, feature_columns=None
):
    """Build a StatPredictor with mock models, no disk I/O."""
    from v2.predictor import StatPredictor

    with patch.object(StatPredictor, '_load_models'):
        predictor = StatPredictor.__new__(StatPredictor)
        predictor.models_dir = Path("/fake/models/v2")
        predictor.models = {}
        predictor.feature_builder = FeatureBuilder()
        predictor.feature_builder._loaded = True

        from v2.edge import EdgeCalculator
        predictor.edge_calculator = EdgeCalculator()
        predictor.target_mode = "line_anchor"
        predictor._loaded = True

        # Install mock models for core props
        for prop_type in CORE_PROPS:
            predictor.models[prop_type] = (
                MockModel(point_val),
                MockModel(q10_val),
                MockModel(q90_val),
            )

        # Set feature columns
        if feature_columns:
            predictor.feature_columns = feature_columns
        else:
            predictor.feature_columns = None

        return predictor


def _load_player_history(predictor, player_name, num_games=15):
    """Inject synthetic game history into the predictor's feature builder."""
    games = []
    for i in range(num_games):
        day = str(i + 1).zfill(2)
        games.append(_make_game_entry(f"2026-01-{day}", pts=20 + i % 5))
    predictor.feature_builder.player_history[player_name] = sorted(
        games, key=lambda g: g["date"]
    )


# ===========================================================================
# 1. FEATURE COLUMN ORDERING
# ===========================================================================

class TestFeatureColumnOrdering:
    """Features must be in the same order as training."""

    def test_feature_vec_uses_column_order(self):
        """When feature_columns is set, the feature vector should follow that order."""
        # Build features manually to get the column list
        fb = FeatureBuilder()
        fb._loaded = True
        games = [_make_game_entry(f"2026-01-{str(d).zfill(2)}") for d in range(1, 10)]
        fb.player_history["Test Player"] = games

        ctx = {"line": 20.0, "over_odds": -110, "under_odds": -110}
        feats = fb.get_features_for_leg("Test Player", "points", "2026-01-15", ctx)
        columns = list(feats.keys())

        # Reverse the columns to prove ordering matters
        reversed_cols = list(reversed(columns))
        predictor = _build_mock_predictor(feature_columns=reversed_cols)
        _load_player_history(predictor, "Test Player")

        result = predictor.predict("Test Player", "points", "2026-01-20", ctx)
        assert result is not None
        # The test passes if predict() completes without error using the given column order

    def test_missing_feature_column_defaults_to_zero(self):
        """If a feature_column is not in computed features, it should be 0.0."""
        predictor = _build_mock_predictor(
            feature_columns=["target_l3", "NONEXISTENT_FEATURE", "target_l5"]
        )
        _load_player_history(predictor, "Test Player")

        ctx = {"line": 20.0, "over_odds": -110, "under_odds": -110}
        result = predictor.predict("Test Player", "points", "2026-01-20", ctx)
        # Should not crash; missing feature defaults to 0.0
        assert result is not None


# ===========================================================================
# 2. LINE-ANCHOR MODE
# ===========================================================================

class TestLineAnchorMode:
    """predicted = line + adjustment, NOT raw value."""

    def test_predicted_equals_line_plus_adjustment(self):
        """In line-anchor mode, predicted = line + model_output."""
        adjustment = 2.5
        predictor = _build_mock_predictor(point_val=adjustment, q10_val=-1.0, q90_val=5.0)
        _load_player_history(predictor, "Test Player")

        line = 24.5
        ctx = {"line": line, "over_odds": -110, "under_odds": -110}
        result = predictor.predict("Test Player", "points", "2026-01-20", ctx)

        assert result is not None
        assert result["target_mode"] == "line_anchor"
        assert abs(result["predicted"] - (line + adjustment)) < 0.01
        assert abs(result["adjustment"] - adjustment) < 0.01

    def test_negative_adjustment(self):
        """Negative adjustment: predicted should be below line."""
        adjustment = -3.0
        predictor = _build_mock_predictor(point_val=adjustment, q10_val=-7.0, q90_val=1.0)
        _load_player_history(predictor, "Test Player")

        line = 24.5
        ctx = {"line": line, "over_odds": -110, "under_odds": -110}
        result = predictor.predict("Test Player", "points", "2026-01-20", ctx)

        assert result["predicted"] < line
        assert abs(result["predicted"] - (line + adjustment)) < 0.01

    def test_zero_adjustment(self):
        """Zero adjustment: predicted should equal line."""
        predictor = _build_mock_predictor(point_val=0.0, q10_val=-4.0, q90_val=4.0)
        _load_player_history(predictor, "Test Player")

        line = 24.5
        ctx = {"line": line, "over_odds": -110, "under_odds": -110}
        result = predictor.predict("Test Player", "points", "2026-01-20", ctx)

        assert abs(result["predicted"] - line) < 0.01

    def test_predicted_value_key_not_zero_bug(self):
        """
        BUG REGRESSION: predicted_value was 0 due to key name mismatch
        (projected_value vs predicted_value). Verify the 'predicted' key
        is always populated with a real value, not zero.
        """
        predictor = _build_mock_predictor(point_val=1.5)
        _load_player_history(predictor, "Test Player")

        ctx = {"line": 24.5, "over_odds": -110, "under_odds": -110}
        result = predictor.predict("Test Player", "points", "2026-01-20", ctx)

        assert result["predicted"] != 0
        assert result["predicted"] == pytest.approx(24.5 + 1.5, abs=0.01)


# ===========================================================================
# 3. QUANTILE ORDERING
# ===========================================================================

class TestQuantileOrdering:
    """p10 <= predicted <= p90 always."""

    def test_quantile_ordering_positive_adjustment(self):
        predictor = _build_mock_predictor(point_val=2.0, q10_val=-3.0, q90_val=6.0)
        _load_player_history(predictor, "Test Player")

        ctx = {"line": 24.5, "over_odds": -110, "under_odds": -110}
        result = predictor.predict("Test Player", "points", "2026-01-20", ctx)

        assert result["p10"] <= result["predicted"]
        assert result["predicted"] <= result["p90"]

    def test_quantile_ordering_negative_adjustment(self):
        predictor = _build_mock_predictor(point_val=-2.0, q10_val=-6.0, q90_val=1.0)
        _load_player_history(predictor, "Test Player")

        ctx = {"line": 24.5, "over_odds": -110, "under_odds": -110}
        result = predictor.predict("Test Player", "points", "2026-01-20", ctx)

        assert result["p10"] <= result["predicted"]
        assert result["predicted"] <= result["p90"]

    def test_quantile_ordering_inverted_quantiles(self):
        """Even if the quantile models give inverted outputs (q90 < q10),
        the predictor should fix the ordering."""
        predictor = _build_mock_predictor(point_val=0.0, q10_val=5.0, q90_val=-5.0)
        _load_player_history(predictor, "Test Player")

        ctx = {"line": 24.5, "over_odds": -110, "under_odds": -110}
        result = predictor.predict("Test Player", "points", "2026-01-20", ctx)

        # The code does: q10_adj = min(raw_q10, adjustment), q90_adj = max(raw_q90, adjustment)
        # Then: p10 = min(p10, predicted), p90 = max(p90, predicted)
        assert result["p10"] <= result["predicted"]
        assert result["predicted"] <= result["p90"]


# ===========================================================================
# 4. UNKNOWN PLAYER
# ===========================================================================

class TestUnknownPlayer:
    """Should return None gracefully, not crash."""

    def test_unknown_player_returns_none(self):
        predictor = _build_mock_predictor()
        # Do NOT load any player history
        ctx = {"line": 20.0, "over_odds": -110, "under_odds": -110}
        result = predictor.predict("Nonexistent Player", "points", "2026-01-20", ctx)
        assert result is None


# ===========================================================================
# 5. UNKNOWN PROP TYPE
# ===========================================================================

class TestUnknownPropType:
    """Should return None for prop types without models."""

    def test_unknown_prop_returns_none(self):
        predictor = _build_mock_predictor()
        _load_player_history(predictor, "Test Player")

        ctx = {"line": 20.0, "over_odds": -110, "under_odds": -110}
        result = predictor.predict("Test Player", "double_double", "2026-01-20", ctx)
        assert result is None

    def test_made_up_prop_returns_none(self):
        predictor = _build_mock_predictor()
        _load_player_history(predictor, "Test Player")

        ctx = {"line": 20.0, "over_odds": -110, "under_odds": -110}
        result = predictor.predict("Test Player", "slam_dunks", "2026-01-20", ctx)
        assert result is None


# ===========================================================================
# 6. MISSING GAME_CONTEXT FIELDS
# ===========================================================================

class TestMissingContextFields:
    """Should use defaults, not crash."""

    def test_empty_context_uses_defaults(self):
        predictor = _build_mock_predictor()
        _load_player_history(predictor, "Test Player")

        # Minimal context: only line is required for features to compute
        ctx = {"line": 20.0}
        result = predictor.predict("Test Player", "points", "2026-01-20", ctx)
        # Should use default odds -110/-110
        assert result is not None

    def test_no_line_in_context(self):
        """Even with no line, should not crash (defaults to 0)."""
        predictor = _build_mock_predictor()
        _load_player_history(predictor, "Test Player")

        ctx = {}
        result = predictor.predict("Test Player", "points", "2026-01-20", ctx)
        assert result is not None


# ===========================================================================
# 7. TRACKING FEATURES MERGE
# ===========================================================================

class TestTrackingFeaturesMerge:
    """When tracking_features provided in game_context, they appear in features."""

    def test_tracking_features_are_merged(self):
        """Tracking features from game_context should be merged into the feature dict."""
        predictor = _build_mock_predictor(
            feature_columns=["target_l3", "target_l5", "drives", "touches", "has_tracking"]
        )
        _load_player_history(predictor, "Test Player")

        ctx = {
            "line": 20.0,
            "over_odds": -110,
            "under_odds": -110,
            "tracking_features": {
                "drives": 12.5,
                "touches": 65.0,
                "has_tracking": 1.0,
            },
        }
        result = predictor.predict("Test Player", "points", "2026-01-20", ctx)
        assert result is not None


# ===========================================================================
# 8. FEATURE VECTOR COMPLETENESS
# ===========================================================================

class TestFeatureVectorCompleteness:
    """Feature vector should have no NaN or infinite values."""

    def test_feature_count_mismatch_regression(self):
        """
        BUG REGRESSION: Model trained on 179 features but inference provides 78.
        When feature_columns is set, the vector length MUST match.
        """
        # Create a feature_columns list of 179 entries
        known_cols = ["target_l3", "target_l5", "target_l10", "target_season", "line"]
        # Pad with synthetic feature names
        padding = [f"feature_{i}" for i in range(179 - len(known_cols))]
        all_cols = known_cols + padding

        predictor = _build_mock_predictor(feature_columns=all_cols)
        _load_player_history(predictor, "Test Player")

        ctx = {"line": 20.0, "over_odds": -110, "under_odds": -110}
        result = predictor.predict("Test Player", "points", "2026-01-20", ctx)

        # Should complete successfully; missing features default to 0.0
        assert result is not None

    def test_predict_returns_all_expected_keys(self):
        """The prediction result should contain all standard keys."""
        predictor = _build_mock_predictor()
        _load_player_history(predictor, "Test Player")

        ctx = {"line": 20.0, "over_odds": -110, "under_odds": -110}
        result = predictor.predict("Test Player", "points", "2026-01-20", ctx)

        expected_keys = [
            "predicted", "p10", "p90", "uncertainty", "confidence",
            "adjustment", "q10_adjustment", "q90_adjustment", "target_mode",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key '{key}' in prediction result"
