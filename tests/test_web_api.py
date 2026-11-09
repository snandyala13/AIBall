"""
tests/test_web_api.py - Web API integration tests.

Tests the FastAPI routes with mock engine data to verify:
- Response structure correctness
- v2 prediction field completeness
- Parlay integrity (no duplicate legs)
- Cross-book data consistency
"""

import os
import sys
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# Path setup is handled by pyproject.toml [tool.pytest.ini_options] pythonpath
# and by conftest.py which ensures the main repo root is on sys.path.


# -------------------------------------------------------------------------
# FastAPI test client setup
# -------------------------------------------------------------------------

@pytest.fixture
def mock_engine(mock_games_api_response, mock_analyze_response):
    """Mock AnalysisEngine that returns predictable data."""
    engine = MagicMock()
    engine.get_games_for_date.return_value = mock_games_api_response["games"]
    engine.analyze_game.return_value = mock_analyze_response
    engine.get_player_profile.return_value = None
    engine.clear_caches.return_value = None
    engine._analysis_cache = {}
    return engine


@pytest.fixture
def test_client(mock_engine):
    """FastAPI TestClient with mocked engine."""
    from fastapi.testclient import TestClient

    with patch("web.app.get_engine", return_value=mock_engine):
        from web.app import app
        client = TestClient(app, raise_server_exceptions=False)
        yield client


# =========================================================================
# /api/games
# =========================================================================

class TestApiGames:
    """/api/games should return structured game data."""

    def test_games_returns_200(self, test_client):
        response = test_client.get("/api/games")
        assert response.status_code == 200

    def test_games_has_games_array(self, test_client):
        data = test_client.get("/api/games").json()
        assert "games" in data
        assert isinstance(data["games"], list)

    def test_games_have_required_fields(self, test_client):
        data = test_client.get("/api/games").json()
        required = ["game_id", "home_team", "away_team", "status_category", "spread", "total"]
        for game in data["games"]:
            for field in required:
                assert field in game, f"Game missing field: {field}"

    def test_games_status_category_valid(self, test_client):
        data = test_client.get("/api/games").json()
        valid_statuses = {"upcoming", "live", "final"}
        for game in data["games"]:
            assert game["status_category"] in valid_statuses

    def test_games_count_matches(self, test_client):
        data = test_client.get("/api/games").json()
        assert data["count"] == len(data["games"])


# =========================================================================
# /api/game/{id}/analyze
# =========================================================================

class TestApiAnalyzeGame:
    """/api/game/{id}/analyze should return v2 predictions."""

    def test_analyze_returns_200(self, test_client):
        response = test_client.get("/api/game/12345/analyze")
        assert response.status_code == 200

    def test_analyze_has_v2_predictions(self, test_client):
        data = test_client.get("/api/game/12345/analyze").json()
        assert "v2_predictions" in data
        assert isinstance(data["v2_predictions"], list)

    def test_v2_predictions_have_required_fields(self, test_client):
        data = test_client.get("/api/game/12345/analyze").json()
        required = [
            "player", "prop_type", "side", "line", "dk_odds",
            "predicted_value", "edge_percent", "p_over", "p_under",
            "over_odds", "under_odds", "confidence",
        ]
        for pred in data["v2_predictions"]:
            for field in required:
                assert field in pred, (
                    f"v2 prediction for {pred.get('player', '?')} missing: {field}"
                )

    def test_no_prediction_has_zero_predicted_value(self, test_client):
        """If model is loaded and player has history, predicted_value should not be 0."""
        data = test_client.get("/api/game/12345/analyze").json()
        for pred in data["v2_predictions"]:
            # The mock data sets predicted_value to non-zero values
            assert pred["predicted_value"] != 0, (
                f"{pred['player']} has predicted_value=0"
            )

    def test_no_prediction_has_edge_over_100(self, test_client):
        """Edge > 100% is impossible and indicates a bug."""
        data = test_client.get("/api/game/12345/analyze").json()
        for pred in data["v2_predictions"]:
            assert abs(pred["edge_percent"]) <= 100, (
                f"{pred['player']} has edge_percent={pred['edge_percent']}"
            )

    def test_side_consistent_with_probability(self, test_client):
        """If side='over', model should believe over has value."""
        data = test_client.get("/api/game/12345/analyze").json()
        for pred in data["v2_predictions"]:
            side = pred["side"]
            edge = pred["edge_percent"]
            # If edge > 0, side should match the positive-edge direction
            if edge > 0 and side == "over":
                assert pred["p_over"] > 0.4, (
                    f"{pred['player']}: side=over but p_over={pred['p_over']}"
                )
            elif edge > 0 and side == "under":
                assert pred["p_under"] > 0.4, (
                    f"{pred['player']}: side=under but p_under={pred['p_under']}"
                )

    def test_over_and_under_odds_always_present(self, test_client):
        data = test_client.get("/api/game/12345/analyze").json()
        for pred in data["v2_predictions"]:
            assert "over_odds" in pred, f"{pred['player']} missing over_odds"
            assert "under_odds" in pred, f"{pred['player']} missing under_odds"
            assert pred["over_odds"] is not None
            assert pred["under_odds"] is not None

    def test_model_version_present(self, test_client):
        data = test_client.get("/api/game/12345/analyze").json()
        assert "model_version" in data

    def test_analyze_404_for_missing_game(self, test_client, mock_engine):
        """Missing game should return 404."""
        mock_engine.analyze_game.return_value = None
        response = test_client.get("/api/game/99999/analyze")
        assert response.status_code == 404


# =========================================================================
# v2 parlays
# =========================================================================

class TestApiParlays:
    """v2_parlays must have no duplicate legs or same-player legs."""

    def test_parlays_no_duplicate_legs(self, mock_v2_parlay_valid):
        """No parlay should have two identical legs (same player+prop)."""
        legs = mock_v2_parlay_valid["legs"]
        leg_keys = [(l["player"], l["prop_type"]) for l in legs]
        assert len(leg_keys) == len(set(leg_keys)), "Duplicate legs in parlay"

    def test_parlays_no_same_player(self, mock_v2_parlay_valid):
        """Each player should appear at most once per parlay."""
        legs = mock_v2_parlay_valid["legs"]
        players = [l["player"] for l in legs]
        assert len(players) == len(set(players)), "Same player appears twice in parlay"

    def test_build_parlays_no_same_player(self):
        """V2AnalysisEngine.build_parlays should never pair same-player legs."""
        from v2.engine import V2AnalysisEngine

        predictions = [
            {
                "player": "Jayson Tatum",
                "prop_type": "points",
                "side": "over",
                "edge_percent": 8.0,
                "dk_odds": -110,
                "our_probability": 0.6,
                "implied_probability": 0.52,
            },
            {
                "player": "Jayson Tatum",
                "prop_type": "rebounds",
                "side": "under",
                "edge_percent": 5.0,
                "dk_odds": -110,
                "our_probability": 0.58,
                "implied_probability": 0.52,
            },
            {
                "player": "Giannis Antetokounmpo",
                "prop_type": "points",
                "side": "over",
                "edge_percent": 6.0,
                "dk_odds": -110,
                "our_probability": 0.59,
                "implied_probability": 0.52,
            },
        ]

        with patch("v2.engine.StatPredictor") as MockPredictor, \
             patch("v2.engine.load_tracking_data", return_value={}), \
             patch("v2.engine.load_pbp_data", return_value={}):
            mock_pred = MagicMock()
            mock_pred.is_ready.return_value = False
            mock_pred.models = {}
            MockPredictor.return_value = mock_pred
            engine = V2AnalysisEngine()

        parlays = engine.build_parlays(predictions)
        for parlay in parlays:
            players = [leg["player"] for leg in parlay["legs"]]
            assert len(players) == len(set(players)), (
                f"Same player in parlay: {players}"
            )

    def test_build_parlays_no_duplicate_legs(self):
        """No parlay should have the exact same player+prop pair twice."""
        from v2.engine import V2AnalysisEngine

        predictions = [
            {
                "player": "Player A",
                "prop_type": "points",
                "side": "over",
                "edge_percent": 8.0,
                "dk_odds": -110,
                "our_probability": 0.6,
                "implied_probability": 0.52,
            },
            {
                "player": "Player B",
                "prop_type": "rebounds",
                "side": "under",
                "edge_percent": 5.0,
                "dk_odds": -110,
                "our_probability": 0.58,
                "implied_probability": 0.52,
            },
            {
                "player": "Player C",
                "prop_type": "assists",
                "side": "over",
                "edge_percent": 4.0,
                "dk_odds": -110,
                "our_probability": 0.57,
                "implied_probability": 0.52,
            },
        ]

        with patch("v2.engine.StatPredictor") as MockPredictor, \
             patch("v2.engine.load_tracking_data", return_value={}), \
             patch("v2.engine.load_pbp_data", return_value={}):
            mock_pred = MagicMock()
            mock_pred.is_ready.return_value = False
            mock_pred.models = {}
            MockPredictor.return_value = mock_pred
            engine = V2AnalysisEngine()

        parlays = engine.build_parlays(predictions)
        for parlay in parlays:
            leg_keys = [(l["player"], l["prop_type"]) for l in parlay["legs"]]
            assert len(leg_keys) == len(set(leg_keys))


# =========================================================================
# all_books consistency
# =========================================================================

class TestAllBooksConsistency:
    """All books data must have consistent line+odds within each book."""

    def test_all_books_have_line_and_odds(self, mock_all_books):
        """Within each book, there should be line, over_odds, and under_odds."""
        for book_name, book_data in mock_all_books.items():
            assert "line" in book_data, f"{book_name} missing 'line'"
            assert "over_odds" in book_data, f"{book_name} missing 'over_odds'"
            assert "under_odds" in book_data, f"{book_name} missing 'under_odds'"

    def test_all_books_odds_are_integers(self, mock_all_books):
        """Odds should be integers (American format)."""
        for book_name, book_data in mock_all_books.items():
            assert isinstance(book_data["over_odds"], int), (
                f"{book_name} over_odds is not int: {type(book_data['over_odds'])}"
            )
            assert isinstance(book_data["under_odds"], int), (
                f"{book_name} under_odds is not int: {type(book_data['under_odds'])}"
            )

    def test_all_books_lines_are_numeric(self, mock_all_books):
        """Lines should be numeric (float or int)."""
        for book_name, book_data in mock_all_books.items():
            assert isinstance(book_data["line"], (int, float)), (
                f"{book_name} line is not numeric: {type(book_data['line'])}"
            )


# =========================================================================
# /api/refresh
# =========================================================================

class TestApiRefresh:
    """POST /api/refresh should clear caches."""

    def test_refresh_returns_200(self, test_client):
        response = test_client.post("/api/refresh")
        assert response.status_code == 200

    def test_refresh_clears_caches(self, test_client, mock_engine):
        test_client.post("/api/refresh")
        mock_engine.clear_caches.assert_called_once()
