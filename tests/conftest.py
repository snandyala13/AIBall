"""
Shared fixtures for the aiBall NBA prop betting test suite.

Provides mock player data, game contexts, prop lines, and prediction
structures that mirror real API responses and internal dataclass shapes.
"""

import os
import sys
import json
import random
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pytest

# Ensure project root is importable.
# When running inside a git worktree the v2/ and web/ packages may only
# exist in the main checkout.  Walk up from this file to find the repo
# root that contains all modules.
def _find_project_root():
    """Return the repo root that has v2/ and web/ directories."""
    # First try: parent of tests/
    candidate = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isdir(os.path.join(candidate, "v2")):
        return candidate
    # Second try: the main aiBall checkout (worktree parent)
    main = os.path.abspath(os.path.join(candidate, "..", "..", "..", ".."))
    if os.path.isdir(os.path.join(main, "v2")):
        return main
    # Hardcoded fallback for this repo layout
    fallback = "/Users/suhaasnandyala/Documents/aiBall"
    if os.path.isdir(os.path.join(fallback, "v2")):
        return fallback
    return candidate

PROJECT_ROOT = _find_project_root()
# Insert at position 0 so the main repo's modules take precedence over
# any worktree copies that may be incomplete or stale.
if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

# Also remove the worktree root from sys.path if it differs, so that
# incomplete worktree copies of data_pipeline.py, etc. don't shadow
# the main repo's complete versions.
_worktree_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _worktree_root != PROJECT_ROOT and _worktree_root in sys.path:
    sys.path.remove(_worktree_root)


# ---------------------------------------------------------------------------
# Deterministic randomness
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def seed_random():
    """Fix random seed for reproducible tests."""
    random.seed(42)


# ---------------------------------------------------------------------------
# Player history fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_player_history():
    """20 games of fake but realistic player data for a starter-caliber player."""
    random.seed(42)
    games = []
    for i in range(20):
        day = 28 - (i % 28)
        month = 10 - (i // 28)
        if month < 1:
            month = 12 + month
        games.append({
            "date": f"2026-{month:02d}-{day:02d}",
            "pts": 22 + random.randint(-5, 5),
            "reb": 5 + random.randint(-2, 2),
            "ast": 4 + random.randint(-2, 2),
            "fg3m": 2 + random.randint(-1, 1),
            "stl": 1,
            "blk": 0,
            "turnover": 2,
            "min": 34 + random.randint(-3, 3),
            "fga": 18,
            "fgm": 9,
            "fta": 5,
            "ftm": 4,
        })
    return games


@pytest.fixture
def mock_minimal_history():
    """Exactly 3 games -- the minimum viable for prediction."""
    random.seed(42)
    return [
        {
            "date": "2026-10-25",
            "pts": 20, "reb": 6, "ast": 3, "fg3m": 2,
            "stl": 1, "blk": 0, "turnover": 2, "min": 32,
            "fga": 16, "fgm": 8, "fta": 4, "ftm": 3,
        },
        {
            "date": "2026-10-23",
            "pts": 18, "reb": 4, "ast": 5, "fg3m": 1,
            "stl": 0, "blk": 1, "turnover": 3, "min": 30,
            "fga": 15, "fgm": 7, "fta": 3, "ftm": 2,
        },
        {
            "date": "2026-10-21",
            "pts": 25, "reb": 7, "ast": 4, "fg3m": 3,
            "stl": 2, "blk": 0, "turnover": 1, "min": 36,
            "fga": 20, "fgm": 10, "fta": 6, "ftm": 5,
        },
    ]


@pytest.fixture
def mock_empty_history():
    """No game history -- should produce None predictions."""
    return []


# ---------------------------------------------------------------------------
# Game context fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_game_context():
    """Standard game context for prediction."""
    return {
        "line": 24.5,
        "over_odds": -110,
        "under_odds": -110,
        "total": 224.0,
        "spread": -3.5,
        "is_home": True,
    }


@pytest.fixture
def mock_game_context_extreme_spread():
    """Blowout game context (15+ point spread)."""
    return {
        "line": 24.5,
        "over_odds": -110,
        "under_odds": -110,
        "total": 230.0,
        "spread": -16.5,
        "is_home": True,
    }


# ---------------------------------------------------------------------------
# All-books / cross-book fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_all_books():
    """Cross-book data for 5 books -- each has line, over_odds, under_odds."""
    return {
        "draftkings": {"line": 24.5, "over_odds": -110, "under_odds": -110},
        "fanduel": {"line": 24.5, "over_odds": -108, "under_odds": -112},
        "betmgm": {"line": 25.0, "over_odds": -115, "under_odds": -105},
        "caesars": {"line": 24.0, "over_odds": -105, "under_odds": -115},
        "pointsbet": {"line": 24.5, "over_odds": -112, "under_odds": -108},
    }


@pytest.fixture
def mock_all_books_19():
    """Full 19-book cross-book data."""
    books = {}
    book_names = [
        "draftkings", "fanduel", "betmgm", "caesars", "pointsbet",
        "bet365", "betrivers", "unibet", "williamhill", "foxbet",
        "wynnbet", "barstool", "twinspires", "sugarhouse", "betus",
        "bovada", "mybookie", "betonline", "betfred",
    ]
    random.seed(42)
    for bk in book_names:
        base_line = 24.5 + random.choice([-0.5, 0, 0, 0, 0.5])
        books[bk] = {
            "line": base_line,
            "over_odds": random.choice([-108, -110, -112, -115]),
            "under_odds": random.choice([-108, -110, -112, -105]),
        }
    return books


# ---------------------------------------------------------------------------
# v2 prediction / edge result fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_v2_raw_prediction():
    """Raw prediction dict as returned by StatPredictor.analyze_prop()."""
    return {
        "player": "Jayson Tatum",
        "prop_type": "points",
        "line": 24.5,
        "predicted": 26.2,
        "p10": 20.1,
        "p90": 32.3,
        "uncertainty": 12.2,
        "confidence": 0.72,
        "adjustment": 1.7,
        "q10_adjustment": -4.4,
        "q90_adjustment": 7.8,
        "target_mode": "line_anchor",
        "p_over": 0.61,
        "p_under": 0.39,
        "implied_over": 0.5238,
        "implied_under": 0.5238,
        "edge_over": 0.0862,
        "edge_under": -0.1338,
        "recommended_side": "OVER",
        "edge": 0.0862,
        "kelly_fraction": 0.01,
        "confidence_stars": 3,
        "confidence_raw": 0.7,
        "sigma": 2.43,
        "z_score": 0.7,
        "p10_prediction": 20.1,
        "p90_prediction": 32.3,
        "over_odds": -110,
        "under_odds": -110,
        "source": "v2_line_anchor",
        "all_books": {},
    }


@pytest.fixture
def mock_v2_formatted_prediction():
    """
    A prediction after format_prediction_for_web().
    This is what serialize_v2_prediction() receives.
    """
    return {
        "player": "Jayson Tatum",
        "prop_type": "points",
        "side": "over",
        "line": 24.5,
        "dk_odds": -110,
        "over_odds": -110,
        "under_odds": -110,
        "predicted_value": 26.2,
        "projected_value": 26.2,
        "predicted_adjustment": 1.7,
        "p10": 20.1,
        "p90": 32.3,
        "uncertainty": 12.2,
        "confidence": 0.72,
        "confidence_stars": 3,
        "edge_percent": 8.62,
        "edge": 0.0862,
        "our_probability": 0.61,
        "implied_probability": 0.5238,
        "p_over": 0.61,
        "p_under": 0.39,
        "kelly_fraction": 0.01,
        "recommended_units": 1.0,
        "all_books": {},
        "sigma": 2.43,
        "z_score": 0.7,
        "cross_book_signal": "NEUTRAL",
        "source": "v2_line_anchor",
        "target_mode": "line_anchor",
    }


@pytest.fixture
def mock_v2_serialized_prediction():
    """Final serialized v2 prediction as returned by the API."""
    return {
        "player": "Jayson Tatum",
        "prop_type": "points",
        "side": "over",
        "line": 24.5,
        "dk_odds": -110,
        "over_odds": -110,
        "under_odds": -110,
        "predicted_value": 26.2,
        "predicted_adjustment": 1.7,
        "p10": 20.1,
        "p90": 32.3,
        "uncertainty": 12.2,
        "confidence": 0.72,
        "confidence_stars": 3,
        "edge_percent": 8.62,
        "edge": 0.0862,
        "our_probability": 0.61,
        "implied_probability": 0.5238,
        "p_over": 0.61,
        "p_under": 0.39,
        "kelly_fraction": 0.01,
        "recommended_units": 1.0,
        "all_books": {},
        "best_book": None,
        "best_odds": None,
        "edge_vs_dk": 0,
        "cross_book_signal": "NEUTRAL",
        "sigma": 2.43,
        "z_score": 0.7,
        "source": "v2_line_anchor",
    }


# ---------------------------------------------------------------------------
# API response fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_games_api_response():
    """Simulated /api/games response."""
    return {
        "games": [
            {
                "game_id": 12345,
                "home_team": "Boston Celtics",
                "home_abbreviation": "BOS",
                "home_team_id": 2,
                "away_team": "Milwaukee Bucks",
                "away_abbreviation": "MIL",
                "away_team_id": 17,
                "status": "7:30 pm ET",
                "status_category": "upcoming",
                "home_score": 0,
                "away_score": 0,
                "date": "2026-10-20",
                "spread": -5.5,
                "total": 224.0,
                "home_injuries": [],
                "away_injuries": [],
                "injury_count": 0,
            }
        ],
        "count": 1,
        "date": "2026-10-20",
    }


@pytest.fixture
def mock_analyze_response(mock_v2_serialized_prediction):
    """Simulated /api/game/{id}/analyze response with v2 predictions."""
    pred1 = dict(mock_v2_serialized_prediction)
    pred2 = dict(mock_v2_serialized_prediction)
    pred2["player"] = "Giannis Antetokounmpo"
    pred2["predicted_value"] = 28.5
    pred2["edge_percent"] = 5.1

    return {
        "matchup": "Bucks at Celtics",
        "game_id": 12345,
        "timestamp": "2026-10-20T18:00:00",
        "spread": -5.5,
        "total": 224.0,
        "best_singles": [],
        "popoff_singles": [],
        "high_confidence_parlays": [],
        "safe_parlays": [],
        "standard_parlays": [],
        "longshot_parlays": [],
        "total_recommended_units": 0,
        "bankroll_allocation": {},
        "projections": [],
        "v2_predictions": [pred1, pred2],
        "model_version": "v2",
        "v2_parlays": [],
    }


# ---------------------------------------------------------------------------
# Parlay fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_v2_parlay_valid():
    """A valid 2-leg parlay: different players, no duplicates."""
    return {
        "legs": [
            {
                "player": "Jayson Tatum",
                "prop_type": "points",
                "side": "over",
                "line": 24.5,
                "dk_odds": -110,
                "edge_percent": 8.62,
                "our_probability": 0.61,
                "implied_probability": 0.5238,
            },
            {
                "player": "Giannis Antetokounmpo",
                "prop_type": "rebounds",
                "side": "under",
                "line": 11.5,
                "dk_odds": -110,
                "edge_percent": 5.1,
                "our_probability": 0.58,
                "implied_probability": 0.5238,
            },
        ],
        "combined_decimal_odds": 3.306,
        "combined_american_odds": 231,
        "combined_edge": 6.86,
        "combined_probability": 0.3538,
        "combined_implied_probability": 0.2744,
        "ev_per_unit": 0.168,
    }


# ---------------------------------------------------------------------------
# EdgeCalculator fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def edge_calculator():
    """Pre-built EdgeCalculator instance."""
    from v2.edge import EdgeCalculator
    return EdgeCalculator()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_game_log_dict(date, pts=22, reb=5, ast=4, fg3m=2, stl=1, blk=0,
                       turnover=2, minutes=34, fga=18, fgm=9, fta=5, ftm=4):
    """Helper to build a single game log dict."""
    return {
        "date": date,
        "pts": pts,
        "reb": reb,
        "ast": ast,
        "fg3m": fg3m,
        "stl": stl,
        "blk": blk,
        "turnover": turnover,
        "min": minutes,
        "fga": fga,
        "fgm": fgm,
        "fta": fta,
        "ftm": ftm,
    }
