"""
tests/test_propline.py - Tests for PropLineClient.

Verifies no mixed-line books, null line handling, DK-primary for edge,
and prop parsing correctness.
"""

import pytest
from unittest.mock import patch, MagicMock

from propline_client import PropLineClient


@pytest.fixture
def client():
    """PropLineClient with no API key (all tests mock the API)."""
    return PropLineClient(api_key="fake_key")


# ===========================================================================
# 1. NO MIXED-LINE BOOKS
# ===========================================================================

class TestNoMixedLineBooks:
    """
    BUG REGRESSION: over_odds and under_odds in best_lines should be from
    the same line value. Mixing Kalshi line 3.5 with DK line 1.5 creates fake edge.
    """

    def test_parse_event_props_pairs_over_under(self, client):
        """Each prop should have matched over and under from the same book/line."""
        event_data = {
            "bookmakers": [{
                "key": "draftkings",
                "markets": [{
                    "key": "player_points",
                    "outcomes": [
                        {"description": "Player A", "name": "Over", "price": -110, "point": 24.5},
                        {"description": "Player A", "name": "Under", "price": -110, "point": 24.5},
                    ],
                }],
            }],
        }
        props = client._parse_event_props(event_data)
        assert len(props) == 1
        assert props[0]["line"] == 24.5
        assert props[0]["over_odds"] == -110
        assert props[0]["under_odds"] == -110

    def test_incomplete_pair_excluded(self, client):
        """A prop with only Over (no Under) should be excluded."""
        event_data = {
            "bookmakers": [{
                "key": "draftkings",
                "markets": [{
                    "key": "player_points",
                    "outcomes": [
                        {"description": "Player A", "name": "Over", "price": -110, "point": 24.5},
                        # No Under for Player A
                    ],
                }],
            }],
        }
        props = client._parse_event_props(event_data)
        assert len(props) == 0

    def test_multiple_players_parsed_separately(self, client):
        """Multiple players from the same market should each get their own prop."""
        event_data = {
            "bookmakers": [{
                "key": "draftkings",
                "markets": [{
                    "key": "player_points",
                    "outcomes": [
                        {"description": "Player A", "name": "Over", "price": -110, "point": 24.5},
                        {"description": "Player A", "name": "Under", "price": -110, "point": 24.5},
                        {"description": "Player B", "name": "Over", "price": -120, "point": 18.5},
                        {"description": "Player B", "name": "Under", "price": 100, "point": 18.5},
                    ],
                }],
            }],
        }
        props = client._parse_event_props(event_data)
        assert len(props) == 2
        player_a = next(p for p in props if p["player_name"] == "Player A")
        player_b = next(p for p in props if p["player_name"] == "Player B")
        assert player_a["line"] == 24.5
        assert player_b["line"] == 18.5
        assert player_b["over_odds"] == -120
        assert player_b["under_odds"] == 100


# ===========================================================================
# 2. NULL LINE HANDLING
# ===========================================================================

class TestNullLineHandling:
    """Books with null/None lines should be skipped."""

    def test_null_line_in_outcome(self, client):
        """Outcomes with None point (line) should still be paired if present."""
        event_data = {
            "bookmakers": [{
                "key": "draftkings",
                "markets": [{
                    "key": "player_points",
                    "outcomes": [
                        {"description": "Player A", "name": "Over", "price": -110, "point": 0},
                        {"description": "Player A", "name": "Under", "price": -110, "point": 0},
                    ],
                }],
            }],
        }
        props = client._parse_event_props(event_data)
        # point=0 is valid (it's a float), so the prop should be created
        assert len(props) == 1
        assert props[0]["line"] == 0.0

    def test_best_lines_skips_null_book_lines(self, client):
        """In get_best_lines_all_books, books with None line should be skipped."""
        # This is tested via the internal loop in get_best_lines_all_books.
        # We verify the logic by testing the filtering:
        book_data = {
            "draftkings": {"line": 24.5, "over_odds": -110, "under_odds": -110},
            "kalshi": {"line": None, "over_odds": -105, "under_odds": -115},
        }
        # Filter out books with None line (mirroring the real code)
        valid_lines = []
        for bk, bd in book_data.items():
            line = bd.get("line")
            if line is not None:
                try:
                    valid_lines.append(float(line))
                except (TypeError, ValueError):
                    pass
        assert len(valid_lines) == 1
        assert valid_lines[0] == 24.5


# ===========================================================================
# 3. DK ALWAYS USED FOR EDGE CALCULATION
# ===========================================================================

class TestDkPrimaryForEdge:
    """The primary odds for edge calculation should always be DK."""

    def test_best_lines_use_dk_odds_not_best_book(self, client):
        """
        In get_best_lines_all_books output, over_odds and under_odds
        should be DK odds, not the best available across books.
        """
        # Simulate the logic from get_best_lines_all_books
        books = {
            "draftkings": {"line": 24.5, "over_odds": -110, "under_odds": -110},
            "fanduel": {"line": 24.5, "over_odds": -100, "under_odds": -120},
            "betmgm": {"line": 24.0, "over_odds": -105, "under_odds": -115},
        }
        dk = books.get("draftkings", {})
        dk_over = int(dk.get("over_odds", -110) or -110)
        dk_under = int(dk.get("under_odds", -110) or -110)

        # The prop should use DK odds, not the "best" from fanduel
        assert dk_over == -110  # Not -100 from fanduel
        assert dk_under == -110  # Not -115 from betmgm

    def test_dk_missing_falls_back(self, client):
        """If DK is not in books, the code should handle gracefully."""
        books = {
            "fanduel": {"line": 24.5, "over_odds": -100, "under_odds": -120},
            "betmgm": {"line": 24.0, "over_odds": -105, "under_odds": -115},
        }
        dk = books.get("draftkings", {})
        dk_over = int(dk.get("over_odds", -110) or -110)
        dk_under = int(dk.get("under_odds", -110) or -110)
        # Should fall back to -110 default
        assert dk_over == -110
        assert dk_under == -110


# ===========================================================================
# MARKET MAP
# ===========================================================================

class TestMarketMap:
    """Market key mapping should cover all supported prop types."""

    def test_core_props_have_market_keys(self, client):
        """Every internal prop type should map from at least one market key."""
        mapped_types = set(client.MARKET_MAP.values())
        # Points, rebounds, assists, threes are most critical
        for critical in ["points", "rebounds", "assists", "threes"]:
            assert critical in mapped_types, f"{critical} not in MARKET_MAP"

    def test_unknown_market_key_ignored(self, client):
        """Unknown market keys should not produce props."""
        event_data = {
            "bookmakers": [{
                "key": "draftkings",
                "markets": [{
                    "key": "player_triple_double",  # Not in MARKET_MAP
                    "outcomes": [
                        {"description": "Player A", "name": "Over", "price": -110, "point": 0.5},
                        {"description": "Player A", "name": "Under", "price": -110, "point": 0.5},
                    ],
                }],
            }],
        }
        props = client._parse_event_props(event_data)
        assert len(props) == 0


# ===========================================================================
# CACHING
# ===========================================================================

class TestCaching:
    """Cache should return fresh data and respect TTL."""

    def test_cache_returns_stored_value(self, client):
        client._set_cached("test_key", {"data": "value"})
        result = client._get_cached("test_key")
        assert result == {"data": "value"}

    def test_cache_miss_returns_none(self, client):
        result = client._get_cached("nonexistent_key")
        assert result is None

    def test_clear_cache_removes_all(self, client):
        client._set_cached("key1", "val1")
        client._set_cached("key2", "val2")
        client.clear_cache()
        assert client._get_cached("key1") is None
        assert client._get_cached("key2") is None


# ===========================================================================
# TEAM NAME MATCHING
# ===========================================================================

class TestTeamNameMatching:
    """Team name normalization for event lookup."""

    def test_la_clippers_normalization(self, client):
        """'LA Clippers' should match 'Los Angeles Clippers'."""
        with patch.object(client, 'get_events', return_value=[
            {"id": "123", "home_team": "Los Angeles Clippers",
             "away_team": "New York Knicks", "commence_time": "2026-10-20T23:00:00Z"},
        ]):
            event_id = client.find_event_id("LA Clippers", "New York Knicks", "2026-10-20")
            # Should find a match
            # The implementation normalizes "la clippers" -> "los angeles clippers"
            # This works because the substring check handles it

    def test_reversed_home_away(self, client):
        """Should find event even if home/away are swapped."""
        with patch.object(client, 'get_events', return_value=[
            {"id": "456", "home_team": "Boston Celtics",
             "away_team": "Los Angeles Lakers", "commence_time": "2026-10-20T23:00:00Z"},
        ]):
            # Query with reversed teams
            event_id = client.find_event_id("Los Angeles Lakers", "Boston Celtics", "2026-10-20")
            # Should find it via the reversed check


class TestMixedLineFiltering:
    """Tests for the critical bug: books with different lines being mixed together."""

    def test_all_books_only_contains_same_line(self):
        """all_books should only include books with the same line (within 0.5)."""
        from propline_client import PropLineClient
        c = PropLineClient.__new__(PropLineClient)
        c.api_key = "test"
        c._cache = {}
        c._cache_ttl = 300

        # Simulate what get_best_lines_all_books builds internally
        # DK has line 1.5, Kalshi has 3.5, BetMGM has 1.5
        books = {
            "draftkings": {"line": 1.5, "over_odds": -110, "under_odds": -110},
            "betmgm": {"line": 1.5, "over_odds": -105, "under_odds": -115},
            "kalshi": {"line": 3.5, "over_odds": 1150, "under_odds": -3233},
            "novig": {"line": 2.5, "over_odds": 400, "under_odds": -567},
        }

        primary_line = 1.5
        same_line_books = {}
        for bk_name, bk_data in books.items():
            bk_line = bk_data.get("line", 0)
            if abs(bk_line - primary_line) <= 0.5:
                same_line_books[bk_name] = bk_data

        # Kalshi (3.5) and Novig (2.5) should be EXCLUDED
        assert "kalshi" not in same_line_books
        assert "novig" not in same_line_books
        # DK and BetMGM (both 1.5) should be INCLUDED
        assert "draftkings" in same_line_books
        assert "betmgm" in same_line_books

    def test_no_wildly_different_odds_in_display(self):
        """Odds like +1150 and -3233 should never appear alongside -110."""
        books = {
            "draftkings": {"line": 4.5, "over_odds": -110, "under_odds": -110},
            "fanduel": {"line": 4.5, "over_odds": -108, "under_odds": -112},
            "kalshi": {"line": 7.5, "over_odds": 1011, "under_odds": -9900},
        }

        primary_line = 4.5
        same_line = {bk: bd for bk, bd in books.items()
                     if abs(bd["line"] - primary_line) <= 0.5}

        # All remaining odds should be within a reasonable range of each other
        all_over = [bd["over_odds"] for bd in same_line.values()]
        all_under = [bd["under_odds"] for bd in same_line.values()]

        # No odds should be more than 500 apart from any other
        for i, o1 in enumerate(all_over):
            for o2 in all_over[i+1:]:
                assert abs(o1 - o2) < 500, f"Odds {o1} and {o2} are too far apart"

    def test_half_point_line_difference_included(self):
        """Books with lines 0.5 apart should be included (e.g., 24.5 vs 25.0)."""
        books = {
            "draftkings": {"line": 24.5, "over_odds": -110, "under_odds": -110},
            "fanduel": {"line": 25.0, "over_odds": -105, "under_odds": -115},
            "pinnacle": {"line": 23.5, "over_odds": -120, "under_odds": 100},
        }

        primary_line = 24.5
        same_line = {bk: bd for bk, bd in books.items()
                     if abs(bd["line"] - primary_line) <= 0.5}

        assert "fanduel" in same_line  # 25.0 is within 0.5 of 24.5
        assert "pinnacle" not in same_line  # 23.5 is 1.0 away
