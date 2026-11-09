"""
tests/test_data_pipeline.py - Tests for data_pipeline.py edge cases.

Verifies season rollover logic, minutes parsing, and related utilities.
Does NOT make API calls - all tests use synthetic data.
"""

import pytest
from datetime import datetime
from unittest.mock import patch


# ===========================================================================
# 1. SEASON ROLLOVER
# ===========================================================================

class TestSeasonRollover:
    """get_current_nba_season() returns correct year in Oct vs April."""

    def test_october_returns_current_year(self):
        """In October, the new season starts -> return current year."""
        from data_pipeline import get_current_nba_season
        with patch('data_pipeline.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2026, 10, 15)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = get_current_nba_season()
            assert result == 2026

    def test_april_returns_previous_year(self):
        """In April (mid-season), return previous year."""
        from data_pipeline import get_current_nba_season
        with patch('data_pipeline.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2026, 4, 15)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = get_current_nba_season()
            assert result == 2025

    def test_january_returns_previous_year(self):
        """In January, still the previous year's season."""
        from data_pipeline import get_current_nba_season
        with patch('data_pipeline.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 7)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = get_current_nba_season()
            assert result == 2025

    def test_september_returns_previous_year(self):
        """In September (offseason), still previous year's season."""
        from data_pipeline import get_current_nba_season
        with patch('data_pipeline.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2026, 9, 30)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = get_current_nba_season()
            assert result == 2025

    def test_season_str_format(self):
        """get_current_nba_season_str() should return 'YYYY-YY' format."""
        from data_pipeline import get_current_nba_season_str
        with patch('data_pipeline.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2026, 10, 15)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = get_current_nba_season_str()
            assert result == "2026-27"

    def test_season_str_format_mid_season(self):
        from data_pipeline import get_current_nba_season_str
        with patch('data_pipeline.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 1)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = get_current_nba_season_str()
            assert result == "2025-26"


# ===========================================================================
# 2. DATE-TO-SEASON MAPPING (v2/enrich_training_data)
# ===========================================================================

class TestDateToSeason:
    """date_to_season() maps game dates to NBA season strings."""

    def test_october_start_of_season(self):
        from v2.enrich_training_data import date_to_season
        result = date_to_season("2025-10-15")
        assert result == "2025-26"

    def test_june_end_of_season(self):
        from v2.enrich_training_data import date_to_season
        result = date_to_season("2026-06-15")
        assert result == "2025-26"

    def test_january_mid_season(self):
        from v2.enrich_training_data import date_to_season
        result = date_to_season("2026-01-05")
        assert result == "2025-26"

    def test_fallback_for_unknown_range(self):
        """Dates outside known SEASON_RANGES should use month-based fallback."""
        from v2.enrich_training_data import date_to_season
        # Far future date, not in SEASON_RANGES
        result = date_to_season("2030-11-15")
        assert result == "2030-31"

    def test_fallback_mid_season_unknown(self):
        from v2.enrich_training_data import date_to_season
        result = date_to_season("2030-02-15")
        assert result == "2029-30"


# ===========================================================================
# 3. MINUTES PARSING
# ===========================================================================

class TestMinutesParsing:
    """
    Minutes come as "MM:SS" strings from the API.
    Parsing: "36:30" -> 36.5, "0:00" -> 0, None -> 0, "" -> 0.
    """

    def _parse_minutes(self, min_str):
        """Replicate the minutes parsing logic from data_pipeline.py."""
        min_str = min_str or "0:00"
        try:
            if ":" in min_str:
                parts = min_str.split(":")
                return int(parts[0]) + int(parts[1]) / 60
            else:
                return float(min_str)
        except (ValueError, TypeError, IndexError):
            return 0.0

    def test_standard_minutes(self):
        assert abs(self._parse_minutes("36:30") - 36.5) < 0.01

    def test_zero_minutes(self):
        assert self._parse_minutes("0:00") == 0.0

    def test_none_minutes(self):
        assert self._parse_minutes(None) == 0.0

    def test_empty_string(self):
        assert self._parse_minutes("") == 0.0

    def test_integer_minutes(self):
        """Some APIs return just the integer minutes."""
        assert abs(self._parse_minutes("32") - 32.0) < 0.01

    def test_float_string(self):
        assert abs(self._parse_minutes("35.5") - 35.5) < 0.01

    def test_full_game_minutes(self):
        assert abs(self._parse_minutes("48:00") - 48.0) < 0.01

    def test_overtime_minutes(self):
        assert abs(self._parse_minutes("52:15") - 52.25) < 0.01

    def test_single_digit_seconds(self):
        assert abs(self._parse_minutes("25:05") - (25 + 5/60)) < 0.01


# ===========================================================================
# 4. PLAYER PROP DATACLASS
# ===========================================================================

class TestPlayerPropDataclass:
    """PlayerProp should preserve all_books data through the pipeline."""

    def test_player_prop_has_all_books(self):
        from data_pipeline import PlayerProp
        all_books = {
            "draftkings": {"line": 24.5, "over_odds": -110, "under_odds": -110},
            "fanduel": {"line": 25.0, "over_odds": -105, "under_odds": -115},
        }
        prop = PlayerProp(
            player_id=123,
            player_name="Test Player",
            prop_type="points",
            line=24.5,
            over_odds=-110,
            under_odds=-110,
            all_books=all_books,
        )
        assert prop.all_books == all_books
        assert len(prop.all_books) == 2

    def test_player_prop_default_all_books_is_empty(self):
        from data_pipeline import PlayerProp
        prop = PlayerProp(
            player_id=123,
            player_name="Test Player",
            prop_type="points",
            line=24.5,
            over_odds=-110,
            under_odds=-110,
        )
        assert prop.all_books == {}

    def test_player_prop_dk_fields(self):
        """DK-specific fields should be preserved."""
        from data_pipeline import PlayerProp
        prop = PlayerProp(
            player_id=123,
            player_name="Test Player",
            prop_type="points",
            line=24.5,
            over_odds=-110,
            under_odds=-110,
            dk_line=24.5,
            dk_over_odds=-110,
            dk_under_odds=-110,
        )
        assert prop.dk_line == 24.5
        assert prop.dk_over_odds == -110
        assert prop.dk_under_odds == -110


# ===========================================================================
# 5. GAMELOG DATACLASS
# ===========================================================================

class TestGameLogDataclass:
    """GameLog should store all stat fields."""

    def test_game_log_creation(self):
        from data_pipeline import GameLog
        gl = GameLog(
            game_id=123, date="2026-01-01", opponent="LAL",
            home=True, minutes=32.5, pts=25, reb=7, ast=5,
            stl=2, blk=1, fg3m=3, turnover=2, fga=18, fgm=10,
            fta=5, ftm=4,
        )
        assert gl.pts == 25
        assert gl.minutes == 32.5
        assert gl.turnover == 2


# ===========================================================================
# 6. ODDS CONVERSION CONSISTENCY
# ===========================================================================

class TestOddsConversion:
    """American-to-probability conversion used across the pipeline."""

    def test_minus_110_both_sides_has_vig(self):
        """Standard -110/-110 should imply >100% total (the vig)."""
        from v2.edge import EdgeCalculator
        calc = EdgeCalculator()
        p_over = calc._american_to_prob(-110)
        p_under = calc._american_to_prob(-110)
        total = p_over + p_under
        assert total > 1.0  # Vig exists
        assert abs(total - (110/210 + 110/210)) < 0.001

    def test_american_to_decimal_negative(self):
        from v2.edge import EdgeCalculator
        calc = EdgeCalculator()
        # -110 -> 1 + 100/110 = 1.909
        dec = calc._american_to_decimal(-110)
        assert abs(dec - (1 + 100/110)) < 0.01

    def test_american_to_decimal_positive(self):
        from v2.edge import EdgeCalculator
        calc = EdgeCalculator()
        # +200 -> 1 + 200/100 = 3.0
        dec = calc._american_to_decimal(200)
        assert abs(dec - 3.0) < 0.01

    def test_american_to_decimal_zero(self):
        from v2.edge import EdgeCalculator
        calc = EdgeCalculator()
        dec = calc._american_to_decimal(0)
        assert abs(dec - 2.0) < 0.01
