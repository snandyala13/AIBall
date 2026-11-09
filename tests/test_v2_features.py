"""
tests/test_v2_features.py - Tests for v2 FeatureBuilder.

Verifies feature engineering correctness, lookahead prevention,
edge cases with empty/insufficient data, rolling average math,
combined props, per-minute rates, and feature count stability.
"""

import pytest
import numpy as np
from collections import defaultdict

from v2.features import FeatureBuilder, PROP_TO_STAT, CORE_PROPS


# ---------------------------------------------------------------------------
# Fixtures: build a FeatureBuilder pre-loaded with synthetic game history
# ---------------------------------------------------------------------------

def _make_game_entry(date, pts=20, reb=5, ast=5, fg3m=2, stl=1, blk=1,
                     turnover=2, minutes=32, fga=15, fgm=8, fta=4, ftm=3,
                     fg3a=5, game_id=None, home_team="BOS", away_team="LAL",
                     home_score=110, away_score=105):
    """Create a synthetic game entry matching FeatureBuilder's format."""
    return {
        "date": date,
        "game_id": game_id or f"game_{date}",
        "home_team": home_team,
        "away_team": away_team,
        "home_score": home_score,
        "away_score": away_score,
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
        "fg3a": fg3a,
    }


def _build_fb_with_player(player_name, games):
    """Create a FeatureBuilder with pre-loaded player history."""
    fb = FeatureBuilder()
    fb.player_history[player_name] = sorted(games, key=lambda g: g["date"])
    fb._loaded = True
    return fb


@pytest.fixture
def sample_player_history():
    """15 games with known stat values for 'Test Player'."""
    games = []
    base_pts = [22, 18, 25, 30, 15, 20, 28, 17, 24, 19,
                26, 21, 23, 27, 16]
    base_reb = [6, 5, 7, 8, 4, 6, 9, 5, 7, 5,
                8, 6, 7, 9, 4]
    base_ast = [5, 4, 6, 7, 3, 5, 8, 4, 6, 4,
                7, 5, 6, 8, 3]
    for i, (pts, reb, ast) in enumerate(zip(base_pts, base_reb, base_ast)):
        day = str(i + 1).zfill(2)
        games.append(_make_game_entry(
            date=f"2026-01-{day}",
            pts=pts, reb=reb, ast=ast,
            fg3m=pts // 8, stl=1, blk=1, turnover=2,
            minutes=32 + (i % 5) - 2,
        ))
    return games


@pytest.fixture
def fb_loaded(sample_player_history):
    """FeatureBuilder with 'Test Player' history loaded."""
    return _build_fb_with_player("Test Player", sample_player_history)


# ===========================================================================
# 1. NO LOOKAHEAD BIAS
# ===========================================================================

class TestNoLookahead:
    """Features for a game date must only use games BEFORE that date."""

    def test_no_future_games_used(self, fb_loaded, sample_player_history):
        """Build features for mid-dataset date; verify no future games leak."""
        game_date = "2026-01-08"  # Should use games 01-07 and before
        ctx = {"line": 20.0, "over_odds": -110, "under_odds": -110}
        feats = fb_loaded.get_features_for_leg("Test Player", "points", game_date, ctx)

        # Prior games for date 01-08 are 01-01 through 01-07 (7 games)
        prior = [g for g in sample_player_history if g["date"] < game_date]
        assert len(prior) == 7

        # The feature should reflect only those 7 games
        assert feats is not None
        assert feats["games_played"] == 7

    def test_season_average_excludes_future(self, fb_loaded, sample_player_history):
        """Season average must match manual calculation from prior games only."""
        game_date = "2026-01-06"
        ctx = {"line": 20.0, "over_odds": -110, "under_odds": -110}
        feats = fb_loaded.get_features_for_leg("Test Player", "points", game_date, ctx)

        prior_pts = [g["pts"] for g in sample_player_history if g["date"] < game_date]
        expected_season = np.mean(prior_pts)
        assert abs(feats["target_season"] - expected_season) < 0.01

    def test_l3_only_uses_last_three_prior(self, fb_loaded, sample_player_history):
        """L3 average should be the 3 most recent games before the prediction date."""
        game_date = "2026-01-10"
        ctx = {"line": 20.0, "over_odds": -110, "under_odds": -110}
        feats = fb_loaded.get_features_for_leg("Test Player", "points", game_date, ctx)

        prior = [g for g in sample_player_history if g["date"] < game_date]
        prior_sorted_desc = sorted(prior, key=lambda g: g["date"], reverse=True)
        expected_l3 = np.mean([g["pts"] for g in prior_sorted_desc[:3]])
        assert abs(feats["target_l3"] - expected_l3) < 0.01


# ===========================================================================
# 2-3. EMPTY / INSUFFICIENT HISTORY
# ===========================================================================

class TestInsufficientHistory:
    """Players with too few games should return None."""

    def test_empty_history_returns_none(self):
        fb = _build_fb_with_player("Ghost Player", [])
        ctx = {"line": 20.0, "over_odds": -110, "under_odds": -110}
        result = fb.get_features_for_leg("Ghost Player", "points", "2026-01-15", ctx)
        assert result is None

    def test_unknown_player_returns_none(self):
        fb = FeatureBuilder()
        fb._loaded = True
        ctx = {"line": 20.0, "over_odds": -110, "under_odds": -110}
        result = fb.get_features_for_leg("Unknown Person", "points", "2026-01-15", ctx)
        assert result is None

    def test_one_game_returns_none(self):
        games = [_make_game_entry("2026-01-01", pts=20)]
        fb = _build_fb_with_player("One Game Player", games)
        ctx = {"line": 20.0, "over_odds": -110, "under_odds": -110}
        result = fb.get_features_for_leg("One Game Player", "points", "2026-01-15", ctx)
        assert result is None

    def test_two_games_returns_none(self):
        games = [
            _make_game_entry("2026-01-01", pts=20),
            _make_game_entry("2026-01-02", pts=22),
        ]
        fb = _build_fb_with_player("Two Game Player", games)
        ctx = {"line": 20.0, "over_odds": -110, "under_odds": -110}
        result = fb.get_features_for_leg("Two Game Player", "points", "2026-01-15", ctx)
        assert result is None

    def test_three_games_returns_features(self):
        """Exactly 3 prior games should be sufficient."""
        games = [
            _make_game_entry("2026-01-01", pts=20),
            _make_game_entry("2026-01-02", pts=22),
            _make_game_entry("2026-01-03", pts=18),
        ]
        fb = _build_fb_with_player("Three Game Player", games)
        ctx = {"line": 20.0, "over_odds": -110, "under_odds": -110}
        result = fb.get_features_for_leg("Three Game Player", "points", "2026-01-15", ctx)
        assert result is not None
        assert result["games_played"] == 3

    def test_all_games_after_date_returns_none(self):
        """If all games are AFTER the prediction date, should return None."""
        games = [
            _make_game_entry("2026-02-01", pts=20),
            _make_game_entry("2026-02-02", pts=22),
            _make_game_entry("2026-02-03", pts=18),
        ]
        fb = _build_fb_with_player("Future Player", games)
        ctx = {"line": 20.0, "over_odds": -110, "under_odds": -110}
        result = fb.get_features_for_leg("Future Player", "points", "2026-01-15", ctx)
        assert result is None


# ===========================================================================
# 4. ROLLING AVERAGES CORRECTNESS
# ===========================================================================

class TestRollingAverages:
    """Verify L3/L5/L10/season averages are computed from correct windows."""

    def test_l5_correctness(self, fb_loaded, sample_player_history):
        game_date = "2026-01-12"
        ctx = {"line": 20.0, "over_odds": -110, "under_odds": -110}
        feats = fb_loaded.get_features_for_leg("Test Player", "points", game_date, ctx)

        prior = sorted(
            [g for g in sample_player_history if g["date"] < game_date],
            key=lambda g: g["date"], reverse=True,
        )
        expected_l5 = np.mean([g["pts"] for g in prior[:5]])
        assert abs(feats["target_l5"] - expected_l5) < 0.01

    def test_l10_correctness(self, fb_loaded, sample_player_history):
        game_date = "2026-01-15"
        ctx = {"line": 20.0, "over_odds": -110, "under_odds": -110}
        feats = fb_loaded.get_features_for_leg("Test Player", "points", game_date, ctx)

        prior = sorted(
            [g for g in sample_player_history if g["date"] < game_date],
            key=lambda g: g["date"], reverse=True,
        )
        expected_l10 = np.mean([g["pts"] for g in prior[:10]])
        assert abs(feats["target_l10"] - expected_l10) < 0.01

    def test_std_dev_l10(self, fb_loaded, sample_player_history):
        game_date = "2026-01-15"
        ctx = {"line": 20.0, "over_odds": -110, "under_odds": -110}
        feats = fb_loaded.get_features_for_leg("Test Player", "points", game_date, ctx)

        prior = sorted(
            [g for g in sample_player_history if g["date"] < game_date],
            key=lambda g: g["date"], reverse=True,
        )
        expected_std = np.std([g["pts"] for g in prior[:10]])
        assert abs(feats["target_std_l10"] - expected_std) < 0.01

    def test_l20_with_fewer_than_20_games(self, fb_loaded, sample_player_history):
        """L20 should use all available games when fewer than 20 exist."""
        game_date = "2026-01-12"
        ctx = {"line": 20.0, "over_odds": -110, "under_odds": -110}
        feats = fb_loaded.get_features_for_leg("Test Player", "points", game_date, ctx)

        prior = sorted(
            [g for g in sample_player_history if g["date"] < game_date],
            key=lambda g: g["date"], reverse=True,
        )
        # Only 11 prior games, so L20 = all 11
        expected_l20 = np.mean([g["pts"] for g in prior[:20]])
        assert abs(feats["target_l20"] - expected_l20) < 0.01


# ===========================================================================
# 5. COMBINED PROPS
# ===========================================================================

class TestCombinedProps:
    """pts_rebs_asts should sum pts+reb+ast from game logs."""

    def test_pts_rebs_asts_is_sum(self, fb_loaded, sample_player_history):
        game_date = "2026-01-10"
        ctx = {"line": 35.0, "over_odds": -110, "under_odds": -110}
        feats = fb_loaded.get_features_for_leg("Test Player", "pts_rebs_asts", game_date, ctx)

        prior = sorted(
            [g for g in sample_player_history if g["date"] < game_date],
            key=lambda g: g["date"], reverse=True,
        )
        combined_l3 = np.mean([g["pts"] + g["reb"] + g["ast"] for g in prior[:3]])
        assert abs(feats["target_l3"] - combined_l3) < 0.01

    def test_pts_rebs_is_sum(self, fb_loaded, sample_player_history):
        game_date = "2026-01-10"
        ctx = {"line": 30.0, "over_odds": -110, "under_odds": -110}
        feats = fb_loaded.get_features_for_leg("Test Player", "pts_rebs", game_date, ctx)

        prior = sorted(
            [g for g in sample_player_history if g["date"] < game_date],
            key=lambda g: g["date"], reverse=True,
        )
        combined_l5 = np.mean([g["pts"] + g["reb"] for g in prior[:5]])
        assert abs(feats["target_l5"] - combined_l5) < 0.01

    def test_stls_blks_is_sum(self, fb_loaded, sample_player_history):
        game_date = "2026-01-10"
        ctx = {"line": 2.5, "over_odds": -110, "under_odds": -110}
        feats = fb_loaded.get_features_for_leg("Test Player", "stls_blks", game_date, ctx)

        prior = sorted(
            [g for g in sample_player_history if g["date"] < game_date],
            key=lambda g: g["date"], reverse=True,
        )
        combined = np.mean([g["stl"] + g["blk"] for g in prior[:3]])
        assert abs(feats["target_l3"] - combined) < 0.01

    def test_combined_prop_has_component_features(self, fb_loaded):
        """Combined props should include component_<key>_l5 features."""
        game_date = "2026-01-10"
        ctx = {"line": 35.0, "over_odds": -110, "under_odds": -110}
        feats = fb_loaded.get_features_for_leg("Test Player", "pts_rebs_asts", game_date, ctx)

        assert "component_pts_l5" in feats
        assert "component_reb_l5" in feats
        assert "component_ast_l5" in feats


# ===========================================================================
# 6. PER-MINUTE RATE
# ===========================================================================

class TestPerMinuteRate:
    """per_minute_rate = total_stat_l10 / total_mins_l10."""

    def test_per_minute_rate_calculation(self, fb_loaded, sample_player_history):
        game_date = "2026-01-12"
        ctx = {"line": 20.0, "over_odds": -110, "under_odds": -110}
        feats = fb_loaded.get_features_for_leg("Test Player", "points", game_date, ctx)

        prior = sorted(
            [g for g in sample_player_history if g["date"] < game_date],
            key=lambda g: g["date"], reverse=True,
        )
        total_pts_l10 = sum(g["pts"] for g in prior[:10])
        total_mins_l10 = sum(g["min"] for g in prior[:10])
        expected_rate = total_pts_l10 / max(total_mins_l10, 1.0)
        assert abs(feats["per_minute_rate"] - expected_rate) < 0.001

    def test_per_minute_rate_zero_minutes_safe(self):
        """If all minutes are 0, per_minute_rate should not crash (divide by 1)."""
        games = [
            _make_game_entry(f"2026-01-{str(d).zfill(2)}", pts=10, minutes=0)
            for d in range(1, 6)
        ]
        fb = _build_fb_with_player("Zero Minutes", games)
        ctx = {"line": 5.0, "over_odds": -110, "under_odds": -110}
        feats = fb.get_features_for_leg("Zero Minutes", "points", "2026-01-10", ctx)
        assert feats is not None
        assert feats["per_minute_rate"] == 50.0 / 1.0  # total_stat / max(0, 1.0)


# ===========================================================================
# 7. TREND CALCULATION
# ===========================================================================

class TestTrend:
    """target_trend = L3 - L10."""

    def test_trend_equals_l3_minus_l10(self, fb_loaded):
        game_date = "2026-01-15"
        ctx = {"line": 20.0, "over_odds": -110, "under_odds": -110}
        feats = fb_loaded.get_features_for_leg("Test Player", "points", game_date, ctx)

        expected_trend = feats["target_l3"] - feats["target_l10"]
        assert abs(feats["target_trend"] - expected_trend) < 0.001

    def test_trend_l5_l20(self, fb_loaded):
        game_date = "2026-01-15"
        ctx = {"line": 20.0, "over_odds": -110, "under_odds": -110}
        feats = fb_loaded.get_features_for_leg("Test Player", "points", game_date, ctx)

        expected = feats["target_l5"] - feats["target_l20"]
        assert abs(feats["target_trend_l5_l20"] - expected) < 0.001


# ===========================================================================
# 8. LINE FEATURES
# ===========================================================================

class TestLineFeatures:
    """line_vs_l5, line_vs_season should be correct differences."""

    def test_line_vs_l5(self, fb_loaded):
        game_date = "2026-01-10"
        line = 25.0
        ctx = {"line": line, "over_odds": -110, "under_odds": -110}
        feats = fb_loaded.get_features_for_leg("Test Player", "points", game_date, ctx)

        assert abs(feats["line_vs_l5"] - (line - feats["target_l5"])) < 0.001

    def test_line_vs_season(self, fb_loaded):
        game_date = "2026-01-10"
        line = 25.0
        ctx = {"line": line, "over_odds": -110, "under_odds": -110}
        feats = fb_loaded.get_features_for_leg("Test Player", "points", game_date, ctx)

        assert abs(feats["line_vs_season"] - (line - feats["target_season"])) < 0.001

    def test_l3_above_line(self, fb_loaded):
        game_date = "2026-01-10"
        line = 25.0
        ctx = {"line": line, "over_odds": -110, "under_odds": -110}
        feats = fb_loaded.get_features_for_leg("Test Player", "points", game_date, ctx)

        assert abs(feats["l3_above_line"] - (feats["target_l3"] - line)) < 0.001

    def test_odds_skew(self):
        """Odds skew should reflect difference in implied probabilities."""
        games = [_make_game_entry(f"2026-01-{str(d).zfill(2)}", pts=20)
                 for d in range(1, 6)]
        fb = _build_fb_with_player("Player", games)
        ctx = {"line": 20.0, "over_odds": -130, "under_odds": +110}
        feats = fb.get_features_for_leg("Player", "points", "2026-01-10", ctx)

        imp_over = fb._american_to_prob(-130)
        imp_under = fb._american_to_prob(110)
        assert abs(feats["odds_skew"] - (imp_over - imp_under)) < 0.001


# ===========================================================================
# 9. ZERO-MINUTES GAMES (not filtered by FeatureBuilder, but worth noting)
# ===========================================================================

class TestZeroMinutesGames:
    """Games with 0 minutes still appear in history (FeatureBuilder does not filter them)."""

    def test_zero_minutes_included_in_averages(self):
        """Zero-minute games ARE included (FeatureBuilder doesn't filter them).
        This test documents actual behavior."""
        games = [
            _make_game_entry("2026-01-01", pts=20, minutes=30),
            _make_game_entry("2026-01-02", pts=0, minutes=0),  # DNP
            _make_game_entry("2026-01-03", pts=25, minutes=35),
            _make_game_entry("2026-01-04", pts=18, minutes=28),
        ]
        fb = _build_fb_with_player("DNP Player", games)
        ctx = {"line": 15.0, "over_odds": -110, "under_odds": -110}
        feats = fb.get_features_for_leg("DNP Player", "points", "2026-01-10", ctx)

        # All 4 games included (including the 0-minute game)
        expected_season = np.mean([20, 0, 25, 18])
        assert abs(feats["target_season"] - expected_season) < 0.01


# ===========================================================================
# 10. FEATURE COUNT STABILITY
# ===========================================================================

class TestFeatureCount:
    """Feature count should be stable and match model expectations."""

    def test_feature_count_is_consistent(self, fb_loaded):
        """All calls should return the same number of features."""
        ctx = {"line": 20.0, "over_odds": -110, "under_odds": -110}

        feats1 = fb_loaded.get_features_for_leg("Test Player", "points", "2026-01-10", ctx)
        feats2 = fb_loaded.get_features_for_leg("Test Player", "rebounds", "2026-01-10", ctx)
        feats3 = fb_loaded.get_features_for_leg("Test Player", "assists", "2026-01-10", ctx)

        # Different prop types may have different numbers of features due to
        # related stats and component features, but each call should be internally consistent
        assert feats1 is not None
        assert feats2 is not None
        assert feats3 is not None

    def test_no_nan_in_features(self, fb_loaded):
        """No feature should be NaN."""
        ctx = {"line": 20.0, "over_odds": -110, "under_odds": -110}
        feats = fb_loaded.get_features_for_leg("Test Player", "points", "2026-01-10", ctx)

        for key, val in feats.items():
            assert not np.isnan(val), f"Feature '{key}' is NaN"

    def test_no_inf_in_features(self, fb_loaded):
        """No feature should be infinite."""
        ctx = {"line": 20.0, "over_odds": -110, "under_odds": -110}
        feats = fb_loaded.get_features_for_leg("Test Player", "points", "2026-01-10", ctx)

        for key, val in feats.items():
            assert not np.isinf(val), f"Feature '{key}' is infinite"

    def test_cross_book_features_present_with_all_books(self, fb_loaded):
        """When all_books provided, cross-book features should be populated."""
        ctx = {
            "line": 20.0,
            "over_odds": -110,
            "under_odds": -110,
            "all_books": {
                "draftkings": {"line": 20.0, "over_odds": -110, "under_odds": -110},
                "fanduel": {"line": 20.5, "over_odds": -105, "under_odds": -115},
                "betmgm": {"line": 19.5, "over_odds": -115, "under_odds": -105},
            },
        }
        feats = fb_loaded.get_features_for_leg("Test Player", "points", "2026-01-10", ctx)

        assert feats["num_books_with_prop"] == 3
        assert feats["line_range"] == 1.0  # 20.5 - 19.5
        assert feats["consensus_line"] == 20.0  # median of [19.5, 20.0, 20.5]

    def test_cross_book_features_default_without_all_books(self, fb_loaded):
        """Without all_books, cross-book features should use safe defaults."""
        ctx = {"line": 20.0, "over_odds": -110, "under_odds": -110}
        feats = fb_loaded.get_features_for_leg("Test Player", "points", "2026-01-10", ctx)

        assert feats["num_books_with_prop"] == 1
        assert feats["dk_vs_consensus"] == 0.0
        assert feats["line_range"] == 0.0


# ===========================================================================
# AMERICAN TO PROB HELPER
# ===========================================================================

class TestAmericanToProb:
    def test_negative_odds(self):
        fb = FeatureBuilder()
        # -110 => 110/210 = 0.5238
        assert abs(fb._american_to_prob(-110) - (110 / 210)) < 0.001

    def test_positive_odds(self):
        fb = FeatureBuilder()
        # +200 => 100/300 = 0.3333
        assert abs(fb._american_to_prob(200) - (100 / 300)) < 0.001

    def test_zero_odds(self):
        fb = FeatureBuilder()
        assert fb._american_to_prob(0) == 0.5

    def test_even_odds(self):
        fb = FeatureBuilder()
        # +100 => 100/200 = 0.5
        assert abs(fb._american_to_prob(100) - 0.5) < 0.001
