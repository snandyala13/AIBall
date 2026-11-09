"""
serializers.py - Convert dataclasses to JSON-safe dicts

Handles Enum -> .value, datetime -> .isoformat(), nested dataclasses, etc.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


def _convert(obj):
    """Recursively convert an object to a JSON-serializable form."""
    if obj is None:
        return None
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (int, float, str, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(item) for item in obj]
    if hasattr(obj, '__dict__'):
        return {k: _convert(v) for k, v in obj.__dict__.items()}
    return str(obj)


def serialize_game_log(log):
    # type: (Any) -> Dict
    """Serialize a GameLog dataclass."""
    return {
        "game_id": log.game_id,
        "date": log.date,
        "opponent": log.opponent,
        "home": log.home,
        "minutes": log.minutes,
        "pts": log.pts,
        "reb": log.reb,
        "ast": log.ast,
        "stl": log.stl,
        "blk": log.blk,
        "fg3m": log.fg3m,
        "turnover": log.turnover,
        "fga": log.fga,
        "fgm": log.fgm,
        "fta": log.fta,
        "ftm": log.ftm,
    }


def serialize_player_profile(profile):
    # type: (Any) -> Dict
    """Serialize a PlayerProfile dataclass."""
    return {
        "id": profile.id,
        "name": profile.name,
        "team": profile.team,
        "team_id": profile.team_id,
        "position": profile.position,
        "season_avg": dict(profile.season_avg),
        "last_3_avg": dict(profile.last_3_avg),
        "last_5_avg": dict(profile.last_5_avg),
        "last_10_avg": dict(profile.last_10_avg),
        "std_dev": dict(profile.std_dev),
        "per_minute": dict(profile.per_minute),
        "usage_pct": profile.usage_pct,
        "home_avg": dict(profile.home_avg),
        "away_avg": dict(profile.away_avg),
        "home_l5_avg": dict(profile.home_l5_avg),
        "away_l5_avg": dict(profile.away_l5_avg),
        "status": profile.status,
        "injury_note": profile.injury_note,
        "game_logs": [serialize_game_log(g) for g in profile.game_logs],
    }


def serialize_projection(proj):
    # type: (Any) -> Dict
    """Serialize a Projection dataclass."""
    return {
        "player_name": proj.player_name,
        "prop_type": proj.prop_type.value if isinstance(proj.prop_type, Enum) else str(proj.prop_type),
        "projected_value": round(proj.projected_value, 2),
        "projected_minutes": round(proj.projected_minutes, 1),
        "std_dev": round(proj.std_dev, 2),
        "confidence": round(proj.confidence, 3),
        "team": proj.team,
        "dk_line": proj.dk_line,
        "dk_over_odds": proj.dk_over_odds,
        "dk_under_odds": proj.dk_under_odds,
        "over_probability": round(proj.over_probability, 4),
        "under_probability": round(proj.under_probability, 4),
        "implied_over_prob": round(proj.implied_over_prob, 4),
        "implied_under_prob": round(proj.implied_under_prob, 4),
        "edge_over": round(proj.edge_over, 4),
        "edge_under": round(proj.edge_under, 4),
        "base_projection": round(proj.base_projection, 2),
        "minutes_adjustment": round(proj.minutes_adjustment, 2),
        "fatigue_adjustment": round(proj.fatigue_adjustment, 2),
        "blowout_adjustment": round(proj.blowout_adjustment, 2),
        "pace_adjustment": round(proj.pace_adjustment, 2),
        "home_away_adjustment": round(proj.home_away_adjustment, 2),
        "defense_adjustment": round(proj.defense_adjustment, 2),
        "tracking_adjustment": round(proj.tracking_adjustment, 2),
        "usage_boost": round(proj.usage_boost, 2),
        "source_layer": proj.source_layer,
        "games_sample": proj.games_sample,
        "role_change": proj.role_change,
        "best_over_book": getattr(proj, 'best_over_book', 'draftkings'),
        "best_under_book": getattr(proj, 'best_under_book', 'draftkings'),
        "all_books": getattr(proj, 'all_books', {}),
    }


def serialize_bet(bet, best_lines=None):
    # type: (Any, Optional[Dict]) -> Dict
    """Serialize a Bet dataclass with optional cross-book line data."""
    result = {
        "tier": bet.tier.value if isinstance(bet.tier, Enum) else str(bet.tier),
        "description": bet.description,
        "player": bet.player,
        "prop_type": bet.prop_type,
        "side": bet.side,
        "line": bet.line,
        "dk_odds": bet.dk_odds,
        "fair_odds": bet.fair_odds,
        "edge_percent": round(bet.edge_percent, 2),
        "confidence": round(bet.confidence, 3),
        "recommended_units": round(bet.recommended_units, 2),
        "ev_per_unit": round(bet.ev_per_unit, 4),
        "reasoning": bet.reasoning,
        "projected_value": round(bet.projected_value, 2),
        "our_probability": round(bet.our_probability, 4),
        "implied_probability": round(bet.implied_probability, 4),
        "player_tier": bet.player_tier.value if isinstance(bet.player_tier, Enum) else str(bet.player_tier),
        "minutes_avg": round(bet.minutes_avg, 1),
        "reasoning_factors": bet.reasoning_factors or [],
        "ml_prob": round(getattr(bet, 'ml_prob', 0), 4),
        "line_signal": getattr(bet, 'line_signal', 'neutral'),
    }

    # Attach cross-book best line info
    # Try best_lines lookup dict first, then check if bet has all_books directly
    all_books = getattr(bet, 'all_books', None)
    best_book = getattr(bet, 'best_over_book' if bet.side == 'over' else 'best_under_book', None)

    if best_lines:
        lookup_key = "{}|{}|{}".format(bet.player, bet.prop_type, bet.side)
        bl = best_lines.get(lookup_key)
        if bl:
            all_books = bl.get("all_books", all_books or {})
            best_book = bl.get("best_book", best_book)

    if all_books and len(all_books) > 1:
        result["best_line_info"] = {
            "best_book": best_book or "draftkings",
            "all_books": {k: {"line": v.get("line"), "over_odds": v.get("over_odds"), "under_odds": v.get("under_odds")} for k, v in all_books.items()},
        }

    return result


def serialize_parlay(parlay, best_lines=None):
    # type: (Any, Optional[Dict]) -> Dict
    """Serialize a ParlayBet dataclass with optional cross-book line data."""
    return {
        "tier": parlay.tier.value if isinstance(parlay.tier, Enum) else str(parlay.tier),
        "legs": [serialize_bet(leg, best_lines=best_lines) for leg in parlay.legs],
        "combined_dk_odds": parlay.combined_dk_odds,
        "true_fair_odds": parlay.true_fair_odds,
        "naive_fair_odds": parlay.naive_fair_odds,
        "edge_percent": round(parlay.edge_percent, 2),
        "recommended_units": round(parlay.recommended_units, 2),
        "ev_per_unit": round(parlay.ev_per_unit, 4),
        "correlation": round(parlay.correlation, 3),
        "true_probability": round(parlay.true_probability, 4),
        "naive_probability": round(parlay.naive_probability, 4),
        "reasoning": parlay.reasoning,
    }


def serialize_game_recommendations(recs, best_lines=None):
    # type: (Any, Optional[Dict]) -> Dict
    """Serialize a GameRecommendations dataclass with optional cross-book data."""
    result = {
        "matchup": recs.matchup,
        "game_id": recs.game_id,
        "timestamp": recs.timestamp.isoformat() if isinstance(recs.timestamp, datetime) else str(recs.timestamp),
        "spread": recs.spread,
        "total": recs.total,
        "best_singles": [serialize_bet(b, best_lines=best_lines) for b in recs.best_singles],
        "popoff_singles": [serialize_bet(b, best_lines=best_lines) for b in getattr(recs, 'popoff_singles', [])],
        "high_confidence_parlays": [serialize_parlay(p, best_lines=best_lines) for p in recs.high_confidence_parlays],
        "safe_parlays": [serialize_parlay(p, best_lines=best_lines) for p in recs.safe_parlays],
        "standard_parlays": [serialize_parlay(p, best_lines=best_lines) for p in recs.standard_parlays],
        "longshot_parlays": [serialize_parlay(p, best_lines=best_lines) for p in recs.longshot_parlays],
        "total_recommended_units": round(recs.total_recommended_units, 2),
        "bankroll_allocation": recs.bankroll_allocation,
    }

    # Line edge signals (keyed by "player|prop_type")
    line_signals = getattr(recs, 'line_signals', None)
    if line_signals:
        result["line_signals"] = _convert(line_signals)

    # CLV stats
    clv_stats = getattr(recs, 'clv_stats', None)
    if clv_stats and clv_stats.get('available'):
        result["clv_stats"] = _convert(clv_stats)

    return result


def serialize_v2_prediction(pred):
    # type: (Dict) -> Dict
    """Serialize a v2 ML model prediction for the web UI.

    The v2 model produces line-anchor predictions with uncertainty bands.
    This formats the output for the frontend bet card display.
    """
    return {
        # Player / prop
        "player": pred.get("player", ""),
        "prop_type": pred.get("prop_type", ""),
        "side": pred.get("side", "over"),
        "line": pred.get("line", 0),
        "dk_odds": pred.get("dk_odds", -110),
        "over_odds": pred.get("over_odds", -110),
        "under_odds": pred.get("under_odds", -110),

        # v2 model output (all None-safe)
        "predicted_value": round(pred.get("predicted_value") or 0, 2),
        "predicted_adjustment": round(pred.get("predicted_adjustment") or 0, 3),
        "p10": round(pred.get("p10") or 0, 1),
        "p90": round(pred.get("p90") or 0, 1),
        "uncertainty": round(pred.get("uncertainty") or 0, 2),
        "confidence": round(pred.get("confidence") or 0, 3),
        "confidence_stars": pred.get("confidence_stars") or 1,

        # Edge
        "edge_percent": round(pred.get("edge_percent") or 0, 2),
        "edge": round(pred.get("edge") or 0, 4),
        "our_probability": round(pred.get("our_probability") or 0.5, 4),
        "implied_probability": round(pred.get("implied_probability") or 0.5, 4),
        "p_over": round(pred.get("p_over") or 0.5, 4),
        "p_under": round(pred.get("p_under") or 0.5, 4),

        # Sizing
        "kelly_fraction": round(pred.get("kelly_fraction") or 0, 5),
        "recommended_units": round(pred.get("recommended_units") or 0, 1),

        # Cross-book
        "all_books": pred.get("all_books", {}),
        "best_book": pred.get("best_book", None),
        "best_odds": pred.get("best_odds", None),
        "edge_vs_dk": pred.get("edge_vs_dk", 0),
        "cross_book_signal": pred.get("cross_book_signal", "NEUTRAL"),

        # Line shopping (EV at each book, sorted best-first)
        "line_shopping": pred.get("line_shopping", []),
        "edge_at_dk": pred.get("edge_at_dk", 0),
        "odds_improvement": pred.get("odds_improvement", 0),
        "odds_tier": pred.get("odds_tier", "standard"),

        # Arbitrage and middle opportunities
        "arbitrage": pred.get("arbitrage", []),
        "middles": pred.get("middles", []),

        # Signal data
        "sigma": round(pred.get("sigma", 0), 3),
        "z_score": round(pred.get("z_score", 0), 3),

        # Empirical filter
        "filter_tier": pred.get("filter_tier", "marginal"),
        "filter_reason": pred.get("filter_reason", ""),

        # Source
        "source": pred.get("source", "v2_line_anchor"),

        # v6 dual-model extras (None when not using v6)
        "p_over_huber": pred.get("p_over_huber"),
        "p_over_rf": pred.get("p_over_rf"),
        "consensus": pred.get("consensus", False),

        # Stale line detection
        "staleness_score": pred.get("staleness_score", 0),
        "is_stale": pred.get("is_stale", False),
        "stale_reason": pred.get("stale_reason", ""),
    }


def serialize_game_info(game, dk_odds=None):
    # type: (Dict, Optional[Dict]) -> Dict
    """Serialize a game dict from the API into a clean format."""
    home = game.get("home_team", {})
    away = game.get("visitor_team", {})
    game_id = game.get("id")
    odds = dk_odds or {}

    return {
        "game_id": game_id,
        "home_team": home.get("full_name", ""),
        "home_abbreviation": home.get("abbreviation", ""),
        "away_team": away.get("full_name", ""),
        "away_abbreviation": away.get("abbreviation", ""),
        "status": game.get("status", ""),
        "date": game.get("date", "")[:10],
        "spread": odds.get("spread", 0),
        "total": odds.get("total", 0),
        "home_team_id": home.get("id"),
        "away_team_id": away.get("id"),
    }
