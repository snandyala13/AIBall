#!/usr/bin/env python3
"""
v2/stale_lines.py - Stale Line Detection System

Identifies DraftKings prop lines that haven't been updated despite new information:
- Recent performance significantly diverges from the line
- Our model projection diverges significantly from the line
- Correlated markets (game total, spread) have moved but props haven't
- Other books have moved their lines but DK hasn't
- Injury report changes (teammate OUT) should shift props but lines are unmoved

A high staleness score (60+) indicates an exploitable inefficiency.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

PROP_TO_STAT = {
    "points": ["pts"],
    "rebounds": ["reb"],
    "assists": ["ast"],
    "threes": ["fg3m"],
    "steals": ["stl"],
    "blocks": ["blk"],
    "turnovers": ["turnover"],
    "pts_rebs_asts": ["pts", "reb", "ast"],
    "pts_rebs": ["pts", "reb"],
    "pts_asts": ["pts", "ast"],
    "rebs_asts": ["reb", "ast"],
    "stls_blks": ["stl", "blk"],
}


class StaleLineSignal:
    """A single staleness signal with score and human-readable reason."""

    def __init__(self, name: str, score: float, reason: str, details: Optional[Dict] = None):
        self.name = name
        self.score = min(max(score, 0.0), 100.0)  # clamp 0-100
        self.reason = reason
        self.details = details or {}

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "score": round(self.score, 1),
            "reason": self.reason,
            "details": self.details,
        }


class StaleLineResult:
    """Complete staleness assessment for a single player prop."""

    def __init__(
        self,
        player: str,
        prop_type: str,
        line: float,
        staleness_score: float,
        signals: List[StaleLineSignal],
        is_stale: bool,
    ):
        self.player = player
        self.prop_type = prop_type
        self.line = line
        self.staleness_score = min(max(staleness_score, 0.0), 100.0)
        self.signals = signals
        self.is_stale = is_stale

    def to_dict(self) -> Dict:
        return {
            "player": self.player,
            "prop_type": self.prop_type,
            "line": self.line,
            "staleness_score": round(self.staleness_score, 1),
            "is_stale": self.is_stale,
            "signals": [s.to_dict() for s in self.signals],
            "top_reason": self.signals[0].reason if self.signals else "",
        }


class StaleLineDetector:
    """
    Detects stale DraftKings prop lines by combining multiple signals.

    Usage:
        detector = StaleLineDetector(feature_builder=fb)
        results = detector.detect_stale_lines(props, game_date, game_context)
    """

    # Signal weights for composite score
    WEIGHTS = {
        "recent_performance": 0.30,
        "model_divergence": 0.25,
        "cross_book": 0.25,
        "correlated_market": 0.10,
        "injury_cascade": 0.10,
    }

    # Thresholds
    STALE_THRESHOLD = 60       # staleness_score >= 60 = "stale"
    RECENT_PERF_GAP = 0.15    # 15% gap between L3 avg and line
    MODEL_DIVERGENCE_GAP = 0.10  # 10% model-vs-line divergence
    MODEL_EDGE_MIN = 0.05     # minimum edge to trigger model divergence signal
    CROSS_BOOK_GAP = 1.0      # 1+ point DK vs median other books
    GAME_TOTAL_MOVE = 3.0     # 3+ point game total movement
    INJURY_CASCADE_MIN = 0.05  # 5% expected boost from teammate injury

    def __init__(self, feature_builder=None, predictor=None):
        """
        Args:
            feature_builder: v2.features.FeatureBuilder instance (with player history loaded)
            predictor: v2.predictor.StatPredictor instance (optional, for model divergence signal)
        """
        self.feature_builder = feature_builder
        self.predictor = predictor

    def _get_stat_value(self, game_entry: dict, prop_type: str) -> float:
        """Extract stat value for a prop type from a game entry."""
        keys = PROP_TO_STAT.get(prop_type)
        if keys is None:
            return 0.0
        return sum(game_entry.get(k, 0) for k in keys)

    def _get_recent_stats(self, player: str, prop_type: str, game_date: str, n: int = 3) -> List[float]:
        """Get last N game stat values for a player/prop before game_date."""
        if not self.feature_builder:
            return []
        history = self.feature_builder.player_history.get(player, [])
        prior = [g for g in history if g["date"] < game_date]
        if not prior:
            return []
        # Most recent first
        prior = list(reversed(prior))
        return [self._get_stat_value(g, prop_type) for g in prior[:n]]

    # ------------------------------------------------------------------
    # Signal 1: Recent Performance Gap
    # ------------------------------------------------------------------
    def _signal_recent_performance(
        self, player: str, prop_type: str, line: float, game_date: str
    ) -> StaleLineSignal:
        """
        Check if recent performance (L3) significantly diverges from line.
        A line set weeks ago that doesn't reflect a hot/cold streak.
        """
        recent = self._get_recent_stats(player, prop_type, game_date, n=3)
        if not recent or line <= 0:
            return StaleLineSignal("recent_performance", 0, "", {})

        recent_avg = np.mean(recent)
        gap = abs(recent_avg - line) / max(line, 1.0)

        # Also check L5 for additional context
        recent_5 = self._get_recent_stats(player, prop_type, game_date, n=5)
        l5_avg = np.mean(recent_5) if recent_5 else recent_avg

        # Consistency bonus: if all recent games are on the same side of the line
        all_above = all(v > line for v in recent)
        all_below = all(v < line for v in recent)
        consistency_bonus = 15.0 if (all_above or all_below) and len(recent) >= 3 else 0.0

        # Direction
        direction = "above" if recent_avg > line else "below"

        if gap < 0.08:
            score = 0.0
            reason = ""
        elif gap < self.RECENT_PERF_GAP:
            score = 20.0 + (gap / self.RECENT_PERF_GAP) * 30.0
            reason = f"L3 avg {recent_avg:.1f} is {gap*100:.0f}% {direction} line {line}"
        elif gap < 0.25:
            score = 55.0 + (gap - self.RECENT_PERF_GAP) / 0.10 * 20.0
            reason = f"Line at {line} but averaging {recent_avg:.1f} over last 3 games ({gap*100:.0f}% gap)"
        else:
            score = 85.0 + min((gap - 0.25) / 0.15 * 15.0, 15.0)
            reason = f"STALE: Line at {line} but averaging {recent_avg:.1f} over last 3 games"

        score = min(score + consistency_bonus, 100.0)

        details = {
            "recent_avg_l3": round(recent_avg, 1),
            "recent_avg_l5": round(l5_avg, 1),
            "line": line,
            "gap_pct": round(gap * 100, 1),
            "direction": direction,
            "recent_values": [round(v, 1) for v in recent],
            "all_same_side": all_above or all_below,
        }

        return StaleLineSignal("recent_performance", score, reason, details)

    # ------------------------------------------------------------------
    # Signal 2: Model Projection Divergence
    # ------------------------------------------------------------------
    def _signal_model_divergence(
        self,
        player: str,
        prop_type: str,
        line: float,
        game_date: str,
        game_context: Optional[dict] = None,
        prediction: Optional[dict] = None,
    ) -> StaleLineSignal:
        """
        Check if our ML model's projection significantly diverges from the line.
        Uses prediction dict if already computed, otherwise runs predictor.
        """
        model_value = None
        edge = None

        if prediction:
            model_value = prediction.get("predicted_value", None)
            edge = prediction.get("edge_percent", prediction.get("edge", 0))
            if edge and abs(edge) < 1:
                edge = edge * 100  # normalize to percentage

        if model_value is None or line <= 0:
            return StaleLineSignal("model_divergence", 0, "", {})

        projection_gap = abs(model_value - line) / max(line, 1.0)
        edge_abs = abs(edge) if edge else 0

        if projection_gap < 0.05 or edge_abs < self.MODEL_EDGE_MIN * 100:
            score = 0.0
            reason = ""
        elif projection_gap < self.MODEL_DIVERGENCE_GAP:
            score = 15.0 + (projection_gap / self.MODEL_DIVERGENCE_GAP) * 25.0
            reason = f"Model projects {model_value:.1f} vs line {line} ({projection_gap*100:.0f}% gap)"
        elif projection_gap < 0.20:
            score = 45.0 + (projection_gap - self.MODEL_DIVERGENCE_GAP) / 0.10 * 25.0
            reason = f"Model projects {model_value:.1f} vs line {line} ({edge_abs:.1f}% edge)"
        else:
            score = 75.0 + min((projection_gap - 0.20) / 0.15 * 25.0, 25.0)
            reason = f"Strong model divergence: projects {model_value:.1f} vs line {line}"

        details = {
            "model_projection": round(model_value, 1),
            "line": line,
            "projection_gap_pct": round(projection_gap * 100, 1),
            "edge_pct": round(edge_abs, 1),
        }

        return StaleLineSignal("model_divergence", score, reason, details)

    # ------------------------------------------------------------------
    # Signal 3: Cross-Book Divergence
    # ------------------------------------------------------------------
    def _signal_cross_book(
        self,
        player: str,
        prop_type: str,
        line: float,
        all_books: Dict,
    ) -> StaleLineSignal:
        """
        Check if DK's line diverges from other books' consensus.
        If DK is at 24.5 but everyone else is at 26.5, DK hasn't updated.
        """
        if not all_books or len(all_books) < 2 or line <= 0:
            return StaleLineSignal("cross_book", 0, "", {})

        other_lines = []
        dk_line = line
        for book, data in all_books.items():
            if book.lower() == "draftkings":
                dk_line_from_books = data.get("line")
                if dk_line_from_books is not None:
                    dk_line = float(dk_line_from_books)
                continue
            bl = data.get("line")
            if bl is not None:
                try:
                    other_lines.append(float(bl))
                except (TypeError, ValueError):
                    pass

        if not other_lines:
            return StaleLineSignal("cross_book", 0, "", {})

        median_other = float(np.median(other_lines))
        gap = abs(dk_line - median_other)

        # More books agreeing on a different line = stronger signal
        books_disagreeing = sum(1 for ol in other_lines if abs(ol - dk_line) >= 0.5)
        book_agreement_pct = books_disagreeing / len(other_lines) if other_lines else 0

        if gap < 0.5:
            score = 0.0
            reason = ""
        elif gap < self.CROSS_BOOK_GAP:
            score = 20.0 + (gap / self.CROSS_BOOK_GAP) * 30.0
            reason = f"DK at {dk_line} vs market median {median_other:.1f} ({gap:.1f}pt gap)"
        elif gap < 2.0:
            score = 55.0 + (gap - self.CROSS_BOOK_GAP) / 1.0 * 25.0
            reason = f"DK at {dk_line} but market consensus at {median_other:.1f}"
        else:
            score = 80.0 + min((gap - 2.0) / 1.0 * 20.0, 20.0)
            reason = f"STALE: DK at {dk_line} but {books_disagreeing} books at {median_other:.1f}"

        # Boost if many books disagree with DK
        if book_agreement_pct > 0.7 and score > 0:
            score = min(score + 10.0, 100.0)

        direction = "higher" if median_other > dk_line else "lower"

        details = {
            "dk_line": dk_line,
            "median_other_books": round(median_other, 1),
            "gap": round(gap, 1),
            "num_other_books": len(other_lines),
            "books_disagreeing": books_disagreeing,
            "direction": direction,
            "other_lines": [round(ol, 1) for ol in sorted(other_lines)],
        }

        return StaleLineSignal("cross_book", score, reason, details)

    # ------------------------------------------------------------------
    # Signal 4: Correlated Market Movement
    # ------------------------------------------------------------------
    def _signal_correlated_market(
        self,
        player: str,
        prop_type: str,
        line: float,
        game_context: Optional[dict] = None,
    ) -> StaleLineSignal:
        """
        Check if game total or spread has moved but individual props haven't.
        If game total went from 215 to 225, scoring props should go up.
        """
        if game_context is None or line <= 0:
            return StaleLineSignal("correlated_market", 0, "", {})

        # Get game total and spread info
        total = None
        opening_total = None
        spread = None

        # Try various attribute/key names
        if hasattr(game_context, 'total'):
            total = getattr(game_context, 'total', None)
        elif isinstance(game_context, dict):
            total = game_context.get('total', None)

        if hasattr(game_context, 'opening_total'):
            opening_total = getattr(game_context, 'opening_total', None)
        elif isinstance(game_context, dict):
            opening_total = game_context.get('opening_total', None)

        if hasattr(game_context, 'spread'):
            spread = getattr(game_context, 'spread', None)
        elif isinstance(game_context, dict):
            spread = game_context.get('spread', None)

        # Without opening_total we can't detect movement
        if total is None or opening_total is None:
            return StaleLineSignal("correlated_market", 0, "", {})

        total_movement = total - opening_total

        # Only scoring-related props should track with game total
        scoring_props = {"points", "threes", "pts_rebs_asts", "pts_rebs", "pts_asts"}

        if abs(total_movement) < self.GAME_TOTAL_MOVE:
            return StaleLineSignal("correlated_market", 0, "", {})

        if prop_type not in scoring_props:
            # Reduced signal for non-scoring props
            score = min(abs(total_movement) / self.GAME_TOTAL_MOVE * 10, 30.0)
            reason = f"Game total moved {total_movement:+.1f} pts" if score > 10 else ""
        else:
            # Scoring props should move with game total
            expected_prop_shift = total_movement * 0.05  # rough: each player ~5% of total
            score = min(abs(total_movement) / self.GAME_TOTAL_MOVE * 25, 80.0)
            direction = "up" if total_movement > 0 else "down"
            reason = f"Game total moved {total_movement:+.1f} but {prop_type} line unchanged at {line}"

        details = {
            "current_total": total,
            "opening_total": opening_total,
            "total_movement": round(total_movement, 1),
            "prop_type": prop_type,
            "is_scoring_prop": prop_type in scoring_props,
        }

        return StaleLineSignal("correlated_market", score, reason, details)

    # ------------------------------------------------------------------
    # Signal 5: Injury Cascade
    # ------------------------------------------------------------------
    def _signal_injury_cascade(
        self,
        player: str,
        prop_type: str,
        line: float,
        injuries: Optional[List[Dict]] = None,
        game_context: Optional[dict] = None,
    ) -> StaleLineSignal:
        """
        Check if a teammate's injury should affect this player's prop but the line
        hasn't moved. E.g., if a team's #1 scorer is OUT, the #2 scorer should see
        higher scoring props.
        """
        if not injuries or line <= 0:
            return StaleLineSignal("injury_cascade", 0, "", {})

        # Identify newly OUT players from the injury report
        out_players = []
        for inj in injuries:
            status = ""
            name = ""
            if isinstance(inj, dict):
                status = inj.get("status", "").lower()
                name = inj.get("name", "") or inj.get("player", "")
                if isinstance(name, dict):
                    name = "{} {}".format(name.get("first_name", ""), name.get("last_name", ""))
            if status == "out" and name and name.strip() != player:
                out_players.append(name.strip())

        if not out_players:
            return StaleLineSignal("injury_cascade", 0, "", {})

        # Estimate cascade boost using available player history
        # Look at how much production the OUT players averaged
        cascade_boost_pct = 0.0
        relevant_injured = []

        if self.feature_builder:
            for injured_player in out_players:
                history = self.feature_builder.player_history.get(injured_player, [])
                if not history:
                    continue
                recent = list(reversed(history))[:10]
                if not recent:
                    continue
                injured_avg = np.mean([self._get_stat_value(g, prop_type) for g in recent])
                # If the injured player had significant production in this stat,
                # teammates should absorb some of it
                if injured_avg > 5.0:  # meaningful producer
                    # Rough cascade: ~20-30% of production redistributed to top options
                    boost = injured_avg * 0.15  # 15% redistribution estimate
                    boost_pct = boost / max(line, 1.0)
                    if boost_pct > 0.03:
                        cascade_boost_pct += boost_pct
                        relevant_injured.append({
                            "player": injured_player,
                            "avg_production": round(injured_avg, 1),
                            "estimated_boost": round(boost, 1),
                        })

        if cascade_boost_pct < self.INJURY_CASCADE_MIN:
            if out_players:
                # Still note the injury, just lower score
                return StaleLineSignal(
                    "injury_cascade", 10.0,
                    f"{len(out_players)} teammate(s) OUT but limited cascade expected",
                    {"out_players": out_players},
                )
            return StaleLineSignal("injury_cascade", 0, "", {})

        score = min(cascade_boost_pct / 0.15 * 60.0, 90.0)

        injured_names = [ri["player"] for ri in relevant_injured]
        boost_total = sum(ri["estimated_boost"] for ri in relevant_injured)
        reason = (
            f"Teammate{'s' if len(injured_names) > 1 else ''} "
            f"{', '.join(injured_names[:2])} OUT, "
            f"expected +{boost_total:.1f} {prop_type} boost, line unmoved at {line}"
        )

        details = {
            "out_players": out_players,
            "relevant_injured": relevant_injured,
            "cascade_boost_pct": round(cascade_boost_pct * 100, 1),
            "estimated_boost": round(boost_total, 1),
        }

        return StaleLineSignal("injury_cascade", score, reason, details)

    # ------------------------------------------------------------------
    # Main detection method
    # ------------------------------------------------------------------
    def detect_stale_lines(
        self,
        props: List[Dict],
        game_date: str,
        game_context: Optional[dict] = None,
        injuries: Optional[List[Dict]] = None,
        predictions: Optional[Dict[str, Dict]] = None,
    ) -> List[StaleLineResult]:
        """
        Run all staleness signals on a list of props.

        Args:
            props: List of prop dicts with player_name, prop_type, line, all_books, etc.
            game_date: ISO date string
            game_context: Game context with total, spread, opening_total, etc.
            injuries: List of injury dicts with name, status
            predictions: Optional dict of {player|prop_type: prediction_dict} from v2 model

        Returns:
            List of StaleLineResult, sorted by staleness_score descending
        """
        results = []
        predictions = predictions or {}

        for prop in props:
            player = prop.get("player_name", prop.get("player", ""))
            prop_type = prop.get("prop_type", "")
            line = prop.get("line", 0)
            all_books = prop.get("all_books", {})

            if not player or not prop_type or line <= 0:
                continue

            # Get prediction for this prop if available
            pred_key = f"{player}|{prop_type}"
            prediction = predictions.get(pred_key, {})

            # Compute all signals
            sig_recent = self._signal_recent_performance(player, prop_type, line, game_date)
            sig_model = self._signal_model_divergence(
                player, prop_type, line, game_date, game_context, prediction
            )
            sig_cross = self._signal_cross_book(player, prop_type, line, all_books)
            sig_market = self._signal_correlated_market(player, prop_type, line, game_context)
            sig_injury = self._signal_injury_cascade(player, prop_type, line, injuries, game_context)

            signals = [sig_recent, sig_model, sig_cross, sig_market, sig_injury]

            # Weighted composite score
            staleness_score = (
                sig_recent.score * self.WEIGHTS["recent_performance"]
                + sig_model.score * self.WEIGHTS["model_divergence"]
                + sig_cross.score * self.WEIGHTS["cross_book"]
                + sig_market.score * self.WEIGHTS["correlated_market"]
                + sig_injury.score * self.WEIGHTS["injury_cascade"]
            )

            # Bonus: if multiple signals fire, boost the score
            active_signals = sum(1 for s in signals if s.score > 30)
            if active_signals >= 3:
                staleness_score = min(staleness_score * 1.15, 100.0)
            elif active_signals >= 2:
                staleness_score = min(staleness_score * 1.08, 100.0)

            # Filter to non-empty signals, sort by score desc
            active = [s for s in signals if s.score > 0]
            active.sort(key=lambda s: s.score, reverse=True)

            is_stale = staleness_score >= self.STALE_THRESHOLD

            results.append(StaleLineResult(
                player=player,
                prop_type=prop_type,
                line=line,
                staleness_score=staleness_score,
                signals=active,
                is_stale=is_stale,
            ))

        # Sort by staleness score descending
        results.sort(key=lambda r: r.staleness_score, reverse=True)
        return results

    def detect_for_game(
        self,
        props: List[Dict],
        game_date: str,
        game_context: Optional[dict] = None,
        injuries: Optional[List[Dict]] = None,
        predictions: Optional[Dict[str, Dict]] = None,
    ) -> Dict:
        """
        Convenience method: detect stale lines and return a summary dict.

        Returns:
            {
                "stale_count": int,
                "total_props": int,
                "stale_lines": [StaleLineResult.to_dict(), ...],
                "all_results": [StaleLineResult.to_dict(), ...],
            }
        """
        results = self.detect_stale_lines(
            props, game_date, game_context, injuries, predictions
        )

        stale = [r for r in results if r.is_stale]

        return {
            "stale_count": len(stale),
            "total_props": len(results),
            "stale_lines": [r.to_dict() for r in stale],
            "all_results": [r.to_dict() for r in results],
        }


def backtest_stale_lines(legs_files=None, game_data_files=None):
    """
    Historical validation: retroactively detect stale lines from backtest data
    and measure hit rates for stale vs non-stale lines.

    Uses the 3-season backtest data (legs + game data) to:
    1. Build player history from game box scores
    2. For each leg, compute staleness signals using only prior data
    3. Compare hit rates: stale lines vs normal lines

    Returns dict with full backtest results.
    """
    import json
    from v2.features import FeatureBuilder, LEGS_FILES, GAME_DATA_FILES

    if legs_files is None:
        legs_files = LEGS_FILES
    if game_data_files is None:
        game_data_files = GAME_DATA_FILES

    # Load feature builder with game data
    fb = FeatureBuilder()
    fb.load_games()
    fb.build_player_history()

    # Load all legs
    all_legs = []
    for path in legs_files:
        if not path.exists():
            logger.warning(f"Missing legs file: {path}")
            continue
        with open(path) as f:
            legs = json.load(f)
        logger.info(f"Loaded {len(legs)} legs from {path.name}")
        all_legs.extend(legs)

    logger.info(f"Total legs for backtest: {len(all_legs)}")

    # Create detector (no predictor needed for backtest -- only recent perf + cross-book signals)
    detector = StaleLineDetector(feature_builder=fb)

    # Process each leg
    stale_hits = 0
    stale_total = 0
    normal_hits = 0
    normal_total = 0

    stale_by_signal = defaultdict(lambda: {"hits": 0, "total": 0})
    stale_by_prop = defaultdict(lambda: {"stale_hits": 0, "stale_total": 0, "normal_hits": 0, "normal_total": 0})
    stale_by_score_bucket = defaultdict(lambda: {"hits": 0, "total": 0})

    # Direction-aware tracking
    stale_over_hits = 0
    stale_over_total = 0
    stale_under_hits = 0
    stale_under_total = 0

    sample_stale_legs = []  # Store examples for inspection

    for i, leg in enumerate(all_legs):
        player = leg.get("player", "")
        prop_type = leg.get("prop_type", "")
        line = leg.get("line", 0)
        actual = leg.get("actual")
        hit = leg.get("hit", False)
        date = leg.get("date", "")
        pick = leg.get("pick", "").upper()

        if not player or not prop_type or line <= 0 or actual is None:
            continue

        # Compute staleness using only data before this game date
        prop_data = [{
            "player_name": player,
            "prop_type": prop_type,
            "line": line,
            "all_books": {},  # No cross-book data in historical backtest
        }]

        results = detector.detect_stale_lines(prop_data, date)
        if not results:
            continue

        result = results[0]
        score = result.staleness_score
        is_stale = result.is_stale

        # Bucket by score
        bucket = int(score // 10) * 10
        stale_by_score_bucket[bucket]["total"] += 1
        if hit:
            stale_by_score_bucket[bucket]["hits"] += 1

        if is_stale:
            stale_total += 1
            if hit:
                stale_hits += 1

            # Track which signal was dominant
            if result.signals:
                top_signal = result.signals[0].name
                stale_by_signal[top_signal]["total"] += 1
                if hit:
                    stale_by_signal[top_signal]["hits"] += 1

            # Direction tracking
            # Determine if the stale direction agrees with the pick
            top_details = result.signals[0].details if result.signals else {}
            stale_direction = top_details.get("direction", "")

            if pick == "OVER":
                stale_over_total += 1
                if hit:
                    stale_over_hits += 1
            elif pick == "UNDER":
                stale_under_total += 1
                if hit:
                    stale_under_hits += 1

            stale_by_prop[prop_type]["stale_total"] += 1
            if hit:
                stale_by_prop[prop_type]["stale_hits"] += 1

            # Keep sample stale legs for inspection
            if len(sample_stale_legs) < 50:
                sample_stale_legs.append({
                    "player": player,
                    "prop_type": prop_type,
                    "line": line,
                    "actual": actual,
                    "hit": hit,
                    "pick": pick,
                    "date": date,
                    "staleness_score": round(score, 1),
                    "top_signal": result.signals[0].name if result.signals else "",
                    "top_reason": result.signals[0].reason if result.signals else "",
                })
        else:
            normal_total += 1
            if hit:
                normal_hits += 1
            stale_by_prop[prop_type]["normal_total"] += 1
            if hit:
                stale_by_prop[prop_type]["normal_hits"] += 1

        if (i + 1) % 10000 == 0:
            logger.info(f"Processed {i+1}/{len(all_legs)} legs...")

    # Compute summary stats
    stale_hit_rate = stale_hits / max(stale_total, 1)
    normal_hit_rate = normal_hits / max(normal_total, 1)

    signal_breakdown = {}
    for sig_name, counts in stale_by_signal.items():
        signal_breakdown[sig_name] = {
            "total": counts["total"],
            "hits": counts["hits"],
            "hit_rate": round(counts["hits"] / max(counts["total"], 1), 4),
        }

    prop_breakdown = {}
    for prop_type, counts in stale_by_prop.items():
        prop_breakdown[prop_type] = {
            "stale_total": counts["stale_total"],
            "stale_hits": counts["stale_hits"],
            "stale_hit_rate": round(counts["stale_hits"] / max(counts["stale_total"], 1), 4),
            "normal_total": counts["normal_total"],
            "normal_hits": counts["normal_hits"],
            "normal_hit_rate": round(counts["normal_hits"] / max(counts["normal_total"], 1), 4),
        }

    score_buckets = {}
    for bucket in sorted(stale_by_score_bucket.keys()):
        counts = stale_by_score_bucket[bucket]
        score_buckets[f"{bucket}-{bucket+9}"] = {
            "total": counts["total"],
            "hits": counts["hits"],
            "hit_rate": round(counts["hits"] / max(counts["total"], 1), 4),
        }

    results = {
        "summary": {
            "total_legs": len(all_legs),
            "stale_total": stale_total,
            "stale_hits": stale_hits,
            "stale_hit_rate": round(stale_hit_rate, 4),
            "normal_total": normal_total,
            "normal_hits": normal_hits,
            "normal_hit_rate": round(normal_hit_rate, 4),
            "hit_rate_advantage": round(stale_hit_rate - normal_hit_rate, 4),
            "stale_over_hit_rate": round(stale_over_hits / max(stale_over_total, 1), 4),
            "stale_under_hit_rate": round(stale_under_hits / max(stale_under_total, 1), 4),
            "stale_over_total": stale_over_total,
            "stale_under_total": stale_under_total,
        },
        "by_signal": signal_breakdown,
        "by_prop": prop_breakdown,
        "by_score_bucket": score_buckets,
        "sample_stale_legs": sample_stale_legs,
    }

    # Print summary
    print("\n" + "=" * 70)
    print("STALE LINE BACKTEST RESULTS")
    print("=" * 70)
    print(f"Total legs: {len(all_legs):,}")
    print(f"Stale lines (score >= {StaleLineDetector.STALE_THRESHOLD}): {stale_total:,} ({stale_total/max(len(all_legs),1)*100:.1f}%)")
    print(f"Normal lines: {normal_total:,}")
    print()
    print(f"Stale line hit rate:  {stale_hit_rate:.1%} ({stale_hits}/{stale_total})")
    print(f"Normal line hit rate: {normal_hit_rate:.1%} ({normal_hits}/{normal_total})")
    print(f"Advantage:            {(stale_hit_rate - normal_hit_rate)*100:+.1f}pp")
    print()
    if stale_over_total:
        print(f"Stale OVER hit rate:  {stale_over_hits/max(stale_over_total,1):.1%} ({stale_over_hits}/{stale_over_total})")
    if stale_under_total:
        print(f"Stale UNDER hit rate: {stale_under_hits/max(stale_under_total,1):.1%} ({stale_under_hits}/{stale_under_total})")
    print()
    print("By staleness score bucket:")
    for bucket_name, data in score_buckets.items():
        print(f"  {bucket_name}: {data['hit_rate']:.1%} ({data['hits']}/{data['total']})")
    print()
    print("By signal type:")
    for sig, data in sorted(signal_breakdown.items(), key=lambda x: -x[1]["total"]):
        print(f"  {sig}: {data['hit_rate']:.1%} ({data['hits']}/{data['total']})")
    print()
    print("By prop type:")
    for pt, data in sorted(prop_breakdown.items(), key=lambda x: -x[1]["stale_total"]):
        print(f"  {pt}: stale {data['stale_hit_rate']:.1%} ({data['stale_hits']}/{data['stale_total']}) vs normal {data['normal_hit_rate']:.1%}")
    print()

    if sample_stale_legs:
        print("Sample stale lines:")
        for s in sample_stale_legs[:10]:
            emoji = "HIT" if s["hit"] else "MISS"
            print(f"  {s['date']} {s['player']} {s['prop_type']} L={s['line']} A={s['actual']} [{emoji}] score={s['staleness_score']} [{s['top_signal']}] {s['top_reason']}")

    return results


if __name__ == "__main__":
    import sys
    import json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("Running stale line backtest...")
    results = backtest_stale_lines()

    # Save results
    output_path = BASE_DIR / "outputs" / "stale_line_backtest.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
