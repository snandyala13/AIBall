"""
line_edge.py - Line Movement Detection & Edge Signal System

Detects betting edge from line movement patterns, cross-book discrepancies,
and closing line value analysis. This is often MORE valuable than projection
accuracy for finding +EV bets because:

1. Opening line value: when a line moves from 24.5 to 25.5, the opener was wrong
2. Sharp vs public money: line moves AGAINST the public side = sharp action
3. Cross-book discrepancies: one book's line differs from consensus
4. Closing Line Value (CLV): proven predictor of long-term profitability

Usage:
    from line_edge import LineEdgeDetector

    detector = LineEdgeDetector(propline_client=propline)
    signals = detector.get_edge_signals(game_props)
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Bookmakers to query for cross-book analysis via PropLine
CROSS_BOOK_BOOKMAKERS = [
    "draftkings",
    "fanduel",
    "betmgm",
    "caesars",
    "pointsbet",
]


class LineEdgeDetector:
    """
    Detects betting edge from line movement patterns, cross-book discrepancies,
    and closing line value analysis.

    This is often MORE valuable than projection accuracy for finding +EV bets.
    """

    # Thresholds for sharp movement detection
    SIGNIFICANT_LINE_MOVE = 0.5   # 0.5+ points = noteworthy
    LARGE_LINE_MOVE = 1.0         # 1.0+ points = strong signal
    SIGNIFICANT_ODDS_MOVE = 15    # 15+ cents of juice shift

    # Cross-book edge thresholds
    CROSS_BOOK_EDGE_MIN = 0.5     # 0.5+ point discrepancy = potential edge
    CROSS_BOOK_EDGE_STRONG = 1.0  # 1.0+ points = strong cross-book edge

    # Confidence multipliers for bet scoring
    SHARP_AGREES_BOOST = 1.15     # Model + sharp agree
    SHARP_DISAGREES_PENALTY = 0.85  # Sharp disagrees with model
    CROSS_BOOK_EDGE_BOOST = 1.10  # Cross-book discrepancy favors the bet
    COMBINED_SIGNAL_BOOST = 1.20  # Both sharp + cross-book agree

    def __init__(self, propline_client=None, line_tracker=None):
        # type: (Optional[object], Optional[object]) -> None
        """
        Initialize the LineEdgeDetector.

        Args:
            propline_client: PropLineClient instance for multi-book data.
            line_tracker: LineTracker instance for historical line data.
        """
        self.propline = propline_client
        self.line_tracker = line_tracker

        # Try to import LineTracker if not provided
        if self.line_tracker is None:
            try:
                from line_tracker import get_line_tracker
                self.line_tracker = get_line_tracker()
            except (ImportError, Exception) as e:
                logger.debug("LineTracker not available: %s", e)

        # Cache for multi-book data (refreshed per analysis run)
        self._multi_book_cache = {}  # type: Dict[str, List[Dict]]

    # ------------------------------------------------------------------
    # Sharp Movement Detection
    # ------------------------------------------------------------------

    def detect_sharp_movement(self, player_name, prop_type, current_line,
                              opening_line=None, current_over_odds=-110,
                              opening_over_odds=-110, books_data=None):
        # type: (str, str, float, Optional[float], int, int, Optional[List[Dict]]) -> Dict
        """
        Detect if sharp money has moved this line.

        Sharp indicators:
        - Line moved 1+ points in one direction (significant move)
        - Line moved AGAINST the public side (reverse line movement)
        - Juice shifted significantly (odds changed without line change)
        - One book has a significantly different line than others (originator)

        Returns:
            Dict with movement, direction, confidence, edge_signal, reasoning.
        """
        result = {
            'movement': 0.0,
            'direction': 'neutral',
            'confidence': 0.0,
            'edge_signal': None,
            'reasoning': 'No line movement data available',
            'opening_line': opening_line,
            'current_line': current_line,
        }

        # Try to get movement from LineTracker if no opening line provided
        if opening_line is None and self.line_tracker is not None:
            movement_data = self._get_tracker_movement(player_name, prop_type)
            if movement_data:
                opening_line = movement_data.get('opening_line')
                opening_over_odds = movement_data.get('opening_over_odds', -110)
                result['opening_line'] = opening_line

        if opening_line is None:
            # No historical data -- check cross-book for originator signals
            if books_data:
                originator = self._detect_originator_book(books_data)
                if originator:
                    result['direction'] = originator['direction']
                    result['confidence'] = originator['confidence']
                    result['edge_signal'] = originator['edge_signal']
                    result['reasoning'] = originator['reasoning']
            return result

        # Calculate line movement
        movement = current_line - opening_line
        result['movement'] = round(movement, 2)

        # Calculate odds movement (juice shift)
        odds_shift = current_over_odds - opening_over_odds

        # Determine direction and confidence
        abs_movement = abs(movement)
        confidence = 0.0

        if abs_movement >= self.LARGE_LINE_MOVE:
            confidence = 0.85
        elif abs_movement >= self.SIGNIFICANT_LINE_MOVE:
            confidence = 0.65
        elif abs(odds_shift) >= self.SIGNIFICANT_ODDS_MOVE:
            # Juice moved without line change -- still a signal
            confidence = 0.50
        else:
            result['direction'] = 'neutral'
            result['confidence'] = 0.0
            result['reasoning'] = 'Minimal line movement ({:+.1f})'.format(movement)
            return result

        # Boost confidence if odds movement confirms line direction
        if movement > 0 and odds_shift < -5:
            # Line went up AND over odds got worse (more juice on over)
            confidence = min(confidence + 0.10, 1.0)
        elif movement < 0 and odds_shift > 5:
            # Line went down AND under odds got worse
            confidence = min(confidence + 0.10, 1.0)

        # Determine sharp direction
        if movement > 0:
            direction = 'sharp_over'
            edge_signal = 'over'
            reasoning = 'Line moved from {:.1f} to {:.1f} ({:+.1f}), suggesting sharp OVER action'.format(
                opening_line, current_line, movement)
        else:
            direction = 'sharp_under'
            edge_signal = 'under'
            reasoning = 'Line moved from {:.1f} to {:.1f} ({:+.1f}), suggesting sharp UNDER action'.format(
                opening_line, current_line, movement)

        # Add odds context
        if abs(odds_shift) >= self.SIGNIFICANT_ODDS_MOVE:
            reasoning += ' (odds shifted {:+d} cents)'.format(odds_shift)

        result.update({
            'direction': direction,
            'confidence': round(confidence, 2),
            'edge_signal': edge_signal,
            'reasoning': reasoning,
        })

        return result

    # ------------------------------------------------------------------
    # Cross-Book Edge Detection
    # ------------------------------------------------------------------

    def find_cross_book_edges(self, player_name, prop_type):
        # type: (str, str) -> Dict
        """
        Find discrepancies between books where one book's line is significantly
        different from the consensus.

        If DraftKings has 24.5 but FanDuel/BetMGM have 25.5, there's potential
        value on the DK over.

        Uses PropLine's multi-book data.

        Returns:
            Dict with best_over_book, best_over_line, consensus_line, edge, etc.
        """
        result = {
            'best_over_book': None,
            'best_over_line': None,
            'best_under_book': None,
            'best_under_line': None,
            'consensus_line': None,
            'edge': 0.0,
            'has_edge': False,
            'books_count': 0,
            'reasoning': 'No cross-book data available',
        }

        if self.propline is None:
            return result

        # Get multi-book data
        books_data = self._get_multi_book_lines(player_name, prop_type)
        if not books_data or len(books_data) < 2:
            result['books_count'] = len(books_data) if books_data else 0
            result['reasoning'] = 'Insufficient book data ({} books)'.format(result['books_count'])
            return result

        # Calculate consensus (median line across books)
        lines = [b['line'] for b in books_data]
        lines.sort()
        n = len(lines)
        if n % 2 == 0:
            consensus = (lines[n // 2 - 1] + lines[n // 2]) / 2.0
        else:
            consensus = lines[n // 2]

        # Find best over line (lowest -- easier to go over)
        best_over = min(books_data, key=lambda b: b['line'])
        # Find best under line (highest -- easier to go under)
        best_under = max(books_data, key=lambda b: b['line'])

        over_edge = consensus - best_over['line']
        under_edge = best_under['line'] - consensus
        max_edge = max(over_edge, under_edge)

        has_edge = max_edge >= self.CROSS_BOOK_EDGE_MIN

        if has_edge:
            if over_edge >= under_edge:
                reasoning = '{} has {:.1f} at {:.1f} vs consensus {:.1f} ({:+.1f} pt edge on OVER)'.format(
                    best_over['book'].title(), best_over['line'], best_over['line'],
                    consensus, over_edge)
            else:
                reasoning = '{} has {:.1f} vs consensus {:.1f} ({:+.1f} pt edge on UNDER)'.format(
                    best_under['book'].title(), best_under['line'],
                    consensus, under_edge)
        else:
            reasoning = 'Lines tight across {} books (consensus {:.1f})'.format(n, consensus)

        result.update({
            'best_over_book': best_over['book'],
            'best_over_line': best_over['line'],
            'best_under_book': best_under['book'],
            'best_under_line': best_under['line'],
            'consensus_line': round(consensus, 2),
            'edge': round(max_edge, 2),
            'over_edge': round(over_edge, 2),
            'under_edge': round(under_edge, 2),
            'has_edge': has_edge,
            'books_count': n,
            'all_books': books_data,
            'reasoning': reasoning,
        })

        return result

    # ------------------------------------------------------------------
    # Closing Line Value (CLV) from History
    # ------------------------------------------------------------------

    def calculate_clv_from_history(self, predictions_file=None):
        # type: (Optional[str]) -> Dict
        """
        Calculate Closing Line Value for past predictions.
        CLV is the gold standard metric -- if you consistently beat the closing line,
        you have real edge regardless of short-term results.

        For each past prediction:
        1. What line did we bet at?
        2. What did the line close at?
        3. CLV = closing_line - our_line (positive = we got a better number)

        Returns:
            Dict with avg_clv, clv_positive_pct, by_prop, total_predictions, etc.
        """
        result = {
            'avg_clv': 0.0,
            'clv_positive_pct': 0.0,
            'total_predictions': 0,
            'predictions_with_clv': 0,
            'by_prop': {},
            'by_side': {'over': 0.0, 'under': 0.0},
            'available': False,
        }

        if predictions_file is None:
            season = datetime.now().year if datetime.now().month >= 10 else datetime.now().year - 1
            predictions_file = 'outputs/predictions_{}-{}.jsonl'.format(season, str(season + 1)[-2:])

        if not os.path.exists(predictions_file):
            result['reasoning'] = 'No predictions file found at {}'.format(predictions_file)
            return result

        if self.line_tracker is None:
            result['reasoning'] = 'No line tracker available for CLV calculation'
            return result

        # Read predictions and match with closing lines
        predictions = []
        try:
            with open(predictions_file, 'r') as f:
                for raw_line in f:
                    raw_line = raw_line.strip()
                    if raw_line:
                        try:
                            predictions.append(json.loads(raw_line))
                        except json.JSONDecodeError:
                            continue
        except IOError:
            result['reasoning'] = 'Could not read predictions file'
            return result

        if not predictions:
            result['reasoning'] = 'No predictions found in file'
            return result

        result['total_predictions'] = len(predictions)

        # Calculate CLV for each prediction
        clv_values = []
        prop_clv = {}  # type: Dict[str, List[float]]
        side_clv = {'over': [], 'under': []}  # type: Dict[str, List[float]]

        for pred in predictions:
            player = pred.get('player', '')
            prop_type = pred.get('prop_type', '')
            bet_line = pred.get('line', 0)
            edge_over = pred.get('edge_over', 0)
            edge_under = pred.get('edge_under', 0)

            if not player or not prop_type or bet_line == 0:
                continue

            # Determine which side we would have bet
            side = 'over' if edge_over > edge_under else 'under'

            # Look up closing line from tracker
            movement_data = self._get_tracker_movement(player, prop_type)
            if movement_data is None:
                continue

            closing_line = movement_data.get('current_line', bet_line)

            # Calculate CLV
            if side == 'over':
                clv = closing_line - bet_line  # positive = we got a lower (better) line
            else:
                clv = bet_line - closing_line  # positive = we got a higher (better) line

            clv_values.append(clv)

            if prop_type not in prop_clv:
                prop_clv[prop_type] = []
            prop_clv[prop_type].append(clv)
            side_clv[side].append(clv)

        if not clv_values:
            result['reasoning'] = 'No CLV data available (no matching closing lines found)'
            return result

        # Aggregate results
        avg_clv = sum(clv_values) / len(clv_values)
        positive_count = sum(1 for v in clv_values if v > 0)
        clv_positive_pct = (positive_count / len(clv_values)) * 100

        by_prop = {}
        for prop, values in prop_clv.items():
            by_prop[prop] = round(sum(values) / len(values), 3) if values else 0.0

        by_side = {}
        for side_key, values in side_clv.items():
            by_side[side_key] = round(sum(values) / len(values), 3) if values else 0.0

        result.update({
            'avg_clv': round(avg_clv, 3),
            'clv_positive_pct': round(clv_positive_pct, 1),
            'predictions_with_clv': len(clv_values),
            'by_prop': by_prop,
            'by_side': by_side,
            'available': True,
        })

        return result

    # ------------------------------------------------------------------
    # Combined Edge Signals for a Game
    # ------------------------------------------------------------------

    def get_edge_signals(self, game_props):
        # type: (List[Dict]) -> Dict[str, Dict]
        """
        For all props in a game, return edge signals from line analysis.

        This runs AFTER the projection model and ADDS to the edge calculation.

        For each prop:
        - Check for sharp movement (from LineTracker history)
        - Check for cross-book discrepancies (from PropLine multi-book)
        - Combine signals for a final edge assessment

        Args:
            game_props: List of prop dicts with player_name, prop_type, line, etc.

        Returns:
            Dict keyed by "player_name|prop_type" with signal details.
        """
        signals = {}

        for prop in game_props:
            player = prop.get('player_name', '')
            prop_type = prop.get('prop_type', '')
            current_line = prop.get('line', 0)
            over_odds = prop.get('over_odds', -110)

            if not player or not prop_type:
                continue

            key = '{}|{}'.format(player, prop_type)

            # Get sharp movement signal
            sharp = self.detect_sharp_movement(
                player_name=player,
                prop_type=prop_type,
                current_line=current_line,
                current_over_odds=over_odds,
            )

            # Get cross-book edge signal
            cross_book = self.find_cross_book_edges(player, prop_type)

            # Combine signals
            combined = self._combine_signals(sharp, cross_book)

            signals[key] = {
                'player_name': player,
                'prop_type': prop_type,
                'line': current_line,
                'sharp_signal': sharp['edge_signal'],
                'sharp_direction': sharp['direction'],
                'sharp_confidence': sharp['confidence'],
                'sharp_movement': sharp['movement'],
                'sharp_reasoning': sharp['reasoning'],
                'cross_book_edge': cross_book['edge'],
                'cross_book_has_edge': cross_book['has_edge'],
                'cross_book_consensus': cross_book['consensus_line'],
                'cross_book_reasoning': cross_book['reasoning'],
                'combined_direction': combined['direction'],
                'combined_confidence': combined['confidence'],
                'combined_reasoning': combined['reasoning'],
            }

        return signals

    # ------------------------------------------------------------------
    # Bet Scoring Enhancement
    # ------------------------------------------------------------------

    def score_bet_with_line_signals(self, bet_confidence, bet_side, line_signals):
        # type: (float, str, Dict) -> Tuple[float, str]
        """
        Enhance a bet's confidence by incorporating line edge signals.

        When model edge AND sharp money agree: BOOST confidence
        When model edge but line moving against: REDUCE confidence
        When cross-book edge favors the bet: BOOST confidence
        When sharp signal but model disagrees: MONITOR only

        Args:
            bet_confidence: Original confidence score from the model.
            bet_side: "over" or "under" -- our bet side.
            line_signals: Signal dict from get_edge_signals for this prop.

        Returns:
            Tuple of (adjusted_confidence, signal_label).
            signal_label is one of: 'sharp_agree', 'sharp_disagree',
            'cross_book_edge', 'combined_agree', 'neutral'.
        """
        sharp_signal = line_signals.get('sharp_signal')
        sharp_confidence = line_signals.get('sharp_confidence', 0)
        cross_book_has_edge = line_signals.get('cross_book_has_edge', False)
        cross_book_edge = line_signals.get('cross_book_edge', 0)

        adjusted = bet_confidence
        label = 'neutral'

        # Check sharp agreement
        sharp_agrees = sharp_signal == bet_side
        sharp_disagrees = sharp_signal is not None and sharp_signal != bet_side

        # Determine cross-book direction
        consensus = line_signals.get('cross_book_consensus')
        line = line_signals.get('line', 0)
        cross_book_favors = False
        if consensus and line:
            if bet_side == 'over' and line < consensus:
                # Our line is lower than consensus -- over is better here
                cross_book_favors = True
            elif bet_side == 'under' and line > consensus:
                # Our line is higher than consensus -- under is better here
                cross_book_favors = True

        # Apply adjustments
        if sharp_agrees and cross_book_favors:
            # Both sharp money and cross-book agree with our side
            adjusted *= self.COMBINED_SIGNAL_BOOST
            label = 'combined_agree'
        elif sharp_agrees:
            # Sharp money agrees with our model
            adjusted *= self.SHARP_AGREES_BOOST
            label = 'sharp_agree'
        elif sharp_disagrees and sharp_confidence >= 0.6:
            # Strong sharp signal disagrees -- reduce confidence
            adjusted *= self.SHARP_DISAGREES_PENALTY
            label = 'sharp_disagree'
        elif cross_book_favors and cross_book_edge >= self.CROSS_BOOK_EDGE_MIN:
            # Cross-book edge favors our bet
            adjusted *= self.CROSS_BOOK_EDGE_BOOST
            label = 'cross_book_edge'

        return (round(adjusted, 4), label)

    # ------------------------------------------------------------------
    # Record lines from PropLine for future CLV tracking
    # ------------------------------------------------------------------

    def record_current_lines(self, game_id, game_props):
        # type: (int, List[Dict]) -> int
        """
        Record current lines from a game into the LineTracker for future CLV.

        This should be called each time we fetch props so we track how
        lines move over time.

        Returns number of lines recorded.
        """
        if self.line_tracker is None:
            return 0

        count = 0
        game_date = datetime.now().strftime('%Y-%m-%d')

        for prop in game_props:
            player = prop.get('player_name', '')
            prop_type = prop.get('prop_type', '')
            line = prop.get('line', 0)
            over_odds = prop.get('over_odds', -110)
            under_odds = prop.get('under_odds', -110)

            if not player or not prop_type or line == 0:
                continue

            try:
                self.line_tracker.record_line(
                    game_id=game_id,
                    player_name=player,
                    prop_type=prop_type,
                    line=line,
                    over_odds=over_odds,
                    under_odds=under_odds,
                    game_date=game_date,
                )
                count += 1
            except Exception as e:
                logger.debug("Failed to record line for %s %s: %s", player, prop_type, e)

        return count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_tracker_movement(self, player_name, prop_type):
        # type: (str, str) -> Optional[Dict]
        """Look up line movement from the LineTracker (searches recent games)."""
        if self.line_tracker is None:
            return None

        # LineTracker stores by game_id -- we need to search broadly
        # Query the line_movements table directly for this player/prop
        try:
            import sqlite3
            conn = sqlite3.connect(self.line_tracker.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT opening_line, current_line, opening_over_odds, current_over_odds,
                       opening_under_odds, current_under_odds, first_seen, last_updated
                FROM line_movements
                WHERE player_name = ? AND prop_type = ?
                ORDER BY last_updated DESC
                LIMIT 1
            """, (player_name, prop_type))
            row = cursor.fetchone()
            conn.close()

            if row:
                return {
                    'opening_line': row[0],
                    'current_line': row[1],
                    'opening_over_odds': row[2] or -110,
                    'current_over_odds': row[3] or -110,
                    'opening_under_odds': row[4] or -110,
                    'current_under_odds': row[5] or -110,
                    'first_seen': row[6],
                    'last_updated': row[7],
                    'movement': row[1] - row[0],
                }
        except Exception as e:
            logger.debug("LineTracker query failed: %s", e)

        return None

    def _get_multi_book_lines(self, player_name, prop_type):
        # type: (str, str) -> List[Dict]
        """
        Get lines from multiple bookmakers via PropLine.

        Fetches from each bookmaker and collects the line for this
        player/prop combination.
        """
        if self.propline is None:
            return []

        cache_key = '{}|{}'.format(player_name, prop_type)
        if cache_key in self._multi_book_cache:
            return self._multi_book_cache[cache_key]

        books_data = []

        for bookmaker in CROSS_BOOK_BOOKMAKERS:
            try:
                all_props = self.propline.get_all_props(bookmaker=bookmaker)
                for event_id, props in all_props.items():
                    for prop in props:
                        if (prop.get('player_name', '').lower() == player_name.lower() and
                                prop.get('prop_type', '') == prop_type):
                            books_data.append({
                                'book': bookmaker,
                                'line': prop['line'],
                                'over_odds': prop.get('over_odds', -110),
                                'under_odds': prop.get('under_odds', -110),
                            })
                            break  # Found this player in this book, move to next book
                    if books_data and books_data[-1]['book'] == bookmaker:
                        break  # Found, stop searching events for this book
            except Exception as e:
                logger.debug("Failed to fetch from %s: %s", bookmaker, e)

        self._multi_book_cache[cache_key] = books_data
        return books_data

    def _detect_originator_book(self, books_data):
        # type: (List[Dict]) -> Optional[Dict]
        """
        Detect if one book is the originator (moved first, others followed).

        If one book has a significantly different line from the rest,
        it may be the originator of a sharp move.
        """
        if not books_data or len(books_data) < 3:
            return None

        lines = [b['line'] for b in books_data]
        avg_line = sum(lines) / len(lines)

        # Find the biggest outlier
        max_dev = 0
        outlier = None
        for b in books_data:
            dev = abs(b['line'] - avg_line)
            if dev > max_dev:
                max_dev = dev
                outlier = b

        if outlier is None or max_dev < self.CROSS_BOOK_EDGE_MIN:
            return None

        if outlier['line'] > avg_line:
            direction = 'sharp_over'
            edge_signal = 'over'
            reasoning = '{} has {:.1f} vs avg {:.1f} -- possible originator for OVER move'.format(
                outlier['book'].title(), outlier['line'], avg_line)
        else:
            direction = 'sharp_under'
            edge_signal = 'under'
            reasoning = '{} has {:.1f} vs avg {:.1f} -- possible originator for UNDER move'.format(
                outlier['book'].title(), outlier['line'], avg_line)

        confidence = min(0.5 + (max_dev / 2.0), 0.85)

        return {
            'direction': direction,
            'edge_signal': edge_signal,
            'confidence': round(confidence, 2),
            'reasoning': reasoning,
        }

    def _combine_signals(self, sharp, cross_book):
        # type: (Dict, Dict) -> Dict
        """
        Combine sharp movement and cross-book signals into a unified assessment.
        """
        sharp_dir = sharp.get('edge_signal')
        sharp_conf = sharp.get('confidence', 0)
        cross_edge = cross_book.get('edge', 0)
        cross_has_edge = cross_book.get('has_edge', False)

        # Determine cross-book direction
        cross_dir = None
        if cross_has_edge:
            over_edge = cross_book.get('over_edge', 0)
            under_edge = cross_book.get('under_edge', 0)
            cross_dir = 'over' if over_edge > under_edge else 'under'

        # Combine
        if sharp_dir and cross_dir and sharp_dir == cross_dir:
            # Both agree
            combined_conf = min(sharp_conf + 0.15, 1.0)
            direction = sharp_dir
            reasoning = 'Sharp money AND cross-book both signal {}'.format(direction.upper())
        elif sharp_dir and sharp_conf >= 0.5:
            # Sharp signal is strong enough on its own
            combined_conf = sharp_conf
            direction = sharp_dir
            reasoning = sharp.get('reasoning', 'Sharp signal detected')
        elif cross_has_edge and cross_edge >= self.CROSS_BOOK_EDGE_STRONG:
            # Strong cross-book edge
            combined_conf = min(0.4 + (cross_edge / 3.0), 0.75)
            direction = cross_dir
            reasoning = cross_book.get('reasoning', 'Cross-book edge detected')
        elif sharp_dir and cross_dir and sharp_dir != cross_dir:
            # Conflicting signals -- low confidence
            combined_conf = 0.2
            direction = 'mixed'
            reasoning = 'Conflicting signals: sharp={}, cross-book={}'.format(sharp_dir, cross_dir)
        else:
            combined_conf = 0.0
            direction = 'neutral'
            reasoning = 'No significant line signals detected'

        return {
            'direction': direction,
            'confidence': round(combined_conf, 2),
            'reasoning': reasoning,
        }

    def clear_cache(self):
        # type: () -> None
        """Clear the multi-book data cache."""
        self._multi_book_cache.clear()
