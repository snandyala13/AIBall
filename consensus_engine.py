#!/usr/bin/env python3
"""
consensus_engine.py — Cross-book consensus & devig engine

Computes devigged consensus probabilities from multi-book odds data.
Core of the "OddsJam-style" strategy: bet when DK deviates from sharp consensus.

Usage as library:
    from consensus_engine import ConsensusEngine
    engine = ConsensusEngine()
    result = engine.analyze_prop("Jayson Tatum", "player_points", "2025-01-15")
    enriched = engine.enrich_legs(legs)

Usage as CLI:
    python consensus_engine.py --date 2025-01-15
    python consensus_engine.py --stats
"""

import json
import logging
import os
import sqlite3
import argparse
from dataclasses import dataclass, asdict
from math import log, exp
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scipy.optimize import brentq

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "data" / "multibook_odds.db"

MARKET_TO_PROP = {
    "player_points": "points",
    "player_rebounds": "rebounds",
    "player_assists": "assists",
    "player_threes": "threes",
    "player_points_rebounds_assists": "pts_rebs_asts",
    "player_double_double": "double_double",
}
PROP_TO_MARKET = {v: k for k, v in MARKET_TO_PROP.items()}

# Sharp books get higher weight in consensus probability
BOOK_SHARPNESS = {
    "pinnacle": 1.0,
    "pinnaclesports": 1.0,
    "circa": 0.85,
    "bookmaker": 0.80,
    "betcris": 0.75,
    "betonlineag": 0.70,
    "bovada": 0.65,
    "lowvig": 0.90,
    "novig": 0.88,
    # Retail books — softer lines, lower weight
    "draftkings": 0.50,
    "fanduel": 0.55,
    "betmgm": 0.50,
    "caesars": 0.50,
    "pointsbetus": 0.45,
    "wynnbet": 0.45,
    "betrivers": 0.50,
    "unibet_us": 0.45,
    "superbook": 0.60,
    "espnbet": 0.50,
    "fliff": 0.40,
    "hardrockbet": 0.45,
    "betparx": 0.40,
    "fanatics": 0.50,
    "windcreek": 0.40,
    "tipico_us": 0.45,
}
DEFAULT_SHARPNESS = 0.45

SOFT_BOOKS = frozenset({"betparx", "betrivers", "espnbet"})


# =============================================================================
# DEVIG MATH
# =============================================================================

def american_to_implied(odds: int) -> float:
    if odds == 0:
        return 0.5
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 100.0 / (odds + 100.0)


def implied_to_american(prob: float) -> int:
    if prob <= 0 or prob >= 1:
        return 0
    if prob >= 0.5:
        return int(round(-100.0 * prob / (1.0 - prob)))
    return int(round(100.0 * (1.0 - prob) / prob))


def devig_multiplicative(over_odds: int, under_odds: int) -> Tuple[float, float]:
    """Remove vig by scaling implied probs to sum to 1."""
    p_over = american_to_implied(over_odds)
    p_under = american_to_implied(under_odds)
    total = p_over + p_under
    if total <= 0:
        return 0.5, 0.5
    return p_over / total, p_under / total


def devig_power(over_odds: int, under_odds: int) -> Tuple[float, float]:
    """Remove vig via the power method (Shin-like). More accurate for skewed odds."""
    p_over_raw = american_to_implied(over_odds)
    p_under_raw = american_to_implied(under_odds)

    if p_over_raw <= 0 or p_under_raw <= 0:
        return devig_multiplicative(over_odds, under_odds)
    if abs(p_over_raw + p_under_raw - 1.0) < 0.001:
        return p_over_raw, p_under_raw

    try:
        def objective(c):
            return p_over_raw ** c + p_under_raw ** c - 1.0

        lo, hi = 0.01, 5.0
        if objective(lo) * objective(hi) > 0:
            return devig_multiplicative(over_odds, under_odds)

        c = brentq(objective, lo, hi, xtol=1e-8, maxiter=100)
        return p_over_raw ** c, p_under_raw ** c
    except (ValueError, RuntimeError):
        return devig_multiplicative(over_odds, under_odds)


def devig(over_odds: int, under_odds: int, method: str = "power") -> Tuple[float, float]:
    if method == "power":
        return devig_power(over_odds, under_odds)
    return devig_multiplicative(over_odds, under_odds)


def compute_vig(over_odds: int, under_odds: int) -> float:
    return american_to_implied(over_odds) + american_to_implied(under_odds) - 1.0


# =============================================================================
# CONSENSUS DATA STRUCTURES
# =============================================================================

@dataclass
class BookLine:
    bookmaker: str
    line: float
    over_odds: int
    under_odds: int
    devigged_over: float
    devigged_under: float
    vig: float
    sharpness: float


@dataclass
class ConsensusResult:
    player: str
    market: str
    prop_type: str
    date: str
    num_books: int
    consensus_line: float
    consensus_over_prob: float
    consensus_under_prob: float
    line_range: float
    # DK-specific
    dk_line: Optional[float]
    dk_over_odds: Optional[int]
    dk_under_odds: Optional[int]
    dk_implied_over: Optional[float]
    dk_vs_consensus_line: Optional[float]
    dk_vs_consensus_prob: Optional[float]
    dk_over_ev: Optional[float]
    dk_under_ev: Optional[float]
    consensus_edge_over: Optional[float]
    consensus_edge_under: Optional[float]
    books: List[BookLine]

    def best_side(self) -> Optional[str]:
        if self.consensus_edge_over is not None and self.consensus_edge_under is not None:
            if self.consensus_edge_over > self.consensus_edge_under and self.consensus_edge_over > 0:
                return "OVER"
            if self.consensus_edge_under > self.consensus_edge_over and self.consensus_edge_under > 0:
                return "UNDER"
        return None

    def best_edge(self) -> float:
        over = self.consensus_edge_over or 0
        under = self.consensus_edge_under or 0
        return max(over, under)


@dataclass
class ConsensusSignal:
    """Lightweight consensus result for live prediction enrichment."""
    consensus_over_prob: float
    consensus_under_prob: float
    consensus_line: float
    line_range: float
    num_books: int
    soft_book_score: float
    dk_vs_consensus_prob: float
    odds_spread_width: float


# =============================================================================
# ENGINE
# =============================================================================

class ConsensusEngine:

    def __init__(self, db_path: str = None):
        self.db_path = str(db_path or DB_PATH)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def analyze_from_books_data(self, player: str, prop_type: str,
                                books_dict: Dict, dk_line: float = 0,
                                dk_over_odds: int = -110,
                                dk_under_odds: int = -110) -> Optional[ConsensusSignal]:
        """Compute consensus from live PropLine all_books dict (no DB needed).

        Args:
            player: player name (for logging only)
            prop_type: e.g. "points", "rebounds"
            books_dict: {book_key: {line, over_odds, under_odds, ...}}
            dk_line: DraftKings line for DK-vs-consensus comparison
            dk_over_odds: DK over odds
            dk_under_odds: DK under odds
        """
        if not books_dict or len(books_dict) < 2:
            return None

        books = []  # type: List[BookLine]
        for bk, bd in books_dict.items():
            line = bd.get("line")
            over_o = bd.get("over_odds")
            under_o = bd.get("under_odds")
            if line is None or not over_o or not under_o:
                continue
            try:
                over_o = int(over_o)
                under_o = int(under_o)
            except (ValueError, TypeError):
                continue
            if over_o == 0 or under_o == 0:
                continue

            dv_over, dv_under = devig_power(over_o, under_o)
            vig = compute_vig(over_o, under_o)
            sharpness = BOOK_SHARPNESS.get(bk.lower(), DEFAULT_SHARPNESS)
            books.append(BookLine(
                bookmaker=bk, line=float(line), over_odds=over_o,
                under_odds=under_o, devigged_over=dv_over,
                devigged_under=dv_under, vig=vig, sharpness=sharpness,
            ))

        if len(books) < 2:
            return None

        cons_line = self._consensus_line(books)
        cons_over, cons_under = self._consensus_prob(books)

        lines = [b.line for b in books]
        line_range = max(lines) - min(lines)

        # Odds spread: max over_odds minus min over_odds across books
        over_odds_list = [b.over_odds for b in books]
        odds_spread_width = max(over_odds_list) - min(over_odds_list) if len(over_odds_list) > 1 else 0

        # DK vs consensus probability
        dk_imp_over = american_to_implied(int(dk_over_odds)) if dk_over_odds else 0.5
        dk_vs_cons_prob = dk_imp_over - cons_over

        # Soft book score: fraction of soft books with best/near-best odds
        # for the consensus-recommended side
        recommended_under = cons_under > cons_over
        soft_with_good_odds = 0
        soft_available = 0
        if books:
            all_side_odds = []
            for b in books:
                odds = b.under_odds if recommended_under else b.over_odds
                all_side_odds.append(odds)
            if all_side_odds:
                best_odds = max(all_side_odds)
                threshold = best_odds - 15  # within 15 cents of best
                for b in books:
                    if b.bookmaker.lower() in SOFT_BOOKS:
                        soft_available += 1
                        odds = b.under_odds if recommended_under else b.over_odds
                        if odds >= threshold:
                            soft_with_good_odds += 1

        soft_book_score = soft_with_good_odds / max(soft_available, 1)

        return ConsensusSignal(
            consensus_over_prob=round(cons_over, 4),
            consensus_under_prob=round(cons_under, 4),
            consensus_line=round(cons_line, 2),
            line_range=round(line_range, 1),
            num_books=len(books),
            soft_book_score=round(soft_book_score, 2),
            dk_vs_consensus_prob=round(dk_vs_cons_prob, 4),
            odds_spread_width=odds_spread_width,
        )

    def _get_book_lines(self, player: str, market: str, date: str) -> List[BookLine]:
        conn = self._conn()
        rows = conn.execute("""
            SELECT p.bookmaker, p.line, p.over_odds, p.under_odds
            FROM props p
            JOIN events e ON p.event_id = e.event_id
            WHERE e.date = ? AND p.player_name = ? AND p.market = ?
        """, (date, player, market)).fetchall()
        conn.close()

        books = []
        for bk, line, over_o, under_o in rows:
            dv_over, dv_under = devig(over_o, under_o)
            vig = compute_vig(over_o, under_o)
            sharpness = BOOK_SHARPNESS.get(bk.lower(), DEFAULT_SHARPNESS)
            books.append(BookLine(
                bookmaker=bk, line=line, over_odds=over_o, under_odds=under_o,
                devigged_over=dv_over, devigged_under=dv_under,
                vig=vig, sharpness=sharpness,
            ))
        return books

    def _consensus_line(self, books: List[BookLine]) -> float:
        pinnacle = [b for b in books if b.bookmaker.lower() in ("pinnacle", "pinnaclesports")]
        if pinnacle:
            pin_line = pinnacle[0].line
            other_lines = [b.line for b in books if b.bookmaker.lower() not in ("pinnacle", "pinnaclesports")]
            if other_lines:
                other_median = sorted(other_lines)[len(other_lines) // 2]
                return 0.5 * pin_line + 0.5 * other_median
            return pin_line

        lines = sorted([b.line for b in books])
        return lines[len(lines) // 2]

    def _consensus_prob(self, books: List[BookLine]) -> Tuple[float, float]:
        if not books:
            return 0.5, 0.5

        total_weight = 0.0
        weighted_over = 0.0
        for b in books:
            w = b.sharpness
            weighted_over += w * b.devigged_over
            total_weight += w

        if total_weight <= 0:
            return 0.5, 0.5

        p_over = weighted_over / total_weight
        return p_over, 1.0 - p_over

    def analyze_prop(self, player: str, market: str, date: str) -> Optional[ConsensusResult]:
        books = self._get_book_lines(player, market, date)
        if not books:
            return None

        prop_type = MARKET_TO_PROP.get(market, market)
        cons_line = self._consensus_line(books)
        cons_over, cons_under = self._consensus_prob(books)

        lines = [b.line for b in books]
        line_range = max(lines) - min(lines) if len(lines) > 1 else 0.0

        dk_books = [b for b in books if b.bookmaker.lower() == "draftkings"]
        dk = dk_books[0] if dk_books else None

        dk_line = dk.line if dk else None
        dk_over_odds = dk.over_odds if dk else None
        dk_under_odds = dk.under_odds if dk else None
        dk_implied_over = american_to_implied(dk.over_odds) if dk else None
        dk_implied_under = american_to_implied(dk.under_odds) if dk else None

        dk_vs_cons_line = (dk.line - cons_line) if dk else None
        dk_vs_cons_prob = (dk_implied_over - cons_over) if dk and dk_implied_over else None

        # EV = true_prob * payout - (1 - true_prob) * stake
        # For American odds: decimal = (100 + abs(odds)) / abs(odds) if neg, else (odds + 100) / 100
        def ev_at_odds(true_prob, odds):
            if odds < 0:
                decimal = 1.0 + 100.0 / abs(odds)
            else:
                decimal = 1.0 + odds / 100.0
            return true_prob * (decimal - 1.0) - (1.0 - true_prob)

        dk_over_ev = ev_at_odds(cons_over, dk.over_odds) if dk else None
        dk_under_ev = ev_at_odds(cons_under, dk.under_odds) if dk else None

        edge_over = (cons_over - dk_implied_over) if dk and dk_implied_over else None
        edge_under = (cons_under - dk_implied_under) if dk and dk_implied_under else None

        return ConsensusResult(
            player=player, market=market, prop_type=prop_type, date=date,
            num_books=len(books),
            consensus_line=cons_line,
            consensus_over_prob=cons_over, consensus_under_prob=cons_under,
            line_range=line_range,
            dk_line=dk_line, dk_over_odds=dk_over_odds, dk_under_odds=dk_under_odds,
            dk_implied_over=dk_implied_over,
            dk_vs_consensus_line=dk_vs_cons_line, dk_vs_consensus_prob=dk_vs_cons_prob,
            dk_over_ev=dk_over_ev, dk_under_ev=dk_under_ev,
            consensus_edge_over=edge_over, consensus_edge_under=edge_under,
            books=books,
        )

    def compute_for_date(self, date: str) -> List[ConsensusResult]:
        conn = self._conn()
        combos = conn.execute("""
            SELECT DISTINCT p.player_name, p.market
            FROM props p
            JOIN events e ON p.event_id = e.event_id
            WHERE e.date = ?
        """, (date,)).fetchall()
        conn.close()

        results = []
        for player, market in combos:
            r = self.analyze_prop(player, market, date)
            if r:
                results.append(r)
        return results

    def enrich_legs(self, legs: List[Dict]) -> List[Dict]:
        """Add consensus fields to existing backtest legs."""
        conn = self._conn()

        # Preload all props grouped by (date, player, market)
        all_props = conn.execute("""
            SELECT e.date, p.player_name, p.market,
                   p.bookmaker, p.line, p.over_odds, p.under_odds
            FROM props p
            JOIN events e ON p.event_id = e.event_id
        """).fetchall()
        conn.close()

        # Index: (date, player_lower, market) -> list of (bookmaker, line, over, under)
        index = {}
        for date, player, market, bk, line, over_o, under_o in all_props:
            key = (date, player.lower(), market)
            if key not in index:
                index[key] = []
            index[key].append((bk, line, over_o, under_o))

        enriched = 0
        for leg in legs:
            date = leg.get("date", "")
            player = leg.get("player", "")
            prop_type = leg.get("prop_type", "")
            market = PROP_TO_MARKET.get(prop_type)
            if not market:
                leg.update(self._empty_consensus_fields())
                continue

            key = (date, player.lower(), market)
            raw = index.get(key)

            if not raw:
                last = player.split()[-1].lower() if player else ""
                if last:
                    for k, v in index.items():
                        if k[0] == date and k[2] == market and last in k[1].split()[-1:]:
                            raw = v
                            break

            if not raw:
                leg.update(self._empty_consensus_fields())
                continue

            books = []
            for bk, line, over_o, under_o in raw:
                dv_over, dv_under = devig(over_o, under_o)
                vig = compute_vig(over_o, under_o)
                sharpness = BOOK_SHARPNESS.get(bk.lower(), DEFAULT_SHARPNESS)
                books.append(BookLine(
                    bookmaker=bk, line=line, over_odds=over_o, under_odds=under_o,
                    devigged_over=dv_over, devigged_under=dv_under,
                    vig=vig, sharpness=sharpness,
                ))

            cons_line = self._consensus_line(books)
            cons_over, cons_under = self._consensus_prob(books)
            lines = [b.line for b in books]

            dk_books = [b for b in books if b.bookmaker.lower() == "draftkings"]
            dk = dk_books[0] if dk_books else None

            dk_imp_over = american_to_implied(dk.over_odds) if dk else None
            dk_imp_under = american_to_implied(dk.under_odds) if dk else None

            def ev_at_odds(true_prob, odds):
                if odds < 0:
                    decimal = 1.0 + 100.0 / abs(odds)
                else:
                    decimal = 1.0 + odds / 100.0
                return true_prob * (decimal - 1.0) - (1.0 - true_prob)

            leg["num_books"] = len(books)
            leg["consensus_line"] = round(cons_line, 2)
            leg["consensus_over_prob"] = round(cons_over, 4)
            leg["consensus_under_prob"] = round(cons_under, 4)
            leg["line_range"] = round(max(lines) - min(lines), 1) if len(lines) > 1 else 0
            leg["dk_vs_consensus"] = round(dk.line - cons_line, 2) if dk else 0
            leg["dk_vs_consensus_prob"] = round((dk_imp_over or 0) - cons_over, 4) if dk else 0
            leg["dk_over_ev"] = round(ev_at_odds(cons_over, dk.over_odds), 4) if dk else None
            leg["dk_under_ev"] = round(ev_at_odds(cons_under, dk.under_odds), 4) if dk else None
            leg["consensus_edge_over"] = round(cons_over - (dk_imp_over or 0), 4) if dk else None
            leg["consensus_edge_under"] = round(cons_under - (dk_imp_under or 0), 4) if dk else None

            # Per-book lines dict for analysis
            leg["all_books"] = {
                b.bookmaker: {"line": b.line, "over_odds": b.over_odds, "under_odds": b.under_odds,
                              "devigged_over": round(b.devigged_over, 4)}
                for b in books
            }

            enriched += 1

        logger.info(f"Enriched {enriched}/{len(legs)} legs with consensus data")
        return legs

    @staticmethod
    def _empty_consensus_fields() -> Dict:
        return {
            "num_books": 0, "consensus_line": None,
            "consensus_over_prob": None, "consensus_under_prob": None,
            "line_range": 0, "dk_vs_consensus": 0, "dk_vs_consensus_prob": 0,
            "dk_over_ev": None, "dk_under_ev": None,
            "consensus_edge_over": None, "consensus_edge_under": None,
            "all_books": {},
        }

    def get_stats(self) -> Dict:
        conn = self._conn()
        try:
            dates = conn.execute("SELECT COUNT(DISTINCT date) FROM events").fetchone()[0]
            events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            props = conn.execute("SELECT COUNT(*) FROM props").fetchone()[0]
            books = conn.execute("SELECT COUNT(DISTINCT bookmaker) FROM props").fetchone()[0]
            players = conn.execute("SELECT COUNT(DISTINCT player_name) FROM props").fetchone()[0]
            dr = conn.execute("SELECT MIN(date), MAX(date) FROM events").fetchone()
            top_books = conn.execute(
                "SELECT bookmaker, COUNT(*) c FROM props GROUP BY bookmaker ORDER BY c DESC LIMIT 15"
            ).fetchall()
            return {
                "dates": dates, "events": events, "props": props,
                "books": books, "players": players,
                "date_range": f"{dr[0]} to {dr[1]}" if dr[0] else "none",
                "top_books": top_books,
            }
        finally:
            conn.close()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Cross-book consensus engine")
    parser.add_argument("--date", help="Analyze all props for a date")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--player", help="Filter to player name")
    parser.add_argument("--min-edge", type=float, default=0.0, help="Min consensus edge to show")
    parser.add_argument("--min-books", type=int, default=3, help="Min books required")
    args = parser.parse_args()

    engine = ConsensusEngine()

    if args.stats:
        s = engine.get_stats()
        print(f"\nMulti-book database: {s['date_range']}")
        print(f"  {s['dates']} dates, {s['events']} events, {s['props']:,} prop rows")
        print(f"  {s['books']} bookmakers, {s['players']} players")
        print("\n  Bookmakers:")
        for bk, cnt in s["top_books"]:
            sharp = BOOK_SHARPNESS.get(bk.lower(), DEFAULT_SHARPNESS)
            print(f"    {bk:20s} {cnt:>8,} props  (sharpness: {sharp:.2f})")
        return

    if args.date:
        results = engine.compute_for_date(args.date)
        if args.player:
            results = [r for r in results if args.player.lower() in r.player.lower()]
        results = [r for r in results if r.num_books >= args.min_books]
        results = [r for r in results if r.best_edge() >= args.min_edge]
        results.sort(key=lambda r: r.best_edge(), reverse=True)

        print(f"\n{'Player':<25} {'Prop':<12} {'Books':>5} {'Cons Line':>9} {'DK Line':>7} "
              f"{'Dev':>6} {'Edge Over':>10} {'Edge Under':>11} {'Best':>5}")
        print("-" * 105)
        for r in results:
            side = r.best_side() or "-"
            print(
                f"{r.player:<25} {r.prop_type:<12} {r.num_books:>5} "
                f"{r.consensus_line:>9.1f} {r.dk_line or 0:>7.1f} "
                f"{r.dk_vs_consensus_line or 0:>+6.1f} "
                f"{(r.consensus_edge_over or 0)*100:>+9.1f}% "
                f"{(r.consensus_edge_under or 0)*100:>+10.1f}% "
                f"{side:>5}"
            )
        print(f"\n{len(results)} props shown (min {args.min_books} books, min {args.min_edge*100:.0f}% edge)")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
