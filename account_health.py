"""
account_health.py — Per-book CLV tracking and limiting-risk detection.

Sportsbooks limit based on CLV patterns, not win/loss records.  Nearly half
of Fanatics' limited customers were net losers when they were limited (ESPN).
Account lifespan at soft books is 3-12 weeks for sharp-patterned accounts.

This module:
  1. Records every bet with book, odds, stake, and our probability
  2. Computes CLV when closing odds become available
  3. Scores each book account 0-100 for limiting risk
  4. Recommends which book to place a bet at (best odds × healthiest account)
  5. Suggests stake disguising and behavioral changes

State persists in data/account_health.json across restarts.
"""

import json
import logging
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_STATE_FILE = str(BASE_DIR / "data" / "account_health.json")

RECREATIONAL_STAKES = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250, 500]


def _american_to_implied(odds: int) -> float:
    if odds == 0:
        return 0.5
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 100.0 / (odds + 100.0)


def _american_to_decimal(odds: int) -> float:
    if odds == 0:
        return 2.0
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def round_to_recreational(raw_stake: float) -> float:
    """Round a precise Kelly-derived stake to the nearest recreational amount."""
    if raw_stake <= 0:
        return 0.0
    best = min(RECREATIONAL_STAKES, key=lambda s: abs(s - raw_stake))
    return float(best)


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class BetRecord:
    bet_id: str
    book: str
    player: str
    prop_type: str
    side: str          # "over" / "under"
    line: float
    odds: int          # American odds at placement
    stake: float
    our_prob: float    # model probability for the chosen side
    placed_at: str     # ISO timestamp
    # Filled later
    closing_over_odds: Optional[int] = None
    closing_under_odds: Optional[int] = None
    clv: Optional[float] = None       # closing line value
    result: Optional[str] = None      # "win" / "loss" / "push"
    profit: Optional[float] = None


@dataclass
class BookAccount:
    book: str
    status: str = "healthy"
    total_bets: int = 0
    total_wagered: float = 0.0
    total_profit: float = 0.0
    avg_clv: float = 0.0
    recent_clv_30d: float = 0.0
    last_bet_date: str = ""
    first_bet_date: str = ""
    days_active: int = 0
    max_single_bet: float = 0.0
    avg_stake: float = 0.0
    limiting_risk_score: float = 0.0


# =============================================================================
# AccountHealthMonitor
# =============================================================================

class AccountHealthMonitor:

    def __init__(self, state_file: str = DEFAULT_STATE_FILE):
        self.state_file = state_file
        self.accounts: Dict[str, BookAccount] = {}
        self.bet_history: List[BetRecord] = []
        self._load_state()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def record_bet(
        self,
        book: str,
        player: str,
        prop_type: str,
        side: str,
        line: float,
        odds: int,
        stake: float,
        our_prob: float,
        closing_over_odds: Optional[int] = None,
        closing_under_odds: Optional[int] = None,
    ) -> str:
        """Record a bet placement.  Returns the generated bet_id."""
        bet_id = uuid.uuid4().hex[:12]
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        rec = BetRecord(
            bet_id=bet_id,
            book=book.lower(),
            player=player,
            prop_type=prop_type,
            side=side.lower(),
            line=line,
            odds=odds,
            stake=stake,
            our_prob=our_prob,
            placed_at=now,
            closing_over_odds=closing_over_odds,
            closing_under_odds=closing_under_odds,
        )

        if closing_over_odds is not None and closing_under_odds is not None:
            rec.clv = self._compute_clv(rec)

        self.bet_history.append(rec)
        self._refresh_account(rec.book)
        self._save_state()
        return bet_id

    def record_result(self, bet_id: str, result: str, profit: float):
        """Record win/loss/push for a bet after the game finishes."""
        for rec in self.bet_history:
            if rec.bet_id == bet_id:
                rec.result = result
                rec.profit = profit
                self._refresh_account(rec.book)
                self._save_state()
                return
        logger.warning("bet_id %s not found", bet_id)

    def update_closing_line(
        self, bet_id: str, closing_over_odds: int, closing_under_odds: int
    ):
        """Set closing odds and compute CLV for a previously recorded bet."""
        for rec in self.bet_history:
            if rec.bet_id == bet_id:
                rec.closing_over_odds = closing_over_odds
                rec.closing_under_odds = closing_under_odds
                rec.clv = self._compute_clv(rec)
                self._refresh_account(rec.book)
                self._save_state()
                return
        logger.warning("bet_id %s not found", bet_id)

    # ------------------------------------------------------------------
    # CLV computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_clv(rec: BetRecord) -> float:
        """CLV = our implied probability at placement minus closing implied probability.

        Positive CLV means we got a better price than the closing line —
        the hallmark sportsbooks use to identify sharp bettors.
        """
        if rec.closing_over_odds is None or rec.closing_under_odds is None:
            return 0.0

        placement_implied = _american_to_implied(rec.odds)

        if rec.side == "over":
            closing_implied = _american_to_implied(rec.closing_over_odds)
        else:
            closing_implied = _american_to_implied(rec.closing_under_odds)

        # CLV = closing implied - placement implied  (positive = we got better price)
        return closing_implied - placement_implied

    # ------------------------------------------------------------------
    # Account health
    # ------------------------------------------------------------------

    def _refresh_account(self, book: str):
        """Recompute BookAccount stats from bet history."""
        bets = [b for b in self.bet_history if b.book == book]
        if not bets:
            self.accounts.pop(book, None)
            return

        acct = self.accounts.get(book) or BookAccount(book=book)

        acct.total_bets = len(bets)
        acct.total_wagered = sum(b.stake for b in bets)
        acct.total_profit = sum(b.profit for b in bets if b.profit is not None)
        acct.avg_stake = acct.total_wagered / max(acct.total_bets, 1)
        acct.max_single_bet = max(b.stake for b in bets)

        dates = sorted(b.placed_at[:10] for b in bets)
        acct.first_bet_date = dates[0]
        acct.last_bet_date = dates[-1]
        try:
            d0 = datetime.fromisoformat(acct.first_bet_date)
            d1 = datetime.fromisoformat(acct.last_bet_date)
            acct.days_active = max((d1 - d0).days, 1)
        except (ValueError, TypeError):
            acct.days_active = 1

        clvs = [b.clv for b in bets if b.clv is not None]
        acct.avg_clv = float(np.mean(clvs)) if clvs else 0.0

        cutoff = (datetime.utcnow() - timedelta(days=30)).isoformat()[:10]
        recent = [b.clv for b in bets if b.clv is not None and b.placed_at[:10] >= cutoff]
        acct.recent_clv_30d = float(np.mean(recent)) if recent else 0.0

        acct.limiting_risk_score = self._compute_limiting_risk_from_bets(bets, acct)
        acct.status = self._status_from_risk(acct.limiting_risk_score)

        self.accounts[book] = acct

    def compute_limiting_risk(self, book: str) -> float:
        acct = self.accounts.get(book)
        if not acct:
            return 0.0
        return acct.limiting_risk_score

    @staticmethod
    def _compute_limiting_risk_from_bets(bets: List[BetRecord], acct: BookAccount) -> float:
        risk = 0.0

        # --- CLV signals ---
        if acct.avg_clv > 0.03:
            risk += 30
        elif acct.avg_clv > 0.015:
            risk += 15
        elif acct.avg_clv > 0.005:
            risk += 5

        # --- Bet frequency ---
        if acct.days_active > 0:
            bets_per_day = acct.total_bets / acct.days_active
            if bets_per_day > 5:
                risk += 10
            elif bets_per_day > 3:
                risk += 5

        # --- Stake consistency (low CV = algorithmic) ---
        stakes = [b.stake for b in bets]
        if len(stakes) >= 5:
            cv = float(np.std(stakes) / max(np.mean(stakes), 1.0))
            if cv < 0.10:
                risk += 10
            elif cv < 0.20:
                risk += 5

        # --- Win rate ---
        resolved = [b for b in bets if b.result in ("win", "loss")]
        if len(resolved) >= 20:
            win_rate = sum(1 for b in resolved if b.result == "win") / len(resolved)
            if win_rate > 0.57:
                risk += 10
            elif win_rate > 0.55:
                risk += 5

        # --- UNDER bias (correlated with sharp bettors) ---
        if len(bets) >= 10:
            under_pct = sum(1 for b in bets if b.side == "under") / len(bets)
            if under_pct > 0.70:
                risk += 5

        # --- New account rapid betting ---
        if acct.days_active < 30 and acct.total_bets >= 50:
            risk += 15
        elif acct.days_active < 30 and acct.total_bets >= 25:
            risk += 8

        # --- Large stakes relative to typical recreational ---
        if acct.max_single_bet > 500:
            risk += 5
        if acct.avg_stake > 200:
            risk += 5

        return min(risk, 100.0)

    @staticmethod
    def _status_from_risk(risk: float) -> str:
        if risk >= 75:
            return "banned"
        if risk >= 50:
            return "limited"
        if risk >= 25:
            return "warning"
        return "healthy"

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_book_health(self, book: str) -> dict:
        acct = self.accounts.get(book.lower())
        if not acct:
            return {"book": book, "status": "unknown", "total_bets": 0}
        return asdict(acct)

    def get_all_health(self) -> dict:
        summary = {
            "books": {bk: asdict(a) for bk, a in self.accounts.items()},
            "total_bets": sum(a.total_bets for a in self.accounts.values()),
            "total_wagered": sum(a.total_wagered for a in self.accounts.values()),
            "total_profit": sum(a.total_profit for a in self.accounts.values()),
            "healthy_books": [bk for bk, a in self.accounts.items() if a.status == "healthy"],
            "warning_books": [bk for bk, a in self.accounts.items() if a.status == "warning"],
            "limited_books": [bk for bk, a in self.accounts.items() if a.status == "limited"],
            "banned_books": [bk for bk, a in self.accounts.items() if a.status == "banned"],
        }
        return summary

    def recommend_book(
        self,
        player: str,
        prop_type: str,
        side: str,
        available_books: Dict[str, dict],
    ) -> Optional[dict]:
        """Pick the best book for a bet, balancing odds quality and account health.

        Args:
            player: player name
            prop_type: e.g. "points"
            side: "over" or "under"
            available_books: {book_key: {"line": ..., "over_odds": ..., "under_odds": ...}}

        Returns:
            {"book": str, "odds": int, "reason": str} or None
        """
        if not available_books:
            return None

        candidates = []
        for bk, bd in available_books.items():
            odds_key = "over_odds" if side == "over" else "under_odds"
            odds = bd.get(odds_key)
            if odds is None:
                continue

            try:
                odds = int(odds)
            except (TypeError, ValueError):
                continue

            acct = self.accounts.get(bk.lower())
            risk = acct.limiting_risk_score if acct else 0.0
            status = acct.status if acct else "healthy"

            if status == "banned":
                continue

            # Score: higher odds (better payout) + lower risk = better
            decimal = _american_to_decimal(odds)
            # Penalize books with high limiting risk
            health_penalty = risk / 100.0  # 0 to 1
            # Penalize books we've used recently (rotation)
            recency_penalty = 0.0
            if acct and acct.last_bet_date:
                try:
                    last = datetime.fromisoformat(acct.last_bet_date)
                    days_since = (datetime.utcnow() - last).days
                    if days_since < 1:
                        recency_penalty = 0.05
                except (ValueError, TypeError):
                    pass

            score = decimal - (health_penalty * 0.15) - recency_penalty

            candidates.append({
                "book": bk,
                "odds": odds,
                "decimal": decimal,
                "risk": risk,
                "status": status,
                "score": score,
            })

        if not candidates:
            return None

        candidates.sort(key=lambda c: c["score"], reverse=True)
        best = candidates[0]

        reason_parts = [f"best adjusted score ({best['score']:.3f})"]
        if best["risk"] > 0:
            reason_parts.append(f"risk={best['risk']:.0f}")
        if best["status"] != "healthy":
            reason_parts.append(f"status={best['status']}")

        return {
            "book": best["book"],
            "odds": best["odds"],
            "reason": ", ".join(reason_parts),
            "alternatives": [
                {"book": c["book"], "odds": c["odds"], "risk": c["risk"]}
                for c in candidates[1:4]
            ],
        }

    def get_disguise_recommendations(self, book: str) -> List[str]:
        acct = self.accounts.get(book.lower())
        if not acct:
            return []

        recs = []
        risk = acct.limiting_risk_score

        if risk < 10:
            return ["Account looks healthy — no changes needed."]

        # Stake consistency
        bets = [b for b in self.bet_history if b.book == book.lower()]
        stakes = [b.stake for b in bets]
        if len(stakes) >= 5:
            cv = float(np.std(stakes) / max(np.mean(stakes), 1.0))
            if cv < 0.20:
                recs.append(
                    "Vary stake amounts more. Use round recreational amounts "
                    "($25, $50, $100) instead of precise Kelly-derived values."
                )

        # UNDER bias
        if len(bets) >= 10:
            under_pct = sum(1 for b in bets if b.side == "under") / len(bets)
            if under_pct > 0.70:
                recs.append(
                    f"Under bias is {under_pct:.0%} — place some over bets "
                    "on popular markets to look recreational."
                )

        # Frequency
        if acct.days_active > 0:
            bpd = acct.total_bets / acct.days_active
            if bpd > 3:
                recs.append(
                    f"Betting frequency ({bpd:.1f}/day) is high. "
                    "Skip some marginal edges and batch bets to look less automated."
                )

        # High CLV
        if acct.avg_clv > 0.015:
            recs.append(
                f"Average CLV ({acct.avg_clv:.1%}) is sharp-level. "
                "Consider waiting 5-10 min after line moves before placing, "
                "and avoid betting the opening line."
            )

        # Large stakes
        if acct.max_single_bet > 500:
            recs.append(
                "Cap single bets at $200-$300 at soft books to stay under the radar."
            )

        if acct.status in ("limited", "banned"):
            recs.append(
                "This account is likely already limited. Focus bets on other books."
            )

        return recs if recs else ["Account looks reasonable — keep monitoring."]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_state(self):
        data = {
            "accounts": {bk: asdict(a) for bk, a in self.accounts.items()},
            "bet_history": [asdict(b) for b in self.bet_history],
            "saved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
        tmp = self.state_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, self.state_file)

    def _load_state(self):
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not load account health state: %s", e)
            return

        for bk, ad in data.get("accounts", {}).items():
            self.accounts[bk] = BookAccount(**{
                k: v for k, v in ad.items() if k in BookAccount.__dataclass_fields__
            })

        for bd in data.get("bet_history", []):
            self.bet_history.append(BetRecord(**{
                k: v for k, v in bd.items() if k in BetRecord.__dataclass_fields__
            }))
