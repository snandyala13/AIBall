#!/usr/bin/env python3
"""
portfolio_manager.py — Portfolio exposure management for sports betting.

Manages bankroll, position sizing, exposure caps, and drawdown protection.
Persists state to data/portfolio_state.json across restarts.

Key rules:
  - Per-player cap: 3% of bankroll
  - Per-team cap: 7% of bankroll
  - Total exposure cap: 20% of bankroll
  - Correlation-adjusted quarter-Kelly sizing
  - Drawdown circuit breaker: 50% stake cut at 15% drawdown from peak
  - Stake rounding to recreational amounts
"""

import json
import logging
import os
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
STATE_PATH = BASE_DIR / "data" / "portfolio_state.json"


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class BetPosition:
    bet_id: str
    player: str
    prop_type: str
    team: str
    game_id: str
    book: str
    side: str
    line: float
    odds: int
    stake: float
    edge: float
    p_win: float
    placed_at: str
    status: str = "open"
    profit: float = 0.0
    actual: Optional[float] = None
    resolved_at: Optional[str] = None
    closing_line_odds: Optional[int] = None
    clv: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BetPosition":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


# =============================================================================
# Portfolio Manager
# =============================================================================

class PortfolioManager:
    """Manages bankroll, position sizing, and exposure limits."""

    def __init__(
        self,
        bankroll: float = 1000.0,
        max_player_pct: float = 0.03,
        max_team_pct: float = 0.07,
        max_total_pct: float = 0.20,
        kelly_fraction: float = 0.25,
        drawdown_threshold: float = 0.15,
        drawdown_recovery: float = 0.10,
        state_path: Optional[str] = None,
    ):
        self.initial_bankroll = bankroll
        self.bankroll = bankroll
        self.peak_bankroll = bankroll
        self.max_player_pct = max_player_pct
        self.max_team_pct = max_team_pct
        self.max_total_pct = max_total_pct
        self.kelly_fraction = kelly_fraction
        self.drawdown_threshold = drawdown_threshold
        self.drawdown_recovery = drawdown_recovery
        self.state_path = Path(state_path) if state_path else STATE_PATH

        self.positions: Dict[str, BetPosition] = {}
        self.history: List[dict] = []

        self._load_state()

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def _load_state(self):
        if not self.state_path.exists():
            return
        try:
            with open(self.state_path) as f:
                state = json.load(f)
            self.bankroll = state.get("bankroll", self.bankroll)
            self.peak_bankroll = state.get("peak_bankroll", self.peak_bankroll)
            self.initial_bankroll = state.get("initial_bankroll", self.initial_bankroll)
            for d in state.get("positions", []):
                pos = BetPosition.from_dict(d)
                self.positions[pos.bet_id] = pos
            self.history = state.get("history", [])
            logger.info(
                "Portfolio loaded: $%.2f bankroll, %d open, %d history",
                self.bankroll, len(self._open_positions()), len(self.history),
            )
        except Exception as e:
            logger.warning("Failed to load portfolio state: %s", e)

    def _save_state(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "bankroll": self.bankroll,
            "peak_bankroll": self.peak_bankroll,
            "initial_bankroll": self.initial_bankroll,
            "positions": [p.to_dict() for p in self.positions.values()],
            "history": self.history[-5000:],
            "saved_at": datetime.utcnow().isoformat(),
        }
        tmp = self.state_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2, default=str)
        tmp.replace(self.state_path)

    # -----------------------------------------------------------------
    # Open position helpers
    # -----------------------------------------------------------------

    def _open_positions(self) -> List[BetPosition]:
        return [p for p in self.positions.values() if p.status == "open"]

    def _player_exposure(self, player: str) -> float:
        return sum(p.stake for p in self._open_positions() if p.player == player)

    def _team_exposure(self, team: str) -> float:
        return sum(p.stake for p in self._open_positions() if p.team == team)

    def _total_exposure(self) -> float:
        return sum(p.stake for p in self._open_positions())

    def _same_game_bets(self, game_id: str) -> List[BetPosition]:
        return [p for p in self._open_positions() if p.game_id == game_id]

    # -----------------------------------------------------------------
    # Drawdown / circuit breaker
    # -----------------------------------------------------------------

    @property
    def drawdown(self) -> float:
        if self.peak_bankroll <= 0:
            return 0.0
        return 1.0 - self.bankroll / self.peak_bankroll

    @property
    def circuit_breaker_active(self) -> bool:
        return self.drawdown >= self.drawdown_threshold

    @property
    def stake_multiplier(self) -> float:
        if self.circuit_breaker_active:
            return 0.5
        return 1.0

    def _update_peak(self):
        if self.bankroll > self.peak_bankroll:
            self.peak_bankroll = self.bankroll

    # -----------------------------------------------------------------
    # Kelly sizing
    # -----------------------------------------------------------------

    @staticmethod
    def _american_to_decimal(odds: int) -> float:
        if odds == 0:
            return 2.0
        if odds > 0:
            return 1.0 + odds / 100.0
        return 1.0 + 100.0 / abs(odds)

    def _kelly_stake(self, p_win: float, odds: int) -> float:
        """Quarter-Kelly stake as fraction of bankroll."""
        b = self._american_to_decimal(odds) - 1.0
        if b <= 0:
            return 0.0
        q = 1.0 - p_win
        f_full = (b * p_win - q) / b
        if f_full <= 0:
            return 0.0
        return self.kelly_fraction * f_full

    def _estimate_correlation(self, game_id: str, existing_bets: List[BetPosition]) -> float:
        """Estimate correlation between new bet and existing same-game bets.

        Same-game props share pace, game script, blowout risk.
        Heuristic: 0.15 base for same game, +0.10 per same-player prop,
        +0.05 per additional same-game bet (diminishing).
        """
        if not existing_bets:
            return 0.0

        same_game = [b for b in existing_bets if b.game_id == game_id]
        if not same_game:
            return 0.0

        base_corr = 0.15
        same_player_count = sum(1 for b in same_game if True)
        extra = min(len(same_game) * 0.05, 0.20)
        return min(base_corr + extra, 0.50)

    @staticmethod
    def round_stake(raw_stake: float) -> float:
        """Round to recreational-looking amounts."""
        if raw_stake <= 0:
            return 0.0
        if raw_stake < 10:
            return round(raw_stake)
        if raw_stake < 50:
            return round(raw_stake / 5) * 5
        if raw_stake < 200:
            return round(raw_stake / 25) * 25
        return round(raw_stake / 50) * 50

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def can_bet(self, player: str, team: str, game_id: str) -> dict:
        """Check if a bet passes exposure caps before sizing."""
        player_exp = self._player_exposure(player)
        team_exp = self._team_exposure(team)
        total_exp = self._total_exposure()

        player_cap = self.bankroll * self.max_player_pct
        team_cap = self.bankroll * self.max_team_pct
        total_cap = self.bankroll * self.max_total_pct

        reasons = []
        if player_exp >= player_cap:
            reasons.append(f"player_cap: ${player_exp:.0f}/${player_cap:.0f}")
        if team_exp >= team_cap:
            reasons.append(f"team_cap: ${team_exp:.0f}/${team_cap:.0f}")
        if total_exp >= total_cap:
            reasons.append(f"total_cap: ${total_exp:.0f}/${total_cap:.0f}")

        approved = len(reasons) == 0
        return {
            "approved": approved,
            "reason": "; ".join(reasons) if reasons else "ok",
            "remaining_player": max(player_cap - player_exp, 0),
            "remaining_team": max(team_cap - team_exp, 0),
            "remaining_total": max(total_cap - total_exp, 0),
            "circuit_breaker": self.circuit_breaker_active,
        }

    def size_bet(
        self,
        player: str,
        team: str,
        game_id: str,
        side: str,
        p_win: float,
        odds: int,
        edge: float,
        correlated_bets: Optional[List[BetPosition]] = None,
    ) -> dict:
        """Compute recommended stake with all adjustments applied."""
        if p_win <= 0 or p_win >= 1 or edge <= 0:
            return {"stake": 0, "approved": False, "reason": "no_edge"}

        kelly_frac = self._kelly_stake(p_win, odds)
        kelly_raw = kelly_frac * self.bankroll

        # Correlation adjustment
        if correlated_bets is None:
            correlated_bets = self._same_game_bets(game_id)
        corr = self._estimate_correlation(game_id, correlated_bets)
        kelly_adjusted = kelly_raw * (1.0 - corr / 2.0)

        # Drawdown circuit breaker
        kelly_adjusted *= self.stake_multiplier

        # Cap at 5% of bankroll for any single bet
        max_single = self.bankroll * 0.05
        capped_reason = None
        stake = kelly_adjusted

        if stake > max_single:
            stake = max_single
            capped_reason = "max_single_5pct"

        # Apply exposure caps
        player_remaining = self.bankroll * self.max_player_pct - self._player_exposure(player)
        team_remaining = self.bankroll * self.max_team_pct - self._team_exposure(team)
        total_remaining = self.bankroll * self.max_total_pct - self._total_exposure()

        if stake > player_remaining:
            stake = max(player_remaining, 0)
            capped_reason = "player_cap"
        if stake > team_remaining:
            stake = max(team_remaining, 0)
            capped_reason = "team_cap"
        if stake > total_remaining:
            stake = max(total_remaining, 0)
            capped_reason = "total_cap"

        stake = self.round_stake(stake)

        return {
            "stake": stake,
            "kelly_raw": round(kelly_raw, 2),
            "kelly_fraction": round(kelly_frac, 4),
            "kelly_adjusted": round(kelly_adjusted, 2),
            "correlation": round(corr, 3),
            "stake_multiplier": self.stake_multiplier,
            "capped_reason": capped_reason,
            "approved": stake > 0,
        }

    def place_bet(self, position: BetPosition) -> bool:
        """Record a placed bet. Returns False if it would violate caps."""
        check = self.can_bet(position.player, position.team, position.game_id)
        if not check["approved"]:
            logger.warning("Bet rejected: %s", check["reason"])
            return False

        if not position.bet_id:
            position.bet_id = str(uuid.uuid4())[:8]
        if not position.placed_at:
            position.placed_at = datetime.utcnow().isoformat()

        self.positions[position.bet_id] = position
        self._save_state()
        logger.info(
            "Placed: %s %s %s %.1f @ %d — $%.0f on %s",
            position.player, position.prop_type, position.side,
            position.line, position.odds, position.stake, position.book,
        )
        return True

    def resolve_bet(self, bet_id: str, result: str, actual: Optional[float] = None,
                    closing_line_odds: Optional[int] = None):
        """Mark a bet as won/lost/push and update bankroll."""
        pos = self.positions.get(bet_id)
        if not pos:
            logger.warning("Unknown bet_id: %s", bet_id)
            return

        pos.status = result
        pos.actual = actual
        pos.resolved_at = datetime.utcnow().isoformat()

        if result == "won":
            decimal = self._american_to_decimal(pos.odds)
            pos.profit = pos.stake * (decimal - 1.0)
            self.bankroll += pos.profit
        elif result == "lost":
            pos.profit = -pos.stake
            self.bankroll -= pos.stake
        else:
            pos.profit = 0.0

        # CLV calculation
        if closing_line_odds is not None:
            pos.closing_line_odds = closing_line_odds
            cl_decimal = self._american_to_decimal(closing_line_odds)
            placed_decimal = self._american_to_decimal(pos.odds)
            pos.clv = placed_decimal - cl_decimal

        self._update_peak()
        self.history.append(pos.to_dict())
        self._save_state()

        logger.info(
            "Resolved %s: %s — profit $%.2f (bankroll $%.2f, dd %.1f%%)",
            bet_id, result, pos.profit, self.bankroll, self.drawdown * 100,
        )

    def get_exposure(self) -> dict:
        """Current exposure summary."""
        open_bets = self._open_positions()

        by_player = defaultdict(float)
        by_team = defaultdict(float)
        for p in open_bets:
            by_player[p.player] += p.stake
            by_team[p.team] += p.stake

        total = self._total_exposure()
        return {
            "total_exposure": round(total, 2),
            "total_pct": round(total / max(self.bankroll, 1) * 100, 1),
            "num_open": len(open_bets),
            "by_player": dict(by_player),
            "by_team": dict(by_team),
            "player_cap": round(self.bankroll * self.max_player_pct, 2),
            "team_cap": round(self.bankroll * self.max_team_pct, 2),
            "total_cap": round(self.bankroll * self.max_total_pct, 2),
        }

    def get_bankroll_status(self) -> dict:
        """Full bankroll status."""
        return {
            "bankroll": round(self.bankroll, 2),
            "initial": round(self.initial_bankroll, 2),
            "peak": round(self.peak_bankroll, 2),
            "drawdown_pct": round(self.drawdown * 100, 1),
            "circuit_breaker_active": self.circuit_breaker_active,
            "stake_multiplier": self.stake_multiplier,
            "total_profit": round(self.bankroll - self.initial_bankroll, 2),
            "roi_pct": round(
                (self.bankroll - self.initial_bankroll) / max(self.initial_bankroll, 1) * 100, 2
            ),
            "bets_placed": len(self.history),
            "open_bets": len(self._open_positions()),
        }

    def set_bankroll(self, amount: float):
        """Manually adjust bankroll (deposit/withdrawal)."""
        self.bankroll = amount
        self._update_peak()
        self._save_state()


# =============================================================================
# Portfolio Analyzer
# =============================================================================

class PortfolioAnalyzer:
    """Computes performance analytics from bet history."""

    def __init__(self, history: List[dict]):
        self.history = [h for h in history if h.get("status") in ("won", "lost", "push")]

    def summary(self) -> dict:
        if not self.history:
            return {"error": "no_history"}

        stakes = [h["stake"] for h in self.history]
        profits = [h.get("profit", 0) for h in self.history]
        wins = sum(1 for h in self.history if h["status"] == "won")
        losses = sum(1 for h in self.history if h["status"] == "lost")

        total_wagered = sum(stakes)
        total_profit = sum(profits)

        return {
            "total_bets": len(self.history),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / max(wins + losses, 1),
            "total_wagered": round(total_wagered, 2),
            "total_profit": round(total_profit, 2),
            "roi_pct": round(total_profit / max(total_wagered, 1) * 100, 2),
            "avg_stake": round(np.mean(stakes), 2) if stakes else 0,
            "avg_profit": round(np.mean(profits), 2) if profits else 0,
            "max_drawdown": self.max_drawdown(),
            "sharpe_weekly": self.sharpe_ratio(),
        }

    def rolling_roi(self, days: int = 30) -> dict:
        """ROI over the last N days."""
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        recent = [h for h in self.history if h.get("resolved_at", h.get("placed_at", "")) >= cutoff]
        if not recent:
            return {"days": days, "bets": 0, "roi_pct": 0}
        wagered = sum(h["stake"] for h in recent)
        profit = sum(h.get("profit", 0) for h in recent)
        return {
            "days": days,
            "bets": len(recent),
            "wagered": round(wagered, 2),
            "profit": round(profit, 2),
            "roi_pct": round(profit / max(wagered, 1) * 100, 2),
        }

    def max_drawdown(self) -> float:
        """Maximum peak-to-trough drawdown from bet history."""
        if not self.history:
            return 0.0
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for h in sorted(self.history, key=lambda x: x.get("resolved_at", x.get("placed_at", ""))):
            cumulative += h.get("profit", 0)
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
        return round(max_dd, 2)

    def sharpe_ratio(self) -> float:
        """Weekly Sharpe ratio (annualized)."""
        if len(self.history) < 7:
            return 0.0

        by_week = defaultdict(float)
        for h in self.history:
            ts = h.get("resolved_at", h.get("placed_at", ""))
            if not ts:
                continue
            try:
                dt = datetime.fromisoformat(ts)
            except (ValueError, TypeError):
                continue
            week_key = dt.isocalendar()[:2]
            by_week[week_key] += h.get("profit", 0)

        if len(by_week) < 3:
            return 0.0

        returns = list(by_week.values())
        mean_r = np.mean(returns)
        std_r = np.std(returns, ddof=1)
        if std_r == 0:
            return 0.0
        return round(mean_r / std_r * np.sqrt(52), 2)

    def by_prop_type(self) -> dict:
        """Win rate and ROI by prop type."""
        groups = defaultdict(lambda: {"bets": 0, "wins": 0, "wagered": 0.0, "profit": 0.0})
        for h in self.history:
            pt = h.get("prop_type", "unknown")
            g = groups[pt]
            g["bets"] += 1
            if h["status"] == "won":
                g["wins"] += 1
            g["wagered"] += h["stake"]
            g["profit"] += h.get("profit", 0)
        result = {}
        for pt, g in groups.items():
            result[pt] = {
                "bets": g["bets"],
                "win_rate": round(g["wins"] / max(g["bets"], 1), 3),
                "roi_pct": round(g["profit"] / max(g["wagered"], 1) * 100, 2),
            }
        return result

    def by_book(self) -> dict:
        """Win rate and ROI by sportsbook."""
        groups = defaultdict(lambda: {"bets": 0, "wins": 0, "wagered": 0.0, "profit": 0.0})
        for h in self.history:
            bk = h.get("book", "unknown")
            g = groups[bk]
            g["bets"] += 1
            if h["status"] == "won":
                g["wins"] += 1
            g["wagered"] += h["stake"]
            g["profit"] += h.get("profit", 0)
        result = {}
        for bk, g in groups.items():
            result[bk] = {
                "bets": g["bets"],
                "win_rate": round(g["wins"] / max(g["bets"], 1), 3),
                "roi_pct": round(g["profit"] / max(g["wagered"], 1) * 100, 2),
            }
        return result

    def clv_summary(self) -> dict:
        """CLV tracking across all bets that have closing line data."""
        with_clv = [h for h in self.history if h.get("clv") is not None]
        if not with_clv:
            return {"n": 0}
        clvs = [h["clv"] for h in with_clv]
        positive = sum(1 for c in clvs if c > 0)
        return {
            "n": len(with_clv),
            "avg_clv": round(float(np.mean(clvs)), 4),
            "median_clv": round(float(np.median(clvs)), 4),
            "pct_positive": round(positive / len(with_clv), 3),
        }
