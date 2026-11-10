"""
push_alerts.py — Real-time edge alert notifications via AWS SNS or webhook.

Checks predictions from the 30-second edge recalc loop and sends alerts
when profitable edges are detected. Deduplicates and rate-limits to avoid
spam.

Configuration via environment:
    AIBALL_SNS_TOPIC_ARN   — AWS SNS topic ARN for push notifications
    AIBALL_WEBHOOK_URL     — Fallback webhook URL (POST with JSON body)
    AIBALL_ALERT_MIN_EDGE  — Minimum edge % to trigger alert (default 5.0)
    AIBALL_ALERT_MAX_PER_HOUR — Max alerts per hour (default 20)
    AIBALL_ALERT_DEDUP_MINUTES — Dedup window in minutes (default 30)
"""

import json
import logging
import os
import time
import urllib.request
import urllib.error
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
HISTORY_PATH = BASE_DIR / "data" / "alert_history.json"

PROP_LABELS = {
    "points": "pts",
    "rebounds": "reb",
    "assists": "ast",
    "threes": "3PM",
    "steals": "stl",
    "blocks": "blk",
    "turnovers": "TO",
    "pts_rebs_asts": "PRA",
    "pts_rebs": "P+R",
    "pts_asts": "P+A",
    "rebs_asts": "R+A",
    "stls_blks": "S+B",
}


@dataclass
class EdgeAlert:
    player: str
    prop_type: str
    side: str
    line: float
    edge_pct: float
    book: str
    odds: int
    consensus_prob: float
    model_prob: float
    blended_prob: float
    kelly_stake_pct: float
    game_info: str
    badges: List[str] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    @property
    def dedup_key(self) -> str:
        return f"{self.player}|{self.prop_type}|{self.side}|{self.line}"


class PushAlertManager:

    def __init__(
        self,
        sns_topic_arn: Optional[str] = None,
        webhook_url: Optional[str] = None,
        min_edge_pct: float = 5.0,
        max_alerts_per_hour: int = 20,
        dedup_window_minutes: int = 30,
    ):
        self.sns_topic_arn = sns_topic_arn or os.environ.get("AIBALL_SNS_TOPIC_ARN")
        self.webhook_url = webhook_url or os.environ.get("AIBALL_WEBHOOK_URL")
        self.min_edge_pct = float(os.environ.get("AIBALL_ALERT_MIN_EDGE", min_edge_pct))
        self.max_alerts_per_hour = int(os.environ.get("AIBALL_ALERT_MAX_PER_HOUR", max_alerts_per_hour))
        self.dedup_window = dedup_window_minutes * 60
        self._dedup: Dict[str, float] = {}  # dedup_key -> last_sent_ts
        self._hourly_log: deque = deque()    # timestamps of sent alerts
        self._lock = Lock()
        self._sns_client = None
        self._history: List[dict] = []
        self._last_flush = 0.0

        if self.sns_topic_arn:
            try:
                import boto3
                self._sns_client = boto3.client("sns")
                logger.info("SNS alerts enabled (topic: %s)", self.sns_topic_arn)
            except ImportError:
                logger.warning("boto3 not installed — SNS alerts disabled")
                self.sns_topic_arn = None
        if self.webhook_url:
            logger.info("Webhook alerts enabled (%s)", self.webhook_url)
        if not self.sns_topic_arn and not self.webhook_url:
            logger.info("No alert channels configured (set AIBALL_SNS_TOPIC_ARN or AIBALL_WEBHOOK_URL)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_and_alert(self, predictions: list) -> List[EdgeAlert]:
        """Screen predictions for alertable edges. Returns alerts that were sent."""
        sent: List[EdgeAlert] = []
        for pred in predictions:
            if not self._should_alert(pred):
                continue
            alert = self._format_alert(pred)
            if self._send(alert):
                sent.append(alert)
        if sent:
            self._maybe_flush_history()
        return sent

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _should_alert(self, pred: dict) -> bool:
        edge = abs(pred.get("edge_percent", 0))
        if edge < self.min_edge_pct:
            return False

        side = (pred.get("side") or "").lower()
        if not side:
            return False

        # Consensus must agree with model direction
        if side == "under" and pred.get("consensus_under_prob", 0) <= 0.50:
            return False
        if side == "over" and pred.get("consensus_over_prob", 0) <= 0.50:
            return False

        key = self._dedup_key(pred)
        now = time.time()

        with self._lock:
            # Dedup check
            if key in self._dedup and (now - self._dedup[key]) < self.dedup_window:
                return False

            # Rate limit
            cutoff = now - 3600
            while self._hourly_log and self._hourly_log[0] < cutoff:
                self._hourly_log.popleft()
            if len(self._hourly_log) >= self.max_alerts_per_hour:
                return False

        return True

    @staticmethod
    def _dedup_key(pred: dict) -> str:
        player = pred.get("player", pred.get("player_name", ""))
        return f"{player}|{pred.get('prop_type', '')}|{pred.get('side', '')}|{pred.get('line', 0)}"

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def _format_alert(self, pred: dict) -> EdgeAlert:
        player = pred.get("player", pred.get("player_name", ""))
        side = (pred.get("side") or "").upper()
        prop_type = pred.get("prop_type", "")

        # Determine recommended book
        book = "DraftKings"
        odds = pred.get("over_odds" if side == "OVER" else "under_odds", -110)
        ls = pred.get("line_shopping", [])
        if ls:
            best = max(ls, key=lambda x: x.get("over_odds" if side == "OVER" else "under_odds", -999))
            book = best.get("book", book)
            odds = best.get("over_odds" if side == "OVER" else "under_odds", odds)

        consensus_prob = (
            pred.get("consensus_over_prob", 0.5) if side == "OVER"
            else pred.get("consensus_under_prob", 0.5)
        )
        model_prob = pred.get("confidence", 0.5)
        if side == "OVER":
            model_prob = pred.get("p_over", model_prob)
        else:
            model_prob = 1 - pred.get("p_over", 1 - model_prob)

        blended = 0.70 * consensus_prob + 0.30 * model_prob

        kelly = self._quarter_kelly(blended, odds)

        badges = []
        if pred.get("soft_book_has_best"):
            badges.append("SOFT BOOK")
        if pred.get("is_star_player") and side == "UNDER":
            badges.append("STAR UNDER")
        if pred.get("line_range", 0) >= 1.0:
            badges.append("WIDE SPREAD")
        if pred.get("consensus_edge", 0) > 0.03:
            badges.append("CONSENSUS")

        game_info = pred.get("game_info", pred.get("matchup", ""))

        return EdgeAlert(
            player=player,
            prop_type=prop_type,
            side=side.lower(),
            line=pred.get("line", 0),
            edge_pct=abs(pred.get("edge_percent", 0)),
            book=book,
            odds=int(odds),
            consensus_prob=round(consensus_prob, 3),
            model_prob=round(model_prob, 3),
            blended_prob=round(blended, 3),
            kelly_stake_pct=round(kelly, 2),
            game_info=game_info,
            badges=badges,
        )

    @staticmethod
    def _quarter_kelly(prob: float, odds: int) -> float:
        if odds == 0:
            return 0.0
        if odds > 0:
            decimal = 1.0 + odds / 100.0
        else:
            decimal = 1.0 + 100.0 / abs(odds)
        b = decimal - 1.0
        q = 1.0 - prob
        if b <= 0:
            return 0.0
        f = (b * prob - q) / b
        return max(0.0, min(f * 0.25, 0.05)) * 100  # as percentage, capped at 5%

    def _format_message(self, alert: EdgeAlert) -> str:
        prop_label = PROP_LABELS.get(alert.prop_type, alert.prop_type)
        side_str = alert.side.upper()

        lines = [
            f"EDGE: {alert.player} {side_str} {alert.line} {prop_label}",
            f"Edge: {alert.edge_pct:.1f}% | {alert.book} {alert.odds:+d} | Kelly: {alert.kelly_stake_pct:.1f}%",
            f"Consensus: {alert.consensus_prob:.1%} | Model: {alert.model_prob:.1%} | Blend: {alert.blended_prob:.1%}",
        ]
        if alert.badges:
            lines.append("[" + "] [".join(alert.badges) + "]")
        if alert.game_info:
            lines.append(alert.game_info)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Delivery
    # ------------------------------------------------------------------

    def _send(self, alert: EdgeAlert) -> bool:
        now = time.time()
        with self._lock:
            self._dedup[alert.dedup_key] = now
            self._hourly_log.append(now)

        message = self._format_message(alert)
        success = False

        if self._sns_client and self.sns_topic_arn:
            success = self._send_sns(alert, message) or success
        if self.webhook_url:
            success = self._send_webhook(alert, message) or success

        if not self._sns_client and not self.webhook_url:
            logger.info("ALERT (no channel): %s", message.replace("\n", " | "))
            success = True

        if success:
            self._history.append(asdict(alert))
        return success

    def _send_sns(self, alert: EdgeAlert, message: str) -> bool:
        prop_label = PROP_LABELS.get(alert.prop_type, alert.prop_type)
        subject = f"Edge {alert.edge_pct:.0f}%: {alert.player} {alert.side.upper()} {alert.line} {prop_label}"
        if len(subject) > 100:
            subject = subject[:97] + "..."
        try:
            self._sns_client.publish(
                TopicArn=self.sns_topic_arn,
                Subject=subject,
                Message=message,
            )
            logger.info("SNS alert sent: %s", subject)
            return True
        except Exception as e:
            logger.error("SNS publish failed: %s", e)
            return False

    def _send_webhook(self, alert: EdgeAlert, message: str) -> bool:
        payload = json.dumps({"text": message, "alert": asdict(alert)}).encode()
        req = urllib.request.Request(
            self.webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                logger.info("Webhook alert sent (%d): %s %s %s",
                            resp.status, alert.player, alert.side, alert.prop_type)
                return True
        except (urllib.error.URLError, OSError) as e:
            logger.error("Webhook failed: %s", e)
            return False

    # ------------------------------------------------------------------
    # History persistence
    # ------------------------------------------------------------------

    def _maybe_flush_history(self):
        now = time.time()
        if now - self._last_flush < 300:
            return
        self._last_flush = now
        try:
            HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            existing = []
            if HISTORY_PATH.exists():
                with open(HISTORY_PATH) as f:
                    existing = json.load(f)
            existing.extend(self._history)
            # Keep last 7 days
            cutoff = (datetime.now(timezone.utc).timestamp() - 7 * 86400)
            existing = [
                a for a in existing
                if datetime.fromisoformat(a.get("timestamp", "2000-01-01")).timestamp() > cutoff
            ]
            with open(HISTORY_PATH, "w") as f:
                json.dump(existing, f, indent=2)
            self._history.clear()
            logger.debug("Flushed %d alerts to %s", len(existing), HISTORY_PATH)
        except Exception as e:
            logger.warning("Failed to flush alert history: %s", e)

    def get_recent_alerts(self, limit: int = 50) -> list:
        alerts = list(self._history)
        try:
            if HISTORY_PATH.exists():
                with open(HISTORY_PATH) as f:
                    alerts = json.load(f) + alerts
        except Exception:
            pass
        return alerts[-limit:]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup_dedup(self):
        now = time.time()
        with self._lock:
            expired = [k for k, ts in self._dedup.items() if now - ts > self.dedup_window]
            for k in expired:
                del self._dedup[k]


# ======================================================================
# Singleton
# ======================================================================

_manager: Optional[PushAlertManager] = None
_manager_lock = Lock()


def get_alert_manager() -> PushAlertManager:
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = PushAlertManager()
    return _manager
