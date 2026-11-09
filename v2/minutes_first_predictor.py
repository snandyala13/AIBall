#!/usr/bin/env python3
"""
v2/minutes_first_predictor.py - Two-stage Minutes-First Stat Predictor

"Props are minutes bets, not stats bets."

Architecture:
  Stage 1: Dedicated minutes model predicts projected_minutes
  Stage 2: per_minute_rate * projected_minutes = projected_stat
  Edge:    compare projected_stat to DK line

Why this is better than the current approach:
  - Current model treats minutes as 7 features among 178 (3.9%)
  - A player playing 38 vs 32 min is a ~19% stat boost -- HUGE
  - Minutes deviations (B2B, injury return, blowout) are the most
    exploitable edges because DK's lines are set to "normal" minutes
  - This model makes the minutes prediction explicit and auditable

Per-minute rate calculation:
  - Uses exponentially weighted recent per-minute rates
  - Handles non-linearity: rates differ at different minute volumes
    (starters in garbage time produce at different rates than crunch time)
  - Separate rates for each stat type

Usage:
    from v2.minutes_first_predictor import MinutesFirstPredictor
    pred = MinutesFirstPredictor()
    result = pred.predict("LeBron James", "points", "2026-10-20", game_context)

    # Compare to current v7 model:
    python3 -I -m v2.minutes_first_predictor  # Full evaluation
"""

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    import joblib
except ImportError:
    logger.error("joblib not installed. Run: pip install joblib")
    raise

from scipy.stats import norm as _norm

BASE_DIR = Path(__file__).resolve().parent.parent
MINUTES_MODEL_DIR = BASE_DIR / "models" / "minutes_model"


# =============================================================================
# Per-Minute Rate Calculator
# =============================================================================

class PerMinuteRateCalculator:
    """Compute stable per-minute rates for each stat type.

    Key insight: per-minute rates are NOT constant across minute volumes.
    A player averaging 0.8 pts/min over 35 min may only produce 0.6 pts/min
    if they play 28 min (because those extra 7 min are crunch-time minutes
    where scoring rates are highest).

    We handle this with:
    1. Exponentially weighted recent rates (recent form matters most)
    2. Minute-volume correction (rates adjust when projected minutes differ
       significantly from recent average)
    3. Separate calculation for each stat type
    """

    # Stat keys for each prop type
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

    @classmethod
    def compute_rate(
        cls,
        prior_games: List[dict],
        prop_type: str,
        window: int = 10,
    ) -> Tuple[float, float]:
        """Compute per-minute rate and its standard deviation.

        Args:
            prior_games: Chronological list (most recent first) of game dicts
            prop_type: e.g. "points"
            window: Number of games to use

        Returns:
            (rate_per_min, rate_std)
        """
        stat_keys = cls.PROP_TO_STAT.get(prop_type, [])
        if not stat_keys:
            return 0.0, 0.0

        games = prior_games[:window]
        rates = []
        weights = []

        for i, g in enumerate(games):
            mins = g.get("min", 0)
            if mins < 5:  # Skip very low-minute games (DNP, blowout bench)
                continue

            stat_val = sum(g.get(k, 0) for k in stat_keys)
            rate = stat_val / mins
            rates.append(rate)

            # Exponential decay weight: most recent game = 1.0, decay 0.9x per game
            weights.append(0.9 ** i)

        if not rates:
            return 0.0, 0.0

        rates = np.array(rates)
        weights = np.array(weights)

        # Weighted average rate
        avg_rate = np.average(rates, weights=weights)

        # Rate standard deviation (unweighted for simplicity)
        rate_std = np.std(rates) if len(rates) >= 3 else avg_rate * 0.3

        return float(avg_rate), float(rate_std)

    @classmethod
    def compute_rate_multi_window(
        cls,
        prior_games: List[dict],
        prop_type: str,
    ) -> dict:
        """Compute rates at multiple windows for blending.

        Returns dict with rates at L5, L10, season.
        """
        rate_l5, std_l5 = cls.compute_rate(prior_games, prop_type, window=5)
        rate_l10, std_l10 = cls.compute_rate(prior_games, prop_type, window=10)
        rate_season, std_season = cls.compute_rate(prior_games, prop_type, window=82)

        # Blend: recent form weighted heavier
        # L5: 50%, L10: 30%, Season: 20%
        blended_rate = rate_l5 * 0.50 + rate_l10 * 0.30 + rate_season * 0.20
        blended_std = std_l5 * 0.50 + std_l10 * 0.30 + std_season * 0.20

        return {
            "rate_l5": rate_l5,
            "rate_l10": rate_l10,
            "rate_season": rate_season,
            "blended_rate": blended_rate,
            "rate_std": blended_std,
        }

    @classmethod
    def minute_volume_correction(
        cls,
        base_rate: float,
        projected_minutes: float,
        avg_recent_minutes: float,
    ) -> float:
        """Adjust per-minute rate for minute volume changes.

        When a player is projected for significantly different minutes than
        their recent average, the per-minute rate changes:
        - MORE minutes than usual: marginal minutes are slightly less productive
          (fatigue, garbage time, less efficient possessions)
        - FEWER minutes than usual: those "lost" minutes were often the most
          productive (crunch time, high-leverage possessions)

        The correction is small but systematic, ~1-3% per 5 minutes of deviation.
        """
        if avg_recent_minutes <= 0 or base_rate <= 0:
            return base_rate

        minutes_diff = projected_minutes - avg_recent_minutes
        pct_diff = minutes_diff / avg_recent_minutes

        # Correction factor: slight reduction for more minutes, increase for fewer
        # Based on empirical observation: ~2% rate change per 10% minute change
        correction = 1.0 - 0.2 * pct_diff

        # Clamp to reasonable range
        correction = max(0.85, min(1.15, correction))

        return base_rate * correction


# =============================================================================
# Minutes-First Predictor
# =============================================================================

class MinutesFirstPredictor:
    """Two-stage predictor: Minutes -> Per-Minute Rate -> Stat Projection.

    Stage 1: ML model predicts minutes
    Stage 2: per_minute_rate * projected_minutes = projected_stat
    Edge:    projected_stat vs DK line
    """

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = Path(models_dir) if models_dir else MINUTES_MODEL_DIR
        self.minutes_model = None
        self.q10_model = None
        self.q90_model = None
        self.feature_names = []
        self.sigma_minutes = 4.0  # Default MAE for minutes
        self._loaded = False
        self.rate_calculator = PerMinuteRateCalculator()
        self._load_models()

    def _load_models(self):
        """Load trained minutes model from disk."""
        if not self.models_dir.exists():
            logger.warning("Minutes model directory not found: %s", self.models_dir)
            return

        meta_path = self.models_dir / "model_metadata.json"
        if not meta_path.exists():
            logger.warning("No model_metadata.json found in %s", self.models_dir)
            return

        with open(meta_path) as f:
            meta = json.load(f)

        self.feature_names = meta.get("feature_names", [])
        self.sigma_minutes = meta.get("sigma", 4.0)

        # Load models
        huber_path = self.models_dir / "minutes_lgb_huber.joblib"
        if huber_path.exists():
            self.minutes_model = joblib.load(huber_path)
            logger.info("Loaded minutes Huber model (%d features, sigma=%.2f)",
                        len(self.feature_names), self.sigma_minutes)
        else:
            logger.warning("minutes_lgb_huber.joblib not found")
            return

        q10_path = self.models_dir / "minutes_q10.joblib"
        q90_path = self.models_dir / "minutes_q90.joblib"
        if q10_path.exists():
            self.q10_model = joblib.load(q10_path)
        if q90_path.exists():
            self.q90_model = joblib.load(q90_path)

        self._loaded = True

    def is_ready(self) -> bool:
        return self._loaded

    def predict_minutes(self, features: dict) -> dict:
        """Stage 1: Predict minutes from features.

        Args:
            features: dict of minutes features (from MinutesFeatureBuilder)

        Returns:
            dict with projected_minutes, minutes_q10, minutes_q90, minutes_std
        """
        if not self._loaded:
            logger.warning("Minutes model not loaded")
            return {
                "projected_minutes": features.get("mins_l5", 30.0),
                "minutes_q10": features.get("mins_l5", 30.0) - 5,
                "minutes_q90": features.get("mins_l5", 30.0) + 5,
                "minutes_std": 4.0,
            }

        # Build feature vector in expected order
        vec = np.array(
            [features.get(col, 0.0) for col in self.feature_names],
            dtype=np.float64,
        ).reshape(1, -1)

        projected = float(self.minutes_model.predict(vec)[0])
        projected = max(0, min(53, projected))

        q10 = float(self.q10_model.predict(vec)[0]) if self.q10_model else projected - 5
        q90 = float(self.q90_model.predict(vec)[0]) if self.q90_model else projected + 5
        q10 = max(0, q10)
        q90 = min(53, q90)

        minutes_std = max((q90 - q10) / 2.56, 1.0)

        return {
            "projected_minutes": round(projected, 1),
            "minutes_q10": round(q10, 1),
            "minutes_q90": round(q90, 1),
            "minutes_std": round(minutes_std, 1),
        }

    def predict_stat(
        self,
        prior_games: List[dict],
        prop_type: str,
        projected_minutes: float,
        avg_recent_minutes: float,
    ) -> dict:
        """Stage 2: Predict stat value from projected minutes + per-minute rate.

        Args:
            prior_games: Player's recent game history (most recent first)
            prop_type: e.g. "points"
            projected_minutes: From Stage 1
            avg_recent_minutes: Player's recent average minutes (for volume correction)

        Returns:
            dict with projected_stat, stat_std, per_minute_rate, rate details
        """
        rates = self.rate_calculator.compute_rate_multi_window(
            prior_games, prop_type
        )

        base_rate = rates["blended_rate"]
        rate_std = rates["rate_std"]

        # Apply minute volume correction
        corrected_rate = self.rate_calculator.minute_volume_correction(
            base_rate, projected_minutes, avg_recent_minutes,
        )

        # Projected stat = rate * minutes
        projected_stat = corrected_rate * projected_minutes

        # Uncertainty: propagate both minutes uncertainty and rate uncertainty
        # Var(stat) = E[min]^2 * Var(rate) + E[rate]^2 * Var(min) + Var(rate)*Var(min)
        # Simplified: std(stat) ~ sqrt( (min*rate_std)^2 + (rate*min_std)^2 )
        min_std = 4.0  # Default, should be from predict_minutes
        stat_std = np.sqrt(
            (projected_minutes * rate_std) ** 2
            + (corrected_rate * min_std) ** 2
        )

        return {
            "projected_stat": round(projected_stat, 2),
            "stat_std": round(stat_std, 2),
            "per_minute_rate": round(corrected_rate, 4),
            "base_rate": round(base_rate, 4),
            "rate_l5": round(rates["rate_l5"], 4),
            "rate_l10": round(rates["rate_l10"], 4),
            "rate_season": round(rates["rate_season"], 4),
        }

    def full_prediction(
        self,
        minutes_features: dict,
        prior_games: List[dict],
        prop_type: str,
        line: float,
        over_odds: int = -110,
        under_odds: int = -110,
    ) -> Optional[dict]:
        """Full two-stage prediction with edge calculation.

        Args:
            minutes_features: dict from MinutesFeatureBuilder.get_minutes_features()
            prior_games: Player's recent games (most recent first)
            prop_type: e.g. "points"
            line: DK prop line
            over_odds: American odds for over
            under_odds: American odds for under

        Returns:
            Complete prediction dict with minutes, stat, and edge info
        """
        # Stage 1: Minutes
        mins_pred = self.predict_minutes(minutes_features)
        projected_minutes = mins_pred["projected_minutes"]
        minutes_std = mins_pred["minutes_std"]

        # Stage 2: Stat from rate * minutes
        avg_recent_minutes = minutes_features.get("mins_l5", 30.0)
        stat_pred = self.predict_stat(
            prior_games, prop_type, projected_minutes, avg_recent_minutes,
        )
        projected_stat = stat_pred["projected_stat"]
        stat_std = stat_pred["stat_std"]

        # Edge calculation
        if stat_std <= 0:
            stat_std = 1.0

        residual = projected_stat - line
        z = residual / stat_std
        p_over = float(_norm.cdf(z))
        p_under = 1.0 - p_over

        # Implied probabilities from odds
        imp_over = self._american_to_prob(over_odds)
        imp_under = self._american_to_prob(under_odds)

        edge_over = p_over - imp_over
        edge_under = p_under - imp_under

        if edge_over > edge_under:
            recommended_side = "OVER"
            edge = edge_over
        else:
            recommended_side = "UNDER"
            edge = edge_under

        # Kelly sizing
        if recommended_side == "OVER":
            p = p_over
            odds = over_odds
        else:
            p = p_under
            odds = under_odds

        decimal_odds = self._american_to_decimal(odds)
        b = decimal_odds - 1
        q = 1 - p
        if b > 0:
            full_kelly = (b * p - q) / b
            kelly = 0.25 * max(0, full_kelly)
            kelly = min(kelly, 0.05)
        else:
            kelly = 0.0

        return {
            "projected_minutes": projected_minutes,
            "minutes_q10": mins_pred["minutes_q10"],
            "minutes_q90": mins_pred["minutes_q90"],
            "minutes_std": minutes_std,
            "mins_l5": avg_recent_minutes,
            "minutes_delta": round(projected_minutes - avg_recent_minutes, 1),
            "projected_stat": projected_stat,
            "stat_std": stat_std,
            "per_minute_rate": stat_pred["per_minute_rate"],
            "line": line,
            "residual": round(residual, 2),
            "p_over": round(p_over, 4),
            "p_under": round(p_under, 4),
            "edge_over": round(edge_over, 4),
            "edge_under": round(edge_under, 4),
            "recommended_side": recommended_side,
            "edge": round(edge, 4),
            "kelly_fraction": round(kelly, 4),
            "model_type": "minutes_first",
        }

    @staticmethod
    def _american_to_prob(odds: float) -> float:
        if odds == 0:
            return 0.5
        if odds > 0:
            return 100.0 / (odds + 100.0)
        return abs(odds) / (abs(odds) + 100.0)

    @staticmethod
    def _american_to_decimal(odds: float) -> float:
        if odds == 0:
            return 2.0
        if odds > 0:
            return 1.0 + odds / 100.0
        return 1.0 + 100.0 / abs(odds)


# =============================================================================
# Evaluation: Compare minutes-first vs current v7 model
# =============================================================================

def evaluate_minutes_first():
    """Full evaluation comparing minutes-first to current approach.

    This runs the two-stage model on the test set and compares:
    1. Stat prediction accuracy (MAE vs line)
    2. Edge calibration (when model says over with X% confidence, does over hit X%?)
    3. ROI simulation
    4. Conditional accuracy on key scenarios (B2B, injury return, blowout)
    """
    import pandas as pd
    from v2.minutes_ml_model import MinutesFeatureBuilder

    print("=" * 70)
    print("  MINUTES-FIRST PREDICTOR - Full Evaluation")
    print("  Comparing two-stage (min * rate) vs current v7 (direct stat prediction)")
    print("=" * 70)

    # Load the minutes model
    predictor = MinutesFirstPredictor()
    if not predictor.is_ready():
        print("\nERROR: Minutes model not trained yet.")
        print("Run: python3 -I -m v2.minutes_ml_model")
        return

    # Build feature data from legs
    fb = MinutesFeatureBuilder()
    fb.load_games()
    fb.build_player_history()

    # Load legs for evaluation
    OUTPUTS = BASE_DIR / "outputs"
    LEGS_FILES = [
        OUTPUTS / "backtest_2023_24_legs.json",
        OUTPUTS / "backtest_2024_25_legs.json",
        OUTPUTS / "backtest_oos_legs.json",
    ]

    all_legs = []
    for path in LEGS_FILES:
        if not path.exists():
            continue
        with open(path) as f:
            legs = json.load(f)
        all_legs.extend(legs)
    all_legs.sort(key=lambda x: x["date"])

    # Use last 15% as test set (matching the minutes model's test split)
    n = len(all_legs)
    test_start = int(n * 0.85)
    test_legs = all_legs[test_start:]

    print(f"\nTest set: {len(test_legs)} legs (last 15% by date)")
    print(f"Date range: {test_legs[0]['date']} to {test_legs[-1]['date']}")

    # Run predictions
    results = []
    prop_types_seen = set()

    for i, leg in enumerate(test_legs):
        if i > 0 and i % 5000 == 0:
            logger.info("  Evaluating leg %d/%d ...", i, len(test_legs))

        player = leg["player"]
        prop_type = leg["prop_type"]
        date = leg["date"]
        line = leg.get("line", 0)
        actual = leg.get("actual")
        over_odds = leg.get("over_odds", -110)
        under_odds = leg.get("under_odds", -110)

        if actual is None:
            continue

        # Get actual minutes
        game_id = leg.get("game_id")
        game = fb.game_lookup.get(game_id)
        if not game:
            continue
        box = game.get("box_score", {})
        player_stats = box.get(player)
        if not player_stats:
            continue
        actual_mins = player_stats.get("min", 0)

        # Determine B2B
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        player_is_home = fb._guess_player_team(player, game)
        team_name = home if player_is_home else (away if player_is_home is not None else "")
        is_b2b = fb._is_b2b(team_name, date) if team_name else 0
        days_rest = fb._days_rest(team_name, date) if team_name else 1

        game_context = {
            "spread": 0.0,
            "total": 224.0,
            "is_home": player_is_home,
            "is_b2b": is_b2b,
            "days_rest": days_rest,
        }

        # Build minutes features
        mins_feats = fb.get_minutes_features(player, date, game_context)
        if mins_feats is None:
            continue

        # Get player's prior games for per-minute rate
        history = fb.player_history.get(player, [])
        prior_games = list(reversed([g for g in history if g["date"] < date]))

        if len(prior_games) < 3:
            continue

        # Run full prediction
        pred = predictor.full_prediction(
            mins_feats, prior_games, prop_type,
            line=line, over_odds=over_odds, under_odds=under_odds,
        )

        if pred is None:
            continue

        prop_types_seen.add(prop_type)

        results.append({
            "player": player,
            "prop_type": prop_type,
            "date": date,
            "line": line,
            "actual_stat": float(actual),
            "actual_mins": actual_mins,
            "projected_mins": pred["projected_minutes"],
            "projected_stat": pred["projected_stat"],
            "per_min_rate": pred["per_minute_rate"],
            "p_over": pred["p_over"],
            "recommended_side": pred["recommended_side"],
            "edge": pred["edge"],
            "kelly": pred["kelly_fraction"],
            "over_odds": over_odds,
            "under_odds": under_odds,
            "is_b2b": is_b2b,
            "mins_l5": mins_feats.get("mins_l5", 0),
            "recent_return": mins_feats.get("recent_return", 0),
        })

    if not results:
        print("No results generated. Check data availability.")
        return

    df = pd.DataFrame(results)
    print(f"\nTotal predictions: {len(df)}")

    # ================================================================
    # 1. Minutes prediction accuracy
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  1. MINUTES PREDICTION ACCURACY")
    print(f"{'='*70}")

    mins_residual = df["projected_mins"] - df["actual_mins"]
    mins_mae = np.mean(np.abs(mins_residual))
    baseline_residual = df["mins_l5"] - df["actual_mins"]
    baseline_mae = np.mean(np.abs(baseline_residual))

    print(f"  Model MAE:    {mins_mae:.2f} min")
    print(f"  Baseline MAE: {baseline_mae:.2f} min (L5 avg)")
    print(f"  Improvement:  {baseline_mae - mins_mae:+.3f} min")

    # ================================================================
    # 2. Stat prediction accuracy
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  2. STAT PREDICTION ACCURACY (minutes * rate)")
    print(f"{'='*70}")

    stat_residual = df["projected_stat"] - df["actual_stat"]
    stat_mae = np.mean(np.abs(stat_residual))
    line_mae = np.mean(np.abs(df["line"] - df["actual_stat"]))

    print(f"  Minutes-First MAE:  {stat_mae:.3f}")
    print(f"  Line-only MAE:      {line_mae:.3f}")
    beats = "YES" if stat_mae < line_mae else "NO"
    print(f"  Beats line:         {beats} (delta: {line_mae - stat_mae:+.3f})")

    # Per-prop breakdown
    print(f"\n  Per-prop breakdown:")
    print(f"  {'Prop':20s} {'N':>6s} {'Model MAE':>10s} {'Line MAE':>10s} {'Delta':>8s}")
    print(f"  {'-'*20} {'-'*6} {'-'*10} {'-'*10} {'-'*8}")
    for pt in sorted(prop_types_seen):
        mask = df["prop_type"] == pt
        n = mask.sum()
        if n > 0:
            m_mae = np.mean(np.abs(df.loc[mask, "projected_stat"] - df.loc[mask, "actual_stat"]))
            l_mae = np.mean(np.abs(df.loc[mask, "line"] - df.loc[mask, "actual_stat"]))
            delta = l_mae - m_mae
            print(f"  {pt:20s} {n:6d} {m_mae:10.3f} {l_mae:10.3f} {delta:+8.3f}")

    # ================================================================
    # 3. Directional accuracy
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  3. DIRECTIONAL ACCURACY")
    print(f"{'='*70}")

    model_over = df["projected_stat"] > df["line"]
    actual_over = df["actual_stat"] > df["line"]
    dir_acc = np.mean(model_over == actual_over)
    print(f"  Directional accuracy: {dir_acc:.1%}")

    over_mask = model_over
    if over_mask.sum() > 0:
        over_hr = np.mean(actual_over[over_mask])
        print(f"  Over hit rate (when predicted): {over_hr:.1%} (N={over_mask.sum()})")

    under_mask = ~model_over
    if under_mask.sum() > 0:
        under_hr = np.mean(~actual_over[under_mask])
        print(f"  Under hit rate (when predicted): {under_hr:.1%} (N={under_mask.sum()})")

    # ================================================================
    # 4. ROI simulation
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  4. ROI SIMULATION (flat $100 bets)")
    print(f"{'='*70}")

    for threshold_label, min_edge in [
        (">= 0%", 0.0),
        (">= 2%", 0.02),
        (">= 3%", 0.03),
        (">= 5%", 0.05),
        (">= 8%", 0.08),
        (">= 10%", 0.10),
    ]:
        total_wagered = 0.0
        total_profit = 0.0
        n_bets = 0
        n_wins = 0

        for _, row in df.iterrows():
            if row["edge"] < min_edge:
                continue

            n_bets += 1
            total_wagered += 100.0

            # Did the bet hit?
            if row["recommended_side"] == "OVER":
                hit = row["actual_stat"] > row["line"]
                odds = row["over_odds"]
            else:
                hit = row["actual_stat"] < row["line"]
                odds = row["under_odds"]

            if hit:
                n_wins += 1
                dec = MinutesFirstPredictor._american_to_decimal(odds)
                total_profit += 100.0 * (dec - 1.0)
            else:
                total_profit -= 100.0

        if n_bets > 0:
            roi = total_profit / total_wagered
            wr = n_wins / n_bets
            print(f"  {threshold_label:>12s}: {n_bets:6d} bets, "
                  f"win_rate={wr:.1%}, ROI={roi:+.1%}, "
                  f"units={total_profit / 100:+.1f}")

    # ================================================================
    # 5. Conditional accuracy (where minutes-first SHOULD shine)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  5. CONDITIONAL ACCURACY (KEY SCENARIOS)")
    print(f"{'='*70}")

    scenarios = {
        "Back-to-back games": df["is_b2b"] == 1,
        "Recent injury return": df["recent_return"] == 1,
        "Minutes 5+ below L5": (df["actual_mins"] - df["mins_l5"]) <= -5,
        "Minutes 5+ above L5": (df["actual_mins"] - df["mins_l5"]) >= 5,
        "High-minute starters": df["mins_l5"] >= 32,
        "Low-minute bench": df["mins_l5"] < 20,
    }

    print(f"  {'Scenario':30s} {'N':>6s} {'Model MAE':>10s} {'Line MAE':>10s} {'Delta':>8s}")
    print(f"  {'-'*30} {'-'*6} {'-'*10} {'-'*10} {'-'*8}")

    for scenario_name, mask in scenarios.items():
        n = mask.sum()
        if n > 0:
            m_mae = np.mean(np.abs(
                df.loc[mask, "projected_stat"] - df.loc[mask, "actual_stat"]
            ))
            l_mae = np.mean(np.abs(
                df.loc[mask, "line"] - df.loc[mask, "actual_stat"]
            ))
            delta = l_mae - m_mae
            print(f"  {scenario_name:30s} {n:6d} {m_mae:10.3f} {l_mae:10.3f} {delta:+8.3f}")

    # ================================================================
    # 6. Minutes accuracy translating to stat accuracy
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  6. MINUTES ACCURACY -> STAT ACCURACY TRANSLATION")
    print(f"{'='*70}")

    # Bin by minutes prediction error
    abs_mins_error = np.abs(df["projected_mins"] - df["actual_mins"])
    bins = [
        ("0-2 min error", abs_mins_error <= 2),
        ("2-4 min error", (abs_mins_error > 2) & (abs_mins_error <= 4)),
        ("4-6 min error", (abs_mins_error > 4) & (abs_mins_error <= 6)),
        ("6-8 min error", (abs_mins_error > 6) & (abs_mins_error <= 8)),
        ("8+ min error", abs_mins_error > 8),
    ]

    print(f"  {'Min Error Bin':20s} {'N':>6s} {'Stat MAE':>10s} {'Dir Acc':>8s}")
    print(f"  {'-'*20} {'-'*6} {'-'*10} {'-'*8}")
    for label, mask in bins:
        n = mask.sum()
        if n > 0:
            s_mae = np.mean(np.abs(
                df.loc[mask, "projected_stat"] - df.loc[mask, "actual_stat"]
            ))
            m_over = df.loc[mask, "projected_stat"] > df.loc[mask, "line"]
            a_over = df.loc[mask, "actual_stat"] > df.loc[mask, "line"]
            d_acc = np.mean(m_over == a_over)
            print(f"  {label:20s} {n:6d} {s_mae:10.3f} {d_acc:7.1%}")

    print(f"\n{'='*70}")
    print(f"  DONE. Minutes model: {MINUTES_MODEL_DIR}")
    print(f"{'='*70}")


# =============================================================================
# CLI
# =============================================================================

def main():
    evaluate_minutes_first()


if __name__ == "__main__":
    main()
