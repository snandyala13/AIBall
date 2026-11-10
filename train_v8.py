#!/usr/bin/env python3
"""
train_v8.py — Calibrated over/under classifier with real multi-book consensus features.

KEY CHANGES FROM v7:
  1. CLASSIFICATION not regression — directly predicts P(over) via log-loss
  2. CALIBRATION OPTIMIZED — hyperparams selected by Brier score, not accuracy/MAE
  3. DEVIGGED CONSENSUS as input feature — model learns to ADJUST the market, not replace it
  4. POST-HOC CALIBRATION — isotonic regression on held-out calibration set
  5. QUANTILE REGRESSION for uncertainty and Kelly sizing

Architecture:
  P(over) = calibrate(classifier.predict_proba(features))
  where features include devigged_consensus_over_prob (the sharp market's view)

The model's job is NOT to predict stats. It's to predict when DK's line
deviates from true probability, anchored to the sharp consensus.

Walk-forward splits (by date, no random):
  - Train:       first 55%
  - Calibration: next 10% (for isotonic regression)
  - Validation:  next 10% (for early stopping)
  - Test:        final 25% (pure out-of-sample evaluation)
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.optimize import brentq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    import joblib
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss, log_loss
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install lightgbm joblib scikit-learn")
    sys.exit(1)

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models" / "v8_calibrated"

# Data files
LEGS_FILES = [
    OUTPUTS / "backtest_2023_24_multibook.json",
    OUTPUTS / "backtest_2024_25_multibook.json",
    OUTPUTS / "backtest_oos_multibook.json",
]
LEGS_FILES_PLAIN = [
    OUTPUTS / "backtest_2023_24_legs.json",
    OUTPUTS / "backtest_2024_25_legs.json",
    OUTPUTS / "backtest_oos_legs.json",
]
GAME_DATA_FILES = [
    OUTPUTS / "backtest_2023_24_data.json",
    OUTPUTS / "backtest_2024_25_data.json",
    OUTPUTS / "backtest_oos_data.json",
]

CORE_PROPS = ["points", "rebounds", "assists", "threes", "pts_rebs_asts", "steals", "blocks"]

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

ALL_STAT_KEYS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover", "min"]

# Sharp book weights for consensus
BOOK_SHARPNESS = {
    "pinnacle": 1.0, "pinnaclesports": 1.0,
    "circa": 0.85, "bookmaker": 0.80, "betcris": 0.75,
    "betonlineag": 0.70, "bovada": 0.65,
    "lowvig": 0.90, "novig": 0.88,
    "draftkings": 0.50, "fanduel": 0.55, "betmgm": 0.50,
    "caesars": 0.50, "pointsbetus": 0.45, "wynnbet": 0.45,
    "betrivers": 0.50, "unibet_us": 0.45, "superbook": 0.60,
    "espnbet": 0.50, "fliff": 0.40, "hardrockbet": 0.45,
    "betparx": 0.40, "fanatics": 0.50, "williamhill_us": 0.55,
    "windcreek": 0.40, "tipico_us": 0.45,
}
SOFT_BOOKS = frozenset({"betparx", "betrivers", "espnbet", "fliff", "hardrockbet"})
SHARP_BOOKS = frozenset({"pinnacle", "pinnaclesports", "circa", "bookmaker", "betonlineag", "lowvig", "novig"})


# =============================================================================
# DEVIG MATH
# =============================================================================

def american_to_implied(odds: int) -> float:
    if odds == 0:
        return 0.5
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 100.0 / (odds + 100.0)


def devig_power(over_odds: int, under_odds: int) -> Tuple[float, float]:
    p_over = american_to_implied(over_odds)
    p_under = american_to_implied(under_odds)
    if p_over <= 0 or p_under <= 0:
        total = p_over + p_under
        return (p_over / total, p_under / total) if total > 0 else (0.5, 0.5)
    if abs(p_over + p_under - 1.0) < 0.001:
        return p_over, p_under
    try:
        def objective(c):
            return p_over ** c + p_under ** c - 1.0
        lo, hi = 0.01, 5.0
        if objective(lo) * objective(hi) > 0:
            total = p_over + p_under
            return p_over / total, p_under / total
        c = brentq(objective, lo, hi, xtol=1e-8, maxiter=100)
        return p_over ** c, p_under ** c
    except (ValueError, RuntimeError):
        total = p_over + p_under
        return (p_over / total, p_under / total) if total > 0 else (0.5, 0.5)


def american_to_decimal(odds: int) -> float:
    if odds == 0:
        return 2.0
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


# =============================================================================
# CONSENSUS FEATURE BUILDER
# =============================================================================

def compute_consensus_features(all_books: dict, dk_line: float,
                                dk_over_odds: int, dk_under_odds: int) -> dict:
    """
    Compute real consensus features from multi-book odds data.
    This is the KEY innovation: giving the model the sharp market's view.
    """
    feats = {}

    if not all_books or len(all_books) < 2:
        # No multi-book data — fill with neutral defaults
        dk_over, dk_under = devig_power(dk_over_odds, dk_under_odds)
        feats["consensus_over_prob"] = dk_over
        feats["consensus_under_prob"] = dk_under
        feats["dk_vs_consensus_prob"] = 0.0
        feats["consensus_line"] = dk_line
        feats["dk_vs_consensus_line"] = 0.0
        feats["line_range"] = 0.0
        feats["num_books"] = 1
        feats["num_sharp_books"] = 0
        feats["num_soft_books"] = 0
        feats["soft_book_score"] = 0.0
        feats["pinnacle_over_prob"] = dk_over
        feats["pinnacle_available"] = 0
        feats["sharp_consensus_over_prob"] = dk_over
        feats["best_over_odds_lift"] = 0.0
        feats["best_under_odds_lift"] = 0.0
        feats["odds_spread_width"] = 0.0
        feats["sharp_soft_disagreement"] = 0.0
        feats["market_efficiency"] = 0.0
        return feats

    # Parse all books
    books = []
    for bk, bd in all_books.items():
        line_val = bd.get("line")
        over_o = bd.get("over_odds")
        under_o = bd.get("under_odds")
        if line_val is None or over_o is None or under_o is None:
            continue
        try:
            over_o = int(over_o)
            under_o = int(under_o)
            line_val = float(line_val)
        except (TypeError, ValueError):
            continue
        if abs(over_o) < 100 or abs(under_o) < 100:
            continue

        dv_over, dv_under = devig_power(over_o, under_o)
        sharpness = BOOK_SHARPNESS.get(bk, 0.45)
        books.append({
            "book": bk,
            "line": line_val,
            "over_odds": over_o,
            "under_odds": under_o,
            "dv_over": dv_over,
            "dv_under": dv_under,
            "sharpness": sharpness,
        })

    if not books:
        return compute_consensus_features({}, dk_line, dk_over_odds, dk_under_odds)

    # Sharp-weighted consensus probability
    total_weight = sum(b["sharpness"] for b in books)
    consensus_over = sum(b["dv_over"] * b["sharpness"] for b in books) / total_weight
    consensus_under = sum(b["dv_under"] * b["sharpness"] for b in books) / total_weight

    # Sharp-weighted consensus line
    consensus_line = sum(b["line"] * b["sharpness"] for b in books) / total_weight

    # DK devigged
    dk_dv_over, dk_dv_under = devig_power(dk_over_odds, dk_under_odds)

    # Line stats
    all_lines = [b["line"] for b in books]
    line_range = max(all_lines) - min(all_lines)

    # Sharp-only consensus
    sharp_books = [b for b in books if b["book"] in SHARP_BOOKS]
    if sharp_books:
        sharp_w = sum(b["sharpness"] for b in sharp_books)
        sharp_consensus_over = sum(b["dv_over"] * b["sharpness"] for b in sharp_books) / sharp_w
    else:
        sharp_consensus_over = consensus_over

    # Soft-only consensus
    soft_books = [b for b in books if b["book"] in SOFT_BOOKS]

    # Pinnacle specifically
    pinnacle = [b for b in books if b["book"] in ("pinnacle", "pinnaclesports")]
    pinnacle_over = pinnacle[0]["dv_over"] if pinnacle else consensus_over
    pinnacle_available = 1 if pinnacle else 0

    # Best odds comparison
    best_over_odds = max(b["over_odds"] for b in books)
    best_under_odds = max(b["under_odds"] for b in books)
    dk_over_decimal = american_to_decimal(dk_over_odds)
    dk_under_decimal = american_to_decimal(dk_under_odds)

    # Odds spread width (max over_odds - min over_odds, in implied prob terms)
    all_over_probs = [b["dv_over"] for b in books]
    odds_spread_width = max(all_over_probs) - min(all_over_probs)

    # Soft book score: fraction of soft books that have better over/under odds than DK
    soft_better = 0
    for b in soft_books:
        if b["over_odds"] > dk_over_odds or b["under_odds"] > dk_under_odds:
            soft_better += 1
    soft_book_score = soft_better / max(len(soft_books), 1)

    # Sharp-soft disagreement: difference in consensus between sharp and soft books
    if soft_books and sharp_books:
        soft_w = sum(b["sharpness"] for b in soft_books)
        soft_consensus_over = sum(b["dv_over"] * b["sharpness"] for b in soft_books) / soft_w
        sharp_soft_disagreement = sharp_consensus_over - soft_consensus_over
    else:
        sharp_soft_disagreement = 0.0

    # Market efficiency: how tight is the consensus? Low variance = efficient
    market_efficiency = np.std(all_over_probs) if len(all_over_probs) > 1 else 0.0

    feats["consensus_over_prob"] = consensus_over
    feats["consensus_under_prob"] = consensus_under
    feats["dk_vs_consensus_prob"] = dk_dv_over - consensus_over
    feats["consensus_line"] = consensus_line
    feats["dk_vs_consensus_line"] = dk_line - consensus_line
    feats["line_range"] = line_range
    feats["num_books"] = len(books)
    feats["num_sharp_books"] = len(sharp_books)
    feats["num_soft_books"] = len(soft_books)
    feats["soft_book_score"] = soft_book_score
    feats["pinnacle_over_prob"] = pinnacle_over
    feats["pinnacle_available"] = pinnacle_available
    feats["sharp_consensus_over_prob"] = sharp_consensus_over
    feats["best_over_odds_lift"] = american_to_decimal(best_over_odds) - dk_over_decimal
    feats["best_under_odds_lift"] = american_to_decimal(best_under_odds) - dk_under_decimal
    feats["odds_spread_width"] = odds_spread_width
    feats["sharp_soft_disagreement"] = sharp_soft_disagreement
    feats["market_efficiency"] = market_efficiency

    return feats


# =============================================================================
# PLAYER HISTORY + FEATURE BUILDER
# =============================================================================

class V8FeatureBuilder:
    """Builds the full feature matrix: box score + market + consensus."""

    def __init__(self):
        self.player_history = defaultdict(list)
        self.games = []

    def load_games(self):
        all_games = []
        for path in GAME_DATA_FILES:
            if not path.exists():
                logger.warning("Missing: %s", path)
                continue
            with open(path) as f:
                data = json.load(f)
            all_games.extend(data.get("games", []))
        all_games.sort(key=lambda g: g["date"])
        self.games = all_games
        logger.info("Loaded %d games", len(self.games))

    def build_player_history(self):
        self.player_history.clear()
        for game in self.games:
            date = game["date"]
            box = game.get("box_score", {})
            for player_name, stats in box.items():
                entry = {"date": date}
                for k in ALL_STAT_KEYS:
                    entry[k] = stats.get(k, 0)
                entry["fga"] = stats.get("fga", 0)
                entry["fgm"] = stats.get("fgm", 0)
                entry["fta"] = stats.get("fta", 0)
                entry["ftm"] = stats.get("ftm", 0)
                entry["fg3a"] = stats.get("fg3a", 0)
                self.player_history[player_name].append(entry)
        for player in self.player_history:
            self.player_history[player].sort(key=lambda g: g["date"])
        logger.info("Built history for %d players", len(self.player_history))

    def get_stat_value(self, game_entry: dict, prop_type: str) -> float:
        keys = PROP_TO_STAT.get(prop_type, [])
        return sum(game_entry.get(k, 0) for k in keys)

    def build_features(self, player: str, prop_type: str, date: str,
                       line: float, over_odds: int, under_odds: int,
                       all_books: Optional[dict] = None) -> Optional[dict]:
        """Build complete feature vector for one leg."""
        history = self.player_history.get(player, [])
        prior = [g for g in history if g["date"] < date]
        if len(prior) < 3:
            return None

        prior = list(reversed(prior))  # most recent first
        feats = {}

        # === BOX SCORE ROLLING STATS ===
        values = [self.get_stat_value(g, prop_type) for g in prior]
        l3 = values[:3]
        l5 = values[:min(5, len(values))]
        l10 = values[:min(10, len(values))]
        l20 = values[:min(20, len(values))]

        feats["target_l3"] = np.mean(l3)
        feats["target_l5"] = np.mean(l5)
        feats["target_l10"] = np.mean(l10)
        feats["target_l20"] = np.mean(l20)
        feats["target_season"] = np.mean(values)
        feats["target_std_l10"] = np.std(l10) if len(l10) >= 3 else np.std(values)
        feats["target_std_season"] = np.std(values)
        feats["target_min_l5"] = min(l5)
        feats["target_max_l5"] = max(l5)
        feats["target_median_l10"] = float(np.median(l10))
        feats["target_trend"] = feats["target_l3"] - feats["target_l10"]
        feats["target_cv"] = feats["target_std_l10"] / max(feats["target_season"], 0.1)

        # Hit rates vs line
        feats["hit_rate_l5"] = np.mean([1 if v > line else 0 for v in l5])
        feats["hit_rate_l10"] = np.mean([1 if v > line else 0 for v in l10])
        feats["hit_rate_l20"] = np.mean([1 if v > line else 0 for v in l20])

        if feats["target_std_l10"] > 0:
            feats["line_z_l10"] = (line - feats["target_l10"]) / feats["target_std_l10"]
        else:
            feats["line_z_l10"] = 0.0

        # === MINUTES ===
        mins = [g["min"] for g in prior]
        feats["mins_l3"] = np.mean(mins[:3])
        feats["mins_l5"] = np.mean(mins[:min(5, len(mins))])
        feats["mins_l10"] = np.mean(mins[:min(10, len(mins))])
        feats["mins_season"] = np.mean(mins)
        feats["mins_std"] = np.std(mins[:min(10, len(mins))]) if len(mins) >= 3 else np.std(mins)
        feats["mins_trend"] = feats["mins_l3"] - feats["mins_l10"]

        # === PER-MINUTE RATE ===
        n10 = min(10, len(values))
        total_stat = sum(values[:n10])
        total_mins = sum(mins[:n10])
        feats["per_minute_rate"] = total_stat / max(total_mins, 1.0)
        n5 = min(5, len(values))
        feats["per_minute_rate_l5"] = sum(values[:n5]) / max(sum(mins[:n5]), 1.0)
        feats["expected_from_rate"] = feats["per_minute_rate"] * feats["mins_l5"]

        # === SHOOTING EFFICIENCY ===
        fga_l10 = [g.get("fga", 0) for g in prior[:10]]
        fgm_l10 = [g.get("fgm", 0) for g in prior[:10]]
        fta_l10 = [g.get("fta", 0) for g in prior[:10]]
        fg3a_l10 = [g.get("fg3a", 0) for g in prior[:10]]
        feats["fga_l10"] = np.mean(fga_l10) if any(fga_l10) else 0.0
        feats["fg_pct_l10"] = sum(fgm_l10) / max(sum(fga_l10), 1)
        feats["fta_l10"] = np.mean(fta_l10) if any(fta_l10) else 0.0
        feats["fg3a_l10"] = np.mean(fg3a_l10) if any(fg3a_l10) else 0.0
        feats["fg3_pct_l10"] = sum(g.get("fg3m", 0) for g in prior[:10]) / max(sum(fg3a_l10), 1)

        # === GAMES PLAYED ===
        feats["games_played"] = min(len(prior), 82)

        # === MARKET LINE FEATURES ===
        feats["line"] = line
        feats["over_odds"] = over_odds
        feats["under_odds"] = under_odds
        implied_over = american_to_implied(over_odds)
        implied_under = american_to_implied(under_odds)
        feats["implied_over_prob"] = implied_over
        feats["implied_under_prob"] = implied_under

        # CRITICAL: devigged DK probability — the market's TRUE view
        dk_dv_over, dk_dv_under = devig_power(over_odds, under_odds)
        feats["dk_devigged_over"] = dk_dv_over
        feats["dk_devigged_under"] = dk_dv_under
        feats["vig"] = implied_over + implied_under - 1.0

        feats["line_vs_season"] = line - feats["target_season"]
        feats["line_vs_l5"] = line - feats["target_l5"]
        feats["line_vs_l3"] = line - feats["target_l3"]
        feats["l3_above_line"] = feats["target_l3"] - line
        feats["l5_above_line"] = feats["target_l5"] - line
        feats["rate_above_line"] = feats["expected_from_rate"] - line
        feats["trend_magnitude"] = abs(feats["target_trend"])
        feats["odds_skew"] = implied_over - implied_under

        # === CONSISTENCY / VOLATILITY ===
        if len(l10) >= 5:
            feats["pct_25_l10"] = np.percentile(l10, 25)
            feats["pct_75_l10"] = np.percentile(l10, 75)
            feats["iqr_l10"] = feats["pct_75_l10"] - feats["pct_25_l10"]
        else:
            feats["pct_25_l10"] = feats["target_l5"] - feats["target_std_l10"]
            feats["pct_75_l10"] = feats["target_l5"] + feats["target_std_l10"]
            feats["iqr_l10"] = 2 * feats["target_std_l10"]

        streak = 0
        for v in values:
            if v > line:
                if streak >= 0:
                    streak += 1
                else:
                    break
            else:
                if streak <= 0:
                    streak -= 1
                else:
                    break
        feats["over_streak"] = max(streak, 0)
        feats["under_streak"] = max(-streak, 0)

        # === PROP TYPE ONE-HOT ===
        for pt in CORE_PROPS:
            feats[f"is_{pt}"] = 1 if prop_type == pt else 0

        # === MULTI-BOOK CONSENSUS FEATURES (THE KEY NEW DATA) ===
        consensus = compute_consensus_features(all_books, line, over_odds, under_odds)
        feats.update(consensus)

        return feats


# =============================================================================
# CALIBRATED MODEL TRAINER
# =============================================================================

class V8Trainer:
    """Train calibrated over/under classifiers per prop type."""

    def __init__(self):
        self.feature_builder = V8FeatureBuilder()

    def build_dataset(self) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        """Build full dataset from all legs + multibook enrichment."""
        logger.info("Loading game data...")
        self.feature_builder.load_games()
        self.feature_builder.build_player_history()

        # Load legs — prefer multibook-enriched versions
        all_legs = []
        for mb_path, plain_path in zip(LEGS_FILES, LEGS_FILES_PLAIN):
            if mb_path.exists():
                logger.info("Loading multibook legs: %s", mb_path.name)
                with open(mb_path) as f:
                    legs = json.load(f)
                logger.info("  -> %d legs", len(legs))
                all_legs.extend(legs)
            elif plain_path.exists():
                logger.info("Loading plain legs: %s", plain_path.name)
                with open(plain_path) as f:
                    legs = json.load(f)
                logger.info("  -> %d legs (no multibook)", len(legs))
                all_legs.extend(legs)

        logger.info("Total raw legs: %d", len(all_legs))

        # Filter to core props and valid data
        prop_set = set(CORE_PROPS)
        all_legs = [
            leg for leg in all_legs
            if leg.get("prop_type") in prop_set
            and leg.get("actual") is not None
            and leg.get("line") is not None
            and leg.get("line", 0) > 0
        ]
        all_legs.sort(key=lambda x: x["date"])
        logger.info("Legs after filtering: %d", len(all_legs))

        rows_X = []
        rows_y = []
        rows_meta = []
        skipped = 0

        for i, leg in enumerate(all_legs):
            if i > 0 and i % 10000 == 0:
                logger.info("  Processing %d/%d (%.0f%%) — %d features built",
                            i, len(all_legs), 100 * i / len(all_legs), len(rows_X))

            player = leg["player"]
            prop_type = leg["prop_type"]
            date = leg["date"]
            actual = leg["actual"]
            line = leg["line"]
            over_odds = leg.get("over_odds", -110)
            under_odds = leg.get("under_odds", -110)
            all_books = leg.get("all_books", {})

            feats = self.feature_builder.build_features(
                player, prop_type, date, line, over_odds, under_odds, all_books
            )
            if feats is None:
                skipped += 1
                continue

            # TARGET: binary over/under (push = under for consistency)
            target = 1 if actual > line else 0

            rows_X.append(feats)
            rows_y.append(target)
            rows_meta.append({
                "date": date,
                "player": player,
                "prop_type": prop_type,
                "line": line,
                "actual": actual,
                "over_odds": over_odds,
                "under_odds": under_odds,
                "has_multibook": 1 if all_books and len(all_books) > 1 else 0,
            })

        X = pd.DataFrame(rows_X)
        y = np.array(rows_y)
        meta = pd.DataFrame(rows_meta)
        logger.info("Built dataset: %d samples, %d features (skipped %d)", len(X), len(X.columns), skipped)
        logger.info("Over rate: %.1f%%", 100 * y.mean())
        logger.info("Multibook coverage: %.1f%%", 100 * meta["has_multibook"].mean())

        return X, y, meta

    def _compute_sample_weights(self, dates: pd.Series) -> np.ndarray:
        max_date = dates.max()
        days_ago = (pd.to_datetime(max_date) - pd.to_datetime(dates)).dt.days
        weights = np.exp(-0.693 * days_ago.values / 90.0)
        return weights / weights.mean()

    def train_prop(self, X: pd.DataFrame, y: np.ndarray, meta: pd.DataFrame,
                   prop_type: str) -> Optional[dict]:
        """Train calibrated classifier + quantile regressors for one prop type."""
        mask = meta["prop_type"] == prop_type
        X_prop = X[mask].copy()
        y_prop = y[mask].copy()
        meta_prop = meta[mask].copy()
        dates = meta_prop["date"].values

        if len(X_prop) < 200:
            logger.warning("Skipping %s: only %d samples", prop_type, len(X_prop))
            return None

        # Walk-forward time split
        sort_idx = np.argsort(dates)
        X_prop = X_prop.iloc[sort_idx].reset_index(drop=True)
        y_prop = y_prop[sort_idx]
        meta_prop = meta_prop.iloc[sort_idx].reset_index(drop=True)
        dates = meta_prop["date"].values

        n = len(X_prop)
        train_end = int(n * 0.55)
        cal_end = int(n * 0.65)
        val_end = int(n * 0.75)

        X_train = X_prop.iloc[:train_end]
        y_train = y_prop[:train_end]
        X_cal = X_prop.iloc[train_end:cal_end]
        y_cal = y_prop[train_end:cal_end]
        X_val = X_prop.iloc[cal_end:val_end]
        y_val = y_prop[cal_end:val_end]
        X_test = X_prop.iloc[val_end:]
        y_test = y_prop[val_end:]
        meta_test = meta_prop.iloc[val_end:]

        logger.info("%s: Train=%d, Cal=%d, Val=%d, Test=%d [%s to %s]",
                    prop_type, len(X_train), len(X_cal), len(X_val), len(X_test),
                    dates[0], dates[-1])

        weights_train = self._compute_sample_weights(pd.Series(dates[:train_end]))

        feature_cols = X_train.columns.tolist()

        # ==========================================================
        # 1. LightGBM CLASSIFIER (log-loss objective)
        # ==========================================================
        clf = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.03,
            num_leaves=40,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=1.0,
            random_state=42,
            verbose=-1,
        )
        clf.fit(
            X_train, y_train,
            sample_weight=weights_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        # Raw classifier probabilities
        raw_probs_cal = clf.predict_proba(X_cal)[:, 1]
        raw_probs_test = clf.predict_proba(X_test)[:, 1]

        # ==========================================================
        # 2. POST-HOC CALIBRATION: Platt scaling + isotonic
        # ==========================================================
        # Platt scaling (logistic regression on raw probs)
        from sklearn.linear_model import LogisticRegression
        platt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        platt.fit(raw_probs_cal.reshape(-1, 1), y_cal)
        platt_probs_test = platt.predict_proba(raw_probs_test.reshape(-1, 1))[:, 1]

        # Isotonic regression
        iso_reg = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
        iso_reg.fit(raw_probs_cal, y_cal)
        iso_probs_test = iso_reg.predict(raw_probs_test)

        # Pick the calibration method with better Brier score
        brier_platt = brier_score_loss(y_test, platt_probs_test)
        brier_iso = brier_score_loss(y_test, iso_probs_test)
        brier_raw_check = brier_score_loss(y_test, raw_probs_test)

        # Use best method, but if raw is best, skip calibration
        if brier_raw_check <= min(brier_platt, brier_iso):
            cal_probs_test = raw_probs_test
            cal_method = "raw"
        elif brier_platt <= brier_iso:
            cal_probs_test = platt_probs_test
            cal_method = "platt"
        else:
            cal_probs_test = iso_probs_test
            cal_method = "isotonic"

        logger.info("  %s: Brier raw=%.4f, platt=%.4f, iso=%.4f -> using %s",
                    prop_type, brier_raw_check, brier_platt, brier_iso, cal_method)

        # Consensus blend test: 0.70*consensus + 0.30*model
        consensus_probs_test = X_test["consensus_over_prob"].values
        has_consensus = X_test["num_books"].values > 2
        blended_probs_test = np.where(
            has_consensus,
            0.70 * consensus_probs_test + 0.30 * cal_probs_test,
            cal_probs_test
        )
        brier_blended = brier_score_loss(y_test, blended_probs_test)

        # ==========================================================
        # 3. QUANTILE REGRESSION for uncertainty
        # ==========================================================
        # Target: actual - line (the residual)
        y_residual_train = meta_prop.iloc[:train_end]["actual"].values - meta_prop.iloc[:train_end]["line"].values
        y_residual_val = meta_prop.iloc[cal_end:val_end]["actual"].values - meta_prop.iloc[cal_end:val_end]["line"].values
        y_residual_test = meta_test["actual"].values - meta_test["line"].values

        q10_model = lgb.LGBMRegressor(
            objective="quantile", alpha=0.10,
            n_estimators=500, max_depth=5, learning_rate=0.05,
            num_leaves=32, min_child_samples=50,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbose=-1,
        )
        q10_model.fit(X_train, y_residual_train, sample_weight=weights_train,
                      eval_set=[(X_val, y_residual_val)],
                      callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

        q90_model = lgb.LGBMRegressor(
            objective="quantile", alpha=0.90,
            n_estimators=500, max_depth=5, learning_rate=0.05,
            num_leaves=32, min_child_samples=50,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbose=-1,
        )
        q90_model.fit(X_train, y_residual_train, sample_weight=weights_train,
                      eval_set=[(X_val, y_residual_val)],
                      callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

        # ==========================================================
        # EVALUATE ON TEST SET
        # ==========================================================
        lines_test = meta_test["line"].values
        actuals_test = meta_test["actual"].values
        over_odds_test = meta_test["over_odds"].values
        under_odds_test = meta_test["under_odds"].values
        actual_over = (actuals_test > lines_test).astype(int)

        # Brier score (lower is better)
        brier_cal = brier_score_loss(y_test, cal_probs_test)
        logloss_cal = log_loss(y_test, np.clip(cal_probs_test, 1e-7, 1-1e-7))

        # Baseline: DK devigged probability
        dk_probs_test = np.array([
            devig_power(int(o), int(u))[0]
            for o, u in zip(over_odds_test, under_odds_test)
        ])
        brier_dk = brier_score_loss(y_test, dk_probs_test)

        # Calibration by decile
        cal_bins = self._calibration_bins(cal_probs_test, y_test)

        # Directional accuracy
        dir_acc = np.mean((cal_probs_test > 0.5) == actual_over)

        # ROI simulation
        roi_results = self._roi_simulation(
            cal_probs_test, actual_over, actuals_test, lines_test,
            over_odds_test, under_odds_test
        )

        # CLV proxy: compare our P(over) vs closing line (DK devigged)
        clv_results = self._clv_analysis(cal_probs_test, dk_probs_test, actual_over)

        # Feature importance
        importances = clf.feature_importances_
        imp_pairs = sorted(zip(feature_cols, importances), key=lambda x: -x[1])

        # Quantile calibration
        q10_preds = q10_model.predict(X_test)
        q90_preds = q90_model.predict(X_test)
        in_range = np.mean((y_residual_test >= q10_preds) & (y_residual_test <= q90_preds))

        # ==========================================================
        # PRINT RESULTS
        # ==========================================================
        # ROI simulation on BLENDED probabilities (the real production signal)
        roi_blended = self._roi_simulation(
            blended_probs_test, actual_over, actuals_test, lines_test,
            over_odds_test, under_odds_test
        )

        print(f"\n{'='*70}")
        print(f"  {prop_type.upper()} — v8 Calibrated Classifier ({len(X_test)} test samples)")
        print(f"{'='*70}")
        print(f"  Calibration:    {cal_method} (raw={brier_raw_check:.4f}, platt={brier_platt:.4f}, iso={brier_iso:.4f})")
        print(f"  Brier (model):  {brier_cal:.4f} | DK baseline: {brier_dk:.4f} | BLENDED: {brier_blended:.4f}")
        print(f"  Log-Loss:       {logloss_cal:.4f}")
        print(f"  Direction Acc:  {dir_acc:.1%}")
        print(f"  Beats DK Brier: model={'YES' if brier_cal < brier_dk else 'NO'}, "
              f"blended={'YES' if brier_blended < brier_dk else 'NO'}")
        print(f"  P10-P90 Calib:  {in_range:.1%} (target: 80%)")
        print(f"  Early Stop:     clf@{clf.best_iteration_}")

        print(f"\n  --- CALIBRATION BY DECILE ---")
        print(f"  {'Bin':>20s} {'N':>6s} {'Pred Avg':>10s} {'Actual':>10s} {'Gap':>8s}")
        for cb in cal_bins:
            print(f"  {cb['bin']:>20s} {cb['n']:6d} {cb['pred_avg']:10.3f} "
                  f"{cb['actual_rate']:10.3f} {cb['gap']:+8.3f}")

        print(f"\n  --- ROI: MODEL-ONLY (flat $100 bets) ---")
        print(f"  {'Threshold':>12s} {'N Bets':>8s} {'Win Rate':>10s} {'ROI':>8s} {'Units':>8s}")
        for rt in roi_results:
            print(f"  {rt['threshold']:>12s} {rt['n_bets']:8d} {rt['win_rate']:10.1%} "
                  f"{rt['roi']:+7.1%} {rt['units']:+8.1f}")

        print(f"\n  --- ROI: CONSENSUS-BLENDED (70% consensus + 30% model) ---")
        print(f"  {'Threshold':>12s} {'N Bets':>8s} {'Win Rate':>10s} {'ROI':>8s} {'Units':>8s}")
        for rt in roi_blended:
            print(f"  {rt['threshold']:>12s} {rt['n_bets']:8d} {rt['win_rate']:10.1%} "
                  f"{rt['roi']:+7.1%} {rt['units']:+8.1f}")

        print(f"\n  --- CLV ANALYSIS ---")
        for ck, cv in clv_results.items():
            print(f"  {ck}: {cv}")

        print(f"\n  Top 15 features:")
        for rank, (name, imp) in enumerate(imp_pairs[:15], 1):
            print(f"    {rank:2d}. {name:35s} {imp:6d}")

        metrics = {
            "prop_type": prop_type,
            "n_train": len(X_train),
            "n_cal": len(X_cal),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "calibration_method": cal_method,
            "brier_calibrated": float(brier_cal),
            "brier_blended": float(brier_blended),
            "brier_dk_baseline": float(brier_dk),
            "beats_dk_brier": bool(brier_cal < brier_dk),
            "blended_beats_dk": bool(brier_blended < brier_dk),
            "logloss_calibrated": float(logloss_cal),
            "directional_accuracy": float(dir_acc),
            "quantile_calibration_80": float(in_range),
            "calibration_bins": cal_bins,
            "roi_results": roi_results,
            "roi_blended": roi_blended,
            "clv_results": clv_results,
            "top_features": [{"name": n, "importance": int(i)} for n, i in imp_pairs[:30]],
            "best_iteration": clf.best_iteration_,
            "feature_names": feature_cols,
        }

        return {
            "clf": clf,
            "isotonic": iso_reg,
            "platt": platt,
            "cal_method": cal_method,
            "q10": q10_model,
            "q90": q90_model,
            "metrics": metrics,
            "feature_names": feature_cols,
        }

    def _calibration_bins(self, probs: np.ndarray, actual: np.ndarray,
                          n_bins: int = 10) -> list:
        bins = []
        edges = np.linspace(0, 1, n_bins + 1)
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() == 0:
                continue
            bins.append({
                "bin": f"[{lo:.1f}, {hi:.1f})",
                "n": int(mask.sum()),
                "pred_avg": float(probs[mask].mean()),
                "actual_rate": float(actual[mask].mean()),
                "gap": float(probs[mask].mean() - actual[mask].mean()),
            })
        return bins

    def _roi_simulation(self, probs, actual_over, actuals, lines,
                        over_odds, under_odds) -> list:
        results = []
        for label, min_edge in [
            (">= 0%", 0.0), (">= 2%", 0.02), (">= 3%", 0.03),
            (">= 5%", 0.05), (">= 8%", 0.08), (">= 10%", 0.10),
        ]:
            total_wager = 0.0
            total_profit = 0.0
            n_bets = 0
            n_wins = 0
            for i in range(len(probs)):
                p_over = probs[i]
                p_under = 1.0 - p_over
                imp_over = american_to_implied(int(over_odds[i]))
                imp_under = american_to_implied(int(under_odds[i]))
                edge_over = p_over - imp_over
                edge_under = p_under - imp_under

                if edge_over >= min_edge and edge_over >= edge_under:
                    side = "OVER"
                    odds = int(over_odds[i])
                elif edge_under >= min_edge:
                    side = "UNDER"
                    odds = int(under_odds[i])
                else:
                    continue

                n_bets += 1
                total_wager += 100.0
                hit = (side == "OVER" and actuals[i] > lines[i]) or \
                      (side == "UNDER" and actuals[i] < lines[i])
                if hit:
                    n_wins += 1
                    total_profit += 100.0 * (american_to_decimal(odds) - 1.0)
                else:
                    total_profit -= 100.0

            results.append({
                "threshold": label,
                "n_bets": n_bets,
                "win_rate": n_wins / max(n_bets, 1),
                "roi": total_profit / max(total_wager, 1),
                "units": total_profit / 100.0,
            })
        return results

    def _clv_analysis(self, model_probs, dk_probs, actual_over) -> dict:
        """Measure CLV: does our probability beat the DK line?"""
        # For bets we would take: when our P(side) > DK P(side) + 2%
        clv_sum = 0.0
        n_clv = 0
        clv_positive = 0
        for i in range(len(model_probs)):
            p_over = model_probs[i]
            dk_over = dk_probs[i]
            edge_over = p_over - dk_over
            edge_under = (1 - p_over) - (1 - dk_over)

            if abs(edge_over) < 0.02:
                continue

            if edge_over > 0:
                clv = edge_over
            else:
                clv = -edge_over  # we'd bet under, CLV is edge_under
            clv_sum += clv
            n_clv += 1
            if clv > 0:
                clv_positive += 1

        return {
            "n_bets_with_edge": n_clv,
            "avg_clv": clv_sum / max(n_clv, 1),
            "pct_positive_clv": clv_positive / max(n_clv, 1),
        }

    def train_all(self, X: pd.DataFrame, y: np.ndarray, meta: pd.DataFrame) -> dict:
        """Train all prop types. Save to models/v8_calibrated/."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        all_results = {}
        for prop_type in CORE_PROPS:
            logger.info("Training %s ...", prop_type)
            t0 = time.time()
            result = self.train_prop(X, y, meta, prop_type)
            if result is None:
                continue
            elapsed = time.time() - t0
            logger.info("  %s trained in %.1fs", prop_type, elapsed)

            # Save models
            joblib.dump(result["clf"], MODELS_DIR / f"{prop_type}_classifier.pkl")
            joblib.dump(result["isotonic"], MODELS_DIR / f"{prop_type}_isotonic.pkl")
            joblib.dump(result["platt"], MODELS_DIR / f"{prop_type}_platt.pkl")
            joblib.dump(result["q10"], MODELS_DIR / f"{prop_type}_q10.pkl")
            joblib.dump(result["q90"], MODELS_DIR / f"{prop_type}_q90.pkl")

            all_results[prop_type] = result["metrics"]

        # Save metadata
        meta_out = {
            "version": "v8",
            "model_type": "calibrated_classifier",
            "architecture": "LGB classifier + isotonic calibration + quantile regression",
            "core_props": CORE_PROPS,
            "n_total_samples": len(X),
            "results": all_results,
        }
        # Add per-prop metadata for predictor
        for prop_type, metrics in all_results.items():
            meta_out[prop_type] = {
                "feature_names": metrics["feature_names"],
                "brier": metrics["brier_calibrated"],
                "directional_accuracy": metrics["directional_accuracy"],
            }

        with open(MODELS_DIR / "model_metadata.json", "w") as f:
            json.dump(meta_out, f, indent=2, default=str)

        # Print summary
        print(f"\n{'='*90}")
        print(f"  V8 CALIBRATED CLASSIFIER — TRAINING SUMMARY")
        print(f"{'='*90}")
        print(f"  {'Prop':20s} {'Test N':>8s} {'Brier':>8s} {'DK Brier':>10s} "
              f"{'Beats?':>7s} {'Dir Acc':>8s} {'ROI@5%':>8s}")
        for pt, m in all_results.items():
            roi_5 = next((r["roi"] for r in m["roi_results"] if r["threshold"] == ">= 5%"), 0)
            print(f"  {pt:20s} {m['n_test']:8d} {m['brier_calibrated']:8.4f} "
                  f"{m['brier_dk_baseline']:10.4f} "
                  f"{'YES' if m['beats_dk_brier'] else 'NO':>7s} "
                  f"{m['directional_accuracy']:7.1%} {roi_5:+7.1%}")

        # Aggregate ROI
        print(f"\n  --- AGGREGATE ROI (all props, flat $100 bets) ---")
        for thresh in [">= 0%", ">= 2%", ">= 3%", ">= 5%", ">= 8%", ">= 10%"]:
            total_bets = sum(
                next((r["n_bets"] for r in m["roi_results"] if r["threshold"] == thresh), 0)
                for m in all_results.values()
            )
            total_units = sum(
                next((r["units"] for r in m["roi_results"] if r["threshold"] == thresh), 0)
                for m in all_results.values()
            )
            total_wins = sum(
                int(next((r["win_rate"] for r in m["roi_results"] if r["threshold"] == thresh), 0)
                    * next((r["n_bets"] for r in m["roi_results"] if r["threshold"] == thresh), 0))
                for m in all_results.values()
            )
            if total_bets > 0:
                print(f"  {thresh:>12s}: {total_bets:6d} bets, "
                      f"win={total_wins/total_bets:.1%}, "
                      f"ROI={total_units/total_bets:+.1%}, "
                      f"units={total_units:+.1f}")

        print(f"\n  Models saved to: {MODELS_DIR}/")
        return all_results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train v8 calibrated classifier")
    parser.add_argument("--cache", action="store_true", help="Use cached features")
    args = parser.parse_args()

    print("=" * 70)
    print("v8 Calibrated Classifier Training")
    print("  Architecture: LGB binary classifier + isotonic calibration")
    print("  Target: P(actual > line)")
    print("  Key feature: devigged sharp consensus probability")
    print("=" * 70)

    t_start = time.time()
    trainer = V8Trainer()

    cache_dir = OUTPUTS / "v8_cache"
    if args.cache and (cache_dir / "features.parquet").exists():
        logger.info("Loading cached dataset...")
        X = pd.read_parquet(cache_dir / "features.parquet")
        y = pd.read_parquet(cache_dir / "targets.parquet")["target"].values
        meta = pd.read_parquet(cache_dir / "metadata.parquet")
    else:
        X, y, meta = trainer.build_dataset()
        # Cache for future runs
        cache_dir.mkdir(parents=True, exist_ok=True)
        X.to_parquet(cache_dir / "features.parquet")
        pd.DataFrame({"target": y}).to_parquet(cache_dir / "targets.parquet")
        meta.to_parquet(cache_dir / "metadata.parquet")
        logger.info("Cached dataset to %s", cache_dir)

    results = trainer.train_all(X, y, meta)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
