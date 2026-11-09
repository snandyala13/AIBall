#!/usr/bin/env python3
"""
v2/train.py - Train line-anchor LightGBM regressors for each prop type.

Architecture:
  predicted_adjustment = model.predict(features)
  final_prediction = DK_line + predicted_adjustment
  Target = actual - DK_line (how wrong was the line?)

For each prop type, trains:
  1. Point estimate model (L2 loss) -> predicted adjustment
  2. P10 quantile model (quantile alpha=0.1) -> lower bound of adjustment
  3. P90 quantile model (quantile alpha=0.9) -> upper bound of adjustment

Uses walk-forward time split (no random splits -- this is time-series):
  - Train:      first 70% by date
  - Validation: next 15% (early stopping)
  - Test:       final 15% (evaluation)

Key evaluations:
  - Does model beat just using the line (adjustment=0)?
  - MONOTONICITY: When model says "over" with high confidence, does over actually hit more?
  - ROI at different edge thresholds

Usage:
    python -m v2.train              # Build features + train all models
    python -m v2.train --from-cache # Use cached features from prior run
    python -m v2.train --raw-target # Legacy mode: predict raw stat value
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Check for lightgbm
try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: lightgbm not installed.")
    print("Install it with:  pip install lightgbm")
    print("Or:               conda install -c conda-forge lightgbm")
    sys.exit(1)

try:
    import joblib
except ImportError:
    print("ERROR: joblib not installed. Install with: pip install joblib")
    sys.exit(1)

from scipy import stats as scipy_stats

from v2.features import FeatureBuilder, CORE_PROPS

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models" / "v2"
CACHE_DIR = BASE_DIR / "outputs" / "v2_training"


# =============================================================================
# ModelTrainer
# =============================================================================

class ModelTrainer:
    """Train and evaluate line-anchor LightGBM models for NBA prop prediction."""

    def __init__(self, features_df: pd.DataFrame, targets: np.ndarray,
                 metadata: pd.DataFrame, target_mode: str = "line_anchor"):
        self.X = features_df
        self.y = targets
        self.meta = metadata
        self.target_mode = target_mode
        self.results = {}  # {prop_type: metrics_dict}

    @classmethod
    def from_feature_builder(cls, prop_types=None, target_mode="line_anchor"):
        """Build features from scratch using FeatureBuilder."""
        fb = FeatureBuilder()
        fb.load_games()
        fb.build_player_history()
        X, y, meta = fb.build_training_data(
            prop_types=prop_types,
            target_mode=target_mode,
        )
        return cls(X, y, meta, target_mode=target_mode)

    @classmethod
    def from_cache(cls, target_mode="line_anchor"):
        """Load cached features from disk."""
        logger.info("Loading cached features from %s", CACHE_DIR)
        X = pd.read_parquet(CACHE_DIR / "features.parquet")
        y = pd.read_parquet(CACHE_DIR / "targets.parquet")["target"].values
        meta = pd.read_parquet(CACHE_DIR / "metadata.parquet")
        logger.info("Loaded %d samples, %d features", len(X), len(X.columns))
        return cls(X, y, meta, target_mode=target_mode)

    # -----------------------------------------------------------------
    # Time-weighted sample weights
    # -----------------------------------------------------------------

    def _compute_sample_weights(self, dates: pd.Series) -> np.ndarray:
        """
        Exponential decay weights favoring recent data.
        Half-life = 90 days.
        """
        max_date = dates.max()
        # Convert to days-ago
        days_ago = (pd.to_datetime(max_date) - pd.to_datetime(dates)).dt.days
        half_life = 90.0
        weights = np.exp(-0.693 * days_ago.values / half_life)
        # Normalize so mean weight = 1
        weights = weights / weights.mean()
        return weights

    # -----------------------------------------------------------------
    # Train a single prop type
    # -----------------------------------------------------------------

    def train_prop_model(self, prop_type: str) -> Tuple:
        """
        Train LightGBM regressor for a specific prop type.

        In line-anchor mode:
          - Target = actual - line (the adjustment)
          - A prediction of +2.0 means "actual will be 2 points above the line"
          - A prediction of -1.5 means "actual will be 1.5 below the line"

        Returns:
            (point_model, q10_model, q90_model, metrics_dict)
        """
        mask = self.meta["prop_type"] == prop_type
        X_prop = self.X[mask].copy()
        y_prop = self.y[mask].copy()
        meta_prop = self.meta[mask].copy()
        dates = meta_prop["date"].copy()

        if len(X_prop) < 100:
            logger.warning("Skipping %s: only %d samples", prop_type, len(X_prop))
            return None, None, None, {}

        # Sort by date (walk-forward)
        sort_idx = dates.argsort()
        X_prop = X_prop.iloc[sort_idx].reset_index(drop=True)
        y_prop = y_prop[sort_idx]
        meta_prop = meta_prop.iloc[sort_idx].reset_index(drop=True)
        dates = dates.iloc[sort_idx].reset_index(drop=True)

        n = len(X_prop)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        X_train = X_prop.iloc[:train_end]
        y_train = y_prop[:train_end]
        dates_train = dates.iloc[:train_end]
        meta_train = meta_prop.iloc[:train_end]

        X_val = X_prop.iloc[train_end:val_end]
        y_val = y_prop[train_end:val_end]

        X_test = X_prop.iloc[val_end:]
        y_test = y_prop[val_end:]
        dates_test = dates.iloc[val_end:]
        meta_test = meta_prop.iloc[val_end:]

        logger.info(
            "%s: Train=%d (to %s), Val=%d (to %s), Test=%d (to %s)",
            prop_type, len(X_train), dates_train.iloc[-1],
            len(X_val), dates.iloc[val_end - 1],
            len(X_test), dates_test.iloc[-1],
        )

        # Sample weights (favor recent data in training set)
        weights_train = self._compute_sample_weights(dates_train)

        # -- Point estimate model (L2 loss) --
        point_model = lgb.LGBMRegressor(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=40,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
        )
        point_model.fit(
            X_train, y_train,
            sample_weight=weights_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )

        # -- P10 quantile model --
        q10_model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=0.10,
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=32,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
        )
        q10_model.fit(
            X_train, y_train,
            sample_weight=weights_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )

        # -- P90 quantile model --
        q90_model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=0.90,
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=32,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
        )
        q90_model.fit(
            X_train, y_train,
            sample_weight=weights_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )

        # ==================================================================
        # Evaluate on TEST set
        # ==================================================================
        pred_adjustments = point_model.predict(X_test)
        p10_adjustments = q10_model.predict(X_test)
        p90_adjustments = q90_model.predict(X_test)

        line_values = X_test["line"].values

        if self.target_mode == "line_anchor":
            # y_test = actual - line (the true adjustment)
            # pred_adjustments = predicted adjustment
            # Baseline: adjustment = 0 (just trust the line)
            actual_adjustments = y_test
            actuals = y_test + line_values  # Reconstruct actual stat values

            # Model's final predictions
            preds_final = line_values + pred_adjustments

            # Model metrics on the ADJUSTMENT
            residuals = pred_adjustments - actual_adjustments
            mae_adj = np.mean(np.abs(residuals))
            baseline_adj_mae = np.mean(np.abs(actual_adjustments))  # MAE of just using line

            # Model metrics on the FINAL prediction (line + adjustment)
            model_mae = np.mean(np.abs(preds_final - actuals))
            line_mae = np.mean(np.abs(line_values - actuals))  # = baseline_adj_mae

            model_rmse = np.sqrt(np.mean((preds_final - actuals) ** 2))
            line_rmse = np.sqrt(np.mean((line_values - actuals) ** 2))

        else:
            # Legacy raw mode
            actuals = y_test
            preds_final = pred_adjustments  # In raw mode, these ARE the predictions
            actual_adjustments = y_test - line_values

            residuals = preds_final - actuals
            model_mae = np.mean(np.abs(residuals))
            line_mae = np.mean(np.abs(line_values - actuals))
            mae_adj = model_mae
            baseline_adj_mae = line_mae

            model_rmse = np.sqrt(np.mean(residuals ** 2))
            line_rmse = np.sqrt(np.mean((line_values - actuals) ** 2))

        # R-squared
        ss_res = np.sum((preds_final - actuals) ** 2)
        ss_tot = np.sum((actuals - actuals.mean()) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-9)

        # Directional accuracy: when model predicts over (adjustment > 0),
        # does the over actually hit?
        if self.target_mode == "line_anchor":
            model_says_over = pred_adjustments > 0
            actual_over = actual_adjustments > 0
        else:
            model_says_over = preds_final > line_values
            actual_over = actuals > line_values
        directional_acc = np.mean(model_says_over == actual_over)

        # Edge signal check: when model says OVER, how often does over hit?
        over_mask = model_says_over
        under_mask = ~model_says_over
        if over_mask.sum() > 0:
            over_hit_rate = np.mean(actual_over[over_mask])
        else:
            over_hit_rate = 0.0
        if under_mask.sum() > 0:
            under_hit_rate = np.mean(~actual_over[under_mask])
        else:
            under_hit_rate = 0.0

        # Quantile calibration
        if self.target_mode == "line_anchor":
            in_range = np.mean(
                (actual_adjustments >= p10_adjustments) &
                (actual_adjustments <= p90_adjustments)
            )
        else:
            in_range = np.mean(
                (actuals >= p10_adjustments) &
                (actuals <= p90_adjustments)
            )

        # ==================================================================
        # MONOTONICITY TEST (the critical v1 failure test)
        # When model predicts larger positive adjustment, does over hit MORE?
        # ==================================================================
        if self.target_mode == "line_anchor":
            adj_for_mono = pred_adjustments
        else:
            adj_for_mono = preds_final - line_values

        sigma_est = np.maximum((p90_adjustments - p10_adjustments) / 2.56, 0.5)
        # Confidence magnitude = |adjustment| / sigma
        confidence = np.abs(adj_for_mono) / sigma_est

        mono_bins = self._monotonicity_test(
            adj_for_mono, actual_over, confidence
        )

        # ==================================================================
        # ROI simulation at edge thresholds
        # ==================================================================
        over_odds_test = X_test["over_odds"].values
        under_odds_test = X_test["under_odds"].values

        roi_results = self._roi_simulation(
            adj_for_mono, sigma_est, actual_over,
            actuals, line_values,
            over_odds_test, under_odds_test,
        )

        metrics = {
            "prop_type": prop_type,
            "target_mode": self.target_mode,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "model_mae": float(model_mae),
            "line_mae": float(line_mae),
            "beats_line_mae": bool(model_mae < line_mae),
            "mae_delta": float(line_mae - model_mae),
            "adjustment_mae": float(mae_adj),
            "baseline_adjustment_mae": float(baseline_adj_mae),
            "model_rmse": float(model_rmse),
            "line_rmse": float(line_rmse),
            "r2": float(r2),
            "directional_accuracy": float(directional_acc),
            "quantile_calibration_80": float(in_range),
            "over_hit_rate": float(over_hit_rate),
            "under_hit_rate": float(under_hit_rate),
            "n_over_preds": int(over_mask.sum()),
            "n_under_preds": int(under_mask.sum()),
            "mean_adjustment": float(np.mean(adj_for_mono)),
            "best_iteration_point": point_model.best_iteration_,
            "best_iteration_q10": q10_model.best_iteration_,
            "best_iteration_q90": q90_model.best_iteration_,
            "monotonicity_bins": mono_bins,
            "roi_results": roi_results,
        }

        # Print results
        print(f"\n{'='*70}")
        print(f"  {prop_type.upper()} - Test Set Results ({metrics['n_test']} samples)")
        if self.target_mode == "line_anchor":
            print(f"  TARGET: actual - line (line-anchor adjustment)")
        else:
            print(f"  TARGET: raw actual value (legacy mode)")
        print(f"{'='*70}")
        print(f"  Model MAE:      {model_mae:.3f}")
        print(f"  Line MAE:       {line_mae:.3f}")
        beats = "YES" if model_mae < line_mae else "NO"
        print(f"  Beats line:     {beats} (delta: {line_mae - model_mae:+.3f})")
        print(f"  Model RMSE:     {model_rmse:.3f}")
        print(f"  Line RMSE:      {line_rmse:.3f}")
        print(f"  R-squared:      {r2:.4f}")
        print(f"  Direction acc:  {directional_acc:.1%}")
        print(f"  P10-P90 calib:  {in_range:.1%} (target: 80%)")
        print(f"  OVER hit rate:  {over_hit_rate:.1%} (when model says over, N={over_mask.sum()})")
        print(f"  UNDER hit rate: {under_hit_rate:.1%} (when model says under, N={under_mask.sum()})")
        print(f"  Mean adjustment:{np.mean(adj_for_mono):+.3f}")
        print(f"  Early stop:     point@{point_model.best_iteration_}, "
              f"q10@{q10_model.best_iteration_}, q90@{q90_model.best_iteration_}")

        # Monotonicity test
        print(f"\n  --- MONOTONICITY TEST ---")
        print(f"  (Higher model confidence -> higher hit rate?)")
        if mono_bins:
            print(f"  {'Confidence Bin':>20s} {'N':>6s} {'Hit Rate':>10s} {'Avg |Adj|':>10s}")
            print(f"  {'-'*20} {'-'*6} {'-'*10} {'-'*10}")
            for mb in mono_bins:
                print(f"  {mb['bin']:>20s} {mb['n']:6d} {mb['hit_rate']:10.1%} "
                      f"{mb['avg_abs_adj']:10.3f}")
            hit_rates = [mb["hit_rate"] for mb in mono_bins if mb["n"] >= 20]
            if len(hit_rates) >= 3:
                is_mono = all(hit_rates[i] <= hit_rates[i + 1]
                              for i in range(len(hit_rates) - 1))
                print(f"  Monotonic: {'YES -- edge signal is real' if is_mono else 'NO -- edge signal is noisy'}")
            else:
                print(f"  (Insufficient bins with N>=20 for monotonicity check)")

        # ROI analysis
        print(f"\n  --- ROI SIMULATION (flat $100 bets) ---")
        if roi_results:
            print(f"  {'Threshold':>12s} {'N Bets':>8s} {'Win Rate':>10s} {'ROI':>8s} {'Units':>8s}")
            print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")
            for rt in roi_results:
                print(f"  {rt['threshold']:>12s} {rt['n_bets']:8d} {rt['win_rate']:10.1%} "
                      f"{rt['roi']:+7.1%} {rt['units']:+8.1f}")

        # Feature importance
        importances = point_model.feature_importances_
        feat_names = X_train.columns.tolist()
        imp_pairs = sorted(zip(feat_names, importances), key=lambda x: -x[1])
        print(f"\n  Top 15 features:")
        for rank, (name, imp) in enumerate(imp_pairs[:15], 1):
            print(f"    {rank:2d}. {name:30s} {imp:6d}")

        self.results[prop_type] = metrics
        return point_model, q10_model, q90_model, metrics

    # -----------------------------------------------------------------
    # Monotonicity test
    # -----------------------------------------------------------------

    def _monotonicity_test(
        self,
        adjustments: np.ndarray,
        actual_over: np.ndarray,
        confidence: np.ndarray,
    ) -> List[dict]:
        """
        Bin predictions by confidence (|adjustment| / sigma).
        For each bin, measure the hit rate (predicted side == actual side).
        A good model shows INCREASING hit rate with INCREASING confidence.
        """
        # Determine if model's predicted side was correct
        model_says_over = adjustments > 0
        correct = (model_says_over == actual_over)

        # Bin by confidence
        bins = [
            ("Low (0-0.3)", (confidence >= 0) & (confidence < 0.3)),
            ("Med-Low (0.3-0.6)", (confidence >= 0.3) & (confidence < 0.6)),
            ("Medium (0.6-1.0)", (confidence >= 0.6) & (confidence < 1.0)),
            ("Med-High (1.0-1.5)", (confidence >= 1.0) & (confidence < 1.5)),
            ("High (>= 1.5)", confidence >= 1.5),
        ]

        results = []
        for label, mask in bins:
            n = mask.sum()
            if n > 0:
                results.append({
                    "bin": label,
                    "n": int(n),
                    "hit_rate": float(correct[mask].mean()),
                    "avg_abs_adj": float(np.abs(adjustments[mask]).mean()),
                })
        return results

    # -----------------------------------------------------------------
    # ROI simulation
    # -----------------------------------------------------------------

    def _roi_simulation(
        self,
        adjustments: np.ndarray,
        sigma: np.ndarray,
        actual_over: np.ndarray,
        actuals: np.ndarray,
        lines: np.ndarray,
        over_odds: np.ndarray,
        under_odds: np.ndarray,
    ) -> List[dict]:
        """
        Simulate flat-bet ROI at different edge thresholds.
        Edge = P(over) - implied_over_prob (for over bets), and vice versa for under.
        """
        # P(over) from the adjustment model
        z_scores = adjustments / sigma
        over_probs = scipy_stats.norm.cdf(z_scores)

        results = []

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

            for i in range(len(adjustments)):
                implied_over = self._american_to_prob(over_odds[i])
                implied_under = self._american_to_prob(under_odds[i])

                edge_over = over_probs[i] - implied_over
                edge_under = (1.0 - over_probs[i]) - implied_under

                if edge_over >= min_edge and edge_over >= edge_under:
                    side = "OVER"
                    odds = over_odds[i]
                elif edge_under >= min_edge:
                    side = "UNDER"
                    odds = under_odds[i]
                else:
                    continue

                n_bets += 1
                total_wagered += 100.0

                hit = (
                    (side == "OVER" and actuals[i] > lines[i]) or
                    (side == "UNDER" and actuals[i] < lines[i])
                )
                if hit:
                    n_wins += 1
                    decimal = self._american_to_decimal(odds)
                    total_profit += 100.0 * (decimal - 1.0)
                else:
                    total_profit -= 100.0

            if n_bets > 0:
                roi = total_profit / total_wagered
                win_rate = n_wins / n_bets
            else:
                roi = 0.0
                win_rate = 0.0

            results.append({
                "threshold": threshold_label,
                "n_bets": n_bets,
                "win_rate": win_rate,
                "roi": roi,
                "units": total_profit / 100.0,
            })

        return results

    # -----------------------------------------------------------------
    # Train all prop types
    # -----------------------------------------------------------------

    def train_all(self) -> Dict[str, dict]:
        """Train models for all core prop types. Save to models/v2/."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        all_metrics = {}
        for prop_type in CORE_PROPS:
            logger.info("Training %s ...", prop_type)
            t0 = time.time()

            point_model, q10_model, q90_model, metrics = self.train_prop_model(prop_type)

            if point_model is None:
                continue

            elapsed = time.time() - t0
            logger.info("  %s trained in %.1fs", prop_type, elapsed)

            # Save models
            joblib.dump(point_model, MODELS_DIR / f"{prop_type}_point.pkl")
            joblib.dump(q10_model, MODELS_DIR / f"{prop_type}_q10.pkl")
            joblib.dump(q90_model, MODELS_DIR / f"{prop_type}_q90.pkl")

            all_metrics[prop_type] = metrics

        # Save training metadata
        meta_out = {
            "target_mode": self.target_mode,
            "feature_columns": self.X.columns.tolist(),
            "core_props": CORE_PROPS,
            "n_total_samples": len(self.X),
            "results": all_metrics,
        }
        with open(MODELS_DIR / "training_meta.json", "w") as f:
            json.dump(meta_out, f, indent=2, default=str)

        # Print summary table
        print(f"\n{'='*90}")
        print(f"  TRAINING SUMMARY ({self.target_mode.upper()} MODE)")
        print(f"{'='*90}")
        print(f"  {'Prop':20s} {'Samples':>8s} {'Model MAE':>10s} {'Line MAE':>10s} "
              f"{'Beats?':>7s} {'Delta':>8s} {'R2':>8s} {'Dir Acc':>8s} {'Over HR':>8s}")
        print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*10} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for pt, m in all_metrics.items():
            print(f"  {pt:20s} {m['n_test']:8d} {m['model_mae']:10.3f} {m['line_mae']:10.3f} "
                  f"{'YES' if m['beats_line_mae'] else 'NO':>7s} {m['mae_delta']:+8.3f} "
                  f"{m['r2']:8.4f} {m['directional_accuracy']:7.1%} "
                  f"{m['over_hit_rate']:7.1%}")

        # Aggregate ROI
        print(f"\n  --- AGGREGATE ROI (all props combined, flat $100 bets) ---")
        # Collect all ROI results at each threshold
        thresholds = [">= 0%", ">= 2%", ">= 3%", ">= 5%", ">= 8%", ">= 10%"]
        for thresh in thresholds:
            total_bets = 0
            total_wins = 0
            total_units = 0.0
            for pt, m in all_metrics.items():
                for rt in m.get("roi_results", []):
                    if rt["threshold"] == thresh:
                        total_bets += rt["n_bets"]
                        total_wins += int(rt["win_rate"] * rt["n_bets"])
                        total_units += rt["units"]
            if total_bets > 0:
                wr = total_wins / total_bets
                roi = total_units / total_bets
                print(f"  {thresh:>12s}: {total_bets:6d} bets, "
                      f"win_rate={wr:.1%}, ROI={roi:+.1%}, units={total_units:+.1f}")

        print(f"\n  Models saved to: {MODELS_DIR}/")
        return all_metrics

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _american_to_prob(odds):
        if odds == 0:
            return 0.5
        if odds > 0:
            return 100.0 / (odds + 100.0)
        return abs(odds) / (abs(odds) + 100.0)

    @staticmethod
    def _american_to_decimal(odds):
        if odds == 0:
            return 2.0
        if odds > 0:
            return 1.0 + odds / 100.0
        return 1.0 + 100.0 / abs(odds)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train v2 LightGBM prop models")
    parser.add_argument("--from-cache", action="store_true",
                        help="Use cached feature files instead of rebuilding")
    parser.add_argument("--raw-target", action="store_true",
                        help="Use raw stat value as target (legacy, not line-anchor)")
    args = parser.parse_args()

    target_mode = "raw" if args.raw_target else "line_anchor"

    print("=" * 70)
    print(f"v2 Model Trainer - LightGBM NBA Prop Prediction")
    print(f"Target mode: {target_mode.upper()}")
    if target_mode == "line_anchor":
        print("  Architecture: predicted = DK_line + ML_adjustment(features)")
        print("  Target = actual - DK_line (how wrong was the line?)")
    else:
        print("  Architecture: predicted = ML_model(features)")
        print("  Target = raw actual stat value (legacy)")
    print("=" * 70)

    t_start = time.time()

    if args.from_cache and (CACHE_DIR / "features.parquet").exists():
        trainer = ModelTrainer.from_cache(target_mode=target_mode)
    else:
        logger.info("Building features from scratch...")
        trainer = ModelTrainer.from_feature_builder(target_mode=target_mode)

    trainer.train_all()

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
