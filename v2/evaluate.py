#!/usr/bin/env python3
"""
v2/evaluate.py - Walk-forward evaluation of v2 line-anchor models.

Answers:
1. Does the model beat the LINE? (MAE comparison)
2. MONOTONICITY: Is the edge signal monotonic? (higher model edge -> higher hit rate?)
3. What's the ROI at different edge thresholds?
4. Does confidence predict accuracy?
5. Calibration: are quantile estimates well-calibrated?
6. Line-anchor specific: What's the distribution of predicted adjustments?

Usage:
    python -m v2.evaluate                # Full evaluation
    python -m v2.evaluate --prop points  # Single prop type
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

try:
    import lightgbm as lgb
    import joblib
except ImportError:
    print("ERROR: lightgbm and joblib required. pip install lightgbm joblib")
    sys.exit(1)

from scipy import stats as scipy_stats

from v2.features import FeatureBuilder, CORE_PROPS, BASE_DIR

MODELS_DIR = BASE_DIR / "models" / "v2"
OUTPUTS_DIR = BASE_DIR / "outputs"


# =============================================================================
# Evaluator
# =============================================================================

class ModelEvaluator:
    """Walk-forward evaluation of trained v2 models."""

    def __init__(self):
        self.feature_builder = FeatureBuilder()
        self.results = {}
        self.target_mode = "line_anchor"  # Will be loaded from metadata

    def run_full_evaluation(self, prop_types: Optional[List[str]] = None):
        """
        Build features and run evaluation on the OOS test portion.
        Uses the same walk-forward split as training (final 15% by date).
        """
        if prop_types is None:
            prop_types = CORE_PROPS

        # Load training metadata for target mode and feature columns
        meta_path = MODELS_DIR / "training_meta.json"
        feature_columns = None
        if meta_path.exists():
            with open(meta_path) as f:
                tmeta = json.load(f)
            feature_columns = tmeta.get("feature_columns")
            self.target_mode = tmeta.get("target_mode", "line_anchor")
        else:
            logger.warning("No training metadata found at %s", meta_path)
            logger.warning("Assuming line_anchor mode. Run 'python -m v2.train' first.")

        logger.info("Building features (target_mode=%s)...", self.target_mode)
        self.feature_builder.load_games()
        self.feature_builder.build_player_history()
        X, y, meta = self.feature_builder.build_training_data(
            prop_types=prop_types,
            target_mode=self.target_mode,
        )

        print(f"\n{'='*80}")
        print(f"  v2 MODEL EVALUATION ({self.target_mode.upper()} MODE)")
        print(f"{'='*80}")
        print(f"  Total samples: {len(X):,}")
        print(f"  Date range: {meta['date'].min()} to {meta['date'].max()}")
        if self.target_mode == "line_anchor":
            print(f"  Target: actual - line (mean={y.mean():+.3f}, std={y.std():.3f})")

        all_results = {}

        for prop_type in prop_types:
            point_path = MODELS_DIR / f"{prop_type}_point.pkl"
            q10_path = MODELS_DIR / f"{prop_type}_q10.pkl"
            q90_path = MODELS_DIR / f"{prop_type}_q90.pkl"

            if not point_path.exists():
                logger.warning("No model for %s, skipping", prop_type)
                continue

            point_model = joblib.load(point_path)
            q10_model = joblib.load(q10_path) if q10_path.exists() else None
            q90_model = joblib.load(q90_path) if q90_path.exists() else None

            # Get test split (final 15% by date for this prop)
            mask = meta["prop_type"] == prop_type
            X_prop = X[mask].copy()
            y_prop = y[mask].copy()
            meta_prop = meta[mask].copy()

            sort_idx = meta_prop["date"].argsort()
            X_prop = X_prop.iloc[sort_idx].reset_index(drop=True)
            y_prop = y_prop[sort_idx]
            meta_prop = meta_prop.iloc[sort_idx].reset_index(drop=True)

            n = len(X_prop)
            val_end = int(n * 0.85)
            X_test = X_prop.iloc[val_end:]
            y_test = y_prop[val_end:]
            meta_test = meta_prop.iloc[val_end:]

            if len(X_test) < 50:
                logger.warning("Only %d test samples for %s, skipping", len(X_test), prop_type)
                continue

            # Align features to expected column order
            if feature_columns:
                for col in feature_columns:
                    if col not in X_test.columns:
                        X_test[col] = 0.0
                X_test = X_test[feature_columns]

            results = self._evaluate_prop(
                prop_type, point_model, q10_model, q90_model,
                X_test, y_test, meta_test,
            )
            all_results[prop_type] = results

        # Print combined summary
        self._print_summary(all_results)

        # Save detailed results
        out_path = OUTPUTS_DIR / "v2_evaluation.json"
        serializable = {}
        for pt, r in all_results.items():
            serializable[pt] = {k: v for k, v in r.items()
                                if not isinstance(v, (np.ndarray, pd.Series, pd.DataFrame))}
        with open(out_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"\n  Detailed results saved to {out_path}")

        return all_results

    def _evaluate_prop(
        self, prop_type, point_model, q10_model, q90_model,
        X_test, y_test, meta_test,
    ) -> dict:
        """Evaluate a single prop type model."""

        raw_preds = point_model.predict(X_test)
        raw_p10 = q10_model.predict(X_test) if q10_model else raw_preds - 5
        raw_p90 = q90_model.predict(X_test) if q90_model else raw_preds + 5

        lines = X_test["line"].values

        if self.target_mode == "line_anchor":
            # raw_preds = predicted adjustments
            # y_test = actual adjustments (actual - line)
            pred_adjustments = raw_preds
            actual_adjustments = y_test
            actuals = y_test + lines  # Reconstruct actual values
            preds_final = lines + pred_adjustments

            model_mae = np.mean(np.abs(preds_final - actuals))
            line_mae = np.mean(np.abs(lines - actuals))  # = mean(|actual_adjustments|)

            model_rmse = np.sqrt(np.mean((preds_final - actuals) ** 2))
            line_rmse = np.sqrt(np.mean((lines - actuals) ** 2))

            actual_over = actual_adjustments > 0
            model_over = pred_adjustments > 0

            # For sigma estimation
            sigma = np.maximum((raw_p90 - raw_p10) / 2.56, 0.5)
            z_scores = pred_adjustments / sigma
            over_probs = scipy_stats.norm.cdf(z_scores)

            # Quantile calibration on adjustments
            in_range = np.mean((actual_adjustments >= raw_p10) & (actual_adjustments <= raw_p90))
            below_p10 = np.mean(actual_adjustments < raw_p10)
            above_p90 = np.mean(actual_adjustments > raw_p90)

        else:
            # Legacy raw mode
            actuals = y_test
            preds_final = raw_preds
            pred_adjustments = raw_preds - lines
            actual_adjustments = y_test - lines

            model_mae = np.mean(np.abs(preds_final - actuals))
            line_mae = np.mean(np.abs(lines - actuals))

            model_rmse = np.sqrt(np.mean((preds_final - actuals) ** 2))
            line_rmse = np.sqrt(np.mean((lines - actuals) ** 2))

            actual_over = actuals > lines
            model_over = preds_final > lines

            sigma = np.maximum((raw_p90 - raw_p10) / 2.56, 0.5)
            z_scores = (preds_final - lines) / sigma
            over_probs = scipy_stats.norm.cdf(z_scores)

            in_range = np.mean((actuals >= raw_p10) & (actuals <= raw_p90))
            below_p10 = np.mean(actuals < raw_p10)
            above_p90 = np.mean(actuals > raw_p90)

        mae_delta = line_mae - model_mae

        # R-squared
        ss_res = np.sum((preds_final - actuals) ** 2)
        ss_tot = np.sum((actuals - actuals.mean()) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-9)

        # Directional accuracy
        directional_acc = np.mean(model_over == actual_over)

        # Edge signal monotonicity
        edge_bins = self._edge_monotonicity_check(
            pred_adjustments, actual_over, over_probs, sigma
        )

        # ROI at different thresholds
        roi_by_threshold = self._roi_analysis(
            pred_adjustments, sigma, actual_over,
            actuals, lines, over_probs,
            X_test["over_odds"].values, X_test["under_odds"].values,
        )

        # Uncertainty vs error correlation
        uncertainties = raw_p90 - raw_p10
        abs_errors = np.abs(preds_final - actuals)
        if len(uncertainties) > 10:
            unc_error_corr = float(np.corrcoef(uncertainties, abs_errors)[0, 1])
        else:
            unc_error_corr = 0.0

        # Print detailed results
        print(f"\n{'='*70}")
        print(f"  {prop_type.upper()} EVALUATION ({len(X_test)} test samples, "
              f"{self.target_mode} mode)")
        print(f"{'='*70}")

        print(f"\n  --- Accuracy ---")
        print(f"  Model MAE:        {model_mae:.3f}")
        print(f"  Line MAE:         {line_mae:.3f}")
        beats = "YES" if model_mae < line_mae else "NO"
        print(f"  Beats line:       {beats} (delta: {mae_delta:+.3f})")
        print(f"  Model RMSE:       {model_rmse:.3f}")
        print(f"  Line RMSE:        {line_rmse:.3f}")
        print(f"  R-squared:        {r2:.4f}")
        print(f"  Direction acc:    {directional_acc:.1%}")

        if self.target_mode == "line_anchor":
            print(f"\n  --- Adjustment Distribution ---")
            print(f"  Mean predicted adj: {pred_adjustments.mean():+.3f}")
            print(f"  Mean actual adj:    {actual_adjustments.mean():+.3f}")
            print(f"  Actual over rate:   {actual_over.mean():.1%}")
            print(f"  Predicted over rate:{model_over.mean():.1%}")

        print(f"\n  --- Quantile Calibration ---")
        print(f"  In P10-P90 range: {in_range:.1%} (target: 80%)")
        print(f"  Below P10:        {below_p10:.1%} (target: 10%)")
        print(f"  Above P90:        {above_p90:.1%} (target: 10%)")

        print(f"\n  --- Edge Signal (MONOTONICITY TEST) ---")
        print(f"  (Higher model confidence should -> higher hit rate)")
        if edge_bins:
            print(f"  {'Confidence Bin':>20s} {'N':>6s} {'Hit Rate':>10s} {'Avg Edge':>10s}")
            print(f"  {'-'*20} {'-'*6} {'-'*10} {'-'*10}")
            for eb in edge_bins:
                print(f"  {eb['bin']:>20s} {eb['n']:6d} {eb['hit_rate']:10.1%} {eb['avg_edge']:+10.1%}")

            # Check monotonicity
            hit_rates = [eb["hit_rate"] for eb in edge_bins if eb["n"] >= 20]
            if len(hit_rates) >= 3:
                is_monotonic = all(hit_rates[i] <= hit_rates[i + 1]
                                   for i in range(len(hit_rates) - 1))
                print(f"  Monotonic: {'YES -- edge signal is real' if is_monotonic else 'NO -- edge signal is noisy'}")

        print(f"\n  --- ROI by Edge Threshold ---")
        if roi_by_threshold:
            print(f"  {'Threshold':>12s} {'N Bets':>8s} {'Win Rate':>10s} {'ROI':>8s} {'Units':>8s}")
            print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")
            for rt in roi_by_threshold:
                print(f"  {rt['threshold']:>12s} {rt['n_bets']:8d} {rt['win_rate']:10.1%} "
                      f"{rt['roi']:+7.1%} {rt['units']:+8.1f}")

        print(f"\n  --- Confidence ---")
        print(f"  Uncertainty-Error corr: {unc_error_corr:.3f} "
              f"({'good' if unc_error_corr > 0.2 else 'weak'}: "
              f"higher uncertainty should mean higher error)")

        return {
            "prop_type": prop_type,
            "target_mode": self.target_mode,
            "n_test": len(X_test),
            "model_mae": float(model_mae),
            "line_mae": float(line_mae),
            "mae_delta": float(mae_delta),
            "beats_line": bool(model_mae < line_mae),
            "model_rmse": float(model_rmse),
            "line_rmse": float(line_rmse),
            "r2": float(r2),
            "directional_accuracy": float(directional_acc),
            "quantile_in_range_80": float(in_range),
            "below_p10": float(below_p10),
            "above_p90": float(above_p90),
            "uncertainty_error_corr": float(unc_error_corr),
            "edge_bins": edge_bins,
            "roi_by_threshold": roi_by_threshold,
        }

    def _edge_monotonicity_check(self, adjustments, actual_over, over_probs, sigma):
        """
        Bin predictions by confidence (|adjustment| / sigma).
        Check if hit rate (predicted side == actual side) increases
        monotonically with confidence.
        """
        model_says_over = adjustments > 0
        correct = (model_says_over == actual_over)

        # Confidence = |adjustment| / sigma
        confidence = np.abs(adjustments) / sigma

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
                    "avg_edge": float(np.abs(adjustments[mask]).mean()),
                })
        return results

    def _roi_analysis(self, adjustments, sigma, actual_over, actuals, lines,
                      over_probs, over_odds, under_odds):
        """
        Simulate flat-bet ROI at different edge thresholds.
        """
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

                hit = (side == "OVER" and actuals[i] > lines[i]) or \
                      (side == "UNDER" and actuals[i] < lines[i])
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

    def _american_to_prob(self, odds):
        if odds == 0:
            return 0.5
        if odds > 0:
            return 100.0 / (odds + 100.0)
        return abs(odds) / (abs(odds) + 100.0)

    def _american_to_decimal(self, odds):
        if odds == 0:
            return 2.0
        if odds > 0:
            return 1.0 + odds / 100.0
        return 1.0 + 100.0 / abs(odds)

    def _print_summary(self, all_results):
        """Print combined summary table."""
        if not all_results:
            return

        print(f"\n{'='*90}")
        print(f"  COMBINED EVALUATION SUMMARY ({self.target_mode.upper()} MODE)")
        print(f"{'='*90}")
        print(f"  {'Prop':20s} {'N Test':>7s} {'Model MAE':>10s} {'Line MAE':>10s} "
              f"{'Beats?':>7s} {'Delta':>8s} {'R2':>7s} {'Dir Acc':>8s} {'P10-P90':>8s}")
        print(f"  {'-'*20} {'-'*7} {'-'*10} {'-'*10} {'-'*7} {'-'*8} {'-'*7} {'-'*8} {'-'*8}")

        for pt, r in all_results.items():
            print(f"  {pt:20s} {r['n_test']:7d} {r['model_mae']:10.3f} {r['line_mae']:10.3f} "
                  f"{'YES' if r['beats_line'] else 'NO':>7s} {r['mae_delta']:+8.3f} "
                  f"{r['r2']:7.4f} {r['directional_accuracy']:7.1%} "
                  f"{r['quantile_in_range_80']:7.1%}")

        # Overall aggregated metrics
        total_test = sum(r["n_test"] for r in all_results.values())
        weighted_model_mae = sum(r["model_mae"] * r["n_test"] for r in all_results.values()) / total_test
        weighted_line_mae = sum(r["line_mae"] * r["n_test"] for r in all_results.values()) / total_test
        weighted_dir_acc = sum(r["directional_accuracy"] * r["n_test"] for r in all_results.values()) / total_test
        n_beating = sum(1 for r in all_results.values() if r["beats_line"])

        print(f"\n  Overall (weighted): Model MAE={weighted_model_mae:.3f}, "
              f"Line MAE={weighted_line_mae:.3f}, "
              f"Dir Acc={weighted_dir_acc:.1%}, "
              f"Beating line: {n_beating}/{len(all_results)} props")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate v2 models")
    parser.add_argument("--prop", type=str, default=None,
                        help="Evaluate single prop type (e.g., points)")
    args = parser.parse_args()

    evaluator = ModelEvaluator()

    if args.prop:
        evaluator.run_full_evaluation(prop_types=[args.prop])
    else:
        evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()
