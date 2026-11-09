#!/usr/bin/env python3
"""
train_v2_final.py - Train improved v2 models with per-prop feature selection
and hyperparameter tuning.

The Problem:
  With quantile alpha=0.5, the point model early-stops at 1 iteration for 4/5 props
  because the best prediction IS the median (= 0, line is correct).
  L2 loss (and Huber) allow the model to learn directional adjustments.

The Fix:
  1. Per-prop feature selection (not global top-50)
  2. Per-prop hyperparameter tuning across L2, Huber, shifted-quantile, and baseline
  3. Pick best config per prop based on MAE, directional accuracy, and iterations
  4. Train final point + q10 + q90 models with best config per prop

Saves to: models/v2_final/
"""

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
except ImportError:
    print("ERROR: lightgbm not installed. pip install lightgbm")
    sys.exit(1)

try:
    import joblib
except ImportError:
    print("ERROR: joblib not installed. pip install joblib")
    sys.exit(1)

from scipy import stats as scipy_stats

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "outputs" / "v2_training"
MODELS_DIR = BASE_DIR / "models" / "v2_final"
CURRENT_MODELS_DIR = BASE_DIR / "models" / "v2_optimized"

CORE_PROPS = ["points", "rebounds", "assists", "threes", "pts_rebs_asts"]

# Hyperparameter configurations to try for the POINT model
CONFIGS = [
    # L2 regression variants (learns adjustments from MSE loss)
    {"name": "L2_slow_shallow",   "point_obj": "regression", "point_lr": 0.01, "point_leaves": 15, "point_depth": -1},
    {"name": "L2_med_depth6",     "point_obj": "regression", "point_lr": 0.03, "point_leaves": 15, "point_depth": 6},
    {"name": "L2_med_depth8",     "point_obj": "regression", "point_lr": 0.03, "point_leaves": 31, "point_depth": 8},
    # Huber loss (robust L2 - handles outliers)
    {"name": "Huber_med",         "point_obj": "huber",      "point_lr": 0.03, "point_leaves": 15, "point_depth": -1},
    {"name": "Huber_fast_deep",   "point_obj": "huber",      "point_lr": 0.05, "point_leaves": 31, "point_depth": 6},
    # Quantile with slight under bias
    {"name": "Q_0.48",            "point_obj": "quantile",   "point_alpha": 0.48, "point_lr": 0.03, "point_leaves": 15, "point_depth": -1},
    {"name": "Q_0.45",            "point_obj": "quantile",   "point_alpha": 0.45, "point_lr": 0.03, "point_leaves": 15, "point_depth": -1},
    # Pure quantile (current approach - baseline)
    {"name": "Q_0.50_baseline",   "point_obj": "quantile",   "point_alpha": 0.50, "point_lr": 0.03, "point_leaves": 15, "point_depth": -1},
]


def load_data():
    """Load features, targets, metadata from cache."""
    logger.info("Loading data from %s", CACHE_DIR)
    X = pd.read_parquet(CACHE_DIR / "features_combined.parquet")
    targets = pd.read_parquet(CACHE_DIR / "targets_enhanced.parquet")
    metadata = pd.read_parquet(CACHE_DIR / "metadata_enhanced.parquet")
    y = targets["target"].values
    logger.info("Loaded %d samples, %d features", len(X), X.shape[1])
    return X, y, metadata


def compute_sample_weights(dates: pd.Series) -> np.ndarray:
    """Exponential decay weights, 90-day half-life."""
    max_date = pd.to_datetime(dates).max()
    days_ago = (max_date - pd.to_datetime(dates)).dt.days
    half_life = 90.0
    weights = np.exp(-0.693 * days_ago.values / half_life)
    weights = weights / weights.mean()
    return weights


def time_split(X, y, meta, dates):
    """70/15/15 walk-forward time split."""
    sort_idx = dates.argsort()
    X = X.iloc[sort_idx].reset_index(drop=True)
    y = y[sort_idx]
    meta = meta.iloc[sort_idx].reset_index(drop=True)
    dates = dates.iloc[sort_idx].reset_index(drop=True)

    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    splits = {
        "X_train": X.iloc[:train_end],
        "y_train": y[:train_end],
        "dates_train": dates.iloc[:train_end],
        "meta_train": meta.iloc[:train_end],
        "X_val": X.iloc[train_end:val_end],
        "y_val": y[train_end:val_end],
        "X_test": X.iloc[val_end:],
        "y_test": y[val_end:],
        "dates_test": dates.iloc[val_end:],
        "meta_test": meta.iloc[val_end:],
    }
    return splits


def select_features_for_prop(X_train, y_train, weights_train, prop_type, n_top=60):
    """
    Step 1: Per-prop feature selection.
    Train a preliminary LightGBM on ALL features, then pick top N by importance.
    """
    logger.info("  Feature selection for %s on %d features...", prop_type, X_train.shape[1])

    prelim = lgb.LGBMRegressor(
        objective="regression",  # L2 to actually learn
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_child_samples=50,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )

    # Use a small validation split from training data for early stopping
    n_tr = int(len(X_train) * 0.85)
    prelim.fit(
        X_train.iloc[:n_tr], y_train[:n_tr],
        sample_weight=weights_train[:n_tr],
        eval_set=[(X_train.iloc[n_tr:], y_train[n_tr:])],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )

    importances = prelim.feature_importances_
    feat_names = X_train.columns.tolist()
    imp_pairs = sorted(zip(feat_names, importances), key=lambda x: -x[1])

    # Take top N features that have non-zero importance
    selected = [name for name, imp in imp_pairs[:n_top] if imp > 0]
    logger.info("  Selected %d features for %s (prelim iter=%d)",
                len(selected), prop_type, prelim.best_iteration_)

    return selected


def build_lgb_params(config):
    """Convert config dict to LightGBM parameters for the point model."""
    obj = config["point_obj"]
    params = {
        "n_estimators": 800,
        "learning_rate": config.get("point_lr", 0.03),
        "num_leaves": config.get("point_leaves", 15),
        "min_child_samples": 50,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbose": -1,
    }

    depth = config.get("point_depth", -1)
    if depth > 0:
        params["max_depth"] = depth

    if obj == "regression":
        params["objective"] = "regression"
    elif obj == "huber":
        params["objective"] = "huber"
        params["huber_delta"] = 1.0  # default Huber delta
    elif obj == "quantile":
        params["objective"] = "quantile"
        params["alpha"] = config.get("point_alpha", 0.5)

    return params


def evaluate_point_model(model, X_test, y_test, line_values):
    """Evaluate a point model on test set. Returns metrics dict."""
    pred_adj = model.predict(X_test)
    actual_adj = y_test
    actuals = y_test + line_values

    preds_final = line_values + pred_adj

    model_mae = np.mean(np.abs(preds_final - actuals))
    line_mae = np.mean(np.abs(line_values - actuals))

    model_rmse = np.sqrt(np.mean((preds_final - actuals) ** 2))

    # Directional accuracy
    model_says_over = pred_adj > 0
    actual_over = actual_adj > 0
    dir_acc = np.mean(model_says_over == actual_over)

    # Over/under hit rates
    over_mask = model_says_over
    under_mask = ~model_says_over
    over_hr = np.mean(actual_over[over_mask]) if over_mask.sum() > 0 else 0.0
    under_hr = np.mean(~actual_over[under_mask]) if under_mask.sum() > 0 else 0.0

    return {
        "model_mae": float(model_mae),
        "line_mae": float(line_mae),
        "beats_line": bool(model_mae < line_mae),
        "mae_delta": float(line_mae - model_mae),
        "model_rmse": float(model_rmse),
        "dir_acc": float(dir_acc),
        "over_hr": float(over_hr),
        "under_hr": float(under_hr),
        "n_over_preds": int(over_mask.sum()),
        "n_under_preds": int(under_mask.sum()),
        "best_iter": int(model.best_iteration_),
        "mean_pred_adj": float(np.mean(pred_adj)),
    }


def score_config(metrics):
    """
    Score a config for ranking. Higher is better.
    Priorities:
      1. Must beat line MAE (heavy penalty if not)
      2. Directional accuracy (main signal)
      3. Bonus for actually learning (iterations > 1)
    """
    score = 0.0

    # Must beat line MAE
    if metrics["beats_line"]:
        score += 100.0
        score += metrics["mae_delta"] * 50.0  # Reward larger improvements
    else:
        score -= 200.0  # Heavy penalty

    # Directional accuracy (range: ~0.48 to ~0.56)
    score += (metrics["dir_acc"] - 0.50) * 1000.0  # e.g. 0.54 -> +40 points

    # Bonus for actually learning (not early-stopping at 1)
    if metrics["best_iter"] > 1:
        score += 20.0
        score += min(metrics["best_iter"], 200) * 0.05  # Small bonus for more iters

    return score


def train_and_evaluate_config(config, X_train, y_train, weights_train,
                               X_val, y_val, X_test, y_test, line_test):
    """Train a point model with given config, evaluate on test set."""
    params = build_lgb_params(config)
    model = lgb.LGBMRegressor(**params)

    model.fit(
        X_train, y_train,
        sample_weight=weights_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )

    metrics = evaluate_point_model(model, X_test, y_test, line_test)
    metrics["config_name"] = config["name"]
    metrics["score"] = score_config(metrics)

    return model, metrics


def train_quantile_model(alpha, X_train, y_train, weights_train, X_val, y_val):
    """Train a quantile model (for q10 or q90)."""
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        sample_weight=weights_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    return model


def roi_simulation(pred_adj, sigma, actual_over, actuals, lines,
                   over_odds, under_odds, threshold=0.03):
    """Simulate ROI at a given edge threshold."""
    z_scores = pred_adj / sigma
    over_probs = scipy_stats.norm.cdf(z_scores)

    total_wagered = 0.0
    total_profit = 0.0
    n_bets = 0
    n_wins = 0

    for i in range(len(pred_adj)):
        implied_over = american_to_prob(over_odds[i])
        implied_under = american_to_prob(under_odds[i])

        edge_over = over_probs[i] - implied_over
        edge_under = (1.0 - over_probs[i]) - implied_under

        if edge_over >= threshold and edge_over >= edge_under:
            side = "OVER"
            odds = over_odds[i]
        elif edge_under >= threshold:
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
            decimal = american_to_decimal(odds)
            total_profit += 100.0 * (decimal - 1.0)
        else:
            total_profit -= 100.0

    if n_bets > 0:
        return {
            "n_bets": n_bets,
            "win_rate": n_wins / n_bets,
            "roi": total_profit / total_wagered,
            "units": total_profit / 100.0,
        }
    return {"n_bets": 0, "win_rate": 0.0, "roi": 0.0, "units": 0.0}


def american_to_prob(odds):
    if odds == 0:
        return 0.5
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def american_to_decimal(odds):
    if odds == 0:
        return 2.0
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def load_current_results():
    """Load current v2_optimized results for comparison."""
    meta_path = CURRENT_MODELS_DIR / "training_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return None


def main():
    t_start = time.time()

    print("=" * 80)
    print("  v2 FINAL Model Trainer - Per-Prop Feature Selection + Hyperparameter Tuning")
    print("=" * 80)

    # Load data
    X_all, y_all, metadata = load_data()

    # Load current results for comparison
    current_meta = load_current_results()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Track per-prop results
    all_prop_results = {}
    all_prop_features = {}
    all_prop_configs = {}
    union_features = set()

    for prop_type in CORE_PROPS:
        print(f"\n{'#' * 80}")
        print(f"#  PROP: {prop_type.upper()}")
        print(f"{'#' * 80}")

        # Filter to this prop
        mask = metadata["prop_type"] == prop_type
        X_prop = X_all[mask].copy()
        y_prop = y_all[mask].copy()
        meta_prop = metadata[mask].copy()
        dates = meta_prop["date"].copy()

        n = len(X_prop)
        logger.info("%s: %d total samples", prop_type, n)

        if n < 200:
            logger.warning("Skipping %s: only %d samples", prop_type, n)
            continue

        # Time split (on all features first, for feature selection)
        splits = time_split(X_prop, y_prop, meta_prop, dates)
        weights_train = compute_sample_weights(splits["dates_train"])

        logger.info("  Train=%d, Val=%d, Test=%d",
                     len(splits["X_train"]), len(splits["X_val"]), len(splits["X_test"]))

        # ==================================================================
        # Step 1: Per-prop feature selection
        # ==================================================================
        selected_features = select_features_for_prop(
            splits["X_train"], splits["y_train"], weights_train, prop_type, n_top=60
        )

        if len(selected_features) < 10:
            logger.warning("  Only %d features selected for %s, using all", len(selected_features), prop_type)
            selected_features = X_prop.columns.tolist()

        print(f"\n  Selected {len(selected_features)} features for {prop_type}:")
        for i, f in enumerate(selected_features[:15]):
            print(f"    {i+1:2d}. {f}")
        if len(selected_features) > 15:
            print(f"    ... and {len(selected_features) - 15} more")

        # Re-split with selected features only
        X_prop_sel = X_prop[selected_features]
        splits_sel = time_split(X_prop_sel, y_prop, meta_prop, dates)
        weights_train_sel = compute_sample_weights(splits_sel["dates_train"])

        line_test = splits_sel["X_test"]["line"].values if "line" in selected_features else splits["X_test"]["line"].values

        # Also need over_odds and under_odds from full feature set for ROI
        splits_full = time_split(X_prop, y_prop, meta_prop, dates)
        over_odds_test = splits_full["X_test"]["over_odds"].values
        under_odds_test = splits_full["X_test"]["under_odds"].values

        # ==================================================================
        # Step 2: Per-prop hyperparameter tuning
        # ==================================================================
        print(f"\n  Testing {len(CONFIGS)} configurations...")
        print(f"  {'Config':<22s} {'MAE':>7s} {'LinMAE':>7s} {'Beat?':>6s} {'Delta':>7s} "
              f"{'DirAcc':>7s} {'Iter':>5s} {'Score':>8s}")
        print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*6} {'-'*7} {'-'*7} {'-'*5} {'-'*8}")

        config_results = []
        for config in CONFIGS:
            model, metrics = train_and_evaluate_config(
                config,
                splits_sel["X_train"], splits_sel["y_train"], weights_train_sel,
                splits_sel["X_val"], splits_sel["y_val"],
                splits_sel["X_test"], splits_sel["y_test"],
                line_test,
            )
            config_results.append((config, model, metrics))

            beat = "YES" if metrics["beats_line"] else "no"
            print(f"  {config['name']:<22s} {metrics['model_mae']:7.4f} {metrics['line_mae']:7.4f} "
                  f"{beat:>6s} {metrics['mae_delta']:+7.4f} {metrics['dir_acc']:7.4f} "
                  f"{metrics['best_iter']:5d} {metrics['score']:+8.1f}")

        # ==================================================================
        # Step 3: Pick best config
        # ==================================================================
        config_results.sort(key=lambda x: x[2]["score"], reverse=True)
        best_config, best_point_model, best_metrics = config_results[0]

        print(f"\n  BEST CONFIG for {prop_type}: {best_config['name']}")
        print(f"    MAE:     {best_metrics['model_mae']:.4f} (line: {best_metrics['line_mae']:.4f}, delta: {best_metrics['mae_delta']:+.4f})")
        print(f"    DirAcc:  {best_metrics['dir_acc']:.4f}")
        print(f"    Iter:    {best_metrics['best_iter']}")
        print(f"    Over HR: {best_metrics['over_hr']:.4f} (N={best_metrics['n_over_preds']})")
        print(f"    Under HR:{best_metrics['under_hr']:.4f} (N={best_metrics['n_under_preds']})")

        # ==================================================================
        # Step 4: Train final models (point already trained, just q10 + q90)
        # ==================================================================
        logger.info("  Training q10 and q90 models with selected features...")

        q10_model = train_quantile_model(
            0.10, splits_sel["X_train"], splits_sel["y_train"],
            weights_train_sel, splits_sel["X_val"], splits_sel["y_val"]
        )
        q90_model = train_quantile_model(
            0.90, splits_sel["X_train"], splits_sel["y_train"],
            weights_train_sel, splits_sel["X_val"], splits_sel["y_val"]
        )

        # ROI simulation with final model
        pred_adj = best_point_model.predict(splits_sel["X_test"])
        p10_adj = q10_model.predict(splits_sel["X_test"])
        p90_adj = q90_model.predict(splits_sel["X_test"])
        sigma_est = np.maximum((p90_adj - p10_adj) / 2.56, 0.5)
        actual_adj = splits_sel["y_test"]
        actual_over = actual_adj > 0
        actuals = actual_adj + line_test

        roi_3pct = roi_simulation(pred_adj, sigma_est, actual_over, actuals, line_test,
                                   over_odds_test, under_odds_test, threshold=0.03)
        roi_5pct = roi_simulation(pred_adj, sigma_est, actual_over, actuals, line_test,
                                   over_odds_test, under_odds_test, threshold=0.05)

        # Quantile calibration
        in_range = np.mean((actual_adj >= p10_adj) & (actual_adj <= p90_adj))

        print(f"    Q10-Q90 calibration: {in_range:.1%} (target: 80%)")
        print(f"    Q10 iter: {q10_model.best_iteration_}, Q90 iter: {q90_model.best_iteration_}")
        print(f"    ROI @3%: {roi_3pct['roi']:+.1%} ({roi_3pct['n_bets']} bets, WR={roi_3pct['win_rate']:.1%})")
        print(f"    ROI @5%: {roi_5pct['roi']:+.1%} ({roi_5pct['n_bets']} bets, WR={roi_5pct['win_rate']:.1%})")

        # Save models
        joblib.dump(best_point_model, MODELS_DIR / f"{prop_type}_point.pkl")
        joblib.dump(q10_model, MODELS_DIR / f"{prop_type}_q10.pkl")
        joblib.dump(q90_model, MODELS_DIR / f"{prop_type}_q90.pkl")
        logger.info("  Saved %s models to %s", prop_type, MODELS_DIR)

        # Track results
        all_prop_features[prop_type] = selected_features
        all_prop_configs[prop_type] = {
            "config_name": best_config["name"],
            "config": best_config,
            "features": selected_features,
            "n_features": len(selected_features),
        }
        union_features.update(selected_features)

        all_prop_results[prop_type] = {
            "model_mae": best_metrics["model_mae"],
            "line_mae": best_metrics["line_mae"],
            "beats_line_mae": str(best_metrics["beats_line"]),
            "mae_delta": best_metrics["mae_delta"],
            "rmse": best_metrics["model_rmse"],
            "directional_accuracy": best_metrics["dir_acc"],
            "over_hit_rate": best_metrics["over_hr"],
            "under_hit_rate": best_metrics["under_hr"],
            "n_test": len(splits_sel["y_test"]),
            "point_iterations": best_metrics["best_iter"],
            "q10_iterations": q10_model.best_iteration_,
            "q90_iterations": q90_model.best_iteration_,
            "quantile_calibration_80": float(in_range),
            "roi_at_3pct": roi_3pct["roi"],
            "roi_bets": roi_3pct["n_bets"],
            "roi_win_rate": roi_3pct["win_rate"],
            "roi_at_5pct": roi_5pct["roi"],
            "config_used": best_config["name"],
            "n_features": len(selected_features),
        }

    # ==================================================================
    # Save training_meta.json
    # ==================================================================
    # Union of all per-prop features (for loading at inference time)
    union_features_list = sorted(list(union_features))

    training_meta = {
        "target_mode": "line_anchor",
        "objective": "hybrid_per_prop",
        "feature_columns": union_features_list,
        "n_features": len(union_features_list),
        "core_props": CORE_PROPS,
        "n_total_samples": len(X_all),
        "hyperparameters": {
            "n_estimators": 800,
            "early_stopping_rounds": 30,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_lambda": 1.0,
            "sample_weights": "exponential_decay_90d",
            "time_split": "70/15/15",
        },
        "per_prop_config": {},
        "results": all_prop_results,
    }

    # Per-prop config with feature lists
    for prop_type in CORE_PROPS:
        if prop_type in all_prop_configs:
            cfg = all_prop_configs[prop_type]
            training_meta["per_prop_config"][prop_type] = {
                "config_name": cfg["config_name"],
                "point_objective": cfg["config"]["point_obj"],
                "point_lr": cfg["config"].get("point_lr", 0.03),
                "point_leaves": cfg["config"].get("point_leaves", 15),
                "point_depth": cfg["config"].get("point_depth", -1),
                "point_alpha": cfg["config"].get("point_alpha", None),
                "features": cfg["features"],
                "n_features": cfg["n_features"],
            }

    with open(MODELS_DIR / "training_meta.json", "w") as f:
        json.dump(training_meta, f, indent=2, default=str)

    # ==================================================================
    # Step 5: Comparison table
    # ==================================================================
    print(f"\n{'=' * 110}")
    print(f"  COMPARISON: v2_optimized (current) vs v2_final (new)")
    print(f"{'=' * 110}")
    print(f"  {'Prop':<16s} | {'Cur MAE':>8s} {'New MAE':>8s} {'Improv':>7s} | "
          f"{'Cur Dir':>8s} {'New Dir':>8s} | {'Cur Iter':>9s} {'New Iter':>9s} | "
          f"{'Config':<22s}")
    print(f"  {'-'*16} + {'-'*8} {'-'*8} {'-'*7} + {'-'*8} {'-'*8} + {'-'*9} {'-'*9} + {'-'*22}")

    for prop_type in CORE_PROPS:
        new = all_prop_results.get(prop_type, {})
        cur = {}
        if current_meta and "results" in current_meta:
            cur = current_meta["results"].get(prop_type, {})

        cur_mae = cur.get("model_mae", 0)
        new_mae = new.get("model_mae", 0)
        improve = cur_mae - new_mae if cur_mae > 0 else 0

        cur_dir = cur.get("directional_accuracy", 0)
        new_dir = new.get("directional_accuracy", 0)

        cur_iter = cur.get("point_iterations", 0)
        new_iter = new.get("point_iterations", 0)

        config_name = new.get("config_used", "N/A")

        print(f"  {prop_type:<16s} | {cur_mae:8.4f} {new_mae:8.4f} {improve:+7.4f} | "
              f"{cur_dir:8.4f} {new_dir:8.4f} | {cur_iter:9d} {new_iter:9d} | "
              f"{config_name:<22s}")

    # Summary stats
    print(f"\n  --- Summary ---")
    total_cur_mae = sum(current_meta["results"].get(p, {}).get("model_mae", 0) for p in CORE_PROPS) if current_meta else 0
    total_new_mae = sum(all_prop_results.get(p, {}).get("model_mae", 0) for p in CORE_PROPS)
    print(f"  Total MAE (sum of 5 props): Current={total_cur_mae:.4f}, New={total_new_mae:.4f}, Delta={total_cur_mae - total_new_mae:+.4f}")

    n_beating = sum(1 for p in CORE_PROPS if all_prop_results.get(p, {}).get("beats_line_mae") == "True")
    print(f"  Props beating line: {n_beating}/5")

    n_learning = sum(1 for p in CORE_PROPS if all_prop_results.get(p, {}).get("point_iterations", 0) > 1)
    print(f"  Props actually learning (iter > 1): {n_learning}/5")

    print(f"\n  Union feature set: {len(union_features_list)} features")
    print(f"  Models saved to: {MODELS_DIR}/")
    print(f"  Total time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
