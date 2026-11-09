#!/usr/bin/env python3
"""
LightGBM Hyperparameter Search for NBA Player Prop Prediction

Three-stage smart search:
  Stage 1: learning_rate x max_depth x num_leaves (64 combos)
  Stage 2: min_child_samples x subsample x colsample_bytree (36 combos)
  Stage 3: reg_alpha x reg_lambda (9 combos)

Uses line-anchor target (actual - DK_line) with exponential decay sample weights.
"""

import json
import time
import warnings
from itertools import product
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_DIR = Path("outputs")
DATA_DIR = OUTPUT_DIR / "v2_training"
RESULTS_PATH = OUTPUT_DIR / "hyperparam_search_results.json"

PROP_TYPES = ["points", "threes"]
DECAY_HALF_LIFE_DAYS = 90


def load_data():
    """Load features, targets, and metadata."""
    features = pd.read_parquet(DATA_DIR / "features_combined.parquet")
    targets = pd.read_parquet(DATA_DIR / "targets_enhanced.parquet")
    metadata = pd.read_parquet(DATA_DIR / "metadata_enhanced.parquet")
    return features, targets, metadata


def compute_sample_weights(dates: pd.Series) -> np.ndarray:
    """Exponential decay weights with 90-day half-life from the most recent date."""
    dates_dt = pd.to_datetime(dates)
    max_date = dates_dt.max()
    days_ago = (max_date - dates_dt).dt.total_seconds() / 86400.0
    decay_rate = np.log(2) / DECAY_HALF_LIFE_DAYS
    weights = np.exp(-decay_rate * days_ago.values)
    return weights


def time_split(features, targets, metadata, prop_type):
    """Filter by prop_type, sort by date, split 70/15/15."""
    mask = metadata["prop_type"] == prop_type
    feat = features.loc[mask].copy()
    targ = targets.loc[mask].copy()
    meta = metadata.loc[mask].copy()

    # Sort by date
    sort_idx = meta["date"].argsort()
    feat = feat.iloc[sort_idx].reset_index(drop=True)
    targ = targ.iloc[sort_idx].reset_index(drop=True)
    meta = meta.iloc[sort_idx].reset_index(drop=True)

    n = len(feat)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    splits = {
        "X_train": feat.iloc[:train_end],
        "y_train": targ["target"].iloc[:train_end],
        "X_val": feat.iloc[train_end:val_end],
        "y_val": targ["target"].iloc[train_end:val_end],
        "X_test": feat.iloc[val_end:],
        "y_test": targ["target"].iloc[val_end:],
        "meta_train": meta.iloc[:train_end],
        "meta_val": meta.iloc[train_end:val_end],
        "meta_test": meta.iloc[val_end:],
    }

    # Sample weights for training set
    splits["w_train"] = compute_sample_weights(splits["meta_train"]["date"])

    return splits


def directional_accuracy(y_true, y_pred):
    """Percentage where sign of prediction matches sign of actual."""
    # Only count cases where actual != 0 (non-trivial)
    nonzero = y_true != 0
    if nonzero.sum() == 0:
        return 0.0
    same_sign = np.sign(y_pred[nonzero]) == np.sign(y_true.values[nonzero])
    return float(same_sign.mean()) * 100.0


def train_and_evaluate(params, splits):
    """Train a LightGBM model and return metrics."""
    full_params = {
        "objective": "regression",
        "metric": "mae",
        "verbosity": -1,
        "n_jobs": -1,
        "random_state": 42,
        **params,
    }

    model = lgb.LGBMRegressor(**full_params)

    model.fit(
        splits["X_train"],
        splits["y_train"],
        sample_weight=splits["w_train"],
        eval_set=[(splits["X_val"], splits["y_val"])],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )

    best_iter = model.best_iteration_

    # Validation MAE
    val_pred = model.predict(splits["X_val"])
    val_mae = float(np.mean(np.abs(val_pred - splits["y_val"].values)))

    # Test MAE
    test_pred = model.predict(splits["X_test"])
    test_mae = float(np.mean(np.abs(test_pred - splits["y_test"].values)))

    # Test directional accuracy
    dir_acc = directional_accuracy(splits["y_test"], test_pred)

    return {
        "best_iteration": best_iter,
        "val_mae": round(val_mae, 5),
        "test_mae": round(test_mae, 5),
        "test_dir_acc": round(dir_acc, 2),
    }


def run_stage(stage_name, param_combos, fixed_params, splits, prop_type):
    """Run a set of param combos, return list of results sorted by val_mae."""
    results = []
    total = len(param_combos)
    print(f"\n{'='*70}")
    print(f"  {prop_type.upper()} - {stage_name}: {total} combinations")
    print(f"{'='*70}")

    for i, combo in enumerate(param_combos, 1):
        params = {**fixed_params, **combo}
        t0 = time.time()
        metrics = train_and_evaluate(params, splits)
        elapsed = time.time() - t0

        result = {**combo, **metrics}
        results.append(result)

        if i % 10 == 0 or i == total:
            print(
                f"  [{i:3d}/{total}] val_mae={metrics['val_mae']:.5f}  "
                f"test_mae={metrics['test_mae']:.5f}  "
                f"dir_acc={metrics['test_dir_acc']:.1f}%  "
                f"iters={metrics['best_iteration']:4d}  "
                f"({elapsed:.1f}s)"
            )

    results.sort(key=lambda x: x["val_mae"])
    return results


def generate_combos(grid: dict) -> list:
    """Generate all combinations from a param grid dict."""
    keys = list(grid.keys())
    vals = list(grid.values())
    combos = []
    for combo in product(*vals):
        combos.append(dict(zip(keys, combo)))
    return combos


def main():
    print("Loading data...")
    features, targets, metadata = load_data()
    print(f"  Total rows: {len(features)}")

    all_results = {}

    for prop_type in PROP_TYPES:
        print(f"\n{'#'*70}")
        print(f"  PROP: {prop_type.upper()}")
        print(f"{'#'*70}")

        splits = time_split(features, targets, metadata, prop_type)
        print(f"  Train: {len(splits['X_train'])}, Val: {len(splits['X_val'])}, Test: {len(splits['X_test'])}")

        prop_results = {"stage1": [], "stage2": [], "stage3": []}

        # ── Stage 1: learning_rate × max_depth × num_leaves ──
        stage1_grid = {
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "max_depth": [4, 6, 8, -1],
            "num_leaves": [15, 31, 63, 127],
        }
        stage1_defaults = {
            "n_estimators": 800,
            "min_child_samples": 50,
            "subsample": 0.85,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        }
        stage1_combos = generate_combos(stage1_grid)
        stage1_results = run_stage(
            "Stage 1 (lr × depth × leaves)", stage1_combos, stage1_defaults, splits, prop_type
        )
        prop_results["stage1"] = stage1_results

        # Extract best from stage 1
        best1 = {k: stage1_results[0][k] for k in stage1_grid.keys()}
        print(f"\n  >> Best Stage 1: {best1}")
        print(
            f"     val_mae={stage1_results[0]['val_mae']:.5f}  "
            f"test_mae={stage1_results[0]['test_mae']:.5f}  "
            f"iters={stage1_results[0]['best_iteration']}"
        )

        # ── Stage 2: min_child_samples × subsample × colsample_bytree ──
        stage2_grid = {
            "min_child_samples": [20, 50, 100, 200],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.5, 0.7, 0.9],
        }
        stage2_fixed = {
            "n_estimators": 800,
            **best1,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        }
        stage2_combos = generate_combos(stage2_grid)
        stage2_results = run_stage(
            "Stage 2 (samples × subsample × colsample)", stage2_combos, stage2_fixed, splits, prop_type
        )
        prop_results["stage2"] = stage2_results

        best2 = {k: stage2_results[0][k] for k in stage2_grid.keys()}
        print(f"\n  >> Best Stage 2: {best2}")
        print(
            f"     val_mae={stage2_results[0]['val_mae']:.5f}  "
            f"test_mae={stage2_results[0]['test_mae']:.5f}  "
            f"iters={stage2_results[0]['best_iteration']}"
        )

        # ── Stage 3: reg_alpha × reg_lambda ──
        stage3_grid = {
            "reg_alpha": [0, 0.1, 1.0],
            "reg_lambda": [0, 1.0, 10.0],
        }
        stage3_fixed = {
            "n_estimators": 800,
            **best1,
            **best2,
        }
        stage3_combos = generate_combos(stage3_grid)
        stage3_results = run_stage(
            "Stage 3 (regularization)", stage3_combos, stage3_fixed, splits, prop_type
        )
        prop_results["stage3"] = stage3_results

        best3 = {k: stage3_results[0][k] for k in stage3_grid.keys()}
        print(f"\n  >> Best Stage 3: {best3}")
        print(
            f"     val_mae={stage3_results[0]['val_mae']:.5f}  "
            f"test_mae={stage3_results[0]['test_mae']:.5f}  "
            f"iters={stage3_results[0]['best_iteration']}"
        )

        # ── Final best config ──
        final_best = {
            "n_estimators": 800,
            **best1,
            **best2,
            **best3,
        }
        print(f"\n  ** FINAL BEST CONFIG for {prop_type.upper()} **")
        print(f"     {json.dumps(final_best, indent=2)}")
        print(
            f"     val_mae={stage3_results[0]['val_mae']:.5f}  "
            f"test_mae={stage3_results[0]['test_mae']:.5f}  "
            f"dir_acc={stage3_results[0]['test_dir_acc']:.1f}%  "
            f"iters={stage3_results[0]['best_iteration']}"
        )

        prop_results["final_best_config"] = final_best
        prop_results["final_best_metrics"] = {
            "val_mae": stage3_results[0]["val_mae"],
            "test_mae": stage3_results[0]["test_mae"],
            "test_dir_acc": stage3_results[0]["test_dir_acc"],
            "best_iteration": stage3_results[0]["best_iteration"],
        }

        all_results[prop_type] = prop_results

    # Save all results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # ── Print Top 10 by test_mae for each prop ──
    print(f"\n{'='*80}")
    print("  TOP 10 CONFIGURATIONS BY TEST MAE")
    print(f"{'='*80}")

    for prop_type in PROP_TYPES:
        print(f"\n  --- {prop_type.upper()} ---")

        # Collect all results across stages, tag each with its full params
        all_configs = []

        # From stage 1
        s1_best = all_results[prop_type]["final_best_config"]
        for r in all_results[prop_type]["stage1"]:
            config = {
                "learning_rate": r["learning_rate"],
                "max_depth": r["max_depth"],
                "num_leaves": r["num_leaves"],
                "min_child_samples": 50,
                "subsample": 0.85,
                "colsample_bytree": 0.7,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "stage": "1",
            }
            config.update({
                "best_iteration": r["best_iteration"],
                "val_mae": r["val_mae"],
                "test_mae": r["test_mae"],
                "test_dir_acc": r["test_dir_acc"],
            })
            all_configs.append(config)

        # From stage 2
        best1 = {
            "learning_rate": s1_best["learning_rate"],
            "max_depth": s1_best["max_depth"],
            "num_leaves": s1_best["num_leaves"],
        }
        for r in all_results[prop_type]["stage2"]:
            config = {
                **best1,
                "min_child_samples": r["min_child_samples"],
                "subsample": r["subsample"],
                "colsample_bytree": r["colsample_bytree"],
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "stage": "2",
            }
            config.update({
                "best_iteration": r["best_iteration"],
                "val_mae": r["val_mae"],
                "test_mae": r["test_mae"],
                "test_dir_acc": r["test_dir_acc"],
            })
            all_configs.append(config)

        # From stage 3
        best2_keys = ["min_child_samples", "subsample", "colsample_bytree"]
        best2 = {k: s1_best[k] for k in best2_keys}
        for r in all_results[prop_type]["stage3"]:
            config = {
                **best1,
                **best2,
                "reg_alpha": r["reg_alpha"],
                "reg_lambda": r["reg_lambda"],
                "stage": "3",
            }
            config.update({
                "best_iteration": r["best_iteration"],
                "val_mae": r["val_mae"],
                "test_mae": r["test_mae"],
                "test_dir_acc": r["test_dir_acc"],
            })
            all_configs.append(config)

        # Sort by test_mae
        all_configs.sort(key=lambda x: x["test_mae"])

        print(f"  {'Rank':>4}  {'Stage':>5}  {'LR':>5}  {'Depth':>5}  {'Leaves':>6}  "
              f"{'MinCS':>5}  {'Sub':>4}  {'ColS':>4}  {'RegA':>5}  {'RegL':>5}  "
              f"{'Iters':>5}  {'ValMAE':>8}  {'TestMAE':>8}  {'DirAcc':>6}")
        print(f"  {'-'*4}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*6}  "
              f"{'-'*5}  {'-'*4}  {'-'*4}  {'-'*5}  {'-'*5}  "
              f"{'-'*5}  {'-'*8}  {'-'*8}  {'-'*6}")

        for rank, c in enumerate(all_configs[:10], 1):
            print(
                f"  {rank:4d}  {c['stage']:>5}  {c['learning_rate']:5.2f}  "
                f"{c['max_depth']:5d}  {c['num_leaves']:6d}  "
                f"{c['min_child_samples']:5d}  {c['subsample']:4.2f}  "
                f"{c['colsample_bytree']:4.2f}  {c['reg_alpha']:5.1f}  "
                f"{c['reg_lambda']:5.1f}  "
                f"{c['best_iteration']:5d}  {c['val_mae']:8.5f}  "
                f"{c['test_mae']:8.5f}  {c['test_dir_acc']:5.1f}%"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
