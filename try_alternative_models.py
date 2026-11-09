#!/usr/bin/env python3
"""
Alternative ML Approaches for NBA Player Prop Prediction

Compares CatBoost, XGBoost, Stacked Ensemble, Huber/Quantile LightGBM,
and a Two-Stage model against the baseline LightGBM regressor.

Data: line-anchor adjustments (actual - DK_line) with 255 features.
Evaluated on points and threes prop types.
"""

import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, accuracy_score

import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
import xgboost as xgb

warnings.filterwarnings("ignore")

# ─── Data Loading ───────────────────────────────────────────────────────────

DATA_DIR = Path("outputs/v2_training")
OUTPUT_PATH = Path("outputs/alternative_models_results.json")

def load_data():
    features = pd.read_parquet(DATA_DIR / "features_combined.parquet")
    targets = pd.read_parquet(DATA_DIR / "targets_enhanced.parquet")
    metadata = pd.read_parquet(DATA_DIR / "metadata_enhanced.parquet")
    return features, targets["target"], metadata


def filter_prop(features, targets, metadata, prop_type):
    mask = metadata["prop_type"] == prop_type
    return (
        features[mask].reset_index(drop=True),
        targets[mask].reset_index(drop=True),
        metadata[mask].reset_index(drop=True),
    )


def time_split(features, targets, metadata):
    """70/15/15 time-based split sorted by date."""
    dates = metadata["date"].values
    sorted_idx = np.argsort(dates)

    n = len(sorted_idx)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_idx = sorted_idx[:train_end]
    val_idx = sorted_idx[train_end:val_end]
    test_idx = sorted_idx[val_end:]

    splits = {}
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        splits[name] = {
            "X": features.iloc[idx],
            "y": targets.iloc[idx],
            "meta": metadata.iloc[idx],
        }

    print(f"  Split sizes - train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")
    print(f"  Train dates: {splits['train']['meta']['date'].min()} to {splits['train']['meta']['date'].max()}")
    print(f"  Val dates:   {splits['val']['meta']['date'].min()} to {splits['val']['meta']['date'].max()}")
    print(f"  Test dates:  {splits['test']['meta']['date'].min()} to {splits['test']['meta']['date'].max()}")

    return splits


def compute_sample_weights(meta, half_life_days=90):
    """Exponential decay weights with 90-day half-life from last date in set."""
    dates = pd.to_datetime(meta["date"])
    max_date = dates.max()
    days_ago = (max_date - dates).dt.days.values.astype(float)
    decay_rate = np.log(2) / half_life_days
    weights = np.exp(-decay_rate * days_ago)
    return weights


# ─── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_model(preds, splits_test, model_name):
    """Compute all metrics for a model's predictions on the test set."""
    y_true = splits_test["y"].values
    meta = splits_test["meta"]
    lines = meta["line"].values
    actuals = meta["actual"].values

    # Model prediction = line + predicted_adjustment
    model_proj = lines + preds
    residual = actuals - model_proj
    mae = float(np.mean(np.abs(residual)))

    # Line MAE (baseline)
    line_residual = actuals - lines
    line_mae = float(np.mean(np.abs(line_residual)))

    beats_line = mae < line_mae

    # Directional accuracy: does the model correctly predict over/under?
    pred_direction = np.sign(preds)  # positive = over, negative = under
    actual_direction = np.sign(actuals - lines)
    # Exclude pushes (actual == line)
    non_push = actual_direction != 0
    if non_push.sum() > 0:
        dir_acc = float(np.mean(pred_direction[non_push] == actual_direction[non_push]))
    else:
        dir_acc = 0.5

    # ROI at >= 3% edge threshold
    roi_info = compute_roi(preds, splits_test)

    result = {
        "model": model_name,
        "mae": round(mae, 4),
        "line_mae": round(line_mae, 4),
        "mae_improvement": round(line_mae - mae, 4),
        "mae_improvement_pct": round((line_mae - mae) / line_mae * 100, 2),
        "beats_line": beats_line,
        "directional_accuracy": round(dir_acc, 4),
        "n_test": int(len(y_true)),
        **roi_info,
    }
    return result


def compute_roi(preds, splits_test, edge_threshold=0.03):
    """
    Compute simulated ROI at >= 3% edge using probability from z-score.
    Uses -110 standard juice.
    """
    meta = splits_test["meta"]
    lines = meta["line"].values
    actuals = meta["actual"].values
    X_test = splits_test["X"]

    # Get std dev from features (target_std_l10)
    if "target_std_l10" in X_test.columns:
        std_devs = X_test["target_std_l10"].values
    elif "target_std_season" in X_test.columns:
        std_devs = X_test["target_std_season"].values
    else:
        std_devs = np.full(len(preds), 5.0)

    # Avoid division by zero
    std_devs = np.maximum(std_devs, 0.5)

    # Model projected value
    model_proj = lines + preds

    # Z-score: how far our projection is from the line
    z_scores = (model_proj - lines) / std_devs  # = preds / std_devs

    # Over probability
    over_prob = 1 - stats.norm.cdf(-z_scores)  # P(actual > line)

    # Implied probability at -110 juice
    implied_prob = 110 / (110 + 100)  # ~0.5238

    # Determine bet direction and edge
    # If over_prob > 0.5, bet OVER; edge = over_prob - implied_prob
    # If over_prob < 0.5, bet UNDER; edge = (1 - over_prob) - implied_prob
    bet_over = over_prob > 0.5
    our_prob = np.where(bet_over, over_prob, 1 - over_prob)
    edge = our_prob - implied_prob

    # Filter to bets above threshold
    qualifies = edge >= edge_threshold
    n_bets = int(qualifies.sum())

    if n_bets == 0:
        return {
            "roi_edge_threshold": edge_threshold,
            "n_bets_above_threshold": 0,
            "roi_pct": 0.0,
            "hit_rate": 0.0,
            "avg_edge": 0.0,
        }

    # Check actual outcomes for qualifying bets
    bet_over_q = bet_over[qualifies]
    actuals_q = actuals[qualifies]
    lines_q = lines[qualifies]

    # Did the bet win?
    actual_over = actuals_q > lines_q
    actual_under = actuals_q < lines_q
    wins = np.where(bet_over_q, actual_over, actual_under)
    pushes = actuals_q == lines_q

    # Simple flat betting at -110
    profit = np.where(wins, 100 / 110, np.where(pushes, 0, -1.0))
    total_profit = float(profit.sum())
    roi = total_profit / n_bets * 100  # per unit

    hit_rate = float(wins.sum() / n_bets)

    return {
        "roi_edge_threshold": edge_threshold,
        "n_bets_above_threshold": n_bets,
        "roi_pct": round(roi, 2),
        "hit_rate": round(hit_rate, 4),
        "avg_edge": round(float(edge[qualifies].mean()) * 100, 2),
    }


# ─── Model Training Functions ──────────────────────────────────────────────

def train_baseline_lgb(splits):
    """Baseline LightGBM with L2 loss (current model)."""
    w_train = compute_sample_weights(splits["train"]["meta"])
    w_val = compute_sample_weights(splits["val"]["meta"])

    train_ds = lgb.Dataset(splits["train"]["X"], splits["train"]["y"], weight=w_train)
    val_ds = lgb.Dataset(splits["val"]["X"], splits["val"]["y"], weight=w_val, reference=train_ds)

    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 6,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
    }

    callbacks = [lgb.early_stopping(30), lgb.log_evaluation(0)]
    model = lgb.train(
        params, train_ds, num_boost_round=500,
        valid_sets=[val_ds], callbacks=callbacks,
    )

    preds = model.predict(splits["test"]["X"])
    best_iter = model.best_iteration
    return preds, best_iter


def train_catboost(splits):
    """Approach 1: CatBoost Regressor."""
    w_train = compute_sample_weights(splits["train"]["meta"])

    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        early_stopping_rounds=30,
        verbose=0,
    )

    model.fit(
        splits["train"]["X"], splits["train"]["y"],
        sample_weight=w_train,
        eval_set=(splits["val"]["X"], splits["val"]["y"]),
    )

    preds = model.predict(splits["test"]["X"])
    best_iter = model.best_iteration_
    return preds, best_iter


def train_xgboost(splits):
    """Approach 2: XGBoost Regressor."""
    w_train = compute_sample_weights(splits["train"]["meta"])

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=50,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=30,
    )

    model.fit(
        splits["train"]["X"], splits["train"]["y"],
        sample_weight=w_train,
        eval_set=[(splits["val"]["X"], splits["val"]["y"])],
        verbose=False,
    )

    preds = model.predict(splits["test"]["X"])
    best_iter = model.best_iteration
    return preds, best_iter


def train_stacked_ensemble(splits, lgb_preds, cat_preds, xgb_preds):
    """
    Approach 3: Stacked ensemble - simple avg + optimized weighted avg.
    Uses validation set to find optimal weights.
    """
    # Get individual model val predictions for weight optimization
    # We need to retrain to get val preds (or we can use a trick)
    # For simplicity, let's compute val preds from the models too
    # Actually, we already have test preds. Let's use val set to optimize weights.

    # Re-train to get val predictions
    w_train = compute_sample_weights(splits["train"]["meta"])

    # LGB val preds
    train_ds = lgb.Dataset(splits["train"]["X"], splits["train"]["y"], weight=w_train)
    val_ds = lgb.Dataset(splits["val"]["X"], splits["val"]["y"], reference=train_ds)
    lgb_params = {
        "objective": "regression", "metric": "mae", "learning_rate": 0.05,
        "num_leaves": 63, "max_depth": 6, "min_child_samples": 50,
        "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1,
        "reg_lambda": 1.0, "verbose": -1,
    }
    lgb_model = lgb.train(
        lgb_params, train_ds, num_boost_round=500,
        valid_sets=[val_ds], callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    lgb_val = lgb_model.predict(splits["val"]["X"])

    # CatBoost val preds
    cat_model = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                                   l2_leaf_reg=3, early_stopping_rounds=30, verbose=0)
    cat_model.fit(splits["train"]["X"], splits["train"]["y"], sample_weight=w_train,
                  eval_set=(splits["val"]["X"], splits["val"]["y"]))
    cat_val = cat_model.predict(splits["val"]["X"])

    # XGBoost val preds
    xgb_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                                   min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
                                   early_stopping_rounds=30)
    xgb_model.fit(splits["train"]["X"], splits["train"]["y"], sample_weight=w_train,
                  eval_set=[(splits["val"]["X"], splits["val"]["y"])], verbose=False)
    xgb_val = xgb_model.predict(splits["val"]["X"])

    y_val = splits["val"]["y"].values

    # Simple average
    simple_avg_test = (lgb_preds + cat_preds + xgb_preds) / 3

    # Optimize weights on val set
    def val_mae(weights):
        w = np.array(weights)
        w = w / w.sum()  # normalize
        pred = w[0] * lgb_val + w[1] * cat_val + w[2] * xgb_val
        return np.mean(np.abs(y_val - pred))

    result = minimize(val_mae, x0=[1/3, 1/3, 1/3],
                      method="Nelder-Mead",
                      options={"maxiter": 1000})
    opt_w = np.array(result.x)
    opt_w = opt_w / opt_w.sum()

    weighted_avg_test = opt_w[0] * lgb_preds + opt_w[1] * cat_preds + opt_w[2] * xgb_preds

    print(f"    Optimized weights: LGB={opt_w[0]:.3f}, Cat={opt_w[1]:.3f}, XGB={opt_w[2]:.3f}")

    return simple_avg_test, weighted_avg_test, opt_w


def train_lgb_huber(splits):
    """Approach 4a: LightGBM with Huber loss (robust to outliers)."""
    w_train = compute_sample_weights(splits["train"]["meta"])
    w_val = compute_sample_weights(splits["val"]["meta"])

    train_ds = lgb.Dataset(splits["train"]["X"], splits["train"]["y"], weight=w_train)
    val_ds = lgb.Dataset(splits["val"]["X"], splits["val"]["y"], weight=w_val, reference=train_ds)

    params = {
        "objective": "huber",
        "huber_delta": 1.0,
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 6,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
    }

    callbacks = [lgb.early_stopping(30), lgb.log_evaluation(0)]
    model = lgb.train(
        params, train_ds, num_boost_round=500,
        valid_sets=[val_ds], callbacks=callbacks,
    )

    preds = model.predict(splits["test"]["X"])
    best_iter = model.best_iteration
    return preds, best_iter


def train_lgb_quantile(splits):
    """Approach 4b: LightGBM with quantile loss (median regression)."""
    w_train = compute_sample_weights(splits["train"]["meta"])
    w_val = compute_sample_weights(splits["val"]["meta"])

    train_ds = lgb.Dataset(splits["train"]["X"], splits["train"]["y"], weight=w_train)
    val_ds = lgb.Dataset(splits["val"]["X"], splits["val"]["y"], weight=w_val, reference=train_ds)

    params = {
        "objective": "quantile",
        "alpha": 0.5,
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 6,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
    }

    callbacks = [lgb.early_stopping(30), lgb.log_evaluation(0)]
    model = lgb.train(
        params, train_ds, num_boost_round=500,
        valid_sets=[val_ds], callbacks=callbacks,
    )

    preds = model.predict(splits["test"]["X"])
    best_iter = model.best_iteration
    return preds, best_iter


def train_two_stage(splits):
    """
    Approach 5: Two-stage model.
    Stage 1: Binary classifier (OVER vs UNDER)
    Stage 2: Regressor predicting magnitude |actual - line|
    Combined: classifier direction * regressor magnitude
    """
    w_train = compute_sample_weights(splits["train"]["meta"])

    y_train = splits["train"]["y"].values
    y_val = splits["val"]["y"].values

    # Stage 1: Binary classifier (is target > 0 => OVER)
    # Exclude pushes (target == 0) from classifier training
    train_nonzero = y_train != 0
    cls_y_train = (y_train[train_nonzero] > 0).astype(int)
    cls_X_train = splits["train"]["X"].iloc[train_nonzero]
    cls_w_train = w_train[train_nonzero]

    val_nonzero = y_val != 0
    cls_y_val = (y_val[val_nonzero] > 0).astype(int)
    cls_X_val = splits["val"]["X"].iloc[val_nonzero]

    # Train classifier with LightGBM
    cls_train_ds = lgb.Dataset(cls_X_train, cls_y_train, weight=cls_w_train)
    cls_val_ds = lgb.Dataset(cls_X_val, cls_y_val, reference=cls_train_ds)

    cls_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 6,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
    }

    cls_model = lgb.train(
        cls_params, cls_train_ds, num_boost_round=500,
        valid_sets=[cls_val_ds],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )

    # Stage 2: Magnitude regressor (predicts |target|)
    mag_y_train = np.abs(y_train)
    mag_y_val = np.abs(y_val)

    mag_train_ds = lgb.Dataset(splits["train"]["X"], mag_y_train, weight=w_train)
    mag_val_ds = lgb.Dataset(splits["val"]["X"], mag_y_val, reference=mag_train_ds)

    mag_params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 6,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
    }

    mag_model = lgb.train(
        mag_params, mag_train_ds, num_boost_round=500,
        valid_sets=[mag_val_ds],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )

    # Combine: direction from classifier, magnitude from regressor
    cls_probs = cls_model.predict(splits["test"]["X"])  # P(over)
    mag_preds = mag_model.predict(splits["test"]["X"])  # |adjustment|

    # Direction: if P(over) > 0.5, positive; else negative
    # Scale by confidence: direction * magnitude * (2 * |P - 0.5|)
    direction = np.where(cls_probs > 0.5, 1, -1)
    confidence = 2 * np.abs(cls_probs - 0.5)  # 0 to 1
    preds = direction * mag_preds * confidence

    cls_iter = cls_model.best_iteration
    mag_iter = mag_model.best_iteration

    # Also report classifier accuracy on test
    y_test = splits["test"]["y"].values
    test_nonzero = y_test != 0
    if test_nonzero.sum() > 0:
        cls_test_probs = cls_probs[test_nonzero]
        cls_test_true = (y_test[test_nonzero] > 0).astype(int)
        cls_acc = float(np.mean((cls_test_probs > 0.5) == cls_test_true))
        print(f"    Classifier accuracy on test: {cls_acc:.4f}")

    return preds, (cls_iter, mag_iter)


# ─── Main Runner ────────────────────────────────────────────────────────────

def run_all_models_for_prop(features, targets, metadata, prop_type):
    """Run all model approaches for a single prop type."""
    print(f"\n{'='*70}")
    print(f"  PROP TYPE: {prop_type.upper()}")
    print(f"{'='*70}")

    X, y, meta = filter_prop(features, targets, metadata, prop_type)
    print(f"  Total samples: {len(X)}")

    splits = time_split(X, y, meta)
    results = []

    # 1) Baseline LightGBM
    print(f"\n  [1/8] Baseline LightGBM (L2)...")
    t0 = time.time()
    lgb_preds, lgb_iter = train_baseline_lgb(splits)
    dt = time.time() - t0
    res = evaluate_model(lgb_preds, splits["test"], "LightGBM (L2 baseline)")
    res["early_stop_iter"] = lgb_iter
    res["train_time_sec"] = round(dt, 1)
    results.append(res)
    print(f"    MAE={res['mae']:.4f}, Line MAE={res['line_mae']:.4f}, Beats={res['beats_line']}, DirAcc={res['directional_accuracy']:.4f}, Iter={lgb_iter}, Time={dt:.1f}s")

    # 2) CatBoost
    print(f"\n  [2/8] CatBoost...")
    t0 = time.time()
    cat_preds, cat_iter = train_catboost(splits)
    dt = time.time() - t0
    res = evaluate_model(cat_preds, splits["test"], "CatBoost")
    res["early_stop_iter"] = cat_iter
    res["train_time_sec"] = round(dt, 1)
    results.append(res)
    print(f"    MAE={res['mae']:.4f}, Line MAE={res['line_mae']:.4f}, Beats={res['beats_line']}, DirAcc={res['directional_accuracy']:.4f}, Iter={cat_iter}, Time={dt:.1f}s")

    # 3) XGBoost
    print(f"\n  [3/8] XGBoost...")
    t0 = time.time()
    xgb_preds, xgb_iter = train_xgboost(splits)
    dt = time.time() - t0
    res = evaluate_model(xgb_preds, splits["test"], "XGBoost")
    res["early_stop_iter"] = xgb_iter
    res["train_time_sec"] = round(dt, 1)
    results.append(res)
    print(f"    MAE={res['mae']:.4f}, Line MAE={res['line_mae']:.4f}, Beats={res['beats_line']}, DirAcc={res['directional_accuracy']:.4f}, Iter={xgb_iter}, Time={dt:.1f}s")

    # 4) Stacked Ensemble (simple avg + weighted avg)
    print(f"\n  [4/8] Stacked Ensemble (simple avg)...")
    print(f"  [5/8] Stacked Ensemble (weighted avg)...")
    t0 = time.time()
    simple_preds, weighted_preds, opt_weights = train_stacked_ensemble(
        splits, lgb_preds, cat_preds, xgb_preds
    )
    dt = time.time() - t0

    res_simple = evaluate_model(simple_preds, splits["test"], "Ensemble (simple avg)")
    res_simple["early_stop_iter"] = "N/A"
    res_simple["train_time_sec"] = round(dt, 1)
    results.append(res_simple)
    print(f"    Simple Avg - MAE={res_simple['mae']:.4f}, Beats={res_simple['beats_line']}, DirAcc={res_simple['directional_accuracy']:.4f}")

    res_weighted = evaluate_model(weighted_preds, splits["test"], "Ensemble (weighted avg)")
    res_weighted["early_stop_iter"] = "N/A"
    res_weighted["ensemble_weights"] = {
        "lgb": round(float(opt_weights[0]), 3),
        "catboost": round(float(opt_weights[1]), 3),
        "xgboost": round(float(opt_weights[2]), 3),
    }
    res_weighted["train_time_sec"] = round(dt, 1)
    results.append(res_weighted)
    print(f"    Weighted Avg - MAE={res_weighted['mae']:.4f}, Beats={res_weighted['beats_line']}, DirAcc={res_weighted['directional_accuracy']:.4f}")

    # 5) LightGBM Huber
    print(f"\n  [6/8] LightGBM (Huber)...")
    t0 = time.time()
    huber_preds, huber_iter = train_lgb_huber(splits)
    dt = time.time() - t0
    res = evaluate_model(huber_preds, splits["test"], "LightGBM (Huber)")
    res["early_stop_iter"] = huber_iter
    res["train_time_sec"] = round(dt, 1)
    results.append(res)
    print(f"    MAE={res['mae']:.4f}, Line MAE={res['line_mae']:.4f}, Beats={res['beats_line']}, DirAcc={res['directional_accuracy']:.4f}, Iter={huber_iter}, Time={dt:.1f}s")

    # 6) LightGBM Quantile (median)
    print(f"\n  [7/8] LightGBM (Quantile, alpha=0.5)...")
    t0 = time.time()
    quantile_preds, quant_iter = train_lgb_quantile(splits)
    dt = time.time() - t0
    res = evaluate_model(quantile_preds, splits["test"], "LightGBM (Quantile)")
    res["early_stop_iter"] = quant_iter
    res["train_time_sec"] = round(dt, 1)
    results.append(res)
    print(f"    MAE={res['mae']:.4f}, Line MAE={res['line_mae']:.4f}, Beats={res['beats_line']}, DirAcc={res['directional_accuracy']:.4f}, Iter={quant_iter}, Time={dt:.1f}s")

    # 7) Two-stage model
    print(f"\n  [8/8] Two-Stage (classifier + magnitude)...")
    t0 = time.time()
    twostage_preds, twostage_iter = train_two_stage(splits)
    dt = time.time() - t0
    res = evaluate_model(twostage_preds, splits["test"], "Two-Stage")
    res["early_stop_iter"] = str(twostage_iter)
    res["train_time_sec"] = round(dt, 1)
    results.append(res)
    print(f"    MAE={res['mae']:.4f}, Line MAE={res['line_mae']:.4f}, Beats={res['beats_line']}, DirAcc={res['directional_accuracy']:.4f}, Iter={twostage_iter}, Time={dt:.1f}s")

    return results


def print_comparison_table(results, prop_type):
    """Print a formatted comparison table."""
    print(f"\n{'='*120}")
    print(f"  COMPARISON TABLE: {prop_type.upper()}")
    print(f"{'='*120}")
    header = f"{'Model':<30} {'MAE':>7} {'LinMAE':>7} {'Improve':>8} {'Impr%':>7} {'Beats':>6} {'DirAcc':>7} {'ROI%':>7} {'Bets':>5} {'HitR':>6} {'Edge%':>6} {'Iter':>6}"
    print(header)
    print("-" * 120)

    # Sort by MAE
    sorted_results = sorted(results, key=lambda x: x["mae"])

    for r in sorted_results:
        iter_str = str(r.get("early_stop_iter", "N/A"))[:6]
        beats_str = "YES" if r["beats_line"] else "no"
        print(
            f"{r['model']:<30} "
            f"{r['mae']:>7.4f} "
            f"{r['line_mae']:>7.4f} "
            f"{r['mae_improvement']:>+8.4f} "
            f"{r['mae_improvement_pct']:>+6.2f}% "
            f"{beats_str:>6} "
            f"{r['directional_accuracy']:>7.4f} "
            f"{r['roi_pct']:>+6.2f}% "
            f"{r['n_bets_above_threshold']:>5} "
            f"{r['hit_rate']:>6.4f} "
            f"{r['avg_edge']:>5.2f}% "
            f"{iter_str:>6}"
        )
    print()


def main():
    print("Loading data...")
    features, targets, metadata = load_data()
    print(f"Loaded {len(features)} rows, {features.shape[1]} features")

    all_results = {}

    for prop_type in ["points", "threes"]:
        results = run_all_models_for_prop(features, targets, metadata, prop_type)
        all_results[prop_type] = results
        print_comparison_table(results, prop_type)

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")

    # Print overall summary
    print(f"\n{'='*120}")
    print("  OVERALL SUMMARY")
    print(f"{'='*120}")
    for prop_type, results in all_results.items():
        print(f"\n  {prop_type.upper()}:")
        best = min(results, key=lambda x: x["mae"])
        best_dir = max(results, key=lambda x: x["directional_accuracy"])
        best_roi = max(results, key=lambda x: x["roi_pct"])
        print(f"    Best MAE:            {best['model']} ({best['mae']:.4f}, {'beats' if best['beats_line'] else 'does NOT beat'} line @ {best['line_mae']:.4f})")
        print(f"    Best Dir. Accuracy:  {best_dir['model']} ({best_dir['directional_accuracy']:.4f})")
        if best_roi['n_bets_above_threshold'] > 0:
            print(f"    Best ROI (>=3% edge):{best_roi['model']} ({best_roi['roi_pct']:+.2f}%, {best_roi['n_bets_above_threshold']} bets, {best_roi['hit_rate']:.4f} hit rate)")
        else:
            print(f"    Best ROI (>=3% edge): No qualifying bets for any model")


if __name__ == "__main__":
    main()
