#!/usr/bin/env python3
"""
Model Experiments A-D: Test fundamentally different modeling approaches
for NBA player prop prediction.

Approach A: Binary Classification (predict OVER/UNDER directly)
Approach B: Game Script -> Minutes -> Stats Pipeline
Approach C: Player Role Clustering (separate models per cluster)
Approach D: Opponent Scheme Interaction Features

All use walk-forward time-split: 70% train, 15% val, 15% test.
Results reported on TEST set only.
"""

import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error
from scipy.stats import norm

warnings.filterwarnings('ignore')

# ─── Data Loading ───────────────────────────────────────────────────────────

def load_data():
    """Load features, targets, metadata and create time-split."""
    base = Path('outputs/v2_training')
    features = pd.read_parquet(base / 'features_combined.parquet')
    targets = pd.read_parquet(base / 'targets_enhanced.parquet')
    meta = pd.read_parquet(base / 'metadata_enhanced.parquet')

    # Align indices
    assert len(features) == len(targets) == len(meta)

    # Sort by date for walk-forward split
    sort_idx = meta['date'].argsort()
    features = features.iloc[sort_idx].reset_index(drop=True)
    targets = targets.iloc[sort_idx].reset_index(drop=True)
    meta = meta.iloc[sort_idx].reset_index(drop=True)

    n = len(features)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    split = {
        'train': slice(0, train_end),
        'val': slice(train_end, val_end),
        'test': slice(val_end, n),
    }

    print(f"Total rows: {n}")
    print(f"Train: {train_end} rows ({meta['date'].iloc[0]} to {meta['date'].iloc[train_end-1]})")
    print(f"Val:   {val_end - train_end} rows ({meta['date'].iloc[train_end]} to {meta['date'].iloc[val_end-1]})")
    print(f"Test:  {n - val_end} rows ({meta['date'].iloc[val_end]} to {meta['date'].iloc[n-1]})")

    return features, targets, meta, split


def compute_roi(predictions_over_prob, meta_slice, threshold, odds=-110):
    """Compute ROI for flat $100 bets at given confidence threshold.
    Bet OVER when P(over) > threshold, UNDER when P(under) > threshold.
    """
    actual = meta_slice['actual'].values
    line = meta_slice['line'].values
    p_over = predictions_over_prob

    # Payout at -110 odds: risk $110 to win $100, so profit = $100, loss = -$110
    # Normalized: bet $100 -> profit = 100/1.1 = $90.91 on win, lose $100 on loss
    payout = 100 / 1.1  # ~90.91

    total_wagered = 0
    total_profit = 0
    n_bets = 0

    for i in range(len(p_over)):
        if p_over[i] >= threshold:
            # Bet OVER
            total_wagered += 100
            n_bets += 1
            if actual[i] > line[i]:
                total_profit += payout
            else:
                total_profit -= 100
        elif (1 - p_over[i]) >= threshold:
            # Bet UNDER
            total_wagered += 100
            n_bets += 1
            if actual[i] < line[i]:
                total_profit += payout
            else:
                total_profit -= 100
        # Skip pushes and bets below threshold

    roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0
    return {
        'threshold': threshold,
        'n_bets': n_bets,
        'total_wagered': total_wagered,
        'total_profit': round(total_profit, 2),
        'roi_pct': round(roi, 2),
        'bet_pct': round(n_bets / len(p_over) * 100, 2) if len(p_over) > 0 else 0,
    }


def compute_accuracy(p_over, meta_slice):
    """Overall directional accuracy: predict over if p_over > 0.5, else under."""
    actual = meta_slice['actual'].values
    line = meta_slice['line'].values

    pred_over = p_over > 0.5
    actual_over = actual > line

    # Only count non-pushes
    mask = actual != line
    if mask.sum() == 0:
        return 0.0
    return accuracy_score(actual_over[mask], pred_over[mask])


def per_month_breakdown(p_over, meta_slice):
    """Per-month accuracy breakdown."""
    df = meta_slice.copy()
    df['p_over'] = p_over
    df['pred_over'] = df['p_over'] > 0.5
    df['actual_over'] = df['actual'] > df['line']
    df['correct'] = df['pred_over'] == df['actual_over']
    df['month'] = pd.to_datetime(df['date']).dt.to_period('M').astype(str)

    result = {}
    for month, grp in df.groupby('month'):
        mask = grp['actual'] != grp['line']
        if mask.sum() > 0:
            result[month] = {
                'accuracy': round(grp.loc[mask, 'correct'].mean(), 4),
                'n_legs': int(mask.sum()),
            }
    return result


def per_prop_breakdown(p_over, meta_slice):
    """Per-prop-type accuracy breakdown."""
    df = meta_slice.copy()
    df['p_over'] = p_over
    df['pred_over'] = df['p_over'] > 0.5
    df['actual_over'] = df['actual'] > df['line']
    df['correct'] = df['pred_over'] == df['actual_over']

    result = {}
    for pt, grp in df.groupby('prop_type'):
        mask = grp['actual'] != grp['line']
        if mask.sum() > 0:
            result[pt] = {
                'accuracy': round(grp.loc[mask, 'correct'].mean(), 4),
                'n_legs': int(mask.sum()),
            }
    return result


def evaluate_model(name, p_over, meta_slice, extra_metrics=None):
    """Full evaluation suite for any model that produces P(over)."""
    acc = compute_accuracy(p_over, meta_slice)
    per_prop = per_prop_breakdown(p_over, meta_slice)
    best_prop = max(per_prop, key=lambda k: per_prop[k]['accuracy']) if per_prop else 'N/A'

    # ROI at various thresholds
    roi_results = {}
    for thresh in [0.50, 0.52, 0.55, 0.58, 0.60, 0.65]:
        roi_results[str(thresh)] = compute_roi(p_over, meta_slice, thresh)

    # Log loss (binary)
    actual_over = (meta_slice['actual'].values > meta_slice['line'].values).astype(int)
    p_clipped = np.clip(p_over, 1e-7, 1 - 1e-7)
    try:
        ll = log_loss(actual_over, p_clipped)
    except Exception:
        ll = None

    result = {
        'name': name,
        'test_size': len(meta_slice),
        'test_date_range': f"{meta_slice['date'].min()} to {meta_slice['date'].max()}",
        'overall_accuracy': round(acc, 4),
        'log_loss': round(ll, 4) if ll is not None else None,
        'best_prop_type': best_prop,
        'per_prop_accuracy': per_prop,
        'roi_at_thresholds': roi_results,
        'per_month': per_month_breakdown(p_over, meta_slice),
    }
    if extra_metrics:
        result.update(extra_metrics)

    return result


# ─── Baseline: Regression Model ────────────────────────────────────────────

def run_baseline_regression(features, targets, meta, split):
    """Baseline: LightGBM regressor predicting (actual - line), then convert to P(over)."""
    print("\n" + "="*80)
    print("BASELINE: Regression Model (predict actual - line)")
    print("="*80)

    X_train = features.iloc[split['train']].values
    y_train = targets['target'].iloc[split['train']].values
    X_val = features.iloc[split['val']].values
    y_val = targets['target'].iloc[split['val']].values
    X_test = features.iloc[split['test']].values
    y_test = targets['target'].iloc[split['test']].values
    meta_test = meta.iloc[split['test']].reset_index(drop=True)

    model = LGBMRegressor(
        n_estimators=1000, learning_rate=0.03, max_depth=6, num_leaves=31,
        min_child_samples=50, subsample=0.7, colsample_bytree=0.7,
        verbosity=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            __import__('lightgbm').early_stopping(30, verbose=False),
            __import__('lightgbm').log_evaluation(0),
        ],
    )

    # Predict adjustment
    preds = model.predict(X_test)

    # Convert to P(over) via normal CDF
    # For each row: predicted adjustment relative to 0 (positive = over expected)
    # Use the test set's empirical std from training data features
    std_col_idx = list(features.columns).index('target_std_l10')
    stds = features.iloc[split['test']].values[:, std_col_idx]
    stds = np.maximum(stds, 0.5)  # floor

    z = preds / stds
    p_over = norm.cdf(z)

    # Also compute regression metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    result = evaluate_model(
        'Baseline Regression',
        p_over, meta_test,
        extra_metrics={'mae': round(mae, 4), 'rmse': round(rmse, 4), 'best_iteration': model.best_iteration_}
    )

    print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    print(f"  Accuracy: {result['overall_accuracy']:.4f}")
    print(f"  Log Loss: {result['log_loss']}")
    print(f"  Best prop: {result['best_prop_type']}")

    return result, model


# ─── Approach A: Binary Classification ──────────────────────────────────────

def run_approach_a(features, targets, meta, split):
    """Approach A: Binary classification - predict OVER/UNDER directly."""
    print("\n" + "="*80)
    print("APPROACH A: Binary Classification (predict P(over) directly)")
    print("="*80)

    # Binary target: 1 if actual > line, 0 if actual < line (drop pushes — none in this data)
    y_class = (meta['actual'] > meta['line']).astype(int)

    X_train = features.iloc[split['train']].values
    y_train = y_class.iloc[split['train']].values
    X_val = features.iloc[split['val']].values
    y_val = y_class.iloc[split['val']].values
    X_test = features.iloc[split['test']].values
    y_test = y_class.iloc[split['test']].values
    meta_test = meta.iloc[split['test']].reset_index(drop=True)

    # ── Global classifier ──
    print("\n  [A1] Global LGBMClassifier...")
    clf = LGBMClassifier(
        n_estimators=800, learning_rate=0.03, max_depth=6, num_leaves=31,
        min_child_samples=50, subsample=0.7, colsample_bytree=0.7,
        verbosity=-1,
    )
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            __import__('lightgbm').early_stopping(30, verbose=False),
            __import__('lightgbm').log_evaluation(0),
        ],
    )

    p_over_global = clf.predict_proba(X_test)[:, 1]
    global_result = evaluate_model(
        'A1: Global Classifier',
        p_over_global, meta_test,
        extra_metrics={'best_iteration': clf.best_iteration_}
    )
    print(f"    Accuracy: {global_result['overall_accuracy']:.4f}")
    print(f"    Log Loss: {global_result['log_loss']}")

    # ── Per-prop classifiers ──
    print("\n  [A2] Per-Prop Classifiers...")
    prop_types = meta['prop_type'].unique()
    per_prop_results = {}
    p_over_perprop = np.full(len(X_test), 0.5)

    for pt in prop_types:
        mask_train = (meta.iloc[split['train']]['prop_type'] == pt).values
        mask_val = (meta.iloc[split['val']]['prop_type'] == pt).values
        mask_test = (meta.iloc[split['test']]['prop_type'] == pt).values

        if mask_train.sum() < 100 or mask_test.sum() < 10:
            print(f"    {pt}: skipped (too few samples)")
            continue

        clf_pt = LGBMClassifier(
            n_estimators=800, learning_rate=0.03, max_depth=6, num_leaves=31,
            min_child_samples=50, subsample=0.7, colsample_bytree=0.7,
            verbosity=-1,
        )
        clf_pt.fit(
            X_train[mask_train], y_train[mask_train],
            eval_set=[(X_val[mask_val], y_val[mask_val])],
            callbacks=[
                __import__('lightgbm').early_stopping(30, verbose=False),
                __import__('lightgbm').log_evaluation(0),
            ],
        )

        p_pt = clf_pt.predict_proba(X_test[mask_test])[:, 1]
        p_over_perprop[mask_test] = p_pt

        meta_pt_test = meta_test.iloc[np.where(mask_test)[0]].reset_index(drop=True)
        acc_pt = compute_accuracy(p_pt, meta_pt_test)
        actual_over_pt = (meta_pt_test['actual'].values > meta_pt_test['line'].values).astype(int)
        ll_pt = log_loss(actual_over_pt, np.clip(p_pt, 1e-7, 1-1e-7))

        per_prop_results[pt] = {
            'accuracy': round(acc_pt, 4),
            'log_loss': round(ll_pt, 4),
            'n_test': int(mask_test.sum()),
            'best_iteration': clf_pt.best_iteration_,
        }
        print(f"    {pt}: acc={acc_pt:.4f}, logloss={ll_pt:.4f}, n={mask_test.sum()}")

    perprop_result = evaluate_model(
        'A2: Per-Prop Classifiers',
        p_over_perprop, meta_test,
        extra_metrics={'per_prop_model_results': per_prop_results}
    )
    print(f"  Combined accuracy: {perprop_result['overall_accuracy']:.4f}")
    print(f"  Combined log loss: {perprop_result['log_loss']}")

    return {
        'A1_global_classifier': global_result,
        'A2_per_prop_classifiers': perprop_result,
    }


# ─── Approach B: Game Script -> Minutes -> Stats Pipeline ──────────────────

def run_approach_b(features, targets, meta, split):
    """Approach B: Three-stage pipeline model."""
    print("\n" + "="*80)
    print("APPROACH B: Game Script -> Minutes -> Stats Pipeline")
    print("="*80)

    feat_df = features.copy()
    meta_df = meta.copy()

    # ── Stage 1: Predict blowout probability ──
    # Blowout proxy: spread_magnitude > 15 (we don't have final scores, but we can
    # use spread_magnitude as a proxy for expected blowout)
    # For the target, we approximate: blowout if |target| is very large (stat deviated a lot)
    # and blowout_risk is high. More precisely, use the blowout_risk feature.
    #
    # Since we lack actual game scores, we use the existing 'blowout_risk' feature
    # and the 'spread_magnitude' to build stage 1. The key insight: we predict whether
    # spread_magnitude > 10 from team-level features only.
    print("\n  [Stage 1] Blowout probability model...")

    # Use only game-context features for stage 1
    blowout_features = [
        'vegas_spread', 'spread_magnitude', 'vegas_total', 'is_home',
        'team_off_rating_l10', 'team_def_rating_l10', 'opp_def_rating_l10',
        'opp_off_rating_l10', 'team_pace_l10', 'opp_pace_l10',
        'off_vs_def_mismatch', 'projected_total', 'total_vs_avg',
        'is_b2b', 'days_rest',
    ]
    blowout_feat_idx = [list(feat_df.columns).index(c) for c in blowout_features if c in feat_df.columns]

    # Blowout target: spread_magnitude > 10 as proxy
    y_blowout = (feat_df['spread_magnitude'] > 10).astype(int).values

    X_train_b = feat_df.iloc[split['train']].values[:, blowout_feat_idx]
    y_train_b = y_blowout[split['train']]
    X_val_b = feat_df.iloc[split['val']].values[:, blowout_feat_idx]
    y_val_b = y_blowout[split['val']]
    X_test_b = feat_df.iloc[split['test']].values[:, blowout_feat_idx]

    clf_blowout = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=4, num_leaves=15,
        min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
        verbosity=-1,
    )
    clf_blowout.fit(
        X_train_b, y_train_b,
        eval_set=[(X_val_b, y_val_b)],
        callbacks=[
            __import__('lightgbm').early_stopping(20, verbose=False),
            __import__('lightgbm').log_evaluation(0),
        ],
    )
    blowout_prob_test = clf_blowout.predict_proba(X_test_b)[:, 1]
    print(f"    Blowout model trained. Mean blowout prob on test: {blowout_prob_test.mean():.3f}")

    # ── Stage 2: Predict minutes ──
    print("\n  [Stage 2] Minutes prediction model...")

    # Minutes target: mins_l3 as recent actual (proxy for actual minutes played in game)
    # We predict from context + blowout_prob
    mins_features = [
        'mins_season', 'mins_l5', 'mins_l10', 'mins_std', 'mins_trend',
        'mins_min_l5', 'is_b2b', 'is_3_in_4', 'days_rest',
        'spread_magnitude', 'vegas_total', 'is_home',
        'high_mins_pct_l10', 'games_played',
    ]
    mins_feat_idx = [list(feat_df.columns).index(c) for c in mins_features if c in feat_df.columns]

    # Target: mins_l3 (most recent minutes average — best proxy we have)
    y_mins = feat_df['mins_l3'].values

    X_train_m = feat_df.iloc[split['train']].values[:, mins_feat_idx]
    y_train_m = y_mins[split['train']]
    X_val_m = feat_df.iloc[split['val']].values[:, mins_feat_idx]
    y_val_m = y_mins[split['val']]
    X_test_m = feat_df.iloc[split['test']].values[:, mins_feat_idx]

    # Add blowout_prob from stage 1 as extra feature
    # We need blowout probs for train/val too
    blowout_prob_train = clf_blowout.predict_proba(feat_df.iloc[split['train']].values[:, blowout_feat_idx])[:, 1]
    blowout_prob_val = clf_blowout.predict_proba(feat_df.iloc[split['val']].values[:, blowout_feat_idx])[:, 1]

    X_train_m = np.column_stack([X_train_m, blowout_prob_train])
    X_val_m = np.column_stack([X_val_m, blowout_prob_val])
    X_test_m = np.column_stack([X_test_m, blowout_prob_test])

    reg_mins = LGBMRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=5, num_leaves=20,
        min_child_samples=50, subsample=0.7, colsample_bytree=0.8,
        verbosity=-1,
    )
    reg_mins.fit(
        X_train_m, y_train_m,
        eval_set=[(X_val_m, y_val_m)],
        callbacks=[
            __import__('lightgbm').early_stopping(20, verbose=False),
            __import__('lightgbm').log_evaluation(0),
        ],
    )
    pred_mins_test = reg_mins.predict(X_test_m)
    pred_mins_test = np.clip(pred_mins_test, 5, 48)  # reasonable bounds
    mae_mins = mean_absolute_error(y_mins[split['test']], pred_mins_test)
    print(f"    Minutes MAE: {mae_mins:.2f}")

    # ── Stage 3: Predict per-minute rate, then combine ──
    print("\n  [Stage 3] Per-minute rate prediction...")

    rate_features = [
        'per_minute_rate', 'per_minute_rate_l5', 'target_l3', 'target_l5',
        'target_l10', 'target_season', 'target_std_l10',
        'opp_def_rating_l10', 'opp_def_efg_allowed_l10', 'opp_pace_l10',
        'team_pace_l10', 'pace_delta', 'projected_pace',
        'is_points', 'is_rebounds', 'is_assists', 'is_threes', 'is_pts_rebs_asts',
        'usage_proxy_l10', 'usage_trend',
        'e_usg_pct', 'e_pace',
        'hit_rate_l5', 'hit_rate_l10',
    ]
    rate_feat_idx = [list(feat_df.columns).index(c) for c in rate_features if c in feat_df.columns]

    # Per-minute rate target
    y_rate = feat_df['per_minute_rate'].values

    X_train_r = feat_df.iloc[split['train']].values[:, rate_feat_idx]
    y_train_r = y_rate[split['train']]
    X_val_r = feat_df.iloc[split['val']].values[:, rate_feat_idx]
    y_val_r = y_rate[split['val']]
    X_test_r = feat_df.iloc[split['test']].values[:, rate_feat_idx]

    reg_rate = LGBMRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=5, num_leaves=20,
        min_child_samples=50, subsample=0.7, colsample_bytree=0.8,
        verbosity=-1,
    )
    reg_rate.fit(
        X_train_r, y_train_r,
        eval_set=[(X_val_r, y_val_r)],
        callbacks=[
            __import__('lightgbm').early_stopping(20, verbose=False),
            __import__('lightgbm').log_evaluation(0),
        ],
    )
    pred_rate_test = reg_rate.predict(X_test_r)

    # Final projection: predicted_stat = predicted_minutes * predicted_rate
    pred_stat = pred_mins_test * pred_rate_test

    # Convert to adjustment relative to line
    meta_test = meta.iloc[split['test']].reset_index(drop=True)
    line = meta_test['line'].values
    pred_adj = pred_stat - line

    # Convert to P(over) using same std approach
    std_col_idx = list(feat_df.columns).index('target_std_l10')
    stds = feat_df.iloc[split['test']].values[:, std_col_idx]
    stds = np.maximum(stds, 0.5)

    z = pred_adj / stds
    p_over = norm.cdf(z)

    # Direct pipeline MAE
    actual_vals = meta_test['actual'].values
    pipeline_mae = mean_absolute_error(actual_vals, pred_stat)
    pipeline_rmse = np.sqrt(mean_squared_error(actual_vals, pred_stat))

    result = evaluate_model(
        'B: Game Script Pipeline',
        p_over, meta_test,
        extra_metrics={
            'pipeline_mae': round(pipeline_mae, 4),
            'pipeline_rmse': round(pipeline_rmse, 4),
            'minutes_mae': round(mae_mins, 4),
        }
    )

    print(f"  Pipeline MAE: {pipeline_mae:.4f}, RMSE: {pipeline_rmse:.4f}")
    print(f"  Accuracy: {result['overall_accuracy']:.4f}")
    print(f"  Log Loss: {result['log_loss']}")

    return result


# ─── Approach C: Player Role Clustering ────────────────────────────────────

def run_approach_c(features, targets, meta, split):
    """Approach C: Cluster players into roles, train separate models per cluster."""
    print("\n" + "="*80)
    print("APPROACH C: Player Role Clustering")
    print("="*80)

    feat_df = features.copy()
    meta_df = meta.copy()

    # Clustering features (player profile)
    cluster_cols = ['target_season', 'mins_season', 'per_minute_rate', 'target_std_season', 'games_played']
    cluster_feat_idx = [list(feat_df.columns).index(c) for c in cluster_cols]

    # Cluster on TRAINING data only to avoid leakage
    train_cluster_data = feat_df.iloc[split['train']].values[:, cluster_feat_idx]

    # Handle any NaN/inf
    train_cluster_data = np.nan_to_num(train_cluster_data, nan=0, posinf=0, neginf=0)

    scaler = StandardScaler()
    train_cluster_scaled = scaler.fit_transform(train_cluster_data)

    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(train_cluster_scaled)

    # Assign clusters to all data
    all_cluster_data = np.nan_to_num(feat_df.values[:, cluster_feat_idx], nan=0, posinf=0, neginf=0)
    all_cluster_scaled = scaler.transform(all_cluster_data)
    all_clusters = kmeans.predict(all_cluster_scaled)

    # Analyze clusters
    print(f"\n  Cluster sizes (train):")
    train_clusters = all_clusters[split['train']]
    for c in range(n_clusters):
        mask = train_clusters == c
        avg_season = train_cluster_data[mask, 0].mean()
        avg_mins = train_cluster_data[mask, 1].mean()
        print(f"    Cluster {c}: n={mask.sum()}, avg_target_season={avg_season:.1f}, avg_mins={avg_mins:.1f}")

    # Binary target
    y_class = (meta_df['actual'] > meta_df['line']).astype(int).values

    X_all = feat_df.values
    meta_test = meta_df.iloc[split['test']].reset_index(drop=True)
    test_clusters = all_clusters[split['test']]

    p_over_clustered = np.full(len(X_all[split['test']]), 0.5)

    cluster_results = {}
    for c in range(n_clusters):
        train_mask = (all_clusters[split['train']] == c)
        val_mask = (all_clusters[split['val']] == c)
        test_mask = (all_clusters[split['test']] == c)

        if train_mask.sum() < 200 or test_mask.sum() < 20:
            print(f"    Cluster {c}: skipped (too few samples)")
            continue

        X_tr = X_all[split['train']][train_mask]
        y_tr = y_class[split['train']][train_mask]
        X_va = X_all[split['val']][val_mask]
        y_va = y_class[split['val']][val_mask]
        X_te = X_all[split['test']][test_mask]
        y_te = y_class[split['test']][test_mask]

        clf = LGBMClassifier(
            n_estimators=600, learning_rate=0.03, max_depth=6, num_leaves=31,
            min_child_samples=30, subsample=0.7, colsample_bytree=0.7,
            verbosity=-1,
        )
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[
                __import__('lightgbm').early_stopping(30, verbose=False),
                __import__('lightgbm').log_evaluation(0),
            ],
        )

        p_c = clf.predict_proba(X_te)[:, 1]
        p_over_clustered[test_mask] = p_c

        acc_c = accuracy_score(y_te, (p_c > 0.5).astype(int))
        ll_c = log_loss(y_te, np.clip(p_c, 1e-7, 1-1e-7))

        cluster_results[f'cluster_{c}'] = {
            'accuracy': round(acc_c, 4),
            'log_loss': round(ll_c, 4),
            'n_train': int(train_mask.sum()),
            'n_test': int(test_mask.sum()),
        }
        print(f"    Cluster {c}: acc={acc_c:.4f}, logloss={ll_c:.4f}, n_test={test_mask.sum()}")

    result = evaluate_model(
        'C: Role Clustering',
        p_over_clustered, meta_test,
        extra_metrics={'cluster_results': cluster_results, 'n_clusters': n_clusters}
    )

    print(f"\n  Combined accuracy: {result['overall_accuracy']:.4f}")
    print(f"  Combined log loss: {result['log_loss']}")

    return result


# ─── Approach D: Opponent Scheme Interaction Features ──────────────────────

def run_approach_d(features, targets, meta, split):
    """Approach D: Add scheme-specific interaction features."""
    print("\n" + "="*80)
    print("APPROACH D: Opponent Scheme Interaction Features")
    print("="*80)

    feat_df = features.copy()

    # Create interaction features
    print("  Creating interaction features...")

    # Player shooting style x opponent defense
    # Guard-like: high 3pt rate, high assists
    # Big-like: high rebounds, paint touches

    # 3PT interactions
    feat_df['player_3pt_x_opp_3pt_def'] = feat_df['fg3_share_l10'] * feat_df['opp_def_fg3_rate_allowed_l10']
    feat_df['catch_shoot_x_opp_3pt_def'] = feat_df['catch_shoot_fg3a'] * feat_df['opp_def_fg3_rate_allowed_l10']
    feat_df['pull_up_x_opp_3pt_def'] = feat_df['pull_up_fg3a'] * feat_df['opp_def_fg3_rate_allowed_l10']

    # Paint / rim interactions
    feat_df['paint_touches_x_opp_def'] = feat_df['paint_touches'] * feat_df['opp_def_rating_l10']
    feat_df['drive_pts_x_opp_def'] = feat_df['drive_pts'] * feat_df['opp_def_rating_l10']
    feat_df['at_rim_fga_x_opp_def'] = feat_df['pbp_at_rim_fga'] * feat_df['opp_def_rating_l10']

    # Assist interactions
    feat_df['ast_ratio_x_opp_tov'] = feat_df['e_ast_ratio'] * feat_df['opp_def_tov_forced_l10']
    feat_df['passes_x_opp_pace'] = feat_df['passes_made'] * feat_df['opp_pace_l10']

    # Rebound interactions
    feat_df['oreb_pct_x_opp_orb_allowed'] = feat_df['e_oreb_pct'] * feat_df['opp_def_orb_allowed_l10']
    feat_df['dreb_pct_x_opp_off_rating'] = feat_df['e_dreb_pct'] * feat_df['opp_off_rating_l10']
    feat_df['reb_chances_x_pace'] = feat_df['reb_chances'] * feat_df['projected_pace']

    # Usage x opponent defense mismatch
    feat_df['usage_x_def_mismatch'] = feat_df['e_usg_pct'] * feat_df['off_vs_def_mismatch']
    feat_df['usage_x_pace_delta'] = feat_df['e_usg_pct'] * feat_df['pace_delta']

    # FT interactions
    feat_df['fta_rate_x_opp_ft_rate_allowed'] = feat_df['fta_rate_l10'] * feat_df['opp_def_ft_rate_allowed_l10']
    feat_df['fouls_drawn_x_opp_def'] = feat_df['pbp_shooting_fouls_drawn'] * feat_df['opp_def_rating_l10']

    # Touches and scoring efficiency interactions
    feat_df['pts_per_touch_x_touches_x_pace'] = feat_df['pts_per_touch'] * feat_df['touches'] * feat_df['projected_pace']
    feat_df['efg_x_opp_efg_allowed'] = feat_df['efg_pct_l10'] * feat_df['opp_def_efg_allowed_l10']

    # Replace nan/inf from multiplications
    feat_df = feat_df.fillna(0)
    feat_df = feat_df.replace([np.inf, -np.inf], 0)

    n_new = len(feat_df.columns) - len(features.columns)
    print(f"  Added {n_new} interaction features (total: {len(feat_df.columns)})")

    # Binary classification with enriched features
    y_class = (meta['actual'] > meta['line']).astype(int)

    X_train = feat_df.iloc[split['train']].values
    y_train = y_class.iloc[split['train']].values
    X_val = feat_df.iloc[split['val']].values
    y_val = y_class.iloc[split['val']].values
    X_test = feat_df.iloc[split['test']].values
    y_test = y_class.iloc[split['test']].values
    meta_test = meta.iloc[split['test']].reset_index(drop=True)

    clf = LGBMClassifier(
        n_estimators=800, learning_rate=0.03, max_depth=6, num_leaves=31,
        min_child_samples=50, subsample=0.7, colsample_bytree=0.7,
        verbosity=-1,
    )
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            __import__('lightgbm').early_stopping(30, verbose=False),
            __import__('lightgbm').log_evaluation(0),
        ],
    )

    p_over = clf.predict_proba(X_test)[:, 1]

    # Feature importance for new features
    importances = clf.feature_importances_
    new_feat_names = list(feat_df.columns[len(features.columns):])
    new_feat_imp = {name: int(importances[len(features.columns) + i])
                    for i, name in enumerate(new_feat_names)}
    new_feat_imp_sorted = dict(sorted(new_feat_imp.items(), key=lambda x: x[1], reverse=True))

    print(f"\n  Top interaction features by importance:")
    for name, imp in list(new_feat_imp_sorted.items())[:10]:
        print(f"    {name}: {imp}")

    result = evaluate_model(
        'D: Opponent Scheme Interactions',
        p_over, meta_test,
        extra_metrics={
            'n_interaction_features': n_new,
            'interaction_feature_importances': new_feat_imp_sorted,
            'best_iteration': clf.best_iteration_,
        }
    )

    print(f"\n  Accuracy: {result['overall_accuracy']:.4f}")
    print(f"  Log Loss: {result['log_loss']}")

    return result


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    start = datetime.now()
    print(f"Starting model experiments at {start}")
    print("="*80)

    features, targets, meta, split = load_data()

    results = {}

    # Baseline
    baseline_result, _ = run_baseline_regression(features, targets, meta, split)
    results['baseline_regression'] = baseline_result

    # Approach A: Binary Classification
    a_results = run_approach_a(features, targets, meta, split)
    results['approach_A'] = a_results

    # Approach B: Game Script Pipeline
    b_result = run_approach_b(features, targets, meta, split)
    results['approach_B'] = b_result

    # Approach C: Player Role Clustering
    c_result = run_approach_c(features, targets, meta, split)
    results['approach_C'] = c_result

    # Approach D: Opponent Scheme Interactions
    d_result = run_approach_d(features, targets, meta, split)
    results['approach_D'] = d_result

    # ── Summary Comparison ──
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    print(f"\n{'Model':<35} {'Accuracy':>9} {'LogLoss':>9} {'ROI@0.55':>9} {'ROI@0.60':>9} {'Bets@0.55':>10}")
    print("-"*83)

    all_models = [
        ('Baseline Regression', results['baseline_regression']),
        ('A1: Global Classifier', results['approach_A']['A1_global_classifier']),
        ('A2: Per-Prop Classifiers', results['approach_A']['A2_per_prop_classifiers']),
        ('B: Game Script Pipeline', results['approach_B']),
        ('C: Role Clustering', results['approach_C']),
        ('D: Scheme Interactions', results['approach_D']),
    ]

    for name, r in all_models:
        acc = r.get('overall_accuracy', 0)
        ll = r.get('log_loss', 0) or 0
        roi55 = r.get('roi_at_thresholds', {}).get('0.55', {}).get('roi_pct', 0)
        roi60 = r.get('roi_at_thresholds', {}).get('0.6', {}).get('roi_pct', 0)
        bets55 = r.get('roi_at_thresholds', {}).get('0.55', {}).get('n_bets', 0)
        print(f"{name:<35} {acc:>9.4f} {ll:>9.4f} {roi55:>8.2f}% {roi60:>8.2f}% {bets55:>10}")

    end = datetime.now()
    print(f"\nCompleted in {(end - start).total_seconds():.1f}s")

    # Save results
    output_path = Path('outputs/model_experiments_A_D.json')

    # Make serializable
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return str(obj)
        return obj

    results_serializable = make_serializable(results)
    results_serializable['_meta'] = {
        'run_date': str(end),
        'runtime_seconds': round((end - start).total_seconds(), 1),
    }

    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
