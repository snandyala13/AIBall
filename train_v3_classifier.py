#!/usr/bin/env python3
"""
Train v3 Production Binary Classifiers for NBA Player Props.

Replaces the v2_final regression model with binary classifiers that predict
P(over) directly. Combines the best elements from experiments:
  - Approach A: Binary classification (54.8% accuracy, +9.6% ROI@55)
  - Approach D: Opponent scheme interaction features (best log loss 0.6869)

Architecture (informed by experiment results):
  - Global classifier trained on ALL prop types with ALL features
    (A1 beat A2 in experiments; 54.75% vs 53.95%)
  - Per-prop calibrators applied only when they improve val log loss
  - Per-prop feature importance tracked for interpretability
  - Saved as per-prop models for production compatibility

Walk-forward split: 70% train / 15% val / 15% test.
Sample weights: exponential decay with 90-day half-life.
"""

import json
import pickle
import warnings
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score, brier_score_loss
)

warnings.filterwarnings('ignore')

# ─── Constants ────────────────────────────────────────────────────────────────

PROP_TYPES = ['points', 'rebounds', 'assists', 'threes', 'pts_rebs_asts']
OUTPUT_DIR = Path('models/v3_classifier')
RANDOM_STATE = 42

HYPERPARAM_GRID = {
    'num_leaves': [15, 31, 63],
    'learning_rate': [0.01, 0.03, 0.05],
    'max_depth': [4, 6, 8, -1],
}

BASE_PARAMS = {
    'n_estimators': 800,
    'min_child_samples': 50,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'verbosity': -1,
    'random_state': RANDOM_STATE,
}

INTERACTION_FEATURES = [
    'player_3pt_x_opp_3pt_def', 'catch_shoot_x_opp_3pt_def', 'pull_up_x_opp_3pt_def',
    'paint_touches_x_opp_def', 'drive_pts_x_opp_def', 'at_rim_fga_x_opp_def',
    'ast_ratio_x_opp_tov', 'passes_x_opp_pace',
    'oreb_pct_x_opp_orb_allowed', 'dreb_pct_x_opp_off_rating', 'reb_chances_x_pace',
    'usage_x_def_mismatch', 'usage_x_pace_delta',
    'fta_rate_x_opp_ft_rate_allowed', 'fouls_drawn_x_opp_def',
    'pts_per_touch_x_touches_x_pace', 'efg_x_opp_efg_allowed',
]


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_data():
    """Load features + metadata, create walk-forward split."""
    base = Path('outputs/v2_training')
    features = pd.read_parquet(base / 'features_combined.parquet')
    meta = pd.read_parquet(base / 'metadata_enhanced.parquet')
    assert len(features) == len(meta)

    sort_idx = meta['date'].argsort()
    features = features.iloc[sort_idx].reset_index(drop=True)
    meta = meta.iloc[sort_idx].reset_index(drop=True)

    n = len(features)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    split = {'train': slice(0, train_end), 'val': slice(train_end, val_end), 'test': slice(val_end, n)}

    print(f"Total rows: {n:,}")
    print(f"Train: {train_end:,} ({meta['date'].iloc[0]} to {meta['date'].iloc[train_end-1]})")
    print(f"Val:   {val_end - train_end:,} ({meta['date'].iloc[train_end]} to {meta['date'].iloc[val_end-1]})")
    print(f"Test:  {n - val_end:,} ({meta['date'].iloc[val_end]} to {meta['date'].iloc[n-1]})")
    return features, meta, split


def add_scheme_interactions(feat_df):
    """Add opponent scheme interaction features from Approach D."""
    df = feat_df.copy()
    df['player_3pt_x_opp_3pt_def'] = df['fg3_share_l10'] * df['opp_def_fg3_rate_allowed_l10']
    df['catch_shoot_x_opp_3pt_def'] = df['catch_shoot_fg3a'] * df['opp_def_fg3_rate_allowed_l10']
    df['pull_up_x_opp_3pt_def'] = df['pull_up_fg3a'] * df['opp_def_fg3_rate_allowed_l10']
    df['paint_touches_x_opp_def'] = df['paint_touches'] * df['opp_def_rating_l10']
    df['drive_pts_x_opp_def'] = df['drive_pts'] * df['opp_def_rating_l10']
    df['at_rim_fga_x_opp_def'] = df['pbp_at_rim_fga'] * df['opp_def_rating_l10']
    df['ast_ratio_x_opp_tov'] = df['e_ast_ratio'] * df['opp_def_tov_forced_l10']
    df['passes_x_opp_pace'] = df['passes_made'] * df['opp_pace_l10']
    df['oreb_pct_x_opp_orb_allowed'] = df['e_oreb_pct'] * df['opp_def_orb_allowed_l10']
    df['dreb_pct_x_opp_off_rating'] = df['e_dreb_pct'] * df['opp_off_rating_l10']
    df['reb_chances_x_pace'] = df['reb_chances'] * df['projected_pace']
    df['usage_x_def_mismatch'] = df['e_usg_pct'] * df['off_vs_def_mismatch']
    df['usage_x_pace_delta'] = df['e_usg_pct'] * df['pace_delta']
    df['fta_rate_x_opp_ft_rate_allowed'] = df['fta_rate_l10'] * df['opp_def_ft_rate_allowed_l10']
    df['fouls_drawn_x_opp_def'] = df['pbp_shooting_fouls_drawn'] * df['opp_def_rating_l10']
    df['pts_per_touch_x_touches_x_pace'] = df['pts_per_touch'] * df['touches'] * df['projected_pace']
    df['efg_x_opp_efg_allowed'] = df['efg_pct_l10'] * df['opp_def_efg_allowed_l10']
    return df.fillna(0).replace([np.inf, -np.inf], 0)


def compute_sample_weights(dates, half_life_days=90):
    """Exponential decay sample weights."""
    dates_ts = pd.to_datetime(dates)
    days_ago = (dates_ts.max() - dates_ts).dt.total_seconds() / 86400.0
    return np.exp(-np.log(2) * days_ago / half_life_days).values


# ─── Training Utilities ──────────────────────────────────────────────────────

def grid_search(X_train, y_train, X_val, y_val, w_train, label=""):
    """Search hyperparameter grid, return best by val log loss."""
    best_params, best_ll, best_model = None, float('inf'), None
    combos = list(itertools.product(
        HYPERPARAM_GRID['num_leaves'],
        HYPERPARAM_GRID['learning_rate'],
        HYPERPARAM_GRID['max_depth'],
    ))
    print(f"    Searching {len(combos)} combos{' for ' + label if label else ''}...")

    for nl, lr, md in combos:
        params = {**BASE_PARAMS, 'num_leaves': nl, 'learning_rate': lr, 'max_depth': md}
        clf = LGBMClassifier(**params)
        clf.fit(
            X_train, y_train, sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                __import__('lightgbm').early_stopping(30, verbose=False),
                __import__('lightgbm').log_evaluation(0),
            ],
        )
        probs = clf.predict_proba(X_val)[:, 1]
        ll = log_loss(y_val, np.clip(probs, 1e-7, 1 - 1e-7))
        if ll < best_ll:
            best_ll = ll
            best_params = {'num_leaves': nl, 'learning_rate': lr, 'max_depth': md}
            best_model = clf

    return best_model, best_params, best_ll


def calibrate(val_probs, val_y):
    """Platt scaling via logistic regression."""
    cal = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    cal.fit(val_probs.reshape(-1, 1), val_y)
    return cal


def apply_cal(calibrator, probs):
    return calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]


# ─── Evaluation ───────────────────────────────────────────────────────────────

def compute_roi(p_over, actual, line, threshold):
    """ROI for flat $100 bets at given confidence threshold."""
    payout = 100 / 1.1
    wagered = profit = bets = wins = 0
    for i in range(len(p_over)):
        if p_over[i] >= threshold:
            wagered += 100; bets += 1
            if actual[i] > line[i]: profit += payout; wins += 1
            else: profit -= 100
        elif (1 - p_over[i]) >= threshold:
            wagered += 100; bets += 1
            if actual[i] < line[i]: profit += payout; wins += 1
            else: profit -= 100
    return {
        'threshold': threshold, 'n_bets': bets,
        'total_wagered': wagered, 'total_profit': round(profit, 2),
        'roi_pct': round(profit / wagered * 100, 2) if wagered > 0 else 0,
        'win_rate': round(wins / bets, 4) if bets > 0 else 0,
        'bet_pct': round(bets / len(p_over) * 100, 2) if len(p_over) > 0 else 0,
    }


def full_eval(p_over, actual, line, dates=None):
    """Full evaluation suite."""
    y = (actual > line).astype(int)
    np_mask = actual != line
    p, y_f, a, l = p_over[np_mask], y[np_mask], actual[np_mask], line[np_mask]

    acc = accuracy_score(y_f, (p > 0.5).astype(int))
    ll = log_loss(y_f, np.clip(p, 1e-7, 1 - 1e-7))
    auc = roc_auc_score(y_f, p) if len(np.unique(y_f)) > 1 else 0.5
    brier = brier_score_loss(y_f, p)

    roi = {}
    for t in [0.52, 0.55, 0.58, 0.60, 0.65]:
        roi[str(t)] = compute_roi(p, a, l, t)

    pred_o = p > 0.5
    over_hit = float(y_f[pred_o].mean()) if pred_o.sum() > 0 else 0
    under_hit = float((1 - y_f[~pred_o]).mean()) if (~pred_o).sum() > 0 else 0

    cal_buckets = []
    for i in range(10):
        lo, hi = i * 0.1, (i + 1) * 0.1
        m = (p >= lo) & (p < hi)
        if m.sum() > 0:
            cal_buckets.append({
                'bucket': f'{lo:.1f}-{hi:.1f}',
                'predicted_mean': round(float(p[m].mean()), 4),
                'actual_mean': round(float(y_f[m].mean()), 4),
                'count': int(m.sum()),
            })

    monthly = {}
    if dates is not None:
        d = dates[np_mask]
        periods = pd.to_datetime(d).to_period('M').astype(str)
        for mo in sorted(pd.unique(periods)):
            m = periods == mo
            if m.sum() > 10:
                monthly[mo] = {
                    'accuracy': round(float(accuracy_score(y_f[m], (p[m] > 0.5).astype(int))), 4),
                    'n_legs': int(m.sum()),
                }

    return {
        'accuracy': round(acc, 4), 'log_loss': round(ll, 4),
        'auc': round(auc, 4), 'brier_score': round(brier, 4),
        'n_test': int(np_mask.sum()),
        'roi_at_thresholds': roi,
        'over_hit_rate': round(over_hit, 4),
        'under_hit_rate': round(under_hit, 4),
        'over_count': int(pred_o.sum()),
        'under_count': int((~pred_o).sum()),
        'calibration_buckets': cal_buckets,
        'per_month': monthly,
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    start = datetime.now()
    print(f"{'='*80}")
    print(f"  v3 CLASSIFIER TRAINING - {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    # ── Load & Prepare ──
    print("[1] Loading data...")
    features_raw, meta, split = load_data()

    print("\n[2] Adding scheme interaction features...")
    features = add_scheme_interactions(features_raw)
    all_cols = list(features.columns)
    print(f"  {len(features_raw.columns)} base + {len(all_cols) - len(features_raw.columns)} interactions = {len(all_cols)} total")

    y = (meta['actual'] > meta['line']).astype(int)
    w_train = compute_sample_weights(meta['date'].iloc[split['train']], half_life_days=90)
    print(f"  Sample weights: [{w_train.min():.4f}, {w_train.max():.4f}]")

    # Clean output directory
    if OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train = features.iloc[split['train']].values
    y_train = y.iloc[split['train']].values
    X_val = features.iloc[split['val']].values
    y_val = y.iloc[split['val']].values
    X_test = features.iloc[split['test']].values
    y_test = y.iloc[split['test']].values

    meta_test = meta.iloc[split['test']].reset_index(drop=True)
    actual_test = meta_test['actual'].values
    line_test = meta_test['line'].values

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 3: GLOBAL CLASSIFIER (all features, all props — matching A1 experiment)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print(f"\n{'='*80}")
    print("  [3] GLOBAL CLASSIFIER TRAINING")
    print(f"{'='*80}")

    global_model, global_params, global_val_ll = grid_search(
        X_train, y_train, X_val, y_val, w_train, label="global"
    )
    print(f"  Best params: {global_params}")
    print(f"  Val log loss: {global_val_ll:.4f}")
    print(f"  Best iteration: {global_model.best_iteration_}")

    global_test_probs = global_model.predict_proba(X_test)[:, 1]
    global_val_probs = global_model.predict_proba(X_val)[:, 1]

    global_metrics = full_eval(global_test_probs, actual_test, line_test, meta_test['date'].values)
    print(f"\n  Global test accuracy: {global_metrics['accuracy']:.4f}")
    print(f"  Global test log loss: {global_metrics['log_loss']:.4f}")
    print(f"  Global test AUC:      {global_metrics['auc']:.4f}")

    for tk, rd in global_metrics['roi_at_thresholds'].items():
        if rd['n_bets'] > 0:
            print(f"  ROI@{tk}: {rd['roi_pct']:+.2f}% ({rd['n_bets']} bets, {rd['win_rate']:.1%} win)")

    # Global feature importance
    g_imp = sorted(zip(all_cols, global_model.feature_importances_.tolist()),
                   key=lambda x: x[1], reverse=True)
    print(f"\n  Global top 15 features:")
    for name, imp in g_imp[:15]:
        print(f"    {name}: {imp}")

    # Save global model
    with open(OUTPUT_DIR / 'global_classifier.pkl', 'wb') as f:
        pickle.dump(global_model, f)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 4: PER-PROP CALIBRATION (only when it improves val performance)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print(f"\n{'='*80}")
    print("  [4] PER-PROP CALIBRATION")
    print(f"{'='*80}")

    final_test_probs = np.copy(global_test_probs)
    per_prop_config = {}
    per_prop_metrics = {}

    for pt in PROP_TYPES:
        print(f"\n  --- {pt.upper()} ---")

        mask_val = (meta.iloc[split['val']]['prop_type'] == pt).values
        mask_test = (meta.iloc[split['test']]['prop_type'] == pt).values
        mask_train = (meta.iloc[split['train']]['prop_type'] == pt).values

        val_probs_pt = global_val_probs[mask_val]
        test_probs_pt = global_test_probs[mask_test]
        y_val_pt = y_val[mask_val]
        y_test_pt = y_test[mask_test]

        n_val_pt = mask_val.sum()
        n_test_pt = mask_test.sum()
        n_train_pt = mask_train.sum()
        print(f"  Samples: {n_train_pt:,} train, {n_val_pt:,} val, {n_test_pt:,} test")

        # Raw global performance on this prop
        raw_val_ll = log_loss(y_val_pt, np.clip(val_probs_pt, 1e-7, 1 - 1e-7))
        raw_test_acc = accuracy_score(y_test_pt, (test_probs_pt > 0.5).astype(int))

        # Try Platt scaling calibration
        cal = calibrate(val_probs_pt, y_val_pt)
        cal_val_probs = apply_cal(cal, val_probs_pt)
        cal_val_ll = log_loss(y_val_pt, np.clip(cal_val_probs, 1e-7, 1 - 1e-7))

        # Decide: use calibration only if it meaningfully improves val log loss
        cal_improvement = raw_val_ll - cal_val_ll
        use_cal = cal_improvement > 0.002
        if use_cal:
            cal_test_probs = apply_cal(cal, test_probs_pt)
            print(f"  Calibration APPLIED: val LL {raw_val_ll:.4f} -> {cal_val_ll:.4f}")
        else:
            cal_test_probs = test_probs_pt
            print(f"  Calibration SKIPPED: val LL {raw_val_ll:.4f} vs {cal_val_ll:.4f} (no improvement)")

        # Store
        test_idx = np.where(mask_test)[0]
        final_test_probs[test_idx] = cal_test_probs

        # Evaluate on test
        meta_pt = meta_test.iloc[test_idx].reset_index(drop=True)
        metrics = full_eval(cal_test_probs, meta_pt['actual'].values,
                           meta_pt['line'].values, meta_pt['date'].values)

        print(f"  Test accuracy: {metrics['accuracy']:.4f}")
        print(f"  Test log loss: {metrics['log_loss']:.4f}")
        print(f"  Test AUC: {metrics['auc']:.4f}")
        print(f"  Over: {metrics['over_hit_rate']:.4f} ({metrics['over_count']}), Under: {metrics['under_hit_rate']:.4f} ({metrics['under_count']})")

        for tk, rd in metrics['roi_at_thresholds'].items():
            if rd['n_bets'] > 0:
                print(f"  ROI@{tk}: {rd['roi_pct']:+.2f}% ({rd['n_bets']} bets, {rd['win_rate']:.1%})")

        # Per-prop feature importance from global model (subset)
        train_imp_mask = mask_train
        if use_cal:
            # Save calibrator
            with open(OUTPUT_DIR / f'{pt}_calibrator.pkl', 'wb') as f:
                pickle.dump(cal, f)
            per_prop_config[pt] = {
                'use_calibration': True,
                'calibrator_coef': float(cal.coef_[0][0]),
                'calibrator_intercept': float(cal.intercept_[0]),
                'val_ll_raw': round(raw_val_ll, 4),
                'val_ll_calibrated': round(cal_val_ll, 4),
            }
        else:
            per_prop_config[pt] = {
                'use_calibration': False,
                'val_ll_raw': round(raw_val_ll, 4),
            }

        per_prop_metrics[pt] = metrics

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 5: PER-PROP SPECIALIST CLASSIFIERS (trained per-prop, checked vs global)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print(f"\n{'='*80}")
    print("  [5] PER-PROP SPECIALIST CLASSIFIERS (vs global baseline)")
    print(f"{'='*80}")

    for pt in PROP_TYPES:
        print(f"\n  --- {pt.upper()} ---")

        mask_train = (meta.iloc[split['train']]['prop_type'] == pt).values
        mask_val = (meta.iloc[split['val']]['prop_type'] == pt).values
        mask_test = (meta.iloc[split['test']]['prop_type'] == pt).values

        X_pt_train = X_train[mask_train]
        y_pt_train = y_train[mask_train]
        w_pt_train = w_train[mask_train]
        X_pt_val = X_val[mask_val]
        y_pt_val = y_val[mask_val]
        X_pt_test = X_test[mask_test]

        # Train per-prop specialist
        pt_model, pt_params, pt_val_ll = grid_search(
            X_pt_train, y_pt_train, X_pt_val, y_pt_val, w_pt_train, label=pt
        )
        print(f"  Specialist params: {pt_params}, val LL: {pt_val_ll:.4f}, iters: {pt_model.best_iteration_}")

        # Compare specialist vs global on this prop's val set
        global_val_pt = global_model.predict_proba(X_pt_val)[:, 1]
        specialist_val_pt = pt_model.predict_proba(X_pt_val)[:, 1]
        global_val_ll = log_loss(y_pt_val, np.clip(global_val_pt, 1e-7, 1 - 1e-7))

        # Try blending
        best_alpha = 0.0
        best_blend_ll = global_val_ll
        for alpha in np.arange(0.0, 1.05, 0.05):
            blended = alpha * specialist_val_pt + (1 - alpha) * global_val_pt
            bl_ll = log_loss(y_pt_val, np.clip(blended, 1e-7, 1 - 1e-7))
            if bl_ll < best_blend_ll:
                best_blend_ll = bl_ll
                best_alpha = round(alpha, 2)

        val_ll_improvement = global_val_ll - best_blend_ll
        print(f"  Global val LL: {global_val_ll:.4f}")
        print(f"  Best blend: {best_alpha:.0%} specialist + {1-best_alpha:.0%} global (LL: {best_blend_ll:.4f}, delta: {val_ll_improvement:.4f})")

        # Require meaningful improvement (>0.002 in val LL) to avoid noise-based upgrades
        MIN_IMPROVEMENT = 0.002
        if best_alpha > 0 and val_ll_improvement > MIN_IMPROVEMENT:
            # Specialist adds meaningful value — apply blend
            specialist_test_pt = pt_model.predict_proba(X_pt_test)[:, 1]
            global_test_pt = global_model.predict_proba(X_pt_test)[:, 1]
            blended_test = best_alpha * specialist_test_pt + (1 - best_alpha) * global_test_pt

            # Try calibration on blended result
            blended_val = best_alpha * specialist_val_pt + (1 - best_alpha) * global_val_pt
            raw_blend_ll = log_loss(y_pt_val, np.clip(blended_val, 1e-7, 1 - 1e-7))

            cal = calibrate(blended_val, y_pt_val)
            cal_blend_val = apply_cal(cal, blended_val)
            cal_blend_ll = log_loss(y_pt_val, np.clip(cal_blend_val, 1e-7, 1 - 1e-7))

            if cal_blend_ll < raw_blend_ll:
                final_probs = apply_cal(cal, blended_test)
                use_cal = True
                print(f"  Calibration APPLIED on blend: {raw_blend_ll:.4f} -> {cal_blend_ll:.4f}")
            else:
                final_probs = blended_test
                use_cal = False
                print(f"  Calibration SKIPPED on blend")

            # Report test set comparison (for information only, not used for decision)
            test_idx = np.where(mask_test)[0]
            current_probs = final_test_probs[test_idx]
            meta_pt = meta_test.iloc[test_idx].reset_index(drop=True)
            y_pt_test = y_test[mask_test]

            current_ll = log_loss(y_pt_test, np.clip(current_probs, 1e-7, 1 - 1e-7))
            new_ll = log_loss(y_pt_test, np.clip(final_probs, 1e-7, 1 - 1e-7))
            current_acc = accuracy_score(y_pt_test, (current_probs > 0.5).astype(int))
            new_acc = accuracy_score(y_pt_test, (final_probs > 0.5).astype(int))

            print(f"  Test comparison (informational):")
            print(f"    Global-only: acc={current_acc:.4f}, LL={current_ll:.4f}")
            print(f"    With blend:  acc={new_acc:.4f}, LL={new_ll:.4f}")

            # Decision made on val set, not test set
            print(f"  >>> UPGRADE: specialist blend (val improvement={val_ll_improvement:.4f} > {MIN_IMPROVEMENT})")
            final_test_probs[test_idx] = final_probs

            # Save specialist model
            with open(OUTPUT_DIR / f'{pt}_classifier.pkl', 'wb') as f_out:
                pickle.dump(pt_model, f_out)
            if use_cal:
                with open(OUTPUT_DIR / f'{pt}_calibrator.pkl', 'wb') as f_out:
                    pickle.dump(cal, f_out)

            per_prop_config[pt].update({
                'has_specialist': True,
                'blend_weight': best_alpha,
                'specialist_params': {**BASE_PARAMS, **pt_params},
                'specialist_best_iteration': pt_model.best_iteration_,
                'use_calibration': use_cal,
            })

            # Re-evaluate
            metrics = full_eval(final_probs, meta_pt['actual'].values,
                               meta_pt['line'].values, meta_pt['date'].values)
            per_prop_metrics[pt] = metrics
        else:
            reason = "specialist weight = 0" if best_alpha == 0 else f"val improvement {val_ll_improvement:.4f} <= {MIN_IMPROVEMENT}"
            print(f"  --- KEEP: global-only for {pt} ({reason})")
            per_prop_config[pt]['has_specialist'] = False

    # Feature importance per prop (from global model, filtered by prop)
    for pt in PROP_TYPES:
        mask_test = (meta.iloc[split['test']]['prop_type'] == pt).values
        test_idx = np.where(mask_test)[0]
        # Use global model importances as proxy
        imp_dict = {n: int(v) for n, v in g_imp[:20]}
        per_prop_metrics[pt]['top_20_features'] = imp_dict

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FINAL OVERALL EVALUATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print(f"\n{'='*80}")
    print("  FINAL OVERALL TEST SET EVALUATION")
    print(f"{'='*80}\n")

    non_push = actual_test != line_test
    y_true = (actual_test > line_test).astype(int)

    overall_acc = accuracy_score(y_true[non_push], (final_test_probs[non_push] > 0.5).astype(int))
    overall_ll = log_loss(y_true[non_push], np.clip(final_test_probs[non_push], 1e-7, 1 - 1e-7))
    overall_auc = roc_auc_score(y_true[non_push], final_test_probs[non_push])
    overall_brier = brier_score_loss(y_true[non_push], final_test_probs[non_push])

    global_only_acc = accuracy_score(y_true[non_push], (global_test_probs[non_push] > 0.5).astype(int))

    print(f"  Global-only accuracy:    {global_only_acc:.4f}")
    print(f"  Final accuracy:          {overall_acc:.4f}")
    print(f"  Final log loss:          {overall_ll:.4f}")
    print(f"  Final AUC:               {overall_auc:.4f}")
    print(f"  Final Brier:             {overall_brier:.4f}")

    # ROI
    print(f"\n  ROI at thresholds:")
    overall_roi = {}
    for t in [0.52, 0.55, 0.58, 0.60, 0.65]:
        r = compute_roi(final_test_probs[non_push], actual_test[non_push], line_test[non_push], t)
        overall_roi[str(t)] = r
        print(f"    @{t:.2f}: {r['roi_pct']:+.2f}% ({r['n_bets']} bets, {r['win_rate']:.1%} win, {r['bet_pct']:.1f}%)")

    # Calibration
    print(f"\n  Calibration:")
    for i in range(10):
        lo, hi = i * 0.1, (i + 1) * 0.1
        m = (final_test_probs >= lo) & (final_test_probs < hi) & non_push
        if m.sum() > 0:
            print(f"    {lo:.1f}-{hi:.1f}: pred={final_test_probs[m].mean():.4f}, actual={y_true[m].mean():.4f}, n={m.sum()}")

    # Per-prop table
    print(f"\n  {'Prop':<16} {'Acc':>7} {'LL':>7} {'AUC':>7} {'Specialist':>11} {'ROI@52':>9} {'ROI@55':>9} {'ROI@58':>9} {'ROI@60':>9}")
    print(f"  {'-'*90}")
    for pt in PROP_TYPES:
        m = per_prop_metrics[pt]
        spec = 'Yes' if per_prop_config[pt].get('has_specialist') else 'No'
        r52 = m['roi_at_thresholds'].get('0.52', {}).get('roi_pct', 0)
        r55 = m['roi_at_thresholds'].get('0.55', {}).get('roi_pct', 0)
        r58 = m['roi_at_thresholds'].get('0.58', {}).get('roi_pct', 0)
        r60 = m['roi_at_thresholds'].get('0.6', {}).get('roi_pct', 0)
        print(f"  {pt:<16} {m['accuracy']:>7.4f} {m['log_loss']:>7.4f} {m['auc']:>7.4f} {spec:>11} {r52:>8.2f}% {r55:>8.2f}% {r58:>8.2f}% {r60:>8.2f}%")

    # Monthly
    print(f"\n  Monthly:")
    dt_periods = pd.to_datetime(meta_test['date'].values).to_period('M').astype(str)
    for mo in sorted(pd.unique(dt_periods)):
        m = (dt_periods == mo) & non_push
        if m.sum() > 10:
            print(f"    {mo}: {accuracy_score(y_true[m], (final_test_probs[m] > 0.5).astype(int)):.4f} ({m.sum()} legs)")

    # Side
    po = final_test_probs[non_push] > 0.5
    print(f"\n  Side breakdown:")
    print(f"    OVER:  {y_true[non_push][po].mean():.4f} hit ({po.sum()} bets)")
    print(f"    UNDER: {(1 - y_true[non_push][~po]).mean():.4f} hit ({(~po).sum()} bets)")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # COMPARISON TABLE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print(f"\n{'='*80}")
    print("  COMPARISON: v2_final REGRESSION vs v3 CLASSIFIER")
    print(f"{'='*80}\n")

    v2_path = Path('models/v2_final/training_meta.json')
    v2m = {}
    if v2_path.exists():
        with open(v2_path) as f:
            v2m = json.load(f).get('results', {})

    print(f"                        REGRESSION (v2_final)    CLASSIFIER (v3)")
    print(f"  {'Prop':<16} {'Accuracy':>9} {'ROI@55':>9}    {'Accuracy':>9} {'ROI@55':>9}    {'Acc +/-':>9}")
    print(f"  {'-'*75}")

    for pt in PROP_TYPES:
        v2a = v2m.get(pt, {}).get('directional_accuracy', 0)
        v2r = v2m.get(pt, {}).get('roi_at_5pct', 0) * 100
        v3a = per_prop_metrics[pt]['accuracy']
        v3r = per_prop_metrics[pt]['roi_at_thresholds'].get('0.55', {}).get('roi_pct', 0)
        d = v3a - v2a
        print(f"  {pt:<16} {v2a:>9.4f} {v2r:>8.2f}%    {v3a:>9.4f} {v3r:>8.2f}%    {d:>+8.4f}")

    v2_avg = np.mean([v2m[pt]['directional_accuracy'] for pt in PROP_TYPES if pt in v2m]) if v2m else 0
    print(f"  {'-'*75}")
    print(f"  {'OVERALL':<16} {v2_avg:>9.4f} {'':>9}    {overall_acc:>9.4f} {'':>9}    {overall_acc - v2_avg:>+8.4f}")

    print(f"\n  vs Experiment A1:  54.75% acc, +9.60% ROI@55, 0.6871 logloss")
    print(f"  vs Experiment D:   54.57% acc, +8.98% ROI@55, 0.6869 logloss")
    print(f"  v3 Production:     {overall_acc:.2%} acc, {overall_roi.get('0.55', {}).get('roi_pct', 0):+.2f}% ROI@55, {overall_ll:.4f} logloss")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SAVE METADATA
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    training_meta = {
        'model_type': 'classifier',
        'model_version': 'v3',
        'architecture': 'global_classifier_with_perprop_calibration',
        'training_date': str(start),
        'n_samples': len(features),
        'n_train': int(len(features.iloc[split['train']])),
        'n_val': int(len(features.iloc[split['val']])),
        'n_test': int(len(features.iloc[split['test']])),
        'train_date_range': f"{meta['date'].iloc[split['train']].iloc[0]} to {meta['date'].iloc[split['train']].iloc[-1]}",
        'val_date_range': f"{meta['date'].iloc[split['val']].iloc[0]} to {meta['date'].iloc[split['val']].iloc[-1]}",
        'test_date_range': f"{meta['date'].iloc[split['test']].iloc[0]} to {meta['date'].iloc[split['test']].iloc[-1]}",
        'calibrated': True,
        'calibration_method': 'platt_scaling_per_prop',
        'sample_weights': 'exponential_decay_90d_half_life',
        'feature_columns': all_cols,
        'interaction_features': INTERACTION_FEATURES,
        'global_hyperparams': {**BASE_PARAMS, **global_params},
        'global_best_iteration': global_model.best_iteration_,
        'global_metrics': global_metrics,
        'global_top_features': {n: v for n, v in g_imp[:20]},
        'per_prop_config': per_prop_config,
        'overall_metrics': {
            'accuracy': round(overall_acc, 4),
            'log_loss': round(overall_ll, 4),
            'auc': round(overall_auc, 4),
            'brier_score': round(overall_brier, 4),
            'roi_at_thresholds': overall_roi,
        },
        'per_prop_metrics': per_prop_metrics,
    }

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return str(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    with open(OUTPUT_DIR / 'training_meta.json', 'w') as f:
        json.dump(make_serializable(training_meta), f, indent=2, default=str)

    end = datetime.now()
    print(f"\n{'='*80}")
    print(f"  TRAINING COMPLETE ({(end - start).total_seconds():.1f}s)")
    print(f"{'='*80}")
    print(f"  Saved to {OUTPUT_DIR}/:")
    for p in sorted(OUTPUT_DIR.iterdir()):
        print(f"    {p.name}")
    print(f"\n  training_meta.json: model_type='classifier'\n")


if __name__ == '__main__':
    main()
