#!/usr/bin/env python3
"""
v3_classifier_holdout: Per-prop binary classifiers with honest time split.

Train: All of 2023-24 season + Jan-Feb 2026 (through 2026-02-28)
Val:   March 2026 (for early stopping + Platt calibration)
Test:  April-June 2026 (PURE HOLDOUT -- never seen during training)

This simulates real deployment: model trained through mid-season, does it work for the rest?
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score, brier_score_loss
)

warnings.filterwarnings('ignore')

OUTPUT_DIR = Path('models/v3_classifier_holdout')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROP_TYPES = ['points', 'rebounds', 'assists', 'threes', 'pts_rebs_asts']

# Hyperparameter grid
NUM_LEAVES_OPTIONS = [15, 31, 63]
LR_OPTIONS = [0.01, 0.03, 0.05]
MAX_DEPTH_OPTIONS = [4, 6, 8]


# ── Data Loading & Splitting ─────────────────────────────────────────────────

def load_and_split():
    """Load data and apply the honest holdout split."""
    base = Path('outputs/v2_training')
    X = pd.read_parquet(base / 'features_combined.parquet')
    y_raw = pd.read_parquet(base / 'targets_enhanced.parquet')['target'].values
    meta = pd.read_parquet(base / 'metadata_enhanced.parquet')

    # Binary target
    y = (meta['actual'] > meta['line']).astype(int).values

    # Remove pushes (none in this dataset but be safe)
    not_push = meta['actual'] != meta['line']
    X = X[not_push].reset_index(drop=True)
    y = y[not_push.values]
    meta = meta[not_push].reset_index(drop=True)

    # Split by date
    train_mask = meta['date'] <= '2026-02-28'
    val_mask = (meta['date'] > '2026-02-28') & (meta['date'] <= '2026-03-31')
    test_mask = meta['date'] > '2026-03-31'

    X_train = X[train_mask].reset_index(drop=True)
    y_train = y[train_mask.values]
    meta_train = meta[train_mask].reset_index(drop=True)

    X_val = X[val_mask].reset_index(drop=True)
    y_val = y[val_mask.values]
    meta_val = meta[val_mask].reset_index(drop=True)

    X_test = X[test_mask].reset_index(drop=True)
    y_test = y[test_mask.values]
    meta_test = meta[test_mask].reset_index(drop=True)

    print(f'Train: {len(X_train)} ({meta_train["date"].min()} to {meta_train["date"].max()})')
    print(f'Val:   {len(X_val)} ({meta_val["date"].min()} to {meta_val["date"].max()})')
    print(f'Test:  {len(X_test)} ({meta_test["date"].min()} to {meta_test["date"].max()})')
    print()

    return (X_train, y_train, meta_train,
            X_val, y_val, meta_val,
            X_test, y_test, meta_test)


# ── Scheme Interaction Features (from Approach D) ────────────────────────────

def add_interaction_features(df):
    """Add opponent scheme interaction features from Approach D."""
    df = df.copy()

    # 3PT interactions
    df['player_3pt_x_opp_3pt_def'] = df['fg3_share_l10'] * df['opp_def_fg3_rate_allowed_l10']
    df['catch_shoot_x_opp_3pt_def'] = df['catch_shoot_fg3a'] * df['opp_def_fg3_rate_allowed_l10']
    df['pull_up_x_opp_3pt_def'] = df['pull_up_fg3a'] * df['opp_def_fg3_rate_allowed_l10']

    # Paint / rim interactions
    df['paint_touches_x_opp_def'] = df['paint_touches'] * df['opp_def_rating_l10']
    df['drive_pts_x_opp_def'] = df['drive_pts'] * df['opp_def_rating_l10']
    df['at_rim_fga_x_opp_def'] = df['pbp_at_rim_fga'] * df['opp_def_rating_l10']

    # Assist interactions
    df['ast_ratio_x_opp_tov'] = df['e_ast_ratio'] * df['opp_def_tov_forced_l10']
    df['passes_x_opp_pace'] = df['passes_made'] * df['opp_pace_l10']

    # Rebound interactions
    df['oreb_pct_x_opp_orb_allowed'] = df['e_oreb_pct'] * df['opp_def_orb_allowed_l10']
    df['dreb_pct_x_opp_off_rating'] = df['e_dreb_pct'] * df['opp_off_rating_l10']
    df['reb_chances_x_pace'] = df['reb_chances'] * df['projected_pace']

    # Usage x opponent defense mismatch
    df['usage_x_def_mismatch'] = df['e_usg_pct'] * df['off_vs_def_mismatch']
    df['usage_x_pace_delta'] = df['e_usg_pct'] * df['pace_delta']

    # FT interactions
    df['fta_rate_x_opp_ft_rate_allowed'] = df['fta_rate_l10'] * df['opp_def_ft_rate_allowed_l10']
    df['fouls_drawn_x_opp_def'] = df['pbp_shooting_fouls_drawn'] * df['opp_def_rating_l10']

    # Touches and scoring efficiency interactions
    df['pts_per_touch_x_touches_x_pace'] = df['pts_per_touch'] * df['touches'] * df['projected_pace']
    df['efg_x_opp_efg_allowed'] = df['efg_pct_l10'] * df['opp_def_efg_allowed_l10']

    # Clean up
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)

    return df


INTERACTION_FEATURES = [
    'player_3pt_x_opp_3pt_def', 'catch_shoot_x_opp_3pt_def', 'pull_up_x_opp_3pt_def',
    'paint_touches_x_opp_def', 'drive_pts_x_opp_def', 'at_rim_fga_x_opp_def',
    'ast_ratio_x_opp_tov', 'passes_x_opp_pace',
    'oreb_pct_x_opp_orb_allowed', 'dreb_pct_x_opp_off_rating', 'reb_chances_x_pace',
    'usage_x_def_mismatch', 'usage_x_pace_delta',
    'fta_rate_x_opp_ft_rate_allowed', 'fouls_drawn_x_opp_def',
    'pts_per_touch_x_touches_x_pace', 'efg_x_opp_efg_allowed',
]


# ── Sample Weights ────────────────────────────────────────────────────────────

def compute_sample_weights(dates, half_life_days=90):
    """Exponential decay weights with given half-life."""
    dates_dt = pd.to_datetime(dates)
    max_date = dates_dt.max()
    days_ago = (max_date - dates_dt).dt.days.values.astype(float)
    weights = np.exp(-np.log(2) * days_ago / half_life_days)
    return weights


# ── Feature Selection ─────────────────────────────────────────────────────────

def select_features_for_prop(X_train_full, y_train_prop, sample_weights, n_features=70):
    """Train preliminary classifier on all features, select top N by importance."""
    prelim = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.7,
        colsample_bytree=0.7,
        verbosity=-1,
        random_state=42,
    )
    prelim.fit(X_train_full.values, y_train_prop, sample_weight=sample_weights)

    importances = prelim.feature_importances_
    feat_imp = pd.Series(importances, index=X_train_full.columns)
    feat_imp = feat_imp.sort_values(ascending=False)

    # Select top N, but always include interaction features that made the cut
    top_feats = list(feat_imp.head(n_features).index)

    # Ensure interaction features with nonzero importance are included
    for f in INTERACTION_FEATURES:
        if f in X_train_full.columns and feat_imp.get(f, 0) > 0 and f not in top_feats:
            top_feats.append(f)

    return top_feats


# ── ROI Computation ───────────────────────────────────────────────────────────

def compute_roi(p_over, meta_slice, threshold):
    """Compute ROI for flat $100 bets at given confidence threshold."""
    actual = meta_slice['actual'].values
    line = meta_slice['line'].values
    payout = 100 / 1.1  # ~90.91 at -110 odds

    total_wagered = 0
    total_profit = 0
    n_bets = 0
    n_wins = 0

    for i in range(len(p_over)):
        if p_over[i] >= threshold:
            # Bet OVER
            total_wagered += 100
            n_bets += 1
            if actual[i] > line[i]:
                total_profit += payout
                n_wins += 1
            else:
                total_profit -= 100
        elif (1 - p_over[i]) >= threshold:
            # Bet UNDER
            total_wagered += 100
            n_bets += 1
            if actual[i] < line[i]:
                total_profit += payout
                n_wins += 1
            else:
                total_profit -= 100

    roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0
    win_rate = n_wins / n_bets if n_bets > 0 else 0
    return {
        'threshold': threshold,
        'n_bets': n_bets,
        'total_wagered': total_wagered,
        'total_profit': round(total_profit, 2),
        'roi_pct': round(roi, 2),
        'win_rate': round(win_rate, 4),
        'bet_pct': round(n_bets / len(p_over) * 100, 2) if len(p_over) > 0 else 0,
    }


# ── Per-Prop Training ────────────────────────────────────────────────────────

def train_prop_classifier(prop_type, X_train_full, y_train_full, meta_train,
                          X_val_full, y_val_full, meta_val):
    """Train best LGBMClassifier for a single prop type with hyperparameter search."""
    print(f'\n{"="*60}')
    print(f'Training: {prop_type}')
    print(f'{"="*60}')

    # Filter to this prop type
    train_mask = meta_train['prop_type'] == prop_type
    val_mask = meta_val['prop_type'] == prop_type

    X_tr = X_train_full[train_mask].reset_index(drop=True)
    y_tr = y_train_full[train_mask.values]
    meta_tr = meta_train[train_mask].reset_index(drop=True)

    X_va = X_val_full[val_mask].reset_index(drop=True)
    y_va = y_val_full[val_mask.values]
    meta_va = meta_val[val_mask].reset_index(drop=True)

    print(f'  Train: {len(X_tr)}, Val: {len(X_va)}')
    print(f'  Train over rate: {y_tr.mean():.4f}, Val over rate: {y_va.mean():.4f}')

    if len(X_tr) < 200 or len(X_va) < 50:
        print(f'  SKIPPED: too few samples')
        return None

    # Sample weights
    sw_train = compute_sample_weights(meta_tr['date'])

    # Feature selection
    selected_features = select_features_for_prop(X_tr, y_tr, sw_train, n_features=70)
    print(f'  Selected {len(selected_features)} features')

    X_tr_sel = X_tr[selected_features].values
    X_va_sel = X_va[selected_features].values

    # Hyperparameter search
    best_ll = float('inf')
    best_config = None
    best_model = None
    best_iter = None

    configs_tested = 0
    for nl in NUM_LEAVES_OPTIONS:
        for lr in LR_OPTIONS:
            for md in MAX_DEPTH_OPTIONS:
                clf = LGBMClassifier(
                    n_estimators=800,
                    num_leaves=nl,
                    learning_rate=lr,
                    max_depth=md,
                    min_child_samples=50,
                    subsample=0.7,
                    colsample_bytree=0.7,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    verbosity=-1,
                    random_state=42,
                )
                clf.fit(
                    X_tr_sel, y_tr,
                    sample_weight=sw_train,
                    eval_set=[(X_va_sel, y_va)],
                    callbacks=[
                        lgb.early_stopping(30, verbose=False),
                        lgb.log_evaluation(0),
                    ],
                )

                p_val = clf.predict_proba(X_va_sel)[:, 1]
                ll = log_loss(y_va, np.clip(p_val, 1e-7, 1 - 1e-7))

                configs_tested += 1
                if ll < best_ll:
                    best_ll = ll
                    best_config = {'num_leaves': nl, 'learning_rate': lr, 'max_depth': md}
                    best_model = clf
                    best_iter = clf.best_iteration_

    print(f'  Tested {configs_tested} configs. Best: num_leaves={best_config["num_leaves"]}, '
          f'lr={best_config["learning_rate"]}, max_depth={best_config["max_depth"]}')
    print(f'  Best val log loss: {best_ll:.4f} at iteration {best_iter}')

    # Platt calibration on validation set
    p_val_raw = best_model.predict_proba(X_va_sel)[:, 1]
    calibrator = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    calibrator.fit(p_val_raw.reshape(-1, 1), y_va)

    p_val_cal = calibrator.predict_proba(p_val_raw.reshape(-1, 1))[:, 1]
    ll_cal = log_loss(y_va, np.clip(p_val_cal, 1e-7, 1 - 1e-7))
    print(f'  Calibrated val log loss: {ll_cal:.4f}')

    # Feature importance (from best model)
    importances = best_model.feature_importances_
    feat_imp = {selected_features[i]: int(importances[i]) for i in range(len(selected_features))}
    feat_imp_sorted = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

    return {
        'model': best_model,
        'calibrator': calibrator,
        'features': selected_features,
        'hyperparams': {
            'n_estimators': 800,
            'min_child_samples': 50,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'verbosity': -1,
            'random_state': 42,
            **best_config,
        },
        'best_iteration': best_iter,
        'best_val_log_loss': round(best_ll, 4),
        'calibrated_val_log_loss': round(ll_cal, 4),
        'calibrator_coef': float(calibrator.coef_[0][0]),
        'calibrator_intercept': float(calibrator.intercept_[0]),
        'n_train': len(X_tr),
        'n_val': len(X_va),
        'feature_importances': feat_imp_sorted,
    }


# ── Test Evaluation ──────────────────────────────────────────────────────────

def evaluate_on_test(prop_models, X_test_full, y_test, meta_test):
    """Full evaluation on pure holdout test set."""
    print(f'\n{"="*80}')
    print('EVALUATING ON PURE HOLDOUT TEST SET (April-June 2026)')
    print(f'{"="*80}')

    all_p_over = np.full(len(X_test_full), 0.5)
    per_prop_metrics = {}

    for prop_type in PROP_TYPES:
        test_mask = meta_test['prop_type'] == prop_type
        if test_mask.sum() == 0 or prop_type not in prop_models:
            continue

        pm = prop_models[prop_type]
        X_te = X_test_full[test_mask][pm['features']].values
        y_te = y_test[test_mask.values]
        meta_te = meta_test[test_mask].reset_index(drop=True)

        # Raw predictions
        p_raw = pm['model'].predict_proba(X_te)[:, 1]

        # Calibrated predictions
        p_cal = pm['calibrator'].predict_proba(p_raw.reshape(-1, 1))[:, 1]

        # Store in full array
        all_p_over[test_mask.values] = p_cal

        # Metrics
        acc_raw = accuracy_score(y_te, (p_raw > 0.5).astype(int))
        acc_cal = accuracy_score(y_te, (p_cal > 0.5).astype(int))
        ll_raw = log_loss(y_te, np.clip(p_raw, 1e-7, 1 - 1e-7))
        ll_cal = log_loss(y_te, np.clip(p_cal, 1e-7, 1 - 1e-7))

        try:
            auc = roc_auc_score(y_te, p_cal)
        except ValueError:
            auc = 0.5

        brier = brier_score_loss(y_te, p_cal)

        # ROI at thresholds
        roi_results = {}
        for thresh in [0.52, 0.55, 0.58, 0.60, 0.65]:
            roi_results[str(thresh)] = compute_roi(p_cal, meta_te, thresh)

        # Over vs Under hit rates
        over_mask = p_cal > 0.5
        under_mask = p_cal <= 0.5
        over_count = int(over_mask.sum())
        under_count = int(under_mask.sum())
        over_hit = accuracy_score(y_te[over_mask], np.ones(over_count)) if over_count > 0 else 0
        under_hit = accuracy_score(1 - y_te[under_mask], np.ones(under_count)) if under_count > 0 else 0

        # Monthly breakdown
        monthly = {}
        meta_te_copy = meta_te.copy()
        meta_te_copy['p_over'] = p_cal
        meta_te_copy['y'] = y_te
        meta_te_copy['month'] = pd.to_datetime(meta_te_copy['date']).dt.to_period('M').astype(str)
        for month, grp in meta_te_copy.groupby('month'):
            month_acc = accuracy_score(grp['y'], (grp['p_over'] > 0.5).astype(int))
            monthly[month] = {'accuracy': round(month_acc, 4), 'n_legs': len(grp)}

        # Calibration check: for predictions in [0.55, 0.60], actual hit rate
        cal_mask = (p_cal >= 0.55) & (p_cal < 0.60)
        cal_check_55_60 = {
            'predicted_range': '[0.55, 0.60)',
            'n_bets': int(cal_mask.sum()),
            'actual_hit_rate': round(y_te[cal_mask].mean(), 4) if cal_mask.sum() > 0 else None,
            'predicted_mean': round(p_cal[cal_mask].mean(), 4) if cal_mask.sum() > 0 else None,
        }

        # Calibration buckets
        cal_buckets = []
        for lo, hi in [(0.3, 0.4), (0.4, 0.45), (0.45, 0.5), (0.5, 0.55), (0.55, 0.6), (0.6, 0.7)]:
            mask = (p_cal >= lo) & (p_cal < hi)
            if mask.sum() > 0:
                cal_buckets.append({
                    'bucket': f'{lo}-{hi}',
                    'predicted_mean': round(float(p_cal[mask].mean()), 4),
                    'actual_mean': round(float(y_te[mask].mean()), 4),
                    'count': int(mask.sum()),
                })

        # Top 20 features
        top_20 = dict(list(pm['feature_importances'].items())[:20])

        per_prop_metrics[prop_type] = {
            'accuracy': round(acc_cal, 4),
            'accuracy_raw': round(acc_raw, 4),
            'log_loss': round(ll_cal, 4),
            'log_loss_raw': round(ll_raw, 4),
            'auc': round(auc, 4),
            'brier_score': round(brier, 4),
            'n_test': int(test_mask.sum()),
            'roi_at_thresholds': roi_results,
            'over_hit_rate': round(over_hit, 4),
            'under_hit_rate': round(under_hit, 4),
            'over_count': over_count,
            'under_count': under_count,
            'calibration_buckets': cal_buckets,
            'calibration_check_55_60': cal_check_55_60,
            'per_month': monthly,
            'top_20_features': top_20,
        }

        print(f'\n  {prop_type}: acc={acc_cal:.4f}, ll={ll_cal:.4f}, auc={auc:.4f}, n={test_mask.sum()}')
        for t in ['0.52', '0.55', '0.58', '0.6', '0.65']:
            r = roi_results[t]
            if r['n_bets'] > 0:
                print(f'    @{t}: ROI={r["roi_pct"]:+.2f}%, bets={r["n_bets"]}, win={r["win_rate"]:.4f}')
        print(f'    Over/Under: {over_count}/{under_count}, hit: {over_hit:.4f}/{under_hit:.4f}')
        print(f'    Cal check [0.55,0.60): n={cal_check_55_60["n_bets"]}, '
              f'actual={cal_check_55_60["actual_hit_rate"]}')

    # Overall metrics
    overall_acc = accuracy_score(y_test, (all_p_over > 0.5).astype(int))
    overall_ll = log_loss(y_test, np.clip(all_p_over, 1e-7, 1 - 1e-7))
    try:
        overall_auc = roc_auc_score(y_test, all_p_over)
    except ValueError:
        overall_auc = 0.5
    overall_brier = brier_score_loss(y_test, all_p_over)

    overall_roi = {}
    for thresh in [0.52, 0.55, 0.58, 0.60, 0.65]:
        overall_roi[str(thresh)] = compute_roi(all_p_over, meta_test, thresh)

    # Overall monthly
    overall_monthly = {}
    meta_test_copy = meta_test.copy()
    meta_test_copy['p_over'] = all_p_over
    meta_test_copy['y'] = y_test
    meta_test_copy['month'] = pd.to_datetime(meta_test_copy['date']).dt.to_period('M').astype(str)
    for month, grp in meta_test_copy.groupby('month'):
        month_acc = accuracy_score(grp['y'], (grp['p_over'] > 0.5).astype(int))
        overall_monthly[month] = {'accuracy': round(month_acc, 4), 'n_legs': len(grp)}

    # Overall calibration check
    cal_mask_all = (all_p_over >= 0.55) & (all_p_over < 0.60)
    cal_check_overall = {
        'predicted_range': '[0.55, 0.60)',
        'n_bets': int(cal_mask_all.sum()),
        'actual_hit_rate': round(float(y_test[cal_mask_all].mean()), 4) if cal_mask_all.sum() > 0 else None,
        'predicted_mean': round(float(all_p_over[cal_mask_all].mean()), 4) if cal_mask_all.sum() > 0 else None,
    }

    overall_metrics = {
        'accuracy': round(overall_acc, 4),
        'log_loss': round(overall_ll, 4),
        'auc': round(overall_auc, 4),
        'brier_score': round(overall_brier, 4),
        'roi_at_thresholds': overall_roi,
        'per_month': overall_monthly,
        'calibration_check_55_60': cal_check_overall,
    }

    print(f'\n  OVERALL: acc={overall_acc:.4f}, ll={overall_ll:.4f}, auc={overall_auc:.4f}')
    for t in ['0.52', '0.55', '0.58', '0.6', '0.65']:
        r = overall_roi[t]
        if r['n_bets'] > 0:
            print(f'    @{t}: ROI={r["roi_pct"]:+.2f}%, bets={r["n_bets"]}, win={r["win_rate"]:.4f}')
    print(f'  Cal check [0.55,0.60): n={cal_check_overall["n_bets"]}, '
          f'actual={cal_check_overall["actual_hit_rate"]}')

    return overall_metrics, per_prop_metrics


# ── Compare to v3_classifier ─────────────────────────────────────────────────

def compare_to_v3(overall_metrics, per_prop_metrics):
    """Compare side-by-side with v3_classifier (70/15/15 split)."""
    v3_path = Path('models/v3_classifier/training_meta.json')
    if not v3_path.exists():
        print('\nv3_classifier not found, skipping comparison.')
        return None

    with open(v3_path) as f:
        v3 = json.load(f)

    print(f'\n{"="*80}')
    print('COMPARISON: v3 (70/15/15) vs v3_holdout (train thru Feb, test Apr-Jun)')
    print(f'{"="*80}')

    # Header
    print(f'\n{"Metric":<30} {"v3 (70/15/15)":>15} {"v3_holdout":>15} {"Delta":>10}')
    print('-' * 70)

    # Overall
    v3_acc = v3['overall_metrics']['accuracy']
    ho_acc = overall_metrics['accuracy']
    print(f'{"Overall accuracy":<30} {v3_acc:>14.4f} {ho_acc:>14.4f} {ho_acc - v3_acc:>+10.4f}')

    v3_ll = v3['overall_metrics']['log_loss']
    ho_ll = overall_metrics['log_loss']
    print(f'{"Log loss":<30} {v3_ll:>14.4f} {ho_ll:>14.4f} {ho_ll - v3_ll:>+10.4f}')

    v3_auc = v3['overall_metrics']['auc']
    ho_auc = overall_metrics['auc']
    print(f'{"AUC":<30} {v3_auc:>14.4f} {ho_auc:>14.4f} {ho_auc - v3_auc:>+10.4f}')

    # ROI at thresholds
    for t in ['0.52', '0.55', '0.58', '0.6']:
        v3_roi = v3['overall_metrics']['roi_at_thresholds'].get(t, {}).get('roi_pct', 0)
        ho_roi = overall_metrics['roi_at_thresholds'].get(t, {}).get('roi_pct', 0)
        v3_bets = v3['overall_metrics']['roi_at_thresholds'].get(t, {}).get('n_bets', 0)
        ho_bets = overall_metrics['roi_at_thresholds'].get(t, {}).get('n_bets', 0)
        label = f'ROI @{t} (n={v3_bets}/{ho_bets})'
        print(f'{label:<30} {v3_roi:>13.2f}% {ho_roi:>13.2f}% {ho_roi - v3_roi:>+9.2f}%')

    # Per-prop comparison
    print(f'\n{"Prop":<18} {"v3 acc":>8} {"ho acc":>8} {"v3 ll":>8} {"ho ll":>8} {"v3 AUC":>8} {"ho AUC":>8}')
    print('-' * 68)
    for pt in PROP_TYPES:
        if pt in v3['per_prop_metrics'] and pt in per_prop_metrics:
            v3p = v3['per_prop_metrics'][pt]
            hop = per_prop_metrics[pt]
            print(f'{pt:<18} {v3p["accuracy"]:>8.4f} {hop["accuracy"]:>8.4f} '
                  f'{v3p["log_loss"]:>8.4f} {hop["log_loss"]:>8.4f} '
                  f'{v3p["auc"]:>8.4f} {hop["auc"]:>8.4f}')

    # Test date range comparison
    print(f'\n  v3 test range:      {v3["test_date_range"]}')
    print(f'  holdout test range: {overall_metrics.get("test_range", "Apr-Jun 2026")}')
    print(f'  v3 train range:     {v3["train_date_range"]}')

    return v3


# ── Save Models ──────────────────────────────────────────────────────────────

def save_models(prop_models, overall_metrics, per_prop_metrics, meta_train, meta_val, meta_test):
    """Save per-prop classifier + calibrator pickle files and training_meta.json."""
    print(f'\nSaving models to {OUTPUT_DIR}/')

    # Save per-prop models
    for prop_type, pm in prop_models.items():
        model_path = OUTPUT_DIR / f'{prop_type}_classifier.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(pm['model'], f)

        cal_path = OUTPUT_DIR / f'{prop_type}_calibrator.pkl'
        with open(cal_path, 'wb') as f:
            pickle.dump(pm['calibrator'], f)

        print(f'  Saved {prop_type}: classifier + calibrator')

    # Build training_meta.json
    all_features = []
    for pm in prop_models.values():
        all_features.extend(pm['features'])
    all_features = sorted(set(all_features))

    per_prop_config = {}
    for prop_type, pm in prop_models.items():
        per_prop_config[prop_type] = {
            'features': pm['features'],
            'n_features': len(pm['features']),
            'hyperparams': pm['hyperparams'],
            'best_iteration': pm['best_iteration'],
            'best_val_log_loss': pm['best_val_log_loss'],
            'calibrated_val_log_loss': pm['calibrated_val_log_loss'],
            'calibrator_coef': pm['calibrator_coef'],
            'calibrator_intercept': pm['calibrator_intercept'],
        }

    meta_json = {
        'model_type': 'classifier',
        'model_version': 'v3_holdout',
        'split': 'holdout_apr_jun_2026',
        'train_end': '2026-02-28',
        'val_range': '2026-03-01 to 2026-03-31',
        'test_range': '2026-04-01 to 2026-06-13',
        'training_date': str(datetime.now()),
        'n_samples': int(len(meta_train) + len(meta_val) + len(meta_test)),
        'n_train': int(len(meta_train)),
        'n_val': int(len(meta_val)),
        'n_test': int(len(meta_test)),
        'train_date_range': f'{meta_train["date"].min()} to {meta_train["date"].max()}',
        'val_date_range': f'{meta_val["date"].min()} to {meta_val["date"].max()}',
        'test_date_range': f'{meta_test["date"].min()} to {meta_test["date"].max()}',
        'calibrated': True,
        'calibration_method': 'platt_scaling',
        'sample_weights': 'exponential_decay_90d_half_life',
        'feature_columns': all_features,
        'interaction_features': INTERACTION_FEATURES,
        'per_prop_config': per_prop_config,
        'overall_metrics': overall_metrics,
        'per_prop_metrics': per_prop_metrics,
    }

    with open(OUTPUT_DIR / 'training_meta.json', 'w') as f:
        json.dump(meta_json, f, indent=2, default=str)

    print(f'  Saved training_meta.json')


# ── Honest Assessment ────────────────────────────────────────────────────────

def print_honest_assessment(overall_metrics, per_prop_metrics, v3_data):
    """Print honest assessment comparing holdout to 70/15/15."""
    print(f'\n{"="*80}')
    print('HONEST ASSESSMENT')
    print(f'{"="*80}')

    if v3_data is None:
        print('No v3 data available for comparison.')
        return

    v3 = v3_data

    # Overall comparison
    v3_acc = v3['overall_metrics']['accuracy']
    ho_acc = overall_metrics['accuracy']
    acc_delta = ho_acc - v3_acc

    v3_ll = v3['overall_metrics']['log_loss']
    ho_ll = overall_metrics['log_loss']

    print(f'\nOverall accuracy: v3={v3_acc:.4f} vs holdout={ho_acc:.4f} (delta={acc_delta:+.4f})')
    print(f'Overall log loss: v3={v3_ll:.4f} vs holdout={ho_ll:.4f}')

    if abs(acc_delta) < 0.005:
        print('\nVERDICT: Holdout performance is VERY CLOSE to 70/15/15 performance.')
        print('This suggests the 70/15/15 results were NOT significantly inflated by data leakage.')
        print('The model generalizes reasonably to future unseen data.')
    elif acc_delta < -0.01:
        print(f'\nVERDICT: Holdout accuracy is WORSE by {abs(acc_delta):.4f}.')
        print('This suggests the 70/15/15 results were OPTIMISTIC.')
        print('The model loses predictive power on genuinely future data.')
        print('Possible causes:')
        print('  - Time-period leakage in 70/15/15 (train/test from similar dates)')
        print('  - Model overfits to within-season patterns that dont persist')
        print('  - Market efficiency increases over the season')
    elif acc_delta > 0.01:
        print(f'\nVERDICT: Holdout accuracy is BETTER by {acc_delta:.4f}.')
        print('This is unusual -- could indicate:')
        print('  - April-June data is easier to predict (playoffs?)')
        print('  - Smaller test set variance')
        print('  - The model genuinely works better on later-season data')
    else:
        print(f'\nVERDICT: Holdout performance is slightly {"worse" if acc_delta < 0 else "better"} '
              f'({acc_delta:+.4f}).')
        print('Within noise range. No strong evidence of data leakage.')

    # ROI comparison
    print('\nROI comparison (the money metric):')
    for t in ['0.52', '0.55', '0.58', '0.6']:
        v3_roi = v3['overall_metrics']['roi_at_thresholds'].get(t, {}).get('roi_pct', 0)
        v3_bets = v3['overall_metrics']['roi_at_thresholds'].get(t, {}).get('n_bets', 0)
        ho_roi = overall_metrics['roi_at_thresholds'].get(t, {}).get('roi_pct', 0)
        ho_bets = overall_metrics['roi_at_thresholds'].get(t, {}).get('n_bets', 0)
        print(f'  @{t}: v3={v3_roi:+.2f}% ({v3_bets} bets) vs holdout={ho_roi:+.2f}% ({ho_bets} bets)')

    # Per-prop comparison
    print('\nPer-prop accuracy deltas (holdout - v3):')
    for pt in PROP_TYPES:
        if pt in v3['per_prop_metrics'] and pt in per_prop_metrics:
            v3a = v3['per_prop_metrics'][pt]['accuracy']
            hoa = per_prop_metrics[pt]['accuracy']
            delta = hoa - v3a
            marker = ' <<< DEGRADED' if delta < -0.01 else ' (stable)' if abs(delta) < 0.005 else ''
            print(f'  {pt:<18}: {delta:+.4f}{marker}')

    # Calibration honesty
    print('\nCalibration reality check:')
    for pt in PROP_TYPES:
        if pt in per_prop_metrics:
            pm = per_prop_metrics[pt]
            check = pm.get('calibration_check_55_60', {})
            if check.get('n_bets', 0) > 0:
                pred = check['predicted_mean']
                actual = check['actual_hit_rate']
                print(f'  {pt} [0.55-0.60): predicted={pred:.4f}, actual={actual:.4f}, '
                      f'n={check["n_bets"]}')
            else:
                print(f'  {pt} [0.55-0.60): no predictions in this range')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    start = datetime.now()
    print(f'Starting v3_holdout training at {start}')
    print(f'{"="*80}')

    # Load data with honest split
    (X_train, y_train, meta_train,
     X_val, y_val, meta_val,
     X_test, y_test, meta_test) = load_and_split()

    # Add interaction features
    print('Adding scheme interaction features...')
    X_train = add_interaction_features(X_train)
    X_val = add_interaction_features(X_val)
    X_test = add_interaction_features(X_test)
    print(f'Feature count after interactions: {X_train.shape[1]}')

    # Train per-prop classifiers
    prop_models = {}
    for prop_type in PROP_TYPES:
        result = train_prop_classifier(
            prop_type, X_train, y_train, meta_train, X_val, y_val, meta_val
        )
        if result is not None:
            prop_models[prop_type] = result

    # Evaluate on pure holdout test set
    overall_metrics, per_prop_metrics = evaluate_on_test(
        prop_models, X_test, y_test, meta_test
    )

    # Compare to v3_classifier
    v3_data = compare_to_v3(overall_metrics, per_prop_metrics)

    # Save models
    save_models(prop_models, overall_metrics, per_prop_metrics,
                meta_train, meta_val, meta_test)

    # Honest assessment
    print_honest_assessment(overall_metrics, per_prop_metrics, v3_data)

    end = datetime.now()
    runtime = (end - start).total_seconds()
    print(f'\nCompleted in {runtime:.1f}s')


if __name__ == '__main__':
    main()
