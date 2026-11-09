"""
Feature selection for LightGBM NBA player prop model.
Identifies and removes noisy features, then tests top-K subsets.
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import defaultdict
from pathlib import Path

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
features = pd.read_parquet("outputs/v2_training/features_combined.parquet")
targets = pd.read_parquet("outputs/v2_training/targets_enhanced.parquet")
metadata = pd.read_parquet("outputs/v2_training/metadata_enhanced.parquet")

all_feature_names = features.columns.tolist()
print(f"  {features.shape[0]} rows x {features.shape[1]} features")
print(f"  Prop types: {metadata['prop_type'].value_counts().to_dict()}")

# ── Helpers ────────────────────────────────────────────────────────────────────

PROP_TYPES = ["points", "rebounds", "assists", "threes", "pts_rebs_asts"]

LGB_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
    n_jobs=-1,
)


def compute_sample_weights(dates, half_life_days=90):
    """Exponential decay weights: more recent games weighted higher."""
    dates_dt = pd.to_datetime(dates)
    max_date = dates_dt.max()
    days_ago = (max_date - dates_dt).dt.days.values.astype(float)
    decay = np.log(2) / half_life_days
    weights = np.exp(-decay * days_ago)
    return weights


def time_split(mask, dates):
    """70/15/15 split sorted by date."""
    idx = np.where(mask)[0]
    d = pd.to_datetime(dates.iloc[idx])
    order = d.argsort()
    idx_sorted = idx[order]
    n = len(idx_sorted)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    train_idx = idx_sorted[:n_train]
    val_idx = idx_sorted[n_train:n_train + n_val]
    test_idx = idx_sorted[n_train + n_val:]
    return train_idx, val_idx, test_idx


def train_and_evaluate(feature_cols, X, y, dates, mask, prop_name):
    """Train LightGBM on given features, return metrics and importances."""
    train_idx, val_idx, test_idx = time_split(mask, dates)

    X_train = X.iloc[train_idx][feature_cols].values
    y_train = y.iloc[train_idx].values.ravel()
    w_train = compute_sample_weights(dates.iloc[train_idx])

    X_val = X.iloc[val_idx][feature_cols].values
    y_val = y.iloc[val_idx].values.ravel()
    w_val = compute_sample_weights(dates.iloc[val_idx])

    X_test = X.iloc[test_idx][feature_cols].values
    y_test = y.iloc[test_idx].values.ravel()

    lines_test = metadata.iloc[test_idx]["line"].values
    actuals_test = metadata.iloc[test_idx]["actual"].values

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[w_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    best_iter = model.best_iteration_
    preds = model.predict(X_test)

    # Metrics
    mae = float(np.mean(np.abs(preds - y_test)))

    # Line MAE: how well line + pred predicts actual
    line_pred_actual = lines_test + preds
    line_mae = float(np.mean(np.abs(line_pred_actual - actuals_test)))

    # Baseline: just using line as prediction
    line_only_mae = float(np.mean(np.abs(lines_test - actuals_test)))

    # Directional accuracy: does our adjustment predict the right side?
    pred_over = preds > 0  # we predict actual > line
    actual_over = (actuals_test - lines_test) > 0
    dir_acc = float(np.mean(pred_over == actual_over))

    # Feature importances (gain-based)
    importances = dict(zip(feature_cols, model.feature_importances_.tolist()))

    return {
        "prop": prop_name,
        "n_features": len(feature_cols),
        "best_iteration": best_iter,
        "mae": round(mae, 4),
        "line_mae": round(line_mae, 4),
        "line_only_mae": round(line_only_mae, 4),
        "improvement_vs_line": round(line_only_mae - line_mae, 4),
        "directional_accuracy": round(dir_acc, 4),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
    }, importances


# ── Phase 1: Full model training for importance analysis ───────────────────────
print("\n" + "=" * 80)
print("PHASE 1: Training full model per prop type for feature importance")
print("=" * 80)

all_importances = {}  # prop -> {feature: importance}
phase1_results = []

for prop in PROP_TYPES:
    mask = (metadata["prop_type"] == prop).values
    n = mask.sum()
    print(f"\n  {prop}: {n} samples")

    result, importances = train_and_evaluate(
        all_feature_names, features, targets, metadata["date"], mask, prop
    )
    phase1_results.append(result)
    all_importances[prop] = importances

    print(f"    Best iter: {result['best_iteration']}, "
          f"MAE: {result['mae']:.4f}, "
          f"Line MAE: {result['line_mae']:.4f} (baseline: {result['line_only_mae']:.4f}), "
          f"Dir acc: {result['directional_accuracy']:.3f}")

# ── Phase 2: Identify zero and near-zero importance features ───────────────────
print("\n" + "=" * 80)
print("PHASE 2: Identifying noisy features")
print("=" * 80)

# Aggregate importances across props
avg_importance = defaultdict(float)
zero_count = defaultdict(int)

for feat in all_feature_names:
    vals = []
    for prop in PROP_TYPES:
        imp = all_importances[prop].get(feat, 0.0)
        vals.append(imp)
        if imp == 0.0:
            zero_count[feat] += 1
    avg_importance[feat] = np.mean(vals)

# Zero importance across ALL 5 props
zero_features = [f for f in all_feature_names if zero_count[f] == 5]
print(f"\n  Features with ZERO importance in all 5 props: {len(zero_features)}")
for f in sorted(zero_features):
    print(f"    - {f}")

# Non-zero features
nonzero_features = [f for f in all_feature_names if f not in zero_features]
print(f"\n  Features with non-zero importance: {len(nonzero_features)}")

# Near-zero: bottom 25% of non-zero features by average importance
if nonzero_features:
    nonzero_importances = {f: avg_importance[f] for f in nonzero_features}
    sorted_by_imp = sorted(nonzero_importances.items(), key=lambda x: x[1])
    cutoff_idx = len(sorted_by_imp) // 4
    near_zero_features = [f for f, _ in sorted_by_imp[:cutoff_idx]]
    print(f"  Near-zero importance (bottom 25%): {len(near_zero_features)}")
    for f, imp in sorted_by_imp[:cutoff_idx]:
        print(f"    - {f}: avg_gain={imp:.1f}")

# Rank all features by average importance
print(f"\n  Top 30 features by average gain:")
sorted_all = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
for i, (f, imp) in enumerate(sorted_all[:30]):
    print(f"    {i+1:3d}. {f}: {imp:.1f}")

# ── Phase 3: Top-K feature subset experiments ─────────────────────────────────
print("\n" + "=" * 80)
print("PHASE 3: Testing top-K feature subsets (excluding zero-importance features)")
print("=" * 80)

# Rank features: sort non-zero features by average importance descending
ranked_features = [f for f, _ in sorted(
    [(f, avg_importance[f]) for f in nonzero_features],
    key=lambda x: x[1], reverse=True
)]

K_VALUES_RAW = [50, 75, 100, 150, 200, len(nonzero_features)]
# Deduplicate (e.g. if nonzero < 200, both cap at same value)
K_VALUES = sorted(set(min(k, len(ranked_features)) for k in K_VALUES_RAW))
topk_results = []

for K in K_VALUES:
    selected = ranked_features[:K]
    actual_k = len(selected)
    print(f"\n  K={actual_k} features:")

    prop_results = []
    for prop in PROP_TYPES:
        mask = (metadata["prop_type"] == prop).values
        result, _ = train_and_evaluate(
            selected, features, targets, metadata["date"], mask, prop
        )
        prop_results.append(result)
        print(f"    {prop:15s}: iter={result['best_iteration']:3d}, "
              f"MAE={result['mae']:.4f}, "
              f"line_MAE={result['line_mae']:.4f}, "
              f"impr={result['improvement_vs_line']:+.4f}, "
              f"dir_acc={result['directional_accuracy']:.3f}")

    # Average across props
    avg_mae = np.mean([r["mae"] for r in prop_results])
    avg_line_mae = np.mean([r["line_mae"] for r in prop_results])
    avg_line_only = np.mean([r["line_only_mae"] for r in prop_results])
    avg_impr = np.mean([r["improvement_vs_line"] for r in prop_results])
    avg_dir = np.mean([r["directional_accuracy"] for r in prop_results])
    avg_iter = np.mean([r["best_iteration"] for r in prop_results])

    topk_results.append({
        "K": actual_k,
        "avg_mae": round(avg_mae, 4),
        "avg_line_mae": round(avg_line_mae, 4),
        "avg_line_only_mae": round(avg_line_only, 4),
        "avg_improvement": round(avg_impr, 4),
        "avg_directional_accuracy": round(avg_dir, 4),
        "avg_best_iteration": round(avg_iter, 1),
        "per_prop": prop_results,
    })

    print(f"    {'AVERAGE':15s}: iter={avg_iter:5.1f}, "
          f"MAE={avg_mae:.4f}, "
          f"line_MAE={avg_line_mae:.4f}, "
          f"impr={avg_impr:+.4f}, "
          f"dir_acc={avg_dir:.3f}")


# ── Summary table ──────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)

header = f"{'K':>5s} | {'Avg Iter':>8s} | {'MAE':>7s} | {'Line MAE':>8s} | {'Baseline':>8s} | {'Improve':>8s} | {'Dir Acc':>7s}"
print(header)
print("-" * len(header))

best_k = None
best_impr = -999

for r in topk_results:
    marker = ""
    if r["avg_improvement"] > best_impr:
        best_impr = r["avg_improvement"]
        best_k = r["K"]

for r in topk_results:
    marker = " <-- BEST" if r["K"] == best_k else ""
    print(f"{r['K']:5d} | {r['avg_best_iteration']:8.1f} | "
          f"{r['avg_mae']:.4f} | {r['avg_line_mae']:.4f}  | "
          f"{r['avg_line_only_mae']:.4f}  | {r['avg_improvement']:+.4f} | "
          f"{r['avg_directional_accuracy']:.3f}{marker}")

print(f"\nBest K = {best_k} (by avg improvement over line baseline)")

# ── Per-prop best K ────────────────────────────────────────────────────────────
print("\nPer-prop best K:")
for prop in PROP_TYPES:
    best_prop_k = None
    best_prop_impr = -999
    for r in topk_results:
        for pr in r["per_prop"]:
            if pr["prop"] == prop and pr["improvement_vs_line"] > best_prop_impr:
                best_prop_impr = pr["improvement_vs_line"]
                best_prop_k = r["K"]
    print(f"  {prop:15s}: best K={best_prop_k}, improvement={best_prop_impr:+.4f}")

# ── Save results ───────────────────────────────────────────────────────────────
output = {
    "phase1_full_model": phase1_results,
    "zero_importance_features": sorted(zero_features),
    "near_zero_features": sorted(near_zero_features) if nonzero_features else [],
    "ranked_features": ranked_features,
    "feature_avg_importance": {f: round(v, 2) for f, v in sorted_all},
    "topk_results": topk_results,
    "best_k": best_k,
    "recommended_features": ranked_features[:best_k],
}

Path("outputs").mkdir(exist_ok=True)
with open("outputs/feature_selection_results.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to outputs/feature_selection_results.json")
print(f"  - {len(zero_features)} zero-importance features to remove")
print(f"  - {len(near_zero_features)} near-zero features identified")
print(f"  - Recommended feature set: top {best_k} features")
print(f"  - Full ranked feature list included")
