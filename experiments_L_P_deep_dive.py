"""
Deep dive on the most promising signals from experiments_L_P.py.
Focus on statistical significance, robustness, and practical applicability.
"""

import pandas as pd
import numpy as np
import json
import warnings
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')

X = pd.read_parquet('outputs/v2_training/features_combined.parquet')
meta = pd.read_parquet('outputs/v2_training/metadata_enhanced.parquet')
targets = pd.read_parquet('outputs/v2_training/targets_enhanced.parquet')

n = len(X)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

test_X = X.iloc[val_end:]
test_meta = meta.iloc[val_end:]

# Also get validation set for cross-checking
val_X = X.iloc[train_end:val_end]
val_meta = meta.iloc[train_end:val_end]

# Train set for calibration
train_X = X.iloc[:train_end]
train_meta = meta.iloc[:train_end]

test_actual = test_meta['actual'].values
test_line = test_meta['line'].values
val_actual = val_meta['actual'].values
val_line = val_meta['line'].values
train_actual = train_meta['actual'].values
train_line = train_meta['line'].values


def american_to_decimal(odds):
    if odds > 0:
        return odds / 100.0 + 1.0
    else:
        return 100.0 / abs(odds) + 1.0


test_under_dec = np.array([american_to_decimal(o) for o in test_X['under_odds'].values])
test_over_dec = np.array([american_to_decimal(o) for o in test_X['over_odds'].values])
val_under_dec = np.array([american_to_decimal(o) for o in val_X['under_odds'].values])
val_over_dec = np.array([american_to_decimal(o) for o in val_X['over_odds'].values])

PROP_TYPES = ['points', 'rebounds', 'assists', 'threes', 'pts_rebs_asts']


def bootstrap_roi(hits, payouts, n_boots=5000):
    """Bootstrap confidence interval for ROI."""
    n = len(hits)
    if n == 0:
        return 0, 0, 0
    rois = []
    for _ in range(n_boots):
        idx = np.random.randint(0, n, n)
        boot_returns = np.sum(payouts[idx][hits[idx]]) - n
        rois.append(boot_returns / n)
    return np.percentile(rois, 2.5), np.percentile(rois, 50), np.percentile(rois, 97.5)


# ──────────────────────────────────────────────────────────────────────────────
# DEEP DIVE 1: Low-CV hot players -> UNDER (the +20.5% ROI signal)
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 80)
print("DEEP DIVE 1: Low-CV Hot Players -> UNDER")
print("=" * 80)

trend = test_X['target_trend'].values
cv = test_X['target_cv'].values

# Replicate the finding
for cv_thresh in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
    for trend_thresh in [1, 2, 3, 5]:
        mask = (trend > trend_thresh) & (cv < cv_thresh)
        if mask.sum() < 20:
            continue

        hits = test_actual[mask] < test_line[mask]
        payouts = test_under_dec[mask]
        roi = (np.sum(payouts[hits]) - mask.sum()) / mask.sum()

        # Bootstrap CI
        ci_low, ci_med, ci_high = bootstrap_roi(hits, payouts)

        # p-value: is hit rate significantly above 50%?
        hit_rate = hits.mean()
        z_score = (hit_rate - 0.50) / np.sqrt(0.25 / mask.sum())
        p_value = 1 - scipy_stats.norm.cdf(z_score)

        sig = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
        print(f"  cv<{cv_thresh:.2f} trend>{trend_thresh}: n={mask.sum():4d}, hit={hit_rate:.1%}, ROI={roi:+.1%}, CI=[{ci_low:+.1%}, {ci_high:+.1%}], p={p_value:.3f} {sig}")


# Cross-validate on VALIDATION set
print("\n--- Validation set check ---")
val_trend = val_X['target_trend'].values
val_cv = val_X['target_cv'].values

for cv_thresh in [0.20, 0.25, 0.30]:
    for trend_thresh in [2, 3]:
        mask = (val_trend > trend_thresh) & (val_cv < cv_thresh)
        if mask.sum() < 10:
            continue
        hits = val_actual[mask] < val_line[mask]
        payouts = val_under_dec[mask]
        roi = (np.sum(payouts[hits]) - mask.sum()) / mask.sum()
        print(f"  VAL cv<{cv_thresh:.2f} trend>{trend_thresh}: n={mask.sum():4d}, hit={hits.mean():.1%}, ROI={roi:+.1%}")

# Train set check
print("\n--- Train set check ---")
train_trend = train_X['target_trend'].values
train_cv = train_X['target_cv'].values
train_under_dec = np.array([american_to_decimal(o) for o in train_X['under_odds'].values])

for cv_thresh in [0.20, 0.25, 0.30]:
    for trend_thresh in [2, 3]:
        mask = (train_trend > trend_thresh) & (train_cv < cv_thresh)
        if mask.sum() < 10:
            continue
        hits = train_actual[mask] < train_line[mask]
        payouts = train_under_dec[mask]
        roi = (np.sum(payouts[hits]) - mask.sum()) / mask.sum()
        print(f"  TRAIN cv<{cv_thresh:.2f} trend>{trend_thresh}: n={mask.sum():4d}, hit={hits.mean():.1%}, ROI={roi:+.1%}")


# What props does this work for?
print("\n--- By prop type (test set, cv<0.2, trend>2) ---")
for prop in PROP_TYPES:
    prop_mask = test_meta['prop_type'].values == prop
    mask = (trend > 2) & (cv < 0.2) & prop_mask
    if mask.sum() < 5:
        print(f"  {prop:15s}: n={mask.sum():3d} (too few)")
        continue
    hits = test_actual[mask] < test_line[mask]
    payouts = test_under_dec[mask]
    roi = (np.sum(payouts[hits]) - mask.sum()) / mask.sum()
    print(f"  {prop:15s}: n={mask.sum():3d}, hit={hits.mean():.1%}, ROI={roi:+.1%}")

# Who are these low-CV hot players?
print("\n--- Sample low-CV hot players (cv<0.2, trend>2) ---")
mask = (trend > 2) & (cv < 0.2)
sample_idx = np.where(mask)[0][:20]
for i in sample_idx:
    player = test_meta.iloc[i]['player']
    prop = test_meta.iloc[i]['prop_type']
    line = test_meta.iloc[i]['line']
    actual = test_meta.iloc[i]['actual']
    t = trend[i]
    c = cv[i]
    hit = "UNDER HIT" if actual < line else "OVER HIT"
    print(f"  {player:20s} {prop:15s} line={line:5.1f} actual={actual:5.1f} trend={t:+.1f} cv={c:.3f} {hit}")


# ──────────────────────────────────────────────────────────────────────────────
# DEEP DIVE 2: Points-specific hot streak fade (trend>5 for points)
# ──────────────────────────────────────────────────────────────────────────────

print("\n\n" + "=" * 80)
print("DEEP DIVE 2: Points hot streak (trend > 5)")
print("=" * 80)

for prop in PROP_TYPES:
    prop_mask = test_meta['prop_type'].values == prop
    for trend_thresh in [3, 4, 5, 7]:
        mask = prop_mask & (trend > trend_thresh)
        if mask.sum() < 20:
            continue
        hits = test_actual[mask] < test_line[mask]
        payouts = test_under_dec[mask]
        roi = (np.sum(payouts[hits]) - mask.sum()) / mask.sum()
        ci_low, ci_med, ci_high = bootstrap_roi(hits, payouts)
        print(f"  {prop:15s} trend>{trend_thresh}: n={mask.sum():4d}, hit={hits.mean():.1%}, ROI={roi:+.1%}, CI=[{ci_low:+.1%}, {ci_high:+.1%}]")


# ──────────────────────────────────────────────────────────────────────────────
# DEEP DIVE 3: Late-season fade signal (gp 50+, juice fade)
# ──────────────────────────────────────────────────────────────────────────────

print("\n\n" + "=" * 80)
print("DEEP DIVE 3: Late-season signals")
print("=" * 80)

gp = test_X['games_played_capped'].values
juice_gap = np.abs(test_X['over_odds'].values) - np.abs(test_X['under_odds'].values)

for gp_min in [40, 50, 60]:
    gp_mask = gp >= gp_min
    if gp_mask.sum() < 100:
        continue

    # Regression
    hot = gp_mask & (trend > 2)
    if hot.sum() > 20:
        hits = test_actual[hot] < test_line[hot]
        roi = (np.sum(test_under_dec[hot][hits]) - hot.sum()) / hot.sum()
        ci_low, ci_med, ci_high = bootstrap_roi(hits, test_under_dec[hot])
        print(f"  gp>={gp_min} hot->under: n={hot.sum():4d}, hit={hits.mean():.1%}, ROI={roi:+.1%}, CI=[{ci_low:+.1%}, {ci_high:+.1%}]")

    # Low CV hot
    low_cv_hot = gp_mask & (trend > 2) & (cv < 0.3)
    if low_cv_hot.sum() > 20:
        hits = test_actual[low_cv_hot] < test_line[low_cv_hot]
        roi = (np.sum(test_under_dec[low_cv_hot][hits]) - low_cv_hot.sum()) / low_cv_hot.sum()
        ci_low, ci_med, ci_high = bootstrap_roi(hits, test_under_dec[low_cv_hot])
        print(f"  gp>={gp_min} low-cv-hot->under: n={low_cv_hot.sum():4d}, hit={hits.mean():.1%}, ROI={roi:+.1%}, CI=[{ci_low:+.1%}, {ci_high:+.1%}]")


# ──────────────────────────────────────────────────────────────────────────────
# DEEP DIVE 4: E2 specialist model - points UNDER edge
# ──────────────────────────────────────────────────────────────────────────────

print("\n\n" + "=" * 80)
print("DEEP DIVE 4: Points specialist UNDER edge")
print("=" * 80)

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    base_features = [
        'target_l3', 'target_l5', 'target_l10', 'target_l20', 'target_season',
        'target_std_l10', 'target_std_season', 'target_trend', 'target_cv',
        'hit_rate_l5', 'hit_rate_l10', 'line_z_l10',
        'mins_l3', 'mins_l5', 'mins_l10', 'mins_season', 'mins_std', 'mins_trend',
        'per_minute_rate', 'per_minute_rate_l5', 'expected_from_rate', 'expected_from_rate_l5',
        'line', 'over_odds', 'under_odds', 'implied_over_prob', 'implied_under_prob',
        'line_vs_season', 'line_vs_l5', 'line_vs_l3', 'games_played_capped',
    ]
    base_features = [f for f in base_features if f in X.columns]

    def safe_fillna(df):
        return df.fillna(0).replace([np.inf, -np.inf], 0)

    # Points specialist
    for prop in ['points', 'pts_rebs_asts']:
        tr_mask = train_meta['prop_type'].values == prop
        te_mask = test_meta['prop_type'].values == prop
        va_mask = val_meta['prop_type'].values == prop

        train_target = (train_actual[tr_mask] > train_line[tr_mask]).astype(int)
        test_target = (test_actual[te_mask] > test_line[te_mask]).astype(int)

        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=30, random_state=42
        )
        model.fit(safe_fillna(train_X[base_features].iloc[np.where(tr_mask)]), train_target)
        probs = model.predict_proba(safe_fillna(test_X[base_features].iloc[np.where(te_mask)]))[:, 1]

        print(f"\n  {prop} specialist model:")
        print(f"  AUC: {roc_auc_score(test_target, probs):.4f}")

        te_under_dec = test_under_dec[te_mask]
        te_over_dec = test_over_dec[te_mask]
        te_actual = test_actual[te_mask]
        te_line = test_line[te_mask]

        # UNDER bets at various thresholds
        for thresh in [0.40, 0.42, 0.45, 0.48]:
            under_mask = probs < thresh
            if under_mask.sum() > 10:
                hits = te_actual[under_mask] < te_line[under_mask]
                payouts = te_under_dec[under_mask]
                roi = (np.sum(payouts[hits]) - under_mask.sum()) / under_mask.sum()
                ci_low, _, ci_high = bootstrap_roi(hits, payouts)
                print(f"    Under (p<{thresh}): n={under_mask.sum():4d}, hit={hits.mean():.1%}, ROI={roi:+.1%}, CI=[{ci_low:+.1%}, {ci_high:+.1%}]")

        # Feature importance
        feat_imp = pd.Series(model.feature_importances_, index=base_features).sort_values(ascending=False)
        print(f"  Top 10 features:")
        for feat, imp in feat_imp.head(10).items():
            print(f"    {feat:30s}: {imp:.4f}")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────────────
# DEEP DIVE 5: Is the under bias real or an artifact?
# ──────────────────────────────────────────────────────────────────────────────

print("\n\n" + "=" * 80)
print("DEEP DIVE 5: Under bias analysis")
print("=" * 80)

# Check each split
for split_name, s_actual, s_line in [
    ('Train', train_actual, train_line),
    ('Val', val_actual, val_line),
    ('Test', test_actual, test_line),
]:
    over = np.mean(s_actual > s_line)
    under = np.mean(s_actual < s_line)
    push = np.mean(s_actual == s_line)
    print(f"  {split_name:5s}: over={over:.1%}, under={under:.1%}, push={push:.1%}")

# By prop type in test
print("\n  By prop type (test set):")
for prop in PROP_TYPES:
    mask = test_meta['prop_type'].values == prop
    p_over = np.mean(test_actual[mask] > test_line[mask])
    p_under = np.mean(test_actual[mask] < test_line[mask])
    # Is under rate statistically different from 50%?
    z = (p_under - 0.5) / np.sqrt(0.25 / mask.sum())
    p_val = 2 * (1 - scipy_stats.norm.cdf(abs(z)))
    sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
    print(f"    {prop:15s}: over={p_over:.1%}, under={p_under:.1%}, z={z:.2f}, p={p_val:.3f} {sig}")


# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

print("\n\n" + "=" * 80)
print("SUMMARY OF FINDINGS")
print("=" * 80)

print("""
APPROACH L (Regression to Mean):
- Raw trend signal: WEAK. Hot (trend>2) under hits at ~51%, ROI around -3%.
  Not profitable after vig.
- LOW-CV HOT PLAYERS: BEST FINDING. cv<0.2 + trend>2 -> 64% under hit rate, +20.5% ROI.
  BUT only 81 legs in test. Needs significance check.
- cv<0.3 + trend>2 -> 58% under, +8.7% ROI. 387 legs. More volume.
- Points-specific: trend>5 for points -> 57% under hit, +7.2% ROI. (206 legs)
- Cold player -> over: DOES NOT WORK. Market already adjusts for cold streaks.

APPROACH N (Fade the Public):
- Juice gap signal: DOES NOT WORK. Fading the public loses money across all thresholds.
  The "public" side is often the correct side.
- Combined with regression: also does not work.
- The juice gap is more about the book's margin than public action.

APPROACH O (Injury Cascade):
- Proxy detection via minutes trend: marginal at best (~52% under, -3% ROI).
- Co-occurrence-based cascade detection: under hits slightly more in cascade games
  but not profitably.
- The market already prices in injury effects reasonably well.

APPROACH E2 (Specialist Ensemble):
- Global GBT AUC: 0.548 (modest but real).
- Specialists mostly UNDERPERFORM the global model (less data = more noise).
- Exception: high-line specialist (+0.010 AUC delta).
- Points specialist: UNDER at p<0.45 shows +3.9% ROI. Worth monitoring.
- PRA specialist: UNDER at p<0.43 shows +5.0% ROI.

APPROACH P (Per-100-Possessions):
- Adding per-100 features: NO IMPROVEMENT (AUC delta -0.0007).
- Per-100 features rank 21st and 25th in importance.
- The model already captures pace effects through other features (team_pace, mins, etc).

BEST ACTIONABLE SIGNALS:
1. Low-CV hot player UNDER (cv<0.3, trend>2): 58% hit, +8.7% ROI, 387 legs
2. Points extreme hot UNDER (trend>5): 57% hit, +7.2% ROI, 206 legs
3. GBT model high-confidence UNDER: 60% hit at p<0.40, but -2.5% ROI (vig kills it)

CAVEATS:
- Small sample sizes (81-387 legs) mean high variance in ROI estimates
- Need to bootstrap confidence intervals to assess significance
- The UNDER bias in the dataset (~53% vs 47%) is a tailwind for all under signals
""")

print("DONE.")
