#!/usr/bin/env python3
"""
train_v3.py — V3 classifier: predict when UNDER hits using multi-book features.

Uses 95K legs with real per-book odds from The Odds API.
Train: 2023-24 + first half 2024-25
Val: second half 2024-25
OOS: 2025-26
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models" / "v3_multibook"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

BOOK_SHARPNESS = {
    'pinnacle': 1.0, 'pinnaclesports': 1.0,
    'williamhill_us': 0.70, 'caesars': 0.70,
    'fanduel': 0.55, 'draftkings': 0.50,
    'betmgm': 0.55, 'bovada': 0.50, 'betrivers': 0.45,
    'pointsbetus': 0.45, 'betonlineag': 0.50, 'mybookieag': 0.40,
    'lowvig': 0.85, 'fliff': 0.40, 'espnbet': 0.45,
    'superbook': 0.50, 'hardrockbet': 0.45, 'ballybet': 0.40,
    'fanatics': 0.45, 'rebet': 0.40, 'betparx': 0.45,
    'unibet_us': 0.45, 'barstool': 0.45, 'wynnbet': 0.45,
    'hardrockbet_az': 0.45, 'hardrockbet_fl': 0.45, 'windcreek': 0.40,
}

PROP_ENCODING = {
    'points': 0, 'rebounds': 1, 'assists': 2, 'threes': 3,
    'steals': 4, 'blocks': 5, 'turnovers': 6,
    'pts_rebs_asts': 7, 'pts_rebs': 8, 'pts_asts': 9,
    'rebs_asts': 10, 'stls_blks': 11,
    'double_double': 12, 'triple_double': 13,
}


def american_to_prob(odds):
    if odds >= 100:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def american_to_decimal(odds):
    if odds >= 100:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def devig(over_odds, under_odds):
    po = american_to_prob(over_odds)
    pu = american_to_prob(under_odds)
    t = po + pu
    if t <= 0:
        return 0.5, 0.5
    return po / t, pu / t


def engineer_features(leg):
    """Extract features from a single leg with multi-book data."""
    books = leg.get("all_books", {})
    if len(books) < 2:
        return None

    dk_over = leg.get("over_odds", -110)
    dk_under = leg.get("under_odds", -110)
    dk_line = leg.get("line", 0)

    # --- DK features ---
    dk_implied_over = american_to_prob(dk_over)
    dk_implied_under = american_to_prob(dk_under)
    dk_vig = dk_implied_over + dk_implied_under - 1.0
    dk_devig_over, dk_devig_under = devig(dk_over, dk_under)

    # --- Per-book devigged probabilities ---
    book_probs_under = []
    book_probs_over = []
    book_lines = []
    sharp_probs_under = []
    sharp_probs_over = []
    sharp_weights = []
    books_favoring_under = 0

    for bk, data in books.items():
        over_o = data.get("over_odds")
        under_o = data.get("under_odds")
        bk_line = data.get("line", dk_line)
        if not over_o or not under_o:
            continue

        po, pu = devig(over_o, under_o)
        book_probs_over.append(po)
        book_probs_under.append(pu)
        book_lines.append(bk_line)

        w = BOOK_SHARPNESS.get(bk, 0.45)
        sharp_probs_over.append(po * w)
        sharp_probs_under.append(pu * w)
        sharp_weights.append(w)

        if pu > 0.5:
            books_favoring_under += 1

    if not book_probs_under:
        return None

    n_books = len(book_probs_under)
    total_w = sum(sharp_weights)

    # --- Consensus features ---
    consensus_under = sum(sharp_probs_under) / total_w
    consensus_over = sum(sharp_probs_over) / total_w
    t = consensus_under + consensus_over
    consensus_under /= t
    consensus_over /= t

    equal_under = np.mean(book_probs_under)
    equal_over = np.mean(book_probs_over)

    # --- Pinnacle features ---
    pin = books.get("pinnacle") or books.get("pinnaclesports")
    has_pinnacle = 0
    pin_under_prob = 0.5
    pin_over_prob = 0.5
    pin_vs_dk_line = 0.0
    if pin and pin.get("over_odds") and pin.get("under_odds"):
        has_pinnacle = 1
        pin_over_prob, pin_under_prob = devig(pin["over_odds"], pin["under_odds"])
        pin_vs_dk_line = pin.get("line", dk_line) - dk_line

    # --- Deviation features ---
    dk_vs_consensus_prob = dk_devig_under - consensus_under
    dk_vs_equal_prob = dk_devig_under - equal_under
    dk_vs_pinnacle_prob = dk_devig_under - pin_under_prob if has_pinnacle else 0.0

    # --- EV features ---
    dec_under = american_to_decimal(dk_under)
    dec_over = american_to_decimal(dk_over)
    ev_under = consensus_under * (dec_under - 1) - (1 - consensus_under)
    ev_over = consensus_over * (dec_over - 1) - (1 - consensus_over)

    # --- Distribution features ---
    prob_std = np.std(book_probs_under) if n_books > 1 else 0.0
    prob_range = max(book_probs_under) - min(book_probs_under) if n_books > 1 else 0.0
    line_range = max(book_lines) - min(book_lines) if len(book_lines) > 1 else 0.0
    line_std = np.std(book_lines) if len(book_lines) > 1 else 0.0

    # --- Book agreement ---
    under_agreement = books_favoring_under / n_books

    # --- Line position features ---
    consensus_line = np.median(book_lines)
    dk_vs_consensus_line = dk_line - consensus_line

    # --- Odds shape features ---
    dk_under_is_plus = 1 if dk_under >= 100 else 0
    dk_over_is_plus = 1 if dk_over >= 100 else 0

    # --- Best available odds ---
    best_under_dec = max(american_to_decimal(d.get("under_odds", -110))
                        for d in books.values() if d.get("under_odds"))
    best_vs_dk_under = best_under_dec - dec_under

    feats = {
        # Multi-book consensus
        "consensus_under_prob": consensus_under,
        "consensus_over_prob": consensus_over,
        "equal_under_prob": equal_under,
        "ev_under": ev_under,
        "ev_over": ev_over,

        # DK position
        "dk_devig_under": dk_devig_under,
        "dk_devig_over": dk_devig_over,
        "dk_vig": dk_vig,
        "dk_implied_under": dk_implied_under,
        "dk_vs_consensus_prob": dk_vs_consensus_prob,
        "dk_vs_equal_prob": dk_vs_equal_prob,
        "dk_vs_pinnacle_prob": dk_vs_pinnacle_prob,
        "dk_vs_consensus_line": dk_vs_consensus_line,

        # Pinnacle
        "has_pinnacle": has_pinnacle,
        "pin_under_prob": pin_under_prob,
        "pin_vs_dk_line": pin_vs_dk_line,

        # Book distribution
        "n_books": n_books,
        "prob_std": prob_std,
        "prob_range": prob_range,
        "line_range": line_range,
        "line_std": line_std,
        "under_agreement": under_agreement,

        # Odds shape
        "dk_under_is_plus": dk_under_is_plus,
        "dk_over_is_plus": dk_over_is_plus,
        "best_vs_dk_under": best_vs_dk_under,

        # Prop context
        "dk_line": dk_line,
        "prop_type": PROP_ENCODING.get(leg.get("prop_type", ""), -1),
    }
    return feats


def load_all_legs():
    """Load and combine all multibook legs."""
    files = [
        ("2023-24", "outputs/backtest_2023_24_multibook.json"),
        ("2024-25", "outputs/backtest_2024_25_multibook.json"),
        ("2025-26", "outputs/backtest_oos_multibook.json"),
    ]
    all_legs = []
    for season, fpath in files:
        with open(fpath) as f:
            legs = json.load(f)
        for leg in legs:
            leg["season"] = season
        all_legs.extend(legs)
    return all_legs


def build_dataset(legs):
    """Build feature matrix + labels from legs."""
    rows = []
    for leg in legs:
        feats = engineer_features(leg)
        if feats is None:
            continue

        actual = leg.get("actual", 0)
        line = leg.get("line", 0)
        if actual == line:
            continue  # skip pushes

        feats["under_hit"] = int(actual < line)
        feats["date"] = leg.get("date", "")
        feats["season"] = leg.get("season", "")
        feats["player"] = leg.get("player", "")
        feats["prop_type_str"] = leg.get("prop_type", "")
        feats["dk_over_odds"] = leg.get("over_odds", -110)
        feats["dk_under_odds"] = leg.get("under_odds", -110)
        rows.append(feats)

    return pd.DataFrame(rows)


def split_data(df):
    """Train: 2023-24 + first half 2024-25. Val: second half 2024-25. OOS: 2025-26."""
    train_mask = (df["season"] == "2023-24") | (
        (df["season"] == "2024-25") & (df["date"] < "2025-02-01")
    )
    val_mask = (df["season"] == "2024-25") & (df["date"] >= "2025-02-01")
    oos_mask = df["season"] == "2025-26"
    return df[train_mask].copy(), df[val_mask].copy(), df[oos_mask].copy()


FEATURE_COLS = [
    "consensus_under_prob", "consensus_over_prob", "equal_under_prob",
    "ev_under", "ev_over",
    "dk_devig_under", "dk_devig_over", "dk_vig", "dk_implied_under",
    "dk_vs_consensus_prob", "dk_vs_equal_prob", "dk_vs_pinnacle_prob",
    "dk_vs_consensus_line",
    "has_pinnacle", "pin_under_prob", "pin_vs_dk_line",
    "n_books", "prob_std", "prob_range", "line_range", "line_std",
    "under_agreement",
    "dk_under_is_plus", "dk_over_is_plus", "best_vs_dk_under",
    "dk_line", "prop_type",
]


def train_model(train_df, val_df):
    """Train LightGBM classifier."""
    import lightgbm as lgb

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["under_hit"].values
    X_val = val_df[FEATURE_COLS].values
    y_val = val_df["under_hit"].values

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_COLS, reference=dtrain)

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
        "seed": 42,
    }

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100),
        ],
    )
    return model


def evaluate(model, df, label):
    """Evaluate model on a dataset with ROI simulation."""
    from sklearn.metrics import roc_auc_score, log_loss

    X = df[FEATURE_COLS].values
    y = df["under_hit"].values
    probs = model.predict(X)

    auc = roc_auc_score(y, probs)
    ll = log_loss(y, probs)
    cal_above_50 = np.mean(y[probs > 0.5])

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Samples: {len(df):,}")
    print(f"  AUC: {auc:.4f}")
    print(f"  Log Loss: {ll:.4f}")
    print(f"  Base UNDER rate: {y.mean():.4f}")
    print(f"  When model says >50% UNDER: actual {cal_above_50:.4f} ({(probs>0.5).sum():,} legs)")

    # ROI simulation at various thresholds
    print(f"\n  --- ROI by model threshold (UNDER only at DK odds) ---")
    print(f"  {'Threshold':<12} {'Bets':>6} {'Win%':>6} {'ROI%':>8} {'Profit':>10}")
    print(f"  {'-'*48}")

    for thresh in [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]:
        mask = probs >= thresh
        if mask.sum() < 10:
            continue
        sub = df[mask]
        hits = sub["under_hit"].values
        odds = sub["dk_under_odds"].values

        profits = []
        for hit, uo in zip(hits, odds):
            if hit:
                payout = (10000.0 / abs(uo)) if uo < 0 else float(uo)
            else:
                payout = -100.0
            profits.append(payout)

        total_profit = sum(profits)
        roi = 100 * total_profit / (len(profits) * 100)
        wr = 100 * hits.mean()
        print(f"  {thresh:<12.2f} {mask.sum():>6} {wr:>5.1f}% {roi:>+7.2f}% ${total_profit:>+9.0f}")

    # ROI by prop type
    print(f"\n  --- ROI by prop (threshold=0.54) ---")
    mask = probs >= 0.54
    if mask.sum() > 0:
        sub = df[mask].copy()
        sub["prob"] = probs[mask]
        for prop in sorted(sub["prop_type_str"].unique()):
            psub = sub[sub["prop_type_str"] == prop]
            if len(psub) < 20:
                continue
            hits = psub["under_hit"].values
            odds = psub["dk_under_odds"].values
            profits = []
            for hit, uo in zip(hits, odds):
                payout = (10000.0 / abs(uo)) if uo < 0 else float(uo) if hit else -100.0
                profits.append(payout)
            roi = 100 * sum(profits) / (len(profits) * 100)
            wr = 100 * hits.mean()
            print(f"    {prop:<20} {len(psub):>5} bets  {wr:>5.1f}% win  {roi:>+7.2f}% ROI")

    # Monthly breakdown at 0.54
    print(f"\n  --- Monthly ROI (threshold=0.54) ---")
    if mask.sum() > 0:
        sub = df[mask].copy()
        sub["month"] = sub["date"].str[:7]
        for month in sorted(sub["month"].unique()):
            msub = sub[sub["month"] == month]
            hits = msub["under_hit"].values
            odds = msub["dk_under_odds"].values
            profits = [((10000.0/abs(uo)) if uo < 0 else float(uo)) if h else -100.0
                       for h, uo in zip(hits, odds)]
            roi = 100 * sum(profits) / (len(profits) * 100)
            print(f"    {month}: {len(msub):>4} bets  {roi:>+7.2f}% ROI")

    return probs


def main():
    print("Loading legs...")
    legs = load_all_legs()
    print(f"Total legs: {len(legs):,}")

    print("Building features...")
    df = build_dataset(legs)
    print(f"Feature matrix: {len(df):,} rows × {len(FEATURE_COLS)} features")

    train_df, val_df, oos_df = split_data(df)
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | OOS: {len(oos_df):,}")
    print(f"Train UNDER rate: {train_df['under_hit'].mean():.4f}")
    print(f"Val UNDER rate: {val_df['under_hit'].mean():.4f}")
    print(f"OOS UNDER rate: {oos_df['under_hit'].mean():.4f}")

    print("\nTraining LightGBM...")
    model = train_model(train_df, val_df)

    # Feature importance
    imp = sorted(zip(FEATURE_COLS, model.feature_importance("gain")),
                 key=lambda x: -x[1])
    print("\nTop features (gain):")
    for name, gain in imp[:15]:
        print(f"  {name:<30} {gain:>10.0f}")

    # Evaluate on all splits
    evaluate(model, train_df, "TRAIN (2023-24 + early 2024-25)")
    evaluate(model, val_df, "VALIDATION (late 2024-25)")
    oos_probs = evaluate(model, oos_df, "OUT-OF-SAMPLE (2025-26)")

    # Save model
    model.save_model(str(MODELS_DIR / "v3_under_classifier.txt"))
    print(f"\nModel saved: {MODELS_DIR / 'v3_under_classifier.txt'}")

    # Save OOS predictions for further analysis
    oos_df = oos_df.copy()
    oos_df["v3_prob"] = oos_probs
    oos_df.to_csv(str(MODELS_DIR / "oos_predictions.csv"), index=False)
    print(f"OOS predictions saved: {MODELS_DIR / 'oos_predictions.csv'}")


if __name__ == "__main__":
    main()
