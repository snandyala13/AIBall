#!/usr/bin/env python3
"""
Fetch historical steals/blocks lines from The Odds API, match to box scores,
build features, and train v3-style binary classifiers.

Steps:
  1. Fetch steals + blocks prop lines from The Odds API (historical endpoint)
  2. Match to actual box score outcomes
  3. Build features via FeatureBuilder
  4. Train LightGBM binary classifiers (same architecture as v3_holdout)
  5. Save models alongside existing v3_classifier_holdout models

Usage:
  python train_steals_blocks.py                    # Full pipeline
  python train_steals_blocks.py --skip-fetch       # Skip API fetch, use cached data
  python train_steals_blocks.py --skip-features    # Skip fetch+features, use cached
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models" / "v3_classifier_holdout"

API_KEY = os.environ.get("ODDS_API_KEY", "")
API_BASE = 'https://api.the-odds-api.com/v4/historical/sports/basketball_nba'

PROP_MARKETS = ['player_steals', 'player_blocks']
PROP_TYPE_MAP = {'player_steals': 'steals', 'player_blocks': 'blocks'}

# Hyperparameter grid (same as v3_holdout)
NUM_LEAVES_OPTIONS = [15, 31, 63]
LR_OPTIONS = [0.01, 0.03, 0.05]
MAX_DEPTH_OPTIONS = [4, 6, 8]


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Fetch historical steals/blocks lines from The Odds API
# ═══════════════════════════════════════════════════════════════════════════════

def generate_sample_dates() -> List[str]:
    """Generate ~200 sample dates spread across both seasons."""
    dates = []

    # 2023-24 season: Oct 2023 - Jun 2024 (every 3rd day)
    d = datetime(2023, 10, 24)
    while d < datetime(2024, 6, 15):
        dates.append(d.strftime('%Y-%m-%dT20:00:00Z'))
        d += timedelta(days=3)

    # 2025-26 season: Jan 2026 - Jun 2026 (every 3rd day)
    d = datetime(2026, 1, 1)
    while d < datetime(2026, 6, 15):
        dates.append(d.strftime('%Y-%m-%dT20:00:00Z'))
        d += timedelta(days=3)

    logger.info("Generated %d sample dates across both seasons", len(dates))
    return dates


def fetch_events_for_date(date_str: str) -> List[dict]:
    """Fetch NBA events for a given historical date."""
    try:
        r = requests.get(
            f'{API_BASE}/events',
            params={'apiKey': API_KEY, 'date': date_str},
            timeout=30,
        )
        if r.status_code != 200:
            logger.warning("Events fetch failed for %s: status %d", date_str, r.status_code)
            return []
        data = r.json()
        events = data.get('data', [])
        return events
    except Exception as e:
        logger.warning("Events fetch error for %s: %s", date_str, e)
        return []


def fetch_props_for_event(event_id: str, date_str: str) -> List[dict]:
    """Fetch steals + blocks props for a specific event from DraftKings."""
    try:
        r = requests.get(
            f'{API_BASE}/events/{event_id}/odds',
            params={
                'apiKey': API_KEY,
                'date': date_str,
                'regions': 'us',
                'markets': ','.join(PROP_MARKETS),
                'oddsFormat': 'american',
                'bookmakers': 'draftkings',
            },
            timeout=30,
        )
        if r.status_code != 200:
            return []
        data = r.json().get('data', r.json())
        bookmakers = data.get('bookmakers', [])
        if not bookmakers:
            return []

        dk = bookmakers[0]  # DraftKings
        legs = []
        for market in dk.get('markets', []):
            market_key = market.get('key', '')
            prop_type = PROP_TYPE_MAP.get(market_key)
            if not prop_type:
                continue

            outcomes = market.get('outcomes', [])
            # Group outcomes by player (description) and point
            player_outcomes = defaultdict(dict)
            for o in outcomes:
                player = o.get('description', '')
                side = o.get('name', '')  # "Over" or "Under"
                price = o.get('price', 0)
                point = o.get('point', 0)
                key = (player, point)
                player_outcomes[key][side] = price

            for (player, point), sides in player_outcomes.items():
                if 'Over' in sides and 'Under' in sides:
                    legs.append({
                        'player': player,
                        'prop_type': prop_type,
                        'line': point,
                        'over_odds': sides['Over'],
                        'under_odds': sides['Under'],
                    })
                elif 'Over' in sides:
                    # Only Over available - still useful
                    legs.append({
                        'player': player,
                        'prop_type': prop_type,
                        'line': point,
                        'over_odds': sides['Over'],
                        'under_odds': -110,  # default
                    })

        remaining = r.headers.get('x-requests-remaining', '?')
        return legs

    except Exception as e:
        logger.warning("Props fetch error for %s: %s", event_id, e)
        return []


def fetch_all_historical_lines() -> List[dict]:
    """Fetch steals/blocks lines from The Odds API across both seasons."""
    cache_path = OUTPUTS / 'steals_blocks_historical.json'

    if cache_path.exists():
        logger.info("Loading cached historical lines from %s", cache_path)
        with open(cache_path) as f:
            data = json.load(f)
        logger.info("Loaded %d cached legs", len(data))
        return data

    dates = generate_sample_dates()
    all_legs = []
    total_api_calls = 0
    dates_with_data = 0

    for i, date_str in enumerate(dates):
        date_display = date_str[:10]
        logger.info(
            "[%d/%d] Fetching events for %s (legs so far: %d)...",
            i + 1, len(dates), date_display, len(all_legs),
        )

        events = fetch_events_for_date(date_str)
        total_api_calls += 1
        time.sleep(0.5)

        if not events:
            continue

        date_legs = 0
        # Limit to 5 events per date to be API-efficient
        for event in events[:5]:
            eid = event.get('id', '')
            home = event.get('home_team', '')
            away = event.get('away_team', '')
            game_date = event.get('commence_time', '')[:10]

            props = fetch_props_for_event(eid, date_str)
            total_api_calls += 1
            time.sleep(0.5)

            for leg in props:
                leg['date'] = game_date
                leg['home_team'] = home
                leg['away_team'] = away
                leg['event_id'] = eid
                leg['matchup'] = f"{away} @ {home}"
                all_legs.append(leg)
                date_legs += 1

        if date_legs > 0:
            dates_with_data += 1

        # Progress check every 20 dates
        if (i + 1) % 20 == 0:
            logger.info(
                "  Progress: %d dates processed, %d with data, %d total legs, ~%d API calls",
                i + 1, dates_with_data, len(all_legs), total_api_calls,
            )

        # Save checkpoint every 50 dates
        if (i + 1) % 50 == 0:
            with open(cache_path, 'w') as f:
                json.dump(all_legs, f, indent=2)
            logger.info("  Checkpoint saved: %d legs", len(all_legs))

    # Final save
    with open(cache_path, 'w') as f:
        json.dump(all_legs, f, indent=2)

    logger.info(
        "Fetch complete: %d legs from %d dates (%d API calls)",
        len(all_legs), dates_with_data, total_api_calls,
    )
    return all_legs


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Match to actual box scores
# ═══════════════════════════════════════════════════════════════════════════════

def load_box_scores() -> dict:
    """Load all box score data from both seasons. Returns {date_team -> game}."""
    game_data_files = [
        OUTPUTS / "backtest_2023_24_data.json",
        OUTPUTS / "backtest_oos_data.json",
    ]
    all_games = []
    for path in game_data_files:
        if not path.exists():
            logger.warning("Missing: %s", path)
            continue
        with open(path) as f:
            data = json.load(f)
        all_games.extend(data.get('games', []))

    logger.info("Loaded %d games with box scores", len(all_games))

    # Build lookup: (date, home_team) -> game and (date, away_team) -> game
    lookup = {}
    for game in all_games:
        date = game['date']
        home = game.get('home_team', '')
        away = game.get('away_team', '')
        lookup[(date, home)] = game
        lookup[(date, away)] = game

    return lookup


def normalize_player_name(name: str) -> str:
    """Normalize player name for matching (handle Jr., III, etc.)."""
    # The Odds API and BallDontLie may have slight name differences
    name = name.strip()
    # Common substitutions
    replacements = {
        'Nic Claxton': 'Nicolas Claxton',
        'Herb Jones': 'Herbert Jones',
        'PJ Washington': 'P.J. Washington',
        'OG Anunoby': 'O.G. Anunoby',
        'Jabari Smith': 'Jabari Smith Jr.',
        'Scotty Pippen': 'Scotty Pippen Jr.',
        'Trey Murphy': 'Trey Murphy III',
        'Dennis Smith': 'Dennis Smith Jr.',
        'Derrick Jones': 'Derrick Jones Jr.',
        'Gary Trent': 'Gary Trent Jr.',
        'Kelly Oubre': 'Kelly Oubre Jr.',
        'Larry Nance': 'Larry Nance Jr.',
        'Marcus Morris': 'Marcus Morris Sr.',
        'Michael Porter': 'Michael Porter Jr.',
        'Otto Porter': 'Otto Porter Jr.',
        'Robert Williams': 'Robert Williams III',
        'Tim Hardaway': 'Tim Hardaway Jr.',
        'Wendell Carter': 'Wendell Carter Jr.',
        'Jaren Jackson': 'Jaren Jackson Jr.',
        'Lonnie Walker': 'Lonnie Walker IV',
        'Kevin Porter': 'Kevin Porter Jr.',
        'Gary Payton': 'Gary Payton II',
        'Derrick White': 'Derrick White',  # no change but for safety
    }
    for key, val in replacements.items():
        if name.startswith(key) and name == key:
            return val
    return name


def match_legs_to_box_scores(legs: List[dict], box_lookup: dict) -> List[dict]:
    """Match historical lines to actual box scores to get outcomes."""
    matched = []
    unmatched_games = 0
    unmatched_players = 0
    matched_count = 0

    for leg in legs:
        date = leg['date']
        home_team = leg['home_team']
        away_team = leg['away_team']

        # Find the game
        game = box_lookup.get((date, home_team)) or box_lookup.get((date, away_team))
        if not game:
            unmatched_games += 1
            continue

        box = game.get('box_score', {})
        player_name = leg['player']

        # Try direct match first
        player_stats = box.get(player_name)

        # Try normalized name
        if not player_stats:
            norm_name = normalize_player_name(player_name)
            player_stats = box.get(norm_name)
            if player_stats:
                player_name = norm_name

        # Try fuzzy match: first+last name matching
        if not player_stats:
            parts = player_name.split()
            if len(parts) >= 2:
                first = parts[0].lower()
                last = parts[-1].lower()
                for bname in box:
                    bparts = bname.split()
                    if len(bparts) >= 2:
                        if bparts[0].lower() == first and bparts[-1].lower() == last:
                            player_stats = box[bname]
                            player_name = bname
                            break

        if not player_stats:
            unmatched_players += 1
            continue

        # Get actual stat value
        prop_type = leg['prop_type']
        if prop_type == 'steals':
            actual = player_stats.get('stl', 0)
        elif prop_type == 'blocks':
            actual = player_stats.get('blk', 0)
        else:
            continue

        hit = 1 if actual > leg['line'] else 0
        matched.append({
            'player': player_name,
            'prop_type': prop_type,
            'line': leg['line'],
            'over_odds': leg['over_odds'],
            'under_odds': leg['under_odds'],
            'actual': actual,
            'hit': hit,
            'date': date,
            'game_id': game.get('game_id'),
            'matchup': leg.get('matchup', f"{away_team} @ {home_team}"),
            'home_team': home_team,
            'away_team': away_team,
        })
        matched_count += 1

    logger.info(
        "Matching complete: %d matched, %d unmatched games, %d unmatched players",
        matched_count, unmatched_games, unmatched_players,
    )

    return matched


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Build features using FeatureBuilder
# ═══════════════════════════════════════════════════════════════════════════════

def build_features_for_legs(legs: List[dict]) -> tuple:
    """Build feature matrix for steals/blocks legs using FeatureBuilder."""
    from v2.features import FeatureBuilder

    fb = FeatureBuilder()
    fb.ensure_loaded()

    rows_X = []
    rows_y = []
    rows_meta = []
    skipped = 0

    for i, leg in enumerate(legs):
        if i > 0 and i % 2000 == 0:
            logger.info("  Building features: %d/%d (built: %d, skipped: %d)",
                        i, len(legs), len(rows_X), skipped)

        game_context = {
            'line': leg['line'],
            'over_odds': leg['over_odds'],
            'under_odds': leg['under_odds'],
            'total': 224.0,
            'spread': 0.0,
        }

        feats = fb.get_features_for_leg(
            leg['player'], leg['prop_type'], leg['date'], game_context
        )

        if feats is None:
            skipped += 1
            continue

        rows_X.append(feats)
        rows_y.append(1 if leg['actual'] > leg['line'] else 0)
        rows_meta.append({
            'date': leg['date'],
            'player': leg['player'],
            'prop_type': leg['prop_type'],
            'game_id': leg.get('game_id'),
            'line': leg['line'],
            'actual': leg['actual'],
            'matchup': leg.get('matchup', ''),
        })

    logger.info("Feature building: %d samples built, %d skipped", len(rows_X), skipped)

    X = pd.DataFrame(rows_X).fillna(0)
    y = np.array(rows_y)
    meta = pd.DataFrame(rows_meta)

    return X, y, meta


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: Add interaction features (same as v3_holdout)
# ═══════════════════════════════════════════════════════════════════════════════

INTERACTION_FEATURES = [
    'player_3pt_x_opp_3pt_def', 'catch_shoot_x_opp_3pt_def', 'pull_up_x_opp_3pt_def',
    'paint_touches_x_opp_def', 'drive_pts_x_opp_def', 'at_rim_fga_x_opp_def',
    'ast_ratio_x_opp_tov', 'passes_x_opp_pace',
    'oreb_pct_x_opp_orb_allowed', 'dreb_pct_x_opp_off_rating', 'reb_chances_x_pace',
    'usage_x_def_mismatch', 'usage_x_pace_delta',
    'fta_rate_x_opp_ft_rate_allowed', 'fouls_drawn_x_opp_def',
    'pts_per_touch_x_touches_x_pace', 'efg_x_opp_efg_allowed',
]


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add opponent scheme interaction features (same as v3_holdout)."""
    df = df.copy()

    # Only add interactions if source columns exist
    def safe_mul(col_a, col_b):
        if col_a in df.columns and col_b in df.columns:
            return df[col_a] * df[col_b]
        return pd.Series(0, index=df.index)

    if 'fg3_share_l10' in df.columns and 'opp_def_fg3_rate_allowed_l10' in df.columns:
        df['player_3pt_x_opp_3pt_def'] = df['fg3_share_l10'] * df['opp_def_fg3_rate_allowed_l10']
    if 'catch_shoot_fg3a' in df.columns and 'opp_def_fg3_rate_allowed_l10' in df.columns:
        df['catch_shoot_x_opp_3pt_def'] = df['catch_shoot_fg3a'] * df['opp_def_fg3_rate_allowed_l10']
    if 'pull_up_fg3a' in df.columns and 'opp_def_fg3_rate_allowed_l10' in df.columns:
        df['pull_up_x_opp_3pt_def'] = df['pull_up_fg3a'] * df['opp_def_fg3_rate_allowed_l10']
    if 'paint_touches' in df.columns and 'opp_def_rating_l10' in df.columns:
        df['paint_touches_x_opp_def'] = df['paint_touches'] * df['opp_def_rating_l10']
    if 'drive_pts' in df.columns and 'opp_def_rating_l10' in df.columns:
        df['drive_pts_x_opp_def'] = df['drive_pts'] * df['opp_def_rating_l10']
    if 'pbp_at_rim_fga' in df.columns and 'opp_def_rating_l10' in df.columns:
        df['at_rim_fga_x_opp_def'] = df['pbp_at_rim_fga'] * df['opp_def_rating_l10']
    if 'e_ast_ratio' in df.columns and 'opp_def_tov_forced_l10' in df.columns:
        df['ast_ratio_x_opp_tov'] = df['e_ast_ratio'] * df['opp_def_tov_forced_l10']
    if 'passes_made' in df.columns and 'opp_pace_l10' in df.columns:
        df['passes_x_opp_pace'] = df['passes_made'] * df['opp_pace_l10']
    if 'e_oreb_pct' in df.columns and 'opp_def_orb_allowed_l10' in df.columns:
        df['oreb_pct_x_opp_orb_allowed'] = df['e_oreb_pct'] * df['opp_def_orb_allowed_l10']
    if 'e_dreb_pct' in df.columns and 'opp_off_rating_l10' in df.columns:
        df['dreb_pct_x_opp_off_rating'] = df['e_dreb_pct'] * df['opp_off_rating_l10']
    if 'reb_chances' in df.columns and 'projected_pace' in df.columns:
        df['reb_chances_x_pace'] = df['reb_chances'] * df['projected_pace']
    if 'e_usg_pct' in df.columns and 'off_vs_def_mismatch' in df.columns:
        df['usage_x_def_mismatch'] = df['e_usg_pct'] * df['off_vs_def_mismatch']
    if 'e_usg_pct' in df.columns and 'pace_delta' in df.columns:
        df['usage_x_pace_delta'] = df['e_usg_pct'] * df['pace_delta']
    if 'fta_rate_l10' in df.columns and 'opp_def_ft_rate_allowed_l10' in df.columns:
        df['fta_rate_x_opp_ft_rate_allowed'] = df['fta_rate_l10'] * df['opp_def_ft_rate_allowed_l10']
    if 'pbp_shooting_fouls_drawn' in df.columns and 'opp_def_rating_l10' in df.columns:
        df['fouls_drawn_x_opp_def'] = df['pbp_shooting_fouls_drawn'] * df['opp_def_rating_l10']
    if 'pts_per_touch' in df.columns and 'touches' in df.columns and 'projected_pace' in df.columns:
        df['pts_per_touch_x_touches_x_pace'] = df['pts_per_touch'] * df['touches'] * df['projected_pace']
    if 'efg_pct_l10' in df.columns and 'opp_def_efg_allowed_l10' in df.columns:
        df['efg_x_opp_efg_allowed'] = df['efg_pct_l10'] * df['opp_def_efg_allowed_l10']

    df = df.fillna(0).replace([np.inf, -np.inf], 0)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: Train classifiers (same architecture as v3_holdout)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_sample_weights(dates, half_life_days=90):
    """Exponential decay weights with given half-life."""
    dates_dt = pd.to_datetime(dates)
    max_date = dates_dt.max()
    days_ago = (max_date - dates_dt).dt.days.values.astype(float)
    weights = np.exp(-np.log(2) * days_ago / half_life_days)
    return weights


def select_features_for_prop(X_train, y_train, sample_weights, n_features=70):
    """Train preliminary classifier and select top features by importance."""
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
    prelim.fit(X_train.values, y_train, sample_weight=sample_weights)

    importances = prelim.feature_importances_
    feat_imp = pd.Series(importances, index=X_train.columns)
    feat_imp = feat_imp.sort_values(ascending=False)

    top_feats = list(feat_imp.head(n_features).index)

    # Ensure interaction features with nonzero importance are included
    for f in INTERACTION_FEATURES:
        if f in X_train.columns and feat_imp.get(f, 0) > 0 and f not in top_feats:
            top_feats.append(f)

    return top_feats


def compute_roi(p_over, meta_slice, threshold):
    """Compute ROI for flat $100 bets at given confidence threshold."""
    actual = meta_slice['actual'].values
    line = meta_slice['line'].values
    payout = 100 / 1.1

    total_wagered = 0
    total_profit = 0
    n_bets = 0
    n_wins = 0

    for i in range(len(p_over)):
        if p_over[i] >= threshold:
            total_wagered += 100
            n_bets += 1
            if actual[i] > line[i]:
                total_profit += payout
                n_wins += 1
            else:
                total_profit -= 100
        elif (1 - p_over[i]) >= threshold:
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


def train_prop_classifier(prop_type, X, y, meta):
    """Train a v3-style classifier for steals or blocks."""
    print(f'\n{"=" * 60}')
    print(f'Training: {prop_type}')
    print(f'{"=" * 60}')

    # Filter to this prop type
    mask = meta['prop_type'] == prop_type
    X_prop = X[mask].reset_index(drop=True)
    y_prop = y[mask.values]
    meta_prop = meta[mask].reset_index(drop=True)

    # Sort by date
    sort_idx = meta_prop['date'].argsort()
    X_prop = X_prop.iloc[sort_idx].reset_index(drop=True)
    y_prop = y_prop[sort_idx]
    meta_prop = meta_prop.iloc[sort_idx].reset_index(drop=True)

    print(f'  Total {prop_type} legs: {len(X_prop)}')

    # Time-based split: try honest holdout first, fall back to walk-forward
    # Honest holdout: Train thru Feb 2026, Val Mar 2026, Test Apr-Jun 2026
    train_mask = meta_prop['date'] <= '2026-02-28'
    val_mask = (meta_prop['date'] > '2026-02-28') & (meta_prop['date'] <= '2026-03-31')
    test_mask = meta_prop['date'] > '2026-03-31'

    n_train_ho = train_mask.sum()
    n_val_ho = val_mask.sum()
    n_test_ho = test_mask.sum()

    # If val set is too small (<100) or test set is too small (<100),
    # fall back to 70/15/15 walk-forward split on ALL data.
    # Steals/blocks have limited 2025-26 data, so the honest holdout
    # often produces unreliable val/test sets.
    use_walkforward = n_val_ho < 100 or n_test_ho < 100

    if use_walkforward:
        print(f'  Holdout val/test too small (val={n_val_ho}, test={n_test_ho}), '
              f'using walk-forward 70/15/15 split')
        n = len(X_prop)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        train_mask = pd.Series([True] * train_end + [False] * (n - train_end))
        val_mask = pd.Series([False] * train_end + [True] * (val_end - train_end) + [False] * (n - val_end))
        test_mask = pd.Series([False] * val_end + [True] * (n - val_end))

    X_train = X_prop[train_mask].reset_index(drop=True)
    y_train = y_prop[train_mask.values]
    meta_train = meta_prop[train_mask].reset_index(drop=True)

    X_val = X_prop[val_mask].reset_index(drop=True)
    y_val = y_prop[val_mask.values]
    meta_val = meta_prop[val_mask].reset_index(drop=True)

    X_test = X_prop[test_mask].reset_index(drop=True)
    y_test = y_prop[test_mask.values]
    meta_test = meta_prop[test_mask].reset_index(drop=True)

    print(f'  Train: {len(X_train)} ({meta_train["date"].min() if len(meta_train) > 0 else "N/A"} to '
          f'{meta_train["date"].max() if len(meta_train) > 0 else "N/A"})')
    print(f'  Val:   {len(X_val)} ({meta_val["date"].min() if len(meta_val) > 0 else "N/A"} to '
          f'{meta_val["date"].max() if len(meta_val) > 0 else "N/A"})')
    print(f'  Test:  {len(X_test)} ({meta_test["date"].min() if len(meta_test) > 0 else "N/A"} to '
          f'{meta_test["date"].max() if len(meta_test) > 0 else "N/A"})')
    print(f'  Train over rate: {y_train.mean():.4f}' if len(y_train) > 0 else '  Train: empty')
    print(f'  Val over rate: {y_val.mean():.4f}' if len(y_val) > 0 else '  Val: empty')

    if len(X_train) < 100 or len(X_val) < 30:
        print(f'  SKIPPED: too few samples (train={len(X_train)}, val={len(X_val)})')
        return None

    # Sample weights
    sw_train = compute_sample_weights(meta_train['date'])

    # Feature selection
    selected_features = select_features_for_prop(X_train, y_train, sw_train, n_features=70)
    print(f'  Selected {len(selected_features)} features')

    X_tr_sel = X_train[selected_features].values
    X_va_sel = X_val[selected_features].values

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
                    X_tr_sel, y_train,
                    sample_weight=sw_train,
                    eval_set=[(X_va_sel, y_val)],
                    callbacks=[
                        lgb.early_stopping(30, verbose=False),
                        lgb.log_evaluation(0),
                    ],
                )

                p_val = clf.predict_proba(X_va_sel)[:, 1]
                ll = log_loss(y_val, np.clip(p_val, 1e-7, 1 - 1e-7))

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
    calibrator.fit(p_val_raw.reshape(-1, 1), y_val)

    p_val_cal = calibrator.predict_proba(p_val_raw.reshape(-1, 1))[:, 1]
    ll_cal = log_loss(y_val, np.clip(p_val_cal, 1e-7, 1 - 1e-7))
    print(f'  Calibrated val log loss: {ll_cal:.4f}')

    # Feature importance
    importances = best_model.feature_importances_
    feat_imp = {selected_features[i]: int(importances[i]) for i in range(len(selected_features))}
    feat_imp_sorted = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

    # ══════ EVALUATE ON TEST SET ══════
    test_metrics = {}
    if len(X_test) > 0:
        X_te_sel = X_test[selected_features].values
        p_raw = best_model.predict_proba(X_te_sel)[:, 1]
        p_cal = calibrator.predict_proba(p_raw.reshape(-1, 1))[:, 1]

        acc_raw = accuracy_score(y_test, (p_raw > 0.5).astype(int))
        acc_cal = accuracy_score(y_test, (p_cal > 0.5).astype(int))
        ll_raw = log_loss(y_test, np.clip(p_raw, 1e-7, 1 - 1e-7))
        ll_cal_test = log_loss(y_test, np.clip(p_cal, 1e-7, 1 - 1e-7))

        try:
            auc = roc_auc_score(y_test, p_cal)
        except ValueError:
            auc = 0.5

        brier = brier_score_loss(y_test, p_cal)

        # ROI
        roi_results = {}
        for thresh in [0.52, 0.55, 0.58, 0.60, 0.65]:
            roi_results[str(thresh)] = compute_roi(p_cal, meta_test, thresh)

        # Over/Under counts
        over_mask = p_cal > 0.5
        under_mask = p_cal <= 0.5
        over_count = int(over_mask.sum())
        under_count = int(under_mask.sum())
        over_hit = float(y_test[over_mask].mean()) if over_count > 0 else 0
        under_hit = float((1 - y_test[under_mask]).mean()) if under_count > 0 else 0

        # Monthly
        monthly = {}
        meta_te_copy = meta_test.copy()
        meta_te_copy['p_over'] = p_cal
        meta_te_copy['y'] = y_test
        meta_te_copy['month'] = pd.to_datetime(meta_te_copy['date']).dt.to_period('M').astype(str)
        for month, grp in meta_te_copy.groupby('month'):
            month_acc = accuracy_score(grp['y'], (grp['p_over'] > 0.5).astype(int))
            monthly[month] = {'accuracy': round(month_acc, 4), 'n_legs': len(grp)}

        # Calibration check
        cal_mask = (p_cal >= 0.55) & (p_cal < 0.60)
        cal_check = {
            'predicted_range': '[0.55, 0.60)',
            'n_bets': int(cal_mask.sum()),
            'actual_hit_rate': round(float(y_test[cal_mask].mean()), 4) if cal_mask.sum() > 0 else None,
            'predicted_mean': round(float(p_cal[cal_mask].mean()), 4) if cal_mask.sum() > 0 else None,
        }

        # Calibration buckets
        cal_buckets = []
        for lo, hi in [(0.3, 0.4), (0.4, 0.45), (0.45, 0.5), (0.5, 0.55), (0.55, 0.6), (0.6, 0.7)]:
            bmask = (p_cal >= lo) & (p_cal < hi)
            if bmask.sum() > 0:
                cal_buckets.append({
                    'bucket': f'{lo}-{hi}',
                    'predicted_mean': round(float(p_cal[bmask].mean()), 4),
                    'actual_mean': round(float(y_test[bmask].mean()), 4),
                    'count': int(bmask.sum()),
                })

        top_20 = dict(list(feat_imp_sorted.items())[:20])

        test_metrics = {
            'accuracy': round(acc_cal, 4),
            'accuracy_raw': round(acc_raw, 4),
            'log_loss': round(ll_cal_test, 4),
            'log_loss_raw': round(ll_raw, 4),
            'auc': round(auc, 4),
            'brier_score': round(brier, 4),
            'n_test': len(X_test),
            'roi_at_thresholds': roi_results,
            'over_hit_rate': round(over_hit, 4),
            'under_hit_rate': round(under_hit, 4),
            'over_count': over_count,
            'under_count': under_count,
            'calibration_buckets': cal_buckets,
            'calibration_check_55_60': cal_check,
            'per_month': monthly,
            'top_20_features': top_20,
        }

        print(f'\n  TEST SET RESULTS:')
        print(f'    Accuracy: {acc_cal:.4f} (raw: {acc_raw:.4f})')
        print(f'    Log loss: {ll_cal_test:.4f}')
        print(f'    AUC:      {auc:.4f}')
        print(f'    Brier:    {brier:.4f}')
        print(f'    Over/Under: {over_count}/{under_count}, hit: {over_hit:.4f}/{under_hit:.4f}')
        for t in ['0.52', '0.55', '0.58', '0.6', '0.65']:
            r = roi_results[t]
            if r['n_bets'] > 0:
                print(f'    @{t}: ROI={r["roi_pct"]:+.2f}%, bets={r["n_bets"]}, win={r["win_rate"]:.4f}')
        print(f'    Top 5 features: {list(feat_imp_sorted.keys())[:5]}')

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
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'feature_importances': feat_imp_sorted,
        'test_metrics': test_metrics,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: Save models and update training_meta.json
# ═══════════════════════════════════════════════════════════════════════════════

def save_models(prop_models: dict):
    """Save steals/blocks models and update training_meta.json."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save model files
    for prop_type, pm in prop_models.items():
        model_path = MODEL_DIR / f'{prop_type}_classifier.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(pm['model'], f)

        cal_path = MODEL_DIR / f'{prop_type}_calibrator.pkl'
        with open(cal_path, 'wb') as f:
            pickle.dump(pm['calibrator'], f)

        print(f'  Saved {prop_type}: classifier + calibrator')

    # Update training_meta.json (preserve existing props)
    meta_path = MODEL_DIR / 'training_meta.json'
    if meta_path.exists():
        with open(meta_path) as f:
            meta_json = json.load(f)
    else:
        meta_json = {}

    # Add steals/blocks to per_prop_config
    if 'per_prop_config' not in meta_json:
        meta_json['per_prop_config'] = {}
    if 'per_prop_metrics' not in meta_json:
        meta_json['per_prop_metrics'] = {}

    for prop_type, pm in prop_models.items():
        meta_json['per_prop_config'][prop_type] = {
            'features': pm['features'],
            'n_features': len(pm['features']),
            'hyperparams': pm['hyperparams'],
            'best_iteration': pm['best_iteration'],
            'best_val_log_loss': pm['best_val_log_loss'],
            'calibrated_val_log_loss': pm['calibrated_val_log_loss'],
            'calibrator_coef': pm['calibrator_coef'],
            'calibrator_intercept': pm['calibrator_intercept'],
        }
        if pm.get('test_metrics'):
            meta_json['per_prop_metrics'][prop_type] = pm['test_metrics']

    # Update feature_columns union
    all_features = set(meta_json.get('feature_columns', []))
    for pm in prop_models.values():
        all_features.update(pm['features'])
    meta_json['feature_columns'] = sorted(all_features)

    with open(meta_path, 'w') as f:
        json.dump(meta_json, f, indent=2, default=str)

    print(f'  Updated training_meta.json with steals/blocks')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train steals/blocks classifiers")
    parser.add_argument('--skip-fetch', action='store_true',
                        help='Skip API fetch, use cached data')
    parser.add_argument('--skip-features', action='store_true',
                        help='Skip fetch+features, use cached')
    args = parser.parse_args()

    start = datetime.now()
    print('=' * 70)
    print('STEALS/BLOCKS CLASSIFIER TRAINING')
    print(f'Started: {start}')
    print('=' * 70)

    legs_cache = OUTPUTS / 'steals_blocks_legs.json'
    features_cache = OUTPUTS / 'steals_blocks_features.parquet'
    meta_cache = OUTPUTS / 'steals_blocks_meta.parquet'
    y_cache = OUTPUTS / 'steals_blocks_y.npy'

    if args.skip_features and features_cache.exists():
        # Load cached features
        print('\nLoading cached features...')
        X = pd.read_parquet(features_cache)
        meta = pd.read_parquet(meta_cache)
        y = np.load(y_cache)
        print(f'  Loaded: {len(X)} samples, {len(X.columns)} features')

    elif args.skip_fetch and legs_cache.exists():
        # Load cached legs, build features
        print('\nLoading cached legs...')
        with open(legs_cache) as f:
            matched_legs = json.load(f)
        print(f'  Loaded: {len(matched_legs)} legs')

        print('\nStep 3: Building features...')
        X, y, meta = build_features_for_legs(matched_legs)

        # Cache features
        X.to_parquet(features_cache, index=False)
        meta.to_parquet(meta_cache, index=False)
        np.save(y_cache, y)

    else:
        # Full pipeline
        print('\nStep 1: Fetching historical lines from The Odds API...')
        raw_legs = fetch_all_historical_lines()
        print(f'  Total raw legs: {len(raw_legs)}')

        # Stats
        prop_counts = Counter(l['prop_type'] for l in raw_legs)
        print(f'  Steals lines: {prop_counts.get("steals", 0)}')
        print(f'  Blocks lines: {prop_counts.get("blocks", 0)}')

        print('\nStep 2: Matching to actual box scores...')
        box_lookup = load_box_scores()
        matched_legs = match_legs_to_box_scores(raw_legs, box_lookup)

        # Save matched legs
        with open(legs_cache, 'w') as f:
            json.dump(matched_legs, f, indent=2)
        print(f'  Saved {len(matched_legs)} matched legs to {legs_cache}')

        # Stats
        for pt in ['steals', 'blocks']:
            pt_legs = [l for l in matched_legs if l['prop_type'] == pt]
            if pt_legs:
                over_rate = sum(1 for l in pt_legs if l['hit']) / len(pt_legs)
                lines = [l['line'] for l in pt_legs]
                print(f'  {pt}: {len(pt_legs)} legs, over rate={over_rate:.3f}, '
                      f'lines: {Counter(lines).most_common(5)}')

        print('\nStep 3: Building features...')
        X, y, meta = build_features_for_legs(matched_legs)

        # Cache features
        X.to_parquet(features_cache, index=False)
        meta.to_parquet(meta_cache, index=False)
        np.save(y_cache, y)

    # Add interaction features
    print('\nAdding interaction features...')
    X = add_interaction_features(X)
    print(f'  Feature count after interactions: {X.shape[1]}')

    # Summary
    print(f'\nData summary:')
    for pt in ['steals', 'blocks']:
        pt_mask = meta['prop_type'] == pt
        n = pt_mask.sum()
        if n > 0:
            rate = y[pt_mask.values].mean()
            print(f'  {pt}: {n} legs, over rate={rate:.3f}')
            # Date breakdown
            dates = meta[pt_mask]['date']
            train_n = (dates <= '2026-02-28').sum()
            val_n = ((dates > '2026-02-28') & (dates <= '2026-03-31')).sum()
            test_n = (dates > '2026-03-31').sum()
            print(f'    Train: {train_n}, Val: {val_n}, Test: {test_n}')

    # Train classifiers
    print('\nStep 4: Training classifiers...')
    prop_models = {}
    for prop_type in ['steals', 'blocks']:
        result = train_prop_classifier(prop_type, X, y, meta)
        if result is not None:
            prop_models[prop_type] = result

    if not prop_models:
        print('\nERROR: No models trained. Check data availability.')
        sys.exit(1)

    # Save
    print('\nStep 5: Saving models...')
    save_models(prop_models)

    # Final report
    print(f'\n{"=" * 70}')
    print('FINAL REPORT')
    print(f'{"=" * 70}')
    print(f'\n{"Prop":<10} {"Train":>6} {"Val":>6} {"Test":>6} '
          f'{"Accuracy":>9} {"AUC":>6} {"LogLoss":>8} {"ROI@55%":>8}')
    print('-' * 63)
    for pt, pm in prop_models.items():
        tm = pm.get('test_metrics', {})
        roi_55 = tm.get('roi_at_thresholds', {}).get('0.55', {}).get('roi_pct', 0)
        print(f'{pt:<10} {pm["n_train"]:>6} {pm["n_val"]:>6} {pm.get("n_test", 0):>6} '
              f'{tm.get("accuracy", 0):>8.4f} {tm.get("auc", 0):>6.4f} '
              f'{tm.get("log_loss", 0):>8.4f} {roi_55:>+7.2f}%')

    # Compare to existing v3 props
    meta_path = MODEL_DIR / 'training_meta.json'
    if meta_path.exists():
        with open(meta_path) as f:
            all_meta = json.load(f)
        existing = all_meta.get('per_prop_metrics', {})
        print(f'\nComparison to existing v3_holdout props:')
        print(f'{"Prop":<15} {"Accuracy":>9} {"AUC":>6} {"LogLoss":>8}')
        print('-' * 42)
        for pt in ['points', 'rebounds', 'assists', 'threes', 'pts_rebs_asts', 'steals', 'blocks']:
            if pt in existing:
                m = existing[pt]
                print(f'{pt:<15} {m.get("accuracy", 0):>8.4f} '
                      f'{m.get("auc", 0):>6.4f} {m.get("log_loss", 0):>8.4f}')

    elapsed = (datetime.now() - start).total_seconds()
    print(f'\nCompleted in {elapsed:.1f}s')


if __name__ == '__main__':
    main()
