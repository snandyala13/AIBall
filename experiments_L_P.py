"""
Experiments L, N, O, E2, P: Regression-to-mean, Fade-the-Public, Injury Cascade,
Specialist Ensembles, and Per-100-Possessions normalization.

Walk-forward split: 70/15/15 by date. Report TEST set only.
"""

import pandas as pd
import numpy as np
import json
import warnings
from collections import defaultdict
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 80)
print("LOADING DATA")
print("=" * 80)

X = pd.read_parquet('outputs/v2_training/features_combined.parquet')
meta = pd.read_parquet('outputs/v2_training/metadata_enhanced.parquet')
targets = pd.read_parquet('outputs/v2_training/targets_enhanced.parquet')

n = len(X)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_X, train_meta, train_y = X.iloc[:train_end], meta.iloc[:train_end], targets.iloc[:train_end]
val_X, val_meta, val_y = X.iloc[train_end:val_end], meta.iloc[train_end:val_end], targets.iloc[train_end:val_end]
test_X, test_meta, test_y = X.iloc[val_end:], meta.iloc[val_end:], targets.iloc[val_end:]

print(f"Total: {n:,} rows")
print(f"Train: {len(train_X):,} ({train_meta['date'].iloc[0]} to {train_meta['date'].iloc[-1]})")
print(f"Val:   {len(val_X):,} ({val_meta['date'].iloc[0]} to {val_meta['date'].iloc[-1]})")
print(f"Test:  {len(test_X):,} ({test_meta['date'].iloc[0]} to {test_meta['date'].iloc[-1]})")
print()

PROP_TYPES = ['points', 'rebounds', 'assists', 'threes', 'pts_rebs_asts']

# Also load the full backtest legs for approach N
with open('outputs/backtest_2023_24_legs.json') as f:
    legs1 = json.load(f)
with open('outputs/backtest_oos_legs.json') as f:
    legs2 = json.load(f)
all_legs = legs1 + legs2
print(f"Total backtest legs: {len(all_legs):,}")
print()


def odds_to_prob(odds):
    """Convert American odds to implied probability."""
    if odds is None or np.isnan(odds):
        return 0.5
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def calc_roi(bets, hit_col='hit', odds_col='payout_odds'):
    """Calculate ROI from a list of bet dicts with hit and payout_odds."""
    if len(bets) == 0:
        return 0.0, 0.0, 0
    total_wagered = len(bets)
    total_returned = sum(b[odds_col] for b in bets if b[hit_col])
    roi = (total_returned - total_wagered) / total_wagered
    hit_rate = sum(1 for b in bets if b[hit_col]) / len(bets)
    return hit_rate, roi, len(bets)


def american_to_decimal(odds):
    """Convert American odds to decimal payout (per $1 wagered)."""
    if odds > 0:
        return odds / 100.0 + 1.0
    else:
        return 100.0 / abs(odds) + 1.0


# ──────────────────────────────────────────────────────────────────────────────
# BASELINE: Naive model (always pick the lower-juice side)
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 80)
print("BASELINE: Always-Under, Always-Over, and Random")
print("=" * 80)

test_actual = test_meta['actual'].values
test_line = test_meta['line'].values
test_over_odds = test_X['over_odds'].values
test_under_odds = test_X['under_odds'].values

overall_over_rate = np.mean(test_actual > test_line)
overall_under_rate = np.mean(test_actual < test_line)
overall_push_rate = np.mean(test_actual == test_line)

print(f"Test set: {len(test_X):,} legs")
print(f"Over rate:  {overall_over_rate:.1%}")
print(f"Under rate: {overall_under_rate:.1%}")
print(f"Push rate:  {overall_push_rate:.1%}")
print()

# Baseline ROI: always bet under at posted odds
under_dec = np.array([american_to_decimal(o) for o in test_under_odds])
over_dec = np.array([american_to_decimal(o) for o in test_over_odds])

under_hits = test_actual < test_line
over_hits = test_actual > test_line

always_under_roi = (np.sum(under_dec[under_hits]) - len(test_X)) / len(test_X)
always_over_roi = (np.sum(over_dec[over_hits]) - len(test_X)) / len(test_X)

print(f"Always-Under: hit={under_hits.mean():.1%}, ROI={always_under_roi:.1%}")
print(f"Always-Over:  hit={over_hits.mean():.1%}, ROI={always_over_roi:.1%}")
print()

baseline_results = {
    'always_under': {'hit_rate': float(under_hits.mean()), 'roi': float(always_under_roi), 'n': int(len(test_X))},
    'always_over': {'hit_rate': float(over_hits.mean()), 'roi': float(always_over_roi), 'n': int(len(test_X))},
    'over_rate': float(overall_over_rate),
    'under_rate': float(overall_under_rate),
    'push_rate': float(overall_push_rate),
}

# Per prop type baselines
for prop in PROP_TYPES:
    mask = test_meta['prop_type'].values == prop
    if mask.sum() == 0:
        continue
    p_actual = test_actual[mask]
    p_line = test_line[mask]
    p_under_dec = under_dec[mask]
    p_over_dec = over_dec[mask]
    p_under_hits = p_actual < p_line
    p_over_hits = p_actual > p_line
    u_roi = (np.sum(p_under_dec[p_under_hits]) - mask.sum()) / mask.sum()
    o_roi = (np.sum(p_over_dec[p_over_hits]) - mask.sum()) / mask.sum()
    print(f"  {prop:15s}: n={mask.sum():5d}, under_hit={p_under_hits.mean():.1%} roi={u_roi:+.1%}, over_hit={p_over_hits.mean():.1%} roi={o_roi:+.1%}")

print()

# ──────────────────────────────────────────────────────────────────────────────
# APPROACH L: Regression to Mean / "Luck" Detection
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 80)
print("APPROACH L: Regression to Mean / Luck Detection")
print("=" * 80)

approach_L_results = {}

# 1) Simple trend signal
print("\n--- L1: target_trend as raw signal ---")
print("When target_trend > 0, player is 'hot' (L3 > L10). Bet UNDER.")
print("When target_trend < 0, player is 'cold' (L3 < L10). Bet OVER.")
print()

trend = test_X['target_trend'].values

for threshold in [0, 1, 2, 3, 5]:
    # Hot players -> bet under
    hot = trend > threshold
    if hot.sum() > 50:
        h_under = test_actual[hot] < test_line[hot]
        h_dec = under_dec[hot]
        h_roi = (np.sum(h_dec[h_under]) - hot.sum()) / hot.sum()
        print(f"  Hot (trend>{threshold}): n={hot.sum():5d}, under_hit={h_under.mean():.1%}, ROI={h_roi:+.1%}")

    # Cold players -> bet over
    cold = trend < -threshold
    if cold.sum() > 50:
        c_over = test_actual[cold] > test_line[cold]
        c_dec = over_dec[cold]
        c_roi = (np.sum(c_dec[c_over]) - cold.sum()) / cold.sum()
        print(f"  Cold (trend<-{threshold}): n={cold.sum():5d}, over_hit={c_over.mean():.1%}, ROI={c_roi:+.1%}")
    print()

# 2) By prop type
print("\n--- L2: Regression signal by prop type ---")
for prop in PROP_TYPES:
    mask = test_meta['prop_type'].values == prop
    if mask.sum() < 100:
        continue
    p_trend = trend[mask]
    p_actual = test_actual[mask]
    p_line = test_line[mask]
    p_under_dec = under_dec[mask]
    p_over_dec = over_dec[mask]

    prop_results = {}
    print(f"\n  {prop}:")
    for threshold in [2, 3, 5]:
        hot = p_trend > threshold
        cold = p_trend < -threshold
        if hot.sum() > 30:
            h_under = p_actual[hot] < p_line[hot]
            h_roi = (np.sum(p_under_dec[hot][h_under]) - hot.sum()) / hot.sum()
            print(f"    Hot (trend>{threshold}): n={hot.sum():4d}, under_hit={h_under.mean():.1%}, ROI={h_roi:+.1%}")
            prop_results[f'hot_gt{threshold}'] = {'n': int(hot.sum()), 'under_hit': float(h_under.mean()), 'roi': float(h_roi)}
        if cold.sum() > 30:
            c_over = p_actual[cold] > p_line[cold]
            c_roi = (np.sum(p_over_dec[cold][c_over]) - cold.sum()) / cold.sum()
            print(f"    Cold (trend<-{threshold}): n={cold.sum():4d}, over_hit={c_over.mean():.1%}, ROI={c_roi:+.1%}")
            prop_results[f'cold_lt{threshold}'] = {'n': int(cold.sum()), 'over_hit': float(c_over.mean()), 'roi': float(c_roi)}

    approach_L_results[prop] = prop_results

# 3) Combined signal: trend + hit_rate_l5
print("\n\n--- L3: Combined regression signal (trend + hit_rate) ---")
hit_rate = test_X['hit_rate_l5'].values

# "Overdue" players: hot + high recent hit rate -> regression down
overdue_hot = (trend > 2) & (hit_rate > 0.8)
overdue_cold = (trend < -2) & (hit_rate < 0.2)

if overdue_hot.sum() > 30:
    h_under = test_actual[overdue_hot] < test_line[overdue_hot]
    h_roi = (np.sum(under_dec[overdue_hot][h_under]) - overdue_hot.sum()) / overdue_hot.sum()
    print(f"  Overdue-hot (trend>2 & hit_rate>.8): n={overdue_hot.sum():4d}, under_hit={h_under.mean():.1%}, ROI={h_roi:+.1%}")

if overdue_cold.sum() > 30:
    c_over = test_actual[overdue_cold] > test_line[overdue_cold]
    c_roi = (np.sum(over_dec[overdue_cold][c_over]) - overdue_cold.sum()) / overdue_cold.sum()
    print(f"  Overdue-cold (trend<-2 & hit_rate<.2): n={overdue_cold.sum():4d}, over_hit={c_over.mean():.1%}, ROI={c_roi:+.1%}")

# 4) CV-based filter: high variance players are less predictable
print("\n\n--- L4: CV filter (low variance = more predictable regression) ---")
cv = test_X['target_cv'].values

for cv_thresh in [0.2, 0.3, 0.5]:
    low_cv_hot = (trend > 2) & (cv < cv_thresh)
    if low_cv_hot.sum() > 30:
        h_under = test_actual[low_cv_hot] < test_line[low_cv_hot]
        h_roi = (np.sum(under_dec[low_cv_hot][h_under]) - low_cv_hot.sum()) / low_cv_hot.sum()
        print(f"  Low-CV hot (trend>2 & cv<{cv_thresh}): n={low_cv_hot.sum():4d}, under_hit={h_under.mean():.1%}, ROI={h_roi:+.1%}")

# 5) Does the signal improve later in the season?
print("\n\n--- L5: Seasonality of regression signal ---")
test_dates = pd.to_datetime(test_meta['date'].values)
test_months = test_dates.month

for month_range, label in [((10, 11, 12), 'Oct-Dec'), ((1, 2), 'Jan-Feb'), ((3, 4, 5, 6), 'Mar-Jun')]:
    month_mask = np.isin(test_months, month_range)
    hot_month = (trend > 2) & month_mask
    if hot_month.sum() > 30:
        h_under = test_actual[hot_month] < test_line[hot_month]
        h_roi = (np.sum(under_dec[hot_month][h_under]) - hot_month.sum()) / hot_month.sum()
        print(f"  {label} hot (trend>2): n={hot_month.sum():4d}, under_hit={h_under.mean():.1%}, ROI={h_roi:+.1%}")


# ──────────────────────────────────────────────────────────────────────────────
# APPROACH N: Fade the Public (Odds Skew as Contrarian Signal)
# ──────────────────────────────────────────────────────────────────────────────

print("\n\n" + "=" * 80)
print("APPROACH N: Fade the Public (Odds Skew)")
print("=" * 80)

approach_N_results = {}

# Using training data features (has over_odds, under_odds)
juice_gap = np.abs(test_over_odds) - np.abs(test_under_odds)
# positive juice_gap = more juice on over = public on over -> fade = bet under
# negative juice_gap = more juice on under = public on under -> fade = bet over

print("\n--- N1: Juice gap signal (test set parquet data) ---")
for gap_thresh in [10, 20, 30, 50]:
    # Public on over -> fade to under
    pub_over = juice_gap > gap_thresh
    if pub_over.sum() > 50:
        fade_under = test_actual[pub_over] < test_line[pub_over]
        fade_roi = (np.sum(under_dec[pub_over][fade_under]) - pub_over.sum()) / pub_over.sum()
        print(f"  Fade-over (gap>{gap_thresh}): n={pub_over.sum():5d}, under_hit={fade_under.mean():.1%}, ROI={fade_roi:+.1%}")

    # Public on under -> fade to over
    pub_under = juice_gap < -gap_thresh
    if pub_under.sum() > 50:
        fade_over = test_actual[pub_under] > test_line[pub_under]
        fade_roi = (np.sum(over_dec[pub_under][fade_over]) - pub_under.sum()) / pub_under.sum()
        print(f"  Fade-under (gap<-{gap_thresh}): n={pub_under.sum():5d}, over_hit={fade_over.mean():.1%}, ROI={fade_roi:+.1%}")
    print()

# N2: By prop type
print("--- N2: Juice gap by prop type ---")
for prop in PROP_TYPES:
    mask = test_meta['prop_type'].values == prop
    if mask.sum() < 100:
        continue
    p_gap = juice_gap[mask]
    p_actual = test_actual[mask]
    p_line = test_line[mask]
    p_under_dec = under_dec[mask]
    p_over_dec = over_dec[mask]

    prop_results = {}
    print(f"\n  {prop}:")
    for gap_thresh in [20, 30, 50]:
        pub_over = p_gap > gap_thresh
        pub_under = p_gap < -gap_thresh
        if pub_over.sum() > 30:
            fade_under = p_actual[pub_over] < p_line[pub_over]
            fade_roi = (np.sum(p_under_dec[pub_over][fade_under]) - pub_over.sum()) / pub_over.sum()
            print(f"    Fade-over (gap>{gap_thresh}): n={pub_over.sum():4d}, under_hit={fade_under.mean():.1%}, ROI={fade_roi:+.1%}")
            prop_results[f'fade_over_gt{gap_thresh}'] = {'n': int(pub_over.sum()), 'under_hit': float(fade_under.mean()), 'roi': float(fade_roi)}
        if pub_under.sum() > 30:
            fade_over = p_actual[pub_under] > p_line[pub_under]
            fade_roi = (np.sum(p_over_dec[pub_under][fade_over]) - pub_under.sum()) / pub_under.sum()
            print(f"    Fade-under (gap<-{gap_thresh}): n={pub_under.sum():4d}, over_hit={fade_over.mean():.1%}, ROI={fade_roi:+.1%}")
            prop_results[f'fade_under_lt{gap_thresh}'] = {'n': int(pub_under.sum()), 'over_hit': float(fade_over.mean()), 'roi': float(fade_roi)}

    approach_N_results[prop] = prop_results

# N3: Full 87K legs analysis
print("\n\n--- N3: Fade the public on ALL 87K backtest legs ---")
# Use walk-forward: only test portion (last 15%)
all_legs_sorted = sorted(all_legs, key=lambda x: x.get('date', ''))
n_legs = len(all_legs_sorted)
legs_test_start = int(n_legs * 0.85)
legs_test = all_legs_sorted[legs_test_start:]
print(f"Test legs: {len(legs_test):,} (dates: {legs_test[0]['date']} to {legs_test[-1]['date']})")

for gap_thresh in [10, 20, 30, 50]:
    fade_bets = []
    for leg in legs_test:
        over_o = leg.get('over_odds')
        under_o = leg.get('under_odds')
        actual = leg.get('actual')
        line = leg.get('line')
        if over_o is None or under_o is None or actual is None or line is None:
            continue
        gap = abs(over_o) - abs(under_o)
        if gap > gap_thresh:
            # Public on over -> bet under
            hit = actual < line
            payout = american_to_decimal(under_o)
            fade_bets.append({'hit': hit, 'payout': payout, 'side': 'under'})
        elif gap < -gap_thresh:
            # Public on under -> bet over
            hit = actual > line
            payout = american_to_decimal(over_o)
            fade_bets.append({'hit': hit, 'payout': payout, 'side': 'over'})

    if len(fade_bets) > 0:
        hr = sum(1 for b in fade_bets if b['hit']) / len(fade_bets)
        roi = (sum(b['payout'] for b in fade_bets if b['hit']) - len(fade_bets)) / len(fade_bets)
        n_under = sum(1 for b in fade_bets if b['side'] == 'under')
        n_over = sum(1 for b in fade_bets if b['side'] == 'over')
        print(f"  Gap>{gap_thresh}: n={len(fade_bets):5d} ({n_under} under, {n_over} over), hit={hr:.1%}, ROI={roi:+.1%}")

# N4: Combined with regression signal
print("\n\n--- N4: Combined fade + regression ---")
# When BOTH signals agree: trend says regression AND juice says public is on the wrong side
combined_under = (trend > 2) & (juice_gap > 20)
combined_over = (trend < -2) & (juice_gap < -20)

if combined_under.sum() > 20:
    h_under = test_actual[combined_under] < test_line[combined_under]
    h_roi = (np.sum(under_dec[combined_under][h_under]) - combined_under.sum()) / combined_under.sum()
    print(f"  Hot + public-over -> UNDER: n={combined_under.sum():4d}, under_hit={h_under.mean():.1%}, ROI={h_roi:+.1%}")

if combined_over.sum() > 20:
    c_over = test_actual[combined_over] > test_line[combined_over]
    c_roi = (np.sum(over_dec[combined_over][c_over]) - combined_over.sum()) / combined_over.sum()
    print(f"  Cold + public-under -> OVER: n={combined_over.sum():4d}, over_hit={c_over.mean():.1%}, ROI={c_roi:+.1%}")


# ──────────────────────────────────────────────────────────────────────────────
# APPROACH O: Injury Cascade Lookup Table
# ──────────────────────────────────────────────────────────────────────────────

print("\n\n" + "=" * 80)
print("APPROACH O: Injury Cascade Effects")
print("=" * 80)

approach_O_results = {}

# We don't have direct game box scores in the parquet, but we have features
# that capture injury effects: related stats, usage, minutes
# Check for injury-related features
injury_features = [c for c in X.columns if 'inj' in c.lower() or 'out' in c.lower() or 'miss' in c.lower()]
print(f"Injury-related features: {injury_features}")

# Alternative: use minutes_trend and usage changes as proxy for injury cascade
# When a teammate is out, remaining players see: higher minutes, higher usage
# Features we have: mins_trend, e_usg_pct, per_minute_rate changes

# Let's detect "injury cascade" situations from the data itself:
# - Player's minutes are trending up (mins_trend > 0)
# - Player's per-minute rate is stable but total stats are up
# This suggests they're getting more opportunity (teammate out)

mins_trend = test_X['mins_trend'].values
per_min_rate = test_X['per_minute_rate'].values
per_min_rate_l5 = test_X['per_minute_rate_l5'].values
expected_rate = test_X['expected_from_rate'].values
expected_rate_l5 = test_X['expected_from_rate_l5'].values

# Proxy for injury cascade: minutes up but per-minute rate stable
# This means player is getting more opportunity, not getting better
mins_up = mins_trend > 2  # 2+ extra minutes
rate_stable = np.abs(per_min_rate - per_min_rate_l5) < 0.05  # stable rate
cascade_proxy = mins_up & rate_stable

print(f"\n--- O1: Injury cascade proxy (mins up + stable rate) ---")
print(f"Cascade situations detected: {cascade_proxy.sum()}")

if cascade_proxy.sum() > 30:
    # In cascade: the line may be set based on recent inflated stats
    # But the player's rate hasn't changed -> over-line -> bet under
    c_under = test_actual[cascade_proxy] < test_line[cascade_proxy]
    c_roi = (np.sum(under_dec[cascade_proxy][c_under]) - cascade_proxy.sum()) / cascade_proxy.sum()
    print(f"  Cascade under: hit={c_under.mean():.1%}, ROI={c_roi:+.1%}")

    c_over = test_actual[cascade_proxy] > test_line[cascade_proxy]
    c_roi_o = (np.sum(over_dec[cascade_proxy][c_over]) - cascade_proxy.sum()) / cascade_proxy.sum()
    print(f"  Cascade over: hit={c_over.mean():.1%}, ROI={c_roi_o:+.1%}")

# O2: Minutes vs expected from rate
print(f"\n--- O2: Rate-based overperformance detection ---")
# If expected_from_rate (season) >> line, player is getting more than line expects
# If expected_from_rate_l5 >> expected_from_rate (season), recent boost
rate_boost = expected_rate_l5 - expected_rate
line_vs_expected = test_line - expected_rate

print(f"Rate boost (L5 rate - season rate) stats:")
print(f"  mean={np.nanmean(rate_boost):.2f}, std={np.nanstd(rate_boost):.2f}")

for boost_thresh in [1, 2, 3]:
    boosted = rate_boost > boost_thresh
    if boosted.sum() > 30:
        b_under = test_actual[boosted] < test_line[boosted]
        b_roi = (np.sum(under_dec[boosted][b_under]) - boosted.sum()) / boosted.sum()
        print(f"  Rate boosted >{boost_thresh}: n={boosted.sum():4d}, under_hit={b_under.mean():.1%}, ROI={b_roi:+.1%}")

    deboosted = rate_boost < -boost_thresh
    if deboosted.sum() > 30:
        d_over = test_actual[deboosted] > test_line[deboosted]
        d_roi = (np.sum(over_dec[deboosted][d_over]) - deboosted.sum()) / deboosted.sum()
        print(f"  Rate deboosted <-{boost_thresh}: n={deboosted.sum():4d}, over_hit={d_over.mean():.1%}, ROI={d_roi:+.1%}")

# O3: Build actual injury lookup from backtest game data
print(f"\n--- O3: Injury cascade from game-level data ---")

# Process legs to find teammate absence patterns
# Group by game_id and team
from collections import defaultdict

player_games = defaultdict(set)  # player -> set of game_ids
game_players = defaultdict(set)  # game_id -> set of players
game_player_stats = {}  # (game_id, player, prop_type) -> actual

# Use ALL legs (sorted by date), only analyze from test split
all_legs_by_date = sorted(all_legs, key=lambda x: x.get('date', ''))

for leg in all_legs_by_date:
    gid = leg.get('game_id')
    player = leg.get('player')
    prop = leg.get('prop_type')
    actual = leg.get('actual')
    if gid and player:
        player_games[player].add(gid)
        game_players[gid].add(player)
        if actual is not None:
            game_player_stats[(gid, player, prop)] = actual

# Identify "usual" players per team (same matchup players)
# Group players by team from matchup strings
player_team = {}
for leg in all_legs_by_date:
    player = leg.get('player')
    matchup = leg.get('matchup', '')
    # Try to identify team from matchup (this is approximate)
    if player and matchup:
        player_team[player] = matchup  # store last seen matchup

# Instead of team mapping, use co-occurrence: players who appear in the same games
# Find player pairs that co-occur in 70%+ of games
# This is expensive for all players, so sample the test period
test_legs = [l for l in all_legs_by_date if l.get('date', '') >= test_meta['date'].iloc[0]]
test_game_ids = set(l['game_id'] for l in test_legs if 'game_id' in l)
print(f"Test period games: {len(test_game_ids)}")

# For each test game, check if any frequent player is missing
# A "frequent player" is one who appeared in 70%+ of the team's last 20 games
# We'll approximate: for each game, look at which players are present
# Then for each player in the game, check if their usual co-players are absent

# Build co-occurrence matrix from training data
train_legs = [l for l in all_legs_by_date if l.get('date', '') < test_meta['date'].iloc[0]]
train_game_players = defaultdict(set)
for leg in train_legs:
    gid = leg.get('game_id')
    player = leg.get('player')
    if gid and player:
        train_game_players[gid].add(player)

# For each player, find their co-players (same game 70%+ of the time)
player_co_counts = defaultdict(lambda: defaultdict(int))
player_game_count = defaultdict(int)

for gid, players in train_game_players.items():
    for p in players:
        player_game_count[p] += 1
        for q in players:
            if p != q:
                player_co_counts[p][q] += 1

# For test period: check if a player's usual teammate is absent
injury_cascade_legs = []
normal_legs = []

test_game_players_map = defaultdict(set)
for leg in test_legs:
    gid = leg.get('game_id')
    player = leg.get('player')
    if gid and player:
        test_game_players_map[gid].add(player)

for leg in test_legs:
    gid = leg.get('game_id')
    player = leg.get('player')
    if not gid or not player or player_game_count[player] < 20:
        continue

    present = test_game_players_map.get(gid, set())
    # Check if any frequent co-player is absent
    teammate_absent = False
    absent_minutes = 0
    for co_player, co_count in player_co_counts[player].items():
        co_rate = co_count / max(player_game_count[player], 1)
        if co_rate > 0.7 and co_player not in present:
            teammate_absent = True
            break

    if teammate_absent:
        injury_cascade_legs.append(leg)
    else:
        normal_legs.append(leg)

print(f"Injury cascade legs: {len(injury_cascade_legs)}")
print(f"Normal legs: {len(normal_legs)}")

# Compare hit rates
if len(injury_cascade_legs) > 50:
    for prop in PROP_TYPES:
        cascade_prop = [l for l in injury_cascade_legs if l['prop_type'] == prop]
        normal_prop = [l for l in normal_legs if l['prop_type'] == prop]
        if len(cascade_prop) > 20 and len(normal_prop) > 20:
            cas_over = np.mean([l['actual'] > l['line'] for l in cascade_prop])
            nor_over = np.mean([l['actual'] > l['line'] for l in normal_prop])
            print(f"  {prop:15s}: cascade_over={cas_over:.1%} (n={len(cascade_prop)}), normal_over={nor_over:.1%} (n={len(normal_prop)}), diff={cas_over-nor_over:+.1%}")
            approach_O_results[prop] = {
                'cascade_over_rate': float(cas_over),
                'normal_over_rate': float(nor_over),
                'n_cascade': len(cascade_prop),
                'n_normal': len(normal_prop),
            }


# ──────────────────────────────────────────────────────────────────────────────
# APPROACH E2: Ensemble of Specialist Models
# ──────────────────────────────────────────────────────────────────────────────

print("\n\n" + "=" * 80)
print("APPROACH E2: Ensemble of Specialist Models")
print("=" * 80)

approach_E2_results = {}

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import log_loss, roc_auc_score

    # Target: did the over hit?
    train_target = (train_meta['actual'].values > train_meta['line'].values).astype(int)
    val_target = (val_meta['actual'].values > val_meta['line'].values).astype(int)
    test_target = (test_meta['actual'].values > test_meta['line'].values).astype(int)

    # Use a fixed set of good features
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

    # E2a: Global model (baseline)
    print("\n--- E2a: Global GBT model (baseline) ---")
    tr_X = safe_fillna(train_X[base_features])
    va_X = safe_fillna(val_X[base_features])
    te_X = safe_fillna(test_X[base_features])

    global_model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=50, random_state=42
    )
    global_model.fit(tr_X, train_target)

    global_probs = global_model.predict_proba(te_X)[:, 1]
    global_auc = roc_auc_score(test_target, global_probs)
    print(f"  Global AUC: {global_auc:.4f}")

    # Betting performance of global model
    for threshold in [0.55, 0.57, 0.60]:
        over_bets = global_probs > threshold
        under_bets = global_probs < (1 - threshold)

        if over_bets.sum() > 30:
            o_hits = test_target[over_bets]
            o_dec = over_dec[over_bets]
            o_roi = (np.sum(o_dec[o_hits.astype(bool)]) - over_bets.sum()) / over_bets.sum()
            print(f"  Over (p>{threshold}): n={over_bets.sum():4d}, hit={o_hits.mean():.1%}, ROI={o_roi:+.1%}")

        if under_bets.sum() > 30:
            u_hits = (~test_target.astype(bool))[under_bets]
            u_dec = under_dec[under_bets]
            u_roi = (np.sum(u_dec[u_hits]) - under_bets.sum()) / under_bets.sum()
            print(f"  Under (p<{1-threshold}): n={under_bets.sum():4d}, hit={u_hits.mean():.1%}, ROI={u_roi:+.1%}")

    approach_E2_results['global'] = {'auc': float(global_auc), 'n_test': int(len(test_X))}

    # E2b: Specialist models
    print("\n--- E2b: Specialist models ---")

    specialists = {
        'high_line': lambda line: line > 20,
        'low_line': lambda line: line <= 5,
        'mid_line': lambda line: (line > 5) & (line <= 20),
    }

    specialist_probs = np.full(len(test_X), np.nan)

    for name, line_filter in specialists.items():
        tr_mask = line_filter(train_X['line'].values)
        va_mask = line_filter(val_X['line'].values)
        te_mask = line_filter(test_X['line'].values)

        if tr_mask.sum() < 500 or te_mask.sum() < 100:
            print(f"  {name}: insufficient data (train={tr_mask.sum()}, test={te_mask.sum()})")
            continue

        spec_model = GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=30, random_state=42
        )
        spec_model.fit(safe_fillna(train_X[base_features].iloc[np.where(tr_mask)]),
                       train_target[tr_mask])

        spec_probs = spec_model.predict_proba(safe_fillna(test_X[base_features].iloc[np.where(te_mask)]))[:, 1]
        spec_auc = roc_auc_score(test_target[te_mask], spec_probs)
        specialist_probs[te_mask] = spec_probs

        # Compare to global on same subset
        global_sub_auc = roc_auc_score(test_target[te_mask], global_probs[te_mask])

        print(f"  {name:12s}: n_test={te_mask.sum():5d}, specialist_AUC={spec_auc:.4f}, global_AUC={global_sub_auc:.4f}, delta={spec_auc-global_sub_auc:+.4f}")
        approach_E2_results[name] = {
            'specialist_auc': float(spec_auc),
            'global_auc_same_subset': float(global_sub_auc),
            'delta': float(spec_auc - global_sub_auc),
            'n_test': int(te_mask.sum()),
        }

    # E2c: Per-prop-type specialists
    print("\n--- E2c: Per-prop-type specialist models ---")
    for prop in PROP_TYPES:
        tr_mask = train_meta['prop_type'].values == prop
        te_mask = test_meta['prop_type'].values == prop

        if tr_mask.sum() < 500 or te_mask.sum() < 100:
            continue

        spec_model = GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=30, random_state=42
        )
        spec_model.fit(safe_fillna(train_X[base_features].iloc[np.where(tr_mask)]),
                       train_target[tr_mask])

        spec_probs = spec_model.predict_proba(safe_fillna(test_X[base_features].iloc[np.where(te_mask)]))[:, 1]
        spec_auc = roc_auc_score(test_target[te_mask], spec_probs)
        global_sub_auc = roc_auc_score(test_target[te_mask], global_probs[te_mask])

        # Betting ROI at threshold
        for thresh in [0.57]:
            over_bets = spec_probs > thresh
            under_bets = spec_probs < (1 - thresh)
            te_tar = test_target[te_mask]
            te_over_dec = over_dec[te_mask]
            te_under_dec = under_dec[te_mask]

            if over_bets.sum() > 10:
                o_hits = te_tar[over_bets]
                o_roi = (np.sum(te_over_dec[over_bets][o_hits.astype(bool)]) - over_bets.sum()) / over_bets.sum()
            else:
                o_roi = float('nan')

            if under_bets.sum() > 10:
                u_hits = (~te_tar.astype(bool))[under_bets]
                u_roi = (np.sum(te_under_dec[under_bets][u_hits]) - under_bets.sum()) / under_bets.sum()
            else:
                u_roi = float('nan')

        print(f"  {prop:15s}: spec_AUC={spec_auc:.4f}, glob_AUC={global_sub_auc:.4f}, delta={spec_auc-global_sub_auc:+.4f}, over_roi={o_roi:+.1%}, under_roi={u_roi:+.1%}")
        approach_E2_results[f'prop_{prop}'] = {
            'specialist_auc': float(spec_auc),
            'global_auc_same_subset': float(global_sub_auc),
            'delta': float(spec_auc - global_sub_auc),
            'n_test': int(te_mask.sum()),
        }

    # E2d: Star vs role player
    print("\n--- E2d: Star vs Role Player specialists ---")
    season_avg = test_X['target_season'].values
    for label, mask_fn in [('star_75pct', lambda sa: sa > np.nanpercentile(sa, 75)),
                            ('role_50pct', lambda sa: sa < np.nanpercentile(sa, 50))]:
        te_mask = mask_fn(season_avg)
        tr_mask = mask_fn(train_X['target_season'].values)

        if tr_mask.sum() < 500 or te_mask.sum() < 100:
            continue

        spec_model = GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=30, random_state=42
        )
        spec_model.fit(safe_fillna(train_X[base_features].iloc[np.where(tr_mask)]),
                       train_target[tr_mask])

        spec_probs = spec_model.predict_proba(safe_fillna(test_X[base_features].iloc[np.where(te_mask)]))[:, 1]
        spec_auc = roc_auc_score(test_target[te_mask], spec_probs)
        global_sub_auc = roc_auc_score(test_target[te_mask], global_probs[te_mask])

        print(f"  {label:15s}: n={te_mask.sum():5d}, spec_AUC={spec_auc:.4f}, glob_AUC={global_sub_auc:.4f}, delta={spec_auc-global_sub_auc:+.4f}")
        approach_E2_results[label] = {
            'specialist_auc': float(spec_auc),
            'global_auc_same_subset': float(global_sub_auc),
            'delta': float(spec_auc - global_sub_auc),
            'n_test': int(te_mask.sum()),
        }

except Exception as e:
    print(f"E2 ERROR: {e}")
    import traceback
    traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────────────
# APPROACH P: Per-100-Possessions Normalization
# ──────────────────────────────────────────────────────────────────────────────

print("\n\n" + "=" * 80)
print("APPROACH P: Per-100-Possessions Normalization")
print("=" * 80)

approach_P_results = {}

# Check if we have pace data
pace_cols = [c for c in X.columns if 'pace' in c.lower()]
print(f"Pace columns: {pace_cols}")

try:
    # Create per-100 features
    # per_100 = (stat / minutes) * (100 / pace * 48) = per_minute_rate * 4800 / pace
    # We have per_minute_rate and e_pace

    if 'e_pace' in X.columns:
        pace_col = 'e_pace'
    elif 'team_pace' in X.columns:
        pace_col = 'team_pace'
    else:
        pace_col = pace_cols[0] if pace_cols else None

    if pace_col:
        print(f"Using pace column: {pace_col}")
        print(f"Pace stats: mean={X[pace_col].mean():.1f}, std={X[pace_col].std():.1f}")

        # Create per-100 features for all splits
        for split_name, split_X, split_meta, split_target in [
            ('train', train_X, train_meta, train_target if 'train_target' in dir() else None),
            ('val', val_X, val_meta, None),
            ('test', test_X, test_meta, None),
        ]:
            pace = split_X[pace_col].values
            # Replace 0 or NaN pace with league average
            league_avg_pace = 100.0
            pace = np.where((pace == 0) | np.isnan(pace), league_avg_pace, pace)

            pmr = split_X['per_minute_rate'].values
            pmr_l5 = split_X['per_minute_rate_l5'].values

            per100 = pmr * 4800.0 / pace
            per100_l5 = pmr_l5 * 4800.0 / pace

            if split_name == 'test':
                test_per100 = per100
                test_per100_l5 = per100_l5

        # P1: Does per-100 normalization improve projection vs line?
        print("\n--- P1: Per-100 vs raw per-minute rate ---")

        # Prediction: if per-100 rate * expected minutes / 48 * pace / 100 > line -> over
        # Simplified: does per-100 rate predict over/under better than raw rate?
        test_pace = test_X[pace_col].values
        test_pace = np.where((test_pace == 0) | np.isnan(test_pace), 100.0, test_pace)

        # Project using per-100 rate: stat_projected = per100_rate * (mins/48) * (opp_pace/100)
        # Since we don't have opp_pace, use: per100_rate * expected_mins / 48 * avg_pace / 100
        test_mins = test_X['mins_l5'].values
        projected_per100 = test_per100 * test_mins / 48.0 * test_pace / 100.0
        projected_raw = test_X['expected_from_rate_l5'].values

        # How well does each projection predict over/under?
        actual = test_meta['actual'].values
        line = test_meta['line'].values

        for proj_name, proj in [('raw_rate_proj', projected_raw), ('per100_proj', projected_per100)]:
            over_pred = proj > line
            actual_over = actual > line

            if np.isnan(proj).all():
                continue

            valid = ~np.isnan(proj)
            if valid.sum() < 100:
                continue

            accuracy = np.mean(over_pred[valid] == actual_over[valid])
            # Correlation between projection error and actual
            proj_error = proj[valid] - line[valid]
            actual_error = actual[valid] - line[valid]
            corr = np.corrcoef(proj_error, actual_error)[0, 1]
            print(f"  {proj_name:15s}: accuracy={accuracy:.1%}, corr={corr:.4f}")

        # P2: Build GBT with per-100 features
        print("\n--- P2: GBT with per-100 features vs without ---")

        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import roc_auc_score

        train_target_p = (train_meta['actual'].values > train_meta['line'].values).astype(int)
        test_target_p = (test_meta['actual'].values > test_meta['line'].values).astype(int)

        # Create per-100 features for train and test
        train_pace = train_X[pace_col].values
        train_pace = np.where((train_pace == 0) | np.isnan(train_pace), 100.0, train_pace)
        train_per100 = train_X['per_minute_rate'].values * 4800.0 / train_pace
        train_per100_l5 = train_X['per_minute_rate_l5'].values * 4800.0 / train_pace

        # Add per-100 features
        per100_features = base_features + ['per100_rate', 'per100_rate_l5']

        tr_with_per100 = safe_fillna(train_X[base_features].copy())
        tr_with_per100['per100_rate'] = np.nan_to_num(train_per100, 0)
        tr_with_per100['per100_rate_l5'] = np.nan_to_num(train_per100_l5, 0)

        te_with_per100 = safe_fillna(test_X[base_features].copy())
        te_with_per100['per100_rate'] = np.nan_to_num(test_per100, 0)
        te_with_per100['per100_rate_l5'] = np.nan_to_num(test_per100_l5, 0)

        # Model without per-100
        model_no_p100 = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=50, random_state=42
        )
        model_no_p100.fit(safe_fillna(train_X[base_features]), train_target_p)
        probs_no = model_no_p100.predict_proba(safe_fillna(test_X[base_features]))[:, 1]
        auc_no = roc_auc_score(test_target_p, probs_no)

        # Model with per-100
        model_with_p100 = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=50, random_state=42
        )
        model_with_p100.fit(tr_with_per100, train_target_p)
        probs_with = model_with_p100.predict_proba(te_with_per100)[:, 1]
        auc_with = roc_auc_score(test_target_p, probs_with)

        print(f"  Without per-100: AUC={auc_no:.4f}")
        print(f"  With per-100:    AUC={auc_with:.4f}")
        print(f"  Delta:           {auc_with-auc_no:+.4f}")

        approach_P_results['gbt_comparison'] = {
            'auc_without_per100': float(auc_no),
            'auc_with_per100': float(auc_with),
            'delta': float(auc_with - auc_no),
        }

        # P3: Per-100 normalization by pace extremes
        print("\n--- P3: Performance at pace extremes ---")
        for pace_label, pace_mask_fn in [
            ('slow (< 97)', lambda p: p < 97),
            ('avg (97-103)', lambda p: (p >= 97) & (p <= 103)),
            ('fast (> 103)', lambda p: p > 103),
        ]:
            pace_mask = pace_mask_fn(test_pace)
            if pace_mask.sum() < 50:
                continue

            sub_probs_no = probs_no[pace_mask]
            sub_probs_with = probs_with[pace_mask]
            sub_target = test_target_p[pace_mask]

            auc_sub_no = roc_auc_score(sub_target, sub_probs_no) if sub_target.sum() > 5 else float('nan')
            auc_sub_with = roc_auc_score(sub_target, sub_probs_with) if sub_target.sum() > 5 else float('nan')

            print(f"  Pace {pace_label}: n={pace_mask.sum():5d}, AUC_no={auc_sub_no:.4f}, AUC_with={auc_sub_with:.4f}, delta={auc_sub_with-auc_sub_no:+.4f}")

        # P4: Feature importance for per-100 model
        print("\n--- P4: Top feature importances (with per-100) ---")
        feature_imp = pd.Series(model_with_p100.feature_importances_,
                                index=tr_with_per100.columns).sort_values(ascending=False)
        for feat, imp in feature_imp.head(15).items():
            print(f"  {feat:30s}: {imp:.4f}")

        # Check where per-100 features rank
        if 'per100_rate' in feature_imp.index:
            print(f"\n  per100_rate rank: {list(feature_imp.index).index('per100_rate')+1}")
        if 'per100_rate_l5' in feature_imp.index:
            print(f"  per100_rate_l5 rank: {list(feature_imp.index).index('per100_rate_l5')+1}")

    else:
        print("No pace column found, skipping per-100 analysis")

except Exception as e:
    print(f"P ERROR: {e}")
    import traceback
    traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────────────
# CROSS-APPROACH ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

print("\n\n" + "=" * 80)
print("CROSS-APPROACH: Combined signals")
print("=" * 80)

# Best combo: regression signal + juice fade + specialist model confidence
try:
    combined_signals = pd.DataFrame({
        'trend': trend,
        'juice_gap': juice_gap,
        'model_prob': global_probs if 'global_probs' in dir() else np.full(len(test_X), 0.5),
        'hit_rate_l5': hit_rate,
        'cv': cv,
        'actual': test_actual,
        'line': test_line,
        'over_dec': over_dec,
        'under_dec': under_dec,
        'prop_type': test_meta['prop_type'].values,
    })

    # Signal agreement: multiple signals all say UNDER
    under_signals = (
        (combined_signals['trend'] > 2).astype(int) +
        (combined_signals['juice_gap'] > 20).astype(int) +
        (combined_signals['model_prob'] < 0.43).astype(int) +
        (combined_signals['hit_rate_l5'] > 0.8).astype(int)
    )

    over_signals = (
        (combined_signals['trend'] < -2).astype(int) +
        (combined_signals['juice_gap'] < -20).astype(int) +
        (combined_signals['model_prob'] > 0.57).astype(int) +
        (combined_signals['hit_rate_l5'] < 0.2).astype(int)
    )

    print("\n--- Signal agreement: UNDER ---")
    for n_agree in [2, 3, 4]:
        mask = under_signals >= n_agree
        if mask.sum() > 20:
            h = combined_signals.loc[mask, 'actual'] < combined_signals.loc[mask, 'line']
            dec = combined_signals.loc[mask, 'under_dec']
            roi = (dec[h].sum() - mask.sum()) / mask.sum()
            print(f"  {n_agree}+ signals agree UNDER: n={mask.sum():4d}, hit={h.mean():.1%}, ROI={roi:+.1%}")

    print("\n--- Signal agreement: OVER ---")
    for n_agree in [2, 3, 4]:
        mask = over_signals >= n_agree
        if mask.sum() > 20:
            h = combined_signals.loc[mask, 'actual'] > combined_signals.loc[mask, 'line']
            dec = combined_signals.loc[mask, 'over_dec']
            roi = (dec[h].sum() - mask.sum()) / mask.sum()
            print(f"  {n_agree}+ signals agree OVER: n={mask.sum():4d}, hit={h.mean():.1%}, ROI={roi:+.1%}")

    # Best combined strategy
    print("\n--- Best combined: strongest signal agreement by prop ---")
    for prop in PROP_TYPES:
        prop_mask = combined_signals['prop_type'] == prop
        for n_agree in [2, 3]:
            under_mask = prop_mask & (under_signals >= n_agree)
            over_mask = prop_mask & (over_signals >= n_agree)

            if under_mask.sum() > 10:
                h = combined_signals.loc[under_mask, 'actual'] < combined_signals.loc[under_mask, 'line']
                dec = combined_signals.loc[under_mask, 'under_dec']
                roi = (dec[h].sum() - under_mask.sum()) / under_mask.sum()
                print(f"  {prop:15s} {n_agree}+ UNDER: n={under_mask.sum():3d}, hit={h.mean():.1%}, ROI={roi:+.1%}")

            if over_mask.sum() > 10:
                h = combined_signals.loc[over_mask, 'actual'] > combined_signals.loc[over_mask, 'line']
                dec = combined_signals.loc[over_mask, 'over_dec']
                roi = (dec[h].sum() - over_mask.sum()) / over_mask.sum()
                print(f"  {prop:15s} {n_agree}+ OVER:  n={over_mask.sum():3d}, hit={h.mean():.1%}, ROI={roi:+.1%}")

except Exception as e:
    print(f"Cross-approach ERROR: {e}")
    import traceback
    traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────────────
# SEASONALITY ANALYSIS (Does signal improve later in season?)
# ──────────────────────────────────────────────────────────────────────────────

print("\n\n" + "=" * 80)
print("SEASONALITY: Does signal improve later in season?")
print("=" * 80)

test_dates = pd.to_datetime(test_meta['date'].values)
test_months = test_dates.month

# Games played as proxy for season progress
gp = test_X['games_played_capped'].values

for gp_label, gp_range in [('early (gp<30)', (0, 30)), ('mid (30-50)', (30, 50)), ('late (50+)', (50, 999))]:
    gp_mask = (gp >= gp_range[0]) & (gp < gp_range[1])
    if gp_mask.sum() < 100:
        continue

    # Regression signal
    hot = gp_mask & (trend > 2)
    if hot.sum() > 20:
        h_under = test_actual[hot] < test_line[hot]
        h_roi = (np.sum(under_dec[hot][h_under]) - hot.sum()) / hot.sum()
        print(f"  {gp_label:20s} hot->under: n={hot.sum():4d}, hit={h_under.mean():.1%}, ROI={h_roi:+.1%}")

    # Juice fade
    fade = gp_mask & (juice_gap > 20)
    if fade.sum() > 20:
        f_under = test_actual[fade] < test_line[fade]
        f_roi = (np.sum(under_dec[fade][f_under]) - fade.sum()) / fade.sum()
        print(f"  {gp_label:20s} fade->under: n={fade.sum():4d}, hit={f_under.mean():.1%}, ROI={f_roi:+.1%}")


# ──────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS
# ──────────────────────────────────────────────────────────────────────────────

print("\n\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

results = {
    'metadata': {
        'total_rows': int(n),
        'test_rows': int(len(test_X)),
        'test_date_range': f"{test_meta['date'].iloc[0]} to {test_meta['date'].iloc[-1]}",
        'split': '70/15/15 walk-forward by date',
    },
    'baseline': baseline_results,
    'approach_L_regression': approach_L_results,
    'approach_N_fade_public': approach_N_results,
    'approach_O_injury_cascade': approach_O_results,
    'approach_E2_specialist': approach_E2_results,
    'approach_P_per100': approach_P_results,
}

with open('outputs/model_experiments_L_P.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"Results saved to outputs/model_experiments_L_P.json")
print("\nDONE.")
