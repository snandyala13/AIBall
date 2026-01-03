# aiBall - NBA Prop Betting Model

A statistical NBA player prop betting model with **ML ensemble filtering** that generates projections and identifies value opportunities. Backtested against real DraftKings/FanDuel lines.

## Performance

### Full Backtest Results (34,697 predictions)

Tested across **1,727 NBA games** from October 2024 - January 2026.

| ML Threshold | Picks | Hit Rate | Edge vs 50% | Simulated ROI |
|--------------|-------|----------|-------------|---------------|
| Baseline (all) | 34,697 | 52.5% | +2.5% | ~0% |
| **ML >= 50%** | 23,709 | **55.2%** | +5.2% | **+3.5%** |
| **ML >= 55%** | 6,613 | **59.8%** | +9.8% | **+7.6%** |
| **ML >= 58%** | 1,305 | **63.5%** | +13.5% | **+10.1%** |
| **ML >= 60%** | 402 | **66.9%** | +16.9% | **+13.4%** |
| ML >= 62% | 89 | 78.7% | +28.7% | ~25% |
| ML >= 65% | 27 | 81.5% | +31.5% | ~28% |

*Tested against real historical lines from The Odds API (DraftKings/FanDuel), not simulated data.*

### Simulated Betting Performance

$100 flat bets at ML >= 55% threshold:
- **6,613 picks** over 14 months
- **$661,300 wagered**
- **+$50,515 profit**
- **+7.6% ROI**

### Performance by Prop Type (ML >= 55%)

| Prop Type | Picks | Hit Rate |
|-----------|-------|----------|
| **Threes** | 851 | **63.7%** |
| **Rebounds** | 1,467 | **60.7%** |
| Points | 1,617 | 59.1% |
| Assists | 1,095 | 58.5% |
| PRA | 1,583 | 58.4% |

### Monthly Trend (ML >= 55%)

| Period | Hit Rate | Notes |
|--------|----------|-------|
| Oct-Dec 2024 | 57.5% | Early season |
| Jan-Apr 2025 | 58.0% | Regular season |
| Oct-Nov 2025 | 62.2% | Current season |
| **Dec 2025** | **74.3%** | Model improvements |

*See [ML_PERFORMANCE_REPORT.md](ML_PERFORMANCE_REPORT.md) for detailed analysis.*

## How It Works

The system runs through 4 stages: Data Collection → Projection → Edge Calculation → ML Filtering.

### 1. Data Collection (`data_pipeline.py`)

Fetches from Balldontlie API:
- **Player Stats**: Last 20 games, season averages, per-minute rates
- **Usage Rates**: Shot attempts, touches, assists opportunities
- **Injuries**: Current injuries for all teams (used for pop-off detection)
- **Live Lines**: DraftKings player props via `/v2/odds/player_props`

Builds two key objects:
- `PlayerProfile`: Rolling averages (L5/L10/season), std devs, home/away splits
- `GameContext`: Opponent pace, defensive ratings, spread, back-to-back status

### 2. Base Projection Model (`model_engine.py`)

**Weighted Average Projection:**
```
Base = (L5_avg × 0.45) + (L10_avg × 0.35) + (Season_avg × 0.20)
```

**Per-Minute Normalization:**
```
Projected = (per_minute_rate × projected_minutes) × adjustments
```

**Adjustment Layers:**
| Factor | Impact | Source |
|--------|--------|--------|
| Back-to-back | -8% stats | Schedule API |
| Blowout risk | -15% minutes | Point spread |
| Defense tier | -13% to +12% | Opponent DRtg |
| Pace | Up to ±5% | Game pace vs league avg |
| Home court | ~3% boost | Home/away split |
| Injury boost | Up to +15% | Teammate injuries (time-decayed) |

### 3. Edge Calculation (`model_engine.py`)

Uses Z-scores to calculate probability:
```python
z_score = (projection - line) / std_dev
prob_over = 1 - norm.cdf(z_score)
prob_under = norm.cdf(z_score)

edge = our_probability - implied_probability
```

Key calibration: `STD_DEV_CALIBRATION = 2.0` widens distributions to reduce overconfidence.

### 4. ML Ensemble Filter (`ml_filter.py`)

Takes the base model's output and predicts whether the pick will actually hit.

**Input Features (11 total):**
```python
features = {
    'probability': 0.62,      # Base model probability
    'edge': 8.5,              # Edge percentage
    'confidence': 0.85,       # Minutes-based confidence
    'diff_pct': 15.0,         # % difference from line
    'line': 25.5,             # Betting line
    'is_over': 1,             # OVER=1, UNDER=0
    'prop_points': 1,         # One-hot encoded prop type
    'edge_squared': 72.25,    # Penalizes extreme edges
    'prob_x_edge': 5.27,      # Interaction term
    'high_edge': 0,           # Flag for edge >= 15%
    'mid_line': 1             # Flag for lines 15-35
}
```

**Ensemble Combination:**
```python
gb_prob = gradient_boosting.predict_proba(X)    # Best at filtering
rf_prob = random_forest.predict_proba(X)        # Robust patterns
lr_prob = logistic_regression.predict_proba(X)  # Baseline

ml_probability = (gb_prob + rf_prob + lr_prob) / 3
```

**What the ML Learned:**
- Base probability is the strongest signal (29% importance)
- Difference from line matters more than raw edge (22% importance)
- Probability × Edge interaction captures quality picks (19% importance)
- Threes actually perform best at 63.7% when ML-filtered
- UNDERs and OVERs perform equally well (~60%) when filtered
- Higher edge tiers (15%+) hit at 63-68% when combined with ML filter

### 5. Pick Selection (`bet_generator.py`)

Final filtering based on ML probability:

| Tier | ML Threshold | Expected Hit Rate | Daily Volume | Action |
|------|--------------|-------------------|--------------|--------|
| ELITE | >= 60% | ~67% | ~1/day | Include in all parlays |
| HIGH | >= 58% | ~64% | ~3/day | Include in recommendations |
| STANDARD | >= 55% | ~60% | ~15/day | Daily betting volume |
| FILTERED | < 55% | ~53% | — | Filter out |

**Parlay Construction:**
- HIGH-CONFIDENCE: 2-leg parlays using ELITE/HIGH picks (ML >= 58%)
- Expected parlay hit rate: 0.64 × 0.64 = 41% (profitable at +300 odds, breakeven ~25%)

## Installation

1. Clone and setup:
```bash
git clone <repo-url>
cd aiBall
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Set up API key:
```bash
export BALLDONTLIE_API_KEY='your-api-key-here'
```

Get an API key from [Balldontlie](https://balldontlie.io).

## Usage

### Generate Today's Picks

```bash
python ball.py --today
```

Output includes:
- **HIGH-CONFIDENCE PARLAYS**: 2-leg parlays with ML-filtered legs (65%+ expected parlay hit rate)
- **Top Singles**: Highest edge single bets
- **Safe/Standard/Longshot Parlays**: Various risk levels

### Save Picks to File

```bash
python ball.py --today > outputs/today_picks.txt
```

### Backtest with Real Lines

```bash
# Run full backtest
python backtest_real_lines.py

# View results
cat outputs/backtest_real_lines.txt
```

### Fetch Historical Odds

```bash
# Check database stats
python fetch_historical_odds.py --stats

# Fetch more data (costs API credits)
python fetch_historical_odds.py --api-key YOUR_KEY --days 30
```

## Project Structure

```
aiBall/
├── ball.py                    # Main entry point
├── data_pipeline.py           # API data fetching and player/team profiles
├── model_engine.py            # Statistical projections and adjustments
├── bet_generator.py           # Pick generation with ML filtering
├── ml_filter.py               # ML ensemble filter module
├── train_ml_model.py          # ML ensemble training with time-based CV
├── analyze_yesterday.py       # Analyze previous day's performance
├── backtest_real_lines.py     # Backtest against real historical lines
├── cache_db.py                # SQLite caching and historical odds storage
├── fetch_historical_odds.py   # Fetch real lines from The Odds API
├── ML_PERFORMANCE_REPORT.md   # Detailed ML filter analysis
├── models/                    # Trained ML models
│   ├── gradient_boosting.joblib
│   ├── random_forest.joblib
│   ├── logistic_regression.joblib
│   ├── scaler.joblib
│   └── ensemble_metadata.json
└── outputs/                   # Generated reports and analysis
```

## Configuration

### Key Parameters
```python
# model_engine.py
EdgeCalculator.STD_DEV_CALIBRATION = 2.0  # Widens probability distribution
PopOffHunter.POPOFF_THRESHOLD = 0.15      # Max 15% injury boost
# Time-decay: Full boost within 3 games, 0% after 21+ games out

# bet_generator.py / ml_filter.py
ML_STANDARD_THRESHOLD = 0.55  # 60% expected hit rate, ~15 picks/day
ML_HIGH_THRESHOLD = 0.58      # 64% expected hit rate, ~3 picks/day
ML_ELITE_THRESHOLD = 0.60     # 67% expected hit rate, ~1 pick/day
```

### Retraining the ML Model

After running a new backtest with more data:

```bash
# 1. Run backtest to generate predictions
python backtest_real_lines.py

# 2. Retrain ensemble models (uses latest backtest_predictions file)
python train_ml_model.py

# 3. Or specify a specific predictions file
python train_ml_model.py --predictions outputs/backtest_predictions_20260103.json
```

## Data Sources

- **Player/Game Data**: [Balldontlie API](https://balldontlie.io)
- **Historical Odds**: [The Odds API](https://the-odds-api.com)
- **Live Odds**: DraftKings, FanDuel

## Current Database Stats

- **Games Backtested**: 1,727
- **Predictions Analyzed**: 34,697
- **Props in Cache**: 673,219+
- **Date Range**: Oct 22, 2024 - Jan 2, 2026
- **ML Training Samples**: 34,697 (time-split validation)

## Disclaimer

This tool is for educational and research purposes only. Sports betting involves risk and past performance does not guarantee future results. Always gamble responsibly.

## License

MIT
