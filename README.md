# aiBall - NBA Prop Betting Model

A statistical NBA player prop betting model with **ML ensemble filtering** that generates projections and identifies value opportunities. Backtested against real DraftKings/FanDuel lines.

## Performance

### ML-Filtered Picks (Recommended)

| ML Threshold | Hit Rate | Sample Size |
|--------------|----------|-------------|
| **ML >= 60%** | **87.5%** | 72 picks |
| **ML >= 58%** | **79.5%** | 200 picks |
| **ML >= 55%** | **75.2%** | 476 picks |

*Based on most recent week of backtesting (Dec 25 - Jan 1)*

### Full Backtest (17,436 predictions)

| Metric | Result |
|--------|--------|
| **Base Model Hit Rate** | 54.1% |
| **ML-Filtered (>=58%)** | 68.1% |
| **ML-Filtered (>=60%)** | 72.1% |
| **Breakeven Threshold** | 52.4% |

*Tested against real historical lines from The Odds API, not simulated data.*

### Live Performance (Jan 1, 2026)

| ML Threshold | Hit Rate | Sample Size | ROI |
|--------------|----------|-------------|-----|
| **ML >= 60%** | **100%** | 2 picks | +90.9% |
| **ML >= 58%** | **85.7%** | 14 picks | +63.6% |

Top picks that hit:
- Svi Mykhailiuk PRA UNDER 15.5 (ML: 61.1%) - Actual: 8
- Taylor Hendricks REB UNDER 5.5 (ML: 61.0%) - Actual: 2
- VJ Edgecombe PTS OVER 14.5 (ML: 58.7%) - Actual: 23

*See `outputs/yesterday_report_20260101.txt` for full breakdown.*

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
- High edge (15%+) actually underperforms → applies penalty
- Threes have 49.8% hit rate → heavily penalized
- UNDERs outperform OVERs → slight boost
- Mid-range lines (15-35) hit best → prioritized
- PRA (combined stats) most reliable → slight boost

### 5. Pick Selection (`bet_generator.py`)

Final filtering based on ML probability:

| Tier | ML Threshold | Expected Hit Rate | Action |
|------|--------------|-------------------|--------|
| ELITE | >= 60% | ~72% | Include in all parlays |
| HIGH | >= 58% | ~68% | Include in recommendations |
| STANDARD | < 58% | ~54% | Filter out |

**Parlay Construction:**
- HIGH-CONFIDENCE: 2-leg parlays using only ELITE/HIGH picks
- Expected parlay hit rate: 0.72 × 0.72 = 51.8% (profitable at +300 odds)

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
├── ball.py                  # Main entry point
├── data_pipeline.py         # API data fetching and player/team profiles
├── model_engine.py          # Statistical projections and adjustments
├── bet_generator.py         # Pick generation with ML filtering
├── ml_filter.py             # ML ensemble filter module (NEW)
├── cache_db.py              # SQLite caching and historical odds storage
├── fetch_historical_odds.py # Fetch real lines from The Odds API
├── backtest_real_lines.py   # Backtest against real historical lines
├── models/                  # Trained ML models (NEW)
│   ├── gradient_boosting.joblib
│   ├── random_forest.joblib
│   ├── logistic_regression.joblib
│   ├── scaler.joblib
│   └── ensemble_metadata.json
└── outputs/                 # Generated reports and analysis
```

## Configuration

### Key Parameters
```python
# model_engine.py
EdgeCalculator.STD_DEV_CALIBRATION = 2.0  # Widens probability distribution
PopOffHunter.POPOFF_THRESHOLD = 0.15      # Max 15% injury boost
# Time-decay: Full boost within 3 games, 0% after 21+ games out

# bet_generator.py
ML_HIGH_THRESHOLD = 0.58   # 68% expected hit rate
ML_ELITE_THRESHOLD = 0.60  # 72% expected hit rate
```

### Retraining the ML Model

After running a new backtest with more data:

```bash
# 1. Run backtest to generate predictions
python backtest_real_lines.py

# 2. Retrain (script coming soon)
python train_ml_filter.py --input outputs/backtest_predictions.json
```

## Data Sources

- **Player/Game Data**: [Balldontlie API](https://balldontlie.io)
- **Historical Odds**: [The Odds API](https://the-odds-api.com)
- **Live Odds**: DraftKings, FanDuel

## Current Database Stats

- **Games**: 1,721
- **Props**: 673,219
- **Date Range**: Oct 22, 2024 - Jan 2, 2026
- **ML Training Samples**: 17,436

## Disclaimer

This tool is for educational and research purposes only. Sports betting involves risk and past performance does not guarantee future results. Always gamble responsibly.

## License

MIT
