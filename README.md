# aiBall - NBA Prop Betting Model

A statistical NBA player prop betting model that generates projections and identifies value opportunities. Backtested against real DraftKings/FanDuel lines with **62.1% hit rate** on individual legs.

## Performance (Dec 25-31, 2025 Backtest)

| Metric | Result |
|--------|--------|
| **Overall Hit Rate** | 62.1% (628/1011) |
| **Breakeven Threshold** | 52.4% |
| **Points Props** | 66.0% |
| **PRA (Combined)** | 68.7% |
| **Rebounds** | 60.5% |
| **Assists** | 58.6% |
| **Threes** | 57.1% |
| **UNDER Picks** | 71.1% |
| **2-Leg Parlays** | 30.2% |

*Tested against real historical lines from The Odds API, not simulated data.*

## Features

### Core Model
- **Weighted Projections**: Per-minute rates (60%) blended with weighted averages (40%)
- **Anti-Overestimation**: 15% regression toward baseline to prevent runaway projections
- **Calibrated Probabilities**: Standard deviation calibration (2.0x) with aggressive damping

### Game Context Adjustments
- Opponent defensive rating by position
- Pace factor based on team matchup
- Back-to-back fatigue (-7%)
- Rest day bonuses (3+ days: +1-2%)
- Blowout risk (spread-based minutes adjustment)
- Home/away splits

### Injury-Based Pop-Off Detection
- Identifies beneficiaries when teammates are injured
- Position-based usage redistribution
- Scarcity multipliers (fewer healthy players at position = higher boost)
- **Capped at 15% boost** - data showed higher boosts hurt performance

### Edge Calculation
- Normal distribution probability model
- Compares model probability vs implied odds
- **Edge capped at 12%** - high edge picks historically underperformed
- Minimum edge threshold of 5% for picks

## Installation

1. Clone the repository:
```bash
git clone <repo-url>
cd aiBall
```

2. Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your API keys:
```bash
export BALLDONTLIE_API_KEY='your-api-key-here'
```

Get an API key from [Balldontlie](https://balldontlie.io).

## Usage

### Generate Picks for Today's Games

```bash
python ball.py
```

This will:
- Fetch today's NBA games
- Build player profiles and projections
- Identify injury-based opportunities
- Generate single picks and parlays
- Display results with edge calculations

### Backtest with Real Historical Lines

```bash
# Run backtest against real DraftKings lines
python backtest_real_lines.py

# Analyze results
cat outputs/backtest_real_lines.txt
```

### Fetch Historical Odds (requires The Odds API key)

```bash
# Check current database stats
python fetch_historical_odds.py --stats

# Estimate API credit costs
python fetch_historical_odds.py --estimate --days 30

# Fetch historical data (costs credits)
python fetch_historical_odds.py --api-key YOUR_KEY --days 30
```

## Project Structure

```
aiBall/
├── ball.py                  # Main entry point
├── data_pipeline.py         # API data fetching and player/team profiles
├── model_engine.py          # Statistical projections and adjustments
├── bet_generator.py         # Pick generation and parlay building
├── cache_db.py              # SQLite caching and historical odds storage
├── fetch_historical_odds.py # Fetch real lines from The Odds API
├── backtest_real_lines.py   # Backtest against real historical lines
├── backtest_full_model.py   # Full model backtest
└── outputs/                 # Generated reports and analysis
```

## Model Architecture

### 1. Data Pipeline (`data_pipeline.py`)
- Fetches player game logs, team stats, injuries
- Builds `PlayerProfile` with per-minute rates, splits, variance
- Builds `GameContext` with matchup data

### 2. Projection Engine (`model_engine.py`)
- `ProjectionEngine`: Base stat projections with adjustments
- `PopOffHunter`: Identifies injury-based opportunities
- `EdgeCalculator`: Probability calculations with calibration
- `CorrelationEngine`: Player stat correlations for parlays

### 3. Bet Generator (`bet_generator.py`)
- Filters projections by edge threshold
- Ranks picks by quality score
- Builds correlation-aware parlays

## Key Learnings from Backtesting

1. **Lower edge = better performance**: 5-10% edge picks hit at 62.3%, while 20%+ edge hit only 36.5% before fixes
2. **Usage boost was broken**: Original model added up to 16 points for injury opportunities; capping at 15% multiplier improved hit rate from 51.6% to 61.3%
3. **PRA props are most reliable**: Combined stats have natural variance smoothing
4. **UNDERs outperform**: Model systematically overestimated, making UNDERs more accurate
5. **Market is efficient**: The Odds API lines are well-calibrated; edge comes from specific situations, not broad mispricing

## Configuration

Key parameters in `model_engine.py`:

```python
# Edge calculation
EdgeCalculator.STD_DEV_CALIBRATION = 2.0  # Widens distribution
EdgeCalculator.MIN_STD_DEV = {...}        # Floors by stat type

# Pop-off detection
PopOffHunter.POPOFF_THRESHOLD = 0.15      # 15% max boost

# Projection
regression_factor = 0.15                   # Blend toward baseline
```

## Data Sources

- **Player/Game Data**: [Balldontlie API](https://balldontlie.io)
- **Historical Odds**: [The Odds API](https://the-odds-api.com) (for backtesting)
- **Live Odds**: DraftKings, FanDuel (via The Odds API)

## Disclaimer

This tool is for educational and research purposes only. Sports betting involves risk and past performance does not guarantee future results. Always gamble responsibly.

## License

MIT
