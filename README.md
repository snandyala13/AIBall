# aiBall - NBA Prop Betting Model

A statistical NBA player prop betting model that generates projections and identifies value opportunities using the Balldontlie API.

## Features

- **Weighted Projections**: Combines last 5 games (45%), last 10 games (35%), and season averages (20%)
- **Game Context Adjustments**:
  - Opponent defensive rating (fetched from API)
  - Pace/game total adjustments
  - Back-to-back fatigue (-7%)
  - Rest day bonuses (3 days: +1%, 4+ days: +2%)
  - Blowout risk (spread-based)
- **Smart Filtering**:
  - Player tier classification (Star/Rotation/Bench)
  - Sample size penalties for low-data players
  - Minimum line thresholds to avoid high-variance props
- **Parlay Generation**: Builds 2-leg, 3-4 leg, and 5-6 leg parlays with quality scoring
- **Backtesting**: Test model performance against historical games

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

4. Set up your API key:
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
- Generate single picks and parlays
- Display results with confidence levels and reasoning

### Backtest the Model

```bash
# Backtest across multiple days
python backtest.py

# Backtest a specific game
python backtest_game.py

# Backtest parlay performance
python backtest_parlays.py
```

## Project Structure

```
aiBall/
├── ball.py              # Main entry point
├── data_pipeline.py     # API data fetching and player/team profiles
├── model_engine.py      # Statistical projections and adjustments
├── bet_generator.py     # Pick generation and parlay building
├── adjustments.py       # Game context adjustment calculations
├── backtest.py          # Multi-day backtesting
├── backtest_game.py     # Single game backtesting
├── backtest_parlays.py  # Parlay backtesting
└── requirements.txt     # Python dependencies
```

## Model Methodology

### Projection Calculation

1. **Base Projection**: Weighted average of recent performance
   - Last 5 games: 45% weight
   - Last 10 games: 35% weight
   - Season average: 20% weight

2. **Adjustments Applied**:
   - **Defense**: Based on opponent's defensive rating vs league average
   - **Pace**: Game total deviation from 225 standard
   - **Fatigue**: B2B games reduce projection by 7%
   - **Rest**: Extra rest (3+ days) adds 1-2% boost
   - **Blowout**: Large spreads affect minutes expectations

3. **Probability Calculation**: Uses normal distribution with player's standard deviation

### Pick Selection

- Minimum 55% probability threshold
- Higher confidence tiers: 60-65%, 65%+
- Quality scoring combines probability, volume, and star status
- Sample size penalties for players with <10 games of data

### Parlay Strategy

- **Safe (2-leg)**: ~+200 payout, higher hit rate
- **Standard (3-4 leg)**: ~+500 payout
- **Longshot (5-6 leg)**: ~+1200 payout, lower hit rate

## Configuration

Key parameters can be adjusted in the respective files:

- `WEIGHTS` in model_engine.py - Adjust period weights
- `AdjustmentFactors` in model_engine.py - Tune adjustment multipliers
- `MIN_EDGE_THRESHOLD` in bet_generator.py - Minimum probability threshold
- Confidence tiers in bet_generator.py

## Disclaimer

This tool is for educational and research purposes only. Sports betting involves risk and past performance does not guarantee future results. Always gamble responsibly.

## License

MIT
