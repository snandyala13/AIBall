# PropTerminal - Quick Start

## Setup

```bash
cd /Users/suhaasnandyala/Documents/aiBall
source .venv/bin/activate
source .env
```

## Web Dashboard (primary way to use)

```bash
python ball.py --web
# Open http://localhost:8000
```

- Pick a date (arrows or date picker, defaults to today)
- Click **Analyze** on any game or **Analyze All Games**
- Top Picks banner appears with best plays ranked by edge
- Click player names for deep dives
- Finished games show hit/miss results with P&L

## CLI Commands

```bash
python ball.py --today                      # Analyze all today's games (text output)
python ball.py --game "Celtics vs Bucks"    # Single game analysis
python ball.py --player "Jayson Tatum"      # Player deep dive
python ball.py --demo                       # Demo mode (no API needed)
python ball.py --check-drift                # Check model health
python ball.py --retrain                    # Retrain ML models
python ball.py --retrain --force            # Force retrain
```

## API Keys (in .env)

| Key | Service | Purpose |
|-----|---------|---------|
| `BALLDONTLIE_API_KEY` | Balldontlie | Player stats, game logs, box scores |
| `PROPLINE_API_KEY` | PropLine | Fast bulk props, graded results |
| `ODDS_API_KEY` | The Odds API | Historical odds for backtesting |

## Backtesting

```bash
python backtest_model_comparison.py              # Old vs new model comparison
python comprehensive_backtest.py                 # Detailed probability analysis
python feature_validation.py                     # Which features help vs hurt
python train_new_filter.py                       # Train new ML filter
python fetch_backtest_data.py --start-date X     # Fetch historical data
```

## Key Directories

```
web/                  Dashboard (FastAPI + Jinja2)
models/               Trained ML models
outputs/              Backtest results, predictions log
archive/              Old output files
```
