#!/bin/bash
# Wait for fetch to finish, then retrain on all data
cd /Users/suhaasnandyala/Documents/aiBall
source .venv/bin/activate

echo "Waiting for 2026 OOS fetch to complete..."
while pgrep -f "fetch_backtest_data" > /dev/null; do
    sleep 30
done

echo "Fetch complete. Checking data..."

# Rename the new OOS data
if [ -f outputs/backtest_oos_data.json ]; then
    python3 -c "
import json
with open('outputs/backtest_oos_legs.json') as f:
    legs = json.load(f)
print(f'2026 OOS: {len(legs)} legs')
"
fi

echo ""
echo "=== COMBINED DATASET ==="
python3 -c "
import json

# Load all three datasets
datasets = []

# 1. 2023-24
with open('outputs/backtest_2023_24_legs.json') as f:
    d1 = json.load(f)
    datasets.extend(d1)
    print(f'2023-24: {len(d1)} legs')

# 2. Original (2024-26)
with open('outputs/backtest_real_lines_full.json') as f:
    d2 = json.load(f)['legs']
    datasets.extend(d2)
    print(f'2024-26: {len(d2)} legs')

# 3. 2026 OOS
with open('outputs/backtest_oos_legs.json') as f:
    d3 = json.load(f)
    datasets.extend(d3)
    print(f'2026 OOS: {len(d3)} legs')

print(f'TOTAL: {len(datasets)} legs')

# Save combined
with open('outputs/backtest_combined_all.json', 'w') as f:
    json.dump({'legs': datasets, 'metadata': {'total': len(datasets), 'sources': ['2023-24','2024-26','2026-oos']}}, f)
print('Saved to outputs/backtest_combined_all.json')
"

echo ""
echo "=== RETRAINING V10 ON FULL DATASET ==="
python3 train_new_filter.py

echo ""
echo "=== DONE ==="
date
