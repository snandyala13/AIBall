#!/usr/bin/env python3
"""
Backtest script for NBA prop betting model.
Tests projections against actual results from past games.

Includes adjustments for:
- Defense matchup
- Pace/game total
- Back-to-back fatigue
- Blowout risk
"""

import os
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from scipy import stats as scipy_stats
import time
import numpy as np

from adjustments import AdjustmentCalculator, GameAdjustments

API_KEY = os.getenv('BALLDONTLIE_API_KEY', '')
HEADERS = {'Authorization': API_KEY}
BASE_URL = 'https://api.balldontlie.io/v1'

# Initialize adjustment calculator
adjustment_calc = AdjustmentCalculator()


def get_games_for_date(date: str) -> List[Dict]:
    """Get all games for a specific date"""
    url = f'{BASE_URL}/games?dates[]={date}'
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code == 200:
        return resp.json().get('data', [])
    return []


def get_box_score(game_id: int) -> List[Dict]:
    """Get box score stats for a game"""
    url = f'{BASE_URL}/stats?game_ids[]={game_id}&per_page=50'
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code == 200:
        return resp.json().get('data', [])
    return []


def get_player_games_before_date(player_id: int, before_date: str, limit: int = 15) -> List[Dict]:
    """Get player's game logs from before a specific date"""
    url = f'{BASE_URL}/stats?player_ids[]={player_id}&seasons[]=2025&per_page=100'
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        return []

    all_games = resp.json().get('data', [])
    # Filter to games before the target date
    filtered = [g for g in all_games if g.get('game', {}).get('date', '') < before_date]
    # Sort by date descending
    filtered.sort(key=lambda x: x.get('game', {}).get('date', ''), reverse=True)
    # Filter out DNP games
    played = [g for g in filtered if (g.get('min') or '0') not in ['0', '0:00', '', None] and g.get('pts', 0) > 0]
    return played[:limit]


def get_game_context(game: Dict, player_team_id: int, game_date: str) -> Dict:
    """Get game context for adjustments (spread, total, B2B)"""
    home_team_id = game.get('home_team', {}).get('id')
    away_team_id = game.get('visitor_team', {}).get('id')
    is_home = player_team_id == home_team_id
    opponent_team_id = away_team_id if is_home else home_team_id

    # Default values (would be replaced with actual odds data in production)
    return {
        'opponent_team_id': opponent_team_id,
        'is_home': is_home,
        'game_total': 225.0,  # Default total
        'spread': 0.0,  # Default spread
        'game_date': game_date
    }


def apply_adjustments(base_projection: float, stat: str, player_team_id: int,
                     game_context: Dict) -> float:
    """Apply game context adjustments to base projection"""
    # Get adjustments from calculator
    adjustments = adjustment_calc.calculate_all_adjustments(
        player_id=0,  # Not used for team-level adjustments
        player_team_id=player_team_id,
        opponent_team_id=game_context['opponent_team_id'],
        game_date=game_context['game_date'],
        game_total=game_context['game_total'],
        spread=game_context['spread'],
        is_home=game_context['is_home']
    )

    # Map stat to adjustment key
    stat_map = {
        'pts': 'pts',
        'reb': 'reb',
        'ast': 'ast',
        'fg3m': 'fg3m',
        'pra': 'pts'  # PRA uses pts defense factor as proxy
    }
    stat_key = stat_map.get(stat, 'pts')

    # Get the multiplier for this stat
    multiplier = adjustments.get_stat_multiplier(stat_key)

    return base_projection * multiplier


def calculate_projection(games: List[Dict], stat: str) -> Tuple[float, float]:
    """Calculate weighted projection and std dev for a stat"""
    if not games:
        return 0.0, 5.0

    # Handle combined stats (PRA)
    if stat == 'pra':
        values = [
            (g.get('pts', 0) or 0) + (g.get('reb', 0) or 0) + (g.get('ast', 0) or 0)
            for g in games
        ]
    else:
        values = [g.get(stat, 0) or 0 for g in games]

    if len(values) < 3:
        return sum(values) / len(values) if values else 0.0, 5.0

    # Weighted average: L5 (45%), L6-10 (35%), L11-15 (20%)
    l5 = values[:5]
    l10 = values[:10]
    l15 = values[:15]

    l5_avg = sum(l5) / len(l5) if l5 else 0
    l10_avg = sum(l10) / len(l10) if l10 else 0
    l15_avg = sum(l15) / len(l15) if l15 else 0

    projection = l5_avg * 0.45 + l10_avg * 0.35 + l15_avg * 0.20

    # Calculate std dev
    std_dev = float(np.std(values[:10])) if len(values) >= 3 else 5.0
    std_dev = max(std_dev, 1.0)  # Minimum std dev

    return projection, std_dev


def calculate_over_probability(projection: float, line: float, std_dev: float) -> float:
    """Calculate probability of going over the line"""
    if std_dev <= 0:
        return 0.55 if projection > line else 0.45

    z_score = (line - projection) / std_dev
    over_prob = 1 - scipy_stats.norm.cdf(z_score)
    return max(0.05, min(0.95, over_prob))


def estimate_dk_line(season_avg: float, stat: str) -> float:
    """Estimate DraftKings line based on season average"""
    # DK typically sets lines slightly below season average
    # Round to nearest 0.5
    base = season_avg * 0.95  # Slightly below average
    return round(base * 2) / 2


def run_backtest(dates: List[str], min_minutes: int = 20) -> Dict:
    """
    Run backtest across multiple dates.

    Args:
        dates: List of dates to backtest (YYYY-MM-DD format)
        min_minutes: Minimum minutes played to include player

    Returns:
        Dictionary with backtest results
    """
    results = {
        'total_props': 0,
        'hits': 0,
        'misses': 0,
        'by_stat': {'pts': {'hits': 0, 'total': 0},
                   'reb': {'hits': 0, 'total': 0},
                   'ast': {'hits': 0, 'total': 0},
                   'fg3m': {'hits': 0, 'total': 0},
                   'pra': {'hits': 0, 'total': 0}},
        'by_edge': {'high': {'hits': 0, 'total': 0},  # >15% edge
                   'medium': {'hits': 0, 'total': 0},  # 10-15%
                   'low': {'hits': 0, 'total': 0}},    # 5-10%
        'predictions': []
    }

    for date in dates:
        print(f"\n{'='*50}")
        print(f"Processing {date}")
        print('='*50)

        games = get_games_for_date(date)
        print(f"Found {len(games)} games")

        for game in games:
            game_id = game['id']
            home = game['home_team']['full_name']
            away = game['visitor_team']['full_name']
            print(f"\n  {away} @ {home}")

            # Get box score
            box_score = get_box_score(game_id)
            if not box_score:
                print("    No box score available")
                continue

            # Process top players (by minutes)
            players_processed = 0
            for player_stats in sorted(box_score, key=lambda x: -(x.get('pts', 0) or 0))[:8]:
                player_id = player_stats['player']['id']
                player_name = f"{player_stats['player']['first_name']} {player_stats['player']['last_name']}"

                # Skip low-minute players
                min_str = player_stats.get('min', '0') or '0'
                try:
                    if ':' in str(min_str):
                        mins = int(min_str.split(':')[0])
                    else:
                        mins = int(float(min_str))
                except:
                    mins = 0

                if mins < min_minutes:
                    continue

                # Get player's games BEFORE this date
                prior_games = get_player_games_before_date(player_id, date)
                if len(prior_games) < 5:
                    continue  # Need enough history

                # Get team ID for adjustments
                player_team_id = player_stats.get('team', {}).get('id', 0)
                if player_team_id == 0:
                    player_team_id = player_stats['player'].get('team_id', 0)

                # Get game context for adjustments
                game_context = get_game_context(game, player_team_id, date)

                # Test each stat (including PRA and 3-pointers)
                for stat in ['pts', 'reb', 'ast', 'fg3m', 'pra']:
                    # Get actual result
                    if stat == 'pra':
                        actual = (player_stats.get('pts', 0) or 0) + \
                                 (player_stats.get('reb', 0) or 0) + \
                                 (player_stats.get('ast', 0) or 0)
                    else:
                        actual = player_stats.get(stat, 0) or 0

                    # Calculate base projection based on games BEFORE this date
                    base_projection, std_dev = calculate_projection(prior_games, stat)

                    # Apply game context adjustments
                    projection = apply_adjustments(base_projection, stat, player_team_id, game_context)

                    # Estimate what the line would have been
                    line = estimate_dk_line(projection, stat)
                    if line <= 0:
                        continue

                    # Calculate our probability
                    over_prob = calculate_over_probability(projection, line, std_dev)

                    # Determine our pick
                    if over_prob > 0.55:
                        our_pick = 'over'
                        our_prob = over_prob
                    elif over_prob < 0.45:
                        our_pick = 'under'
                        our_prob = 1 - over_prob
                    else:
                        continue  # Skip if too close to call

                    # Calculate edge
                    implied_prob = 0.52  # Typical -110 odds
                    edge = (our_prob - implied_prob) * 100

                    if edge < 5:
                        continue  # Skip low-edge plays

                    # Did it hit?
                    if our_pick == 'over':
                        hit = actual > line
                    else:
                        hit = actual < line

                    # Record result
                    results['total_props'] += 1
                    results['by_stat'][stat]['total'] += 1

                    if hit:
                        results['hits'] += 1
                        results['by_stat'][stat]['hits'] += 1
                    else:
                        results['misses'] += 1

                    # Track by edge level
                    if edge >= 15:
                        edge_level = 'high'
                    elif edge >= 10:
                        edge_level = 'medium'
                    else:
                        edge_level = 'low'

                    results['by_edge'][edge_level]['total'] += 1
                    if hit:
                        results['by_edge'][edge_level]['hits'] += 1

                    # Store prediction details
                    results['predictions'].append({
                        'date': date,
                        'player': player_name,
                        'stat': stat,
                        'projection': round(projection, 1),
                        'line': line,
                        'pick': our_pick,
                        'probability': round(our_prob, 3),
                        'edge': round(edge, 1),
                        'actual': actual,
                        'hit': hit
                    })

                players_processed += 1

            print(f"    Processed {players_processed} players")
            time.sleep(0.1)  # Rate limiting

    return results


def print_results(results: Dict):
    """Print backtest results summary"""
    print("\n" + "="*60)
    print("BACKTEST RESULTS SUMMARY")
    print("="*60)

    total = results['total_props']
    hits = results['hits']

    if total == 0:
        print("No props tested")
        return

    hit_rate = hits / total * 100

    print(f"\nOverall: {hits}/{total} ({hit_rate:.1f}%)")
    print(f"  Need >52.4% to beat -110 juice")
    print(f"  {'PROFITABLE' if hit_rate > 52.4 else 'NOT PROFITABLE'}")

    print("\nBy Stat Type:")
    for stat, data in results['by_stat'].items():
        if data['total'] > 0:
            rate = data['hits'] / data['total'] * 100
            print(f"  {stat.upper()}: {data['hits']}/{data['total']} ({rate:.1f}%)")

    print("\nBy Edge Level:")
    for level, data in results['by_edge'].items():
        if data['total'] > 0:
            rate = data['hits'] / data['total'] * 100
            print(f"  {level.upper()} (edge): {data['hits']}/{data['total']} ({rate:.1f}%)")

    # Show sample predictions
    print("\nSample Predictions (last 10):")
    for pred in results['predictions'][-10:]:
        status = '✅' if pred['hit'] else '❌'
        print(f"  {pred['player']} {pred['stat'].upper()} {pred['pick'].upper()} {pred['line']}: "
              f"Proj {pred['projection']} | Actual {pred['actual']} | Edge {pred['edge']:.0f}% {status}")


if __name__ == '__main__':
    # Backtest last 7 days
    dates = [
        '2025-12-25',
        '2025-12-26',
        '2025-12-27',
        '2025-12-28',
        '2025-12-29',
        '2025-12-30',
        '2025-12-31'
    ]

    print("Starting backtest...")
    print(f"Testing {len(dates)} days of games")

    results = run_backtest(dates)
    print_results(results)
