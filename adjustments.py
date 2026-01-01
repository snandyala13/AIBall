"""
Game Context Adjustments Module

Provides adjustment factors for:
1. Defense matchup (opponent defensive rating)
2. Pace/game total
3. Back-to-back fatigue
4. Blowout risk (based on spread)
5. Opponent-specific historical performance
"""

import os
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

API_KEY = os.getenv('BALLDONTLIE_API_KEY', '')
HEADERS = {'Authorization': API_KEY}
BASE_URL = 'https://api.balldontlie.io/v1'


@dataclass
class GameAdjustments:
    """Adjustment factors for a specific game context"""
    defense_factor: float = 1.0      # Multiplier based on opponent defense
    pace_factor: float = 1.0         # Multiplier based on game pace/total
    fatigue_factor: float = 1.0      # Multiplier for B2B fatigue
    blowout_factor: float = 1.0      # Multiplier for blowout risk (minutes)
    opponent_factor: float = 1.0     # Multiplier based on historical vs opponent

    # Per-stat adjustments (some stats affected more by defense)
    pts_defense_mult: float = 1.0
    reb_defense_mult: float = 1.0
    ast_defense_mult: float = 1.0
    fg3m_defense_mult: float = 1.0

    def get_stat_multiplier(self, stat: str) -> float:
        """Get total multiplier for a specific stat"""
        base = self.pace_factor * self.fatigue_factor * self.blowout_factor * self.opponent_factor

        if stat == 'pts':
            return base * self.pts_defense_mult
        elif stat == 'reb':
            return base * self.reb_defense_mult
        elif stat == 'ast':
            return base * self.ast_defense_mult
        elif stat == 'fg3m':
            return base * self.fg3m_defense_mult
        else:
            return base * self.defense_factor


class AdjustmentCalculator:
    """Calculate game context adjustments"""

    # League average defensive rating (approx)
    LEAGUE_AVG_DEF_RTG = 112.0

    # League average pace
    LEAGUE_AVG_PACE = 100.0

    # Standard game total
    STANDARD_TOTAL = 225.0

    def __init__(self):
        self._team_stats_cache: Dict[int, Dict] = {}
        self._schedule_cache: Dict[int, List] = {}

    def get_team_defensive_rating(self, team_id: int) -> float:
        """Get team's defensive rating for current season"""
        if team_id in self._team_stats_cache:
            return self._team_stats_cache[team_id].get('def_rtg', self.LEAGUE_AVG_DEF_RTG)

        # Get team's advanced stats (aggregate from player stats)
        try:
            url = f'{BASE_URL}/stats/advanced?seasons[]=2025&team_ids[]={team_id}&per_page=100'
            resp = requests.get(url, headers=HEADERS)
            if resp.status_code == 200:
                data = resp.json().get('data', [])
                if data:
                    # Average defensive rating across team's games
                    def_rtgs = [d.get('defensive_rating', 0) for d in data if d.get('defensive_rating')]
                    if def_rtgs:
                        avg_def = sum(def_rtgs) / len(def_rtgs)
                        self._team_stats_cache[team_id] = {'def_rtg': avg_def}
                        return avg_def
        except Exception as e:
            logger.warning(f"Failed to get defensive rating for team {team_id}: {e}")

        return self.LEAGUE_AVG_DEF_RTG

    def calculate_defense_adjustment(self, opponent_team_id: int) -> Dict[str, float]:
        """
        Calculate adjustment based on opponent's defensive rating.

        Returns multipliers for each stat type.
        Elite defense (top 5) = reduce by 8-12%
        Poor defense (bottom 5) = increase by 5-8%
        """
        def_rtg = self.get_team_defensive_rating(opponent_team_id)

        # How much better/worse than average?
        # Lower def rating = better defense
        diff_from_avg = self.LEAGUE_AVG_DEF_RTG - def_rtg

        # Convert to adjustment factor
        # Elite D (def_rtg ~105) = diff of +7, adjustment of -7%
        # Poor D (def_rtg ~118) = diff of -6, adjustment of +6%

        pts_adj = 1.0 - (diff_from_avg * 0.010)  # Points most affected
        reb_adj = 1.0 - (diff_from_avg * 0.005)  # Rebounds less affected
        ast_adj = 1.0 - (diff_from_avg * 0.008)  # Assists moderately affected
        fg3m_adj = 1.0 - (diff_from_avg * 0.012) # 3s highly affected by perimeter D

        # Clamp to reasonable range
        pts_adj = max(0.85, min(1.12, pts_adj))
        reb_adj = max(0.92, min(1.08, reb_adj))
        ast_adj = max(0.88, min(1.10, ast_adj))
        fg3m_adj = max(0.82, min(1.15, fg3m_adj))

        return {
            'pts': pts_adj,
            'reb': reb_adj,
            'ast': ast_adj,
            'fg3m': fg3m_adj,
            'overall': (pts_adj + reb_adj + ast_adj) / 3
        }

    def calculate_pace_adjustment(self, game_total: float) -> float:
        """
        Calculate adjustment based on expected game pace (using game total).

        High total (240+) = more possessions = more stats
        Low total (210-) = fewer possessions = fewer stats
        """
        if game_total <= 0:
            return 1.0

        # Deviation from standard
        diff = game_total - self.STANDARD_TOTAL

        # ~1% adjustment per 3 points of total
        adjustment = 1.0 + (diff / 300)

        # Clamp to ±10%
        return max(0.90, min(1.10, adjustment))

    def is_back_to_back(self, team_id: int, game_date: str) -> bool:
        """Check if team played yesterday (back-to-back)"""
        try:
            target_date = datetime.strptime(game_date, '%Y-%m-%d')
            yesterday = (target_date - timedelta(days=1)).strftime('%Y-%m-%d')

            url = f'{BASE_URL}/games?team_ids[]={team_id}&dates[]={yesterday}'
            resp = requests.get(url, headers=HEADERS)
            if resp.status_code == 200:
                games = resp.json().get('data', [])
                return len(games) > 0
        except Exception as e:
            logger.warning(f"Failed to check B2B for team {team_id}: {e}")

        return False

    def calculate_fatigue_adjustment(self, is_b2b: bool, days_rest: int = 1) -> float:
        """
        Calculate fatigue/rest adjustment.

        B2B (1 day rest) = reduce projection by ~7%
        Normal (2 days) = no adjustment
        Extra rest (3+ days) = small boost ~1-2% (light weight per user request)
        """
        if is_b2b or days_rest <= 1:
            return 0.93  # 7% reduction for B2B
        elif days_rest >= 4:
            return 1.02  # 2% boost for 4+ days rest (light)
        elif days_rest >= 3:
            return 1.01  # 1% boost for 3 days rest (very light)
        return 1.0  # Normal rest (2 days)

    def calculate_blowout_adjustment(self, spread: float, is_favorite: bool) -> float:
        """
        Calculate adjustment for potential blowouts.

        Big favorites (-12+) may rest starters in Q4
        Big underdogs (+12+) may get garbage time stats
        """
        abs_spread = abs(spread)

        if abs_spread < 8:
            return 1.0  # Close game, no adjustment

        if is_favorite and abs_spread >= 12:
            # Big favorite - starters might rest
            # Reduce minutes expectation by 3-8%
            reduction = min(0.08, (abs_spread - 12) * 0.01 + 0.03)
            return 1.0 - reduction

        if not is_favorite and abs_spread >= 12:
            # Big underdog - bench might get more run
            # Slight reduction for starters
            return 0.97

        return 1.0

    def get_player_vs_opponent_history(
        self,
        player_id: int,
        opponent_team_id: int,
        stat: str,
        games_back: int = 10
    ) -> Optional[float]:
        """
        Get player's historical performance vs specific opponent.

        Returns: adjustment factor based on historical over/under performance
        """
        try:
            url = f'{BASE_URL}/stats?player_ids[]={player_id}&seasons[]=2025&seasons[]=2024&per_page=100'
            resp = requests.get(url, headers=HEADERS)
            if resp.status_code != 200:
                return None

            all_games = resp.json().get('data', [])

            # Filter to games vs this opponent
            vs_opponent = []
            for g in all_games:
                game_info = g.get('game', {})
                home_id = game_info.get('home_team_id')
                away_id = game_info.get('visitor_team_id')

                if opponent_team_id in [home_id, away_id]:
                    stat_val = g.get(stat, 0) or 0
                    if stat_val > 0:  # Filter DNP
                        vs_opponent.append(stat_val)

            if len(vs_opponent) < 2:
                return None  # Not enough history

            # Get overall average for comparison
            all_stats = [g.get(stat, 0) or 0 for g in all_games if (g.get(stat, 0) or 0) > 0]
            if not all_stats:
                return None

            overall_avg = sum(all_stats[:20]) / min(20, len(all_stats))
            opponent_avg = sum(vs_opponent[:games_back]) / len(vs_opponent[:games_back])

            if overall_avg <= 0:
                return None

            # How much better/worse vs this opponent?
            ratio = opponent_avg / overall_avg

            # Clamp adjustment to ±15%
            return max(0.85, min(1.15, ratio))

        except Exception as e:
            logger.warning(f"Failed to get opponent history: {e}")
            return None

    def calculate_all_adjustments(
        self,
        player_id: int,
        player_team_id: int,
        opponent_team_id: int,
        game_date: str,
        game_total: float = 225.0,
        spread: float = 0.0,
        is_home: bool = True
    ) -> GameAdjustments:
        """
        Calculate all adjustment factors for a player in a specific game context.
        """
        adjustments = GameAdjustments()

        # 1. Defense adjustment
        defense_adj = self.calculate_defense_adjustment(opponent_team_id)
        adjustments.pts_defense_mult = defense_adj['pts']
        adjustments.reb_defense_mult = defense_adj['reb']
        adjustments.ast_defense_mult = defense_adj['ast']
        adjustments.fg3m_defense_mult = defense_adj['fg3m']
        adjustments.defense_factor = defense_adj['overall']

        # 2. Pace adjustment
        adjustments.pace_factor = self.calculate_pace_adjustment(game_total)

        # 3. Fatigue adjustment
        is_b2b = self.is_back_to_back(player_team_id, game_date)
        adjustments.fatigue_factor = self.calculate_fatigue_adjustment(is_b2b)

        # 4. Blowout adjustment
        is_favorite = (is_home and spread < 0) or (not is_home and spread > 0)
        adjustments.blowout_factor = self.calculate_blowout_adjustment(spread, is_favorite)

        # 5. Opponent-specific (optional - can be slow)
        # This would need to be called per-stat, so we leave it at 1.0 here
        adjustments.opponent_factor = 1.0

        return adjustments


# Convenience function for quick adjustments
def get_game_adjustments(
    player_id: int,
    player_team_id: int,
    opponent_team_id: int,
    game_date: str,
    game_total: float = 225.0,
    spread: float = 0.0,
    is_home: bool = True
) -> GameAdjustments:
    """Get all adjustments for a player/game context"""
    calc = AdjustmentCalculator()
    return calc.calculate_all_adjustments(
        player_id, player_team_id, opponent_team_id,
        game_date, game_total, spread, is_home
    )
