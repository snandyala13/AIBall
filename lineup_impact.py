"""
lineup_impact.py - Lineup Context Model

Adjusts player projections based on WHO they're playing WITH and AGAINST tonight.

Basketball is 5-on-5. A player's production depends on lineup context:
- Jokic with Murray: Murray handles the ball, Jokic posts up -> Jokic scores more
- Jokic without Murray: Jokic becomes primary playmaker -> assists up, scoring down
- A spot-up shooter next to a ball-dominant guard: more open 3s
- Two ball-dominant players together: usage cannibalization

The current model projects each player independently. This module fixes that by
computing ON/OFF splits from game logs and applying lineup adjustments.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

from data_pipeline import PlayerProfile, GameLog

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Stats we track for ON/OFF splits
LINEUP_STAT_KEYS = ["pts", "reb", "ast", "stl", "blk", "fg3m", "turnover"]

# Minimum games in both ON and OFF samples to trust the split
MIN_SAMPLE_ON = 5
MIN_SAMPLE_OFF = 5

# Maximum adjustment per single teammate influence (+/- 10%)
MAX_PER_TEAMMATE_ADJ = 0.10

# Maximum total lineup adjustment across all teammates (+/- 15%)
MAX_TOTAL_ADJ = 0.15


def _get_stat_from_log(log: GameLog, stat_key: str) -> float:
    """Extract a stat value from a GameLog by key."""
    return getattr(log, stat_key, 0) or 0


def build_game_rosters(all_profiles_for_team: List[PlayerProfile]) -> Dict[int, Set[str]]:
    """
    Build {game_id: set of player_names who played} from all team profiles.

    This lets us determine for any historical game which players were active,
    which is the foundation for ON/OFF split calculations.
    """
    rosters = {}
    for profile in all_profiles_for_team:
        for log in profile.game_logs:
            if log.minutes > 0:
                if log.game_id not in rosters:
                    rosters[log.game_id] = set()
                rosters[log.game_id].add(profile.name)
    return rosters


def get_on_off_for_player(
    player_profile: PlayerProfile,
    teammate_name: str,
    game_rosters: Dict[int, Set[str]]
) -> Tuple[List[GameLog], List[GameLog]]:
    """
    Split a player's game logs into games where a teammate played vs sat.

    Filters out games where the player played fewer than 5 minutes
    (DNP or garbage time) to avoid noise.

    Returns: (on_games, off_games)
    """
    on_games = []
    off_games = []
    for log in player_profile.game_logs:
        if log.minutes < 5:
            continue
        roster = game_rosters.get(log.game_id, set())
        if teammate_name in roster:
            on_games.append(log)
        else:
            off_games.append(log)
    return on_games, off_games


def _compute_averages(logs: List[GameLog], stat_keys: List[str]) -> Dict[str, float]:
    """Compute average stats from a list of game logs."""
    if not logs:
        return {}
    totals = {k: 0.0 for k in stat_keys}
    for log in logs:
        for k in stat_keys:
            totals[k] += _get_stat_from_log(log, k)
    n = len(logs)
    return {k: round(v / n, 2) for k, v in totals.items()}


class LineupImpactModel:
    """
    Adjusts player projections based on lineup context -- who's playing tonight
    vs who was playing during the games that built the player's averages.

    Key insight: a player's L10 average was built in specific lineup contexts.
    If a key teammate is now out (or back), the production changes.
    """

    def __init__(self):
        # Cache: player_name -> {teammate_name: split_data}
        self._on_off_cache: Dict[str, Dict[str, Dict[str, float]]] = {}
        # Cache: frozenset of team profile names -> game rosters
        self._roster_cache: Dict[str, Dict[int, Set[str]]] = {}

    def _get_or_build_rosters(
        self,
        team_profiles: List[PlayerProfile]
    ) -> Dict[int, Set[str]]:
        """Get cached game rosters or build them."""
        cache_key = ",".join(sorted(p.name for p in team_profiles))
        if cache_key not in self._roster_cache:
            self._roster_cache[cache_key] = build_game_rosters(team_profiles)
        return self._roster_cache[cache_key]

    def calculate_on_off_splits(
        self,
        player_profile: PlayerProfile,
        team_profiles: List[PlayerProfile]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate how a player's stats change when each teammate plays vs sits.

        Uses game logs: for each teammate, split the player's games into
        "teammate played" vs "teammate didn't play" and compute stat averages.

        Returns: {
            teammate_name: {
                "pts_on": 22.5,    # Player's avg when teammate plays
                "pts_off": 26.1,   # Player's avg when teammate sits
                "pts_diff": +3.6,  # Positive = player better WITHOUT teammate
                "reb_on": 5.2,
                "reb_off": 6.8,
                "reb_diff": +1.6,
                "games_on": 35,
                "games_off": 8,
            }
        }

        Only include teammates with MIN_SAMPLE_ON games ON and MIN_SAMPLE_OFF
        games OFF.
        """
        # Check cache
        if player_profile.name in self._on_off_cache:
            return self._on_off_cache[player_profile.name]

        game_rosters = self._get_or_build_rosters(team_profiles)
        splits = {}

        for teammate in team_profiles:
            if teammate.name == player_profile.name:
                continue

            on_games, off_games = get_on_off_for_player(
                player_profile, teammate.name, game_rosters
            )

            if len(on_games) < MIN_SAMPLE_ON or len(off_games) < MIN_SAMPLE_OFF:
                continue

            on_avgs = _compute_averages(on_games, LINEUP_STAT_KEYS)
            off_avgs = _compute_averages(off_games, LINEUP_STAT_KEYS)

            teammate_split = {
                "games_on": len(on_games),
                "games_off": len(off_games),
            }

            for stat_key in LINEUP_STAT_KEYS:
                on_val = on_avgs.get(stat_key, 0)
                off_val = off_avgs.get(stat_key, 0)
                teammate_split[f"{stat_key}_on"] = on_val
                teammate_split[f"{stat_key}_off"] = off_val
                teammate_split[f"{stat_key}_diff"] = round(off_val - on_val, 2)

            splits[teammate.name] = teammate_split

        self._on_off_cache[player_profile.name] = splits
        return splits

    def calculate_lineup_adjustment(
        self,
        player_profile: PlayerProfile,
        active_players: List[PlayerProfile],
        scratched_players: List[PlayerProfile],
        team_profiles: List[PlayerProfile] = None
    ) -> Dict[str, float]:
        """
        Calculate net stat adjustment based on who's playing tonight vs
        historical context.

        Logic:
        1. Get ON/OFF splits for this player vs each teammate
        2. For each scratched teammate: if player was BETTER without them, boost.
           If WORSE without them, reduce.
        3. For returning players who are active but might not have been in recent
           games: apply reverse logic.
        4. Net the adjustments, cap at +/- MAX_TOTAL_ADJ per stat.

        Returns: dict of {stat_key: adjustment_multiplier}
            e.g. {"pts": 1.05, "ast": 0.97} means +5% points, -3% assists
        """
        if team_profiles is None:
            # Combine active + scratched as the full team roster for building rosters
            team_profiles = list(active_players) + list(scratched_players)

        # Only consider teammates on the same team
        player_team = player_profile.team
        same_team_profiles = [p for p in team_profiles if p.team == player_team]

        if len(same_team_profiles) < 2:
            return {k: 1.0 for k in LINEUP_STAT_KEYS}

        splits = self.calculate_on_off_splits(player_profile, same_team_profiles)

        if not splits:
            return {k: 1.0 for k in LINEUP_STAT_KEYS}

        # Collect scratched teammate names on same team
        scratched_names = set(
            p.name for p in scratched_players if p.team == player_team
        )

        # Calculate per-stat adjustments
        net_adjustments = {k: 0.0 for k in LINEUP_STAT_KEYS}

        for teammate_name, split_data in splits.items():
            if teammate_name not in scratched_names:
                # Teammate is active (playing tonight) -- no adjustment needed
                # unless they were recently absent and are now returning.
                # For simplicity, we only adjust for scratched players.
                continue

            # Teammate is OUT tonight. How does the player perform without them?
            for stat_key in LINEUP_STAT_KEYS:
                on_val = split_data.get(f"{stat_key}_on", 0)
                off_val = split_data.get(f"{stat_key}_off", 0)

                if on_val <= 0:
                    continue

                # Percentage change when teammate is OUT
                # Positive = player is better without this teammate
                # Negative = player is worse without this teammate
                pct_diff = (off_val - on_val) / on_val

                # Cap per-teammate influence
                capped_diff = max(-MAX_PER_TEAMMATE_ADJ, min(MAX_PER_TEAMMATE_ADJ, pct_diff))

                # Weight by sample quality: more OFF games = more reliable
                off_games = split_data.get("games_off", 0)
                # Scale from 0.5 (5 games) to 1.0 (10+ games)
                sample_weight = min(1.0, 0.5 + (off_games - MIN_SAMPLE_OFF) * 0.1)

                net_adjustments[stat_key] += capped_diff * sample_weight

        # Cap total adjustment per stat and convert to multipliers
        result = {}
        for stat_key in LINEUP_STAT_KEYS:
            capped = max(-MAX_TOTAL_ADJ, min(MAX_TOTAL_ADJ, net_adjustments[stat_key]))
            result[stat_key] = round(1.0 + capped, 4)

        return result

    def get_usage_redistribution_from_logs(
        self,
        beneficiary_profile: PlayerProfile,
        scratched_profile: PlayerProfile,
        team_profiles: List[PlayerProfile] = None
    ) -> Optional[Dict[str, float]]:
        """
        DATA-DRIVEN alternative to the hardcoded USAGE_REDISTRIBUTION matrix.

        Look at games where the scratched player was out and compute how much
        the beneficiary's stats actually changed. This is empirical, not guesswork.

        Returns: {stat_key: actual_multiplier} or None if insufficient data.

        Example: If Jaylen Brown averages 22 PPG with Tatum and 28 PPG without,
        this returns {"pts": 1.273, ...} (a 27.3% boost).
        """
        if team_profiles is None:
            team_profiles = [beneficiary_profile, scratched_profile]

        # Both must be on the same team
        if beneficiary_profile.team != scratched_profile.team:
            return None

        same_team_profiles = [
            p for p in team_profiles if p.team == beneficiary_profile.team
        ]

        splits = self.calculate_on_off_splits(beneficiary_profile, same_team_profiles)

        if scratched_profile.name not in splits:
            return None

        split_data = splits[scratched_profile.name]
        result = {}

        for stat_key in LINEUP_STAT_KEYS:
            on_val = split_data.get(f"{stat_key}_on", 0)
            off_val = split_data.get(f"{stat_key}_off", 0)

            if on_val <= 0:
                continue

            # The multiplier is how much the player's stat changes when
            # the scratched player is OUT.
            # off_val / on_val: e.g. 28 / 22 = 1.273
            multiplier = off_val / on_val

            # Cap at reasonable bounds to prevent crazy projections
            # Allow up to +/- 20% since this is empirical data
            multiplier = max(0.80, min(1.20, multiplier))

            result[stat_key] = round(multiplier, 4)

        if not result:
            return None

        return result

    def clear_cache(self):
        """Clear all caches (call between games or when refreshing data)."""
        self._on_off_cache.clear()
        self._roster_cache.clear()
