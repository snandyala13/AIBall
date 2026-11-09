"""
quantile_engine.py - Empirical Quantile Probability Estimation

Replaces the normal distribution assumption in EdgeCalculator with empirical
CDF interpolation from actual game logs. NBA stats are bounded at 0,
left-skewed, heavy-tailed, and sometimes bimodal (steals/blocks), so
quantile-based estimation improves calibration.

Approach:
1. Build empirical CDF from a player's game logs for each prop type
2. Interpolate to find the percentile at the DK line
3. P(over) = 1 - percentile
4. Blend with z-score estimate for stability (especially low-sample players)
5. Apply the same damping as EdgeCalculator to prevent overconfidence
"""

import logging
from typing import Dict, Optional

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Stat key mapping matching model_engine.py STAT_MAPPING values
# PropType enum value -> game log attribute name
PROP_TO_STAT_KEY = {
    "points": "pts",
    "rebounds": "reb",
    "assists": "ast",
    "threes": "fg3m",
    "steals": "stl",
    "blocks": "blk",
    "turnovers": "turnover",
}

# Combined stat props -> list of game log attribute names
COMBINED_PROPS = {
    "pts_rebs_asts": ["pts", "reb", "ast"],
    "pts_rebs": ["pts", "reb"],
    "pts_asts": ["pts", "ast"],
    "rebs_asts": ["reb", "ast"],
    "stls_blks": ["stl", "blk"],
}

# Position-average quantile reference data.
# Derived from league-wide distributions by position.
# Each entry: {stat_key: (values_at_percentiles_10_25_50_75_90)}
# These serve as fallback when a player has fewer than 10 game logs.
POSITION_AVERAGE_QUANTILES = {
    "PG": {
        "pts": (8.0, 13.0, 18.0, 24.0, 32.0),
        "reb": (1.0, 2.0, 4.0, 5.0, 7.0),
        "ast": (2.0, 4.0, 6.0, 8.0, 11.0),
        "fg3m": (0.0, 1.0, 2.0, 3.0, 5.0),
        "stl": (0.0, 0.0, 1.0, 2.0, 3.0),
        "blk": (0.0, 0.0, 0.0, 1.0, 1.0),
        "turnover": (0.0, 1.0, 2.0, 3.0, 5.0),
    },
    "SG": {
        "pts": (7.0, 12.0, 17.0, 23.0, 30.0),
        "reb": (1.0, 2.0, 3.0, 5.0, 7.0),
        "ast": (1.0, 2.0, 3.0, 5.0, 7.0),
        "fg3m": (0.0, 1.0, 2.0, 3.0, 5.0),
        "stl": (0.0, 0.0, 1.0, 2.0, 3.0),
        "blk": (0.0, 0.0, 0.0, 1.0, 1.0),
        "turnover": (0.0, 1.0, 2.0, 3.0, 4.0),
    },
    "SF": {
        "pts": (6.0, 11.0, 16.0, 22.0, 29.0),
        "reb": (2.0, 3.0, 5.0, 7.0, 9.0),
        "ast": (1.0, 1.0, 2.0, 4.0, 6.0),
        "fg3m": (0.0, 0.0, 1.0, 2.0, 4.0),
        "stl": (0.0, 0.0, 1.0, 1.0, 2.0),
        "blk": (0.0, 0.0, 0.0, 1.0, 2.0),
        "turnover": (0.0, 1.0, 1.0, 2.0, 4.0),
    },
    "PF": {
        "pts": (5.0, 10.0, 15.0, 21.0, 28.0),
        "reb": (2.0, 4.0, 6.0, 8.0, 11.0),
        "ast": (0.0, 1.0, 2.0, 3.0, 5.0),
        "fg3m": (0.0, 0.0, 1.0, 2.0, 3.0),
        "stl": (0.0, 0.0, 1.0, 1.0, 2.0),
        "blk": (0.0, 0.0, 1.0, 1.0, 2.0),
        "turnover": (0.0, 1.0, 1.0, 2.0, 3.0),
    },
    "C": {
        "pts": (4.0, 8.0, 13.0, 19.0, 26.0),
        "reb": (3.0, 5.0, 8.0, 11.0, 14.0),
        "ast": (0.0, 1.0, 2.0, 3.0, 4.0),
        "fg3m": (0.0, 0.0, 0.0, 1.0, 2.0),
        "stl": (0.0, 0.0, 1.0, 1.0, 2.0),
        "blk": (0.0, 0.0, 1.0, 2.0, 3.0),
        "turnover": (0.0, 1.0, 1.0, 2.0, 3.0),
    },
}

# The percentile breakpoints corresponding to the tuples above
_POSITION_PERCENTILES = np.array([10.0, 25.0, 50.0, 75.0, 90.0])

# Minimum number of game logs for reliable empirical CDF
MIN_GAMES_FOR_EMPIRICAL = 10


class QuantileEngine:
    """Estimates P(stat > line) using empirical quantile interpolation
    instead of z-scores.

    The engine builds an empirical CDF from a player's recent game logs for
    each prop type, interpolates to find the percentile at the DraftKings line,
    and returns P(over) = 1 - percentile. Results are blended with the z-score
    estimate for stability and damped to prevent overconfidence.
    """

    def __init__(self, game_logs_by_player: Optional[Dict] = None):
        # Cache: key = "player_name::prop_type" -> {"values": ndarray, "percentiles": ndarray}
        self.quantile_cache: Dict[str, Dict[str, np.ndarray]] = {}

        # Pre-populate cache if bulk data provided
        if game_logs_by_player:
            for player_name, logs in game_logs_by_player.items():
                for prop_type in PROP_TO_STAT_KEY:
                    self.build_quantile_function(player_name, prop_type, logs)

    def _extract_stat_values(self, prop_type: str, game_logs: list) -> np.ndarray:
        """Extract stat values from game logs for a given prop type.

        Filters out zero-minute games (DNPs) to avoid polluting the distribution.

        Args:
            prop_type: String prop type (e.g., "points", "pts_rebs_asts").
            game_logs: List of GameLog dataclass instances.

        Returns:
            1-D numpy array of stat values from qualifying games.
        """
        values = []

        for log in game_logs:
            # Skip zero-minute games (DNP / injury scratch)
            if getattr(log, "minutes", 0) <= 0:
                continue

            if prop_type in COMBINED_PROPS:
                # Sum the component stats
                stat_keys = COMBINED_PROPS[prop_type]
                total = sum(getattr(log, k, 0) for k in stat_keys)
                values.append(float(total))
            else:
                stat_key = PROP_TO_STAT_KEY.get(prop_type)
                if stat_key is None:
                    continue
                values.append(float(getattr(log, stat_key, 0)))

        return np.array(values, dtype=float)

    def build_quantile_function(
        self,
        player_name: str,
        prop_type: str,
        game_logs: list,
        position: str = None,
    ) -> Dict[str, np.ndarray]:
        """Build empirical CDF from game logs for a specific player-prop combo.

        If fewer than MIN_GAMES_FOR_EMPIRICAL qualifying games exist, falls back
        to position-average quantiles.

        Args:
            player_name: Player name for cache key.
            prop_type: String prop type (e.g., "points", "rebounds").
            game_logs: List of GameLog dataclass instances.
            position: Player position (e.g., "PG"). Used for fallback.

        Returns:
            Dict with 'values' (sorted stat values) and 'percentiles'
            (corresponding CDF percentages 0-100).
        """
        cache_key = f"{player_name}::{prop_type}"

        values = self._extract_stat_values(prop_type, game_logs)

        if len(values) < MIN_GAMES_FOR_EMPIRICAL:
            # Not enough data -- use position-average fallback
            result = self._position_average_quantiles(position or "SF", prop_type)
            self.quantile_cache[cache_key] = result
            return result

        # Sort values and compute corresponding CDF percentiles
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        # Use midpoint CDF formula: percentile_i = (i + 0.5) / n * 100
        percentiles = (np.arange(n) + 0.5) / n * 100.0

        result = {
            "values": sorted_vals,
            "percentiles": percentiles,
        }
        self.quantile_cache[cache_key] = result
        return result

    def calculate_over_probability(
        self,
        player_name: str,
        prop_type: str,
        line: float,
        projection: float,
        game_logs: list = None,
        position: str = None,
    ) -> float:
        """Calculate P(stat > line) using empirical quantile interpolation.

        Steps:
            1. Build/retrieve empirical CDF for this player-prop.
            2. Interpolate to find the percentile at the line value.
            3. P(over) = 1 - percentile / 100.
            4. Apply projection-based shift: if the model projects meaningfully
               above/below the historical mean, nudge the empirical estimate.
            5. Apply damping (same philosophy as EdgeCalculator).

        Args:
            player_name: Player name.
            prop_type: String prop type.
            line: DraftKings line value.
            projection: Our model's projected value for this stat.
            game_logs: List of GameLog instances. If None, uses cached CDF.
            position: Player position for fallback. Defaults to "SF".

        Returns:
            Empirical P(over) after projection adjustment and damping,
            clamped to [0.28, 0.75].
        """
        cache_key = f"{player_name}::{prop_type}"

        # Build CDF if not cached or if fresh logs provided
        if cache_key not in self.quantile_cache or game_logs is not None:
            if game_logs is None:
                # No logs and no cache -- return neutral
                return 0.50
            self.build_quantile_function(
                player_name, prop_type, game_logs, position
            )

        cdf_data = self.quantile_cache[cache_key]
        sorted_vals = cdf_data["values"]
        percentiles = cdf_data["percentiles"]

        # Interpolate to find the CDF value at the line
        # np.interp: if line is below all values -> returns first percentile
        #            if line is above all values -> returns last percentile
        cdf_at_line = np.interp(line, sorted_vals, percentiles)

        # Convert to probability (percentiles are 0-100)
        empirical_over = 1.0 - (cdf_at_line / 100.0)

        # --- Projection-based shift ---
        # If our projection differs meaningfully from the historical mean,
        # shift the empirical estimate to account for current-form adjustments
        # that the raw game logs don't capture (pace, defense, injuries, etc.)
        hist_mean = float(np.mean(sorted_vals)) if len(sorted_vals) > 0 else line
        if hist_mean > 0:
            projection_ratio = projection / hist_mean
            # Clamp the ratio to avoid extreme shifts
            projection_ratio = max(0.80, min(1.20, projection_ratio))
            # The shift: if projection is 10% above historical mean,
            # nudge the over probability up proportionally
            shift = (projection_ratio - 1.0) * 0.30  # 30% pass-through
            empirical_over = empirical_over + shift

        # Clamp before damping
        empirical_over = max(0.05, min(0.95, empirical_over))

        # --- Damping (matches EdgeCalculator philosophy) ---
        if empirical_over > 0.65:
            empirical_over = 0.65 + (empirical_over - 0.65) * 0.5
        elif empirical_over < 0.35:
            empirical_over = 0.35 - (0.35 - empirical_over) * 0.5

        return max(0.28, min(0.75, empirical_over))

    def _position_average_quantiles(
        self, position: str, prop_type: str
    ) -> Dict[str, np.ndarray]:
        """Fallback quantile functions based on position averages.

        Used for rookies or players with fewer than MIN_GAMES_FOR_EMPIRICAL
        qualifying game logs.

        Args:
            position: Player position (PG, SG, SF, PF, C).
            prop_type: String prop type.

        Returns:
            Dict with 'values' and 'percentiles' arrays.
        """
        # Resolve the stat key for lookup
        if prop_type in COMBINED_PROPS:
            # For combined props, sum the component position averages
            component_keys = COMBINED_PROPS[prop_type]
            pos_data = POSITION_AVERAGE_QUANTILES.get(position, POSITION_AVERAGE_QUANTILES["SF"])
            combined_values = np.zeros(len(_POSITION_PERCENTILES))
            for key in component_keys:
                component_vals = pos_data.get(key)
                if component_vals is not None:
                    combined_values += np.array(component_vals, dtype=float)
            return {
                "values": combined_values,
                "percentiles": _POSITION_PERCENTILES.copy(),
            }

        stat_key = PROP_TO_STAT_KEY.get(prop_type)
        if stat_key is None:
            # Unknown prop type -- return flat distribution
            return {
                "values": np.array([0.0, 5.0, 10.0, 15.0, 20.0]),
                "percentiles": _POSITION_PERCENTILES.copy(),
            }

        pos_data = POSITION_AVERAGE_QUANTILES.get(position, POSITION_AVERAGE_QUANTILES["SF"])
        stat_quantiles = pos_data.get(stat_key)

        if stat_quantiles is None:
            return {
                "values": np.array([0.0, 5.0, 10.0, 15.0, 20.0]),
                "percentiles": _POSITION_PERCENTILES.copy(),
            }

        return {
            "values": np.array(stat_quantiles, dtype=float),
            "percentiles": _POSITION_PERCENTILES.copy(),
        }

    def get_num_qualifying_games(self, game_logs: list) -> int:
        """Count game logs with non-zero minutes (for blend weight calculation).

        Args:
            game_logs: List of GameLog instances.

        Returns:
            Number of games where minutes > 0.
        """
        return sum(1 for log in game_logs if getattr(log, "minutes", 0) > 0)
