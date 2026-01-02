"""
model_engine.py - The Analysis Engine

Calculates projections with multiple adjustment layers:
- Minutes projection (separate from per-minute rates)
- Back-to-back / fatigue adjustment
- Blowout risk (spread-based minutes reduction)
- Pace adjustment
- Home/away splits
- Position-specific defense matchups
- Usage redistribution (injury pop-off)
- Real correlation matrix from game logs

Edge calculation uses z-scores and normal distribution.
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
from scipy import stats as scipy_stats

from data_pipeline import (
    DataPipeline, PlayerProfile, TeamProfile, GameContext,
    PlayerProp, GameLog
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class PropType(Enum):
    """All DraftKings prop types"""
    # Single stats
    POINTS = "points"
    REBOUNDS = "rebounds"
    ASSISTS = "assists"
    THREES = "threes"
    STEALS = "steals"
    BLOCKS = "blocks"
    TURNOVERS = "turnovers"

    # Combined stats
    PRA = "pts_rebs_asts"
    PR = "pts_rebs"
    PA = "pts_asts"
    RA = "rebs_asts"
    STEALS_BLOCKS = "stls_blks"

    # Milestone props
    DOUBLE_DOUBLE = "double_double"
    TRIPLE_DOUBLE = "triple_double"


# Mapping from PropType to stat keys in our data
STAT_MAPPING = {
    PropType.POINTS: "pts",
    PropType.REBOUNDS: "reb",
    PropType.ASSISTS: "ast",
    PropType.THREES: "fg3m",
    PropType.STEALS: "stl",
    PropType.BLOCKS: "blk",
    PropType.TURNOVERS: "turnover",
}

COMBINED_STAT_MAPPING = {
    PropType.PRA: ["pts", "reb", "ast"],
    PropType.PR: ["pts", "reb"],
    PropType.PA: ["pts", "ast"],
    PropType.RA: ["reb", "ast"],
    PropType.STEALS_BLOCKS: ["stl", "blk"],
}


@dataclass
class Projection:
    """Complete projection for a player prop"""
    player_name: str
    prop_type: PropType

    # Core projection
    projected_value: float
    projected_minutes: float
    std_dev: float
    confidence: float  # 0-1

    # Market data
    dk_line: float = 0.0
    dk_over_odds: int = -110
    dk_under_odds: int = -110

    # Probability calculations
    over_probability: float = 0.5
    under_probability: float = 0.5
    implied_over_prob: float = 0.5
    implied_under_prob: float = 0.5

    # Edge
    edge_over: float = 0.0  # Positive = value on over
    edge_under: float = 0.0

    # Adjustments breakdown
    base_projection: float = 0.0
    minutes_adjustment: float = 0.0
    fatigue_adjustment: float = 0.0
    blowout_adjustment: float = 0.0
    pace_adjustment: float = 0.0
    home_away_adjustment: float = 0.0
    defense_adjustment: float = 0.0
    usage_boost: float = 0.0  # From injuries

    # Metadata
    source_layer: str = "baseline"  # baseline, popoff, correlation
    games_sample: int = 0
    role_change: str = ""  # "expanded", "reduced", or ""


@dataclass
class CorrelatedPair:
    """A pair of correlated props for SGP building"""
    player_a: str
    prop_a: PropType
    player_b: str
    prop_b: PropType
    correlation: float
    sample_size: int

    # Joint probabilities
    joint_over_prob: float = 0.0
    naive_prob: float = 0.0
    correlation_boost: float = 0.0  # How much correlation helps


# =============================================================================
# ADJUSTMENT FACTORS
# =============================================================================

class AdjustmentFactors:
    """Constants for various adjustments"""

    # Back-to-back fatigue reduction
    B2B_FATIGUE_FACTOR = 0.92  # 8% reduction on B2B

    # Blowout risk (minutes reduction based on spread)
    # Spread -> expected minutes reduction factor
    BLOWOUT_FACTORS = {
        15: 0.85,  # 15+ point favorite = 15% minutes reduction risk
        12: 0.90,
        10: 0.93,
        8: 0.96,
        5: 0.98,
        0: 1.00,
    }

    # Position defense multipliers (league average = 1.0)
    # How much a position scores against different defensive archetypes
    DEFENSE_VS_POSITION = {
        # Defensive team archetype: {position: multiplier}
        "elite": {"PG": 0.90, "SG": 0.88, "SF": 0.89, "PF": 0.91, "C": 0.87},
        "good": {"PG": 0.95, "SG": 0.94, "SF": 0.94, "PF": 0.95, "C": 0.93},
        "average": {"PG": 1.00, "SG": 1.00, "SF": 1.00, "PF": 1.00, "C": 1.00},
        "poor": {"PG": 1.05, "SG": 1.06, "SF": 1.05, "PF": 1.04, "C": 1.07},
        "bad": {"PG": 1.10, "SG": 1.12, "SF": 1.10, "PF": 1.08, "C": 1.13},
    }

    # Usage redistribution when star is out
    # What % of scratched player's stats go to each position
    USAGE_REDISTRIBUTION = {
        "PG": {"PG": 0.20, "SG": 0.25, "SF": 0.20, "PF": 0.18, "C": 0.17},
        "SG": {"PG": 0.25, "SG": 0.20, "SF": 0.22, "PF": 0.18, "C": 0.15},
        "SF": {"PG": 0.20, "SG": 0.22, "SF": 0.20, "PF": 0.22, "C": 0.16},
        "PF": {"PG": 0.18, "SG": 0.18, "SF": 0.22, "PF": 0.20, "C": 0.22},
        "C": {"PG": 0.17, "SG": 0.15, "SF": 0.18, "PF": 0.25, "C": 0.25},
    }

    # Referee tendencies (placeholder - would need real data)
    # ref_name: {pace_factor, foul_factor}
    REFEREE_PROFILES = {
        "Scott Foster": {"pace": 1.02, "fouls": 1.05},
        "Tony Brothers": {"pace": 0.98, "fouls": 1.08},
        "Marc Davis": {"pace": 1.01, "fouls": 0.97},
    }

    # League averages
    LEAGUE_AVG_PACE = 100.0
    LEAGUE_AVG_DEF_RATING = 112.0


# =============================================================================
# PROJECTION ENGINE
# =============================================================================

class ProjectionEngine:
    """
    Core projection engine.

    Calculates fair value using per-minute rates × projected minutes,
    then applies all adjustments.
    """

    # Base weights for averaging periods (favor recent form)
    WEIGHTS = {
        "last_5": 0.55,
        "last_10": 0.30,
        "season": 0.15
    }

    # Weights when role change detected (heavily favor recent)
    ROLE_CHANGE_WEIGHTS = {
        "last_5": 0.70,
        "last_10": 0.20,
        "season": 0.10
    }

    def __init__(self, pipeline: DataPipeline):
        self.pipeline = pipeline

    def detect_role_change(self, profile: PlayerProfile) -> Tuple[bool, str]:
        """
        Detect if player's role has recently changed based on minutes or scoring trend.

        Returns: (role_changed, direction)
            - role_changed: True if significant change detected
            - direction: "expanded" or "reduced" or ""
        """
        l5_min = profile.last_5_avg.get("min", 0)
        l10_min = profile.last_10_avg.get("min", 0)
        l5_pts = profile.last_5_avg.get("pts", 0)
        l10_pts = profile.last_10_avg.get("pts", 0)

        min_change = 0.0
        pts_change = 0.0

        # Calculate minutes trend
        if l10_min > 0:
            min_change = (l5_min - l10_min) / l10_min

        # Calculate scoring trend
        if l10_pts > 0:
            pts_change = (l5_pts - l10_pts) / l10_pts

        # Expanded role: 15%+ increase in minutes OR 10%+ increase in scoring
        if min_change >= 0.15 or pts_change >= 0.10:
            return True, "expanded"

        # Reduced role: 15%+ decrease in minutes OR 10%+ decrease in scoring
        if min_change <= -0.15 or pts_change <= -0.10:
            return True, "reduced"

        return False, ""

    def project_minutes(
        self,
        profile: PlayerProfile,
        is_home: bool,
        is_b2b: bool,
        spread: float,
        days_rest: int = 2
    ) -> Tuple[float, float, float, float]:
        """
        Project minutes with adjustments.

        Returns: (projected_minutes, fatigue_adj, blowout_adj, base_minutes)
        """
        # Base minutes from recent games
        base_minutes = profile.last_10_avg.get("min", 0)
        if base_minutes == 0:
            base_minutes = profile.season_avg.get("min", 28)

        projected = base_minutes
        fatigue_adj = 0.0
        blowout_adj = 0.0

        # B2B fatigue (7% reduction)
        if is_b2b or days_rest <= 1:
            reduction = base_minutes * (1 - AdjustmentFactors.B2B_FATIGUE_FACTOR)
            projected -= reduction
            fatigue_adj = -reduction
        # Extra rest bonus (light weight per user request)
        elif days_rest >= 4:
            boost = base_minutes * 0.02  # 2% boost for 4+ days rest
            projected += boost
            fatigue_adj = boost
        elif days_rest >= 3:
            boost = base_minutes * 0.01  # 1% boost for 3 days rest
            projected += boost
            fatigue_adj = boost

        # Blowout risk (large favorites)
        abs_spread = abs(spread)
        blowout_factor = 1.0
        for threshold, factor in sorted(AdjustmentFactors.BLOWOUT_FACTORS.items(), reverse=True):
            if abs_spread >= threshold:
                blowout_factor = factor
                break

        if blowout_factor < 1.0:
            reduction = projected * (1 - blowout_factor)
            projected -= reduction
            blowout_adj = -reduction

        return projected, fatigue_adj, blowout_adj, base_minutes

    def get_weighted_stat(
        self,
        profile: PlayerProfile,
        stat_key: str,
        use_role_change: bool = True
    ) -> float:
        """Get weighted average for a single stat, adjusting for role changes"""
        values = []
        weights = []

        # Detect role change and select appropriate weights
        role_changed, direction = self.detect_role_change(profile) if use_role_change else (False, "")
        weight_set = self.ROLE_CHANGE_WEIGHTS if role_changed else self.WEIGHTS

        if stat_key in profile.last_5_avg:
            values.append(profile.last_5_avg[stat_key])
            weights.append(weight_set["last_5"])

        if stat_key in profile.last_10_avg:
            values.append(profile.last_10_avg[stat_key])
            weights.append(weight_set["last_10"])

        if stat_key in profile.season_avg:
            values.append(profile.season_avg[stat_key])
            weights.append(weight_set["season"])

        if not values:
            return 0.0

        total_weight = sum(weights)
        return sum(v * w / total_weight for v, w in zip(values, weights))

    def project_stat(
        self,
        profile: PlayerProfile,
        prop_type: PropType,
        projected_minutes: float,
        is_home: bool,
        opponent_def_rating: float,
        pace_factor: float
    ) -> Tuple[float, Dict[str, float]]:
        """
        Project a single stat with all adjustments.

        Returns: (projected_value, adjustments_dict)
        """
        adjustments = {
            "base": 0.0,
            "minutes": 0.0,
            "home_away": 0.0,
            "defense": 0.0,
            "pace": 0.0,
        }

        # Handle combined stats
        if prop_type in COMBINED_STAT_MAPPING:
            return self._project_combined_stat(
                profile, prop_type, projected_minutes,
                is_home, opponent_def_rating, pace_factor
            )

        # Handle milestone props
        if prop_type in [PropType.DOUBLE_DOUBLE, PropType.TRIPLE_DOUBLE]:
            return self._project_milestone(profile, prop_type)

        # Get stat key
        stat_key = STAT_MAPPING.get(prop_type)
        if not stat_key:
            return 0.0, adjustments

        # Method 1: Use per-minute rate × projected minutes
        per_min_rate = profile.per_minute.get(stat_key, 0)
        avg_minutes = profile.last_10_avg.get("min", 30)

        if per_min_rate > 0 and avg_minutes > 0:
            # Base projection from per-minute
            base_from_rate = per_min_rate * projected_minutes
            # Also get weighted average
            weighted_avg = self.get_weighted_stat(profile, stat_key)
            # Blend them (per-minute accounts for minutes changes)
            base_projection = (base_from_rate * 0.6) + (weighted_avg * 0.4)
        else:
            base_projection = self.get_weighted_stat(profile, stat_key)

        adjustments["base"] = base_projection
        projected = base_projection

        # Minutes adjustment (difference from average minutes)
        if avg_minutes > 0 and per_min_rate > 0:
            minutes_diff = projected_minutes - avg_minutes
            minutes_adj = per_min_rate * minutes_diff * 0.5  # Dampened
            projected += minutes_adj
            adjustments["minutes"] = minutes_adj

        # Home/away adjustment
        if is_home and profile.home_avg.get(stat_key):
            home_diff = profile.home_avg[stat_key] - self.get_weighted_stat(profile, stat_key)
            home_adj = home_diff * 0.3  # Dampened
            projected += home_adj
            adjustments["home_away"] = home_adj
        elif not is_home and profile.away_avg.get(stat_key):
            away_diff = profile.away_avg[stat_key] - self.get_weighted_stat(profile, stat_key)
            away_adj = away_diff * 0.3
            projected += away_adj
            adjustments["home_away"] = away_adj

        # Defense adjustment
        def_factor = self._get_defense_factor(
            opponent_def_rating,
            profile.position,
            prop_type
        )
        defense_adj = projected * (def_factor - 1)
        projected *= def_factor
        adjustments["defense"] = defense_adj

        # Pace adjustment
        pace_adj = projected * (pace_factor - 1)
        projected *= pace_factor
        adjustments["pace"] = pace_adj

        # ANTI-OVERESTIMATION: Regress toward baseline
        # Data showed we overestimate by +2.79 on average (74.7% of picks)
        # This dampens extreme projections while keeping direction
        # Blend: 85% adjusted projection + 15% baseline
        # This prevents runaway projections while preserving edge
        base = adjustments["base"]
        if base > 0:
            regression_factor = 0.15
            projected = (projected * (1 - regression_factor)) + (base * regression_factor)

        return max(0, projected), adjustments

    def _project_combined_stat(
        self,
        profile: PlayerProfile,
        prop_type: PropType,
        projected_minutes: float,
        is_home: bool,
        opponent_def_rating: float,
        pace_factor: float
    ) -> Tuple[float, Dict[str, float]]:
        """Project combined stats (PRA, PR, etc.)"""
        stat_keys = COMBINED_STAT_MAPPING.get(prop_type, [])
        total = 0.0
        adjustments = {"base": 0.0, "minutes": 0.0, "home_away": 0.0, "defense": 0.0, "pace": 0.0}

        for stat_key in stat_keys:
            # Map stat key to PropType for individual projection
            prop_map = {v: k for k, v in STAT_MAPPING.items()}
            individual_prop = prop_map.get(stat_key)
            if individual_prop:
                val, adj = self.project_stat(
                    profile, individual_prop, projected_minutes,
                    is_home, opponent_def_rating, pace_factor
                )
                total += val
                for k in adjustments:
                    adjustments[k] += adj.get(k, 0)

        return total, adjustments

    def _project_milestone(
        self,
        profile: PlayerProfile,
        prop_type: PropType
    ) -> Tuple[float, Dict[str, float]]:
        """Project double-double/triple-double probability"""
        game_logs = profile.game_logs
        if not game_logs:
            return 0.0, {}

        milestone_count = 0
        for log in game_logs:
            cats_over_10 = sum(1 for stat in [log.pts, log.reb, log.ast, log.stl, log.blk] if stat >= 10)

            if prop_type == PropType.DOUBLE_DOUBLE and cats_over_10 >= 2:
                milestone_count += 1
            elif prop_type == PropType.TRIPLE_DOUBLE and cats_over_10 >= 3:
                milestone_count += 1

        probability = milestone_count / len(game_logs) if game_logs else 0
        return probability, {"base": probability}

    def _get_defense_factor(
        self,
        opponent_def_rating: float,
        position: str,
        prop_type: PropType
    ) -> float:
        """Get defensive adjustment factor"""
        # Categorize defense
        if opponent_def_rating <= 108:
            def_tier = "elite"
        elif opponent_def_rating <= 111:
            def_tier = "good"
        elif opponent_def_rating <= 114:
            def_tier = "average"
        elif opponent_def_rating <= 117:
            def_tier = "poor"
        else:
            def_tier = "bad"

        # Get position factor
        pos = position.upper() if position else "SF"
        if pos not in ["PG", "SG", "SF", "PF", "C"]:
            # Handle combo positions like "G" or "F"
            if "G" in pos:
                pos = "SG"
            elif "F" in pos:
                pos = "SF"
            else:
                pos = "SF"

        factor = AdjustmentFactors.DEFENSE_VS_POSITION.get(def_tier, {}).get(pos, 1.0)

        # Defense matters less for some props
        if prop_type in [PropType.TURNOVERS, PropType.STEALS]:
            factor = 1 + (factor - 1) * 0.5  # Dampen

        return factor

    def calculate_std_dev(
        self,
        profile: PlayerProfile,
        prop_type: PropType
    ) -> float:
        """Get standard deviation for a prop type"""
        if prop_type in COMBINED_STAT_MAPPING:
            # For combined stats, need to account for covariance
            stat_keys = COMBINED_STAT_MAPPING[prop_type]
            if not profile.game_logs:
                return 5.0  # Default

            combined_values = []
            for log in profile.game_logs:
                total = sum(getattr(log, k, 0) for k in stat_keys)
                combined_values.append(total)

            return float(np.std(combined_values)) if combined_values else 5.0

        stat_key = STAT_MAPPING.get(prop_type)
        if stat_key and stat_key in profile.std_dev:
            return profile.std_dev[stat_key]

        return 3.0  # Default


# =============================================================================
# EDGE CALCULATOR
# =============================================================================

class EdgeCalculator:
    """
    Calculates betting edge using probability distributions.

    Edge = Our probability - Implied probability from odds
    """

    # Calibration factor for std_dev - backtest showed we're overconfident
    # When we predicted 65%, actual hit rate was 50%
    # Increasing std_dev widens the distribution toward 50%
    STD_DEV_CALIBRATION = 2.0  # Multiply std_dev by this factor (increased from 1.6)

    # Minimum std_dev by stat type (prevents overconfidence on "consistent" players)
    # Increased floors based on backtest showing high confidence still underperforming
    MIN_STD_DEV = {
        "pts": 6.0,      # Points: raised floor (was 4.5)
        "reb": 3.5,      # Rebounds (was 2.5)
        "ast": 3.0,      # Assists (was 2.0)
        "fg3m": 1.5,     # Threes (was 1.2)
        "stl": 1.0,      # Steals (was 0.8)
        "blk": 1.0,      # Blocks (was 0.8)
        "turnover": 1.2, # Turnovers (was 1.0)
        "pra": 9.0,      # PRA combined - higher floor (was 7.0)
    }

    @staticmethod
    def american_to_prob(odds: int) -> float:
        """Convert American odds to implied probability"""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)

    @staticmethod
    def prob_to_american(prob: float) -> int:
        """Convert probability to American odds"""
        if prob >= 0.5:
            return int(-100 * prob / (1 - prob))
        else:
            return int(100 * (1 - prob) / prob)

    @classmethod
    def calculate_over_probability(
        cls,
        projection: float,
        line: float,
        std_dev: float,
        stat_type: str = "pts"
    ) -> float:
        """
        Calculate probability of going OVER the line.

        Uses normal distribution with player's projection and variance.
        P(X > line) = 1 - CDF((line - projection) / std_dev)

        CALIBRATION: Applies std_dev multiplier and minimum floors
        to prevent overconfidence (backtest showed 65% predicted -> 50% actual)
        """
        if std_dev <= 0:
            return 0.55 if projection > line else 0.45

        # Apply calibration: multiply std_dev to widen distribution
        calibrated_std = std_dev * cls.STD_DEV_CALIBRATION

        # Apply minimum floor based on stat type
        min_std = cls.MIN_STD_DEV.get(stat_type, 3.0)
        calibrated_std = max(calibrated_std, min_std)

        z_score = (line - projection) / calibrated_std
        over_prob = 1 - scipy_stats.norm.cdf(z_score)

        # Aggressive damping: pull probabilities toward 55%
        # Backtest showed 65%+ predictions hit at 41%, so we need strong damping
        # Any probability over 60% gets pulled back hard
        if over_prob > 0.60:
            # Aggressive damping: limit how far above 60% we can go
            # 70% -> 60 + (70-60)*0.3 = 63%
            # 80% -> 60 + (80-60)*0.3 = 66%
            over_prob = 0.60 + (over_prob - 0.60) * 0.3
        elif over_prob < 0.40:
            # Same aggressive damping on under side
            over_prob = 0.40 - (0.40 - over_prob) * 0.3

        # Clamp to realistic range - never claim > 68% or < 32%
        return max(0.32, min(0.68, over_prob))

    def calculate_edge(
        self,
        projection: float,
        line: float,
        std_dev: float,
        over_odds: int,
        under_odds: int,
        stat_type: str = "pts"
    ) -> Tuple[float, float, float, float]:
        """
        Calculate edge on over and under.

        Returns: (edge_over, edge_under, our_over_prob, our_under_prob)

        IMPORTANT: Backtest data showed high edge picks underperform:
        - 5-7% edge: 56.6% hit rate (good!)
        - 20-30% edge: 36.5% hit rate (disaster!)

        The market is more efficient than we think. High edge claims are overconfident.
        We cap reported edge to prevent overconfidence.
        """
        our_over_prob = self.calculate_over_probability(projection, line, std_dev, stat_type)
        our_under_prob = 1 - our_over_prob

        implied_over = self.american_to_prob(over_odds)
        implied_under = self.american_to_prob(under_odds)

        edge_over = (our_over_prob - implied_over) * 100  # As percentage
        edge_under = (our_under_prob - implied_under) * 100

        # CAP EDGE: Data showed >12% edge picks underperform
        # The market is smarter than we think - high edge = overconfidence
        # Cap at 12% to keep picks in the profitable zone
        MAX_EDGE = 12.0
        edge_over = min(edge_over, MAX_EDGE)
        edge_under = min(edge_under, MAX_EDGE)

        return edge_over, edge_under, our_over_prob, our_under_prob


# =============================================================================
# CORRELATION ENGINE
# =============================================================================

class CorrelationEngine:
    """
    Calculates real correlations between player props using game logs.

    Uses Pearson correlation on shared games.
    """

    MIN_SAMPLE_SIZE = 5
    SIGNIFICANT_CORRELATION = 0.25

    def __init__(self, pipeline: DataPipeline):
        self.pipeline = pipeline
        self._cache: Dict[str, float] = {}

    def calculate_correlation(
        self,
        profile_a: PlayerProfile,
        stat_a: str,
        profile_b: PlayerProfile,
        stat_b: str
    ) -> Tuple[float, int]:
        """
        Calculate Pearson correlation between two players' stats.

        Returns: (correlation, sample_size)
        """
        cache_key = f"{profile_a.id}_{stat_a}_{profile_b.id}_{stat_b}"
        if cache_key in self._cache:
            return self._cache[cache_key], 10  # Cached

        # Get game logs
        logs_a = profile_a.game_logs
        logs_b = profile_b.game_logs

        if not logs_a or not logs_b:
            return 0.0, 0

        # Build game_id -> value maps (only games where player actually played)
        a_by_game = {}
        for log in logs_a:
            if log.minutes > 0:  # Filter out DNP games
                a_by_game[log.game_id] = getattr(log, stat_a, 0)

        b_by_game = {}
        for log in logs_b:
            if log.minutes > 0:  # Filter out DNP games
                b_by_game[log.game_id] = getattr(log, stat_b, 0)

        # Find shared games where both played
        shared_games = set(a_by_game.keys()) & set(b_by_game.keys())

        if len(shared_games) < self.MIN_SAMPLE_SIZE:
            return 0.0, len(shared_games)

        a_values = [a_by_game[g] for g in shared_games]
        b_values = [b_by_game[g] for g in shared_games]

        # Check for zero variance (would cause NaN)
        if np.std(a_values) == 0 or np.std(b_values) == 0:
            return 0.0, len(shared_games)

        try:
            corr, _ = scipy_stats.pearsonr(a_values, b_values)
            if np.isnan(corr):
                return 0.0, len(shared_games)
            self._cache[cache_key] = corr
            return round(corr, 3), len(shared_games)
        except Exception:
            return 0.0, 0

    def find_correlated_pairs(
        self,
        players: List[PlayerProfile],
        prop_types: List[PropType] = None
    ) -> List[CorrelatedPair]:
        """Find all significantly correlated pairs among players"""
        if prop_types is None:
            prop_types = [PropType.POINTS, PropType.REBOUNDS, PropType.ASSISTS, PropType.THREES, PropType.PRA]

        pairs = []
        seen = set()

        for i, player_a in enumerate(players):
            for player_b in players[i+1:]:
                # Only same team
                if player_a.team != player_b.team:
                    continue

                for prop_a in prop_types:
                    for prop_b in prop_types:
                        key = f"{player_a.id}_{prop_a}_{player_b.id}_{prop_b}"
                        if key in seen:
                            continue
                        seen.add(key)

                        stat_a = STAT_MAPPING.get(prop_a)
                        stat_b = STAT_MAPPING.get(prop_b)

                        if not stat_a or not stat_b:
                            continue

                        corr, sample = self.calculate_correlation(
                            player_a, stat_a,
                            player_b, stat_b
                        )

                        if abs(corr) >= self.SIGNIFICANT_CORRELATION and sample >= self.MIN_SAMPLE_SIZE:
                            pairs.append(CorrelatedPair(
                                player_a=player_a.name,
                                prop_a=prop_a,
                                player_b=player_b.name,
                                prop_b=prop_b,
                                correlation=corr,
                                sample_size=sample
                            ))

        return pairs

    def calculate_joint_probability(
        self,
        prob_a: float,
        prob_b: float,
        correlation: float
    ) -> float:
        """
        Calculate P(A AND B) accounting for correlation.

        P(A ∩ B) ≈ P(A) × P(B) + ρ × σ_A × σ_B

        Where σ = sqrt(p × (1-p)) for binary events.
        """
        if correlation == 0:
            return prob_a * prob_b

        sigma_a = math.sqrt(prob_a * (1 - prob_a))
        sigma_b = math.sqrt(prob_b * (1 - prob_b))

        joint = prob_a * prob_b + correlation * sigma_a * sigma_b
        return max(0.01, min(0.99, joint))


# =============================================================================
# POP-OFF HUNTER
# =============================================================================

class PopOffHunter:
    """
    Identifies players with inflated projections due to injuries.

    When a star is out, usage redistributes to teammates.
    """

    POPOFF_THRESHOLD = 0.15  # 15% boost = pop-off candidate

    def __init__(self, pipeline: DataPipeline):
        self.pipeline = pipeline

    def calculate_usage_boost(
        self,
        beneficiary: PlayerProfile,
        scratched: PlayerProfile,
        prop_type: PropType
    ) -> float:
        """
        Calculate percentage boost when a teammate is out.

        Returns a MULTIPLIER (e.g., 1.08 = 8% boost), not raw points.
        This is more stable and prevents the crazy projections we were seeing.

        Key insight from backtest: High boosts (>1.10) actually hurt performance.
        The market already prices in injuries. Our edge is in being more accurate
        about WHICH players benefit and by how much.
        """
        scratched_pos = scratched.position.upper() if scratched.position else "SF"
        beneficiary_pos = beneficiary.position.upper() if beneficiary.position else "SF"

        # Normalize positions
        if scratched_pos not in ["PG", "SG", "SF", "PF", "C"]:
            scratched_pos = "SG" if "G" in scratched_pos else "SF"
        if beneficiary_pos not in ["PG", "SG", "SF", "PF", "C"]:
            beneficiary_pos = "SG" if "G" in beneficiary_pos else "SF"

        # Get redistribution factor based on position compatibility
        redistribution = AdjustmentFactors.USAGE_REDISTRIBUTION.get(scratched_pos, {})
        base_factor = redistribution.get(beneficiary_pos, 0.10)

        # Get stat key
        stat_key = STAT_MAPPING.get(prop_type)
        if not stat_key:
            return 1.0  # No boost (multiplier of 1)

        # Get scratched player's production
        scratched_avg = scratched.last_10_avg.get(stat_key, 0)
        if scratched_avg == 0:
            scratched_avg = scratched.season_avg.get(stat_key, 0)

        # Get beneficiary's baseline for this stat
        beneficiary_avg = beneficiary.last_10_avg.get(stat_key, 0)
        if beneficiary_avg == 0:
            beneficiary_avg = beneficiary.season_avg.get(stat_key, 0)

        if beneficiary_avg == 0:
            return 1.0  # Can't calculate boost without baseline

        # Calculate boost as percentage of beneficiary's baseline
        # If star scoring 25ppg is out and I average 15ppg:
        # I might get 25 * 0.20 = 5 extra points = 5/15 = 33% boost
        # But that's way too aggressive. Scale it down.
        raw_boost_pct = (scratched_avg * base_factor) / beneficiary_avg

        # Apply usage weight - high usage players absorb more
        # But cap at 1.2x to prevent runaway boosts
        usage_weight = 1.0
        if beneficiary.usage_pct > 0:
            usage_weight = min(1.2, 0.8 + (beneficiary.usage_pct / 50))

        # Calculate final boost percentage
        boost_pct = raw_boost_pct * usage_weight

        # Convert to multiplier (e.g., 0.08 boost -> 1.08 multiplier)
        # Cap individual injury boost at 8% (data showed >10% hurts performance)
        capped_boost = min(0.08, boost_pct)

        return 1.0 + capped_boost

    def _get_position_group(self, position: str) -> str:
        """
        Map position to broader group for scarcity calculation.
        Groups: BIG (C, PF), WING (SF, SG), GUARD (PG)
        """
        pos = position.upper() if position else "SF"
        if pos in ["C", "PF"] or "C" in pos:
            return "BIG"
        elif pos in ["SF", "SG", "F"] or "F" in pos:
            return "WING"
        elif pos in ["PG", "G"]:
            return "GUARD"
        return "WING"  # Default

    def _calculate_position_scarcity(
        self,
        active_players: List[PlayerProfile],
        scratched_players: List[PlayerProfile],
        team: str
    ) -> Dict[str, float]:
        """
        Calculate scarcity multiplier for each position group on a team.

        If 3 bigs are out and 1 is active, the active big gets 3x multiplier.
        Returns: {position_group: scarcity_multiplier}
        """
        # Count by position group for this team
        active_by_group = {"BIG": 0, "WING": 0, "GUARD": 0}
        scratched_by_group = {"BIG": 0, "WING": 0, "GUARD": 0}

        for p in active_players:
            if p.team == team:
                group = self._get_position_group(p.position)
                active_by_group[group] += 1

        for p in scratched_players:
            if p.team == team:
                group = self._get_position_group(p.position)
                scratched_by_group[group] += 1

        # Calculate scarcity multiplier
        # More scratched + fewer active = higher multiplier
        # Use sqrt for softer scaling - a player can only absorb so much
        scarcity = {}
        for group in ["BIG", "WING", "GUARD"]:
            active = max(1, active_by_group[group])  # Avoid division by zero
            scratched = scratched_by_group[group]

            if scratched == 0:
                scarcity[group] = 1.0
            else:
                # Softer formula using sqrt to avoid unrealistic projections
                # e.g., 4 scratched, 1 active = 1 + sqrt(4/1) = 3.0x (was 5.0x)
                # e.g., 2 scratched, 2 active = 1 + sqrt(2/2) = 2.0x (was 2.0x)
                # Cap at 2.5x to stay realistic (one player can only do so much)
                scarcity[group] = min(2.5, 1.0 + math.sqrt(scratched / active))

        return scarcity

    def find_popoff_candidates(
        self,
        active_players: List[PlayerProfile],
        scratched_players: List[PlayerProfile],
        prop_types: List[PropType] = None
    ) -> Dict[str, Dict[PropType, float]]:
        """
        Find all pop-off opportunities.

        Returns: {player_name: {prop_type: usage_boost_multiplier}}

        The boost is now a MULTIPLIER (e.g., 1.12 = 12% boost) not raw points.
        Multiple injuries combine multiplicatively but are capped at 1.15 total.
        """
        if prop_types is None:
            prop_types = [PropType.POINTS, PropType.REBOUNDS, PropType.ASSISTS, PropType.THREES, PropType.PRA]

        # Pre-calculate position scarcity for each team
        teams = set(p.team for p in active_players)
        team_scarcity = {}
        for team in teams:
            team_scarcity[team] = self._calculate_position_scarcity(
                active_players, scratched_players, team
            )

        boosts = {}

        for active in active_players:
            player_boosts = {}

            # Get position scarcity multiplier for this player
            pos_group = self._get_position_group(active.position)
            scarcity_mult = team_scarcity.get(active.team, {}).get(pos_group, 1.0)

            for prop_type in prop_types:
                # Start with base multiplier of 1.0 (no boost)
                combined_multiplier = 1.0

                for scratched in scratched_players:
                    # Only same team
                    if active.team != scratched.team:
                        continue

                    # Get boost multiplier for this injury
                    boost_mult = self.calculate_usage_boost(active, scratched, prop_type)

                    if boost_mult > 1.0:
                        # Combine multipliers: (1.05) * (1.03) = 1.0815
                        # This is more realistic than adding raw points
                        combined_multiplier *= boost_mult

                if combined_multiplier > 1.0:
                    # Apply position scarcity as additional boost on the margin
                    # e.g., 1.10 boost with 1.5x scarcity = 1.0 + (0.10 * 1.5) = 1.15
                    boost_portion = combined_multiplier - 1.0
                    scarcity_adjusted = 1.0 + (boost_portion * min(1.5, scarcity_mult))

                    # Cap total boost at 15% - data showed higher hurts performance
                    # The 1.05-1.10 range had best hit rate (65%)
                    final_multiplier = min(1.15, scarcity_adjusted)

                    player_boosts[prop_type] = final_multiplier

            if player_boosts:
                boosts[active.name] = player_boosts

        return boosts


# =============================================================================
# MODEL ENGINE - MAIN ORCHESTRATOR
# =============================================================================

class ModelEngine:
    """
    Main orchestrator combining all analysis layers.

    Usage:
        engine = ModelEngine(pipeline)
        projections = engine.analyze_game(context, player_names)
    """

    def __init__(self, pipeline: DataPipeline):
        self.pipeline = pipeline
        self.projection_engine = ProjectionEngine(pipeline)
        self.edge_calculator = EdgeCalculator()
        self.correlation_engine = CorrelationEngine(pipeline)
        self.popoff_hunter = PopOffHunter(pipeline)

    def create_projection(
        self,
        profile: PlayerProfile,
        prop_type: PropType,
        context: GameContext,
        usage_boost: float = 1.0
    ) -> Projection:
        """Create a full projection for a player prop

        Args:
            usage_boost: Multiplier for injury opportunity (1.0 = no boost, 1.10 = 10% boost)
        """

        # Determine if home/away
        is_home = profile.team == context.home_team.name

        # Get opponent info
        opponent = context.away_team if is_home else context.home_team

        # Project minutes
        is_b2b = context.home_team.is_back_to_back if is_home else context.away_team.is_back_to_back
        days_rest = context.home_team.days_rest if is_home else context.away_team.days_rest
        spread = context.spread if is_home else -context.spread

        proj_minutes, fatigue_adj, blowout_adj, base_min = self.projection_engine.project_minutes(
            profile, is_home, is_b2b, spread, days_rest
        )

        # Calculate pace factor
        team_pace = context.home_team.pace if is_home else context.away_team.pace
        opp_pace = context.away_team.pace if is_home else context.home_team.pace
        expected_pace = (team_pace + opp_pace) / 2
        pace_factor = expected_pace / AdjustmentFactors.LEAGUE_AVG_PACE

        # Project the stat
        projected_value, adjustments = self.projection_engine.project_stat(
            profile, prop_type, proj_minutes, is_home,
            opponent.defensive_rating, pace_factor
        )

        # Apply usage boost from injuries as a MULTIPLIER
        # e.g., usage_boost=1.10 means 10% increase in projection
        if usage_boost > 1.0:
            projected_value *= usage_boost

        # Get std dev
        std_dev = self.projection_engine.calculate_std_dev(profile, prop_type)

        # Detect role change
        _, role_direction = self.projection_engine.detect_role_change(profile)

        # Create projection
        # Store usage_boost as the multiplier (1.10 for 10% boost)
        # source_layer is "popoff" if we applied any injury boost
        has_boost = usage_boost > 1.0
        proj = Projection(
            player_name=profile.name,
            prop_type=prop_type,
            projected_value=round(projected_value, 2),
            projected_minutes=round(proj_minutes, 1),
            std_dev=round(std_dev, 2),
            confidence=self._calculate_confidence(profile, prop_type),
            base_projection=adjustments.get("base", 0),
            minutes_adjustment=round(adjustments.get("minutes", 0) + fatigue_adj + blowout_adj, 2),
            fatigue_adjustment=round(fatigue_adj, 2),
            blowout_adjustment=round(blowout_adj, 2),
            pace_adjustment=round(adjustments.get("pace", 0), 2),
            home_away_adjustment=round(adjustments.get("home_away", 0), 2),
            defense_adjustment=round(adjustments.get("defense", 0), 2),
            usage_boost=round(usage_boost, 3),  # Now a multiplier like 1.10
            source_layer="popoff" if has_boost else "baseline",
            games_sample=len(profile.game_logs),
            role_change=role_direction
        )

        return proj

    def add_market_data(
        self,
        projection: Projection,
        prop: PlayerProp
    ) -> Projection:
        """Add market data and calculate edge"""
        projection.dk_line = prop.line
        projection.dk_over_odds = prop.over_odds
        projection.dk_under_odds = prop.under_odds

        # Determine stat_type for calibration
        stat_type = STAT_MAPPING.get(projection.prop_type, "pts")
        if projection.prop_type == PropType.PRA:
            stat_type = "pra"

        # Calculate probabilities and edge
        edge_over, edge_under, our_over, our_under = self.edge_calculator.calculate_edge(
            projection.projected_value,
            prop.line,
            projection.std_dev,
            prop.over_odds,
            prop.under_odds,
            stat_type
        )

        projection.over_probability = round(our_over, 3)
        projection.under_probability = round(our_under, 3)
        projection.implied_over_prob = round(self.edge_calculator.american_to_prob(prop.over_odds), 3)
        projection.implied_under_prob = round(self.edge_calculator.american_to_prob(prop.under_odds), 3)
        projection.edge_over = round(edge_over, 2)
        projection.edge_under = round(edge_under, 2)

        return projection

    def _calculate_confidence(
        self,
        profile: PlayerProfile,
        prop_type: PropType
    ) -> float:
        """Calculate confidence score (0-1)"""
        # Factors: sample size, variance, consistency
        sample_size = len(profile.game_logs)

        # Sample size factor (max at 15 games)
        sample_factor = min(1.0, sample_size / 15)

        # Variance factor (lower variance = higher confidence)
        std_dev = self.projection_engine.calculate_std_dev(profile, prop_type)
        stat_key = STAT_MAPPING.get(prop_type, "pts")
        mean_val = profile.last_10_avg.get(stat_key, 10)

        if mean_val > 0:
            cv = std_dev / mean_val  # Coefficient of variation
            variance_factor = max(0.3, 1 - cv)
        else:
            variance_factor = 0.5

        confidence = sample_factor * 0.6 + variance_factor * 0.4
        return round(min(1.0, max(0.1, confidence)), 2)

    def analyze_game(
        self,
        context: GameContext,
        player_names: List[str],
        prop_types: List[PropType] = None,
        injured_player_names: List[str] = None
    ) -> Dict:
        """
        Full game analysis.

        Args:
            context: Game context with teams, props, etc.
            player_names: Names of active players to analyze
            prop_types: Which prop types to project
            injured_player_names: Names of injured (OUT) players for pop-off boosts

        Returns:
        {
            "projections": [Projection, ...],
            "correlations": [CorrelatedPair, ...],
            "injuries": [...],
            "context": {...}
        }
        """
        if prop_types is None:
            prop_types = list(PropType)

        results = {
            "game_id": context.game_id,
            "matchup": f"{context.away_team.name} @ {context.home_team.name}",
            "spread": context.spread,
            "total": context.total,
            "projections": [],
            "correlations": [],
            "popoff_boosts": {},
            "injuries": context.injuries,
        }

        # Build active player profiles
        profiles = []
        for name in player_names:
            profile = self.pipeline.build_player_profile(name)
            if profile:
                profiles.append(profile)
                context.players[name] = profile

        active = [p for p in profiles if p.status not in ["out", "doubtful"]]

        # Build injured player profiles for pop-off analysis
        scratched = []
        if injured_player_names:
            for name in injured_player_names:
                profile = self.pipeline.build_player_profile(name)
                if profile:
                    scratched.append(profile)

        # Calculate pop-off boosts from injured players
        popoff_boosts = self.popoff_hunter.find_popoff_candidates(
            active, scratched, prop_types
        )
        results["popoff_boosts"] = popoff_boosts

        # Log pop-off boosts for debugging
        if popoff_boosts:
            for player, boosts in popoff_boosts.items():
                for prop_type, boost in boosts.items():
                    if boost > 1.05:  # Log significant boosts (>5%)
                        boost_pct = (boost - 1.0) * 100
                        logger.info(f"Pop-off: {player} +{boost_pct:.0f}% {prop_type.value} from injuries")

        # Create projections for each player/prop
        for profile in active:
            for prop_type in prop_types:
                # Get usage boost multiplier if applicable (default 1.0 = no boost)
                usage_boost = popoff_boosts.get(profile.name, {}).get(prop_type, 1.0)

                projection = self.create_projection(
                    profile, prop_type, context, usage_boost
                )

                # Find matching DK prop
                for prop in context.props:
                    prop_name = prop.player_name.lower()
                    profile_name = profile.name.lower()

                    # Match player name and prop type
                    if prop_name in profile_name or profile_name in prop_name:
                        if self._match_prop_type(prop.prop_type, prop_type):
                            projection = self.add_market_data(projection, prop)
                            break

                results["projections"].append(projection)

        # Find correlations
        results["correlations"] = self.correlation_engine.find_correlated_pairs(
            active,
            [PropType.POINTS, PropType.REBOUNDS, PropType.ASSISTS, PropType.THREES, PropType.PRA]
        )

        return results

    def _match_prop_type(self, dk_prop_type: str, our_prop_type: PropType) -> bool:
        """Match DraftKings prop type string to our PropType enum"""
        dk_lower = dk_prop_type.lower()

        matches = {
            PropType.POINTS: ["points", "pts"],
            PropType.REBOUNDS: ["rebounds", "rebs", "reb"],
            PropType.ASSISTS: ["assists", "asts", "ast"],
            PropType.THREES: ["threes", "3pt", "3-pointers", "three"],
            PropType.STEALS: ["steals", "stl"],
            PropType.BLOCKS: ["blocks", "blk"],
            PropType.TURNOVERS: ["turnovers", "to"],
            PropType.PRA: ["pts_rebs_asts", "pra", "points_rebounds_assists"],
            PropType.PR: ["pts_rebs", "points_rebounds"],
            PropType.PA: ["pts_asts", "points_assists"],
            PropType.RA: ["rebs_asts", "rebounds_assists"],
            PropType.STEALS_BLOCKS: ["stls_blks", "steals_blocks"],
            PropType.DOUBLE_DOUBLE: ["double_double", "double-double"],
            PropType.TRIPLE_DOUBLE: ["triple_double", "triple-double"],
        }

        for term in matches.get(our_prop_type, []):
            if term in dk_lower:
                return True
        return False


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    API_KEY = os.getenv("BALLDONTLIE_API_KEY")
    if not API_KEY:
        print("Error: BALLDONTLIE_API_KEY not found")
        exit(1)

    # Initialize
    pipeline = DataPipeline(API_KEY)
    engine = ModelEngine(pipeline)

    # Demo: Show edge calculation
    print("\n=== Edge Calculation Demo ===\n")

    calc = EdgeCalculator()

    # Example: Player projects to 26.5 pts, line is 24.5, std_dev is 5.5
    projection = 26.5
    line = 24.5
    std_dev = 5.5

    over_prob = calc.calculate_over_probability(projection, line, std_dev)
    edge_over, edge_under, _, _ = calc.calculate_edge(
        projection, line, std_dev, -115, -105
    )

    print(f"Projection: {projection}")
    print(f"DK Line: {line}")
    print(f"Std Dev: {std_dev}")
    print(f"Z-Score: {(line - projection) / std_dev:.3f}")
    print(f"Over Probability: {over_prob*100:.1f}%")
    print(f"Implied (at -115): {calc.american_to_prob(-115)*100:.1f}%")
    print(f"Edge on Over: {edge_over:.1f}%")
    print(f"Edge on Under: {edge_under:.1f}%")

    # Demo: Correlation math
    print("\n=== Correlation Math Demo ===\n")

    corr_engine = CorrelationEngine(pipeline)

    prob_a = 0.55  # 55% chance player A hits over
    prob_b = 0.52  # 52% chance player B hits over
    correlation = 0.45  # Positive correlation (teammates)

    naive = prob_a * prob_b
    joint = corr_engine.calculate_joint_probability(prob_a, prob_b, correlation)

    print(f"Player A over prob: {prob_a*100:.0f}%")
    print(f"Player B over prob: {prob_b*100:.0f}%")
    print(f"Correlation: {correlation}")
    print(f"Naive parlay prob: {naive*100:.1f}%")
    print(f"True parlay prob: {joint*100:.1f}%")
    print(f"Correlation boost: +{(joint-naive)*100:.1f}%")
    print(f"\nNaive fair odds: +{calc.prob_to_american(naive)}")
    print(f"True fair odds: +{calc.prob_to_american(joint)}")
