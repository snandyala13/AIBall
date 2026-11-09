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

import json
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
from scipy import stats as scipy_stats

from data_pipeline import (
    DataPipeline, PlayerProfile, TeamProfile, GameContext,
    PlayerProp, GameLog, TrackingStats, PBPStats, CTGStats,
    ProjectionFeatures, CTGTeamClient, CTGLineupClient, CleaningTheGlassClient
)
from lineup_impact import LineupImpactModel
from minutes_model import MinutesPredictor

# ML model imports
import os
import joblib

from quantile_engine import QuantileEngine

# Minimum game logs for full 70/30 quantile/z-score blend weight
FULL_BLEND_GAMES = 15

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ML PROJECTION ENGINE
# =============================================================================

class MLProjectionEngine:
    """
    ML-driven projection engine using stacked ensemble models.

    Uses LightGBM + XGBoost + CatBoost → Ridge meta-learner for each prop type.
    Falls back to rule-based projections if ML fails or model unavailable.
    """

    MODEL_DIR = "models"

    # Map prop types to model filenames
    PROP_TO_MODEL = {
        "points": "points_ensemble_model.joblib",
        "rebounds": "rebounds_ensemble_model.joblib",
        "assists": "assists_ensemble_model.joblib",
        "threes": "threes_ensemble_model.joblib",
        "steals": "steals_ensemble_model.joblib",
        "blocks": "blocks_ensemble_model.joblib",
        "turnovers": "turnovers_ensemble_model.joblib",
    }

    def __init__(self, pipeline: DataPipeline):
        self.pipeline = pipeline
        self.models = {}
        self.ctg_client = None
        self.ctg_team_client = None
        self.ctg_lineup_client = None
        self._load_models()
        self._init_data_clients()

    def _load_models(self):
        """Load trained ensemble models from disk."""
        for prop_type, filename in self.PROP_TO_MODEL.items():
            filepath = os.path.join(self.MODEL_DIR, filename)
            if os.path.exists(filepath):
                try:
                    self.models[prop_type] = joblib.load(filepath)
                    logger.info(f"Loaded ML model: {prop_type}")
                except Exception as e:
                    logger.warning(f"Failed to load {prop_type} model: {e}")

        logger.info(f"Loaded {len(self.models)} ML models")

    def _init_data_clients(self):
        """Initialize CTG data clients for feature building."""
        try:
            self.ctg_client = CleaningTheGlassClient()
            self.ctg_team_client = CTGTeamClient()
            self.ctg_lineup_client = CTGLineupClient()
        except Exception as e:
            logger.warning(f"Failed to initialize CTG clients: {e}")

    def build_features(
        self,
        profile: PlayerProfile,
        prop_type: str,
        context: GameContext,
        injury_boost: float = 1.0
    ) -> ProjectionFeatures:
        """
        Build ProjectionFeatures from player profile and game context.

        This creates the ~66 feature vector needed by the ML model.
        """
        features = ProjectionFeatures()

        # Determine stat key for this prop type
        stat_mapping = {
            "points": "pts", "rebounds": "reb", "assists": "ast",
            "threes": "fg3m", "steals": "stl", "blocks": "blk", "turnovers": "turnover"
        }
        stat_key = stat_mapping.get(prop_type, "pts")

        # === BASE STATS ===
        features.l3_avg = profile.last_3_avg.get(stat_key, 0)
        features.l5_avg = profile.last_5_avg.get(stat_key, 0)
        features.l10_avg = profile.last_10_avg.get(stat_key, 0)
        features.season_avg = profile.season_avg.get(stat_key, 0)
        features.std_dev = profile.std_dev.get(stat_key, 3.0)
        features.per_minute = profile.per_minute.get(stat_key, 0)

        # === MINUTES CONTEXT ===
        features.proj_minutes = profile.last_10_avg.get("min", 30)
        features.avg_minutes_l5 = profile.last_5_avg.get("min", 30)
        l3_min = profile.last_3_avg.get("min", 30)
        l10_min = profile.last_10_avg.get("min", 30)
        if l10_min > 0:
            features.minutes_trend = (l3_min - l10_min) / l10_min

        # === GAME CONTEXT ===
        is_home = profile.team == context.home_team.name
        features.is_home = is_home

        opponent = context.away_team if is_home else context.home_team
        features.opp_def_rating = opponent.defensive_rating

        # Pace factor
        team_pace = context.home_team.pace if is_home else context.away_team.pace
        opp_pace = context.away_team.pace if is_home else context.home_team.pace
        features.pace_factor = ((team_pace + opp_pace) / 2) / 100.0

        features.spread = context.spread if is_home else -context.spread
        features.total = context.total

        team_profile = context.home_team if is_home else context.away_team
        features.is_b2b = team_profile.is_back_to_back
        features.days_rest = team_profile.days_rest

        # === CTG PLAYER STATS ===
        if self.ctg_client and profile.ctg:
            ctg = profile.ctg
            features.ctg_psa_percentile = ctg.psa_percentile
            features.ctg_usage_percentile = ctg.usage_percentile
            features.ctg_ast_percentile = ctg.ast_percentile
            features.ctg_ast_to_usage = ctg.ast_to_usage
            features.ctg_tov_percentile = ctg.tov_percentile
            features.has_ctg = True

        # === CTG TEAM STATS ===
        if self.ctg_team_client:
            try:
                # Own team stats
                team_stats = self.ctg_team_client.get_team_stats(profile.team)
                if team_stats:
                    features.team_off_efg_rank = team_stats.off_efg_rank
                    features.team_off_tov_pct = team_stats.off_tov_pct
                    features.team_transition_freq = team_stats.transition_freq
                    features.team_transition_pts_play = team_stats.transition_pts_play

                # Opponent defense stats
                opp_team = opponent.name
                opp_stats = self.ctg_team_client.get_team_stats(opp_team)
                if opp_stats:
                    features.opp_def_efg_rank = opp_stats.def_efg_rank
                    features.opp_def_tov_forced_pct = opp_stats.def_tov_pct
                    features.opp_def_orb_allowed_pct = opp_stats.def_orb_pct
                    features.opp_def_transition_pts = opp_stats.def_transition_pts
                    features.opp_def_halfcourt_pts = opp_stats.def_halfcourt_pts
                    features.has_ctg_team = True
            except Exception as e:
                logger.debug(f"CTG team stats error: {e}")

        # === CTG LINEUP STATS ===
        if self.ctg_lineup_client:
            try:
                lineup = self.ctg_lineup_client.get_best_lineup_for_player(profile.name)
                if lineup:
                    features.lineup_net_rating = lineup.net_rating
                    features.lineup_off_efg = lineup.off_efg_pct
                    features.lineup_off_pts_poss = lineup.off_pts_poss
                    features.lineup_def_pts_poss = lineup.def_pts_poss
                    features.lineup_possessions = lineup.possessions
                    features.lineup_found = True
            except Exception as e:
                logger.debug(f"CTG lineup stats error: {e}")

        # === INJURY/ROLE CONTEXT ===
        features.injury_boost = injury_boost

        # Detect role change
        l3_pts = profile.last_3_avg.get("pts", 0)
        l10_pts = profile.last_10_avg.get("pts", 0)
        if l10_pts > 0:
            pts_change = (l3_pts - l10_pts) / l10_pts
            if pts_change >= 0.10:
                features.role_change = 1  # Expanded
            elif pts_change <= -0.10:
                features.role_change = -1  # Reduced

        return features

    def predict(
        self,
        profile: PlayerProfile,
        prop_type: str,
        context: GameContext,
        injury_boost: float = 1.0
    ) -> Optional[float]:
        """
        Get ML prediction for a player prop.

        Returns None if model unavailable or prediction fails.
        """
        if prop_type not in self.models:
            return None

        try:
            # Build features
            features = self.build_features(profile, prop_type, context, injury_boost)
            feature_array = features.to_array().reshape(1, -1)

            # Get ensemble prediction
            ensemble = self.models[prop_type]
            base_preds = np.column_stack([
                ensemble['base_models'][name].predict(feature_array)
                for name in ['lgbm', 'xgb', 'catboost']
            ])
            prediction = ensemble['meta_model'].predict(base_preds)[0]

            # Sanity check - ensure prediction is reasonable
            season_avg = profile.season_avg.get(
                {"points": "pts", "rebounds": "reb", "assists": "ast",
                 "threes": "fg3m", "steals": "stl", "blocks": "blk", "turnovers": "turnover"}.get(prop_type, "pts"),
                10
            )

            # Cap at 2.5x season average to prevent runaway predictions
            max_val = max(season_avg * 2.5, season_avg + 15)
            prediction = min(prediction, max_val)

            return max(0, prediction)

        except Exception as e:
            logger.warning(f"ML prediction failed for {profile.name} {prop_type}: {e}")
            return None

    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        info = {
            "models_loaded": list(self.models.keys()),
            "model_count": len(self.models),
        }

        # Add test accuracy from metadata if available
        for prop_type, ensemble in self.models.items():
            if isinstance(ensemble, dict):
                info[f"{prop_type}_test_mae"] = ensemble.get("test_mae")
                info[f"{prop_type}_accuracy"] = ensemble.get("accuracy")

        return info


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
    team: str = ""  # Player's team (for parlay correlation)
    dk_line: float = 0.0
    dk_over_odds: int = -110
    dk_under_odds: int = -110
    # Cross-book data (best available line across all sportsbooks)
    best_over_book: str = "draftkings"
    best_under_book: str = "draftkings"
    all_books: Dict = field(default_factory=dict)

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
    tracking_adjustment: float = 0.0  # From NBA.com tracking stats
    pbp_adjustment: float = 0.0  # From PBPStats.com
    ctg_adjustment: float = 0.0  # From Cleaning the Glass (percentile context)
    lineup_adjustment: float = 0.0  # From lineup context (ON/OFF splits)
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

    # Usage redistribution when a player is out — depth-chart aware
    # Real NBA depth charts: minutes go to the SAME position group
    # Guards (PG/SG) → other guards. Bigs (PF/C) → other bigs.
    # Wings (SF) bridge both groups slightly.
    # Cross-group: near zero. Kornet out does NOT help Pritchard.
    USAGE_REDISTRIBUTION = {
        "PG": {"PG": 0.40, "SG": 0.35, "SF": 0.15, "PF": 0.05, "C": 0.00},
        "SG": {"PG": 0.35, "SG": 0.35, "SF": 0.20, "PF": 0.05, "C": 0.00},
        "SF": {"PG": 0.10, "SG": 0.15, "SF": 0.30, "PF": 0.25, "C": 0.15},
        "PF": {"PG": 0.00, "SG": 0.05, "SF": 0.20, "PF": 0.35, "C": 0.35},
        "C":  {"PG": 0.00, "SG": 0.00, "SF": 0.10, "PF": 0.40, "C": 0.45},
    }

    # League averages
    LEAGUE_AVG_PACE = 100.0
    LEAGUE_AVG_DEF_RATING = 113.0
    LEAGUE_AVG_TOTAL = 224.0  # ~112 points per team


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
    # L3 weighted highest for recency, then L5, L10, season
    WEIGHTS = {
        "last_3": 0.40,
        "last_5": 0.30,
        "last_10": 0.20,
        "season": 0.10
    }

    # Weights when role change detected (heavily favor most recent)
    ROLE_CHANGE_WEIGHTS = {
        "last_3": 0.55,
        "last_5": 0.30,
        "last_10": 0.10,
        "season": 0.05
    }

    def __init__(self, pipeline: DataPipeline):
        self.pipeline = pipeline
        self.adj_config = self._load_adjustment_config()
        self.minutes_predictor = MinutesPredictor()

    @staticmethod
    def get_game_environment_factor(game_total: float) -> float:
        """
        Scale stats based on Vegas total (over/under).

        The Vegas total is the market's best estimate of total scoring.
        It encodes pace, defense quality, and game script expectations.

        A 235-point game is ~5% above average -> everyone's stats up ~5%.
        A 210-point game is ~6% below average -> everyone's stats down ~6%.
        """
        if game_total <= 0:
            return 1.0
        return game_total / AdjustmentFactors.LEAGUE_AVG_TOTAL

    def project_points_from_possessions(
        self,
        profile: PlayerProfile,
        projected_minutes: float,
        game_pace: float,
        usage_pct: float,
    ) -> float:
        """
        Model points from expected possessions rather than per-minute rates.

        Points = possessions x points_per_possession

        possessions = game_pace * (minutes / 48) * (usage_rate / 100)
        PPP = historical points / historical possessions (from game logs)
        """
        if usage_pct <= 0:
            # Fallback to per-minute if no usage data
            return profile.per_minute.get("pts", 0) * projected_minutes

        # Expected possessions for this player in this game
        player_possessions = game_pace * (projected_minutes / 48.0) * (usage_pct / 100.0)

        # Points per possession from game history
        ppp = self._calculate_player_ppp(profile)

        return player_possessions * ppp

    def _calculate_player_ppp(self, profile: PlayerProfile) -> float:
        """
        Calculate a player's points per possession from game logs.

        Uses the possession estimate: game_pace * (minutes / 48) * (usage / 100).
        Falls back to a ratio derived from season averages if game logs are sparse.
        """
        if not profile.game_logs or profile.usage_pct <= 0:
            # Fallback: estimate from season averages
            avg_min = profile.season_avg.get("min", 30)
            avg_pts = profile.season_avg.get("pts", 0)
            if avg_min > 0 and profile.usage_pct > 0:
                est_poss = AdjustmentFactors.LEAGUE_AVG_PACE * (avg_min / 48.0) * (profile.usage_pct / 100.0)
                if est_poss > 0:
                    return avg_pts / est_poss
            return 1.0  # League-average PPP fallback

        total_pts = 0.0
        total_poss = 0.0

        for log in profile.game_logs:
            if log.minutes <= 0:
                continue
            # Estimate possessions for this game using league-avg pace
            # (we don't have per-game pace, so league avg is our best proxy)
            est_poss = AdjustmentFactors.LEAGUE_AVG_PACE * (log.minutes / 48.0) * (profile.usage_pct / 100.0)
            total_pts += log.pts
            total_poss += est_poss

        if total_poss > 0:
            return total_pts / total_poss
        return 1.0

    @staticmethod
    def _normalize_position(position: str) -> str:
        """Normalize a position string to one of PG/SG/SF/PF/C."""
        pos = position.upper() if position else "SF"
        if pos in ("PG", "SG", "SF", "PF", "C"):
            return pos
        if "G" in pos and "F" not in pos:
            return "SG"
        if "F" in pos and "G" not in pos:
            return "SF"
        if "C" in pos:
            return "C"
        return "SF"

    @staticmethod
    def _load_adjustment_config() -> dict:
        """Load adjustment config from JSON file, falling back to conservative defaults."""
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'adjustment_config.json')
        try:
            with open(config_path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load adjustment_config.json ({e}), using defaults")
            return ProjectionEngine._default_adjustment_config()

    @staticmethod
    def _default_adjustment_config() -> dict:
        """Minimal conservative defaults when config file is missing."""
        return {
            "version": "2.0-defaults",
            "tracking": {"enabled": False, "max_total_adjustment": 2.0},
            "pbp": {"enabled": False, "max_total_adjustment": 1.5},
            "ctg": {"enabled": False, "max_total_adjustment": 1.0, "percentile_scale": 0.03}
        }

    @staticmethod
    def _cfg_adj(value: float, ref: float, scale: float, neg_cap: float, pos_cap: float) -> float:
        """Compute a capped linear adjustment: (value - ref) * scale, clamped to [neg_cap, pos_cap]."""
        raw = (value - ref) * scale
        return max(neg_cap, min(pos_cap, raw))

    def detect_role_change(self, profile: PlayerProfile) -> Tuple[bool, str]:
        """
        Detect if player's role has recently changed based on minutes or scoring trend.
        Uses L3 vs L10 comparison for better recency detection.

        Returns: (role_changed, direction)
            - role_changed: True if significant change detected
            - direction: "expanded" or "reduced" or ""
        """
        # Use L3 for most recent, L10 for baseline
        l3_min = profile.last_3_avg.get("min", 0)
        l10_min = profile.last_10_avg.get("min", 0)
        l3_pts = profile.last_3_avg.get("pts", 0)
        l10_pts = profile.last_10_avg.get("pts", 0)

        min_change = 0.0
        pts_change = 0.0

        # Calculate minutes trend (L3 vs L10)
        if l10_min > 0:
            min_change = (l3_min - l10_min) / l10_min

        # Calculate scoring trend (L3 vs L10)
        if l10_pts > 0:
            pts_change = (l3_pts - l10_pts) / l10_pts

        # Expanded role: 12%+ increase in minutes OR 10%+ increase in scoring
        if min_change >= 0.12 or pts_change >= 0.10:
            return True, "expanded"

        # Reduced role: 12%+ decrease in minutes OR 10%+ decrease in scoring
        if min_change <= -0.12 or pts_change <= -0.10:
            return True, "reduced"

        return False, ""

    def detect_minutes_trend(self, profile: PlayerProfile) -> Tuple[float, str]:
        """
        Detect if player's minutes are trending up or down.
        Compares L3 minutes vs L10 minutes.

        Returns: (trend_factor, direction)
            - trend_factor: Multiplier to apply (0.95 to 1.05)
            - direction: "up", "down", or "stable"
        """
        l3_min = profile.last_3_avg.get("min", 0)
        l10_min = profile.last_10_avg.get("min", 0)

        if l10_min <= 0:
            return 1.0, "stable"

        pct_change = (l3_min - l10_min) / l10_min

        # Strong upward trend (10%+ increase in L3 vs L10)
        if pct_change >= 0.10:
            # Cap at 5% boost to avoid overreaction
            return min(1.05, 1.0 + (pct_change * 0.4)), "up"

        # Strong downward trend (10%+ decrease in L3 vs L10)
        if pct_change <= -0.10:
            # Cap at 5% reduction
            return max(0.95, 1.0 + (pct_change * 0.4)), "down"

        return 1.0, "stable"

    def detect_streak(self, profile: PlayerProfile, stat_key: str) -> Tuple[float, str]:
        """
        Detect if player is on a hot or cold streak for a specific stat.
        Compares L3 performance to L10 baseline using standard deviation.

        Returns: (streak_factor, direction)
            - streak_factor: Multiplier to apply (0.97 to 1.03)
            - direction: "hot", "cold", or "neutral"
        """
        l3_val = profile.last_3_avg.get(stat_key, 0)
        l10_val = profile.last_10_avg.get(stat_key, 0)
        std_dev = profile.std_dev.get(stat_key, 0)

        if l10_val <= 0 or std_dev <= 0:
            return 1.0, "neutral"

        # Calculate how many std devs L3 is from L10
        z_score = (l3_val - l10_val) / std_dev

        # Hot streak: L3 is 1+ std dev above L10
        if z_score >= 1.0:
            # Cap at 3% boost, scale by z-score
            factor = min(1.03, 1.0 + (z_score * 0.02))
            return factor, "hot"

        # Cold streak: L3 is 1+ std dev below L10
        if z_score <= -1.0:
            # Cap at 3% reduction
            factor = max(0.97, 1.0 + (z_score * 0.02))
            return factor, "cold"

        return 1.0, "neutral"

    def project_minutes(
        self,
        profile: PlayerProfile,
        is_home: bool,
        is_b2b: bool,
        spread: float,
        days_rest: int = 2
    ) -> Tuple[float, float, float, float]:
        """
        Project minutes with adjustments using the ML-ready MinutesPredictor.

        Delegates to MinutesPredictor for the actual prediction, then maps
        the result back to the legacy (projected, fatigue_adj, blowout_adj,
        base_minutes) tuple that the rest of the system expects.

        Returns: (projected_minutes, fatigue_adj, blowout_adj, base_minutes)
        """
        # Build a lightweight context object for the MinutesPredictor.
        # The predictor expects a GameContext-like object with spread and
        # team info. We construct a minimal shim from the arguments we have.
        prediction = self._predict_minutes_via_model(
            profile, is_home, is_b2b, spread, days_rest
        )

        # Map the detailed MinutesPrediction back to the legacy 4-tuple.
        # fatigue_adj = b2b + rest + trend (everything non-spread)
        # blowout_adj = spread adjustment (game script)
        fatigue_adj = prediction.b2b_adj + prediction.rest_adj + prediction.trend_adj + prediction.depth_adj
        blowout_adj = prediction.spread_adj

        return (
            prediction.expected_minutes,
            fatigue_adj,
            blowout_adj,
            prediction.base_minutes
        )

    def _predict_minutes_via_model(
        self,
        profile: PlayerProfile,
        is_home: bool,
        is_b2b: bool,
        spread: float,
        days_rest: int
    ):
        """
        Call MinutesPredictor with a shim context built from the legacy args.

        Returns a MinutesPrediction dataclass.
        """
        from minutes_model import MinutesPrediction

        # Build a minimal shim that quacks like GameContext/TeamProfile
        # so MinutesPredictor.predict_minutes_full() can read spread,
        # is_back_to_back, days_rest, and team name.
        class _TeamShim:
            def __init__(self, name, b2b, rest):
                self.name = name
                self.is_back_to_back = b2b
                self.days_rest = rest

        class _ContextShim:
            def __init__(self, home_team, away_team, spread_val):
                self.home_team = home_team
                self.away_team = away_team
                self.spread = spread_val

        if is_home:
            home_shim = _TeamShim(profile.team, is_b2b, days_rest)
            away_shim = _TeamShim("_opponent", False, 1)
        else:
            home_shim = _TeamShim("_opponent", False, 1)
            away_shim = _TeamShim(profile.team, is_b2b, days_rest)

        # spread convention: positive = home favored.
        # If is_home, spread is from home perspective already.
        # If not home, the caller already negated it so spread is from
        # the player's team perspective. We need to convert back to
        # home-perspective for the context shim.
        if is_home:
            context_spread = spread
        else:
            context_spread = -spread

        ctx = _ContextShim(home_shim, away_shim, context_spread)

        return self.minutes_predictor.predict_minutes_full(profile, ctx)

    def get_weighted_stat(
        self,
        profile: PlayerProfile,
        stat_key: str,
        use_role_change: bool = True
    ) -> float:
        """Get weighted average for a single stat, adjusting for role changes

        When roster_filtering_applied is True, uses roster-filtered averages
        (games where currently injured teammates were also out) to prevent
        double-counting injury boosts.
        """
        values = []
        weights = []

        # Detect role change and select appropriate weights
        role_changed, direction = self.detect_role_change(profile) if use_role_change else (False, "")
        weight_set = self.ROLE_CHANGE_WEIGHTS if role_changed else self.WEIGHTS

        # Check if we should use roster-filtered averages
        # This happens when injured teammates' absence is already reflected in recent games
        use_roster_filtered = (
            profile.roster_filtering_applied and
            profile.roster_filtered_games_count >= 3
        )

        if use_roster_filtered:
            # Use roster-filtered averages (games with current roster)
            # These already reflect production without injured teammates
            if stat_key in profile.roster_filtered_l3_avg:
                values.append(profile.roster_filtered_l3_avg[stat_key])
                weights.append(weight_set["last_3"])

            if stat_key in profile.roster_filtered_l5_avg:
                values.append(profile.roster_filtered_l5_avg[stat_key])
                weights.append(weight_set["last_5"])

            # Still use unfiltered L10/season as stabilizer (lower weight)
            # These provide floor/ceiling context even if roster was different
            if stat_key in profile.last_10_avg:
                values.append(profile.last_10_avg[stat_key])
                weights.append(weight_set["last_10"] * 0.5)  # Reduced weight

            if stat_key in profile.season_avg:
                values.append(profile.season_avg[stat_key])
                weights.append(weight_set["season"] * 0.5)  # Reduced weight
        else:
            # Standard weighted average (no roster filtering)
            # L3 gets highest weight for recency
            if stat_key in profile.last_3_avg:
                values.append(profile.last_3_avg[stat_key])
                weights.append(weight_set["last_3"])

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
            # Fallback for rookies / players with no history
            # Use position-average defaults so they don't project to 0
            POSITION_DEFAULTS = {
                "PG": {"pts": 14, "reb": 3.5, "ast": 5, "fg3m": 1.5, "stl": 1.0, "blk": 0.3, "turnover": 2.0},
                "SG": {"pts": 15, "reb": 3.5, "ast": 3, "fg3m": 1.8, "stl": 0.9, "blk": 0.3, "turnover": 1.8},
                "SF": {"pts": 14, "reb": 5.0, "ast": 2.5, "fg3m": 1.3, "stl": 0.8, "blk": 0.5, "turnover": 1.5},
                "PF": {"pts": 13, "reb": 6.0, "ast": 2.0, "fg3m": 1.0, "stl": 0.6, "blk": 0.7, "turnover": 1.3},
                "C":  {"pts": 12, "reb": 7.5, "ast": 1.5, "fg3m": 0.5, "stl": 0.5, "blk": 1.2, "turnover": 1.5},
            }
            pos = (profile.position or "SF").upper()
            if pos not in POSITION_DEFAULTS:
                pos = "SF"
            return POSITION_DEFAULTS[pos].get(stat_key, 0.0)

        total_weight = sum(weights)
        return sum(v * w / total_weight for v, w in zip(values, weights))

    def project_stat(
        self,
        profile: PlayerProfile,
        prop_type: PropType,
        projected_minutes: float,
        is_home: bool,
        opponent_def_rating: float,
        pace_factor: float,
        opponent_team_id: int = None,
        context: 'GameContext' = None,
        opponent_team: 'TeamProfile' = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Project a single stat with all adjustments.

        Uses an expected-possessions model for points when usage data is
        available, and a Vegas-total game environment factor that replaces
        the old pace-only adjustment.

        Args:
            context: GameContext with Vegas total, teams, etc.
            opponent_team: Full TeamProfile for position-specific defense.
            opponent_def_rating / pace_factor: Legacy params kept for
                backward compat; ignored when context/opponent_team provided.

        Returns: (projected_value, adjustments_dict)
        """
        adjustments = {
            "base": 0.0,
            "minutes": 0.0,
            "home_away": 0.0,
            "defense": 0.0,
            "pace": 0.0,
            "streak": 0.0,
            "opponent": 0.0,
        }

        # Handle combined stats
        if prop_type in COMBINED_STAT_MAPPING:
            return self._project_combined_stat(
                profile, prop_type, projected_minutes,
                is_home, opponent_def_rating, pace_factor, opponent_team_id,
                context=context, opponent_team=opponent_team
            )

        # Handle milestone props
        if prop_type in [PropType.DOUBLE_DOUBLE, PropType.TRIPLE_DOUBLE]:
            return self._project_milestone(profile, prop_type)

        # Get stat key
        stat_key = STAT_MAPPING.get(prop_type)
        if not stat_key:
            return 0.0, adjustments

        # -----------------------------------------------------------------
        # 1. BASE PROJECTION
        # -----------------------------------------------------------------
        # For POINTS: use expected-possessions model when usage data exists.
        # For all other stats: per-minute rate blend (same as before).

        per_min_rate = profile.per_minute.get(stat_key, 0)
        avg_minutes = profile.last_10_avg.get("min", 30)

        # Compute expected game pace from both teams (used by possessions model)
        if context is not None:
            expected_game_pace = (context.home_team.pace + context.away_team.pace) / 2
        else:
            expected_game_pace = AdjustmentFactors.LEAGUE_AVG_PACE * pace_factor

        if prop_type == PropType.POINTS and profile.usage_pct > 0:
            # Possessions-driven points projection
            base_from_poss = self.project_points_from_possessions(
                profile, projected_minutes, expected_game_pace, profile.usage_pct
            )
            # Blend with weighted average to stabilize (possessions model
            # can be noisy when usage_pct or pace data is imprecise)
            weighted_avg = self.get_weighted_stat(profile, stat_key)
            base_projection = (base_from_poss * 0.55) + (weighted_avg * 0.45)
        elif per_min_rate > 0 and avg_minutes > 0:
            base_from_rate = per_min_rate * projected_minutes
            weighted_avg = self.get_weighted_stat(profile, stat_key)
            base_projection = (base_from_rate * 0.6) + (weighted_avg * 0.4)
        else:
            base_projection = self.get_weighted_stat(profile, stat_key)

        adjustments["base"] = base_projection
        projected = base_projection

        # Minutes adjustment (difference from average minutes) -- skip for
        # possessions-based points since minutes are already baked in.
        if prop_type != PropType.POINTS or profile.usage_pct <= 0:
            if avg_minutes > 0 and per_min_rate > 0:
                minutes_diff = projected_minutes - avg_minutes
                minutes_adj = per_min_rate * minutes_diff * 0.5  # Dampened
                projected += minutes_adj
                adjustments["minutes"] = minutes_adj

        # Home/away adjustment - prefer L5 splits for recency
        if is_home:
            # Use L5 home splits if available, fallback to full home splits
            home_stat = profile.home_l5_avg.get(stat_key) or profile.home_avg.get(stat_key)
            if home_stat:
                home_diff = home_stat - self.get_weighted_stat(profile, stat_key)
                home_adj = home_diff * 0.3  # Dampened
                projected += home_adj
                adjustments["home_away"] = home_adj
        else:
            # Use L5 away splits if available, fallback to full away splits
            away_stat = profile.away_l5_avg.get(stat_key) or profile.away_avg.get(stat_key)
            if away_stat:
                away_diff = away_stat - self.get_weighted_stat(profile, stat_key)
                away_adj = away_diff * 0.3
                projected += away_adj
                adjustments["home_away"] = away_adj

        # Streak adjustment (hot/cold detection)
        streak_factor, streak_direction = self.detect_streak(profile, stat_key)

        # Opponent-specific adjustment (historical performance vs this team)
        opp_factor = 1.0
        if opponent_team_id and str(opponent_team_id) in profile.opponent_history:
            opp_history = profile.opponent_history[str(opponent_team_id)]
            games_vs_opp = opp_history.get("games", 0)

            # Only apply if 2+ games against this opponent for reliability
            if games_vs_opp >= 2 and stat_key in opp_history:
                opp_avg = opp_history[stat_key]
                overall_avg = self.get_weighted_stat(profile, stat_key)

                if overall_avg > 0:
                    opp_diff_pct = (opp_avg - overall_avg) / overall_avg
                    sample_weight = min(games_vs_opp / 5.0, 1.0)
                    capped_diff = max(-0.15, min(0.15, opp_diff_pct)) * sample_weight * 0.5
                    opp_factor = 1.0 + capped_diff

        # Defense adjustment — position-specific when available
        def_factor = self._get_defense_factor(
            opponent_def_rating,
            profile.position,
            prop_type,
            opponent_team=opponent_team
        )

        # Game environment factor — Vegas total replaces old pace adjustment
        # when context is available; falls back to raw pace_factor
        if context is not None and context.total > 0:
            env_factor = self.get_game_environment_factor(context.total)
        else:
            env_factor = pace_factor

        # ADDITIVE stacking: sum all factor deltas, apply once
        # This prevents multiplicative compounding of small adjustments
        total_factor_delta = (
            (streak_factor - 1.0) +
            (opp_factor - 1.0) +
            (def_factor - 1.0) +
            (env_factor - 1.0)
        )
        # Cap total adjustment at +/-25% to prevent runaway projections
        total_factor_delta = max(-0.25, min(0.25, total_factor_delta))

        # Track individual contributions (proportional to their share)
        adjustments["streak"] = projected * (streak_factor - 1.0) if streak_factor != 1.0 else 0
        adjustments["opponent"] = projected * (opp_factor - 1.0) if opp_factor != 1.0 else 0
        adjustments["defense"] = projected * (def_factor - 1.0)
        adjustments["pace"] = projected * (env_factor - 1.0)

        # Apply combined adjustment once
        projected *= (1.0 + total_factor_delta)

        # Apply tracking-based adjustments (from NBA.com tracking data)
        tracking_adj, tracking_conf = self._apply_tracking_adjustment(profile, prop_type, projected)
        if tracking_adj != 0:
            projected += tracking_adj
            adjustments["tracking"] = tracking_adj
            logger.debug(f"Tracking adjustment for {profile.name} {prop_type.value}: {tracking_adj:+.2f} (conf: {tracking_conf})")

        # Apply PBP Stats adjustments (from pbpstats.com)
        pbp_adj, pbp_conf = self._apply_pbp_adjustment(profile, prop_type, projected)
        if pbp_adj != 0:
            projected += pbp_adj
            adjustments["pbp"] = pbp_adj
            logger.debug(f"PBP adjustment for {profile.name} {prop_type.value}: {pbp_adj:+.2f} (conf: {pbp_conf})")

        # Apply Cleaning the Glass adjustments (percentile context)
        ctg_adj, ctg_conf = self._apply_ctg_adjustment(profile, prop_type, projected)
        if ctg_adj != 0:
            projected += ctg_adj
            adjustments["ctg"] = ctg_adj
            logger.debug(f"CTG adjustment for {profile.name} {prop_type.value}: {ctg_adj:+.2f} (conf: {ctg_conf})")

        # ANTI-OVERESTIMATION: Heavy regression toward the DK line
        # Honest evaluation (122K legs) showed:
        #   - Model MAE 4.0 vs Line MAE 3.4 (line is more accurate)
        #   - Model bias: -0.45 (systematically over-projects)
        #   - Optimal blend: 30% model / 70% line (best Brier score)
        #   - UNDER picks hit 53% vs OVER 48% (over-projection kills value)
        base = adjustments["base"]
        if base > 0:
            regression_factor = 0.30
            projected = (projected * (1 - regression_factor)) + (base * regression_factor)

        return max(0, projected), adjustments

    def _project_combined_stat(
        self,
        profile: PlayerProfile,
        prop_type: PropType,
        projected_minutes: float,
        is_home: bool,
        opponent_def_rating: float,
        pace_factor: float,
        opponent_team_id: int = None,
        context: 'GameContext' = None,
        opponent_team: 'TeamProfile' = None
    ) -> Tuple[float, Dict[str, float]]:
        """Project combined stats (PRA, PR, etc.)"""
        stat_keys = COMBINED_STAT_MAPPING.get(prop_type, [])
        total = 0.0
        adjustments = {
            "base": 0.0, "minutes": 0.0, "home_away": 0.0,
            "defense": 0.0, "pace": 0.0, "streak": 0.0, "opponent": 0.0
        }

        for stat_key in stat_keys:
            # Map stat key to PropType for individual projection
            prop_map = {v: k for k, v in STAT_MAPPING.items()}
            individual_prop = prop_map.get(stat_key)
            if individual_prop:
                val, adj = self.project_stat(
                    profile, individual_prop, projected_minutes,
                    is_home, opponent_def_rating, pace_factor, opponent_team_id,
                    context=context, opponent_team=opponent_team
                )
                total += val
                for k in adjustments:
                    adjustments[k] += adj.get(k, 0)

        # Apply small dampening for combined props to account for negative
        # correlations between components (e.g., high scoring load reduces assists)
        if len(stat_keys) >= 3:
            total *= 0.97  # 3% dampening for 3+ component props (PRA)
        elif len(stat_keys) == 2:
            total *= 0.985  # 1.5% dampening for 2-component props (PR, PA)

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
        prop_type: PropType,
        opponent_team: 'TeamProfile' = None
    ) -> float:
        """
        Get defensive adjustment factor.

        Prefers position-specific multipliers from TeamProfile
        (def_vs_pg ... def_vs_c) when they contain real data.
        Falls back to the overall-DRtg tier approach when the
        position-specific fields are still at their 1.0 defaults.
        """
        pos = self._normalize_position(position)

        # ----- Try position-specific multipliers first -----
        if opponent_team is not None:
            pos_defense_map = {
                "PG": opponent_team.def_vs_pg,
                "SG": opponent_team.def_vs_sg,
                "SF": opponent_team.def_vs_sf,
                "PF": opponent_team.def_vs_pf,
                "C": opponent_team.def_vs_c,
            }
            base_factor = pos_defense_map.get(pos, 1.0)

            # If we have real data (not all defaults at 1.0), use it
            has_real_data = any(
                v != 1.0 for v in pos_defense_map.values()
            )
            if has_real_data:
                factor = base_factor

                # Defense matters less for volatile/steal/block props
                if prop_type in (PropType.TURNOVERS, PropType.STEALS, PropType.BLOCKS):
                    factor = 1 + (factor - 1) * 0.5

                return factor

        # ----- Fallback: overall DRtg tier approach -----
        return self._get_defense_factor_from_drtg(opponent_def_rating, pos, prop_type)

    def _get_defense_factor_from_drtg(
        self,
        opponent_def_rating: float,
        pos: str,
        prop_type: PropType
    ) -> float:
        """Legacy defense factor derived from overall defensive rating tiers."""
        # Categorize defense tier relative to league average
        league_avg_drtg = 113.0
        if opponent_def_rating <= league_avg_drtg - 5:
            def_tier = "elite"
        elif opponent_def_rating <= league_avg_drtg - 2:
            def_tier = "good"
        elif opponent_def_rating <= league_avg_drtg + 2:
            def_tier = "average"
        elif opponent_def_rating <= league_avg_drtg + 4:
            def_tier = "poor"
        else:
            def_tier = "bad"

        # Get base position factor from tier
        factor = AdjustmentFactors.DEFENSE_VS_POSITION.get(def_tier, {}).get(pos, 1.0)

        # Enhanced position-specific adjustments
        # Guards face more perimeter pressure from elite defenses
        if pos in ["PG", "SG"] and def_tier in ["elite", "good"]:
            if prop_type == PropType.POINTS:
                factor = factor * 0.97 if def_tier == "elite" else factor * 0.98
            elif prop_type == PropType.ASSISTS:
                factor = factor * 0.96 if def_tier == "elite" else factor * 0.97
            elif prop_type == PropType.THREES:
                factor = factor * 0.95 if def_tier == "elite" else factor * 0.97

        # Bigs face more interior pressure
        if pos in ["PF", "C"] and def_tier in ["elite", "good"]:
            if prop_type == PropType.REBOUNDS:
                factor = factor * 0.96 if def_tier == "elite" else factor * 0.97
            elif prop_type == PropType.POINTS:
                factor = factor * 0.95 if def_tier == "elite" else factor * 0.97

        # Conversely, bad defenses get exploited more by specific positions
        if def_tier in ["poor", "bad"]:
            if pos in ["PG", "SG"] and prop_type in [PropType.POINTS, PropType.THREES]:
                factor = factor * 1.03 if def_tier == "bad" else factor * 1.02
            elif pos in ["PF", "C"] and prop_type in [PropType.POINTS, PropType.REBOUNDS]:
                factor = factor * 1.04 if def_tier == "bad" else factor * 1.02

        # Defense matters less for some props
        if prop_type in [PropType.TURNOVERS, PropType.STEALS, PropType.BLOCKS]:
            factor = 1 + (factor - 1) * 0.5  # Dampen

        return factor

    def _apply_tracking_adjustment(
        self,
        profile: PlayerProfile,
        prop_type: PropType,
        base_projection: float
    ) -> Tuple[float, float]:
        """
        Apply tracking-based adjustments using NBA.com tracking data.

        All thresholds and multipliers are loaded from adjustment_config.json
        so they can be tuned via backtest without code changes.

        Returns: (adjustment_amount, confidence_factor)
            - adjustment_amount: Points to add/subtract
            - confidence_factor: 0-1 indicating reliability of adjustment
        """
        config = self.adj_config.get("tracking", {})
        if not config.get("enabled", True):
            return 0.0, 0.0

        tracking = profile.tracking
        if not tracking:
            return 0.0, 0.0

        adjustment = 0.0
        confidence = 0.0
        prop_key = prop_type.value  # "points", "rebounds", etc.
        prop_cfg = config.get(prop_key, {})

        if prop_type == PropType.POINTS:
            # Drives: more drives = more scoring opportunities
            cfg = prop_cfg.get("drives", {})
            if cfg.get("enabled") and tracking.drives > 0:
                league_avg = cfg["league_avg"]
                drive_factor = (tracking.drives - league_avg) / max(league_avg, 1)
                drive_eff = tracking.drive_fg_pct if tracking.drive_fg_pct > 0 else cfg.get("efficiency_default", 0.45)
                raw = drive_factor * drive_eff * cfg.get("efficiency_scale", 1.5)
                cap = cfg["cap"]
                adjustment += max(-cap, min(cap, raw))
                confidence += cfg.get("confidence", 0.2)

            # Catch-and-shoot efficiency
            cfg = prop_cfg.get("catch_shoot", {})
            if cfg.get("enabled") and tracking.catch_shoot_fg_pct > cfg.get("min_pct", 0.40):
                raw = (tracking.catch_shoot_fg_pct - cfg["league_avg_pct"]) * tracking.catch_shoot_fga * cfg["scale"]
                adjustment += max(0, min(cfg["cap"], raw))
                confidence += cfg.get("confidence", 0.15)

            # Pull-up efficiency (disabled by default -- high variance, small sample)
            cfg = prop_cfg.get("pull_up", {})
            if cfg.get("enabled") and tracking.pull_up_fg_pct > cfg.get("min_pct", 0.42):
                raw = (tracking.pull_up_fg_pct - cfg["league_avg_pct"]) * tracking.pull_up_fga * cfg["scale"]
                adjustment += max(0, min(cfg["cap"], raw))
                confidence += cfg.get("confidence", 0.1)

            # Free throw creation from drives
            cfg = prop_cfg.get("free_throws", {})
            if cfg.get("enabled") and tracking.drive_fta > 0:
                raw = (tracking.drive_fta - cfg["league_avg_fta"]) * cfg["scale"]
                adjustment += max(cfg.get("neg_cap", -0.5), min(cfg.get("pos_cap", 1.0), raw))
                confidence += cfg.get("confidence", 0.1)

        elif prop_type == PropType.ASSISTS:
            # Potential assists vs actual: regression toward mean
            cfg = prop_cfg.get("potential_assists", {})
            if cfg.get("enabled") and tracking.potential_assists > 0 and profile.last_10_avg.get("ast", 0) > 0:
                actual_ast = profile.last_10_avg["ast"]
                assist_gap = tracking.potential_assists - actual_ast
                if assist_gap > cfg.get("positive_gap_threshold", 1.5):
                    adjustment += min(cfg["positive_cap"], assist_gap * cfg["positive_scale"])
                    confidence += cfg.get("positive_confidence", 0.25)
                elif assist_gap < cfg.get("negative_gap_threshold", -0.5):
                    adjustment += max(cfg["negative_cap"], assist_gap * cfg["negative_scale"])
                    confidence += cfg.get("negative_confidence", 0.15)

            # Passing efficiency (ast_to_pass_pct)
            cfg = prop_cfg.get("passing_efficiency", {})
            if cfg.get("enabled") and tracking.ast_to_pass_pct > 0:
                raw = (tracking.ast_to_pass_pct - cfg["league_avg_pct"]) * cfg["scale"]
                cap = cfg["cap"]
                adjustment += max(-cap, min(cap, raw))
                confidence += cfg.get("confidence", 0.15)

            # Drive assists
            cfg = prop_cfg.get("drive_assists", {})
            if cfg.get("enabled") and tracking.drive_ast > cfg.get("min_val", 0.5):
                raw = (tracking.drive_ast - cfg["min_val"]) * cfg["scale"]
                adjustment += max(0, min(cfg["cap"], raw))
                confidence += cfg.get("confidence", 0.1)

        elif prop_type == PropType.REBOUNDS:
            # Defensive rebound chances & conversion
            cfg = prop_cfg.get("dreb_chances", {})
            if cfg.get("enabled") and tracking.dreb_chances > 0 and tracking.dreb_chance_pct > 0:
                raw = (tracking.dreb_chance_pct - cfg["league_avg_pct"]) * cfg["scale"]
                cap = cfg["cap"]
                adjustment += max(-cap, min(cap, raw))
                confidence += cfg.get("confidence", 0.2)

            # Offensive rebound chances
            cfg = prop_cfg.get("oreb_chances", {})
            if cfg.get("enabled") and tracking.oreb_chances > 0 and tracking.oreb_chance_pct > 0:
                raw = (tracking.oreb_chance_pct - cfg["league_avg_pct"]) * cfg["scale"]
                cap = cfg["cap"]
                adjustment += max(-cap, min(cap, raw))
                confidence += cfg.get("confidence", 0.15)

            # Box outs (disabled by default -- too noisy)
            cfg = prop_cfg.get("box_outs", {})
            if cfg.get("enabled") and tracking.box_outs > 0:
                raw = (tracking.box_outs - cfg["league_avg"]) * cfg["scale"]
                cap = cfg["cap"]
                adjustment += max(-cap, min(cap, raw))
                confidence += cfg.get("confidence", 0.1)

            # Average rebound distance (disabled by default -- too noisy)
            cfg = prop_cfg.get("avg_reb_dist", {})
            if cfg.get("enabled") and tracking.avg_reb_dist > 0:
                raw = (cfg["neutral_dist"] - tracking.avg_reb_dist) * cfg["scale"]
                cap = cfg["cap"]
                adjustment += max(-cap, min(cap, raw))
                confidence += cfg.get("confidence", 0.05)

        elif prop_type == PropType.STEALS:
            # Deflections
            cfg = prop_cfg.get("deflections", {})
            if cfg.get("enabled") and tracking.deflections > 0:
                raw = (tracking.deflections - cfg["league_avg"]) * cfg["scale"]
                adjustment += max(cfg.get("neg_cap", -0.3), min(cfg.get("pos_cap", 0.3), raw))
                confidence += cfg.get("confidence", 0.25)

            # Loose balls recovered (disabled by default -- too noisy)
            cfg = prop_cfg.get("loose_balls", {})
            if cfg.get("enabled") and tracking.loose_balls_recovered > 0:
                raw = (tracking.loose_balls_recovered - cfg["league_avg"]) * cfg["scale"]
                adjustment += max(cfg.get("neg_cap", -0.15), min(cfg.get("pos_cap", 0.25), raw))
                confidence += cfg.get("confidence", 0.1)

        elif prop_type == PropType.BLOCKS:
            # Contested shots
            cfg = prop_cfg.get("contested_shots", {})
            if cfg.get("enabled") and tracking.contested_shots > 0:
                raw = (tracking.contested_shots - cfg["league_avg"]) * cfg["scale"]
                adjustment += max(cfg.get("neg_cap", -0.2), min(cfg.get("pos_cap", 0.2), raw))
                confidence += cfg.get("confidence", 0.2)

            # Rim defense FGA
            cfg = prop_cfg.get("rim_defense", {})
            if cfg.get("enabled") and tracking.def_rim_fga > 0:
                raw = (tracking.def_rim_fga - cfg["league_avg_fga"]) * cfg["scale"]
                adjustment += max(cfg.get("neg_cap", -0.15), min(cfg.get("pos_cap", 0.2), raw))
                confidence += cfg.get("confidence", 0.15)

        elif prop_type == PropType.THREES:
            # Catch-and-shoot 3P%
            cfg = prop_cfg.get("catch_shoot_3pct", {})
            if cfg.get("enabled") and tracking.catch_shoot_fg3_pct > 0:
                raw = (tracking.catch_shoot_fg3_pct - cfg["league_avg"]) * tracking.catch_shoot_fga * cfg["scale"]
                adjustment += max(cfg.get("neg_cap", -0.3), min(cfg.get("pos_cap", 0.5), raw))
                confidence += cfg.get("confidence", 0.2)

            # Pull-up 3P% (disabled by default -- high variance)
            cfg = prop_cfg.get("pull_up_3pct", {})
            if cfg.get("enabled") and tracking.pull_up_fg3_pct > 0 and tracking.pull_up_fga > cfg.get("min_fga", 2.0):
                raw = (tracking.pull_up_fg3_pct - cfg["league_avg"]) * tracking.pull_up_fga * cfg["scale"]
                adjustment += max(cfg.get("neg_cap", -0.2), min(cfg.get("pos_cap", 0.3), raw))
                confidence += cfg.get("confidence", 0.1)

        # Normalize confidence to 0-1 range
        confidence = min(1.0, confidence)

        # Cap total adjustment
        max_adj = config.get("max_total_adjustment", 2.0)
        adjustment = max(-max_adj, min(max_adj, adjustment))

        return round(adjustment, 2), round(confidence, 2)

    def _apply_pbp_adjustment(
        self,
        profile: PlayerProfile,
        prop_type: PropType,
        base_projection: float
    ) -> Tuple[float, float]:
        """
        Apply PBP Stats adjustments using pbpstats.com data.

        All thresholds and multipliers are loaded from adjustment_config.json.
        The old shot_quality formula (shot_quality_avg * 2 as TS% proxy) was
        mathematically incorrect and is disabled by default.

        Returns: (adjustment_amount, confidence_factor)
        """
        config = self.adj_config.get("pbp", {})
        if not config.get("enabled", True):
            return 0.0, 0.0

        pbp = profile.pbp
        if not pbp:
            return 0.0, 0.0

        adjustment = 0.0
        confidence = 0.0
        prop_key = prop_type.value
        prop_cfg = config.get(prop_key, {})

        if prop_type == PropType.POINTS:
            # Shot quality regression (DISABLED by default -- old formula was wrong)
            # The previous code did: expected_ts = shot_quality_avg * 2
            # That's not how TS% relates to shot quality. Only re-enable after
            # proper calibration with backtest data.
            cfg = prop_cfg.get("shot_quality", {})
            if cfg.get("enabled") and pbp.shot_quality_avg > 0 and pbp.ts_pct > 0:
                # If re-enabled, use shot quality as relative deviation from league avg
                # rather than a bogus TS% conversion
                league_avg_sq = cfg.get("league_avg", 0.5)
                sq_diff = (pbp.shot_quality_avg - league_avg_sq) / max(league_avg_sq, 0.01)
                scale = cfg.get("scale", 0.05)
                cap = cfg.get("cap", 1.0)
                raw = sq_diff * base_projection * scale
                adjustment += max(-cap, min(cap, raw))
                confidence += cfg.get("confidence", 0.1)

            # At-rim efficiency
            cfg = prop_cfg.get("rim_efficiency", {})
            if cfg.get("enabled") and pbp.at_rim_fga > 0 and pbp.at_rim_accuracy > 0:
                rim_diff = pbp.at_rim_accuracy - cfg["league_avg_pct"]
                raw = rim_diff * pbp.at_rim_fga * cfg["scale"]
                adjustment += max(cfg.get("neg_cap", -0.75), min(cfg.get("pos_cap", 1.0), raw))
                confidence += cfg.get("confidence", 0.1)

            # Self offensive rebounds (second-chance points)
            cfg = prop_cfg.get("self_oreb_points", {})
            if cfg.get("enabled") and pbp.self_oreb > 0:
                raw = (pbp.self_oreb - cfg["league_avg"]) * cfg["scale"]
                adjustment += max(cfg.get("neg_cap", -0.3), min(cfg.get("pos_cap", 0.5), raw))
                confidence += cfg.get("confidence", 0.05)

        elif prop_type == PropType.ASSISTS:
            # Playmaker detection via assisted shot rate
            cfg = prop_cfg.get("playmaker", {})
            if cfg.get("enabled") and pbp.assisted_2s_pct > 0 and pbp.assisted_3s_pct > 0:
                avg_assisted = (pbp.assisted_2s_pct + pbp.assisted_3s_pct) / 2
                threshold = cfg.get("assisted_threshold", 0.50)
                if avg_assisted < threshold:
                    raw = (threshold - avg_assisted) * cfg["scale"]
                    adjustment += max(0, min(cfg["cap"], raw))
                    confidence += cfg.get("confidence", 0.15)

            # Rim assists (drive-and-kick)
            cfg = prop_cfg.get("rim_assists", {})
            if cfg.get("enabled") and pbp.at_rim_assists > 0:
                raw = (pbp.at_rim_assists - cfg["league_avg"]) * cfg["scale"]
                adjustment += max(cfg.get("neg_cap", -0.3), min(cfg.get("pos_cap", 0.5), raw))
                confidence += cfg.get("confidence", 0.1)

        elif prop_type == PropType.REBOUNDS:
            # Self offensive rebounds (effort metric)
            cfg = prop_cfg.get("self_oreb", {})
            if cfg.get("enabled") and pbp.self_oreb > 0:
                raw = (pbp.self_oreb - cfg["league_avg"]) * cfg["scale"]
                adjustment += max(cfg.get("neg_cap", -0.25), min(cfg.get("pos_cap", 0.5), raw))
                confidence += cfg.get("confidence", 0.15)

            # Defensive rebound %
            cfg = prop_cfg.get("dreb_pct", {})
            if cfg.get("enabled") and pbp.def_reb_pct > 0:
                raw = (pbp.def_reb_pct - cfg["league_avg"]) * cfg["scale"]
                adjustment += max(cfg.get("neg_cap", -0.5), min(cfg.get("pos_cap", 1.0), raw))
                confidence += cfg.get("confidence", 0.15)

        elif prop_type == PropType.STEALS:
            # Bad pass steals
            cfg = prop_cfg.get("bad_pass_steals", {})
            if cfg.get("enabled") and pbp.bad_pass_steals > 0:
                raw = (pbp.bad_pass_steals - cfg["league_avg"]) * cfg["scale"]
                adjustment += max(cfg.get("neg_cap", -0.2), min(cfg.get("pos_cap", 0.3), raw))
                confidence += cfg.get("confidence", 0.2)

            # Lost ball steals
            cfg = prop_cfg.get("lost_ball_steals", {})
            if cfg.get("enabled") and pbp.lost_ball_steals > 0:
                raw = (pbp.lost_ball_steals - cfg["league_avg"]) * cfg["scale"]
                adjustment += max(cfg.get("neg_cap", -0.15), min(cfg.get("pos_cap", 0.2), raw))
                confidence += cfg.get("confidence", 0.1)

        elif prop_type == PropType.BLOCKS:
            # FG2A blocked %
            cfg = prop_cfg.get("fg2a_blocked_pct", {})
            if cfg.get("enabled") and pbp.fg2a_pct_blocked > 0:
                raw = (pbp.fg2a_pct_blocked - cfg["league_avg"]) * cfg["scale"]
                adjustment += max(cfg.get("neg_cap", -0.2), min(cfg.get("pos_cap", 0.3), raw))
                confidence += cfg.get("confidence", 0.2)

        elif prop_type == PropType.THREES:
            # Corner 3 efficiency
            cfg = prop_cfg.get("corner3", {})
            if cfg.get("enabled") and pbp.corner3_fga > 0 and pbp.corner3_accuracy > 0:
                raw = (pbp.corner3_accuracy - cfg["league_avg_pct"]) * pbp.corner3_fga * cfg["scale"]
                adjustment += max(cfg.get("neg_cap", -0.2), min(cfg.get("pos_cap", 0.3), raw))
                confidence += cfg.get("confidence", 0.1)

            # Arc 3 efficiency
            cfg = prop_cfg.get("arc3", {})
            if cfg.get("enabled") and pbp.arc3_fga > 0 and pbp.arc3_accuracy > 0:
                raw = (pbp.arc3_accuracy - cfg["league_avg_pct"]) * pbp.arc3_fga * cfg["scale"]
                adjustment += max(cfg.get("neg_cap", -0.2), min(cfg.get("pos_cap", 0.3), raw))
                confidence += cfg.get("confidence", 0.1)

        elif prop_type == PropType.TURNOVERS:
            # Bad pass turnovers (disabled by default -- rarely bet prop)
            cfg = prop_cfg.get("bad_pass_turnovers", {})
            if cfg.get("enabled") and pbp.bad_pass_turnovers > 0:
                raw = (pbp.bad_pass_turnovers - cfg["league_avg"]) * cfg["scale"]
                adjustment += max(cfg.get("neg_cap", -0.3), min(cfg.get("pos_cap", 0.4), raw))
                confidence += cfg.get("confidence", 0.15)

            # Lost ball turnovers (disabled by default)
            cfg = prop_cfg.get("lost_ball_turnovers", {})
            if cfg.get("enabled") and pbp.lost_ball_turnovers > 0:
                raw = (pbp.lost_ball_turnovers - cfg["league_avg"]) * cfg["scale"]
                adjustment += max(cfg.get("neg_cap", -0.25), min(cfg.get("pos_cap", 0.35), raw))
                confidence += cfg.get("confidence", 0.1)

        confidence = min(1.0, confidence)

        # Cap total adjustment
        max_adj = config.get("max_total_adjustment", 1.5)
        adjustment = max(-max_adj, min(max_adj, adjustment))

        return round(adjustment, 2), round(confidence, 2)

    def _apply_ctg_adjustment(
        self,
        profile: PlayerProfile,
        prop_type: PropType,
        base_projection: float
    ) -> Tuple[float, float]:
        """
        Apply Cleaning the Glass adjustments using percentile context.

        CTG provides garbage-time filtered stats with league percentile rankings.
        Percentiles are uniform [0, 100], NOT normally distributed, so we do NOT
        convert them to z-scores. Instead we use linear deviation from median:
            pct_deviation = (percentile - 50) / 100  # Range [-0.5, +0.5]
            adjustment = pct_deviation * base_projection * scale

        All thresholds loaded from adjustment_config.json.

        Returns: (adjustment_amount, confidence_factor)
        """
        config = self.adj_config.get("ctg", {})
        if not config.get("enabled", True):
            return 0.0, 0.0

        ctg = profile.ctg
        if not ctg:
            return 0.0, 0.0

        adjustment = 0.0
        confidence = 0.0

        def _percentile_adj(percentile: float, scale: float, cap: float) -> float:
            """Convert a uniform percentile to a capped linear adjustment.
            Percentiles are uniform [0,100]. We use (pct - 50)/100 to get
            a deviation in [-0.5, +0.5], then scale by base_projection.
            """
            pct_deviation = (percentile - 50) / 100.0
            raw = pct_deviation * base_projection * scale
            return max(-cap, min(cap, raw))

        if prop_type == PropType.POINTS:
            prop_cfg = config.get("points", {})

            # PSA (Points per Shot Attempt) percentile
            cfg = prop_cfg.get("psa", {})
            if cfg.get("enabled") and ctg.psa_percentile > 0:
                adj = _percentile_adj(ctg.psa_percentile, cfg["scale"], cfg["cap"])
                adjustment += adj
                confidence += cfg.get("confidence", 0.15)

            # Star treatment: high usage + high efficiency
            cfg = prop_cfg.get("star_treatment", {})
            if cfg.get("enabled"):
                if (ctg.usage_percentile > cfg.get("usage_min_percentile", 80)
                        and ctg.psa_percentile > cfg.get("psa_min_percentile", 60)):
                    raw = base_projection * cfg["scale"]
                    adjustment += min(cfg["cap"], raw)
                    confidence += cfg.get("confidence", 0.1)

        elif prop_type == PropType.ASSISTS:
            prop_cfg = config.get("assists", {})

            # AST% percentile
            cfg = prop_cfg.get("ast_pct", {})
            if cfg.get("enabled") and ctg.ast_percentile > 0:
                adj = _percentile_adj(ctg.ast_percentile, cfg["scale"], cfg["cap"])
                adjustment += adj
                confidence += cfg.get("confidence", 0.2)

            # Assist-to-Usage ratio percentile
            cfg = prop_cfg.get("ast_to_usage", {})
            if cfg.get("enabled") and ctg.ast_to_usage_percentile > 0:
                high_thresh = cfg.get("high_threshold", 70)
                low_thresh = cfg.get("low_threshold", 30)
                if ctg.ast_to_usage_percentile > high_thresh:
                    raw = (ctg.ast_to_usage_percentile - high_thresh) / 100 * base_projection * cfg["high_scale"]
                    adjustment += max(0, min(cfg["high_cap"], raw))
                    confidence += cfg.get("confidence", 0.1)
                elif ctg.ast_to_usage_percentile < low_thresh:
                    raw = (low_thresh - ctg.ast_to_usage_percentile) / 100 * base_projection * cfg["low_scale"]
                    adjustment += max(cfg["low_cap"], min(0, raw))
                    confidence += cfg.get("confidence", 0.1)

        elif prop_type == PropType.TURNOVERS:
            # TOV% percentile (disabled by default -- rarely bet prop)
            prop_cfg = config.get("turnovers", {})
            cfg = prop_cfg.get("tov_pct", {})
            if cfg.get("enabled") and ctg.tov_percentile > 0:
                low_thresh = cfg.get("low_threshold", 30)
                high_thresh = cfg.get("high_threshold", 70)
                if ctg.tov_percentile < low_thresh:
                    # Low percentile = high TOV% = more turnovers expected
                    raw = (low_thresh - ctg.tov_percentile) / 100 * cfg["low_scale"]
                    adjustment += max(0, min(cfg["low_cap"], raw))
                    confidence += cfg.get("low_confidence", 0.2)
                elif ctg.tov_percentile > high_thresh:
                    # High percentile = low TOV% = fewer turnovers
                    raw = (ctg.tov_percentile - high_thresh) / 100 * cfg["high_scale"]
                    adjustment += max(cfg["high_cap"], min(0, raw))
                    confidence += cfg.get("high_confidence", 0.15)

        elif prop_type in (PropType.PRA, PropType.PA):
            # Combined props: scorer + playmaker bonus
            combo_cfg = config.get("combined", {}).get("pra_pa", {})
            if combo_cfg.get("enabled"):
                psa_min = combo_cfg.get("psa_min_percentile", 70)
                ast_min = combo_cfg.get("ast_min_percentile", 50)
                alt_psa = combo_cfg.get("alt_psa_min_percentile", 50)
                alt_ast = combo_cfg.get("alt_ast_min_percentile", 70)
                if ((ctg.psa_percentile > psa_min and ctg.ast_percentile > ast_min)
                        or (ctg.psa_percentile > alt_psa and ctg.ast_percentile > alt_ast)):
                    raw = base_projection * combo_cfg["scale"]
                    adjustment += min(combo_cfg["cap"], raw)
                    confidence += combo_cfg.get("confidence", 0.1)

        # Position-specific adjustments
        pos_cfg = config.get("position", {})
        if ctg.position == "Point" and prop_type == PropType.ASSISTS:
            cfg = pos_cfg.get("point_assists", {})
            if cfg.get("enabled") and ctg.ast_percentile > cfg.get("min_ast_percentile", 60):
                raw = base_projection * cfg["scale"]
                adjustment += min(cfg["cap"], raw)
                confidence += cfg.get("confidence", 0.05)

        elif ctg.position == "Big" and prop_type == PropType.REBOUNDS:
            cfg = pos_cfg.get("big_rebounds", {})
            if cfg.get("enabled") and ctg.usage_percentile < cfg.get("max_usage_percentile", 40):
                raw = base_projection * cfg["scale"]
                adjustment += min(cfg["cap"], raw)
                confidence += cfg.get("confidence", 0.1)

        confidence = min(1.0, confidence)

        # Cap total adjustment
        max_adj = config.get("max_total_adjustment", 1.0)
        adjustment = max(-max_adj, min(max_adj, adjustment))

        return round(adjustment, 2), round(confidence, 2)

    def calculate_std_dev(
        self,
        profile: PlayerProfile,
        prop_type: PropType
    ) -> float:
        """Get standard deviation for a prop type, with minimum floors."""
        if prop_type in COMBINED_STAT_MAPPING:
            stat_keys = COMBINED_STAT_MAPPING[prop_type]
            if not profile.game_logs:
                return EdgeCalculator.MIN_STD_DEV.get("pra", 9.0)

            combined_values = []
            for log in profile.game_logs:
                total = sum(getattr(log, k, 0) for k in stat_keys)
                combined_values.append(total)

            raw = float(np.std(combined_values)) if combined_values else 5.0
            return max(raw, EdgeCalculator.MIN_STD_DEV.get("pra", 9.0))

        stat_key = STAT_MAPPING.get(prop_type)
        if stat_key and stat_key in profile.std_dev:
            raw = profile.std_dev[stat_key]
            min_floor = EdgeCalculator.MIN_STD_DEV.get(stat_key, 3.0)
            return max(raw, min_floor)

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
    # Honest evaluation (122K legs) showed higher calibration = better Brier score:
    # 2.0 → 0.2527, 3.0 → 0.2512, 4.0 → 0.2504, 5.0 → 0.2500
    # Using 3.0 as a balance between calibration and retaining signal
    STD_DEV_CALIBRATION = 3.0

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

        # Honest evaluation (122K legs, no lookahead):
        # Calibration table shows model is overconfident at ALL levels above 50%
        # Predicted 55-60% → actual 49.7%, Predicted 60-65% → actual 53.3%
        # With STD_DEV_CALIBRATION=3.0, probabilities are already more conservative
        # Apply moderate damping to prevent extreme confidence
        if over_prob > 0.58:
            over_prob = 0.58 + (over_prob - 0.58) * 0.25
        elif over_prob < 0.42:
            over_prob = 0.42 - (0.42 - over_prob) * 0.25

        # Tight cap: honest data shows nothing above 63% is calibrated
        return max(0.37, min(0.63, over_prob))

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

    # Time-decay thresholds for injury boosts
    # If a player has been out for a long time, teammates have adjusted
    INJURY_FRESH_GAMES = 3      # Full boost: out 0-3 games
    INJURY_RECENT_GAMES = 7     # 75% boost: out 4-7 games
    INJURY_MEDIUM_GAMES = 14    # 40% boost: out 8-14 games
    INJURY_OLD_GAMES = 21       # 15% boost: out 15-21 games
    # Beyond 21 games: no boost (team fully adjusted)

    def __init__(self, pipeline: DataPipeline, lineup_model: LineupImpactModel = None):
        self.pipeline = pipeline
        self.lineup_model = lineup_model

    def _calculate_injury_time_decay(self, scratched: PlayerProfile) -> float:
        """
        Calculate time-decay multiplier based on how long player has been out.

        If a player has been out for weeks, their teammates' recent stats
        ALREADY reflect the increased usage. Boosting again would double-count.

        IMPORTANT: Check for actual playing time (minutes > 0), not just game dates.
        The API returns DNP entries where players have 0 minutes - those don't count.

        Returns: multiplier from 0.0 to 1.0
        - 1.0 = fresh injury, apply full boost
        - 0.0 = long-term injury, teammates have fully adjusted
        """
        from datetime import datetime, timedelta

        if not scratched.game_logs:
            # No game logs = likely out for a very long time
            return 0.0

        # Find most recent game WHERE PLAYER ACTUALLY PLAYED (minutes > 0)
        most_recent_played = None
        for log in scratched.game_logs:
            # Skip DNP entries - player didn't actually play
            if log.minutes <= 0:
                continue

            game_date = log.date
            if isinstance(game_date, str):
                try:
                    game_date = datetime.strptime(game_date, "%Y-%m-%d")
                except:
                    continue
            if most_recent_played is None or game_date > most_recent_played:
                most_recent_played = game_date

        if most_recent_played is None:
            # No games with actual playing time = out for a long time
            return 0.0

        # Calculate days since last ACTUAL game
        today = datetime.now()
        if isinstance(most_recent_played, datetime):
            days_out = (today - most_recent_played).days
        else:
            return 0.5  # Default if we can't determine

        # Check if the game was from a previous season
        # NBA season runs Oct-Apr. If last game was before Oct 1 of current season year,
        # treat as long-term absence (player hasn't played this season)
        current_year = today.year
        if today.month >= 10:
            season_start = datetime(current_year, 10, 1)
        else:
            season_start = datetime(current_year - 1, 10, 1)

        if most_recent_played < season_start:
            # Player hasn't played this season at all - no boost
            # Teammates have fully adjusted over months
            return 0.0

        # Convert days to approximate games (roughly 1 game every 2 days during season)
        games_out = days_out // 2

        # Apply time-decay based on games missed
        if games_out <= self.INJURY_FRESH_GAMES:
            return 1.0  # Full boost
        elif games_out <= self.INJURY_RECENT_GAMES:
            return 0.75  # Recent injury
        elif games_out <= self.INJURY_MEDIUM_GAMES:
            return 0.40  # Team partially adjusted
        elif games_out <= self.INJURY_OLD_GAMES:
            return 0.15  # Team mostly adjusted
        else:
            return 0.0  # Fully adjusted, no boost

    def calculate_usage_boost(
        self,
        beneficiary: PlayerProfile,
        scratched: PlayerProfile,
        prop_type: PropType,
        team_profiles: List[PlayerProfile] = None
    ) -> float:
        """
        Calculate percentage boost when a teammate is out.

        Returns a MULTIPLIER (e.g., 1.08 = 8% boost), not raw points.
        This is more stable and prevents the crazy projections we were seeing.

        Key insight from backtest: High boosts (>1.10) actually hurt performance.
        The market already prices in injuries. Our edge is in being more accurate
        about WHICH players benefit and by how much.

        Uses data-driven ON/OFF splits when available, falls back to the
        hardcoded USAGE_REDISTRIBUTION matrix otherwise.
        """
        # Try data-driven redistribution first (from actual game logs)
        if self.lineup_model is not None:
            try:
                empirical = self.lineup_model.get_usage_redistribution_from_logs(
                    beneficiary, scratched, team_profiles
                )
                stat_key = STAT_MAPPING.get(prop_type)
                if empirical and stat_key and stat_key in empirical:
                    empirical_mult = empirical[stat_key]

                    # Apply time-decay: if player has been out for weeks,
                    # the recent averages already reflect the absence
                    time_decay = self._calculate_injury_time_decay(scratched)

                    # Scale the empirical boost by time decay
                    # e.g. empirical says 1.15 (15% boost), time_decay=0.4
                    # -> effective boost = 1.0 + (0.15 * 0.4) = 1.06
                    boost_portion = (empirical_mult - 1.0) * time_decay
                    # Cap at 8% per individual injury (same as hardcoded path)
                    capped_boost = max(-0.08, min(0.08, boost_portion))
                    return 1.0 + capped_boost
            except Exception as e:
                logger.debug(f"Data-driven redistribution failed, using fallback: {e}")

        # Fall back to hardcoded matrix if insufficient data
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

        # ROLE IMPACT: Weight by injured player's minutes (depth chart proxy)
        # A starter (34 mpg) being out is a massive deal.
        # A 12th man (8 mpg) being out is nearly irrelevant.
        scratched_mins = scratched.last_10_avg.get("min", 0)
        if scratched_mins == 0:
            scratched_mins = scratched.season_avg.get("min", 0)
            # Season avg minutes come as "MM:SS" string sometimes
            if isinstance(scratched_mins, str) and ":" in scratched_mins:
                try:
                    parts = scratched_mins.split(":")
                    scratched_mins = int(parts[0]) + int(parts[1]) / 60
                except (ValueError, IndexError):
                    scratched_mins = 0

        scratched_mins = float(scratched_mins) if scratched_mins else 0

        # Depth chart role classification by minutes:
        #   Star/Starter: 28+ min → full impact (1.0)
        #   Rotation:     18-28 min → moderate impact (0.4-0.8)
        #   Bench:        10-18 min → minimal impact (0.1-0.3)
        #   Deep bench:   <10 min → near-zero impact (0.0-0.05)
        if scratched_mins >= 28:
            role_weight = 1.0
        elif scratched_mins >= 18:
            role_weight = 0.4 + (scratched_mins - 18) / 10 * 0.4  # 0.4 to 0.8
        elif scratched_mins >= 10:
            role_weight = 0.1 + (scratched_mins - 10) / 8 * 0.2   # 0.1 to 0.3
        else:
            role_weight = max(0.0, scratched_mins / 10 * 0.05)     # 0.0 to 0.05

        # If the injured player barely plays, skip entirely
        if role_weight < 0.05:
            return 1.0

        # Get scratched player's production
        scratched_avg = scratched.last_10_avg.get(stat_key, 0)
        if scratched_avg == 0:
            scratched_avg = scratched.season_avg.get(stat_key, 0)

        # Get beneficiary's baseline for this stat
        beneficiary_avg = beneficiary.last_10_avg.get(stat_key, 0)
        if beneficiary_avg == 0:
            beneficiary_avg = beneficiary.season_avg.get(stat_key, 0)

        if beneficiary_avg == 0:
            return 1.0

        # Raw boost: scratched player's production * position redistribution / beneficiary baseline
        raw_boost_pct = (scratched_avg * base_factor) / beneficiary_avg

        # Apply role weight — bench player injuries barely matter
        raw_boost_pct *= role_weight

        # Apply usage weight — high-usage beneficiaries absorb more
        usage_weight = 1.0
        if beneficiary.usage_pct > 0:
            usage_weight = min(1.2, 0.8 + (beneficiary.usage_pct / 50))
        boost_pct = raw_boost_pct * usage_weight

        # Time-decay: if injured player has been out for weeks, team adjusted
        time_decay = self._calculate_injury_time_decay(scratched)
        boost_pct *= time_decay

        # Cap individual injury boost at 8%
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
        Also populates self.boost_sources for transparency:
            {player_name: [{"injured": name, "mpg": float, "boost_pct": float}, ...]}
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

        all_team_profiles = list(active_players) + list(scratched_players)

        boosts = {}
        self.boost_sources = {}  # Track which injuries cause each boost

        for active in active_players:
            player_boosts = {}
            player_sources = []

            pos_group = self._get_position_group(active.position)
            scarcity_mult = team_scarcity.get(active.team, {}).get(pos_group, 1.0)

            for prop_type in prop_types:
                combined_multiplier = 1.0

                for scratched in scratched_players:
                    if active.team != scratched.team:
                        continue

                    boost_mult = self.calculate_usage_boost(
                        active, scratched, prop_type, all_team_profiles
                    )

                    if boost_mult > 1.01:
                        combined_multiplier *= boost_mult
                        # Track the source (only once per injured player)
                        scratched_mins = scratched.last_10_avg.get("min", 0)
                        if isinstance(scratched_mins, str) and ":" in scratched_mins:
                            try:
                                parts = scratched_mins.split(":")
                                scratched_mins = int(parts[0]) + int(parts[1]) / 60
                            except (ValueError, IndexError):
                                scratched_mins = 0
                        scratched_mins = float(scratched_mins) if scratched_mins else 0

                        if not any(s["injured"] == scratched.name for s in player_sources):
                            player_sources.append({
                                "injured": scratched.name,
                                "position": scratched.position,
                                "mpg": round(scratched_mins, 1),
                                "ppg": round(scratched.last_10_avg.get("pts", 0), 1),
                                "boost_pct": round((boost_mult - 1.0) * 100, 1),
                            })

                if combined_multiplier > 1.01:
                    boost_portion = combined_multiplier - 1.0
                    scarcity_adjusted = 1.0 + (boost_portion * min(1.5, scarcity_mult))
                    final_multiplier = min(1.10, scarcity_adjusted)
                    player_boosts[prop_type] = final_multiplier

            if player_boosts:
                boosts[active.name] = player_boosts
            if player_sources:
                self.boost_sources[active.name] = player_sources

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

    def __init__(self, pipeline: DataPipeline, use_quantile: bool = False):
        self.pipeline = pipeline
        self.projection_engine = ProjectionEngine(pipeline)
        self.edge_calculator = EdgeCalculator()
        self.correlation_engine = CorrelationEngine(pipeline)
        self.lineup_model = LineupImpactModel()
        self.popoff_hunter = PopOffHunter(pipeline, self.lineup_model)
        self.quantile_engine = QuantileEngine()
        self.use_quantile = use_quantile

    def create_projection(
        self,
        profile: PlayerProfile,
        prop_type: PropType,
        context: GameContext,
        usage_boost: float = 1.0,
        lineup_adj: Dict[str, float] = None
    ) -> Projection:
        """Create a full projection for a player prop

        Args:
            usage_boost: Multiplier for injury opportunity (1.0 = no boost, 1.10 = 10% boost)
            lineup_adj: Per-stat multipliers from lineup context model
                        e.g. {"pts": 1.05, "ast": 0.97}
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

        # Calculate pace factor (kept for legacy compat; project_stat
        # now prefers context.total via game environment factor)
        team_pace = context.home_team.pace if is_home else context.away_team.pace
        opp_pace = context.away_team.pace if is_home else context.home_team.pace
        expected_pace = (team_pace + opp_pace) / 2
        pace_factor = expected_pace / AdjustmentFactors.LEAGUE_AVG_PACE

        # Project the stat -- pass full context and opponent TeamProfile
        # so project_stat can use Vegas total and position-specific defense
        projected_value, adjustments = self.projection_engine.project_stat(
            profile, prop_type, proj_minutes, is_home,
            opponent.defensive_rating, pace_factor,
            context=context, opponent_team=opponent
        )

        # Apply lineup context adjustment (ON/OFF splits)
        lineup_adj_value = 0.0
        if lineup_adj:
            lineup_mult = self._get_lineup_multiplier(prop_type, lineup_adj)
            if lineup_mult != 1.0:
                lineup_adj_value = projected_value * (lineup_mult - 1.0)
                projected_value *= lineup_mult

        # Apply usage boost from injuries as a MULTIPLIER
        # BUT: Reduce/skip if roster_filtering_applied is True
        # When roster filtering is used, the base projection already reflects
        # production without injured teammates (from filtered game logs)
        effective_usage_boost = usage_boost
        if profile.roster_filtering_applied and profile.roster_filtered_games_count >= 3:
            # Roster-filtered averages already account for injuries
            # Apply only a small fraction of the boost for very fresh injuries
            # (in case the filtered games don't fully capture the opportunity yet)
            if usage_boost > 1.0:
                # Reduce boost to 25% of original (e.g., 1.10 -> 1.025)
                effective_usage_boost = 1.0 + (usage_boost - 1.0) * 0.25
                projected_value *= effective_usage_boost
                logger.debug(
                    f"{profile.name}: Roster filtering applied, "
                    f"reducing injury boost {usage_boost:.3f} -> {effective_usage_boost:.3f}"
                )
        elif usage_boost > 1.0:
            effective_usage_boost = usage_boost
            projected_value *= usage_boost

        # Get std dev
        std_dev = self.projection_engine.calculate_std_dev(profile, prop_type)

        # Detect role change
        _, role_direction = self.projection_engine.detect_role_change(profile)

        # Create projection
        # Store effective_usage_boost (may be reduced if roster filtering applied)
        # source_layer indicates if boost was applied and whether it was reduced
        has_boost = effective_usage_boost > 1.0
        roster_filtered_boost = profile.roster_filtering_applied and usage_boost > 1.0

        # Determine source layer for debugging/tracking
        if roster_filtered_boost:
            source_layer = "roster_filtered"  # Boost was reduced due to roster filtering
        elif has_boost:
            source_layer = "popoff"  # Full injury boost applied
        else:
            source_layer = "baseline"  # No injury boost

        proj = Projection(
            player_name=profile.name,
            prop_type=prop_type,
            team=profile.team,  # For parlay correlation tracking
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
            tracking_adjustment=round(adjustments.get("tracking", 0), 2),  # From NBA.com tracking
            pbp_adjustment=round(adjustments.get("pbp", 0), 2),  # From PBPStats.com
            ctg_adjustment=round(adjustments.get("ctg", 0), 2),  # From Cleaning the Glass
            lineup_adjustment=round(lineup_adj_value, 2),
            usage_boost=round(effective_usage_boost, 3),  # Effective boost (may be reduced)
            source_layer=source_layer,
            games_sample=len(profile.game_logs),
            role_change=role_direction
        )

        return proj

    def _get_lineup_multiplier(
        self,
        prop_type: PropType,
        lineup_adj: Dict[str, float]
    ) -> float:
        """
        Get the lineup adjustment multiplier for a specific prop type.

        For single stats, use the direct stat key mapping.
        For combined stats (PRA, PR, etc.), average the component multipliers.
        """
        if prop_type in STAT_MAPPING:
            stat_key = STAT_MAPPING[prop_type]
            return lineup_adj.get(stat_key, 1.0)

        if prop_type in COMBINED_STAT_MAPPING:
            stat_keys = COMBINED_STAT_MAPPING[prop_type]
            multipliers = [lineup_adj.get(k, 1.0) for k in stat_keys]
            if multipliers:
                return sum(multipliers) / len(multipliers)

        return 1.0

    def add_market_data(
        self,
        projection: Projection,
        prop: PlayerProp,
        profile: PlayerProfile = None
    ) -> Projection:
        """Add market data and calculate edge.

        When use_quantile is enabled and a player profile is provided, blends
        empirical quantile-based probability with the z-score estimate.
        """
        projection.dk_line = prop.line
        projection.dk_over_odds = prop.over_odds
        projection.dk_under_odds = prop.under_odds
        # Cross-book data (from PropLine multi-book fetch)
        projection.best_over_book = getattr(prop, 'best_over_book', 'draftkings')
        projection.best_under_book = getattr(prop, 'best_under_book', 'draftkings')
        projection.all_books = getattr(prop, 'all_books', {})

        # Determine stat_type for calibration
        stat_type = STAT_MAPPING.get(projection.prop_type, "pts")
        if projection.prop_type == PropType.PRA:
            stat_type = "pra"

        # Z-score based probabilities (always computed as baseline)
        edge_over, edge_under, zscore_over, zscore_under = self.edge_calculator.calculate_edge(
            projection.projected_value,
            prop.line,
            projection.std_dev,
            prop.over_odds,
            prop.under_odds,
            stat_type
        )

        our_over = zscore_over
        our_under = zscore_under

        # Blend with quantile-based probability when enabled and data available
        if self.use_quantile and profile is not None and profile.game_logs:
            prop_type_str = projection.prop_type.value
            num_logs = self.quantile_engine.get_num_qualifying_games(profile.game_logs)

            if num_logs >= 3:
                empirical_over = self.quantile_engine.calculate_over_probability(
                    player_name=profile.name,
                    prop_type=prop_type_str,
                    line=prop.line,
                    projection=projection.projected_value,
                    game_logs=profile.game_logs,
                    position=profile.position,
                )

                # Adaptive blend weight based on sample size
                if num_logs >= FULL_BLEND_GAMES:
                    empirical_weight = 0.7
                else:
                    empirical_weight = max(0.3, num_logs / FULL_BLEND_GAMES * 0.7)

                blended_over = empirical_weight * empirical_over + (1 - empirical_weight) * zscore_over

                # Apply same data-driven damping as EdgeCalculator
                if blended_over > 0.60:
                    blended_over = 0.60 + (blended_over - 0.60) * 0.30
                elif blended_over < 0.40:
                    blended_over = 0.40 - (0.40 - blended_over) * 0.30
                blended_over = max(0.32, min(0.68, blended_over))

                our_over = blended_over
                our_under = 1.0 - our_over

                # Recalculate edge with blended probabilities
                implied_over = self.edge_calculator.american_to_prob(prop.over_odds)
                implied_under = self.edge_calculator.american_to_prob(prop.under_odds)
                edge_over = (our_over - implied_over) * 100
                edge_under = (our_under - implied_under) * 100
                edge_over = min(edge_over, 12.0)
                edge_under = min(edge_under, 12.0)

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

        # Build active player profiles IN PARALLEL (biggest speed win)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        profiles = []
        # Dedup player names (can get duplicates from overlapping prop sources)
        all_names = list(dict.fromkeys(player_names))  # preserves order, removes dupes
        if injured_player_names:
            for name in injured_player_names:
                if name not in all_names:
                    all_names.append(name)

        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_name = {
                executor.submit(self.pipeline.build_player_profile, name): name
                for name in all_names
            }
            profile_map = {}
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    profile = future.result()
                    if profile:
                        profile_map[name] = profile
                except Exception as e:
                    logger.warning(f"Failed to build profile for {name}: {e}")

        seen_names = set()
        for name in player_names:
            if name in profile_map and name not in seen_names:
                profiles.append(profile_map[name])
                context.players[name] = profile_map[name]
                seen_names.add(name)

        active = [p for p in profiles if p.status not in ["out", "doubtful"]]

        # Injured profiles (already fetched in parallel above)
        scratched = []
        if injured_player_names:
            for name in injured_player_names:
                if name in profile_map:
                    scratched.append(profile_map[name])

        # Apply roster-aware filtering to active players
        # This prevents double-counting injury boosts by using game logs
        # from games where the injured teammates were also out
        if scratched:
            # Group injured players by team
            injured_by_team: Dict[str, List[str]] = {}
            for inj_profile in scratched:
                team = inj_profile.team
                if team not in injured_by_team:
                    injured_by_team[team] = []
                injured_by_team[team].append(inj_profile.name)

            # Apply roster filtering to each active player
            for profile in active:
                team_injured = injured_by_team.get(profile.team, [])
                if team_injured:
                    self.pipeline.apply_roster_filtering_to_profile(profile, team_injured)

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

        # Calculate lineup context adjustments (ON/OFF splits)
        # Needs all team profiles (active + scratched) to build game rosters
        all_team_profiles = list(active) + list(scratched)
        lineup_adjustments = {}
        for profile in active:
            try:
                adj = self.lineup_model.calculate_lineup_adjustment(
                    profile, active, scratched, all_team_profiles
                )
                if adj and any(v != 1.0 for v in adj.values()):
                    lineup_adjustments[profile.name] = adj
                    # Log significant lineup adjustments
                    for stat_key, mult in adj.items():
                        if abs(mult - 1.0) >= 0.03:  # Log 3%+ adjustments
                            pct = (mult - 1.0) * 100
                            direction = "+" if pct > 0 else ""
                            logger.info(
                                f"Lineup: {profile.name} {direction}{pct:.1f}% {stat_key} "
                                f"(from ON/OFF splits)"
                            )
            except Exception as e:
                logger.warning(f"Lineup adjustment failed for {profile.name}: {e}")
        results["lineup_adjustments"] = lineup_adjustments

        # Create projections for each player/prop
        for profile in active:
            for prop_type in prop_types:
                # Get usage boost multiplier if applicable (default 1.0 = no boost)
                usage_boost = popoff_boosts.get(profile.name, {}).get(prop_type, 1.0)

                # Get lineup adjustment for this player (if any)
                lineup_adj = lineup_adjustments.get(profile.name)

                projection = self.create_projection(
                    profile, prop_type, context, usage_boost, lineup_adj
                )

                # Find matching DK prop
                for prop in context.props:
                    prop_name = prop.player_name.lower()
                    profile_name = profile.name.lower()

                    # Match player name and prop type
                    if prop_name in profile_name or profile_name in prop_name:
                        if self._match_prop_type(prop.prop_type, prop_type):
                            projection = self.add_market_data(projection, prop, profile)
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
        dk_lower = dk_prop_type.lower().strip()

        # Use EXACT matching to avoid "reb" matching "pts_rebs_asts"
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

        # Exact match required to prevent "rebounds" matching "pts_rebs_asts"
        for term in matches.get(our_prop_type, []):
            if dk_lower == term:
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
