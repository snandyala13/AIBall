#!/usr/bin/env python3
"""
ML Filter Module for NBA Prop Betting Model

This module provides prop-specific ML models that adjust probability estimates
based on historical backtest performance. Each prop type (points, rebounds,
assists, threes, pts_rebs_asts) has its own trained model with custom features.

For props without specific models, falls back to unified ensemble or raw probability.

Usage:
    from ml_filter import MLFilter

    ml = MLFilter()
    adjusted_prob = ml.get_ensemble_probability(
        probability=0.62,
        edge=8.5,
        confidence=0.85,
        diff_pct=15.0,
        line=25.5,
        is_over=True,
        prop_type='points',
        projected_minutes=34.5,
        std_dev=5.2
    )

    # Check if pick passes ML filter
    if ml.passes_filter(adjusted_prob, threshold=0.58):
        # Include in recommendations
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from typing import Optional, Dict, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Default thresholds based on backtest analysis
DEFAULT_THRESHOLD = 0.55      # v10 default
HIGH_CONF_THRESHOLD = 0.58    # High confidence tier
ELITE_THRESHOLD = 0.62        # Elite tier

# ML Confidence Tiers (based on v10 model validation)
# These define pick quality based on ML probability output
ML_TIERS = {
    'elite': {
        'min_prob': 0.58,
        'expected_hr': 0.65,
        'label': 'ELITE',
        'parlay_priority': 1,  # Must have in parlays
        'description': 'Must-have for parlays'
    },
    'high': {
        'min_prob': 0.53,
        'expected_hr': 0.62,
        'label': 'HIGH',
        'parlay_priority': 2,
        'description': 'Strong single bet'
    },
    'standard': {
        'min_prob': 0.48,
        'expected_hr': 0.58,
        'label': 'STD',
        'parlay_priority': 3,
        'description': 'Standard pick'
    },
    'low': {
        'min_prob': 0.0,
        'expected_hr': 0.50,
        'label': 'LOW',
        'parlay_priority': 4,
        'description': 'Below average - consider fading'
    }
}

# Props with dedicated models
PROP_SPECIFIC_MODELS = ['points', 'rebounds', 'assists', 'threes', 'pts_rebs_asts']

# Prop-specific optimal thresholds from v10 training
# Trained on 34K legs (Oct 2024 - Jan 2026), validated on last 30%
# v10 uses richer features and per-prop threshold optimization
PROP_THRESHOLDS = {
    'points': 0.53,       # ROI: +18.2% on test set
    'rebounds': 0.50,     # ROI: +19.0% on test set
    'assists': 0.48,      # ROI: +16.6% on test set
    'threes': 0.50,       # ROI: +25.0% on test set
    'pts_rebs_asts': 0.56 # ROI: +21.3% on test set
}

# Props where raw probability should be used instead of ML
# v10 models handle all props — empty set
USE_RAW_PROBABILITY = set()


class MLFilter:
    """
    Prop-specific ML filter for adjusting prop bet probabilities.

    Uses dedicated models for each major prop type (points, rebounds, assists,
    threes, pts_rebs_asts). Each model was trained on prop-specific features
    that capture unique dynamics (e.g., threes prefer UNDERs, PRA prefers OVERs).

    Falls back to unified ensemble for other props (steals, blocks, etc.).
    """

    def __init__(self, models_dir: str = None):
        """
        Initialize the ML filter by loading trained models.

        Args:
            models_dir: Path to directory containing saved models.
                       Defaults to 'models/' in the project root.
        """
        if models_dir is None:
            models_dir = Path(__file__).parent / 'models'
        else:
            models_dir = Path(models_dir)

        self.models_dir = models_dir
        self.loaded = False

        # Prop-specific models
        self.prop_models = {}
        self.prop_metadata = None
        self.prop_models_loaded = False

        # Legacy unified ensemble (fallback)
        self.gb = None
        self.rf = None
        self.lr = None
        self.scaler = None
        self.metadata = None
        self.feature_cols = None
        self.legacy_loaded = False

        # Per-prop optimized ensemble weights
        self.ensemble_weights = {}
        self._default_weights = [1/3, 1/3, 1/3]

        # Try to load models
        self._load_models()
        self._load_ensemble_weights()

    def _load_models(self) -> bool:
        """Load trained models from disk."""
        # First, try v10 models (best, most recent)
        self._load_v10_models()

        # If v10 not available, try older prop-specific models
        if not self.prop_models_loaded:
            self._load_prop_specific_models()

        # Then, try to load legacy ensemble (fallback)
        self._load_legacy_ensemble()

        # At least one system must be loaded
        self.loaded = self.prop_models_loaded or self.legacy_loaded
        self.is_stale = False
        self.model_age_days = 0

        if self.loaded:
            # Check model staleness
            self._check_staleness()
            if self.prop_models_loaded:
                logger.info(f"Prop-specific models loaded: {list(self.prop_models.keys())}")
            if self.legacy_loaded:
                logger.info("Legacy ensemble loaded (fallback)")
            if self.is_stale:
                logger.warning(
                    f"ML models are {self.model_age_days} days old — "
                    f"predictions may be less reliable for current season"
                )
        else:
            logger.warning("No ML models loaded. Filter will be disabled.")

        return self.loaded

    def _check_staleness(self):
        """Check if models are older than 90 days."""
        from datetime import datetime
        metadata_files = [
            self.models_dir / 'v10' / 'metadata.json',
            self.models_dir / 'prop_models_v2_metadata.json',
            self.models_dir / 'prop_models_metadata.json',
        ]
        for mf in metadata_files:
            if mf.exists():
                try:
                    import json
                    with open(mf) as f:
                        meta = json.load(f)
                    trained_date = meta.get('trained_date') or meta.get('timestamp', '')
                    if trained_date:
                        dt = datetime.fromisoformat(trained_date.replace('Z', '+00:00'))
                        self.model_age_days = (datetime.now(dt.tzinfo) - dt).days
                        if self.model_age_days > 90:
                            self.is_stale = True
                        return
                except Exception:
                    pass
        # If we can't determine age, check file modification time
        for mf in metadata_files:
            if mf.exists():
                import os
                mtime = datetime.fromtimestamp(os.path.getmtime(mf))
                self.model_age_days = (datetime.now() - mtime).days
                if self.model_age_days > 90:
                    self.is_stale = True
                return

    def _load_v10_models(self) -> bool:
        """Load v10 models from models/v10/ directory.

        v10 models use a richer feature set with prop-type indicators,
        interaction features, and per-prop optimized thresholds.
        """
        v10_dir = self.models_dir / 'v10'
        v10_metadata_path = v10_dir / 'metadata.json'

        if not v10_metadata_path.exists():
            logger.debug("v10 models not found")
            return False

        try:
            with open(v10_metadata_path, 'r') as f:
                self.prop_metadata = json.load(f)

            self.model_version = 'v10'
            self.v10_feature_cols = self.prop_metadata.get('feature_cols', [])

            # Load per-prop models
            models_info = self.prop_metadata.get('models', {})
            for prop_type, info in models_info.items():
                if not info.get('use_model', True):
                    continue
                model_file = info.get('model_file')
                if model_file:
                    model_path = v10_dir / model_file
                    if model_path.exists():
                        self.prop_models[prop_type] = {
                            'model': joblib.load(model_path),
                            'feature_names': info.get('feature_names', []),
                            'config': {},
                            'metrics': info.get('metrics', {}),
                            'threshold': info.get('threshold', 0.55),
                        }
                        # Update PROP_THRESHOLDS from metadata
                        PROP_THRESHOLDS[prop_type] = info.get('threshold', PROP_THRESHOLDS.get(prop_type, 0.55))

            # Load unified fallback
            unified_info = self.prop_metadata.get('unified_model', {})
            unified_file = unified_info.get('model_file')
            if unified_file:
                unified_path = v10_dir / unified_file
                if unified_path.exists():
                    self._v10_unified_model = joblib.load(unified_path)
                    self._v10_unified_threshold = unified_info.get('threshold', 0.53)

            self.prop_models_loaded = len(self.prop_models) > 0
            if self.prop_models_loaded:
                logger.info(f"Loaded v10 models for: {list(self.prop_models.keys())}")

            return self.prop_models_loaded

        except Exception as e:
            logger.warning(f"Error loading v10 models: {e}")
            return False

    def _build_v10_features(self, probability: float, edge: float,
                            confidence: float, diff_pct: float,
                            line: float, is_over: bool,
                            prop_type: str,
                            projected_minutes: float = 32.0,
                            std_dev: float = 5.0,
                            implied_prob: float = 0.5) -> pd.DataFrame:
        """Build feature DataFrame matching v10 training format.

        v10 features include prop-type indicators, interaction terms, and CV.
        """
        features = {}

        # Base model signals
        features['probability'] = probability
        features['edge'] = edge
        features['diff_pct'] = diff_pct
        features['confidence'] = confidence if confidence else 0.8
        features['std_dev'] = std_dev

        # Prop-type one-hot
        features['is_points'] = 1.0 if prop_type == 'points' else 0.0
        features['is_rebounds'] = 1.0 if prop_type == 'rebounds' else 0.0
        features['is_assists'] = 1.0 if prop_type == 'assists' else 0.0
        features['is_threes'] = 1.0 if prop_type == 'threes' else 0.0
        features['is_pra'] = 1.0 if prop_type == 'pts_rebs_asts' else 0.0

        # Player quality
        features['line_value'] = line
        features['projected_minutes'] = projected_minutes

        # Coefficient of variation
        denom = line + 0.5 if line > 0 else 1.0
        features['cv'] = min(std_dev / denom, 3.0)

        # Game context
        features['is_over'] = 1.0 if is_over else 0.0
        features['implied_prob'] = implied_prob

        # Derived interactions
        features['edge_squared'] = edge ** 2
        features['prob_x_edge'] = probability * edge
        features['edge_x_line'] = edge * line

        # Direction-prop interactions
        features['over_points'] = features['is_over'] * features['is_points']
        features['under_rebounds'] = (1.0 - features['is_over']) * features['is_rebounds']
        features['over_pra'] = features['is_over'] * features['is_pra']

        df = pd.DataFrame([features])

        # Ensure correct column order from metadata
        feature_names = self.v10_feature_cols if hasattr(self, 'v10_feature_cols') and self.v10_feature_cols else list(features.keys())
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]

        return df

    def _load_ensemble_weights(self):
        """Load per-prop optimized ensemble weights from optimize_ensemble.py output."""
        weights_path = self.models_dir / 'ensemble_weights.json'
        try:
            with open(weights_path, 'r') as f:
                data = json.load(f)
            self.ensemble_weights = data.get('weights', {})
            self._default_weights = data.get('default_weights', [1/3, 1/3, 1/3])
            logger.info(f"Loaded optimized ensemble weights for {len(self.ensemble_weights)} prop types")
        except FileNotFoundError:
            logger.debug("No ensemble_weights.json found, using equal weighting")
        except Exception as e:
            logger.warning(f"Error loading ensemble weights: {e}")

    def _load_prop_specific_models(self) -> bool:
        """Load prop-specific models from disk. Prefers v2 models with game features."""
        try:
            # Try v2 models first (with game-level features)
            v2_metadata_path = self.models_dir / 'prop_models_v2_metadata.json'
            v1_metadata_path = self.models_dir / 'prop_models_metadata.json'

            if v2_metadata_path.exists():
                with open(v2_metadata_path, 'r') as f:
                    self.prop_metadata = json.load(f)
                self.model_version = 'v2'
                logger.info("Loading v2 models with game-level features")
            elif v1_metadata_path.exists():
                with open(v1_metadata_path, 'r') as f:
                    self.prop_metadata = json.load(f)
                self.model_version = 'v1'
                logger.info("Loading v1 models (no game features)")
            else:
                logger.debug("Prop-specific models metadata not found")
                return False

            # Load each prop model
            models_info = self.prop_metadata.get('models', {})
            for prop_type, info in models_info.items():
                # Skip models marked as disabled
                if not info.get('use_model', True):
                    logger.debug(f"Skipping {prop_type} model (disabled in metadata)")
                    continue

                model_file = info.get('model_file')
                if model_file:
                    model_path = self.models_dir / model_file
                    if model_path.exists():
                        self.prop_models[prop_type] = {
                            'model': joblib.load(model_path),
                            'feature_names': info.get('feature_names', []),
                            'config': info.get('config', {}),
                            'metrics': info.get('metrics', {}),
                            'threshold': info.get('threshold', 0.58)
                        }
                        best_hr = info.get('metrics', {}).get('best_filtered_hr', 0)
                        logger.debug(f"Loaded {prop_type} model: {best_hr:.1%} best filtered HR")

            self.prop_models_loaded = len(self.prop_models) > 0
            if self.prop_models_loaded:
                version = self.prop_metadata.get('version', 'unknown')
                logger.info(f"Loaded {len(self.prop_models)} prop-specific models ({version})")

            return self.prop_models_loaded

        except Exception as e:
            logger.warning(f"Error loading prop-specific models: {e}")
            self.prop_models_loaded = False
            return False

    def _load_legacy_ensemble(self) -> bool:
        """Load legacy unified ensemble models (fallback for props without specific models)."""
        try:
            self.gb = joblib.load(self.models_dir / 'gradient_boosting.joblib')
            self.rf = joblib.load(self.models_dir / 'random_forest.joblib')
            self.lr = joblib.load(self.models_dir / 'logistic_regression.joblib')
            self.scaler = joblib.load(self.models_dir / 'scaler.joblib')

            with open(self.models_dir / 'ensemble_metadata.json', 'r') as f:
                self.metadata = json.load(f)

            self.feature_cols = self.metadata['feature_cols']
            self.legacy_loaded = True
            return True

        except FileNotFoundError:
            logger.debug("Legacy ensemble models not found")
            self.legacy_loaded = False
            return False
        except Exception as e:
            logger.warning(f"Error loading legacy ensemble: {e}")
            self.legacy_loaded = False
            return False

    def _build_prop_specific_features(self, probability: float, edge: float,
                                        confidence: float, diff_pct: float,
                                        line: float, is_over: bool,
                                        prop_type: str,
                                        projected_minutes: float = 32.0,
                                        std_dev: float = 5.0,
                                        game_total_pts: float = 70.0,
                                        game_avg_mins: float = 32.0,
                                        game_num_props: int = 20,
                                        implied_prob: float = 0.5) -> pd.DataFrame:
        """
        Build feature DataFrame for prop-specific models.

        Args:
            probability: Original model probability (0-1)
            edge: Edge percentage (e.g., 8.5 for 8.5%)
            confidence: Model confidence score (0-1)
            diff_pct: Percentage difference from line
            line: The betting line value
            is_over: True for OVER picks, False for UNDER
            prop_type: Type of prop ('points', 'rebounds', etc.)
            projected_minutes: Player's projected minutes
            std_dev: Standard deviation of the stat
            game_total_pts: Estimated total points in the game (v2 feature)
            game_avg_mins: Average projected minutes across all players (v2 feature)
            game_num_props: Number of props available in the game (v2 feature)

        Returns:
            DataFrame with features matching prop-specific model training format
        """
        model_info = self.prop_models.get(prop_type, {})
        feature_names = model_info.get('feature_names', [])
        config = model_info.get('config', {})

        # Base features that all prop models use
        features = {
            'probability': probability,
            'edge': edge,
            'confidence': confidence if confidence else 0.8,
            'diff_pct': diff_pct,
            'line': line,
            'is_over': 1 if is_over else 0,
            'projected_minutes': projected_minutes,
            'std_dev': std_dev,
            'implied_prob': implied_prob,  # DraftKings juice - helps model avoid bad lines
        }

        # Derived features
        features['edge_squared'] = edge ** 2
        features['prob_x_edge'] = probability * edge

        # Prop-specific features
        # Points
        features['high_line'] = 1 if line >= 25 else 0
        features['star_player'] = 1 if line >= 28 else 0

        # Rebounds/Threes
        features['low_line'] = 1 if line <= 6 else 0
        features['very_low_line'] = 1 if line <= 2.5 else 0
        features['high_minutes'] = 1 if projected_minutes >= 34 else 0

        # Assists
        features['high_prob'] = 1 if probability >= 0.6 else 0

        # Threes (penalize OVERs)
        features['is_over_penalty'] = (1 if is_over else 0) * edge

        # PRA (boost OVERs)
        features['is_over_boost'] = (1 if is_over else 0) * 1.0

        # V2: Game-level features
        features['game_total_pts'] = game_total_pts
        features['game_avg_mins'] = game_avg_mins
        features['game_num_props'] = game_num_props
        features['high_mins_game'] = 1 if game_avg_mins >= 33 else 0
        features['high_total_game'] = 1 if game_total_pts >= 80 else 0

        # Create DataFrame
        df = pd.DataFrame([features])

        # Ensure all expected columns exist and match order
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0

        # Reorder to match training
        if feature_names:
            df = df[feature_names]

        return df

    def _build_legacy_features(self, probability: float, edge: float, confidence: float,
                               diff_pct: float, line: float, is_over: bool,
                               prop_type: str) -> pd.DataFrame:
        """
        Build feature DataFrame matching legacy ensemble training format.

        Args:
            probability: Original model probability (0-1)
            edge: Edge percentage (e.g., 8.5 for 8.5%)
            confidence: Model confidence score (0-1)
            diff_pct: Percentage difference from line
            line: The betting line value
            is_over: True for OVER picks, False for UNDER
            prop_type: Type of prop ('points', 'rebounds', etc.)

        Returns:
            DataFrame with features matching training format
        """
        # Known prop types from training
        known_props = self.metadata.get('prop_types',
            ['assists', 'points', 'pts_rebs_asts', 'rebounds', 'threes'])

        # Build base features
        features = {
            'probability': probability,
            'edge': edge,
            'confidence': confidence if confidence else 0.8,
            'diff_pct': diff_pct,
            'line': line,
            'is_over': 1 if is_over else 0,
        }

        # Add prop type dummies (all zeros first)
        for prop in known_props:
            features[f'prop_{prop}'] = 0

        # Set the actual prop type
        prop_key = f'prop_{prop_type}'
        if prop_key in features:
            features[prop_key] = 1

        # Add derived features
        features['line_percentile'] = 0.5  # Default to median (will be approximate)
        features['edge_squared'] = edge ** 2
        features['prob_x_edge'] = probability * edge
        features['edge_over_interaction'] = edge * (1 if is_over else 0)
        features['high_edge'] = 1 if edge >= 15 else 0
        features['mid_line'] = 1 if 15 <= line <= 35 else 0

        # Create DataFrame with correct column order
        df = pd.DataFrame([features])

        # Ensure all expected columns exist
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0

        # Reorder to match training
        df = df[self.feature_cols]

        return df

    def get_ensemble_probability(self, probability: float, edge: float,
                                  confidence: float, diff_pct: float,
                                  line: float, is_over: bool,
                                  prop_type: str,
                                  projected_minutes: float = 32.0,
                                  std_dev: float = 5.0,
                                  game_total_pts: float = 70.0,
                                  game_avg_mins: float = 32.0,
                                  game_num_props: int = 20,
                                  implied_prob: float = 0.5) -> float:
        """
        Get ML-adjusted probability for a pick.

        Uses prop-specific models when available, otherwise falls back to
        legacy ensemble or returns original probability.

        Args:
            probability: Original model probability (0-1)
            edge: Edge percentage
            confidence: Model confidence score
            diff_pct: Percentage difference from line
            line: The betting line value
            is_over: True for OVER, False for UNDER
            prop_type: Type of prop
            projected_minutes: Player's projected minutes (for prop-specific models)
            std_dev: Standard deviation of the stat (for prop-specific models)
            game_total_pts: Estimated total points in the game (v2 feature)
            game_avg_mins: Average projected minutes across all players (v2 feature)
            game_num_props: Number of props available in the game (v2 feature)

        Returns:
            ML-adjusted probability (0-1)
        """
        if not self.loaded:
            logger.debug("ML filter not loaded, returning original probability")
            return probability

        # Check if this prop type should use raw probability (model is anti-predictive)
        if prop_type in USE_RAW_PROBABILITY:
            logger.debug(f"Using raw probability for {prop_type} (model disabled)")
            return probability

        # Try prop-specific model first
        if self.prop_models_loaded and prop_type in self.prop_models:
            return self._get_prop_specific_probability(
                probability=probability,
                edge=edge,
                confidence=confidence,
                diff_pct=diff_pct,
                line=line,
                is_over=is_over,
                prop_type=prop_type,
                projected_minutes=projected_minutes,
                std_dev=std_dev,
                game_total_pts=game_total_pts,
                game_avg_mins=game_avg_mins,
                game_num_props=game_num_props,
                implied_prob=implied_prob
            )

        # Fall back to legacy ensemble
        if self.legacy_loaded:
            return self._get_legacy_probability(
                probability=probability,
                edge=edge,
                confidence=confidence,
                diff_pct=diff_pct,
                line=line,
                is_over=is_over,
                prop_type=prop_type
            )

        # No model available, return original probability
        return probability

    def _get_prop_specific_probability(self, probability: float, edge: float,
                                        confidence: float, diff_pct: float,
                                        line: float, is_over: bool,
                                        prop_type: str,
                                        projected_minutes: float = 32.0,
                                        std_dev: float = 5.0,
                                        game_total_pts: float = 70.0,
                                        game_avg_mins: float = 32.0,
                                        game_num_props: int = 20,
                                        implied_prob: float = 0.5) -> float:
        """Get probability from prop-specific model."""
        try:
            model_info = self.prop_models[prop_type]
            model = model_info['model']

            # Use v10 feature builder if model version is v10
            if getattr(self, 'model_version', '') == 'v10':
                X = self._build_v10_features(
                    probability=probability,
                    edge=edge,
                    confidence=confidence,
                    diff_pct=diff_pct,
                    line=line,
                    is_over=is_over,
                    prop_type=prop_type,
                    projected_minutes=projected_minutes,
                    std_dev=std_dev,
                    implied_prob=implied_prob,
                )
            else:
                # Legacy v1/v2 feature builder
                X = self._build_prop_specific_features(
                    probability=probability,
                    edge=edge,
                    confidence=confidence,
                    diff_pct=diff_pct,
                    line=line,
                    is_over=is_over,
                    prop_type=prop_type,
                    projected_minutes=projected_minutes,
                    std_dev=std_dev,
                    game_total_pts=game_total_pts,
                    game_avg_mins=game_avg_mins,
                    game_num_props=game_num_props,
                    implied_prob=implied_prob,
                )

            # Get prediction
            prob = model.predict_proba(X)[0, 1]
            return float(prob)

        except Exception as e:
            logger.warning(f"Error in prop-specific model for {prop_type}: {e}")
            # Fall back to legacy if available
            if self.legacy_loaded:
                return self._get_legacy_probability(
                    probability, edge, confidence, diff_pct, line, is_over, prop_type
                )
            return probability

    def _get_legacy_probability(self, probability: float, edge: float,
                                 confidence: float, diff_pct: float,
                                 line: float, is_over: bool,
                                 prop_type: str) -> float:
        """Get probability from legacy ensemble model."""
        try:
            # Build features
            X = self._build_legacy_features(
                probability=probability,
                edge=edge,
                confidence=confidence,
                diff_pct=diff_pct,
                line=line,
                is_over=is_over,
                prop_type=prop_type
            )

            # Get predictions from each model
            gb_prob = self.gb.predict_proba(X)[0, 1]
            rf_prob = self.rf.predict_proba(X)[0, 1]

            # LogisticRegression needs scaled features
            X_scaled = self.scaler.transform(X)
            lr_prob = self.lr.predict_proba(X_scaled)[0, 1]

            # Weighted ensemble (optimized per-prop or equal fallback)
            weights = self.ensemble_weights.get(prop_type, self._default_weights)
            model_probs = [gb_prob, rf_prob, lr_prob]
            ensemble_prob = sum(w * p for w, p in zip(weights, model_probs))

            return float(ensemble_prob)

        except Exception as e:
            logger.warning(f"Error in legacy ensemble: {e}. Returning original probability.")
            return probability

    def passes_filter(self, ensemble_prob: float,
                      threshold: float = DEFAULT_THRESHOLD) -> bool:
        """
        Check if a pick passes the ML filter threshold.

        Args:
            ensemble_prob: The ensemble-adjusted probability
            threshold: Minimum probability to pass (default 0.58 = 68% hit rate)

        Returns:
            True if pick passes filter, False otherwise
        """
        return ensemble_prob >= threshold

    def get_filter_tier(self, ensemble_prob: float) -> Tuple[str, float]:
        """
        Get the quality tier for a pick based on ensemble probability.

        Returns:
            Tuple of (tier_name, expected_hit_rate)
        """
        if ensemble_prob >= ELITE_THRESHOLD:
            return ('elite', 0.65)
        elif ensemble_prob >= HIGH_CONF_THRESHOLD:
            return ('high', 0.62)
        elif ensemble_prob >= DEFAULT_THRESHOLD:
            return ('standard', 0.58)
        else:
            return ('below_threshold', 0.50)

    def get_model_info(self) -> Dict:
        """Get information about the loaded models."""
        if not self.loaded:
            return {'loaded': False}

        info = {
            'loaded': True,
            'model_version': getattr(self, 'model_version', 'unknown'),
            'prop_models_loaded': self.prop_models_loaded,
            'legacy_loaded': self.legacy_loaded,
            'thresholds': {
                'standard': {'threshold': DEFAULT_THRESHOLD, 'expected_hit_rate': 0.58},
                'high': {'threshold': HIGH_CONF_THRESHOLD, 'expected_hit_rate': 0.62},
                'elite': {'threshold': ELITE_THRESHOLD, 'expected_hit_rate': 0.65}
            }
        }

        # Add prop-specific model info
        if self.prop_models_loaded:
            info['prop_models'] = {}
            info['training_date'] = self.prop_metadata.get('trained_date') or self.prop_metadata.get('trained_at')
            for prop_type, model_info in self.prop_models.items():
                metrics = model_info.get('metrics', {})
                info['prop_models'][prop_type] = {
                    'accuracy': metrics.get('accuracy', 0),
                    'test_samples': metrics.get('test_samples', 0),
                    'test_hit_rate': metrics.get('test_hit_rate') or metrics.get('best_filtered_hr', 0),
                    'threshold': model_info.get('threshold', DEFAULT_THRESHOLD),
                    'auc': metrics.get('auc', 0),
                }

        # Add legacy ensemble info
        if self.legacy_loaded and self.metadata:
            info['legacy'] = {
                'training_date': self.metadata.get('training_date'),
                'training_samples': self.metadata.get('training_samples'),
                'base_hit_rate': self.metadata.get('base_hit_rate')
            }

        return info

    def get_model_for_prop(self, prop_type: str) -> str:
        """Get which model will be used for a given prop type."""
        if prop_type in USE_RAW_PROBABILITY:
            return "disabled (using raw probability)"
        if self.prop_models_loaded and prop_type in self.prop_models:
            return f"prop_specific ({prop_type})"
        elif self.legacy_loaded:
            return "legacy_ensemble"
        else:
            return "none (using raw probability)"

    def get_threshold_for_prop(self, prop_type: str) -> float:
        """Get the optimal threshold for a given prop type."""
        return PROP_THRESHOLDS.get(prop_type, DEFAULT_THRESHOLD) or DEFAULT_THRESHOLD

    def passes_prop_filter(self, ensemble_prob: float, prop_type: str) -> bool:
        """
        Check if a pick passes the ML filter using prop-specific threshold.

        Args:
            ensemble_prob: The ensemble-adjusted probability
            prop_type: Type of prop for threshold lookup

        Returns:
            True if pick passes filter, False otherwise
        """
        threshold = self.get_threshold_for_prop(prop_type)
        return ensemble_prob >= threshold

    def get_ml_tier(self, ml_prob: float) -> Dict:
        """
        Get the ML confidence tier for a pick based on ML probability.

        Args:
            ml_prob: The ML-adjusted probability

        Returns:
            Dict with tier info: {
                'tier': 'elite'|'high'|'standard'|'low',
                'label': 'ELITE'|'HIGH'|'STD'|'LOW',
                'expected_hr': float,
                'parlay_priority': int (1=must have, 4=avoid),
                'description': str
            }
        """
        for tier_name in ['elite', 'high', 'standard', 'low']:
            tier = ML_TIERS[tier_name]
            if ml_prob >= tier['min_prob']:
                return {
                    'tier': tier_name,
                    'label': tier['label'],
                    'expected_hr': tier['expected_hr'],
                    'parlay_priority': tier['parlay_priority'],
                    'description': tier['description'],
                    'ml_prob': ml_prob
                }

        # Fallback (shouldn't happen)
        return ML_TIERS['low']

    def classify_pick(self, probability: float, edge: float,
                      confidence: float, diff_pct: float,
                      line: float, is_over: bool,
                      prop_type: str,
                      projected_minutes: float = 32.0,
                      std_dev: float = 5.0,
                      game_total_pts: float = 70.0,
                      game_avg_mins: float = 32.0,
                      game_num_props: int = 20) -> Dict:
        """
        Classify a pick with ML probability and tier.

        Returns complete classification including:
        - ml_prob: The ML-adjusted probability
        - tier: Classification tier (elite/high/standard/low)
        - expected_hr: Expected hit rate for this tier
        - parlay_priority: 1 (must-have) to 4 (avoid)
        - is_parlay_lock: True if this is an elite pick for parlays

        Args:
            (same as get_ensemble_probability)

        Returns:
            Dict with full classification
        """
        ml_prob = self.get_ensemble_probability(
            probability=probability,
            edge=edge,
            confidence=confidence,
            diff_pct=diff_pct,
            line=line,
            is_over=is_over,
            prop_type=prop_type,
            projected_minutes=projected_minutes,
            std_dev=std_dev,
            game_total_pts=game_total_pts,
            game_avg_mins=game_avg_mins,
            game_num_props=game_num_props
        )

        # For props using raw probability (no ML model), use more conservative tiers
        # Raw probability isn't as calibrated as ML probability
        uses_ml = prop_type not in USE_RAW_PROBABILITY and prop_type in self.prop_models

        if uses_ml:
            tier_info = self.get_ml_tier(ml_prob)
        else:
            # More conservative tiers for raw probability
            # Raw prob of 65% ~ ML prob of 55% in terms of hit rate
            if ml_prob >= 0.65:
                tier_info = self.get_ml_tier(0.55)  # Downgrade to 'high'
                tier_info['ml_prob'] = ml_prob
            elif ml_prob >= 0.62:
                tier_info = self.get_ml_tier(0.52)  # Downgrade to 'standard'
                tier_info['ml_prob'] = ml_prob
            else:
                tier_info = self.get_ml_tier(0.45)  # Low tier
                tier_info['ml_prob'] = ml_prob

        return {
            'ml_prob': ml_prob,
            'raw_prob': probability,
            'tier': tier_info['tier'],
            'label': tier_info['label'],
            'expected_hr': tier_info['expected_hr'],
            'parlay_priority': tier_info['parlay_priority'],
            'description': tier_info['description'],
            'is_parlay_lock': tier_info['tier'] == 'elite' and uses_ml,
            'is_strong_single': tier_info['tier'] in ['elite', 'high'],
            'uses_ml_model': uses_ml,
            'prop_type': prop_type
        }


# Singleton instance for easy import
_ml_filter_instance = None

def get_ml_filter() -> MLFilter:
    """Get the singleton MLFilter instance."""
    global _ml_filter_instance
    if _ml_filter_instance is None:
        _ml_filter_instance = MLFilter()
    return _ml_filter_instance


if __name__ == "__main__":
    # Test the ML filter
    print("Testing ML Filter...")
    print("=" * 60)

    ml = MLFilter()
    print(f"Loaded: {ml.loaded}")
    print(f"  Prop-specific models: {ml.prop_models_loaded}")
    print(f"  Legacy ensemble: {ml.legacy_loaded}")

    if ml.loaded:
        # Show which models are available
        print("\nModel routing and thresholds:")
        for prop in ['points', 'rebounds', 'assists', 'threes', 'pts_rebs_asts', 'steals', 'blocks']:
            model = ml.get_model_for_prop(prop)
            threshold = ml.get_threshold_for_prop(prop)
            print(f"  {prop}: {model} (threshold: {threshold:.0%})")

        # Test with sample predictions
        test_cases = [
            {'probability': 0.65, 'edge': 8.0, 'confidence': 0.85, 'diff_pct': 15.0,
             'line': 25.5, 'is_over': True, 'prop_type': 'points',
             'projected_minutes': 34.0, 'std_dev': 5.5},
            {'probability': 0.62, 'edge': 18.0, 'confidence': 0.90, 'diff_pct': 30.0,
             'line': 2.5, 'is_over': True, 'prop_type': 'threes',
             'projected_minutes': 32.0, 'std_dev': 1.2},
            {'probability': 0.58, 'edge': 6.0, 'confidence': 0.82, 'diff_pct': 10.0,
             'line': 32.0, 'is_over': False, 'prop_type': 'pts_rebs_asts',
             'projected_minutes': 35.0, 'std_dev': 8.0},
            {'probability': 0.60, 'edge': 7.5, 'confidence': 0.80, 'diff_pct': 12.0,
             'line': 6.5, 'is_over': False, 'prop_type': 'rebounds',
             'projected_minutes': 30.0, 'std_dev': 2.5},
            {'probability': 0.55, 'edge': 5.0, 'confidence': 0.75, 'diff_pct': 8.0,
             'line': 5.5, 'is_over': True, 'prop_type': 'assists',
             'projected_minutes': 33.0, 'std_dev': 2.0},
        ]

        print("\n" + "-" * 60)
        print("Test predictions with ML Classification:")
        for i, tc in enumerate(test_cases, 1):
            # Use the new classify_pick method
            classification = ml.classify_pick(**tc)
            model_used = ml.get_model_for_prop(tc['prop_type'])

            print(f"\n  Case {i}: {tc['prop_type'].upper()} {'OVER' if tc['is_over'] else 'UNDER'} {tc['line']}")
            print(f"    Model: {model_used}")
            print(f"    Raw prob: {classification['raw_prob']:.1%} -> ML prob: {classification['ml_prob']:.1%}")
            print(f"    Tier: [{classification['label']}] {classification['description']}")
            print(f"    Expected HR: {classification['expected_hr']:.0%} | Parlay Priority: {classification['parlay_priority']}")
            if classification['is_parlay_lock']:
                print(f"    >>> PARLAY LOCK - Must include in parlays!")

        print("\n" + "=" * 60)
        print("Model Info:")
        info = ml.get_model_info()

        if info.get('prop_models'):
            print(f"\n  Prop-specific models (trained: {info.get('training_date', 'unknown')[:10]}):")
            for prop, metrics in info['prop_models'].items():
                print(f"    {prop}: {metrics['accuracy']:.1%} accuracy, {metrics['test_samples']} test samples")

        if info.get('legacy'):
            print(f"\n  Legacy ensemble:")
            print(f"    Training samples: {info['legacy'].get('training_samples', 'N/A'):,}")
            print(f"    Base hit rate: {info['legacy'].get('base_hit_rate', 0):.1%}")
