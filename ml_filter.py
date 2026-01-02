#!/usr/bin/env python3
"""
ML Filter Module for NBA Prop Betting Model

This module provides an ensemble ML model that adjusts probability estimates
based on historical backtest performance. The ensemble combines:
- GradientBoosting (best at high-confidence filtering)
- RandomForest (robust feature learning)
- LogisticRegression (interpretable baseline)

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
        prop_type='points'
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
DEFAULT_THRESHOLD = 0.58      # 68% hit rate, ~30 picks/game
HIGH_CONF_THRESHOLD = 0.60    # 72% hit rate, ~11 picks/game
ELITE_THRESHOLD = 0.65        # 84% hit rate, ~1 pick/game


class MLFilter:
    """
    Ensemble ML filter for adjusting prop bet probabilities.

    The model was trained on 17K+ historical predictions and learns to:
    - Correct overconfidence in high-edge picks
    - Identify prop types that perform better/worse than expected
    - Adjust for line value and direction (OVER/UNDER)
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
        self.gb = None
        self.rf = None
        self.lr = None
        self.scaler = None
        self.metadata = None
        self.feature_cols = None

        # Try to load models
        self._load_models()

    def _load_models(self) -> bool:
        """Load trained models from disk."""
        try:
            self.gb = joblib.load(self.models_dir / 'gradient_boosting.joblib')
            self.rf = joblib.load(self.models_dir / 'random_forest.joblib')
            self.lr = joblib.load(self.models_dir / 'logistic_regression.joblib')
            self.scaler = joblib.load(self.models_dir / 'scaler.joblib')

            with open(self.models_dir / 'ensemble_metadata.json', 'r') as f:
                self.metadata = json.load(f)

            self.feature_cols = self.metadata['feature_cols']
            self.loaded = True
            logger.info(f"ML Filter loaded successfully. Training date: {self.metadata.get('training_date', 'unknown')}")
            return True

        except FileNotFoundError as e:
            logger.warning(f"ML models not found: {e}. Filter will be disabled.")
            self.loaded = False
            return False
        except Exception as e:
            logger.error(f"Error loading ML models: {e}")
            self.loaded = False
            return False

    def _build_features(self, probability: float, edge: float, confidence: float,
                        diff_pct: float, line: float, is_over: bool,
                        prop_type: str) -> pd.DataFrame:
        """
        Build feature DataFrame matching training format.

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
                                  prop_type: str) -> float:
        """
        Get ensemble-adjusted probability for a pick.

        Args:
            probability: Original model probability (0-1)
            edge: Edge percentage
            confidence: Model confidence score
            diff_pct: Percentage difference from line
            line: The betting line value
            is_over: True for OVER, False for UNDER
            prop_type: Type of prop

        Returns:
            Ensemble-adjusted probability (0-1)
        """
        if not self.loaded:
            logger.debug("ML filter not loaded, returning original probability")
            return probability

        try:
            # Build features
            X = self._build_features(
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

            # Simple average ensemble
            ensemble_prob = (gb_prob + rf_prob + lr_prob) / 3

            return float(ensemble_prob)

        except Exception as e:
            logger.warning(f"Error in ML filter: {e}. Returning original probability.")
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
            return ('elite', 0.84)
        elif ensemble_prob >= HIGH_CONF_THRESHOLD:
            return ('high', 0.72)
        elif ensemble_prob >= DEFAULT_THRESHOLD:
            return ('standard', 0.68)
        else:
            return ('below_threshold', 0.50)

    def get_model_info(self) -> Dict:
        """Get information about the loaded models."""
        if not self.loaded:
            return {'loaded': False}

        return {
            'loaded': True,
            'training_date': self.metadata.get('training_date'),
            'training_samples': self.metadata.get('training_samples'),
            'base_hit_rate': self.metadata.get('base_hit_rate'),
            'model_scores': self.metadata.get('model_scores'),
            'thresholds': {
                'standard': {'threshold': DEFAULT_THRESHOLD, 'expected_hit_rate': 0.68},
                'high': {'threshold': HIGH_CONF_THRESHOLD, 'expected_hit_rate': 0.72},
                'elite': {'threshold': ELITE_THRESHOLD, 'expected_hit_rate': 0.84}
            }
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

    ml = MLFilter()
    print(f"Loaded: {ml.loaded}")

    if ml.loaded:
        # Test with a sample prediction
        test_cases = [
            {'probability': 0.65, 'edge': 8.0, 'confidence': 0.85, 'diff_pct': 15.0,
             'line': 25.5, 'is_over': True, 'prop_type': 'points'},
            {'probability': 0.62, 'edge': 18.0, 'confidence': 0.90, 'diff_pct': 30.0,
             'line': 8.5, 'is_over': True, 'prop_type': 'threes'},
            {'probability': 0.58, 'edge': 6.0, 'confidence': 0.82, 'diff_pct': 10.0,
             'line': 32.0, 'is_over': False, 'prop_type': 'pts_rebs_asts'},
        ]

        print("\nTest predictions:")
        for i, tc in enumerate(test_cases, 1):
            ensemble_prob = ml.get_ensemble_probability(**tc)
            tier, expected_hr = ml.get_filter_tier(ensemble_prob)
            passes = ml.passes_filter(ensemble_prob)

            print(f"\n  Case {i}: {tc['prop_type'].upper()} {'OVER' if tc['is_over'] else 'UNDER'} {tc['line']}")
            print(f"    Original prob: {tc['probability']:.1%}")
            print(f"    Ensemble prob: {ensemble_prob:.1%}")
            print(f"    Tier: {tier} (expected {expected_hr:.0%} hit rate)")
            print(f"    Passes filter: {passes}")

        print("\n" + "=" * 50)
        print("Model Info:")
        info = ml.get_model_info()
        print(f"  Training samples: {info['training_samples']:,}")
        print(f"  Base hit rate: {info['base_hit_rate']:.1%}")
