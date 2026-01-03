#!/usr/bin/env python3
"""
Train ML Ensemble with Time-Based Cross-Validation

This script trains the ML filter models using temporal splits to prevent
data leakage. Training on older data and validating on recent data mimics
real-world betting conditions.

Usage:
    python train_ml_model.py                    # Train with latest backtest data
    python train_ml_model.py --predictions FILE # Use specific predictions file
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Feature columns for the model
FEATURE_COLS = [
    'probability', 'edge', 'confidence', 'diff_pct', 'line', 'is_over',
    'prop_assists', 'prop_points', 'prop_pts_rebs_asts', 'prop_rebounds', 'prop_threes',
    'edge_squared', 'prob_x_edge', 'edge_over_interaction', 'high_edge',
    'mid_line', 'line_percentile'
]

PROP_TYPES = ['assists', 'points', 'pts_rebs_asts', 'rebounds', 'threes']


def load_predictions(predictions_file: str) -> pd.DataFrame:
    """Load predictions from JSON file and convert to DataFrame."""
    logger.info(f"Loading predictions from: {predictions_file}")

    with open(predictions_file, 'r') as f:
        data = json.load(f)

    predictions = data.get('predictions', data.get('legs', []))
    logger.info(f"Loaded {len(predictions)} predictions")

    return pd.DataFrame(predictions)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare feature DataFrame from raw predictions."""
    # Create feature columns
    features = pd.DataFrame()

    # Base features
    features['probability'] = df['probability'].astype(float)
    features['edge'] = df['edge'].astype(float)
    features['confidence'] = df.get('confidence', 0.8).fillna(0.8).astype(float)
    features['diff_pct'] = df.get('diff_pct', df['edge'] * 2).astype(float)
    features['line'] = df['line'].astype(float)
    features['is_over'] = (df['pick'] == 'OVER').astype(int)

    # Prop type dummies
    for prop in PROP_TYPES:
        features[f'prop_{prop}'] = (df['prop_type'] == prop).astype(int)

    # Derived features
    features['edge_squared'] = features['edge'] ** 2
    features['prob_x_edge'] = features['probability'] * features['edge']
    features['edge_over_interaction'] = features['edge'] * features['is_over']
    features['high_edge'] = (features['edge'] >= 15).astype(int)
    features['mid_line'] = ((features['line'] >= 15) & (features['line'] <= 35)).astype(int)
    features['line_percentile'] = features['line'].rank(pct=True)

    # Target variable
    features['hit'] = df['hit'].astype(int)
    features['date'] = pd.to_datetime(df['game_date'])

    return features


def time_based_split(df: pd.DataFrame, train_end_date: str = None,
                     test_start_date: str = None) -> tuple:
    """
    Split data temporally for realistic validation.

    By default:
    - Train: All data before the last 30 days
    - Test: Last 30 days
    """
    df = df.sort_values('date').reset_index(drop=True)

    if train_end_date is None:
        # Default: use last 30 days for testing
        max_date = df['date'].max()
        train_end_date = (max_date - pd.Timedelta(days=30)).strftime('%Y-%m-%d')

    if test_start_date is None:
        test_start_date = train_end_date

    train_mask = df['date'] < train_end_date
    test_mask = df['date'] >= test_start_date

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    logger.info(f"Time-based split:")
    logger.info(f"  Train: {train_df['date'].min()} to {train_df['date'].max()} ({len(train_df)} samples)")
    logger.info(f"  Test:  {test_df['date'].min()} to {test_df['date'].max()} ({len(test_df)} samples)")

    return train_df, test_df


def train_ensemble(train_df: pd.DataFrame, test_df: pd.DataFrame,
                   models_dir: str) -> dict:
    """Train the ensemble models and save to disk."""

    # Prepare feature matrices
    feature_cols = [c for c in FEATURE_COLS if c in train_df.columns]

    X_train = train_df[feature_cols].values
    y_train = train_df['hit'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['hit'].values

    # Initialize models
    logger.info("Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        min_samples_leaf=20,
        random_state=42
    )
    gb.fit(X_train, y_train)
    gb_score = accuracy_score(y_test, gb.predict(X_test))
    logger.info(f"  GB Test Accuracy: {gb_score:.4f}")

    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_score = accuracy_score(y_test, rf.predict(X_test))
    logger.info(f"  RF Test Accuracy: {rf_score:.4f}")

    logger.info("Training Logistic Regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(
        C=0.1,
        max_iter=1000,
        random_state=42
    )
    lr.fit(X_train_scaled, y_train)
    lr_score = accuracy_score(y_test, lr.predict(X_test_scaled))
    logger.info(f"  LR Test Accuracy: {lr_score:.4f}")

    # Ensemble predictions
    gb_probs = gb.predict_proba(X_test)[:, 1]
    rf_probs = rf.predict_proba(X_test)[:, 1]
    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
    ensemble_probs = (gb_probs + rf_probs + lr_probs) / 3

    # Evaluate at different thresholds
    logger.info("\nEnsemble Performance at Different Thresholds:")
    thresholds = [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65]
    for thresh in thresholds:
        mask = ensemble_probs >= thresh
        if mask.sum() > 0:
            hit_rate = y_test[mask].mean()
            count = mask.sum()
            logger.info(f"  ML >= {thresh:.0%}: {hit_rate:.1%} hit rate ({count} picks)")

    # Save models
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(gb, models_path / 'gradient_boosting.joblib')
    joblib.dump(rf, models_path / 'random_forest.joblib')
    joblib.dump(lr, models_path / 'logistic_regression.joblib')
    joblib.dump(scaler, models_path / 'scaler.joblib')

    # Save metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'training_samples': len(train_df),
        'test_samples': len(test_df),
        'base_hit_rate': float(y_train.mean()),
        'test_hit_rate': float(y_test.mean()),
        'feature_cols': feature_cols,
        'prop_types': PROP_TYPES,
        'model_scores': {
            'gradient_boosting': float(gb_score),
            'random_forest': float(rf_score),
            'logistic_regression': float(lr_score)
        },
        'split_info': {
            'train_start': str(train_df['date'].min()),
            'train_end': str(train_df['date'].max()),
            'test_start': str(test_df['date'].min()),
            'test_end': str(test_df['date'].max())
        }
    }

    with open(models_path / 'ensemble_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\nModels saved to: {models_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description='Train ML ensemble with time-based CV')
    parser.add_argument('--predictions', type=str,
                        help='Path to predictions JSON file')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Directory to save trained models')
    parser.add_argument('--train-end', type=str,
                        help='End date for training data (YYYY-MM-DD)')
    parser.add_argument('--test-start', type=str,
                        help='Start date for test data (YYYY-MM-DD)')
    args = parser.parse_args()

    # Find latest predictions file if not specified
    if args.predictions:
        predictions_file = args.predictions
    else:
        outputs_dir = Path(__file__).parent / 'outputs'
        prediction_files = list(outputs_dir.glob('backtest_predictions_*.json'))
        prediction_files += list(outputs_dir.glob('backtest_real_lines.json'))

        if not prediction_files:
            logger.error("No predictions file found. Run backtest first or specify --predictions")
            sys.exit(1)

        predictions_file = str(max(prediction_files, key=lambda p: p.stat().st_mtime))
        logger.info(f"Using latest predictions: {predictions_file}")

    # Load and prepare data
    raw_df = load_predictions(predictions_file)
    df = prepare_features(raw_df)

    # Time-based split
    train_df, test_df = time_based_split(
        df,
        train_end_date=args.train_end,
        test_start_date=args.test_start
    )

    if len(test_df) < 100:
        logger.warning(f"Test set only has {len(test_df)} samples. Consider adjusting split dates.")

    # Train and save models
    models_dir = str(Path(__file__).parent / args.models_dir)
    metadata = train_ensemble(train_df, test_df, models_dir)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Training samples: {metadata['training_samples']:,}")
    print(f"Test samples: {metadata['test_samples']:,}")
    print(f"Base hit rate (train): {metadata['base_hit_rate']:.1%}")
    print(f"Test hit rate: {metadata['test_hit_rate']:.1%}")
    print(f"\nModel Scores:")
    for model, score in metadata['model_scores'].items():
        print(f"  {model}: {score:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
