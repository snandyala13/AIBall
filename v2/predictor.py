#!/usr/bin/env python3
"""
v2/predictor.py - Inference engine for trained v2 line-anchor models.

Loads trained LightGBM models and makes predictions with uncertainty estimates.
Used by the main ball.py pipeline for live predictions.

Architecture (v6 dual-model):
  1. LGB Huber regressor predicts residual = actual - line
  2. RF Classifier predicts P(over) directly
  3. Combine: Huber is primary signal; RF provides consensus confirmation

  P(over) from Huber:
    p_over = norm.cdf(residual / sigma)
    where sigma is the MAE from training (per-prop, in model_metadata.json)

  When both models agree on direction AND edge >= 5%, that is the
  strongest signal (highest confidence).

Fallback: v3 classifier models if v6 not available.

Usage:
    # As a module
    from v2.predictor import StatPredictor
    pred = StatPredictor()
    result = pred.predict("LeBron James", "points", "2026-10-20", game_context)

    # CLI demo
    python -m v2.predictor
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    import joblib
except ImportError:
    logger.error("joblib not installed. Run: pip install joblib")
    raise

from scipy import stats as scipy_stats
from scipy.stats import norm as _norm

from v2.features import FeatureBuilder, CORE_PROPS, BASE_DIR
from v2.edge import EdgeCalculator

# v7 ensemble: enhanced tabular + sequence model (primary)
V7_MODELS_DIR = BASE_DIR / "models" / "v7_ensemble"
# v6 models trained on 131K dataset (fallback 1)
V6_MODELS_DIR = BASE_DIR / "models" / "v6_full"
# v3 classifier holdout (fallback 2)
V3_MODELS_DIR = BASE_DIR / "models" / "v3_classifier_holdout"

def _resolve_models_dir() -> Path:
    env = os.environ.get("AIBALL_MODELS_DIR")
    if env:
        return Path(env)
    if V7_MODELS_DIR.exists():
        return V7_MODELS_DIR
    if V6_MODELS_DIR.exists():
        return V6_MODELS_DIR
    return V3_MODELS_DIR

MODELS_DIR = _resolve_models_dir()


# =============================================================================
# StatPredictor
# =============================================================================

class StatPredictor:
    """Loads trained models and makes predictions with uncertainty.

    Supports three model formats:
    - v6 dual-model (primary): LGB Huber + RF Classifier per prop
    - v3 classifier: single LightGBM binary classifier per prop
    - v2 regression: LightGBM Huber + quantile regressors per prop
    """

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.models = {}  # {prop_type: model_tuple} - legacy v3/v2
        self.feature_builder = FeatureBuilder()
        self.edge_calculator = EdgeCalculator()
        self.feature_columns = None  # Expected column order
        self.target_mode = "line_anchor"  # Default; loaded from metadata
        self.model_type = "regression"  # "regression", "classifier", or "v6_dual"
        self.calibrators = {}  # {prop_type: calibrator} for classifier mode
        self.per_prop_features = {}  # {prop_type: [feature_list]}

        # v6 dual-model state
        self.v6_huber = {}      # {prop_type: lgb_huber_model}
        self.v6_rf = {}         # {prop_type: rf_classifier_model}
        self.v6_sigma = {}      # {prop_type: float} MAE from training
        self.v6_metadata = {}   # raw model_metadata.json
        self.v6_feature_names = {}  # {prop_type: [feature_names]}

        self._loaded = False
        self._load_models()

    # -----------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------

    def _load_models(self):
        """Load models from disk. Tries v6 dual-model first, falls back to v3/v2."""
        if not self.models_dir.exists():
            logger.warning("Models directory not found: %s", self.models_dir)
            logger.warning("Run 'python -m v2.train' first to train models.")
            return

        # Try v6 format first (model_metadata.json + *_lgb_huber.joblib)
        v6_meta_path = self.models_dir / "model_metadata.json"
        if v6_meta_path.exists():
            try:
                self._load_v6_models(v6_meta_path)
                if self._loaded:
                    return  # v6 loaded successfully
            except Exception as e:
                logger.warning("v6 model loading failed: %s -- falling back to v3/v2", e)
                import traceback
                logger.debug(traceback.format_exc())

        # Fall back to v3/v2 format (training_meta.json)
        self._load_legacy_models()

    def _load_v6_models(self, meta_path: Path):
        """Load v6/v7 model format: LGB Huber (primary) + optional RF Classifier.

        v7 enhanced models use {prop}_lgb_huber_enhanced.joblib filenames and
        have per-prop advanced_features + market_features in metadata.
        v6 models use {prop}_lgb_huber.joblib + {prop}_rf_classifier.joblib.
        Both share the same inference path (Huber residual → sigma → P(over)).
        """
        with open(meta_path) as f:
            self.v6_metadata = json.load(f)

        version = self.v6_metadata.get("version", "v6")

        trained_props = self.v6_metadata.get("_training_config", {}).get(
            "props_trained", []
        )
        if not trained_props:
            trained_props = [
                k for k in self.v6_metadata
                if not k.startswith("_") and isinstance(self.v6_metadata[k], dict)
                and "sigma" in self.v6_metadata[k]
            ]

        loaded_count = 0
        for prop_type in trained_props:
            prop_meta = self.v6_metadata.get(prop_type)
            if not prop_meta:
                continue

            # Try v7 enhanced filename first, then v6 filename
            huber_path = self.models_dir / f"{prop_type}_lgb_huber_enhanced.joblib"
            if not huber_path.exists():
                huber_path = self.models_dir / f"{prop_type}_lgb_huber.joblib"
            if not huber_path.exists():
                logger.debug("No Huber model for %s", prop_type)
                continue

            try:
                huber_model = joblib.load(huber_path)
            except Exception as e:
                logger.error("Failed to load Huber for %s: %s", prop_type, e)
                continue

            # Load RF Classifier (consensus signal, v6 only)
            rf_model = None
            rf_path = self.models_dir / f"{prop_type}_rf_classifier.joblib"
            if rf_path.exists():
                try:
                    rf_model = joblib.load(rf_path)
                except Exception as e:
                    logger.warning("Failed to load RF for %s: %s (continuing without)", prop_type, e)

            self.v6_huber[prop_type] = huber_model
            if rf_model is not None:
                self.v6_rf[prop_type] = rf_model
            self.v6_sigma[prop_type] = prop_meta.get("sigma", prop_meta.get("sigma_huber", 3.0))
            self.v6_feature_names[prop_type] = prop_meta.get("feature_names", [])

            self.models[prop_type] = (huber_model, rf_model, None)

            loaded_count += 1
            has_rf = "yes" if rf_model is not None else "no"
            logger.info(
                "Loaded %s %s: Huber (sigma=%.2f) + RF=%s, %d features",
                version,
                prop_type,
                self.v6_sigma[prop_type],
                has_rf,
                len(self.v6_feature_names[prop_type]),
            )

        if loaded_count > 0:
            self.model_type = "v6_dual"
            self.target_mode = "v6_dual"
            self.per_prop_features = dict(self.v6_feature_names)
            self._loaded = True
            logger.info(
                "%s loaded: %d props (LGB Huber primary)",
                version,
                loaded_count,
            )
        else:
            logger.warning("No models loaded from %s", self.models_dir)

    def _load_legacy_models(self):
        """Load v3 classifier or v2 regression models (fallback)."""
        meta_path = self.models_dir / "training_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.feature_columns = meta.get("feature_columns") or meta.get("features")
            self.target_mode = meta.get("target_mode", "line_anchor")
            self.model_type = meta.get("model_type", "regression")
            ppc = meta.get("per_prop_config", {})
            for pt, cfg in ppc.items():
                feats = cfg.get("features", [])
                if feats:
                    self.per_prop_features[pt] = feats

        available_props = list(self.per_prop_features.keys()) if self.per_prop_features else list(CORE_PROPS)

        if self.model_type == "classifier":
            for prop_type in available_props:
                cls_path = self.models_dir / f"{prop_type}_classifier.pkl"
                cal_path = self.models_dir / f"{prop_type}_calibrator.pkl"
                if not cls_path.exists():
                    continue
                try:
                    classifier = joblib.load(cls_path)
                    calibrator = joblib.load(cal_path) if cal_path.exists() else None
                    self.models[prop_type] = (classifier, None, None)
                    self.calibrators[prop_type] = calibrator
                    logger.info("Loaded %s classifier (v3 fallback)", prop_type)
                except Exception as e:
                    logger.error("Failed to load %s classifier: %s", prop_type, e)
        else:
            for prop_type in available_props:
                point_path = self.models_dir / f"{prop_type}_point.pkl"
                if not point_path.exists():
                    continue
                try:
                    point_model = joblib.load(point_path)
                    q10_path = self.models_dir / f"{prop_type}_q10.pkl"
                    q90_path = self.models_dir / f"{prop_type}_q90.pkl"
                    q10_model = joblib.load(q10_path) if q10_path.exists() else None
                    q90_model = joblib.load(q90_path) if q90_path.exists() else None
                    self.models[prop_type] = (point_model, q10_model, q90_model)
                    logger.info("Loaded %s model (%s mode, fallback)", prop_type, self.target_mode)
                except Exception as e:
                    logger.error("Failed to load %s model: %s", prop_type, e)

        self._loaded = len(self.models) > 0
        if self._loaded:
            logger.info("Legacy models loaded: %d props (target_mode=%s)",
                        len(self.models), self.target_mode)
        else:
            logger.warning("No models loaded. Run 'python -m v2.train' first.")

    def is_ready(self) -> bool:
        """Check if models are loaded and feature builder is ready."""
        return self._loaded

    def ensure_data_loaded(self):
        """Load game data into feature builder if not already loaded."""
        self.feature_builder.ensure_loaded()

    # -----------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------

    def predict(
        self,
        player_name: str,
        prop_type: str,
        game_date: str,
        game_context: dict,
    ) -> Optional[dict]:
        """
        Predict stat value with uncertainty.

        For v6 dual-model:
          1. LGB Huber predicts residual (actual - line)
          2. RF Classifier predicts P(over) directly
          3. Combine: Huber primary, RF consensus confirmation

        For v3 classifier: directly predicts P(over)
        For v2 regression: predicts adjustment + quantile uncertainty

        Args:
            player_name: e.g. "LeBron James"
            prop_type: e.g. "points"
            game_date: ISO date string e.g. "2026-10-20"
            game_context: dict with keys:
                - line (float): DK prop line
                - over_odds (int): e.g. -110
                - under_odds (int): e.g. -110
                - total (float, optional): game total
                - spread (float, optional): spread
                - is_home (bool, optional): home team?
                - all_books (dict, optional): cross-book data from PropLine

        Returns:
            dict with predicted, p10, p90, uncertainty, confidence,
            adjustment (in line-anchor mode),
            or None if prediction not possible.
        """
        if not self._loaded:
            logger.warning("Models not loaded")
            return None

        if prop_type not in self.models:
            logger.debug("No model for prop type: %s", prop_type)
            return None

        self.ensure_data_loaded()

        # Build features
        feats = self.feature_builder.get_features_for_leg(
            player_name, prop_type, game_date, game_context
        )
        if feats is None:
            logger.debug("Insufficient history for %s", player_name)
            return None

        # Merge tracking/PBP features if provided in game_context
        tracking_feats = game_context.get("tracking_features", {})
        if tracking_feats:
            feats.update(tracking_feats)

        # Merge enhanced features (team, travel, calendar, derived) if provided
        enhanced_feats = game_context.get("enhanced_features", {})
        if enhanced_feats:
            feats.update(enhanced_feats)

        line = game_context.get("line", 0)

        # Compute v7 market/advanced features from available data
        if self.model_type == "v6_dual" and self.v6_metadata.get("version", "").startswith("v7"):
            self._compute_v7_extras(feats, prop_type, game_context)

        # Dispatch to the right prediction path
        if self.model_type == "v6_dual":
            return self._predict_v6(feats, prop_type, line)
        elif self.model_type == "classifier":
            return self._predict_v3_classifier(feats, prop_type, line)
        else:
            return self._predict_v2_regression(feats, prop_type, line)

    # -----------------------------------------------------------------
    # v7 extra feature computation
    # -----------------------------------------------------------------

    @staticmethod
    def _american_to_prob(odds: float) -> float:
        if odds == 0:
            return 0.5
        if odds > 0:
            return 100.0 / (odds + 100.0)
        return abs(odds) / (abs(odds) + 100.0)

    def _compute_v7_extras(self, feats: dict, prop_type: str, ctx: dict):
        """Derive v7 market signal and advanced features from available data.

        Fills in feats dict in-place. Features that can't be computed stay at
        their default (0.0 via _build_feature_vec).
        """
        line = ctx.get("line", 0)
        over_odds = ctx.get("over_odds", -110)
        under_odds = ctx.get("under_odds", -110)

        # --- Market features ---
        p_over = self._american_to_prob(over_odds)
        p_under = self._american_to_prob(under_odds)
        total_p = max(p_over + p_under, 0.01)
        vig = total_p - 1.0

        feats.setdefault("vig_asymmetry", (p_over / total_p) - 0.5)

        # Vig vs line bucket average (trained averages per bucket)
        bucket_avg_vig = {1: 0.042, 2: 0.045, 3: 0.048, 4: 0.046, 5: 0.043, 6: 0.040}
        if line <= 2.5:
            bkt = 1
        elif line <= 5.5:
            bkt = 2
        elif line <= 10.5:
            bkt = 3
        elif line <= 20.5:
            bkt = 4
        elif line <= 50.5:
            bkt = 5
        else:
            bkt = 6
        feats.setdefault("vig_vs_line_bucket", vig - bucket_avg_vig.get(bkt, 0.045))

        # line_vs_recent_actual: (avg_recent_actual - line) / max(line, 1)
        target_l3 = feats.get("target_l3", 0)
        if target_l3 and line:
            feats.setdefault("line_vs_recent_actual", (target_l3 - line) / max(line, 1.0))

        # recent_over_hit_rate: use existing hit_rate_l5 (tracks over-line rate)
        hr_l5 = feats.get("hit_rate_l5", 0.5)
        feats.setdefault("recent_over_hit_rate", hr_l5)

        # player_line_trend, odds_trend, line_staleness, line_trend_x_odds_trend,
        # reverse_line_movement: require historical line/odds data not available
        # at single-game inference. Default to 0.0 (neutral).
        for feat_name in ("player_line_trend", "odds_trend", "line_staleness",
                          "line_trend_x_odds_trend", "reverse_line_movement"):
            feats.setdefault(feat_name, 0.0)

        # --- Advanced features ---
        # l5_zscore_vs_season
        target_l5 = feats.get("target_l5", 0)
        target_season = feats.get("target_season", 0)
        target_std = feats.get("target_std_season", 1)
        if target_std and target_std > 0:
            feats.setdefault("l5_zscore_vs_season",
                             (target_l5 - target_season) / target_std)

        # ewm_target_l10: approximate as weighted blend of L5 and L10
        target_l10 = feats.get("target_l10", 0)
        if target_l5 and target_l10:
            feats.setdefault("ewm_target_l10", 0.6 * target_l5 + 0.4 * target_l10)

        # home_stand_length: from game_context schedule info if available
        feats.setdefault("home_stand_length", ctx.get("home_stand_length", 0))

        # matchup_avg_mins, matchup_std: from game_context matchup data
        feats.setdefault("matchup_avg_mins", ctx.get("matchup_avg_mins", 0))
        feats.setdefault("matchup_std", ctx.get("matchup_std", 0))

    def _build_feature_vec(self, feats: dict, prop_type: str) -> np.ndarray:
        """Assemble a feature vector in the expected column order for a prop type."""
        prop_features = self.per_prop_features.get(prop_type, self.feature_columns)
        if prop_features:
            return np.array(
                [feats.get(col, 0.0) for col in prop_features], dtype=np.float64,
            ).reshape(1, -1)
        # Fallback: DataFrame (lets the model figure out column order)
        import pandas as pd
        return pd.DataFrame([feats])

    # -----------------------------------------------------------------
    # v6 dual-model: Huber (primary) + RF (consensus)
    # -----------------------------------------------------------------

    def _predict_v6(self, feats: dict, prop_type: str, line: float) -> Optional[dict]:
        """Dual-model prediction using v6 LGB Huber + RF Classifier.

        Strategy:
          - LGB Huber predicts residual, converted to P(over) via sigma
          - RF Classifier predicts P(over) directly
          - When both agree on direction: weighted blend (70% Huber, 30% RF)
          - When they disagree: use Huber only (more calibrated)
        """
        feature_vec = self._build_feature_vec(feats, prop_type)
        sigma = self.v6_sigma.get(prop_type, 3.0)

        # 1. LGB Huber: predict residual (actual - line)
        huber_model = self.v6_huber[prop_type]
        residual = float(huber_model.predict(feature_vec)[0])

        # Convert residual to P(over) via normal CDF
        # P(over) = P(actual > line) = P(residual > 0 | predicted residual)
        z_huber = residual / sigma
        p_over_huber = float(_norm.cdf(z_huber))

        # 2. RF Classifier: predict P(over) directly
        p_over_rf = None
        rf_model = self.v6_rf.get(prop_type)
        if rf_model is not None:
            try:
                rf_proba = rf_model.predict_proba(feature_vec)[0]
                p_over_rf = float(rf_proba[1]) if len(rf_proba) > 1 else float(rf_proba[0])
            except Exception as e:
                logger.debug("v6 RF predict failed for %s: %s", prop_type, e)

        # 3. Combine: Huber primary, RF consensus
        consensus = False
        if p_over_rf is not None:
            huber_over = p_over_huber > 0.5
            rf_over = p_over_rf > 0.5
            if huber_over == rf_over:
                # Both agree on direction -- blend with 70/30 weighting
                p_over = 0.7 * p_over_huber + 0.3 * p_over_rf
                consensus = True
            else:
                # Disagreement -- trust Huber only (better calibrated)
                p_over = p_over_huber
        else:
            p_over = p_over_huber

        p_under = 1.0 - p_over
        adjustment = residual
        predicted = line + adjustment
        confidence = max(p_over, p_under)

        return {
            "predicted": round(predicted, 2),
            "p10": round(line - sigma * 1.28, 2),
            "p90": round(line + sigma * 1.28, 2),
            "uncertainty": round(sigma * 2.56, 2),
            "confidence": round(confidence, 3),
            "adjustment": round(adjustment, 3),
            "q10_adjustment": round(-sigma * 1.28, 3),
            "q90_adjustment": round(sigma * 1.28, 3),
            "p_over": round(p_over, 4),
            "p_under": round(p_under, 4),
            "p_over_huber": round(p_over_huber, 4),
            "p_over_rf": round(p_over_rf, 4) if p_over_rf is not None else None,
            "consensus": consensus,
            "sigma": round(sigma, 3),
            "target_mode": "v6_dual",
            "model_type": "v6_dual",
        }

    # -----------------------------------------------------------------
    # v3 classifier (fallback)
    # -----------------------------------------------------------------

    def _predict_v3_classifier(self, feats: dict, prop_type: str, line: float) -> dict:
        """v3 single classifier: directly predicts P(over)."""
        feature_vec = self._build_feature_vec(feats, prop_type)
        classifier = self.models[prop_type][0]
        calibrator = self.calibrators.get(prop_type)

        raw_proba = classifier.predict_proba(feature_vec)[0]
        p_over_raw = float(raw_proba[1]) if len(raw_proba) > 1 else float(raw_proba[0])

        if calibrator is not None:
            try:
                calibrated = calibrator.predict_proba(np.array([[p_over_raw]]))[0]
                p_over = float(calibrated[1]) if len(calibrated) > 1 else float(calibrated[0])
            except Exception:
                p_over = p_over_raw
        else:
            p_over = p_over_raw

        p_under = 1.0 - p_over
        sigma = 3.0
        if 0.01 < p_over < 0.99:
            adjustment = _norm.ppf(p_over) * sigma
        else:
            adjustment = 5.0 if p_over > 0.5 else -5.0

        predicted = line + adjustment
        confidence = max(p_over, p_under)

        return {
            "predicted": round(predicted, 2),
            "p10": round(line - sigma * 1.28, 2),
            "p90": round(line + sigma * 1.28, 2),
            "uncertainty": round(sigma * 2.56, 2),
            "confidence": round(confidence, 3),
            "adjustment": round(adjustment, 3),
            "q10_adjustment": round(-sigma * 1.28, 3),
            "q90_adjustment": round(sigma * 1.28, 3),
            "p_over": round(p_over, 4),
            "p_under": round(p_under, 4),
            "target_mode": "classifier",
            "model_type": "classifier",
        }

    # -----------------------------------------------------------------
    # v2 regression (legacy fallback)
    # -----------------------------------------------------------------

    def _predict_v2_regression(self, feats: dict, prop_type: str, line: float) -> dict:
        """v2 regression: Huber point + quantile uncertainty."""
        feature_vec = self._build_feature_vec(feats, prop_type)
        point_model, q10_model, q90_model = self.models[prop_type]

        raw_point = float(point_model.predict(feature_vec)[0])
        raw_q10 = float(q10_model.predict(feature_vec)[0]) if q10_model else raw_point - 5.0
        raw_q90 = float(q90_model.predict(feature_vec)[0]) if q90_model else raw_point + 5.0

        if self.target_mode == "line_anchor":
            adjustment = raw_point
            q10_adj = min(raw_q10, adjustment)
            q90_adj = max(raw_q90, adjustment)
            predicted = line + adjustment
            p10 = line + q10_adj
            p90 = line + q90_adj
        else:
            predicted = raw_point
            p10 = min(raw_q10, predicted)
            p90 = max(raw_q90, predicted)
            adjustment = predicted - line
            q10_adj = p10 - line
            q90_adj = p90 - line

        p10 = min(p10, predicted)
        p90 = max(p90, predicted)
        uncertainty = p90 - p10

        if predicted > 0:
            confidence = max(0.0, min(1.0, 1.0 - (uncertainty / (2.0 * max(predicted, 1.0)))))
        else:
            confidence = 0.3

        return {
            "predicted": round(predicted, 2),
            "p10": round(p10, 2),
            "p90": round(p90, 2),
            "uncertainty": round(uncertainty, 2),
            "confidence": round(confidence, 3),
            "adjustment": round(adjustment, 3),
            "q10_adjustment": round(q10_adj, 3),
            "q90_adjustment": round(q90_adj, 3),
            "target_mode": self.target_mode,
        }

    # -----------------------------------------------------------------
    # Edge calculation (delegates to EdgeCalculator)
    # -----------------------------------------------------------------

    def calculate_edge(
        self,
        prediction: dict,
        line: float,
        over_odds: int,
        under_odds: int,
        consensus_line: Optional[float] = None,
        line_range: Optional[float] = None,
    ) -> dict:
        """
        Calculate betting edge from prediction vs market line.

        In line-anchor mode, uses the EdgeCalculator which works directly
        with the adjustment values.

        Returns:
            dict with over_prob, under_prob, edge_over, edge_under,
                 recommended_side, kelly_fraction
        """
        return self.edge_calculator.from_adjustment(
            adjustment=prediction["adjustment"],
            q10=prediction["q10_adjustment"],
            q90=prediction["q90_adjustment"],
            dk_line=line,
            over_odds=over_odds,
            under_odds=under_odds,
            consensus_line=consensus_line,
            line_range=line_range,
        )

    # -----------------------------------------------------------------
    # Full analysis: predict + edge in one call
    # -----------------------------------------------------------------

    def analyze_prop(
        self,
        player_name: str,
        prop_type: str,
        game_date: str,
        line: float,
        over_odds: int,
        under_odds: int,
        game_context: Optional[dict] = None,
    ) -> Optional[dict]:
        """
        Full analysis: predict stat value and calculate edge vs market.

        Returns combined dict with prediction + edge info, or None.
        """
        ctx = game_context or {}
        ctx.setdefault("line", line)
        ctx.setdefault("over_odds", over_odds)
        ctx.setdefault("under_odds", under_odds)

        prediction = self.predict(player_name, prop_type, game_date, ctx)
        if prediction is None:
            return None

        # Extract cross-book info from context if available
        consensus_line = None
        line_range = None
        all_books = ctx.get("all_books", {})
        if all_books and len(all_books) > 1:
            book_lines = []
            for bd in all_books.values():
                bl = bd.get("line")
                if bl is not None:
                    try:
                        book_lines.append(float(bl))
                    except (TypeError, ValueError):
                        pass
            if book_lines:
                consensus_line = float(np.median(book_lines))
                line_range = max(book_lines) - min(book_lines)

        edge_info = self.calculate_edge(
            prediction, line, over_odds, under_odds,
            consensus_line=consensus_line,
            line_range=line_range,
        )

        return {
            "player": player_name,
            "prop_type": prop_type,
            "line": line,
            **prediction,
            **edge_info,
        }


# =============================================================================
# CLI demo
# =============================================================================

def main():
    print("=" * 70)
    print("v2 StatPredictor - Loading models and running demo")
    print("=" * 70)

    predictor = StatPredictor()
    if not predictor.is_ready():
        print("\nNo trained models found. Run 'python -m v2.train' first.")
        return

    print(f"  Target mode: {predictor.target_mode}")
    predictor.ensure_data_loaded()

    # Demo: pick some players from the data
    players = list(predictor.feature_builder.player_history.keys())
    # Pick players with enough history
    demo_players = []
    for p in players:
        if len(predictor.feature_builder.player_history[p]) >= 20:
            demo_players.append(p)
        if len(demo_players) >= 5:
            break

    if not demo_players:
        print("No players with sufficient history for demo.")
        return

    print(f"\nDemo predictions for {len(demo_players)} players:\n")

    for player in demo_players:
        history = predictor.feature_builder.player_history[player]
        last_game = history[-1]
        game_date = last_game["date"]

        for prop_type in ["points", "rebounds", "assists"]:
            if prop_type not in predictor.models:
                continue

            # Use season average as a fake line
            stat_vals = [predictor.feature_builder.get_stat_value(g, prop_type)
                         for g in history[-20:]]
            fake_line = round(np.mean(stat_vals) - 0.5, 1)

            result = predictor.analyze_prop(
                player, prop_type, game_date,
                line=fake_line, over_odds=-110, under_odds=-110,
            )
            if result:
                adj_str = ""
                if "adjustment" in result:
                    adj_str = f"adj={result['adjustment']:+.2f} "

                side = result.get("recommended_side", "?")
                edge = result.get("edge", 0)
                stars = result.get("confidence_stars", 0)

                print(f"  {player:25s} {prop_type:15s} line={fake_line:5.1f} -> "
                      f"pred={result['predicted']:5.1f} "
                      f"[{result['p10']:.1f}-{result['p90']:.1f}] "
                      f"{adj_str}"
                      f"edge={edge:+.1%} {side} "
                      f"{'*' * stars}")

    print()


if __name__ == "__main__":
    main()
