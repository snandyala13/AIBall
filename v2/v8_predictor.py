#!/usr/bin/env python3
"""
v2/v8_predictor.py — Inference engine for v8 calibrated classifier models.

Architecture:
  1. LGB classifier predicts raw P(over) via log-loss
  2. Blend with sharp consensus: final_prob = w_market * consensus + w_model * model
     - 3+ books: 70% consensus, 30% model
     - 2 books:  50/50
     - 1 book:   100% model
  3. Quantile regressors provide sigma for uncertainty and Kelly sizing
  4. Edge = final_prob - DK implied probability

The model's job is to ADJUST the market, not replace it. The sharp consensus
is the anchor; the model is the filter.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import joblib
except ImportError:
    logger.error("joblib not installed. Run: pip install joblib")
    raise

from scipy.stats import norm as _norm

from train_v8 import (
    V8FeatureBuilder,
    compute_consensus_features,
    american_to_implied,
    american_to_decimal,
    devig_power,
    CORE_PROPS,
)

BASE_DIR = Path(__file__).resolve().parent.parent
V8_MODELS_DIR = BASE_DIR / "models" / "v8_calibrated"

# Fallback dirs for older models
V7_MODELS_DIR = BASE_DIR / "models" / "v7_ensemble"
V6_MODELS_DIR = BASE_DIR / "models" / "v6_full"


class V8Predictor:
    """Calibrated classifier predictor with consensus blending.

    Loads v8 models: LGB classifier + isotonic calibrator + quantile regressors.
    Falls back to v7/v6 StatPredictor if v8 not available.
    """

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = Path(models_dir) if models_dir else V8_MODELS_DIR
        self.classifiers = {}      # {prop_type: lgb_classifier}
        self.isotonics = {}        # {prop_type: isotonic_regressor}
        self.q10_models = {}       # {prop_type: lgb_quantile}
        self.q90_models = {}       # {prop_type: lgb_quantile}
        self.feature_names = {}    # {prop_type: [feature_list]}
        self.metadata = {}
        self.feature_builder = V8FeatureBuilder()
        self._loaded = False
        self._data_loaded = False

        self._load_models()

    def _load_models(self):
        meta_path = self.models_dir / "model_metadata.json"
        if not meta_path.exists():
            logger.warning("v8 model_metadata.json not found at %s", self.models_dir)
            return

        with open(meta_path) as f:
            self.metadata = json.load(f)

        if self.metadata.get("version") != "v8":
            logger.warning("Expected v8 metadata, got %s", self.metadata.get("version"))
            return

        loaded = 0
        for prop_type in CORE_PROPS:
            prop_meta = self.metadata.get(prop_type)
            if not prop_meta:
                continue

            clf_path = self.models_dir / f"{prop_type}_classifier.pkl"
            iso_path = self.models_dir / f"{prop_type}_isotonic.pkl"
            q10_path = self.models_dir / f"{prop_type}_q10.pkl"
            q90_path = self.models_dir / f"{prop_type}_q90.pkl"

            if not clf_path.exists():
                logger.debug("No classifier for %s", prop_type)
                continue

            try:
                self.classifiers[prop_type] = joblib.load(clf_path)
                self.isotonics[prop_type] = (
                    joblib.load(iso_path) if iso_path.exists() else None
                )
                self.q10_models[prop_type] = (
                    joblib.load(q10_path) if q10_path.exists() else None
                )
                self.q90_models[prop_type] = (
                    joblib.load(q90_path) if q90_path.exists() else None
                )
                self.feature_names[prop_type] = prop_meta.get("feature_names", [])
                loaded += 1
                logger.info(
                    "v8 loaded %s: classifier + isotonic + quantiles, %d features",
                    prop_type, len(self.feature_names[prop_type]),
                )
            except Exception as e:
                logger.error("Failed to load v8 %s: %s", prop_type, e)

        if loaded > 0:
            self._loaded = True
            logger.info("v8 predictor ready: %d props loaded", loaded)
        else:
            logger.warning("No v8 models loaded from %s", self.models_dir)

    def is_ready(self) -> bool:
        return self._loaded

    def ensure_data_loaded(self):
        if not self._data_loaded:
            self.feature_builder.load_games()
            self.feature_builder.build_player_history()
            self._data_loaded = True

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        player_name: str,
        prop_type: str,
        game_date: str,
        game_context: dict,
    ) -> Optional[dict]:
        """Predict P(over) with consensus blending and full edge analysis.

        Args:
            player_name: e.g. "LeBron James"
            prop_type: e.g. "points"
            game_date: ISO date e.g. "2026-10-20"
            game_context: dict with line, over_odds, under_odds, all_books, etc.

        Returns:
            dict with predicted, p_over, p_under, edge_over, edge_under, sigma,
            confidence, side, adjustment, model_prob, consensus_prob, blended_prob,
            kelly_fraction, recommended_bet_pct, or None.
        """
        if not self._loaded:
            return None
        if prop_type not in self.classifiers:
            return None

        self.ensure_data_loaded()

        # Skip players with thin game history (rookies, new arrivals)
        min_games = game_context.get("min_games", 10)
        history = self.feature_builder.player_history.get(player_name, [])
        if len(history) < min_games:
            return None

        line = game_context.get("line", 0)
        over_odds = game_context.get("over_odds", -110)
        under_odds = game_context.get("under_odds", -110)
        all_books = game_context.get("all_books", {})

        # Build features
        feats = self.feature_builder.build_features(
            player_name, prop_type, game_date,
            line, over_odds, under_odds, all_books,
        )
        if feats is None:
            return None

        # Assemble feature vector in trained column order (as DataFrame to keep names)
        feat_names = self.feature_names[prop_type]
        feature_vec = pd.DataFrame(
            [[feats.get(col, 0.0) for col in feat_names]],
            columns=feat_names,
        )

        # 1. Raw model probability
        clf = self.classifiers[prop_type]
        raw_proba = clf.predict_proba(feature_vec)[0]
        model_prob = float(raw_proba[1]) if len(raw_proba) > 1 else float(raw_proba[0])

        # 2. Consensus probability (sharp-weighted devigged)
        consensus_feats = compute_consensus_features(
            all_books, line, over_odds, under_odds,
        )
        consensus_prob = consensus_feats["consensus_over_prob"]
        num_books = consensus_feats["num_books"]

        # 3. Blend: market gets heavy weight when available
        if num_books >= 3:
            blended_prob = 0.70 * consensus_prob + 0.30 * model_prob
        elif num_books == 2:
            blended_prob = 0.50 * consensus_prob + 0.50 * model_prob
        else:
            blended_prob = model_prob

        p_over = float(np.clip(blended_prob, 0.01, 0.99))
        p_under = 1.0 - p_over

        # 4. Quantile-based uncertainty
        q10_model = self.q10_models.get(prop_type)
        q90_model = self.q90_models.get(prop_type)
        if q10_model is not None and q90_model is not None:
            q10_adj = float(q10_model.predict(feature_vec)[0])
            q90_adj = float(q90_model.predict(feature_vec)[0])
            sigma = max((q90_adj - q10_adj) / 2.56, 0.5)
        else:
            sigma = 3.0
            q10_adj = -sigma * 1.28
            q90_adj = sigma * 1.28

        # 5. Edge computation
        edge_result = self._compute_edge(
            p_over, over_odds, under_odds,
        )

        # Derive an adjustment for compatibility with downstream code
        if 0.01 < p_over < 0.99:
            adjustment = float(_norm.ppf(p_over)) * sigma
        else:
            adjustment = 5.0 if p_over > 0.5 else -5.0

        predicted = line + adjustment
        confidence = max(p_over, p_under)

        return {
            "predicted": round(predicted, 2),
            "p10": round(line + q10_adj, 2),
            "p90": round(line + q90_adj, 2),
            "uncertainty": round(q90_adj - q10_adj, 2),
            "confidence": round(confidence, 3),
            "adjustment": round(adjustment, 3),
            "q10_adjustment": round(q10_adj, 3),
            "q90_adjustment": round(q90_adj, 3),
            "p_over": round(p_over, 4),
            "p_under": round(p_under, 4),
            "model_prob": round(model_prob, 4),
            "consensus_prob": round(consensus_prob, 4),
            "blended_prob": round(p_over, 4),
            "consensus_num_books": num_books,
            "sigma": round(sigma, 3),
            "side": edge_result["side"],
            "edge_over": round(edge_result["edge_over"], 4),
            "edge_under": round(edge_result["edge_under"], 4),
            "edge": round(edge_result["edge"], 4),
            "kelly_fraction": round(edge_result["kelly_fraction"], 5),
            "recommended_bet_pct": round(edge_result["recommended_bet_pct"], 4),
            "target_mode": "v8_calibrated",
            "model_type": "v8_calibrated",
        }

    # ------------------------------------------------------------------
    # Edge and Kelly
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_edge(p_over: float, over_odds: int, under_odds: int) -> dict:
        """Compute edge, recommended side, and Kelly fraction.

        Edge = our probability - DK implied probability (for the recommended side).
        Kelly: f* = (b*p - q) / b  where b = decimal_odds - 1, p = prob, q = 1-p.
        We use quarter-Kelly and cap at 5% of bankroll.
        """
        imp_over = american_to_implied(over_odds)
        imp_under = american_to_implied(under_odds)

        edge_over = p_over - imp_over
        edge_under = (1.0 - p_over) - imp_under

        if edge_over >= edge_under:
            side = "over"
            edge = edge_over
            prob = p_over
            odds = over_odds
        else:
            side = "under"
            edge = edge_under
            prob = 1.0 - p_over
            odds = under_odds

        # Kelly criterion
        decimal_odds = american_to_decimal(odds)
        b = decimal_odds - 1.0
        q = 1.0 - prob
        if b > 0:
            full_kelly = (b * prob - q) / b
        else:
            full_kelly = 0.0

        kelly_fraction = max(0.25 * full_kelly, 0.0)
        recommended_bet_pct = min(kelly_fraction, 0.05)

        return {
            "side": side,
            "edge_over": edge_over,
            "edge_under": edge_under,
            "edge": edge,
            "kelly_fraction": kelly_fraction,
            "recommended_bet_pct": recommended_bet_pct,
        }
