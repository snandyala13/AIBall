#!/usr/bin/env python3
"""
v2/engine.py - Production analysis engine using v2 line-anchor ML models.

Replaces the v1 ModelEngine for game analysis in the web pipeline.
Uses LightGBM models trained on 179 features to predict adjustments to DK lines.

Architecture:
  predicted_adjustment = model.predict(features)
  final_prediction = DK_line + predicted_adjustment

For each player prop:
  1. Build features from player history + tracking + PBP + market data
  2. Run v2 LightGBM model to predict adjustment to line
  3. Calculate edge and confidence via EdgeCalculator
  4. Rank by edge * confidence

Usage:
    from v2.engine import V2AnalysisEngine
    engine = V2AnalysisEngine()
    result = engine.get_predictions_for_game(game_id, game_context, player_profiles, props)
"""

import logging
import time
from typing import Dict, List, Optional

from v2.predictor import StatPredictor
from v2.features import CORE_PROPS
from v2.edge import EdgeCalculator
from v2.enrich_training_data import (
    load_tracking_data,
    load_pbp_data,
    date_to_season,
    TRACKING_FEATURES,
    PBP_FEATURES,
)

# Optional enhanced feature builders (graceful degradation if unavailable)
try:
    from v2.team_features import TeamStatsBuilder
    _HAS_TEAM_FEATURES = True
except ImportError:
    _HAS_TEAM_FEATURES = False

try:
    from v2.derived_features import DerivedFeatures
    _HAS_DERIVED_FEATURES = True
except ImportError:
    _HAS_DERIVED_FEATURES = False

try:
    from v2.travel_features import (
        get_travel_features,
        get_calendar_features,
        build_team_schedules,
    )
    _HAS_TRAVEL_FEATURES = True
except ImportError:
    _HAS_TRAVEL_FEATURES = False

logger = logging.getLogger(__name__)

# Prop types the v2 model can handle
V2_PROP_TYPES = set(CORE_PROPS)  # points, rebounds, assists, threes, pts_rebs_asts


class V2AnalysisEngine:
    """
    Production engine using v2 line-anchor ML models.
    Replaces the v1 ModelEngine for game analysis.
    """

    def __init__(self):
        logger.info("Initializing V2AnalysisEngine...")
        self.predictor = StatPredictor()
        self.feature_builder = self.predictor.feature_builder
        self.edge_calculator = EdgeCalculator()

        # Load tracking/PBP data for feature enrichment during inference
        self.tracking_data = {}  # {season: {player: {feature: value}}}
        self.pbp_data = {}  # {season: {player: {feature: value}}}
        self._tracking_loaded_at = 0  # timestamp of last load
        self._tracking_cache_ttl = 86400  # 24 hours

        self._load_tracking_data()

        # Initialize supplemental feature builders (optional, graceful degradation)
        self.team_builder = None
        self.derived_builder = None
        self.team_schedules = None

        self._init_enhanced_features()

        self._ready = self.predictor.is_ready()
        if self._ready:
            logger.info(
                "V2AnalysisEngine ready: %d prop models loaded",
                len(self.predictor.models),
            )
        else:
            logger.warning(
                "V2AnalysisEngine: no models loaded. "
                "Run 'python -m v2.train' to train models."
            )

    def is_ready(self) -> bool:
        """Check if v2 models are loaded and ready for inference."""
        return self._ready

    # -----------------------------------------------------------------
    # Tracking / PBP data loading
    # -----------------------------------------------------------------

    def _load_tracking_data(self):
        """Load tracking and PBP data from cached JSON files."""
        try:
            self.tracking_data = load_tracking_data()
            self.pbp_data = load_pbp_data()
            self._tracking_loaded_at = time.time()
            total_tracking = sum(len(v) for v in self.tracking_data.values())
            total_pbp = sum(len(v) for v in self.pbp_data.values())
            logger.info(
                "Loaded tracking data: %d players across %d seasons",
                total_tracking,
                len(self.tracking_data),
            )
            logger.info(
                "Loaded PBP data: %d players across %d seasons",
                total_pbp,
                len(self.pbp_data),
            )
        except Exception as e:
            logger.warning("Could not load tracking/PBP data: %s", e)
            self.tracking_data = {}
            self.pbp_data = {}

    def _ensure_tracking_fresh(self):
        """Reload tracking data if cache is stale."""
        if time.time() - self._tracking_loaded_at > self._tracking_cache_ttl:
            logger.info("Tracking data cache expired, reloading...")
            self._load_tracking_data()

    # -----------------------------------------------------------------
    # Enhanced feature builders (team, derived, travel/calendar)
    # -----------------------------------------------------------------

    def _init_enhanced_features(self):
        """Initialize supplemental feature builders from training game data.

        Each builder is wrapped in try/except so failures don't break the engine.
        If a builder can't be initialized, the engine continues without those
        enhanced features (the model defaults missing features to 0.0).
        """
        # Load training game data for building team history and schedules
        game_data = self._load_training_games()

        # TeamStatsBuilder: team-level four factors, pace, matchup features
        if _HAS_TEAM_FEATURES and game_data:
            try:
                self.team_builder = TeamStatsBuilder()
                self.team_builder.build_team_history(game_data)
                logger.info(
                    "TeamStatsBuilder initialized: %d teams",
                    len(self.team_builder.team_history),
                )
            except Exception as e:
                logger.warning("Failed to initialize TeamStatsBuilder: %s", e)
                self.team_builder = None
        else:
            if not _HAS_TEAM_FEATURES:
                logger.info("TeamStatsBuilder not available (import failed)")

        # DerivedFeatures: stateless player-level derived features
        if _HAS_DERIVED_FEATURES:
            try:
                self.derived_builder = DerivedFeatures()
                logger.info("DerivedFeatures initialized")
            except Exception as e:
                logger.warning("Failed to initialize DerivedFeatures: %s", e)
                self.derived_builder = None
        else:
            logger.info("DerivedFeatures not available (import failed)")

        # Travel/calendar: build team schedules from training data
        if _HAS_TRAVEL_FEATURES and game_data:
            try:
                self.team_schedules = build_team_schedules(game_data)
                logger.info(
                    "Team schedules built: %d teams",
                    len(self.team_schedules),
                )
            except Exception as e:
                logger.warning("Failed to build team schedules: %s", e)
                self.team_schedules = None
        else:
            if not _HAS_TRAVEL_FEATURES:
                logger.info("Travel features not available (import failed)")

    def _load_training_games(self) -> List[dict]:
        """Load training game data from backtest JSON files.

        Used to build team history and schedules for enhanced features.
        Returns empty list on failure.
        """
        import json
        from pathlib import Path

        base_dir = Path(__file__).resolve().parent.parent
        outputs = base_dir / "outputs"
        game_files = [
            outputs / "backtest_2023_24_data.json",
            outputs / "backtest_oos_data.json",
        ]

        all_games = []
        for path in game_files:
            if not path.exists():
                continue
            try:
                with open(path) as f:
                    data = json.load(f)
                games = data.get("games", [])
                all_games.extend(games)
            except Exception as e:
                logger.warning("Failed to load game data from %s: %s", path, e)

        all_games.sort(key=lambda g: g.get("date", ""))
        if all_games:
            logger.info("Loaded %d training games for enhanced features", len(all_games))
        return all_games

    def _build_enhanced_features(
        self,
        player_name: str,
        prop_type: str,
        game_date: str,
        is_home: Optional[bool],
        team_name: str,
        opp_name: str,
        line: float = 0.0,
    ) -> Dict:
        """Build all enhanced features for a single prediction.

        Collects team features, travel features, calendar features, and
        derived player features. Each group is wrapped in try/except so
        a failure in one group does not block the others.

        Args:
            player_name: e.g. "LeBron James"
            prop_type: e.g. "points"
            game_date: ISO date string
            is_home: whether player is on the home team (may be None)
            team_name: player's team name
            opp_name: opponent team name
            line: DK prop line value (for streak features)

        Returns:
            dict of enhanced feature_name -> float
        """
        enhanced = {}

        # 1. Team features (four factors, matchup, pace)
        if self.team_builder is not None and team_name and opp_name:
            try:
                team_feats = self.team_builder.get_all_features_for_game(
                    team_name, opp_name, game_date
                )
                if team_feats:
                    enhanced.update(team_feats)
            except Exception as e:
                logger.debug("Team features failed for %s: %s", player_name, e)

        # 2. Travel features
        if _HAS_TRAVEL_FEATURES and self.team_schedules and team_name:
            try:
                schedule = self.team_schedules.get(team_name, [])
                if schedule and is_home is not None:
                    travel_feats = get_travel_features(
                        team_name, opp_name, game_date, is_home, schedule
                    )
                    enhanced.update(travel_feats)
            except Exception as e:
                logger.debug("Travel features failed for %s: %s", player_name, e)

        # 3. Calendar features
        if _HAS_TRAVEL_FEATURES:
            try:
                cal_feats = get_calendar_features(game_date)
                enhanced.update(cal_feats)
            except Exception as e:
                logger.debug("Calendar features failed for %s: %s", player_name, e)

        # 4. Derived player features
        if self.derived_builder is not None:
            try:
                # Get prior games from the feature builder's player history
                player_history = self.feature_builder.player_history.get(player_name, [])
                if player_history:
                    # player_history is sorted most-recent-first after injection
                    # Filter to games before game_date
                    prior_games = [
                        g for g in player_history
                        if g.get("date", "") < game_date
                    ]
                    # Ensure most-recent-first ordering
                    prior_games.sort(key=lambda g: g.get("date", ""), reverse=True)

                    if len(prior_games) >= 3:
                        derived_ctx = {
                            "line": line,
                            "is_home": is_home,
                            "opponent": opp_name,
                            "player_name": player_name,
                        }
                        derived_feats = self.derived_builder.compute_all(
                            prior_games, prop_type, derived_ctx
                        )
                        enhanced.update(derived_feats)
            except Exception as e:
                logger.debug("Derived features failed for %s: %s", player_name, e)

        return enhanced

    def _get_player_tracking(self, player_name: str, game_date: str) -> Dict:
        """Get tracking features for a player at a given date."""
        self._ensure_tracking_fresh()
        season = date_to_season(game_date)
        if not season:
            return {}

        result = {}

        # Tracking stats — fall back to most recent season if current not available
        season_tracking = self.tracking_data.get(season, {})
        if not season_tracking:
            # Try previous seasons as fallback
            available = sorted(self.tracking_data.keys(), reverse=True)
            for s in available:
                if self.tracking_data[s]:
                    season_tracking = self.tracking_data[s]
                    break
        player_tracking = season_tracking.get(player_name)
        if player_tracking:
            result["has_tracking"] = 1.0
            for feat in TRACKING_FEATURES:
                val = player_tracking.get(feat, 0)
                try:
                    result[feat] = float(val) if val is not None else 0.0
                except (ValueError, TypeError):
                    result[feat] = 0.0
        else:
            result["has_tracking"] = 0.0

        # PBP stats — fall back to most recent season if current not available
        season_pbp = self.pbp_data.get(season, {})
        if not season_pbp:
            available = sorted(self.pbp_data.keys(), reverse=True)
            for s in available:
                if self.pbp_data[s]:
                    season_pbp = self.pbp_data[s]
                    break
        player_pbp = season_pbp.get(player_name)
        if not player_pbp:
            # Case-insensitive fallback
            for pname, pdata in season_pbp.items():
                if pname.lower() == player_name.lower():
                    player_pbp = pdata
                    break
        if player_pbp:
            result["has_pbp"] = 1.0
            for feat in PBP_FEATURES:
                val = player_pbp.get(feat, 0)
                try:
                    result[feat] = float(val) if val is not None else 0.0
                except (ValueError, TypeError):
                    result[feat] = 0.0
        else:
            result["has_pbp"] = 0.0

        return result

    # -----------------------------------------------------------------
    # Core analysis
    # -----------------------------------------------------------------

    def analyze_game(self, game_context, player_profiles, props, game_date=None):
        """
        Analyze all props for a game using v2 models.

        Args:
            game_context: GameContext object or dict with spread, total, pace, etc.
            player_profiles: list of PlayerProfile objects or dicts with player data
            props: list of dicts with keys:
                player_name, prop_type, line, over_odds, under_odds,
                and optionally all_books for cross-book data
            game_date: ISO date string (defaults to today)

        Returns:
            list of prediction dicts, sorted by edge * confidence descending.
            Each dict contains:
              - player, prop_type, line, over_odds, under_odds
              - predicted_value, p10, p90, adjustment
              - edge, recommended_side, confidence_stars
              - kelly_fraction, p_over, p_under
              - best_book info (if cross-book data available)
        """
        if not self._ready:
            logger.warning("V2 models not loaded, returning empty results")
            return []

        if game_date is None:
            from datetime import datetime
            game_date = datetime.now().strftime("%Y-%m-%d")

        # Ensure feature builder data is loaded
        self.predictor.ensure_data_loaded()

        # Extract context values
        if hasattr(game_context, '__dict__'):
            spread = getattr(game_context, 'spread', 0.0)
            total = getattr(game_context, 'total', 224.0)
        elif isinstance(game_context, dict):
            spread = game_context.get('spread', 0.0)
            total = game_context.get('total', 224.0)
        else:
            spread = 0.0
            total = 224.0

        # Extract home/away team names from game context
        if hasattr(game_context, 'home_team'):
            home_team = getattr(game_context, 'home_team', '')
            away_team = getattr(game_context, 'away_team', '')
        elif isinstance(game_context, dict):
            home_team = game_context.get('home_team', '')
            away_team = game_context.get('away_team', '')
        else:
            home_team = ''
            away_team = ''

        # Build a lookup for player home/away status and team assignment
        player_home_map = {}
        player_team_map = {}  # {player_name: team_name}
        for profile in player_profiles:
            if hasattr(profile, 'team'):
                pteam = profile.team
            elif isinstance(profile, dict):
                pteam = profile.get('team', '')
            else:
                pteam = ''
            pname = profile.name if hasattr(profile, 'name') else profile.get('name', '')
            if pteam:
                player_team_map[pname] = pteam
                if home_team:
                    player_home_map[pname] = (pteam.lower() in home_team.lower() or
                                               home_team.lower() in pteam.lower())

        predictions = []
        t0 = time.time()

        # Ensure feature builder data is loaded FIRST, then inject live game logs
        # so they don't get overwritten by ensure_data_loaded() inside predict()
        self.predictor.ensure_data_loaded()

        for profile in player_profiles:
            pname = profile.name if hasattr(profile, 'name') else profile.get('name', '')
            if pname and hasattr(profile, 'game_logs') and profile.game_logs:
                live_games = []
                for gl in profile.game_logs:
                    live_games.append({
                        'date': gl.date, 'pts': gl.pts, 'reb': gl.reb, 'ast': gl.ast,
                        'fg3m': gl.fg3m, 'stl': gl.stl, 'blk': gl.blk,
                        'turnover': gl.turnover, 'min': gl.minutes,
                        'fga': getattr(gl, 'fga', 0), 'fgm': getattr(gl, 'fgm', 0),
                        'fta': getattr(gl, 'fta', 0), 'ftm': getattr(gl, 'ftm', 0),
                    })
                self.predictor.feature_builder.player_history[pname] = sorted(
                    live_games, key=lambda g: g['date'], reverse=True
                )

        for prop in props:
            player_name = prop.get("player_name", "")
            prop_type = prop.get("prop_type", "")
            line = prop.get("line", 0)
            over_odds = prop.get("over_odds", -110)
            under_odds = prop.get("under_odds", -110)

            # Skip prop types v2 can't handle
            if prop_type not in V2_PROP_TYPES:
                continue

            # Build game context dict for the predictor
            ctx = {
                "line": line,
                "over_odds": over_odds,
                "under_odds": under_odds,
                "total": total,
                "spread": spread,
                "is_home": player_home_map.get(player_name),
                "all_books": prop.get("all_books", {}),
            }

            # Enrich context with tracking/PBP features for this player
            tracking_feats = self._get_player_tracking(player_name, game_date)
            if tracking_feats:
                ctx["tracking_features"] = tracking_feats

            # Build enhanced features (team, travel, calendar, derived)
            is_home = player_home_map.get(player_name)
            p_team = player_team_map.get(player_name, '')
            if is_home is True:
                opp_team = away_team
            elif is_home is False:
                opp_team = home_team
            else:
                # Fallback: if we know the player's team, infer opponent
                if p_team and home_team and (p_team.lower() in home_team.lower()
                                              or home_team.lower() in p_team.lower()):
                    opp_team = away_team
                elif p_team and away_team and (p_team.lower() in away_team.lower()
                                                or away_team.lower() in p_team.lower()):
                    opp_team = home_team
                else:
                    opp_team = ''

            enhanced_feats = self._build_enhanced_features(
                player_name=player_name,
                prop_type=prop_type,
                game_date=game_date,
                is_home=is_home,
                team_name=p_team,
                opp_name=opp_team,
                line=line,
            )
            if enhanced_feats:
                ctx["enhanced_features"] = enhanced_feats

            # Run v2 prediction
            result = self.predictor.analyze_prop(
                player_name=player_name,
                prop_type=prop_type,
                game_date=game_date,
                line=line,
                over_odds=over_odds,
                under_odds=under_odds,
                game_context=ctx,
            )

            if result is None:
                continue

            # Add source metadata and original odds (not in analyze_prop output)
            if result.get("model_type") == "v6_dual":
                result["source"] = "v6_dual"
            else:
                result["source"] = "v2_line_anchor"
            result["over_odds"] = over_odds
            result["under_odds"] = under_odds

            # Include best book info from cross-book data
            all_books = prop.get("all_books", {})
            if all_books:
                result["all_books"] = all_books

            predictions.append(result)

        elapsed = time.time() - t0
        logger.info(
            "V2 analyzed %d props in %.2fs (%.0f props/sec)",
            len(predictions),
            elapsed,
            len(predictions) / max(elapsed, 0.001),
        )

        # Sort by abs(edge) * confidence descending
        predictions.sort(
            key=lambda p: abs(p.get("edge", 0)) * p.get("confidence", 0),
            reverse=True,
        )

        return predictions

    def build_parlays(self, predictions, max_parlays=5):
        """
        Build 2-leg parlays from v2 predictions.

        Rules:
        - Each leg must have edge > 2%
        - Each leg must be from a DIFFERENT player
        - Prefer under picks (proven +5.2% ROI bias)
        - Sort parlays by combined edge
        - Max 5 parlays
        - No duplicate legs across parlays

        Args:
            predictions: list of formatted v2 prediction dicts
                (output of format_prediction_for_web or serialize_v2_prediction)
            max_parlays: maximum number of parlays to return

        Returns:
            list of parlay dicts, each with:
              - legs: list of 2 prediction dicts
              - combined_decimal_odds: float
              - combined_american_odds: int
              - combined_edge: float (average edge of legs)
              - combined_probability: float
        """
        # Filter to actionable picks with edge > 2%
        good_picks = [
            p for p in predictions
            if p.get("edge_percent", 0) > 2 and p.get("side", "pass") != "pass"
        ]

        # Sort: prefer unders first (proven ROI bias), then by edge descending
        good_picks.sort(
            key=lambda x: (
                0 if x.get("side") == "under" else 1,
                -x.get("edge_percent", 0),
            )
        )

        parlays = []
        used_players = set()

        for i, leg1 in enumerate(good_picks):
            if len(parlays) >= max_parlays:
                break
            player1 = leg1.get("player", "")
            if player1 in used_players:
                continue

            for leg2 in good_picks[i + 1:]:
                player2 = leg2.get("player", "")
                if player2 == player1:
                    continue
                if player2 in used_players:
                    continue

                # Calculate combined parlay odds
                dec1 = self._american_to_decimal(leg1.get("dk_odds", -110))
                dec2 = self._american_to_decimal(leg2.get("dk_odds", -110))
                combined_decimal = dec1 * dec2

                # Combined implied probability (independent)
                imp1 = leg1.get("implied_probability", 0.5)
                imp2 = leg2.get("implied_probability", 0.5)
                combined_implied = imp1 * imp2

                # Combined model probability (independent)
                prob1 = leg1.get("our_probability", 0.5)
                prob2 = leg2.get("our_probability", 0.5)
                combined_model_prob = prob1 * prob2

                # Combined edge: model prob vs implied prob
                combined_edge = combined_model_prob - combined_implied
                combined_edge_pct = round(combined_edge * 100, 2)

                # American odds from decimal
                combined_american = self._decimal_to_american(combined_decimal)

                # EV per unit
                ev = combined_model_prob * (combined_decimal - 1) - (1 - combined_model_prob)

                parlays.append({
                    "legs": [leg1, leg2],
                    "combined_decimal_odds": round(combined_decimal, 3),
                    "combined_american_odds": combined_american,
                    "combined_edge": combined_edge_pct,
                    "combined_probability": round(combined_model_prob, 4),
                    "combined_implied_probability": round(combined_implied, 4),
                    "ev_per_unit": round(ev, 4),
                })

                used_players.add(player1)
                used_players.add(player2)
                break

        # Sort by combined edge descending
        parlays.sort(key=lambda x: x["combined_edge"], reverse=True)
        return parlays[:max_parlays]

    @staticmethod
    def _american_to_decimal(american_odds):
        """Convert American odds to decimal odds."""
        if american_odds >= 100:
            return 1 + american_odds / 100.0
        elif american_odds <= -100:
            return 1 + 100.0 / abs(american_odds)
        else:
            return 1.91  # fallback for -110

    @staticmethod
    def _decimal_to_american(decimal_odds):
        """Convert decimal odds to American odds."""
        if decimal_odds >= 2.0:
            return int(round((decimal_odds - 1) * 100))
        elif decimal_odds > 1.0:
            return int(round(-100 / (decimal_odds - 1)))
        else:
            return -110  # fallback

    def get_top_picks(self, predictions, min_edge=0.03, max_picks=10):
        """
        Filter and rank predictions into top picks.

        Args:
            predictions: list from analyze_game()
            min_edge: minimum edge threshold (default 3%)
            max_picks: maximum picks to return

        Returns:
            list of top picks with edge >= min_edge, sorted by edge descending
        """
        picks = []
        for pred in predictions:
            side = pred.get("recommended_side", "PASS")
            edge = pred.get("edge", 0)
            if side != "PASS" and edge >= min_edge:
                picks.append(pred)

        picks.sort(key=lambda p: p.get("edge", 0), reverse=True)
        return picks[:max_picks]

    def format_prediction_for_web(self, pred):
        """
        Format a v2 prediction dict for the web UI serializer.

        Maps v2 output fields to the format expected by serialize_bet() and
        the frontend JavaScript.

        Returns:
            dict compatible with the web UI bet card format
        """
        line = pred.get("line", 0)
        predicted = pred.get("predicted", line)
        adjustment = pred.get("adjustment", 0)
        side = pred.get("recommended_side", "PASS")
        edge = pred.get("edge", 0)
        stars = pred.get("confidence_stars", 1)

        # Determine the displayed side: use edge calculator's pick, or model probability if PASS
        if side in ("OVER", "UNDER"):
            displayed_side = side.lower()
        else:
            displayed_side = "under" if pred.get("p_under", 0) > pred.get("p_over", 0) else "over"

        return {
            # Core bet info
            "player": pred.get("player", ""),
            "prop_type": pred.get("prop_type", ""),
            "side": displayed_side,
            "line": line,
            "dk_odds": pred.get("over_odds", -110) if displayed_side == "over" else pred.get("under_odds", -110),
            "over_odds": pred.get("over_odds", -110),
            "under_odds": pred.get("under_odds", -110),

            # Projection data (v2 specific)
            "predicted_value": round(predicted, 2),
            "projected_value": round(predicted, 2),  # alias
            "predicted_adjustment": round(adjustment, 3),
            "p10": pred.get("p10_prediction", round(predicted - 5, 1)),
            "p90": pred.get("p90_prediction", round(predicted + 5, 1)),
            "uncertainty": pred.get("uncertainty", 10.0),
            "confidence": pred.get("confidence", 0.5),
            "confidence_stars": stars,

            # Edge data — compute for the DISPLAYED side (not the raw edge calculator side)
            "edge_percent": round(
                (pred.get("edge_over", 0) if displayed_side == "over" else pred.get("edge_under", 0)) * 100, 2
            ) if side == "PASS" else round(edge * 100, 2),
            "edge": round(
                pred.get("edge_over", 0) if displayed_side == "over" else pred.get("edge_under", 0), 4
            ) if side == "PASS" else round(edge, 4),
            "our_probability": pred.get("p_over", 0.5) if displayed_side == "over" else pred.get("p_under", 0.5),
            "implied_probability": pred.get("implied_over", 0.5) if displayed_side == "over" else pred.get("implied_under", 0.5),
            "p_over": pred.get("p_over", 0.5),
            "p_under": pred.get("p_under", 0.5),

            # Sizing
            "kelly_fraction": pred.get("kelly_fraction", 0.0),
            "recommended_units": round(pred.get("kelly_fraction", 0.0) * 100, 1),

            # Cross-book
            "all_books": pred.get("all_books", {}),

            # Signal info
            "sigma": pred.get("sigma", 0),
            "z_score": pred.get("z_score", 0),
            "cross_book_signal": pred.get("cross_book_signal", "NEUTRAL"),

            # Source
            "source": pred.get("source", "v2_line_anchor"),
            "target_mode": pred.get("target_mode", "line_anchor"),

            # v6 dual-model extras (passed through for logging/display)
            "p_over_huber": pred.get("p_over_huber"),
            "p_over_rf": pred.get("p_over_rf"),
            "consensus": pred.get("consensus", False),
        }
