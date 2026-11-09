#!/usr/bin/env python3
"""
v2/ref_features.py - Referee data pipeline for predicting player prop outcomes.

Different refs call different foul rates (impacts FTA -> points), allow different
pace (impacts possessions -> all stats), and some have home whistle bias.

Data source: nba_api endpoints from NBA.com
  - BoxScoreSummaryV2: officials per game
  - ScoreboardV2: today's ref assignments (published ~1-2hrs before tip)
  - LeagueGameLog: game-level stats for computing ref tendencies

Features produced per ref crew (6 features + 1 flag):
  ref_avg_fta_per_game     - crew's average total FTA across their games
  ref_avg_total_score      - crew's average total points
  ref_avg_pace             - crew's average game pace
  ref_fta_vs_league_avg    - ratio of crew FTA to league average
  ref_home_fta_bias        - fraction of FTA going to home team
  ref_games_sample         - sample size (crew games worked)
  has_ref_data             - 1 if ref data available, 0 otherwise

Rate limiting: 0.6s sleep between nba_api calls.
Caching: results saved to disk to avoid re-fetching.

Usage:
    python -m v2.ref_features                      # Build ref DB for training seasons
    python -m v2.ref_features --season 2023-24     # Single season
    python -m v2.ref_features --today              # Fetch today's ref assignments
"""

import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS = BASE_DIR / "outputs"
REF_CACHE_DIR = OUTPUTS / "ref_cache"

# NBA API rate limit: sleep between calls
API_SLEEP = 0.6

# Seasons to fetch ref data for (matching training data)
TRAINING_SEASONS = ["2023-24", "2025-26"]

# Season string -> nba_api season format (e.g. "2023-24" -> "2023-24")
# nba_api uses the same format, so no conversion needed.


# =============================================================================
# RefStatsBuilder
# =============================================================================

class RefStatsBuilder:
    """Builds referee tendency database from NBA.com game data.

    Pipeline:
      1. fetch_ref_assignments(season) -> {game_id: [ref1, ref2, ref3]}
      2. fetch_game_stats(season) -> {game_id: {home_fta, away_fta, total_pts, pace, ...}}
      3. build_ref_tendencies(assignments, game_stats) -> {ref_name: {tendency_stats}}
      4. get_ref_features(ref_names) -> feature dict for a game's ref crew
    """

    def __init__(self):
        self.ref_assignments: Dict[str, Dict[int, List[str]]] = {}  # season -> {game_id: [refs]}
        self.game_stats: Dict[str, Dict[int, Dict]] = {}            # season -> {game_id: stats}
        self.ref_tendencies: Dict[str, Dict[str, Dict]] = {}        # season -> {ref: tendencies}
        REF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # 1. Fetch ref assignments for a season
    # -----------------------------------------------------------------

    def fetch_ref_assignments(self, season: str) -> Dict[int, List[str]]:
        """Fetch referee assignments for all games in a season.

        Uses BoxScoreSummaryV2 for each game to get the Officials dataset.
        This is expensive (one API call per game), so results are cached.

        Args:
            season: NBA season string, e.g. "2023-24"

        Returns:
            dict mapping game_id -> [ref_name_1, ref_name_2, ref_name_3]
        """
        cache_path = REF_CACHE_DIR / f"ref_assignments_{season}.json"
        if cache_path.exists():
            logger.info("Loading cached ref assignments for %s", season)
            with open(cache_path) as f:
                data = json.load(f)
            # JSON keys are strings; convert back to int
            assignments = {int(k): v for k, v in data.items()}
            self.ref_assignments[season] = assignments
            logger.info("  -> %d games with ref data", len(assignments))
            return assignments

        logger.info("Fetching ref assignments for season %s (this will take a while)...", season)

        try:
            from nba_api.stats.endpoints import LeagueGameLog, BoxScoreSummaryV2
        except ImportError:
            logger.error("nba_api not installed. Run: pip install nba_api")
            return {}

        # Step 1: Get all game IDs for the season
        time.sleep(API_SLEEP)
        game_log = LeagueGameLog(
            season=season,
            season_type_all_star="Regular Season",
        )
        games_df = game_log.get_data_frames()[0]

        # Each row is a team-game, so unique game IDs appear twice
        game_ids = sorted(games_df["GAME_ID"].unique())
        logger.info("  Found %d unique games for %s", len(game_ids), season)

        assignments = {}
        fetched = 0
        errors = 0

        for i, game_id in enumerate(game_ids):
            if i > 0 and i % 50 == 0:
                logger.info(
                    "  Progress: %d/%d games (%.0f%%), %d errors",
                    i, len(game_ids), 100 * i / len(game_ids), errors,
                )
                # Save intermediate results
                self._save_assignments_cache(assignments, season, cache_path)

            try:
                time.sleep(API_SLEEP)
                box = BoxScoreSummaryV2(game_id=str(game_id))
                officials_df = box.officials.get_data_frame()

                if officials_df is not None and len(officials_df) > 0:
                    ref_names = []
                    for _, row in officials_df.iterrows():
                        first = row.get("FIRST_NAME", "")
                        last = row.get("LAST_NAME", "")
                        name = f"{first} {last}".strip()
                        if name:
                            ref_names.append(name)
                    if ref_names:
                        assignments[int(game_id)] = ref_names
                        fetched += 1

            except Exception as e:
                errors += 1
                if errors <= 5:
                    logger.warning("  Error fetching game %s: %s", game_id, e)

        self._save_assignments_cache(assignments, season, cache_path)
        self.ref_assignments[season] = assignments
        logger.info(
            "Ref assignments for %s: %d games fetched, %d errors",
            season, fetched, errors,
        )
        return assignments

    def _save_assignments_cache(
        self, assignments: Dict[int, List[str]], season: str, path: Path
    ):
        """Save assignments to cache file."""
        # Convert int keys to strings for JSON
        data = {str(k): v for k, v in assignments.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("  Saved %d assignments to %s", len(assignments), path.name)

    # -----------------------------------------------------------------
    # 2. Fetch game-level stats for computing ref tendencies
    # -----------------------------------------------------------------

    def fetch_game_stats(self, season: str) -> Dict[int, Dict]:
        """Fetch game-level stats (FTA, total points, pace) for a season.

        Uses LeagueGameLog which returns team-level stats per game.
        Two rows per game (home + away), which we combine.

        Args:
            season: NBA season string

        Returns:
            dict mapping game_id -> {home_fta, away_fta, total_pts, total_fta, pace_proxy}
        """
        cache_path = REF_CACHE_DIR / f"game_stats_{season}.json"
        if cache_path.exists():
            logger.info("Loading cached game stats for %s", season)
            with open(cache_path) as f:
                data = json.load(f)
            stats = {int(k): v for k, v in data.items()}
            self.game_stats[season] = stats
            return stats

        logger.info("Fetching game stats for season %s ...", season)

        try:
            from nba_api.stats.endpoints import LeagueGameLog
        except ImportError:
            logger.error("nba_api not installed.")
            return {}

        time.sleep(API_SLEEP)
        game_log = LeagueGameLog(
            season=season,
            season_type_all_star="Regular Season",
        )
        df = game_log.get_data_frames()[0]

        # Group by GAME_ID, aggregate home/away stats
        # MATCHUP contains "vs." for home games and "@" for away games
        game_stats = {}
        grouped = df.groupby("GAME_ID")

        for game_id, group in grouped:
            home_rows = group[group["MATCHUP"].str.contains("vs.", na=False)]
            away_rows = group[group["MATCHUP"].str.contains("@", na=False)]

            home_fta = float(home_rows["FTA"].sum()) if len(home_rows) > 0 else 0.0
            away_fta = float(away_rows["FTA"].sum()) if len(away_rows) > 0 else 0.0
            home_pts = float(home_rows["PTS"].sum()) if len(home_rows) > 0 else 0.0
            away_pts = float(away_rows["PTS"].sum()) if len(away_rows) > 0 else 0.0

            # Pace proxy: total FGA + 0.44*FTA - OREB + TOV (per team, then average)
            # Simplified: use total possessions estimate
            total_fga = float(group["FGA"].sum())
            total_fta = home_fta + away_fta
            total_oreb = float(group["OREB"].sum())
            total_tov = float(group["TOV"].sum())

            # Simple pace proxy: average possessions per team
            # possessions ~ FGA + 0.44*FTA - OREB + TOV
            poss_estimate = total_fga + 0.44 * total_fta - total_oreb + total_tov
            pace_proxy = poss_estimate / 2.0  # per team

            game_date = None
            if len(group) > 0:
                game_date = str(group.iloc[0].get("GAME_DATE", ""))

            game_stats[int(game_id)] = {
                "home_fta": home_fta,
                "away_fta": away_fta,
                "total_fta": total_fta,
                "total_pts": home_pts + away_pts,
                "home_pts": home_pts,
                "away_pts": away_pts,
                "pace_proxy": pace_proxy,
                "game_date": game_date,
            }

        # Cache
        data = {str(k): v for k, v in game_stats.items()}
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)

        self.game_stats[season] = game_stats
        logger.info("Game stats for %s: %d games", season, len(game_stats))
        return game_stats

    # -----------------------------------------------------------------
    # 3. Build ref tendencies from assignments + game stats
    # -----------------------------------------------------------------

    def build_ref_tendencies(self, season: str) -> Dict[str, Dict]:
        """Compute per-referee tendency stats for a season.

        For each ref, aggregates stats across all games they officiated:
          - avg_fta_per_game: average total FTA
          - avg_total_score: average total points
          - avg_pace: average pace proxy
          - home_fta_share: fraction of total FTA going to home team
          - games_reffed: sample size

        Args:
            season: NBA season string

        Returns:
            dict mapping ref_name -> {tendency stats}
        """
        assignments = self.ref_assignments.get(season, {})
        stats = self.game_stats.get(season, {})

        if not assignments or not stats:
            logger.warning(
                "Missing data for %s: %d assignments, %d game stats",
                season, len(assignments), len(stats),
            )
            return {}

        # Accumulate per ref
        ref_data = defaultdict(lambda: {
            "total_fta": [],
            "total_pts": [],
            "pace": [],
            "home_fta_share": [],
        })

        matched = 0
        for game_id, ref_names in assignments.items():
            game = stats.get(game_id)
            if game is None:
                continue
            matched += 1

            total_fta = game["total_fta"]
            total_pts = game["total_pts"]
            pace = game["pace_proxy"]
            home_fta = game["home_fta"]
            home_share = home_fta / max(total_fta, 1.0)

            for ref in ref_names:
                ref_data[ref]["total_fta"].append(total_fta)
                ref_data[ref]["total_pts"].append(total_pts)
                ref_data[ref]["pace"].append(pace)
                ref_data[ref]["home_fta_share"].append(home_share)

        logger.info(
            "Building tendencies for %d refs (%d/%d games matched)",
            len(ref_data), matched, len(assignments),
        )

        # Compute league averages for normalization
        all_fta = [s["total_fta"] for s in stats.values()]
        league_avg_fta = np.mean(all_fta) if all_fta else 44.0

        tendencies = {}
        for ref, data in ref_data.items():
            games_reffed = len(data["total_fta"])
            if games_reffed < 3:
                continue  # Not enough sample

            avg_fta = float(np.mean(data["total_fta"]))
            tendencies[ref] = {
                "avg_fta_per_game": avg_fta,
                "avg_total_score": float(np.mean(data["total_pts"])),
                "avg_pace": float(np.mean(data["pace"])),
                "fta_vs_league_avg": avg_fta / max(league_avg_fta, 1.0),
                "home_fta_share": float(np.mean(data["home_fta_share"])),
                "games_reffed": games_reffed,
            }

        # Cache tendencies
        cache_path = REF_CACHE_DIR / f"ref_tendencies_{season}.json"
        with open(cache_path, "w") as f:
            json.dump(tendencies, f, indent=2)

        self.ref_tendencies[season] = tendencies
        logger.info(
            "Ref tendencies for %s: %d refs (min %d games, max %d games)",
            season,
            len(tendencies),
            min(t["games_reffed"] for t in tendencies.values()) if tendencies else 0,
            max(t["games_reffed"] for t in tendencies.values()) if tendencies else 0,
        )
        return tendencies

    # -----------------------------------------------------------------
    # 4. Get features for a game's ref crew
    # -----------------------------------------------------------------

    def get_ref_features(
        self,
        ref_names: List[str],
        season: Optional[str] = None,
    ) -> Dict[str, float]:
        """Compute ref features for a game's officiating crew.

        Averages each ref's tendencies across the crew (typically 3 refs).
        Returns zeros with has_ref_data=0 if no ref data available.

        Args:
            ref_names: list of referee names for the game
            season: season to look up tendencies for (uses most recent if None)

        Returns:
            dict of 7 features
        """
        empty = {
            "ref_avg_fta_per_game": 0.0,
            "ref_avg_total_score": 0.0,
            "ref_avg_pace": 0.0,
            "ref_fta_vs_league_avg": 1.0,
            "ref_home_fta_bias": 0.5,
            "ref_games_sample": 0.0,
            "has_ref_data": 0,
        }

        if not ref_names:
            return empty

        # Find the right season's tendencies
        tendencies = None
        if season and season in self.ref_tendencies:
            tendencies = self.ref_tendencies[season]
        elif self.ref_tendencies:
            # Use most recent season
            latest = sorted(self.ref_tendencies.keys())[-1]
            tendencies = self.ref_tendencies[latest]

        if not tendencies:
            return empty

        # Collect stats for matched refs
        matched_stats = []
        for ref in ref_names:
            if ref in tendencies:
                matched_stats.append(tendencies[ref])
            else:
                # Try fuzzy match (last name only)
                ref_last = ref.split()[-1] if ref.strip() else ""
                for known_ref, stats in tendencies.items():
                    if known_ref.split()[-1] == ref_last:
                        matched_stats.append(stats)
                        break

        if not matched_stats:
            return empty

        # Average across matched crew members
        return {
            "ref_avg_fta_per_game": float(np.mean([s["avg_fta_per_game"] for s in matched_stats])),
            "ref_avg_total_score": float(np.mean([s["avg_total_score"] for s in matched_stats])),
            "ref_avg_pace": float(np.mean([s["avg_pace"] for s in matched_stats])),
            "ref_fta_vs_league_avg": float(np.mean([s["fta_vs_league_avg"] for s in matched_stats])),
            "ref_home_fta_bias": float(np.mean([s["home_fta_share"] for s in matched_stats])),
            "ref_games_sample": float(np.mean([s["games_reffed"] for s in matched_stats])),
            "has_ref_data": 1,
        }

    # -----------------------------------------------------------------
    # 5. Fetch today's ref assignments
    # -----------------------------------------------------------------

    def fetch_todays_refs(
        self, game_date: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """Fetch referee assignments for today's (or a given date's) games.

        Ref assignments are published ~1-2 hours before tip on NBA.com.
        Uses ScoreboardV2 which includes an Officials dataset per game.

        Args:
            game_date: date string "YYYY-MM-DD" or None for today

        Returns:
            dict mapping matchup string (e.g. "BOS vs. NYK") -> [ref1, ref2, ref3]
        """
        try:
            from nba_api.stats.endpoints import ScoreboardV2
        except ImportError:
            logger.error("nba_api not installed.")
            return {}

        if game_date is None:
            game_date = datetime.now().strftime("%Y-%m-%d")

        # ScoreboardV2 expects MM/DD/YYYY format
        parts = game_date.split("-")
        formatted_date = f"{parts[1]}/{parts[2]}/{parts[0]}"

        logger.info("Fetching ref assignments for %s ...", game_date)

        try:
            time.sleep(API_SLEEP)
            scoreboard = ScoreboardV2(game_date=formatted_date)

            game_header = scoreboard.game_header.get_data_frame()
            officials_info = {}

            # ScoreboardV2 does not directly map officials to games in all
            # versions. Fall back to per-game BoxScoreSummaryV2 if needed.
            if hasattr(scoreboard, "available") and len(game_header) > 0:
                # Try to get officials from the scoreboard data
                pass

            # More reliable: fetch BoxScoreSummaryV2 for each game
            from nba_api.stats.endpoints import BoxScoreSummaryV2

            for _, row in game_header.iterrows():
                game_id = row.get("GAME_ID")
                home_team = row.get("HOME_TEAM_ID", "")
                visitor_team = row.get("VISITOR_TEAM_ID", "")
                matchup = f"{row.get('GAMECODE', game_id)}"

                try:
                    time.sleep(API_SLEEP)
                    box = BoxScoreSummaryV2(game_id=str(game_id))
                    officials_df = box.officials.get_data_frame()

                    if officials_df is not None and len(officials_df) > 0:
                        refs = []
                        for _, orow in officials_df.iterrows():
                            first = orow.get("FIRST_NAME", "")
                            last = orow.get("LAST_NAME", "")
                            name = f"{first} {last}".strip()
                            if name:
                                refs.append(name)
                        if refs:
                            officials_info[matchup] = refs
                except Exception as e:
                    logger.warning("Error fetching officials for game %s: %s", game_id, e)

            logger.info("Found refs for %d/%d games", len(officials_info), len(game_header))
            return officials_info

        except Exception as e:
            logger.error("Error fetching scoreboard for %s: %s", game_date, e)
            return {}

    # -----------------------------------------------------------------
    # 6. Load cached tendencies (for use in enrichment without re-fetching)
    # -----------------------------------------------------------------

    def load_cached_tendencies(self, season: Optional[str] = None) -> bool:
        """Load ref tendencies from disk cache.

        Args:
            season: specific season to load, or None to load all cached seasons

        Returns:
            True if any tendencies were loaded
        """
        loaded = False

        if season:
            paths = [REF_CACHE_DIR / f"ref_tendencies_{season}.json"]
        else:
            paths = sorted(REF_CACHE_DIR.glob("ref_tendencies_*.json"))

        for path in paths:
            if path.exists():
                with open(path) as f:
                    tendencies = json.load(f)
                # Extract season from filename
                s = path.stem.replace("ref_tendencies_", "")
                self.ref_tendencies[s] = tendencies
                logger.info("Loaded %d ref tendencies for %s", len(tendencies), s)
                loaded = True

        return loaded

    # -----------------------------------------------------------------
    # 7. Enrich training data with ref features
    # -----------------------------------------------------------------

    def get_ref_features_for_game(
        self,
        game_id: int,
        season: str,
    ) -> Dict[str, float]:
        """Get ref features for a specific historical game.

        Looks up the ref crew from assignments, then computes crew features.

        Args:
            game_id: NBA game ID
            season: season string

        Returns:
            dict of ref features (zeros + has_ref_data=0 if not available)
        """
        assignments = self.ref_assignments.get(season, {})
        ref_names = assignments.get(game_id, [])
        return self.get_ref_features(ref_names, season=season)

    # -----------------------------------------------------------------
    # Full pipeline: build everything for a season
    # -----------------------------------------------------------------

    def build_season(self, season: str):
        """Run the full pipeline for one season: assignments, stats, tendencies.

        Args:
            season: NBA season string, e.g. "2023-24"
        """
        logger.info("=" * 60)
        logger.info("Building ref data for season %s", season)
        logger.info("=" * 60)

        self.fetch_ref_assignments(season)
        self.fetch_game_stats(season)
        self.build_ref_tendencies(season)


# =============================================================================
# Feature names (for use by enrich_training_data and train modules)
# =============================================================================

REF_FEATURES = [
    "ref_avg_fta_per_game",
    "ref_avg_total_score",
    "ref_avg_pace",
    "ref_fta_vs_league_avg",
    "ref_home_fta_bias",
    "ref_games_sample",
    "has_ref_data",
]


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build referee tendency database")
    parser.add_argument(
        "--season",
        type=str,
        default=None,
        help="Single season to fetch (e.g. 2023-24). Default: all training seasons.",
    )
    parser.add_argument(
        "--today",
        action="store_true",
        help="Fetch today's ref assignments and print features.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Fetch ref assignments for a specific date (YYYY-MM-DD).",
    )
    args = parser.parse_args()

    builder = RefStatsBuilder()

    if args.today or args.date:
        # Live inference mode: fetch today's refs and show features
        game_date = args.date
        officials = builder.fetch_todays_refs(game_date)

        if not officials:
            print("No ref assignments found (may not be published yet).")
            return

        # Load tendencies for feature computation
        builder.load_cached_tendencies()

        print(f"\n{'='*70}")
        print(f"REFEREE ASSIGNMENTS - {game_date or 'Today'}")
        print(f"{'='*70}")

        for matchup, refs in officials.items():
            print(f"\n  {matchup}")
            print(f"    Refs: {', '.join(refs)}")
            features = builder.get_ref_features(refs)
            if features["has_ref_data"]:
                print(f"    FTA/game:        {features['ref_avg_fta_per_game']:.1f}")
                print(f"    Total score:     {features['ref_avg_total_score']:.1f}")
                print(f"    Pace:            {features['ref_avg_pace']:.1f}")
                print(f"    FTA vs league:   {features['ref_fta_vs_league_avg']:.3f}")
                print(f"    Home FTA bias:   {features['ref_home_fta_bias']:.3f}")
                print(f"    Sample size:     {features['ref_games_sample']:.0f}")
            else:
                print(f"    (no tendency data available for these refs)")

    else:
        # Build mode: fetch all data for training season(s)
        seasons = [args.season] if args.season else TRAINING_SEASONS

        for season in seasons:
            builder.build_season(season)

        # Print summary
        print(f"\n{'='*70}")
        print(f"REFEREE TENDENCY DATABASE SUMMARY")
        print(f"{'='*70}")

        for season in seasons:
            tendencies = builder.ref_tendencies.get(season, {})
            if not tendencies:
                print(f"\n  {season}: No data")
                continue

            fta_values = [t["avg_fta_per_game"] for t in tendencies.values()]
            score_values = [t["avg_total_score"] for t in tendencies.values()]
            games_values = [t["games_reffed"] for t in tendencies.values()]

            print(f"\n  {season}: {len(tendencies)} referees")
            print(f"    FTA/game range:    {min(fta_values):.1f} - {max(fta_values):.1f} "
                  f"(league avg: {np.mean(fta_values):.1f})")
            print(f"    Total score range: {min(score_values):.1f} - {max(score_values):.1f}")
            print(f"    Games/ref range:   {min(games_values)} - {max(games_values)}")

            # Top 5 highest FTA refs
            sorted_refs = sorted(tendencies.items(), key=lambda x: x[1]["avg_fta_per_game"], reverse=True)
            print(f"\n    Top 5 highest FTA refs:")
            for ref, stats in sorted_refs[:5]:
                print(f"      {ref:25s}  FTA={stats['avg_fta_per_game']:.1f}  "
                      f"({stats['games_reffed']} games)")

            # Top 5 lowest FTA refs
            print(f"\n    Top 5 lowest FTA refs:")
            for ref, stats in sorted_refs[-5:]:
                print(f"      {ref:25s}  FTA={stats['avg_fta_per_game']:.1f}  "
                      f"({stats['games_reffed']} games)")

        print()


if __name__ == "__main__":
    main()
