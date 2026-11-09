"""
injury_cascade.py - Historical Injury Cascade Impact Tables

Analyzes 3 seasons of game data to compute: when Player X sits out,
how do each teammate's stats change? This is the foundation for detecting
mispriced props immediately after injury news breaks.

Usage:
    python -m v2.injury_cascade              # Build tables from all backtest data
    python -m v2.injury_cascade --team BOS   # Build for a specific team only
"""

import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILES = [
    os.path.join(PROJECT_ROOT, "outputs", "backtest_2023_24_data.json"),
    os.path.join(PROJECT_ROOT, "outputs", "backtest_2024_25_data.json"),
    os.path.join(PROJECT_ROOT, "outputs", "backtest_oos_data.json"),
]

CASCADE_TABLE_PATH = os.path.join(PROJECT_ROOT, "data", "injury_cascade_tables.json")

# Props we track for cascade impacts
STAT_KEYS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover", "min"]

# Prop name mapping (stat key -> DK prop name)
STAT_TO_PROP = {
    "pts": "points",
    "reb": "rebounds",
    "ast": "assists",
    "fg3m": "threes",
    "stl": "steals",
    "blk": "blocks",
    "turnover": "turnovers",
    "min": "minutes",
}

# Minimum sample requirements
MIN_GAMES_WITH = 5     # Minimum games the absent player played in
MIN_GAMES_WITHOUT = 5  # Minimum games the absent player sat out
MIN_GAMES_PLAYED = 20  # Minimum total games for a player to be considered a "regular"
MIN_BENEF_GAMES = 20   # Beneficiary must also be a rotation player
P_VALUE_THRESHOLD = 0.10  # Statistical significance threshold

# Minimum baseline stat thresholds (filters out garbage-time bench players)
# Only compute boost if the "with" average exceeds this floor
MIN_BASELINE = {
    "pts": 4.0,
    "reb": 1.5,
    "ast": 1.0,
    "fg3m": 0.5,
    "stl": 0.3,
    "blk": 0.2,
    "turnover": 0.5,
}

# NBA team abbreviation -> full name mapping
TEAM_ABBREVS = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies", "MIA": "Miami Heat", "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves", "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs", "TOR": "Toronto Raptors", "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}

FULL_TO_ABBREV = {v: k for k, v in TEAM_ABBREVS.items()}

# Handle variant team names (BDL uses "LA Clippers" sometimes, "Los Angeles Clippers" others)
TEAM_NAME_ALIASES = {
    "Los Angeles Clippers": "LA Clippers",
}


def normalize_team_name(name: str) -> str:
    """Normalize team name variants to a canonical form."""
    return TEAM_NAME_ALIASES.get(name, name)


# -------------------------------------------------------------------------
# Data Loading & Player-Team Mapping
# -------------------------------------------------------------------------

def load_all_game_data() -> List[dict]:
    """Load and merge game data from all 3 seasons."""
    all_games = []
    for path in DATA_FILES:
        if not os.path.exists(path):
            logger.warning("Data file not found: %s", path)
            continue
        with open(path) as f:
            data = json.load(f)
        season_games = data.get("games", [])
        season_label = data.get("metadata", {}).get("date_range", path)
        logger.info("Loaded %d games from %s", len(season_games), season_label)
        all_games.extend(season_games)
    logger.info("Total games loaded: %d", len(all_games))
    return all_games


def build_player_team_mapping(games: List[dict]) -> Dict[str, str]:
    """
    Determine which team each player belongs to.

    The backtest data has a quirk: box_score for a given date contains ALL
    players from ALL games that day, not just from the specific game. So we
    use a voting approach: for each game a player appears in, we "vote" for
    both the home and away team. The player's actual team gets the most votes
    because they appear in every game their team plays, but only appear in
    other games when those games fall on the same date.
    """
    player_team_votes = defaultdict(lambda: defaultdict(int))

    for game in games:
        home = normalize_team_name(game["home_team"])
        away = normalize_team_name(game["away_team"])
        for player_name in game.get("box_score", {}):
            player_team_votes[player_name][home] += 1
            player_team_votes[player_name][away] += 1

    player_to_team = {}
    for player, votes in player_team_votes.items():
        if not votes:
            continue
        best_team = max(votes, key=votes.get)
        player_to_team[player] = best_team

    logger.info("Mapped %d players to teams", len(player_to_team))
    return player_to_team


def build_player_team_mapping_by_season(games: List[dict]) -> Dict[str, Dict[str, str]]:
    """
    Build per-season player-team mappings to handle mid-season trades.

    Returns: {season_str: {player_name: team_name}}
    where season_str is derived from the game date year boundaries.
    """
    from collections import defaultdict

    # Group games by season
    season_games = defaultdict(list)
    for game in games:
        date_str = game.get("date", "")
        if not date_str:
            continue
        year = int(date_str[:4])
        month = int(date_str[5:7])
        # NBA season spans Oct-June: Oct+ = start year, Jan-June = start year - 1
        season_year = year if month >= 10 else year - 1
        season_games[season_year].append(game)

    mappings = {}
    for season_year, sg in season_games.items():
        season_str = f"{season_year}-{str(season_year + 1)[-2:]}"
        mappings[season_str] = build_player_team_mapping(sg)
        logger.info("Season %s: %d players mapped", season_str, len(mappings[season_str]))

    return mappings


# -------------------------------------------------------------------------
# Core Cascade Analysis
# -------------------------------------------------------------------------

def get_season_key(date_str: str) -> int:
    """Return the season start year for a game date."""
    year = int(date_str[:4])
    month = int(date_str[5:7])
    return year if month >= 10 else year - 1


def build_team_game_logs(
    games: List[dict],
    player_to_team: Dict[str, str],
) -> Dict[str, Dict[str, List[dict]]]:
    """
    Organize box scores by team and player.

    Returns: {team_name: {player_name: [{"date": ..., "pts": ..., ...}, ...]}}
    Each entry is one game's stats for that player.
    """
    # Deduplicate: since all games on a date share the same box_score,
    # we only process each (date, player) pair once.
    seen = set()
    team_logs = defaultdict(lambda: defaultdict(list))

    for game in games:
        date = game["date"]
        home_team = normalize_team_name(game["home_team"])
        away_team = normalize_team_name(game["away_team"])
        box = game.get("box_score", {})

        for player_name, stat_line in box.items():
            # Skip if we already processed this player on this date
            key = (date, player_name)
            if key in seen:
                continue
            seen.add(key)

            team = player_to_team.get(player_name)
            if not team:
                continue

            # Only include if this is actually the player's team's game
            if team != home_team and team != away_team:
                continue

            minutes = stat_line.get("min", 0) or 0
            if minutes <= 0:
                continue  # DNP / inactive

            entry = {"date": date}
            for sk in STAT_KEYS:
                entry[sk] = stat_line.get(sk, 0) or 0
            team_logs[team][player_name].append(entry)

    # Sort each player's logs by date
    for team in team_logs:
        for player in team_logs[team]:
            team_logs[team][player].sort(key=lambda x: x["date"])

    return dict(team_logs)


def compute_cascade_impacts(
    team_logs: Dict[str, Dict[str, List[dict]]],
    team_filter: Optional[str] = None,
) -> dict:
    """
    For every team, compute how each player's absence affects teammates.

    Algorithm:
        For each team:
            For each player A on the team (with >= MIN_GAMES_PLAYED games):
                Identify dates where A played
                For each other player B on the team:
                    B's games where A also played  -> "with_A" stats
                    B's games where A did NOT play  -> "without_A" stats
                    Compute boost_pct and statistical significance

    Returns nested dict:
        {team: {absent_player: {beneficiary: {prop: {boost_pct, ...}}}}}
    """
    cascade = {}
    teams_to_process = [team_filter] if team_filter else list(team_logs.keys())

    for team in teams_to_process:
        if team not in team_logs:
            logger.warning("Team %s not found in game logs", team)
            continue

        players = team_logs[team]
        team_cascade = {}
        logger.info("Processing %s (%d players)", team, len(players))

        # Identify "regulars" - players with enough games to be meaningful
        regulars = {
            name: logs
            for name, logs in players.items()
            if len(logs) >= MIN_GAMES_PLAYED
        }

        # All dates for this team (union of all player game dates)
        all_team_dates = set()
        for logs in players.values():
            for entry in logs:
                all_team_dates.add(entry["date"])

        for absent_player, absent_logs in regulars.items():
            absent_dates = {e["date"] for e in absent_logs}
            missed_dates = all_team_dates - absent_dates

            # Need enough "without" games
            if len(missed_dates) < MIN_GAMES_WITHOUT:
                continue
            if len(absent_dates) < MIN_GAMES_WITH:
                continue

            beneficiary_impacts = {}

            for benef_name, benef_logs in players.items():
                if benef_name == absent_player:
                    continue
                if len(benef_logs) < MIN_BENEF_GAMES:
                    continue  # Not enough data for beneficiary to be rotation

                # Split beneficiary's games into with/without absent player
                with_games = [e for e in benef_logs if e["date"] in absent_dates]
                without_games = [e for e in benef_logs if e["date"] in missed_dates]

                if len(with_games) < MIN_GAMES_WITH:
                    continue
                if len(without_games) < MIN_GAMES_WITHOUT:
                    continue

                prop_impacts = {}

                for stat_key in STAT_KEYS:
                    if stat_key == "min":
                        continue  # Minutes tracked separately

                    with_vals = np.array([g[stat_key] for g in with_games], dtype=float)
                    without_vals = np.array([g[stat_key] for g in without_games], dtype=float)

                    avg_with = np.mean(with_vals)
                    avg_without = np.mean(without_vals)

                    # Skip if baseline is too low (filters garbage-time noise)
                    min_base = MIN_BASELINE.get(stat_key, 0.5)
                    if avg_with < min_base:
                        continue

                    boost_pct = (avg_without - avg_with) / avg_with

                    # Statistical significance via Welch's t-test
                    if np.std(with_vals) == 0 and np.std(without_vals) == 0:
                        continue
                    t_stat, p_value = scipy_stats.ttest_ind(
                        without_vals, with_vals, equal_var=False
                    )

                    if p_value > P_VALUE_THRESHOLD:
                        continue  # Not statistically significant

                    # Per-minute rate comparison (controls for minutes variance)
                    with_mins = np.array([g["min"] for g in with_games], dtype=float)
                    without_mins = np.array([g["min"] for g in without_games], dtype=float)
                    avg_min_with = np.mean(with_mins)
                    avg_min_without = np.mean(without_mins)
                    minutes_change = 0.0
                    if avg_min_with > 0:
                        minutes_change = (avg_min_without - avg_min_with) / avg_min_with

                    # Per-minute rate boost (separates volume from efficiency)
                    rate_with = avg_with / max(avg_min_with, 1.0) * 36.0
                    rate_without = avg_without / max(avg_min_without, 1.0) * 36.0
                    rate_boost = 0.0
                    if rate_with > 0:
                        rate_boost = (rate_without - rate_with) / rate_with

                    # Confidence level
                    n_without = len(without_games)
                    if p_value < 0.01 and n_without >= 15:
                        confidence = "high"
                    elif p_value < 0.05 and n_without >= 10:
                        confidence = "medium"
                    else:
                        confidence = "low"

                    prop_name = STAT_TO_PROP.get(stat_key, stat_key)
                    prop_impacts[prop_name] = {
                        "boost_pct": round(boost_pct * 100, 2),
                        "avg_with": round(avg_with, 2),
                        "avg_without": round(avg_without, 2),
                        "minutes_change_pct": round(minutes_change * 100, 2),
                        "per36_rate_boost_pct": round(rate_boost * 100, 2),
                        "sample_with": len(with_games),
                        "sample_without": n_without,
                        "p_value": round(p_value, 4),
                        "t_statistic": round(t_stat, 3),
                        "confidence": confidence,
                    }

                if prop_impacts:
                    beneficiary_impacts[benef_name] = prop_impacts

            if beneficiary_impacts:
                team_cascade[absent_player] = beneficiary_impacts

        if team_cascade:
            cascade[team] = team_cascade
            logger.info(
                "  %s: %d key absences tracked, %d total cascade pairs",
                team,
                len(team_cascade),
                sum(len(v) for v in team_cascade.values()),
            )

    return cascade


# -------------------------------------------------------------------------
# Cascade Table I/O
# -------------------------------------------------------------------------

def save_cascade_tables(cascade: dict, path: str = CASCADE_TABLE_PATH) -> None:
    """Save cascade tables to JSON with metadata."""
    output = {
        "metadata": {
            "built_at": datetime.now().isoformat(),
            "data_files": [os.path.basename(f) for f in DATA_FILES],
            "min_games_with": MIN_GAMES_WITH,
            "min_games_without": MIN_GAMES_WITHOUT,
            "min_games_played": MIN_GAMES_PLAYED,
            "p_value_threshold": P_VALUE_THRESHOLD,
            "total_teams": len(cascade),
            "total_absent_players": sum(len(v) for v in cascade.values()),
            "total_cascade_pairs": sum(
                sum(len(benefs) for benefs in absent.values())
                for absent in cascade.values()
            ),
        },
        "cascade": cascade,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved cascade tables to %s", path)


def load_cascade_tables(path: str = CASCADE_TABLE_PATH) -> dict:
    """Load cascade tables from JSON. Returns the cascade dict."""
    if not os.path.exists(path):
        logger.warning("Cascade tables not found at %s", path)
        return {}
    with open(path) as f:
        data = json.load(f)
    cascade = data.get("cascade", {})
    meta = data.get("metadata", {})
    logger.info(
        "Loaded cascade tables: %d teams, %d absent players, %d pairs (built %s)",
        meta.get("total_teams", 0),
        meta.get("total_absent_players", 0),
        meta.get("total_cascade_pairs", 0),
        meta.get("built_at", "unknown"),
    )
    return cascade


# -------------------------------------------------------------------------
# Lookup API
# -------------------------------------------------------------------------

class InjuryCascadeLookup:
    """
    Fast lookup interface for injury cascade data.

    Usage:
        lookup = InjuryCascadeLookup()
        impacts = lookup.get_impacts("Jaylen Brown")
        # -> [{"player": "Jayson Tatum", "prop": "points", "boost": 8.2, ...}, ...]
    """

    def __init__(self, cascade_path: str = CASCADE_TABLE_PATH):
        self._cascade = load_cascade_tables(cascade_path)
        # Build reverse index: player_name -> (team, is_absent_player)
        self._player_index = {}
        for team, absents in self._cascade.items():
            for absent_player in absents:
                self._player_index[absent_player] = team

    @property
    def is_loaded(self) -> bool:
        return bool(self._cascade)

    def get_team_for_player(self, player_name: str) -> Optional[str]:
        """Look up which team a player belongs to in the cascade data."""
        return self._player_index.get(player_name)

    def get_impacts(
        self,
        absent_player: str,
        min_confidence: str = "low",
    ) -> List[dict]:
        """
        Get all cascade impacts when a specific player sits out.

        Args:
            absent_player: Name of the player who is out
            min_confidence: Minimum confidence level ("low", "medium", "high")

        Returns:
            List of impact dicts sorted by absolute boost magnitude:
            [{"player": ..., "prop": ..., "boost_pct": ..., "confidence": ..., ...}]
        """
        conf_order = {"low": 0, "medium": 1, "high": 2}
        min_conf_val = conf_order.get(min_confidence, 0)

        team = self._player_index.get(absent_player)
        if not team:
            return []

        absents = self._cascade.get(team, {})
        impacts_raw = absents.get(absent_player, {})

        impacts = []
        for benef_name, props in impacts_raw.items():
            for prop_name, data in props.items():
                if conf_order.get(data.get("confidence", "low"), 0) < min_conf_val:
                    continue
                impacts.append({
                    "player": benef_name,
                    "team": team,
                    "prop": prop_name,
                    "boost_pct": data["boost_pct"],
                    "avg_with": data["avg_with"],
                    "avg_without": data["avg_without"],
                    "minutes_change_pct": data.get("minutes_change_pct", 0),
                    "per36_rate_boost_pct": data.get("per36_rate_boost_pct", 0),
                    "sample_with": data["sample_with"],
                    "sample_without": data["sample_without"],
                    "p_value": data["p_value"],
                    "confidence": data["confidence"],
                })

        # Sort by absolute boost magnitude (biggest impacts first)
        impacts.sort(key=lambda x: abs(x["boost_pct"]), reverse=True)
        return impacts

    def get_impacts_for_player(
        self,
        beneficiary: str,
        absent_player: str,
    ) -> Dict[str, dict]:
        """
        Get cascade impacts on a specific beneficiary when a specific player sits.

        Returns: {prop_name: impact_data}
        """
        team = self._player_index.get(absent_player)
        if not team:
            return {}
        return (
            self._cascade
            .get(team, {})
            .get(absent_player, {})
            .get(beneficiary, {})
        )

    def get_all_absences_for_team(self, team_name: str) -> List[str]:
        """Get all players whose absence is tracked for a team."""
        return list(self._cascade.get(team_name, {}).keys())

    def get_top_cascade_targets(
        self,
        absent_player: str,
        prop_type: Optional[str] = None,
        top_n: int = 5,
    ) -> List[dict]:
        """
        Get the top N beneficiaries when a player sits, optionally filtered by prop.
        Sorted by boost magnitude.
        """
        impacts = self.get_impacts(absent_player)
        if prop_type:
            impacts = [i for i in impacts if i["prop"] == prop_type]
        return impacts[:top_n]

    def suggest_bets(
        self,
        absent_player: str,
        min_boost_pct: float = 5.0,
        min_confidence: str = "medium",
    ) -> List[dict]:
        """
        Suggest bets based on cascade impacts.
        Returns OVER recommendations for positive boosts, UNDER for negative.
        """
        impacts = self.get_impacts(absent_player, min_confidence=min_confidence)
        suggestions = []
        for imp in impacts:
            if abs(imp["boost_pct"]) < min_boost_pct:
                continue
            side = "OVER" if imp["boost_pct"] > 0 else "UNDER"
            suggestions.append({
                "player": imp["player"],
                "prop": imp["prop"],
                "side": side,
                "boost_pct": imp["boost_pct"],
                "confidence": imp["confidence"],
                "sample_without": imp["sample_without"],
                "reason": (
                    f"{absent_player} out, "
                    f"{'+' if imp['boost_pct'] > 0 else ''}{imp['boost_pct']:.1f}% "
                    f"historical boost ({imp['sample_without']} game sample)"
                ),
            })
        return suggestions

    def summary(self) -> dict:
        """Return a summary of cascade table coverage."""
        total_pairs = 0
        total_absences = 0
        teams = []
        for team, absents in self._cascade.items():
            n_abs = len(absents)
            n_pairs = sum(len(benefs) for benefs in absents.values())
            total_absences += n_abs
            total_pairs += n_pairs
            teams.append({
                "team": team,
                "abbrev": FULL_TO_ABBREV.get(team, "???"),
                "tracked_absences": n_abs,
                "cascade_pairs": n_pairs,
            })
        return {
            "total_teams": len(teams),
            "total_absences": total_absences,
            "total_cascade_pairs": total_pairs,
            "teams": sorted(teams, key=lambda x: x["cascade_pairs"], reverse=True),
        }


# -------------------------------------------------------------------------
# CLI Entry Point
# -------------------------------------------------------------------------

def build_tables(team_filter: Optional[str] = None) -> dict:
    """Full pipeline: load data, build mappings, compute cascades, save."""
    logger.info("=== Building Injury Cascade Tables ===")

    # Load all game data
    games = load_all_game_data()
    if not games:
        logger.error("No game data found. Check outputs/ directory.")
        return {}

    # Build player-team mapping
    player_to_team = build_player_team_mapping(games)

    # Resolve team filter (abbreviation -> full name)
    resolved_filter = None
    if team_filter:
        if team_filter in TEAM_ABBREVS:
            resolved_filter = TEAM_ABBREVS[team_filter]
        elif team_filter in FULL_TO_ABBREV:
            resolved_filter = team_filter
        else:
            logger.error("Unknown team: %s", team_filter)
            return {}
        logger.info("Filtering to team: %s", resolved_filter)

    # Build per-team game logs
    logger.info("Building team game logs...")
    team_logs = build_team_game_logs(games, player_to_team)
    logger.info("Built logs for %d teams", len(team_logs))

    # Compute cascade impacts
    logger.info("Computing cascade impacts...")
    cascade = compute_cascade_impacts(team_logs, team_filter=resolved_filter)

    # Save
    save_cascade_tables(cascade)

    return cascade


def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Build injury cascade impact tables")
    parser.add_argument(
        "--team",
        type=str,
        default=None,
        help="Team abbreviation to filter (e.g., BOS, LAL). Default: all teams.",
    )
    args = parser.parse_args()

    cascade = build_tables(team_filter=args.team)

    if cascade:
        # Print summary
        lookup = InjuryCascadeLookup()
        summary = lookup.summary()
        print(f"\n=== Cascade Table Summary ===")
        print(f"Teams: {summary['total_teams']}")
        print(f"Tracked absences: {summary['total_absences']}")
        print(f"Cascade pairs: {summary['total_cascade_pairs']}")
        print()
        for t in summary["teams"][:10]:
            print(f"  {t['abbrev']:>3} {t['team']:<30} "
                  f"{t['tracked_absences']:>3} absences, "
                  f"{t['cascade_pairs']:>4} cascade pairs")

        # Show top impacts
        print(f"\n=== Top Cascade Impacts (All Teams) ===")
        all_impacts = []
        for team, absents in cascade.items():
            for absent, benefs in absents.items():
                for benef, props in benefs.items():
                    for prop, data in props.items():
                        if data.get("confidence") in ("medium", "high"):
                            all_impacts.append({
                                "team": FULL_TO_ABBREV.get(team, "???"),
                                "absent": absent,
                                "beneficiary": benef,
                                "prop": prop,
                                "boost": data["boost_pct"],
                                "sample": data["sample_without"],
                                "confidence": data["confidence"],
                            })
        all_impacts.sort(key=lambda x: abs(x["boost"]), reverse=True)
        for imp in all_impacts[:20]:
            sign = "+" if imp["boost"] > 0 else ""
            print(
                f"  When {imp['absent']} OUT ({imp['team']}): "
                f"{imp['beneficiary']} {imp['prop']} {sign}{imp['boost']:.1f}% "
                f"({imp['sample']}g, {imp['confidence']})"
            )
    else:
        print("No cascade data generated.")


if __name__ == "__main__":
    main()
