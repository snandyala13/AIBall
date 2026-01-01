#!/usr/bin/env python3
"""
ball.py - NBA Betting System Main Entry Point

Usage:
    python ball.py                         # Demo mode (no API key needed)
    python ball.py --game "BOS vs MIL"     # Analyze specific game
    python ball.py --player "Jaylen Brown" # Analyze single player
    python ball.py --today                 # Analyze all today's games
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import List

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_demo():
    """Run demonstration with math examples"""
    print("\n" + "=" * 70)
    print("    NBA BETTING SYSTEM - DEMONSTRATION")
    print("=" * 70)

    from bet_generator import KellyCriterion, SGPJuiceCalculator

    kelly = KellyCriterion()
    sgp = SGPJuiceCalculator()

    # Edge Calculation Demo
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                     EDGE CALCULATION EXPLAINED                        ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  Our model calculates edge using Z-scores and normal distribution:   ║
║                                                                       ║
║  1. Build player projection from:                                    ║
║     - Last 5 games (45% weight)                                      ║
║     - Last 10 games (35% weight)                                     ║
║     - Season average (20% weight)                                    ║
║                                                                       ║
║  2. Apply adjustments:                                               ║
║     - Per-minute rates × projected minutes                           ║
║     - Home/away splits                                               ║
║     - Pace (team vs opponent)                                        ║
║     - Defense matchup                                                ║
║     - Fatigue (back-to-back)                                        ║
║     - Blowout risk (minutes reduction)                               ║
║                                                                       ║
║  3. Calculate probability:                                           ║
║     Z = (projection - line) / std_dev                                ║
║     P(over) = 1 - CDF(Z)                                            ║
║                                                                       ║
║  4. Find edge:                                                       ║
║     Edge = Our probability - DraftKings implied probability          ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Example calculation
    print("\n--- EXAMPLE: Points Over Bet ---\n")

    projection = 26.8
    line = 24.5
    std_dev = 5.2
    dk_odds = -115

    from model_engine import EdgeCalculator
    calc = EdgeCalculator()

    z_score = (line - projection) / std_dev
    over_prob = calc.calculate_over_probability(projection, line, std_dev)
    implied = calc.american_to_prob(dk_odds)
    edge = (over_prob - implied) * 100

    print(f"  Projection:    {projection:.1f} points")
    print(f"  DK Line:       {line}")
    print(f"  Std Dev:       {std_dev:.1f}")
    print(f"  Z-Score:       {z_score:.3f}")
    print(f"")
    print(f"  Our Prob:      {over_prob*100:.1f}%")
    print(f"  DK Implied:    {implied*100:.1f}% (at {dk_odds})")
    print(f"  Edge:          {edge:+.1f}%")

    if edge > 3:
        units = kelly.calculate_units(over_prob, dk_odds)
        ev = kelly.calculate_ev(over_prob, dk_odds)
        print(f"\n  ✓ VALUE FOUND!")
        print(f"    Kelly units: {units}")
        print(f"    EV per unit: ${ev:.3f}")

    # Correlation Demo
    print("\n" + "=" * 70)
    print("    CORRELATION IN SGPs")
    print("=" * 70)

    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                      WHY CORRELATION MATTERS                          ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  Independent events: P(A ∩ B) = P(A) × P(B)                          ║
║                                                                       ║
║  Correlated events:  P(A ∩ B) = P(A) × P(B) + ρ × σ_A × σ_B         ║
║                                                                       ║
║  Examples of POSITIVE correlation:                                   ║
║    - PG assists + C points (the PG feeds the C)                      ║
║    - Same player points + rebounds (more minutes = both go up)       ║
║    - High pace game = everyone's stats go up                         ║
║                                                                       ║
║  DraftKings KNOWS this. They juice SGPs to account for it.          ║
║  Our advantage: We calculate the TRUE joint probability.             ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    print("\n--- EXAMPLE: Correlated SGP ---\n")

    prob_a = 0.54  # PG assists over
    prob_b = 0.52  # C points over
    correlation = 0.42

    naive = sgp.calculate_naive_probability([prob_a, prob_b])
    true = sgp.calculate_true_parlay_probability([prob_a, prob_b], [correlation])

    naive_odds = sgp.probability_to_american(naive)
    true_odds = sgp.probability_to_american(true)

    print(f"  Leg A (PG assists over): {prob_a*100:.0f}%")
    print(f"  Leg B (C points over):   {prob_b*100:.0f}%")
    print(f"  Correlation:             {correlation}")
    print(f"")
    print(f"  Naive prob (independent): {naive*100:.1f}% → fair odds {naive_odds:+d}")
    print(f"  True prob (correlated):   {true*100:.1f}% → fair odds {true_odds:+d}")
    print(f"")
    print(f"  Correlation boost: +{(true-naive)*100:.1f} percentage points")
    print(f"  This is {((true/naive)-1)*100:.0f}% more likely than naive calculation!")

    # Kelly Demo
    print("\n" + "=" * 70)
    print("    FRACTIONAL KELLY CRITERION")
    print("=" * 70)

    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                        POSITION SIZING                               ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  Full Kelly: f* = (bp - q) / b                                       ║
║    where b = decimal odds - 1, p = win prob, q = 1-p                ║
║                                                                       ║
║  We use 0.25 Kelly (Quarter Kelly) because:                          ║
║    - Full Kelly is too aggressive                                    ║
║    - Reduces variance significantly                                  ║
║    - Industry standard for sports betting                            ║
║                                                                       ║
║  Position limits:                                                    ║
║    - Single bets: max 5% of bankroll                                ║
║    - Parlays: max 2% of bankroll                                    ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    print("\n--- EXAMPLE: Kelly Sizing ---\n")

    win_prob = 0.56
    odds = -110

    full_kelly = kelly.calculate_kelly_fraction(win_prob, odds) / kelly.KELLY_FRACTION
    quarter_kelly = kelly.calculate_kelly_fraction(win_prob, odds)
    units = kelly.calculate_units(win_prob, odds)

    print(f"  Win probability: {win_prob*100:.0f}%")
    print(f"  Odds:            {odds}")
    print(f"")
    print(f"  Full Kelly:      {full_kelly*100:.2f}% of bankroll")
    print(f"  Quarter Kelly:   {quarter_kelly*100:.2f}% of bankroll")
    print(f"  Recommended:     {units} units")

    # How to use
    print("\n" + "=" * 70)
    print("    HOW TO USE")
    print("=" * 70)

    print("""
    1. Set your API key:
       export BALLDONTLIE_API_KEY="your_key"

    2. Analyze a specific game:
       python ball.py --game "Boston Celtics vs Milwaukee Bucks"

    3. Analyze a single player:
       python ball.py --player "Jaylen Brown"

    4. Analyze all today's games:
       python ball.py --today
    """)


def run_player_analysis(player_name: str):
    """Analyze a single player"""
    api_key = os.getenv("BALLDONTLIE_API_KEY")
    if not api_key:
        print("Error: BALLDONTLIE_API_KEY not set")
        print("Set it with: export BALLDONTLIE_API_KEY='your_key'")
        return

    from data_pipeline import DataPipeline

    print(f"\n=== PLAYER ANALYSIS: {player_name} ===\n")

    pipeline = DataPipeline(api_key)
    profile = pipeline.build_player_profile(player_name)

    if not profile:
        print(f"Player not found: {player_name}")
        return

    print(f"Name:     {profile.name}")
    print(f"Team:     {profile.team}")
    print(f"Position: {profile.position}")
    print(f"Status:   {profile.status}")
    if profile.injury_note:
        print(f"Injury:   {profile.injury_note}")
    print()

    print("=== Season Averages ===")
    for stat, val in profile.season_avg.items():
        if stat != "games_played":
            print(f"  {stat}: {val}")
    print()

    print("=== Last 5 Games ===")
    for stat, val in profile.last_5_avg.items():
        print(f"  {stat}: {val}")
    print()

    print("=== Last 10 Games ===")
    for stat, val in profile.last_10_avg.items():
        print(f"  {stat}: {val}")
    print()

    print("=== Standard Deviation ===")
    for stat, val in profile.std_dev.items():
        print(f"  {stat}: {val}")
    print()

    print("=== Per-Minute Rates ===")
    for stat, val in profile.per_minute.items():
        print(f"  {stat}: {val:.3f}")
    print()

    if profile.usage_pct:
        print(f"Usage Rate: {profile.usage_pct:.1f}%")

    print("\n=== Home/Away Splits ===")
    print("Home:")
    for stat, val in profile.home_avg.items():
        print(f"  {stat}: {val}")
    print("Away:")
    for stat, val in profile.away_avg.items():
        print(f"  {stat}: {val}")


def run_game_analysis(game_desc: str, player_names: List[str] = None):
    """Analyze a specific game"""
    api_key = os.getenv("BALLDONTLIE_API_KEY")
    if not api_key:
        print("Error: BALLDONTLIE_API_KEY not set")
        print("Set it with: export BALLDONTLIE_API_KEY='your_key'")
        return

    from data_pipeline import DataPipeline
    from model_engine import ModelEngine, PropType
    from bet_generator import BetGenerator

    # Parse game description (e.g., "Boston Celtics vs Milwaukee Bucks")
    parts = game_desc.replace(" @ ", " vs ").split(" vs ")
    if len(parts) != 2:
        print("Error: Game should be in format 'Team A vs Team B'")
        return

    away_team = parts[0].strip()
    home_team = parts[1].strip()

    print(f"\n{'='*70}")
    print(f"  GAME ANALYSIS: {away_team} @ {home_team}")
    print(f"{'='*70}\n")

    # Initialize
    pipeline = DataPipeline(api_key)
    engine = ModelEngine(pipeline)
    generator = BetGenerator(engine)

    # Find the game ID first (needed for active roster)
    games = pipeline.client.get_todays_games()
    game_id = None
    for game in games:
        h_team = game.get("home_team", {}).get("full_name", "")
        a_team = game.get("visitor_team", {}).get("full_name", "")
        if (h_team == home_team and a_team == away_team) or \
           (h_team == away_team and a_team == home_team):
            game_id = game.get("id")
            break

    if not game_id:
        print(f"Game not found for today. Check team names.")
        print(f"Tried: '{away_team}' @ '{home_team}'")
        return

    # Build game context
    context = pipeline.build_game_context(home_team, away_team)
    if not context:
        print(f"Could not build game context.")
        return

    # Get injuries for context - collect injured player names for pop-off analysis
    home_id = pipeline.client.TEAM_IDS.get(home_team)
    away_id = pipeline.client.TEAM_IDS.get(away_team)
    injuries = []
    injured_names = []  # List of injured player names for pop-off boosts

    if home_id:
        for inj in pipeline.client.get_injuries(home_id):
            player = inj.get('player', {})
            name = f"{player.get('first_name')} {player.get('last_name')}"
            status = inj.get('status', '').lower()
            if status == 'out':
                injured_names.append(name)
                injuries.append(f"{name} (OUT)")

    if away_id:
        for inj in pipeline.client.get_injuries(away_id):
            player = inj.get('player', {})
            name = f"{player.get('first_name')} {player.get('last_name')}"
            status = inj.get('status', '').lower()
            if status == 'out':
                injured_names.append(name)
                injuries.append(f"{name} (OUT)")

    if injuries:
        print(f"🚑 INJURIES: {', '.join(injuries)}")
        print(f"   Pop-off candidates will get usage boosts\n")

    # If no players specified, get ACTIVE players from DraftKings props
    # This is the BEST source - if DK has props on you, you're playing tonight
    if not player_names:
        print("Detecting active players from DraftKings props...")
        active_players = pipeline.get_active_players_for_game(game_id)

        if active_players:
            player_names = [p["player_name"] for p in active_players]
            print(f"Found {len(player_names)} players with DK props\n")
        else:
            print("No DraftKings props available yet. Try specifying players manually.")
            return

    if not player_names:
        print("No players found.")
        return

    print(f"Analyzing {len(player_names)} players...")
    print(f"Players: {', '.join(player_names[:10])}{'...' if len(player_names) > 10 else ''}\n")

    # Run analysis - pass injured players for pop-off boosts
    analysis = engine.analyze_game(context, player_names, injured_player_names=injured_names)

    # Generate recommendations
    recs = generator.generate_recommendations(analysis)

    # Output
    print(generator.format_recommendations(recs))

    # Save to file
    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"outputs/game_{timestamp}.json"
    with open(filename, "w") as f:
        f.write(generator.export_to_json(recs))
    print(f"\nSaved to: {filename}")


def run_today_analysis():
    """Analyze all games today"""
    api_key = os.getenv("BALLDONTLIE_API_KEY")
    if not api_key:
        print("Error: BALLDONTLIE_API_KEY not set")
        return

    from data_pipeline import DataPipeline
    from model_engine import ModelEngine
    from bet_generator import BetGenerator

    pipeline = DataPipeline(api_key)
    engine = ModelEngine(pipeline)
    generator = BetGenerator(engine)

    print("\n" + "=" * 70)
    print("  TODAY'S GAMES")
    print("=" * 70 + "\n")

    games = pipeline.client.get_todays_games()

    if not games:
        print("No games found for today.")
        return

    # Get DK odds for all games
    dk_odds = pipeline.client.get_draftkings_game_odds()

    print(f"Found {len(games)} games:\n")

    for game in games:
        home = game.get("home_team", {}).get("full_name", "Home")
        away = game.get("visitor_team", {}).get("full_name", "Away")
        game_id = game.get("id")
        odds = dk_odds.get(game_id, {})
        spread = odds.get("spread", 0)
        total = odds.get("total", 0)

        spread_str = f"{spread:+.1f}" if spread else "N/A"
        total_str = f"{total:.1f}" if total else "N/A"

        print(f"  {away} @ {home}")
        print(f"    Spread: {spread_str} | Total: {total_str}")

    print("\n" + "-" * 70)
    print("  Analyzing each game...")
    print("-" * 70)

    all_recommendations = []

    for game in games:
        home = game.get("home_team", {}).get("full_name")
        away = game.get("visitor_team", {}).get("full_name")
        game_id = game.get("id")

        if not home or not away or not game_id:
            continue

        print(f"\n{'='*70}")
        print(f"  {away} @ {home}")
        print(f"{'='*70}")

        # Get injuries
        home_id = game.get("home_team", {}).get("id")
        away_id = game.get("visitor_team", {}).get("id")
        injuries = []

        if home_id:
            for inj in pipeline.client.get_injuries(home_id):
                player = inj.get('player', {})
                name = f"{player.get('first_name')} {player.get('last_name')}"
                status = inj.get('status', '').lower()
                if status == 'out':
                    injuries.append(f"{name} (OUT)")

        if away_id:
            for inj in pipeline.client.get_injuries(away_id):
                player = inj.get('player', {})
                name = f"{player.get('first_name')} {player.get('last_name')}"
                status = inj.get('status', '').lower()
                if status == 'out':
                    injuries.append(f"{name} (OUT)")

        if injuries:
            print(f"🚑 INJURIES: {', '.join(injuries)}\n")

        # Get ACTIVE players from DraftKings props
        active_players = pipeline.get_active_players_for_game(game_id)

        if not active_players:
            print("  No DraftKings props available yet")
            continue

        player_names = [p["player_name"] for p in active_players]
        print(f"  Active players: {len(player_names)}")

        # Build context and analyze
        context = pipeline.build_game_context(home, away, player_names)
        if not context:
            print("  Could not build game context")
            continue

        analysis = engine.analyze_game(context, player_names)
        recs = generator.generate_recommendations(analysis)
        print(generator.format_recommendations(recs))

        all_recommendations.append({
            "game": f"{away} @ {home}",
            "recs": recs
        })

    # Summary
    print("\n" + "=" * 70)
    print("  DAILY SUMMARY")
    print("=" * 70)

    total_picks = sum(
        len(r["recs"].tier_1_singles) + len(r["recs"].tier_2_popoffs) + len(r["recs"].tier_3_parlays)
        for r in all_recommendations
    )
    total_units = sum(r["recs"].total_recommended_units for r in all_recommendations)

    print(f"\n  Total picks across all games: {total_picks}")
    print(f"  Total units recommended: {total_units:.1f}")

    # Save all recommendations to files
    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    date_str = datetime.now().strftime("%Y-%m-%d")

    # Save human-readable text file
    txt_filename = f"outputs/today_{timestamp}.txt"
    with open(txt_filename, "w") as f:
        f.write(f"NBA BETTING PICKS - {date_str}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total Picks: {total_picks}\n")
        f.write(f"Total Units: {total_units:.1f}\n\n")

        for rec in all_recommendations:
            f.write(generator.format_recommendations(rec["recs"]))
            f.write("\n")

    print(f"\n  Saved to: {txt_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="NBA Betting System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ball.py                              # Demo mode
  python ball.py --player "Jaylen Brown"      # Analyze player
  python ball.py --game "Celtics vs Bucks"    # Analyze game
  python ball.py --today                      # All today's games
        """
    )
    parser.add_argument(
        "--player", "-p",
        type=str,
        help="Analyze a single player"
    )
    parser.add_argument(
        "--game", "-g",
        type=str,
        help="Analyze a game (format: 'Team A vs Team B')"
    )
    parser.add_argument(
        "--players",
        type=str,
        help="Comma-separated list of players to analyze"
    )
    parser.add_argument(
        "--today", "-t",
        action="store_true",
        help="Analyze all games today"
    )
    parser.add_argument(
        "--demo", "-d",
        action="store_true",
        help="Run demonstration mode"
    )

    args = parser.parse_args()

    if args.player:
        run_player_analysis(args.player)
    elif args.game:
        players = args.players.split(",") if args.players else None
        run_game_analysis(args.game, players)
    elif args.today:
        run_today_analysis()
    else:
        run_demo()


if __name__ == "__main__":
    main()
