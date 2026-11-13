#!/usr/bin/env python3
"""
fetch_multibook_odds.py — Fetch ALL bookmaker lines from The Odds API

Enriches existing backtest legs with multi-book data for consensus analysis.
Stores per-book lines so we can compute devigged consensus, dk_vs_consensus,
line_range, and num_books_with_prop.

Credit budget: ~60 credits per game (6 markets × 10 × 1 region)
Full season: ~263K credits. We have ~5M remaining.

Usage:
    # Dry run — see credit cost without fetching
    python fetch_multibook_odds.py --season 2024-25 --dry-run

    # Fetch all books for 2024-25 season
    python fetch_multibook_odds.py --season 2024-25

    # Fetch specific date range
    python fetch_multibook_odds.py --start 2024-10-22 --end 2025-06-22

    # Resume interrupted fetch
    python fetch_multibook_odds.py --season 2024-25 --resume

    # Check what we have
    python fetch_multibook_odds.py --stats
"""

import os
import sys
import json
import time
import argparse
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from clean_backtest import SNAPSHOT_LAG_MINUTES, norm_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "data" / "multibook_odds.db"
OUTPUT_DIR = BASE_DIR / "outputs"

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"

MARKETS = [
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_threes",
    "player_points_rebounds_assists",
    "player_double_double",
]

MARKET_TO_PROP = {
    "player_points": "points",
    "player_rebounds": "rebounds",
    "player_assists": "assists",
    "player_threes": "threes",
    "player_points_rebounds_assists": "pts_rebs_asts",
    "player_double_double": "double_double",
}

SEASONS = {
    "2023-24": ("2023-10-24", "2024-06-17"),
    "2024-25": ("2024-10-22", "2025-06-22"),
    "2025-26": ("2025-10-21", "2026-06-15"),
}

CREDITS_PER_GAME = 10 * len(MARKETS) * 1  # 60
REQUEST_DELAY = 0.35


# =============================================================================
# DATABASE
# =============================================================================

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            date TEXT NOT NULL,
            home_team TEXT,
            away_team TEXT,
            commence_time TEXT,
            fetched_at TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS props (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL,
            bookmaker TEXT NOT NULL,
            market TEXT NOT NULL,
            player_name TEXT NOT NULL,
            line REAL,
            over_odds INTEGER,
            under_odds INTEGER,
            snapshot_time TEXT,
            UNIQUE(event_id, bookmaker, market, player_name)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS fetch_log (
            date TEXT PRIMARY KEY,
            num_events INTEGER,
            num_props INTEGER,
            credits_used INTEGER,
            fetched_at TEXT
        )
    """)

    c.execute("CREATE INDEX IF NOT EXISTS idx_props_event ON props(event_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_props_player ON props(player_name)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_props_market ON props(market)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_props_book ON props(bookmaker)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_events_date ON events(date)")

    conn.commit()
    conn.close()


def get_fetched_dates() -> set:
    conn = sqlite3.connect(str(DB_PATH))
    dates = {r[0] for r in conn.execute("SELECT date FROM fetch_log").fetchall()}
    conn.close()
    return dates


def save_event(conn, event_id, date, home, away, commence_time):
    conn.execute(
        "INSERT OR IGNORE INTO events VALUES (?, ?, ?, ?, ?, ?)",
        (event_id, date, home, away, commence_time, datetime.now().isoformat()),
    )


def save_props_batch(conn, rows: List[tuple]):
    conn.executemany(
        "INSERT OR IGNORE INTO props (event_id, bookmaker, market, player_name, line, over_odds, under_odds, snapshot_time) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )


def log_fetch(conn, date, num_events, num_props, credits_used):
    conn.execute(
        "INSERT OR REPLACE INTO fetch_log VALUES (?, ?, ?, ?, ?)",
        (date, num_events, num_props, credits_used, datetime.now().isoformat()),
    )


# =============================================================================
# API
# =============================================================================

def api_get(url, params, retries=3):
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
            remaining = resp.headers.get("x-requests-remaining", "?")
            if resp.status_code == 429:
                wait = 2 ** attempt
                logger.warning(f"Rate limited. Waiting {wait}s... (remaining: {remaining})")
                time.sleep(wait)
                continue
            if resp.status_code == 422:
                logger.debug(f"No data (422) for params: {params.get('date', '?')}")
                return None, 0, remaining
            resp.raise_for_status()
            credits = int(resp.headers.get("x-requests-last", 0))
            return resp.json(), credits, remaining
        except requests.RequestException as e:
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"Request failed: {e}")
                return None, 0, "?"
    return None, 0, "?"


def fetch_events_for_date(date_str: str) -> Tuple[List[Dict], int]:
    url = f"{ODDS_API_BASE}/historical/sports/{SPORT}/events"
    params = {"apiKey": ODDS_API_KEY, "date": f"{date_str}T18:00:00Z"}
    data, credits, remaining = api_get(url, params)
    if data and isinstance(data, dict):
        return data.get("data", []), credits
    return [], credits


def fetch_event_odds(event_id: str, snapshot_time: str) -> Tuple[Dict, int, str]:
    """Fetch ALL bookmakers for an event. No bookmaker filter."""
    url = f"{ODDS_API_BASE}/historical/sports/{SPORT}/events/{event_id}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us,us2,eu",
        "markets": ",".join(MARKETS),
        "oddsFormat": "american",
        "date": snapshot_time,
    }
    data, credits, remaining = api_get(url, params)
    if data and isinstance(data, dict):
        return data.get("data", data), credits, remaining
    return {}, credits, remaining


def compute_snapshot_time(commence_time: str, minutes_before: int = 10) -> str:
    game_time = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
    snap = game_time - timedelta(minutes=minutes_before)
    minute = (snap.minute // 5) * 5
    snap = snap.replace(minute=minute, second=0, microsecond=0)
    return snap.strftime("%Y-%m-%dT%H:%M:%SZ")


def _implied(odds) -> float:
    return abs(odds) / (abs(odds) + 100.0) if odds < 0 else 100.0 / (odds + 100.0)


def parse_all_books(event_data: Dict, event_id: str, snapshot_time: str) -> List[tuple]:
    rows = []
    for bk in event_data.get("bookmakers", []):
        bookmaker = bk.get("key", "")
        for market in bk.get("markets", []):
            mkt = market.get("key", "")
            # Some books list several lines per player in one market. Pair each over with the
            # under at the same line, then keep the line priced closest to even (the main line).
            # Taking the first line with the last prices stores a price that belongs to another line.
            by_line = {}
            for outcome in market.get("outcomes", []):
                player = outcome.get("description", "")
                direction = outcome.get("name", "")
                if not player or direction not in ("Over", "Under"):
                    continue
                by_line.setdefault((player, float(outcome.get("point", 0))), {})[direction] = outcome.get("price", 0)

            main_line = {}
            for (player, point), prices in by_line.items():
                if "Over" not in prices or "Under" not in prices:
                    continue
                balance = abs(_implied(prices["Over"]) - _implied(prices["Under"]))
                if player not in main_line or balance < main_line[player][0]:
                    main_line[player] = (balance, point, prices["Over"], prices["Under"])

            for player, (_, point, over_o, under_o) in main_line.items():
                rows.append((event_id, bookmaker, mkt, player, point, over_o, under_o, snapshot_time))
    return rows


# =============================================================================
# MAIN FETCH
# =============================================================================

def date_range(start: str, end: str) -> List[str]:
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    dates = []
    cur = s
    while cur <= e:
        dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return dates


def fetch_multibook(start: str, end: str, max_credits: int = 1_000_000,
                    resume: bool = False, dry_run: bool = False):
    init_db()
    done_dates = get_fetched_dates() if resume else set()

    dates = date_range(start, end)
    dates_todo = [d for d in dates if d not in done_dates]

    total_credits = 0
    total_events = 0
    total_props = 0

    logger.info("=" * 60)
    logger.info(f"MULTI-BOOK FETCH {'(DRY RUN)' if dry_run else ''}")
    logger.info(f"Range: {start} to {end} ({len(dates)} days)")
    logger.info(f"Already done: {len(dates) - len(dates_todo)} | Remaining: {len(dates_todo)}")
    logger.info(f"Credit budget: {max_credits:,} | Per game: {CREDITS_PER_GAME}")
    logger.info("=" * 60)

    if dry_run:
        est_games = len(dates_todo) * 8
        est_credits = len(dates_todo) * 1 + est_games * CREDITS_PER_GAME
        logger.info(f"Estimated: {est_games} games, {est_credits:,} credits")
        return

    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")

    for i, date_str in enumerate(dates_todo):
        if total_credits + CREDITS_PER_GAME > max_credits:
            logger.warning(f"Credit limit approaching ({total_credits:,}/{max_credits:,}). Stopping.")
            break

        events, ev_credits = fetch_events_for_date(date_str)
        total_credits += ev_credits
        time.sleep(REQUEST_DELAY)

        if not events:
            log_fetch(conn, date_str, 0, 0, ev_credits)
            conn.commit()
            continue

        day_props = 0
        for ev in events:
            if total_credits + CREDITS_PER_GAME > max_credits:
                break

            eid = ev.get("id", "")
            home = ev.get("home_team", "")
            away = ev.get("away_team", "")
            commence = ev.get("commence_time", "")
            if not eid or not commence:
                continue

            snap_time = compute_snapshot_time(commence, minutes_before=10)
            save_event(conn, eid, date_str, home, away, commence)

            odds_data, od_credits, remaining = fetch_event_odds(eid, snap_time)
            total_credits += od_credits
            time.sleep(REQUEST_DELAY)

            if odds_data:
                rows = parse_all_books(odds_data, eid, snap_time)
                if rows:
                    save_props_batch(conn, rows)
                    day_props += len(rows)

            total_events += 1

        log_fetch(conn, date_str, len(events), day_props, total_credits)
        total_props += day_props
        conn.commit()

        if (i + 1) % 7 == 0 or i == len(dates_todo) - 1:
            logger.info(
                f"[{date_str}] Week checkpoint: {total_events} games, "
                f"{total_props:,} prop rows, {total_credits:,} credits used, "
                f"remaining: {remaining}"
            )

    conn.close()

    logger.info("=" * 60)
    logger.info("FETCH COMPLETE")
    logger.info(f"  Events: {total_events}")
    logger.info(f"  Prop rows: {total_props:,}")
    logger.info(f"  Credits used: {total_credits:,}")
    logger.info("=" * 60)


def show_stats():
    init_db()
    conn = sqlite3.connect(str(DB_PATH))

    dates = conn.execute("SELECT COUNT(DISTINCT date) FROM fetch_log").fetchone()[0]
    events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    props = conn.execute("SELECT COUNT(*) FROM props").fetchone()[0]
    books = conn.execute("SELECT COUNT(DISTINCT bookmaker) FROM props").fetchone()[0]
    players = conn.execute("SELECT COUNT(DISTINCT player_name) FROM props").fetchone()[0]

    date_range_r = conn.execute("SELECT MIN(date), MAX(date) FROM fetch_log").fetchone()
    top_books = conn.execute(
        "SELECT bookmaker, COUNT(*) FROM props GROUP BY bookmaker ORDER BY 2 DESC LIMIT 10"
    ).fetchall()
    credits = conn.execute("SELECT SUM(credits_used) FROM fetch_log").fetchone()[0] or 0

    conn.close()

    print("\n" + "=" * 50)
    print("MULTI-BOOK ODDS DATABASE")
    print("=" * 50)
    print(f"  Dates fetched: {dates}")
    print(f"  Date range: {date_range_r[0]} to {date_range_r[1]}" if date_range_r[0] else "  No data yet")
    print(f"  Events: {events}")
    print(f"  Prop rows: {props:,}")
    print(f"  Unique books: {books}")
    print(f"  Unique players: {players}")
    print(f"  Credits used: {credits:,}")
    print()
    print("  Top bookmakers:")
    for bk, cnt in top_books:
        print(f"    {bk}: {cnt:,}")
    print("=" * 50)


def _closing_quotes(conn) -> dict:
    """(ET game date, normalized player, market) -> {book: quote} from each event's closing snapshot."""
    et = ZoneInfo("America/New_York")
    quotes, in_window, game_date, names = {}, {}, {}, {}
    rows = conn.execute("""
        SELECT e.commence_time, p.snapshot_time, p.bookmaker, p.market, p.player_name,
               p.line, p.over_odds, p.under_odds
        FROM props p
        JOIN events e ON p.event_id = e.event_id
    """)
    for commence, snap, book, market, player, line_val, over_o, under_o in rows:
        timing = (commence, snap)
        if timing not in in_window:
            tip = datetime.fromisoformat(commence.replace("Z", "+00:00"))
            lag = (datetime.fromisoformat(snap.replace("Z", "+00:00")) - tip).total_seconds() / 60.0
            in_window[timing] = SNAPSHOT_LAG_MINUTES[0] <= lag <= SNAPSHOT_LAG_MINUTES[1]
            # events.date is the day the event was first listed, often days before it was played.
            game_date[commence] = tip.astimezone(et).strftime("%Y-%m-%d")
        if not in_window[timing]:
            continue
        if player not in names:
            names[player] = norm_name(player)
        quotes.setdefault((game_date[commence], names[player], market), {}).setdefault(
            book, {"line": line_val, "over_odds": over_o, "under_odds": under_o, "snapshot_time": snap})
    return quotes


def export_enriched_legs(season: str):
    """Join multi-book data with existing legs to create enriched dataset."""
    init_db()

    legs_file = OUTPUT_DIR / f"backtest_{season.replace('-', '_')}_legs.json"
    if not legs_file.exists():
        logger.error(f"Legs file not found: {legs_file}")
        return

    with open(legs_file) as f:
        legs = json.load(f)

    conn = sqlite3.connect(str(DB_PATH))
    quotes = _closing_quotes(conn)

    enriched = []
    matched = 0
    for leg in legs:
        date = leg.get("date", "")
        player = leg.get("player", "")
        prop_type = leg.get("prop_type", "")

        market_key = next((k for k, v in MARKET_TO_PROP.items() if v == prop_type), None)
        if not market_key:
            enriched.append(leg)
            continue

        by_book = quotes.get((date, norm_name(player), market_key), {})
        dk = by_book.get("draftkings")

        if dk:
            matched += 1
            # The other books were captured minutes before tip; the leg's own DraftKings price was
            # captured hours earlier. Reprice the leg from the same snapshot, or the consensus
            # would already know where that price ended up.
            books = {bk: {"line": q["line"], "over_odds": q["over_odds"], "under_odds": q["under_odds"]}
                     for bk, q in by_book.items() if q["snapshot_time"] == dk["snapshot_time"]}
            leg["line"], leg["over_odds"], leg["under_odds"] = dk["line"], dk["over_odds"], dk["under_odds"]
            if leg.get("actual") is not None:
                leg["hit"] = leg["actual"] > leg["line"]

            leg["all_books"] = books
            leg["num_books"] = len(books)

            lines = [b["line"] for b in books.values()]
            leg["line_range"] = max(lines) - min(lines) if len(lines) > 1 else 0
            leg["consensus_line"] = sorted(lines)[len(lines) // 2]
            leg["dk_vs_consensus"] = dk["line"] - leg["consensus_line"]
        else:
            leg["all_books"] = {}
            leg["num_books"] = 0
            leg["line_range"] = 0
            leg["consensus_line"] = leg.get("line", 0)
            leg["dk_vs_consensus"] = 0

        enriched.append(leg)

    conn.close()

    out_file = OUTPUT_DIR / f"backtest_{season.replace('-', '_')}_multibook.json"
    with open(out_file, "w") as f:
        json.dump(enriched, f)

    logger.info(f"Enriched {matched}/{len(legs)} legs ({100*matched/max(1,len(legs)):.1f}%)")
    logger.info(f"Saved: {out_file}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fetch multi-book NBA player props")
    parser.add_argument("--season", choices=list(SEASONS.keys()), help="Season to fetch")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument("--max-credits", type=int, default=1_000_000)
    parser.add_argument("--resume", action="store_true", help="Skip already-fetched dates")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--export", help="Export enriched legs for season (e.g. 2024-25)")
    args = parser.parse_args()

    if args.stats:
        show_stats()
        return

    if args.export:
        export_enriched_legs(args.export)
        return

    if args.season:
        start, end = SEASONS[args.season]
    elif args.start and args.end:
        start, end = args.start, args.end
    else:
        parser.error("Provide --season or --start/--end")
        return

    fetch_multibook(start, end, max_credits=args.max_credits,
                    resume=args.resume, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
