#!/usr/bin/env python3
"""
fetch_early_odds.py — pre-game multi-book prop snapshots from The Odds API's historical endpoint.

One snapshot is the prices every named book showed at a chosen UTC hour on the Eastern game date
(18 = 18:00 UTC, about 1-2 p.m. Eastern). The event list comes from the closing-odds database, so every
row joins to its close by event_id. Raw responses are kept so parsing can be redone without paying again.

Each snapshot hour has its own database and raw folder:
    hour 18      data/early_odds.db        data/early_raw/
    other hours  data/early_odds_hHH.db    data/early_raw_hHH/

Usage:
    python fetch_early_odds.py credits                                   # balance (free call)
    python fetch_early_odds.py fetch --start 2024-11-06 --end 2024-11-06 --limit 1
    python fetch_early_odds.py fetch --start 2023-10-01 --end 2026-06-30 --hour 20 --max-credits 199000
    python fetch_early_odds.py status
"""

import argparse
import gzip
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

from fetch_multibook_odds import parse_all_books  # noqa: E402  (reads the key from the environment loaded above)

CLOSE_DB = BASE_DIR / "data" / "multibook_odds.db"
DATA_DIR = BASE_DIR / "data"

API = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"
MARKETS = ["player_points", "player_rebounds", "player_assists", "player_threes", "player_points_rebounds_assists"]
# Other pull sets are kept in their own databases and raw folders (early_odds_<set>_hHH.db, early_raw_<set>_hHH/).
ALT_MARKETS = ["player_points_alternate", "player_rebounds_alternate", "player_assists_alternate", "player_threes_alternate"]
MARKET_SETS = {
    "main": MARKETS,
    "combos": ["player_points_rebounds", "player_points_assists", "player_rebounds_assists"],
    # Thinly quoted markets: a few books, a handful of players. Double-double is yes/no, so only the raw file keeps it.
    "small": ["player_steals", "player_blocks", "player_blocks_steals", "player_turnovers", "player_double_double", "player_points_q1"],
    # Exchanges and pick'em apps: the main markets plus the alternates, where Kalshi's contracts and the apps' ladders sit.
    "venues": MARKETS + ALT_MARKETS,
}
VENUES = ["kalshi", "novig", "prophetx", "polymarket", "polymarket_us", "prizepicks", "underdog", "pick6", "parlayplay", "dabble_us_dfs"]
SET_BOOKS = {"venues": VENUES}   # every other set asks for BOOKS
# Up to ten named books are billed as one region. The first seven are the shopping set plus Pinnacle;
# the last three fill the group.
BOOKS = ["draftkings", "fanduel", "betmgm", "espnbet", "betparx", "betrivers", "pinnacle",
         "williamhill_us", "hardrockbet", "fanatics"]
REGIONS = "us,us2,eu"
ET = ZoneInfo("America/New_York")
DEFAULT_HOUR_UTC = 18
# The early build drops games tipping within 30 minutes of the snapshot, so they are not bought.
MIN_LEAD_MINUTES = 30
DELAY = 0.3


CLOSE = -1  # stands for the closing snapshot: ten minutes before tip, floored to five minutes
EXTRA_EVENTS_FILE = DATA_DIR / "extra_events.json"


def _tag(hour: int) -> str:
    return "close" if hour == CLOSE else f"h{hour:02d}"


def db_path(hour: int, mset: str = "main") -> Path:
    if mset != "main":
        return DATA_DIR / f"early_odds_{mset}_{_tag(hour)}.db"
    return DATA_DIR / ("early_odds.db" if hour == DEFAULT_HOUR_UTC else f"early_odds_{_tag(hour)}.db")


def raw_dir(hour: int, mset: str = "main") -> Path:
    if mset != "main":
        return DATA_DIR / f"early_raw_{mset}_{_tag(hour)}"
    return DATA_DIR / ("early_raw" if hour == DEFAULT_HOUR_UTC else f"early_raw_{_tag(hour)}")


def requested_time(et_date: str, hour: int, tip: Optional[datetime] = None) -> datetime:
    """The UTC instant asked for: `hour` hours after 00:00 UTC on the Eastern game date (24+ runs into the next UTC day),
    or, for the close, ten minutes before tip floored to five minutes."""
    if hour == CLOSE:
        t = tip - timedelta(minutes=10)
        return t.replace(minute=(t.minute // 5) * 5, second=0, microsecond=0)
    return datetime.fromisoformat(f"{et_date}T00:00:00+00:00") + timedelta(hours=hour)


def _key() -> str:
    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        sys.exit("ODDS_API_KEY is not set (expected in .env)")
    return key


THROTTLED = 0


def get(url: str, params: dict):
    """One GET. Returns (http status, parsed body or None, credits charged, credits remaining).
    Nothing here prints the URL or the exception text: both carry the key."""
    global THROTTLED
    for attempt in range(1, 7):
        try:
            r = requests.get(url, params=params, timeout=40)
        except requests.RequestException as ex:
            print(f"   request failed ({type(ex).__name__}), retry {attempt}", flush=True)
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 429:
            THROTTLED += 1
            time.sleep(2 ** attempt)
            continue
        try:
            body = r.json()
        except ValueError:
            body = None
        charged = int(r.headers.get("x-requests-last") or 0)
        return r.status_code, body, charged, r.headers.get("x-requests-remaining")
    return 0, None, 0, None


def init_db(hour: int = DEFAULT_HOUR_UTC, mset: str = "main") -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path(hour, mset)), timeout=60)
    conn.execute("""create table if not exists events (
        event_id text primary key, et_date text, home_team text, away_team text, commence_time text,
        requested_time text, snapshot_time text, mode text, status text, http_status integer,
        credits integer, n_rows integer, fetched_at text)""")
    conn.execute("""create table if not exists props (
        event_id text not null, bookmaker text not null, market text not null, player_name text not null,
        line real, over_odds integer, under_odds integer, snapshot_time text, last_update text,
        unique(event_id, bookmaker, market, player_name))""")
    conn.execute("create index if not exists idx_early_props_event on props(event_id)")
    conn.commit()
    return conn


def events_between(start: str, end: str) -> list:
    """Events that have closing props, keyed by their Eastern game date (the database's own date column is unreliable)."""
    conn = sqlite3.connect(str(CLOSE_DB))
    rows = conn.execute("select event_id, home_team, away_team, commence_time from events "
                        "where event_id in (select distinct event_id from props)").fetchall()
    conn.close()
    if EXTRA_EVENTS_FILE.exists():
        known = {r[0] for r in rows}
        rows += [(e["event_id"], e["home_team"], e["away_team"], e["commence_time"])
                 for e in json.loads(EXTRA_EVENTS_FILE.read_text()) if e["event_id"] not in known]
    out = []
    for eid, home, away, commence in rows:
        tip = datetime.fromisoformat(commence.replace("Z", "+00:00"))
        et_date = tip.astimezone(ET).strftime("%Y-%m-%d")
        if start <= et_date <= end:
            out.append({"event_id": eid, "home": home, "away": away, "commence_time": commence, "tip": tip, "et_date": et_date})
    return sorted(out, key=lambda e: (e["et_date"], e["commence_time"]))


def fetch(start: str, end: str, max_credits: int, mode: str, limit: Optional[int] = None, hour: int = DEFAULT_HOUR_UTC, mset: str = "main"):
    key = _key()
    markets, books = MARKET_SETS[mset], SET_BOOKS.get(mset, BOOKS)
    conn = init_db(hour, mset)
    done = {r[0] for r in conn.execute("select event_id from events where status != 'error'")}
    todo = [e for e in events_between(start, end) if e["event_id"] not in done]
    if limit:
        todo = todo[:limit]
    per_call = 10 * len(markets) * (1 if mode == "books" else len(REGIONS.split(",")))
    print(f"{mset} {_tag(hour)}: {len(todo):,} events to fetch between {start} and {end}; mode {mode}; "
          f"at most {per_call} credits each; budget {max_credits:,}", flush=True)

    spent = n_ok = n_rows = n_skipped = 0
    remaining = None
    started = time.time()
    for i, e in enumerate(todo, 1):
        req_dt = requested_time(e["et_date"], hour, e["tip"])
        requested = req_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        base = (e["event_id"], e["et_date"], e["home"], e["away"], e["commence_time"], requested)
        if hour != CLOSE and e["tip"] < req_dt + timedelta(minutes=MIN_LEAD_MINUTES):
            conn.execute("insert or replace into events values (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (*base, None, mode, "tips_before_snapshot", None, 0, 0, datetime.now().isoformat(timespec="seconds")))
            conn.commit()
            n_skipped += 1
            continue
        if spent + per_call > max_credits:
            print(f"budget reached at {spent:,} credits; stopping before event {i}", flush=True)
            break

        params = {"apiKey": key, "markets": ",".join(markets), "oddsFormat": "american", "date": requested}
        if mode == "books":
            params["bookmakers"] = ",".join(books)
        else:
            params["regions"] = REGIONS
        status, body, charged, remaining = get(f"{API}/historical/sports/{SPORT}/events/{e['event_id']}/odds", params)
        spent += charged
        if charged > per_call:
            print(f"charged {charged} for one event, above the expected {per_call}; stopping", flush=True)
            conn.execute("insert or replace into events values (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (*base, None, mode, "error", status, charged, 0, datetime.now().isoformat(timespec="seconds")))
            conn.commit()
            break

        rows, snap, state = [], None, "error"
        if status == 200 and isinstance(body, dict):
            snap = body.get("timestamp")
            data = body.get("data") or {}
            raw_path = raw_dir(hour, mset) / e["et_date"] / f"{e['event_id']}.json.gz"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(raw_path, "wt") as fh:
                json.dump(body, fh)
            updated = {(bk.get("key"), m.get("key")): m.get("last_update") or bk.get("last_update")
                       for bk in data.get("bookmakers", []) for m in bk.get("markets", [])}
            rows = [(*r, updated.get((r[1], r[2]))) for r in parse_all_books(data, e["event_id"], snap)]
            state = "ok" if rows else "empty"
        elif status in (404, 422):
            state = "no_snapshot"
        else:
            message = str((body or {}).get("message", ""))[:160].replace(key, "<key>") if isinstance(body, dict) else ""
            print(f"   {e['et_date']} {e['away']} @ {e['home']}: http {status} {message}", flush=True)

        if rows:
            conn.executemany("insert or replace into props values (?,?,?,?,?,?,?,?,?)", rows)
        conn.execute("insert or replace into events values (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (*base, snap, mode, state, status, charged, len(rows), datetime.now().isoformat(timespec="seconds")))
        conn.commit()
        n_ok += state == "ok"
        n_rows += len(rows)
        if (limit and limit <= 20) or i % 100 == 0 or i == len(todo):
            print(f"{_tag(hour)} [{i:>5}/{len(todo)}] {e['et_date']} {e['away']} @ {e['home']}: {state}, {len(rows)} rows, snapshot {snap}, "
                  f"charged {charged}; run total {spent:,}; balance {remaining}; throttled {THROTTLED}; {time.time() - started:.0f}s", flush=True)
        time.sleep(DELAY)

    print(f"{_tag(hour)} done: {n_ok:,} events with prices, {n_rows:,} rows, {n_skipped:,} skipped (tip too close), "
          f"{spent:,} credits charged this run; balance {remaining}; throttled {THROTTLED}", flush=True)
    conn.close()


def list_events(start: str, end: str):
    """Add the NBA events of each Eastern date in the range to extra_events.json (historical events list, one credit a date)."""
    key = _key()
    have = {e["event_id"]: e for e in json.loads(EXTRA_EVENTS_FILE.read_text())} if EXTRA_EVENTS_FILE.exists() else {}
    day, last, spent, added = datetime.strptime(start, "%Y-%m-%d"), datetime.strptime(end, "%Y-%m-%d"), 0, 0
    while day <= last:
        d = day.strftime("%Y-%m-%d")
        status, body, charged, remaining = get(f"{API}/historical/sports/{SPORT}/events", {"apiKey": key, "date": f"{d}T14:00:00Z"})
        spent += charged
        for ev in (body or {}).get("data", []) if isinstance(body, dict) else []:
            tip = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
            if tip.astimezone(ET).strftime("%Y-%m-%d") == d and ev["id"] not in have:
                have[ev["id"]] = {"event_id": ev["id"], "home_team": ev.get("home_team"), "away_team": ev.get("away_team"),
                                  "commence_time": ev["commence_time"]}
                added += 1
        day += timedelta(days=1)
        time.sleep(DELAY)
    EXTRA_EVENTS_FILE.write_text(json.dumps(sorted(have.values(), key=lambda e: e["commence_time"]), indent=1))
    print(f"events added {added}; total listed {len(have)}; credits charged {spent}; balance {remaining}")


def credits():
    status, _, _, remaining = get(f"{API}/sports", {"apiKey": _key()})
    print(f"http {status}; credits remaining: {remaining}")


def status():
    hours = sorted({DEFAULT_HOUR_UTC} | {int(p.stem.split("_h")[1]) for p in DATA_DIR.glob("early_odds_h*.db")})
    if db_path(CLOSE).exists():
        hours.append(CLOSE)
    total = 0
    for hour in hours:
        if not db_path(hour).exists():
            continue
        conn = init_db(hour)
        print(f"{_tag(hour)}  ({db_path(hour).name})")
        for r in conn.execute("select status, count(*), sum(credits), sum(n_rows), min(et_date), max(et_date) from events group by 1"):
            total += r[2] or 0
            print(f"   {r[0]:<22} events {r[1]:>5,}  credits {r[2] or 0:>8,}  rows {r[3] or 0:>9,}  {r[4]} to {r[5]}")
        books = conn.execute("select bookmaker, count(*) from props group by 1 order by 2 desc").fetchall()
        print("   rows by book: " + ", ".join(f"{b} {n:,}" for b, n in books))
        conn.close()
    print(f"credits charged across all snapshot hours: {total:,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pre-game multi-book prop snapshots at a chosen UTC hour on game day")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--start", required=True, help="first Eastern game date, YYYY-MM-DD")
    f.add_argument("--end", required=True, help="last Eastern game date, YYYY-MM-DD")
    f.add_argument("--hour", type=int, default=DEFAULT_HOUR_UTC, help="snapshot hour, UTC, counted from 00:00 UTC on the Eastern game date")
    f.add_argument("--close", action="store_true", help="take the closing snapshot (ten minutes before tip) instead of a fixed hour")
    f.add_argument("--max-credits", type=int, default=5_000)
    f.add_argument("--mode", choices=["books", "regions"], default="books",
                   help="books: the ten named books (one region's charge); regions: every book in us, us2, eu")
    f.add_argument("--limit", type=int, default=None, help="stop after this many events")
    f.add_argument("--set", dest="mset", choices=sorted(MARKET_SETS), default="main", help="which markets to pull (each set has its own database)")
    sub.add_parser("credits")
    sub.add_parser("status")
    ev = sub.add_parser("events", help="list a date range's events from the API into data/extra_events.json")
    ev.add_argument("--start", required=True)
    ev.add_argument("--end", required=True)
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch(a.start, a.end, a.max_credits, a.mode, a.limit, CLOSE if a.close else a.hour, a.mset)
    elif a.cmd == "events":
        list_events(a.start, a.end)
    elif a.cmd == "credits":
        credits()
    else:
        status()
