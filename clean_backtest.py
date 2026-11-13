#!/usr/bin/env python3
"""Clean backtest data: box scores from Balldontlie, closing odds from data/multibook_odds.db."""

import argparse
import json
import logging
import os
import re
import sqlite3
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("clean_backtest")

BASE_DIR = Path(__file__).resolve().parent
CLEAN_DIR = BASE_DIR / "outputs" / "clean"
BOX_RAW_DIR = CLEAN_DIR / "box_raw"
BOX_FILE = CLEAN_DIR / "box_scores.parquet"

DB_PATH = BASE_DIR / "data" / "multibook_odds.db"
LEGS_FILE = CLEAN_DIR / "legs.parquet"
BOOK_LINES_FILE = CLEAN_DIR / "book_lines.parquet"
GAMES_FILE = CLEAN_DIR / "games.json"
MATCH_REPORT_FILE = CLEAN_DIR / "match_report.json"
SEASON_RAW_FILE = BASE_DIR / "outputs" / "season_raw_books.json"
SEASON_LEGS_FILE = CLEAN_DIR / "season_legs.parquet"
SEASON_BOOK_LINES_FILE = CLEAN_DIR / "season_book_lines.parquet"

ET = ZoneInfo("America/New_York")
BET_BOOK = "draftkings"
MARKET_TO_PROP = {
    "player_points": "points",
    "player_rebounds": "rebounds",
    "player_assists": "assists",
    "player_threes": "threes",
    "player_points_rebounds_assists": "pts_rebs_asts",
}
PROP_TO_STAT = {
    "points": ["pts"],
    "rebounds": ["reb"],
    "assists": ["ast"],
    "threes": ["fg3m"],
    "pts_rebs_asts": ["pts", "reb", "ast"],
}
TEAM_ALIASES = {"LA Clippers": "Los Angeles Clippers"}
NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
# Odds-feed spellings that neither normalize to nor prefix-match the Balldontlie name. Each was checked by hand,
# and an alias still only resolves when that player is on the roster of the game in question.
NAME_ALIASES = {
    "ace bailey": "airious bailey",
    "moe wagner": "moritz wagner",
    "bones hyland": "nahshon hyland",
    "gregory jackson": "gg jackson",
    "kj martin": "kenyon martin",
    "sasha vezenkov": "aleksandar vezenkov",
    "james huff": "jay huff",
    "obadiah toppin": "obi toppin",
    "nicholas batum": "nicolas batum",
    "bojan bogdanovich": "bojan bogdanovic",
    "danilo galinari": "danilo gallinari",
    "bj boston": "brandon boston",
    "trayce jackson": "trayce jackson davis",
    "christian james mccollum": "cj mccollum",
    "alfred joel horford reynoso": "al horford",
    "ishmael larry smith": "ish smith",
    "edrice femi adebayo": "bam adebayo",
}
# Closing snapshots are requested 10 minutes before tip and floored to 5 minutes.
SNAPSHOT_LAG_MINUTES = (-20, -5)
# The odds database was parsed keeping a book's first listed line with its last listed prices, so
# where a book listed several lines the stored price belongs to a different line. A book's rows for
# a season and market are dropped when too many sit implausibly far from DraftKings on the same line.
PRICE_GAP_LIMIT = 0.08
MAX_FAR_SHARE = 0.02
MIN_CELL_ROWS = 100

BDL_BASE = "https://api.balldontlie.io/v1"
SEASON_WINDOWS = [
    ("2023-10-01", "2024-06-30"),
    ("2024-10-01", "2025-06-30"),
    ("2025-10-01", "2026-06-30"),
    ("2026-10-01", None),
]
STAT_FIELDS = ["pts", "reb", "ast", "fg3m", "fg3a", "fgm", "fga", "ftm", "fta", "stl", "blk", "turnover", "oreb", "dreb", "pf"]


# =============================================================================
# BOX SCORES
# =============================================================================

def _month_chunks(today: date):
    chunks = []
    for start, end in SEASON_WINDOWS:
        cur = datetime.strptime(start, "%Y-%m-%d").date()
        last = datetime.strptime(end, "%Y-%m-%d").date() if end else today
        while cur <= last:
            nxt = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
            chunks.append((cur.strftime("%Y-%m"), cur, min(nxt - timedelta(days=1), last)))
            cur = nxt
    return chunks


def _trim(s: dict) -> dict:
    p, t, g = s.get("player") or {}, s.get("team") or {}, s.get("game") or {}
    row = {
        "game_id": g.get("id"), "date": str(g.get("date", ""))[:10], "season": g.get("season"),
        "postseason": g.get("postseason"), "status": g.get("status"),
        "home_team_id": g.get("home_team_id"), "visitor_team_id": g.get("visitor_team_id"),
        "home_score": g.get("home_team_score"), "visitor_score": g.get("visitor_team_score"),
        "player_id": p.get("id"), "first_name": p.get("first_name"), "last_name": p.get("last_name"),
        "team_id": t.get("id"), "team": t.get("full_name"), "min": s.get("min"),
        "position": p.get("position"), "height": p.get("height"),
    }
    for k in STAT_FIELDS:
        row[k] = s.get(k)
    return row


def _fetch_month(label: str, start: date, end: date, api_key: str) -> int:
    out = BOX_RAW_DIR / f"{label}.json"
    session = requests.Session()
    session.headers["Authorization"] = api_key
    rows, cursor = [], None
    while True:
        params = {"start_date": start.isoformat(), "end_date": end.isoformat(), "per_page": 100}
        if cursor:
            params["cursor"] = cursor
        for attempt in range(1, 7):
            try:
                r = session.get(f"{BDL_BASE}/stats", params=params, timeout=30)
            except requests.RequestException as e:
                logger.warning("%s: %s (attempt %d)", label, type(e).__name__, attempt)
                time.sleep(3 * attempt)
                continue
            if r.status_code == 200:
                break
            if r.status_code in (401, 403):
                raise RuntimeError(f"Balldontlie rejected the API key (HTTP {r.status_code})")
            time.sleep(6 * attempt if r.status_code == 429 else 3 * attempt)
        else:
            raise RuntimeError(f"{label}: giving up after repeated failures")
        payload = r.json()
        rows.extend(_trim(s) for s in payload.get("data", []))
        cursor = (payload.get("meta") or {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(0.2)
    out.write_text(json.dumps(rows))
    logger.info("%s: %d stat rows", label, len(rows))
    return len(rows)


def parse_minutes(v) -> float:
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    v = str(v).strip()
    if not v:
        return 0.0
    try:
        if ":" in v:
            m, s = v.split(":", 1)
            return float(m or 0) + float(s or 0) / 60.0
        return float(v)
    except ValueError:
        return 0.0


def fetch_box(workers: int = 4, refresh: bool = False):
    api_key = os.environ.get("BALLDONTLIE_API_KEY", "")
    if not api_key:
        raise SystemExit("BALLDONTLIE_API_KEY is not set (expected in the environment or .env)")
    BOX_RAW_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today()
    todo = []
    for label, start, end in _month_chunks(today):
        settled = end < today - timedelta(days=3)
        if settled and not refresh and (BOX_RAW_DIR / f"{label}.json").exists():
            continue
        todo.append((label, start, end))
    logger.info("Fetching %d month(s) of box scores", len(todo))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        total = sum(pool.map(lambda c: _fetch_month(c[0], c[1], c[2], api_key), todo))
    logger.info("Fetched %d stat rows", total)
    consolidate_box()


def consolidate_box() -> pd.DataFrame:
    rows = []
    for f in sorted(BOX_RAW_DIR.glob("*.json")):
        rows.extend(json.loads(f.read_text()))
    df = pd.DataFrame(rows)
    df = df[df["player_id"].notna() & df["game_id"].notna()].drop_duplicates(["game_id", "player_id"])
    df["player"] = (df["first_name"].fillna("") + " " + df["last_name"].fillna("")).str.strip()
    df["minutes"] = df["min"].map(parse_minutes)
    for k in STAT_FIELDS:
        df[k] = pd.to_numeric(df[k], errors="coerce").fillna(0)
    df["played"] = (df["minutes"] > 0) | (df[STAT_FIELDS].sum(axis=1) > 0)
    df = df.drop(columns=["min", "first_name", "last_name"]).sort_values(["date", "game_id", "team_id", "player"])
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(BOX_FILE, index=False)
    logger.info("Box scores: %d rows, %d games, %s to %s -> %s", len(df), df["game_id"].nunique(),
                df["date"].min(), df["date"].max(), BOX_FILE)
    return df


# =============================================================================
# PLAYER IDENTITY
# =============================================================================

def norm_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[.'`]", "", s).replace("-", " ")
    if "," in s:
        last, _, first = s.partition(",")
        s = f"{first} {last}"
    toks = [t for t in s.split() if t not in NAME_SUFFIXES]
    merged, in_initials = [], False
    for t in toks:
        if len(t) == 1 and in_initials:
            merged[-1] += t
        else:
            merged.append(t)
        in_initials = len(t) == 1
    return " ".join(merged)


class Roster:
    """Players listed for one game (or one date) with a strict name resolver: never a surname on its own."""

    def __init__(self, rows):
        self.by_norm = defaultdict(list)
        self.entries = []
        for player_id, name in rows:
            n = norm_name(name)
            self.by_norm[n].append(player_id)
            toks = n.split()
            if len(toks) >= 2:
                self.entries.append((player_id, toks[0], " ".join(toks[1:])))

    def _unique(self, n):
        ids = self.by_norm.get(n, [])
        return ids[0] if len(ids) == 1 else None

    def resolve_one(self, name):
        n = norm_name(name)
        n = NAME_ALIASES.get(n, n)
        pid = self._unique(n)
        if pid is not None:
            return pid, "exact"
        toks = n.split()
        if len(toks) < 2:
            return None
        pid = self._unique(" ".join(toks[-1:] + toks[:-1]))
        if pid is not None:
            return pid, "reordered"
        if len(toks) >= 3:
            pid = self._unique(f"{toks[0]} {toks[-1]}")
            if pid is not None:
                return pid, "middle-dropped"
        first, rest = toks[0], " ".join(toks[1:])
        cands = [p for p, f, r in self.entries if r == rest and (f.startswith(first) or first.startswith(f))]
        if len(cands) == 1:
            return cands[0], "prefix"
        return None

    def resolve(self, names):
        """Map each feed name to (player_id, how). Unresolved names are omitted."""
        out = {}
        for name in names:
            hit = self.resolve_one(name)
            if hit:
                out[name] = hit
        return out


def player_keys(box: pd.DataFrame) -> dict:
    """player_id -> history key. The display name, unless two different players share it."""
    names = box.drop_duplicates("player_id").set_index("player_id")["player"]
    shared = names[names.duplicated(keep=False)]
    return {pid: (f"{nm} ({pid})" if pid in shared.index else nm) for pid, nm in names.items()}


# =============================================================================
# LEGS
# =============================================================================

def load_box() -> pd.DataFrame:
    box = pd.read_parquet(BOX_FILE)
    box = box[box["status"].astype(str).str.lower().str.startswith("final")].copy()
    box["team"] = box["team"].replace(TEAM_ALIASES)
    return box


def _games_table(box: pd.DataFrame) -> pd.DataFrame:
    team_name = box.drop_duplicates("team_id").set_index("team_id")["team"]
    g = box.drop_duplicates("game_id")[["game_id", "date", "season", "postseason", "home_team_id",
                                         "visitor_team_id", "home_score", "visitor_score"]].copy()
    g["home"] = g["home_team_id"].map(team_name)
    g["away"] = g["visitor_team_id"].map(team_name)
    pts = box.groupby(["game_id", "team_id"])["pts"].sum()
    g["box_adds_up"] = [
        pts.get((gid, h), -1) == hs and pts.get((gid, v), -1) == vs
        for gid, h, v, hs, vs in zip(g["game_id"], g["home_team_id"], g["visitor_team_id"], g["home_score"], g["visitor_score"])
    ]
    return g


def write_games_file(box: pd.DataFrame, keys: dict):
    """Per-game box scores of players who actually played, in the layout the feature builder reads."""
    games = _games_table(box).set_index("game_id")
    played = box[box["played"]]
    stat_cols = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover", "fga", "fgm", "fta", "ftm", "fg3a"]
    out = []
    for gid, grp in played.groupby("game_id", sort=False):
        g = games.loc[gid]
        bs = {}
        for r in grp.itertuples(index=False):
            entry = {k: int(getattr(r, k)) for k in stat_cols}
            entry["min"] = round(float(r.minutes), 1)
            bs[keys[r.player_id]] = entry
        out.append({"game_id": int(gid), "date": g["date"], "home_team": g["home"], "away_team": g["away"], "box_score": bs})
    out.sort(key=lambda x: (x["date"], x["game_id"]))
    GAMES_FILE.write_text(json.dumps({"games": out}))
    logger.info("Games file: %d games -> %s", len(out), GAMES_FILE)


def build_legs():
    box = load_box()
    keys = player_keys(box)
    games = _games_table(box)
    write_games_file(box, keys)

    by_pair = defaultdict(list)
    for r in games.itertuples(index=False):
        by_pair[frozenset((r.home, r.away))].append((r.date, r.game_id))

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    events = pd.read_sql("select event_id, home_team, away_team, commence_time from events", conn)
    events["commence"] = pd.to_datetime(events["commence_time"], utc=True)
    events["et_date"] = events["commence"].dt.tz_convert(ET).dt.strftime("%Y-%m-%d")

    def link(home, away, et_date):
        cands = by_pair.get(frozenset((home, away)), [])
        d0 = datetime.strptime(et_date, "%Y-%m-%d").date()
        best = None
        for gdate, gid in cands:
            gap = abs((datetime.strptime(gdate, "%Y-%m-%d").date() - d0).days)
            if gap <= 1 and (best is None or gap < best[0]):
                best = (gap, gid)
        return best[1] if best else None

    events["game_id"] = [link(h, a, d) for h, a, d in zip(events["home_team"], events["away_team"], events["et_date"])]
    linked = events[events["game_id"].notna()].copy()
    linked["game_id"] = linked["game_id"].astype(int)
    if linked["game_id"].duplicated().any():
        linked = linked.sort_values("commence").drop_duplicates("game_id", keep="last")

    props = pd.read_sql(
        "select event_id, bookmaker, market, player_name, line, over_odds, under_odds, snapshot_time from props", conn)
    conn.close()
    n_props_raw = len(props)
    props = props.merge(linked[["event_id", "game_id", "commence", "commence_time"]], on="event_id")
    lag = (pd.to_datetime(props["snapshot_time"], utc=True) - props["commence"]).dt.total_seconds() / 60.0
    in_window = (lag >= SNAPSHOT_LAG_MINUTES[0]) & (lag <= SNAPSHOT_LAG_MINUTES[1])
    n_off_window = int((~in_window).sum())
    props = props[in_window & props["market"].isin(MARKET_TO_PROP)
                  & (props["over_odds"].abs() >= 100) & (props["under_odds"].abs() >= 100)].copy()

    rosters = {gid: Roster(zip(grp["player_id"], grp["player"])) for gid, grp in box.groupby("game_id")}
    resolved, how_counts, prefix_pairs, unresolved, unresolved_dk = {}, Counter(), Counter(), Counter(), Counter()
    id_to_name = box.drop_duplicates("player_id").set_index("player_id")["player"].to_dict()
    for (eid, gid), grp in props.groupby(["event_id", "game_id"]):
        names = grp["player_name"].unique()
        dk_names = set(grp.loc[grp["bookmaker"] == BET_BOOK, "player_name"])
        res = rosters[gid].resolve(names)
        for name in names:
            if name in res:
                pid, how = res[name]
                resolved[(eid, name)] = pid
                how_counts[how] += 1
                if how != "exact":
                    prefix_pairs[f"{name} -> {id_to_name[pid]} [{how}]"] += 1
            else:
                unresolved[name] += 1
                if name in dk_names:
                    unresolved_dk[name] += 1
    props["player_id"] = [resolved.get(k) for k in zip(props["event_id"], props["player_name"])]
    n_unresolved_rows = int(props["player_id"].isna().sum())
    props = props[props["player_id"].notna()].copy()
    props["player_id"] = props["player_id"].astype(int)
    props = props.drop_duplicates(["event_id", "market", "player_id", "bookmaker"])

    io, iu = _imp(props["over_odds"]), _imp(props["under_odds"])
    props["fair_over"] = io / (io + iu)
    props["season"] = props["game_id"].map(games.set_index("game_id")["season"])
    dk_quote = props[props["bookmaker"] == BET_BOOK].set_index(["event_id", "market", "player_id"])
    quote_key = pd.MultiIndex.from_frame(props[["event_id", "market", "player_id"]])
    on_dk_line = (props["line"] - dk_quote["line"].reindex(quote_key).to_numpy()).abs() < 1e-6
    far = (props["fair_over"] - dk_quote["fair_over"].reindex(quote_key).to_numpy()).abs() > PRICE_GAP_LIMIT
    cells = props[on_dk_line].assign(far=far[on_dk_line]).groupby(["bookmaker", "season", "market"])["far"].agg(["mean", "size"])
    trusted = cells[(cells["mean"] <= MAX_FAR_SHARE) & (cells["size"] >= MIN_CELL_ROWS)].index
    cell_key = pd.MultiIndex.from_frame(props[["bookmaker", "season", "market"]])
    keep = cell_key.isin(trusted) | (props["bookmaker"] == BET_BOOK).to_numpy()
    bad_cells = cells[~cells.index.isin(trusted) & (cells.index.get_level_values("bookmaker") != BET_BOOK)]
    n_untrusted_rows = int((~keep).sum())
    props = props[keep].copy()

    dk = props[props["bookmaker"] == BET_BOOK].copy()
    stats = box.set_index(["game_id", "player_id"])
    idx = pd.MultiIndex.from_arrays([dk["game_id"], dk["player_id"]])
    dk["played"] = stats["played"].reindex(idx).to_numpy()
    dk["minutes"] = stats["minutes"].reindex(idx).to_numpy()
    dk["team"] = stats["team"].reindex(idx).to_numpy()
    dk["prop_type"] = dk["market"].map(MARKET_TO_PROP)
    dk["actual"] = np.nan
    for prop, cols in PROP_TO_STAT.items():
        m = (dk["prop_type"] == prop).to_numpy()
        vals = stats[cols].sum(axis=1).reindex(idx[m]).to_numpy()
        dk.loc[m, "actual"] = vals
    g = games.set_index("game_id")
    for col in ["date", "season", "postseason", "home", "away", "box_adds_up"]:
        dk[col] = dk["game_id"].map(g[col])
    dk["player"] = dk["player_id"].map(keys)

    n_dk = len(dk)
    void = dk[dk["played"] != True]  # noqa: E712
    bad_box = dk[(dk["played"] == True) & (~dk["box_adds_up"].astype(bool))]  # noqa: E712
    legs = dk[(dk["played"] == True) & dk["box_adds_up"].astype(bool)].copy()  # noqa: E712
    legs = legs.sort_values(["date", "game_id", "player", "prop_type"]).reset_index(drop=True)
    legs["leg_id"] = np.arange(len(legs))

    book_lines = props.merge(
        legs[["leg_id", "event_id", "market", "player_id", "snapshot_time"]],
        on=["event_id", "market", "player_id", "snapshot_time"])[["leg_id", "bookmaker", "line", "over_odds", "under_odds"]]
    book_lines = book_lines.rename(columns={"bookmaker": "book"}).sort_values(["leg_id", "book"])
    legs["n_books"] = legs["leg_id"].map(book_lines.groupby("leg_id").size()).fillna(1).astype(int)

    cols = ["leg_id", "event_id", "game_id", "date", "commence_time", "snapshot_time", "season", "postseason",
            "home", "away", "player_id", "player", "team", "prop_type", "line", "over_odds", "under_odds",
            "actual", "minutes", "n_books"]
    legs[cols].to_parquet(LEGS_FILE, index=False)
    book_lines.to_parquet(BOOK_LINES_FILE, index=False)

    report = {
        "events_in_db": int(len(events)), "events_linked_to_a_game": int(len(linked)),
        "prop_rows_in_db": int(n_props_raw), "prop_rows_outside_closing_window": n_off_window,
        "prop_rows_unresolved_player": n_unresolved_rows,
        "name_matches": dict(how_counts), "non_exact_matches": dict(prefix_pairs.most_common()),
        "unresolved_names": dict(unresolved.most_common()),
        "unresolved_names_on_bet_book": dict(unresolved_dk.most_common()),
        "prop_rows_dropped_untrustworthy_book_cells": n_untrusted_rows,
        "untrustworthy_book_cells": {f"{b} | {int(s)} | {m}": {"share_far_from_dk": round(float(r["mean"]), 4), "rows": int(r["size"])}
                                     for (b, s, m), r in bad_cells.iterrows()},
        "dk_props_resolved": int(n_dk), "dk_props_void_did_not_play": int(len(void)),
        "dk_props_dropped_box_does_not_add_up": int(len(bad_box)), "legs": int(len(legs)),
        "games_whose_box_does_not_add_up": int((~games["box_adds_up"]).sum()), "games": int(len(games)),
    }
    MATCH_REPORT_FILE.write_text(json.dumps(report, indent=1))
    logger.info("Legs: %d (void %d, bad box %d) -> %s", len(legs), len(void), len(bad_box), LEGS_FILE)
    return legs


def build_season_legs():
    """This season's six-book snapshots (same timestamp for every book), graded on clean box scores."""
    box = load_box()
    keys = player_keys(box)
    raw = json.loads(SEASON_RAW_FILE.read_text())
    props = pd.DataFrame(raw["props"])
    props = props[props["prop_type"].isin(PROP_TO_STAT) & (props["over_odds"].abs() >= 100) & (props["under_odds"].abs() >= 100)]

    resolved, unresolved, prefix_pairs = {}, Counter(), Counter()
    id_to_name = box.drop_duplicates("player_id").set_index("player_id")["player"].to_dict()
    box_by_date = {d: grp for d, grp in box.groupby("date")}
    for d, grp in props.groupby("date"):
        day = box_by_date.get(d)
        if day is None:
            continue
        res = Roster(zip(day["player_id"], day["player"])).resolve(grp["player"].unique())
        for name in grp["player"].unique():
            if name in res:
                resolved[(d, name)] = res[name][0]
                if res[name][1] != "exact":
                    prefix_pairs[f"{name} -> {id_to_name[res[name][0]]} [{res[name][1]}]"] += 1
            else:
                unresolved[name] += 1
    props = props.assign(player_id=[resolved.get(k) for k in zip(props["date"], props["player"])])
    props = props[props["player_id"].notna()].copy()
    props["player_id"] = props["player_id"].astype(int)
    props = props.drop_duplicates(["date", "player_id", "prop_type", "book"])

    dk = props[props["book"] == BET_BOOK].copy()
    stats = box.set_index(["date", "player_id"])
    stats = stats[~stats.index.duplicated()]
    idx = pd.MultiIndex.from_arrays([dk["date"], dk["player_id"]])
    dk["played"] = stats["played"].reindex(idx).to_numpy()
    dk["minutes"] = stats["minutes"].reindex(idx).to_numpy()
    dk["game_id"] = stats["game_id"].reindex(idx).to_numpy()
    dk["actual"] = np.nan
    for prop, cols in PROP_TO_STAT.items():
        m = (dk["prop_type"] == prop).to_numpy()
        dk.loc[m, "actual"] = stats[cols].sum(axis=1).reindex(idx[m]).to_numpy()
    dk["player"] = dk["player_id"].map(keys)
    n_void = int((dk["played"] != True).sum())  # noqa: E712
    legs = dk[dk["played"] == True].sort_values(["date", "player", "prop_type"]).reset_index(drop=True)  # noqa: E712
    legs["leg_id"] = np.arange(len(legs))
    book_lines = props.merge(legs[["leg_id", "date", "player_id", "prop_type"]], on=["date", "player_id", "prop_type"])
    book_lines = book_lines[["leg_id", "book", "line", "over_odds", "under_odds"]].sort_values(["leg_id", "book"])
    legs["n_books"] = legs["leg_id"].map(book_lines.groupby("leg_id").size()).fillna(1).astype(int)
    legs[["leg_id", "game_id", "date", "player_id", "player", "prop_type", "line", "over_odds", "under_odds",
          "actual", "minutes", "n_books"]].to_parquet(SEASON_LEGS_FILE, index=False)
    book_lines.to_parquet(SEASON_BOOK_LINES_FILE, index=False)
    logger.info("Season legs: %d (void %d, unresolved names %d, prefix matches %s)", len(legs), n_void,
                sum(unresolved.values()), dict(prefix_pairs))
    return legs


# =============================================================================
# EVALUATION
# =============================================================================

SIX_BOOKS = ["draftkings", "fanduel", "betmgm", "betrivers", "espnbet", "betparx"]
REPORT_FILE = CLEAN_DIR / "backtest_report.json"


def _dec(o):
    o = np.asarray(o, dtype=float)
    return np.where(o > 0, 1 + o / 100.0, 1 + 100.0 / np.abs(o))


def _imp(o):
    o = np.asarray(o, dtype=float)
    return np.where(o < 0, -o / (-o + 100.0), 100.0 / (o + 100.0))


def score(df: pd.DataFrame, side, od: str = "od", ud: str = "ud") -> dict:
    """Flat $100 bets. The interval treats each date as one cluster, since same-night props move together."""
    n = len(df)
    if n == 0:
        return {"n": 0, "wr": float("nan"), "roi": float("nan"), "ci": float("nan"), "profit": 0.0}
    side = np.full(n, side) if isinstance(side, str) else np.asarray(side)
    d = np.where(side == "over", df[od], df[ud])
    won = np.where(side == "over", df["won_over"], df["won_under"])
    pnl = np.where(won, 100 * (d - 1), -100.0)
    roi = pnl.mean()
    by_day = pd.DataFrame({"date": df["date"].to_numpy(), "pnl": pnl}).groupby("date")["pnl"].agg(["sum", "size"])
    g = len(by_day)
    if g > 1:
        resid = by_day["sum"] - roi * by_day["size"]
        ci = 1.96 * np.sqrt(g / (g - 1) * (resid ** 2).sum()) / n
    else:
        ci = float("nan")
    return {"n": int(n), "wr": float(won.mean() * 100), "roi": float(roi), "ci": float(ci), "profit": float(pnl.sum())}


def fmt(s: dict) -> str:
    if not s["n"]:
        return "no bets"
    return f"{s['n']:>6,} bets  {s['wr']:5.1f}% won  ROI {s['roi']:+6.2f}% [±{s['ci']:.2f}]"


def _add_market_columns(df: pd.DataFrame, book_lines: pd.DataFrame) -> pd.DataFrame:
    from train_v8 import devig_power

    df = df.copy()
    df["io"], df["iu"] = _imp(df["over_odds"]), _imp(df["under_odds"])
    df["od"], df["ud"] = _dec(df["over_odds"]), _dec(df["under_odds"])
    df["hold"] = df["io"] + df["iu"] - 1
    df["won_over"] = df["actual"] > df["line"]
    df["won_under"] = df["actual"] < df["line"]
    df["p"] = np.clip(np.where(df["nb"] >= 3, 0.70 * df["cons"] + 0.30 * df["p_model"],
                               np.where(df["nb"] == 2, 0.50 * df["cons"] + 0.50 * df["p_model"], df["p_model"])), 0.01, 0.99)
    for name, col in [("blend", "p"), ("model", "p_model"), ("cons", "cons")]:
        eo = (df[col] - df["io"]) * 100
        eu = ((1 - df[col]) - df["iu"]) * 100
        df[f"eo_{name}"], df[f"eu_{name}"] = eo, eu
        df[f"side_{name}"] = np.where(eo >= eu, "over", "under")
        df[f"edge_{name}"] = np.maximum(eo, eu)
    df["conf"] = np.maximum(df["p"], 1 - df["p"])

    bl = book_lines.merge(df[["leg_id", "line"]].rename(columns={"line": "dk_line"}), on="leg_id")
    same = bl[(bl["line"] - bl["dk_line"]).abs() < 1e-6]
    for tag, rows in [("all", same), ("six", same[same["book"].isin(SIX_BOOKS)])]:
        best = rows.groupby("leg_id")[["over_odds", "under_odds"]].max()
        df[f"od_{tag}"] = _dec(df["leg_id"].map(best["over_odds"]).fillna(df["over_odds"]))
        df[f"ud_{tag}"] = _dec(df["leg_id"].map(best["under_odds"]).fillna(df["under_odds"]))
    pin = same[same["book"] == "pinnacle"].drop_duplicates("leg_id").set_index("leg_id")
    if len(pin):
        fair = np.array([devig_power(int(o), int(u))[0] for o, u in zip(pin["over_odds"], pin["under_odds"])])
        df["pin_over"] = df["leg_id"].map(pd.Series(fair, index=pin.index))
    else:
        df["pin_over"] = np.nan
    return df


def backtest_frame(models_dir: Path) -> pd.DataFrame:
    """The held-out last 25% of each prop, scored by the saved classifiers exactly as training split it."""
    import joblib
    from train_v8 import DATASETS

    cache = DATASETS["clean"]["cache_dir"]
    X = pd.read_parquet(cache / "features.parquet")
    meta = pd.read_parquet(cache / "metadata.parquet")
    info = json.loads((models_dir / "model_metadata.json").read_text())
    frames = []
    for prop in info["results"]:
        mask = (meta["prop_type"] == prop).to_numpy()
        Xp, mp = X[mask], meta[mask]
        order = np.argsort(mp["date"].values)
        Xp, mp = Xp.iloc[order], mp.iloc[order]
        cut = int(len(Xp) * 0.75)
        Xt, mt = Xp.iloc[cut:], mp.iloc[cut:].copy()
        clf = joblib.load(models_dir / f"{prop}_classifier.pkl")
        raw = clf.predict_proba(Xt[info[prop]["feature_names"]])[:, 1]
        method = info[prop].get("calibration_method", "raw")
        if method == "platt":
            raw = joblib.load(models_dir / f"{prop}_platt.pkl").predict_proba(raw.reshape(-1, 1))[:, 1]
        elif method == "isotonic":
            raw = joblib.load(models_dir / f"{prop}_isotonic.pkl").predict(raw)
        mt["p_model"] = raw
        mt["cons"] = Xt["consensus_over_prob"].to_numpy()
        mt["nb"] = Xt["num_books"].to_numpy()
        mt["line_range"] = Xt["line_range"].to_numpy()
        frames.append(mt)
    df = pd.concat(frames, ignore_index=True)
    return _add_market_columns(df, pd.read_parquet(BOOK_LINES_FILE))


def season_frame(models_dir: Path) -> pd.DataFrame:
    """This season's legs, priced through the production predictor with history up to the day before each game."""
    from v2.v8_predictor import V8Predictor

    legs = pd.read_parquet(SEASON_LEGS_FILE)
    bl = pd.read_parquet(SEASON_BOOK_LINES_FILE)
    books = defaultdict(dict)
    for r in bl.itertuples(index=False):
        books[r.leg_id][r.book] = {"line": r.line, "over_odds": r.over_odds, "under_odds": r.under_odds}
    predictor = V8Predictor(models_dir=str(models_dir))
    rows = []
    for r in legs.itertuples(index=False):
        out = predictor.predict(r.player, r.prop_type, r.date, {
            "line": r.line, "over_odds": r.over_odds, "under_odds": r.under_odds,
            "all_books": books[r.leg_id], "min_games": 3})
        if out is not None:
            rows.append((r.leg_id, out["model_prob"], out["consensus_prob"], out["consensus_num_books"]))
    pred = pd.DataFrame(rows, columns=["leg_id", "p_model", "cons", "nb"])
    df = legs.merge(pred, on="leg_id")
    lines = bl.groupby("leg_id")["line"].agg(["max", "min"])
    df["line_range"] = df["leg_id"].map(lines["max"] - lines["min"]).fillna(0.0)
    return _add_market_columns(df, bl)


STRATEGIES = {
    "Tier 1: no assists, best side, edge>=8, conf>=60":
        lambda d: (d[(d["prop_type"] != "assists") & (d["edge_blend"] >= 8) & (d["conf"] >= 0.60)], None),
    "Tier 2: pts+reb UNDER, edge 5-15, conf>=56, 5+ books":
        lambda d: (d[d["prop_type"].isin(["points", "rebounds"]) & d["eu_blend"].between(5, 15)
                     & (1 - d["p"] >= 0.56) & (d["nb"] >= 5)], "under"),
    "Tier 3: assists UNDER, edge 3-10, conf>=56, 2+ books":
        lambda d: (d[(d["prop_type"] == "assists") & d["eu_blend"].between(3, 10) & (1 - d["p"] >= 0.56) & (d["nb"] >= 2)], "under"),
    "All UNDER, edge>=3, conf>=56":
        lambda d: (d[(d["eu_blend"] >= 3) & (1 - d["p"] >= 0.56)], "under"),
    "All UNDER, edge>=5, conf>=56":
        lambda d: (d[(d["eu_blend"] >= 5) & (1 - d["p"] >= 0.56)], "under"),
    "Best side, edge>=8, conf>=58":
        lambda d: (d[(d["edge_blend"] >= 8) & (d["conf"] >= 0.58)], None),
    "Best side, any claimed edge > 0":
        lambda d: (d[d["edge_blend"] > 0], None),
}


def _strategy_scores(df: pd.DataFrame, od: str = "od", ud: str = "ud") -> dict:
    out = {}
    for name, pick in STRATEGIES.items():
        sub, side = pick(df)
        out[name] = score(sub, sub["side_blend"].to_numpy() if side is None else side, od, ud)
    return out


def _edge_buckets(df: pd.DataFrame, signal: str) -> list:
    rows = []
    for lo, hi in [(-100, 0), (0, 2), (2, 4), (4, 8), (8, 100)]:
        sub = df[(df[f"edge_{signal}"] > lo) & (df[f"edge_{signal}"] <= hi)]
        rows.append((f"{lo} to {hi}" if lo > -100 else "0 or less", score(sub, sub[f"side_{signal}"].to_numpy())))
    return rows


def _brier(p, won_over) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(won_over, dtype=float)) ** 2))


def _grid(bt: pd.DataFrame, cs: pd.DataFrame) -> pd.DataFrame:
    """The session's strategy search (prop set x side x edge x confidence x consensus x books) on both samples."""
    props = {
        "all": None, "no_ast": ["points", "rebounds", "threes", "pts_rebs_asts"],
        "no_pra": ["points", "rebounds", "assists", "threes"], "no_ast_pra": ["points", "rebounds", "threes"],
        "pts": ["points"], "reb": ["rebounds"], "ast": ["assists"], "threes": ["threes"], "pra": ["pts_rebs_asts"],
        "pts+reb": ["points", "rebounds"], "pts+threes": ["points", "threes"], "reb+threes": ["rebounds", "threes"],
        "pts+ast": ["points", "assists"], "reb+ast": ["rebounds", "assists"],
    }
    edges = {"e0": (0, None), "e1": (1, None), "e2": (2, None), "e3": (3, None), "e5": (5, None), "e8": (8, None),
             "e10": (10, None), "e3-10": (3, 10), "e5-15": (5, 15), "e3-8": (3, 8), "e1-5": (1, 5), "any": (None, None)}
    confs = {"c50": 0.50, "c52": 0.52, "c54": 0.54, "c56": 0.56, "c58": 0.58, "c60": 0.60, "cany": None}
    conss = {"cons_any": None, "cons50": 0.50, "cons51": 0.51, "cons52": 0.52, "cons53": 0.53, "cons54": 0.54, "cons55": 0.55}
    books = {"bany": None, "b2": 2, "b3": 3, "b5": 5}

    def arrays(df):
        a = {"n": len(df)}
        for side in ("best", "under", "over"):
            pick_over = (df["side_blend"] == "over").to_numpy() if side == "best" else np.full(len(df), side == "over")
            a[side] = {
                "edge": np.where(pick_over, df["eo_blend"], df["eu_blend"]),
                "conf": np.where(pick_over, df["p"], 1 - df["p"]),
                "cons": np.where(pick_over, df["cons"], 1 - df["cons"]),
                "pnl": np.where(np.where(pick_over, df["won_over"], df["won_under"]),
                                100 * (np.where(pick_over, df["od"], df["ud"]) - 1), -100.0),
            }
        a["prop"] = {k: (np.ones(len(df), bool) if v is None else df["prop_type"].isin(v).to_numpy()) for k, v in props.items()}
        a["books"] = {k: (np.ones(len(df), bool) if v is None else (df["nb"] >= v).to_numpy()) for k, v in books.items()}
        return a

    A, B = arrays(bt), arrays(cs)
    rows = []
    for side in ("best", "under", "over"):
        for ek, (lo, hi) in edges.items():
            if ek == "any" and side == "best":
                continue
            me = [np.ones(x["n"], bool) if lo is None else ((x[side]["edge"] >= lo) & (True if hi is None else x[side]["edge"] <= hi)) for x in (A, B)]
            for ck, cv in confs.items():
                mc = [m if cv is None else m & (x[side]["conf"] >= cv) for m, x in zip(me, (A, B))]
                for sk, sv in conss.items():
                    ms = [m if sv is None else m & (x[side]["cons"] >= sv) for m, x in zip(mc, (A, B))]
                    for bk in books:
                        mb = [m & x["books"][bk] for m, x in zip(ms, (A, B))]
                        for pk in props:
                            m1, m2 = mb[0] & A["prop"][pk], mb[1] & B["prop"][pk]
                            n1, n2 = int(m1.sum()), int(m2.sum())
                            if n1 < 50 or n2 < 10:
                                continue
                            rows.append((f"{pk}|{side}|{ek}|{ck}|{sk}|{bk}", side, n1, A[side]["pnl"][m1].mean(),
                                         n2, B[side]["pnl"][m2].mean()))
    return pd.DataFrame(rows, columns=["label", "side", "bt_n", "bt_roi", "cs_n", "cs_roi"])


def evaluate(models_dir: Path):
    bt = backtest_frame(models_dir)
    cs = season_frame(models_dir)
    bt.to_parquet(CLEAN_DIR / "backtest_predictions.parquet", index=False)
    cs.to_parquet(CLEAN_DIR / "season_predictions.parquet", index=False)
    legs = _add_market_columns(pd.read_parquet(LEGS_FILE).assign(p_model=0.5, cons=0.5, nb=1), pd.read_parquet(BOOK_LINES_FILE))
    report = {}
    line = "=" * 118

    print(line)
    print(f"1. BASELINES AT DRAFTKINGS' CLOSING PRICE  (all {len(legs):,} graded bets, three seasons)")
    print(line)
    print(f"   average hold {legs['hold'].mean() * 100:.2f}%   {legs['date'].min()} to {legs['date'].max()}")
    report["baseline"] = {}
    for label, sub in [("all three seasons", legs)] + [(f"season starting {s}", g) for s, g in legs.groupby("season")]:
        u, o = score(sub, "under"), score(sub, "over")
        report["baseline"][label] = {"under": u, "over": o}
        print(f"   {label:<22} blind UNDER {fmt(u)}    blind OVER  ROI {o['roi']:+6.2f}%")
    for prop, sub in legs.groupby("prop_type"):
        print(f"   {prop:<22} blind UNDER {fmt(score(sub, 'under'))}")
    u, o = score(cs, "under"), score(cs, "over")
    report["baseline"]["this season (Oct 22 - Nov 9, 20:00 UTC prices)"] = {"under": u, "over": o}
    print(f"   {'this season, 18 days':<22} blind UNDER {fmt(u)}    blind OVER  ROI {o['roi']:+6.2f}%   (hold {cs['hold'].mean() * 100:.2f}%)")

    print("\n" + line)
    print(f"2. HELD-OUT TEST SLICE  ({len(bt):,} bets, {bt['date'].min()} to {bt['date'].max()}; the model never saw these)")
    print(line)
    report["brier"] = {}
    print(f"   {'prop':<16}{'bets':>8}{'DK price':>11}{'consensus':>11}{'classifier':>12}{'70/30 blend':>13}   (Brier score, lower is better)")
    for prop, sub in list(bt.groupby("prop_type")) + [("ALL", bt)]:
        dk_fair = sub["io"] / (sub["io"] + sub["iu"])
        b = {"dk": _brier(dk_fair, sub["won_over"]), "consensus": _brier(sub["cons"], sub["won_over"]),
             "classifier": _brier(sub["p_model"], sub["won_over"]), "blend": _brier(sub["p"], sub["won_over"])}
        report["brier"][prop] = b
        print(f"   {prop:<16}{len(sub):>8,}{b['dk']:>11.4f}{b['consensus']:>11.4f}{b['classifier']:>12.4f}{b['blend']:>13.4f}")

    print("\n" + line)
    print("3. DOES A CLAIMED EDGE PAY?  (bet the side with the larger claimed edge, at DraftKings)")
    print(line)
    report["edge_buckets"] = {}
    for tag, df in [("backtest test slice", bt), ("this season", cs)]:
        for signal, label in [("blend", "70/30 blend (what the system uses)"), ("model", "classifier alone"), ("cons", "line-matched consensus alone")]:
            print(f"   [{tag}] {label}")
            rows = _edge_buckets(df, signal)
            report["edge_buckets"][f"{tag} | {signal}"] = {k: v for k, v in rows}
            for k, v in rows:
                print(f"      claimed edge {k:<11} {fmt(v)}")

    print("\n" + line)
    print("4. THE STRATEGIES REPORTED EARLIER, ON CLEAN DATA")
    print(line)
    report["strategies"] = {"backtest": _strategy_scores(bt), "season": _strategy_scores(cs)}
    for name in STRATEGIES:
        print(f"   {name}")
        print(f"      backtest test slice  {fmt(report['strategies']['backtest'][name])}")
        print(f"      this season          {fmt(report['strategies']['season'][name])}")
    print("\n   'All UNDER, edge>=3, conf>=56' by month (backtest test slice):")
    sub, _ = STRATEGIES["All UNDER, edge>=3, conf>=56"](bt)
    for month, g in sub.groupby(sub["date"].str[:7]):
        print(f"      {month}  {fmt(score(g, 'under'))}")

    print("\n" + line)
    print("5. LINE SHOPPING  (same bets, best price among books quoting DraftKings' line at the same moment)")
    print(line)
    report["shopping"] = {}
    for tag, df in [("all three seasons", legs), ("this season", cs)]:
        row = {"dk": score(df, "under"), "six": score(df, "under", "od_six", "ud_six"), "all": score(df, "under", "od_all", "ud_all")}
        report["shopping"][f"blind under | {tag}"] = row
        print(f"   blind UNDER, {tag}:  DraftKings {row['dk']['roi']:+.2f}%   best of your six books {row['six']['roi']:+.2f}%"
              + (f"   best of all 23 books {row['all']['roi']:+.2f}%" if tag != "this season" else ""))
    for name in ["All UNDER, edge>=3, conf>=56", "Best side, any claimed edge > 0"]:
        for tag, df in [("backtest test slice", bt), ("this season", cs)]:
            sub, side = STRATEGIES[name](df)
            s = sub["side_blend"].to_numpy() if side is None else side
            row = {"dk": score(sub, s), "six": score(sub, s, "od_six", "ud_six"), "all": score(sub, s, "od_all", "ud_all")}
            report["shopping"][f"{name} | {tag}"] = row
            print(f"   {name}, {tag}:  DraftKings {row['dk']['roi']:+.2f}%   best of six {row['six']['roi']:+.2f}% [±{row['six']['ci']:.2f}]"
                  + (f"   best of all {row['all']['roi']:+.2f}%" if tag != "this season" else ""))

    print("\n" + line)
    print("6. PINNACLE AS THE FAIR PRICE  (Pinnacle on the same line at the same moment; bet the best of the six books)")
    print(line)
    report["pinnacle"] = {}
    for tag, df in [("all three seasons", legs), ("backtest test slice", bt)]:
        pin = df[df["pin_over"].notna()]
        ev_o = pin["pin_over"] * pin["od_six"] - 1
        ev_u = (1 - pin["pin_over"]) * pin["ud_six"] - 1
        side = np.where(ev_o >= ev_u, "over", "under")
        ev = np.maximum(ev_o, ev_u) * 100
        print(f"   [{tag}] bets with Pinnacle on DraftKings' line: {len(pin):,}")
        for lo in (0, 1, 2, 4):
            m = (ev > lo).to_numpy()
            s = score(pin[m], side[m], "od_six", "ud_six")
            report["pinnacle"][f"{tag} | EV>{lo}%"] = s
            print(f"      expected value above {lo}%   {fmt(s)}")
        m = ((1 - pin["pin_over"]) >= 0.55) & (pin["under_odds"] > 0)
        s = score(pin[m], "under")
        report["pinnacle"][f"{tag} | pinnacle under>=55% and DK plus money"] = s
        print(f"      Pinnacle says under >= 55% and DraftKings pays plus money   {fmt(s)}")

    print("\n" + line)
    print("7. THE STRATEGY SEARCH, RE-RUN ON CLEAN DATA  (prop set x side x edge x confidence x consensus x books)")
    print(line)
    grid = _grid(bt, cs)
    grid.to_parquet(CLEAN_DIR / "strategy_grid.parquet", index=False)
    pb, pc = (grid["bt_roi"] > 0).mean(), (grid["cs_roi"] > 0).mean()
    both = ((grid["bt_roi"] > 0) & (grid["cs_roi"] > 0)).mean()
    report["grid"] = {"combinations_with_volume": int(len(grid)), "profitable_backtest": float(pb), "profitable_season": float(pc),
                      "profitable_both": float(both), "expected_both_if_unrelated": float(pb * pc),
                      "rank_correlation": float(grid["bt_roi"].corr(grid["cs_roi"], method="spearman"))}
    print(f"   combinations with enough bets in both samples: {len(grid):,}")
    print(f"   profitable in backtest {pb * 100:.1f}%   this season {pc * 100:.1f}%   both {both * 100:.1f}%   (expected if unrelated {pb * pc * 100:.1f}%)")
    print(f"   rank correlation between backtest ROI and season ROI: {report['grid']['rank_correlation']:+.3f}")
    big = grid[grid["bt_n"] >= 200]
    top = big.nlargest(max(len(big) // 10, 1), "bt_roi")
    report["grid"]["top_decile"] = {"n": int(len(top)), "backtest_roi": float(top["bt_roi"].mean()), "season_roi": float(top["cs_roi"].mean()),
                                    "season_profitable": float((top["cs_roi"] > 0).mean())}
    print(f"   top tenth by backtest ROI (200+ bets, {len(top):,} combos): mean backtest ROI {top['bt_roi'].mean():+.1f}%  ->  "
          f"mean season ROI {top['cs_roi'].mean():+.1f}%, {(top['cs_roi'] > 0).mean() * 100:.0f}% profitable this season")
    print("   best ten by backtest ROI with 500+ bets:")
    for r in grid[grid["bt_n"] >= 500].nlargest(10, "bt_roi").itertuples(index=False):
        print(f"      {r.label:<44} backtest {r.bt_n:>6,} bets {r.bt_roi:+6.2f}%   season {r.cs_n:>5,} bets {r.cs_roi:+6.2f}%")

    REPORT_FILE.write_text(json.dumps(report, indent=1))
    print(f"\nSaved: {REPORT_FILE}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    fb = sub.add_parser("fetch-box", help="Download box scores for every season window")
    fb.add_argument("--refresh", action="store_true", help="Re-download months that are already cached")
    sub.add_parser("build-legs", help="Join closing odds to box scores for the three-season backtest")
    sub.add_parser("build-season-legs", help="Join this season's snapshots to box scores")
    ev = sub.add_parser("evaluate", help="Backtest the clean model on the held-out slice and on this season")
    ev.add_argument("--models-dir", default=str(BASE_DIR / "models" / "v8_clean"))
    args = parser.parse_args()
    if args.cmd == "fetch-box":
        fetch_box(refresh=args.refresh)
    elif args.cmd == "build-legs":
        build_legs()
    elif args.cmd == "build-season-legs":
        build_season_legs()
    elif args.cmd == "evaluate":
        evaluate(Path(args.models_dir))


if __name__ == "__main__":
    main()
