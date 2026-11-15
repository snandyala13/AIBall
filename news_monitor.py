#!/usr/bin/env python3
"""
news_monitor.py — paper-mode live monitor for the news-speed effect. No bets are placed.

When DraftKings stops quoting a key player's points prop before tip (in the tests he then sat out 75-100% of the time),
teammates' props at slower books stay at their pre-news prices for a few minutes. This watches for that live and logs the
stale teammate quotes that the teammate model rates at 3%+ value, with timestamps, so the effect is measured at real speed
before any money is used. Tested in news_lab.py and news_confirm.py
(research_notes/Quant sports betting techniques/00_inhouse_backtest_audit.md, sections 8j and 8k).

Frozen rules (outputs/clean/feature_lab/protocol_news_speed.json and protocol_news_confirm.json):
  key player   a DraftKings points line of 15.5 or more
  trigger      DraftKings stops quoting a key player's points prop while still quoting 3+ other players' points in that game,
               5 minutes or more before tip; once per key player and game
  teammate     a player whose latest game this season (box scores before the date) was for the key player's team
  stale quote  a teammate's prop at one of the six bet books whose line and both prices equal the baseline poll's
  flag         predicted post-news fair = normal CDF(probit of the baseline S1 fair at that line + the teammate model's shift);
               the side with the higher predicted value, at 3%+; one per teammate, prop and side, at the best book
The fair price uses the line-conversion spreads frozen in paper_log_sigma.json (2023-24 to 2025-26).

Fast-book rule: MEASUREMENT ONLY, logged in its own table (ff_flags), never changes the rule above. It did not pass its
test (fast_follow_lab.py, protocol_fast_follow.json, audit section 8m): its value held up (+7.6% on 275 bets), but only
35% of its flagged prices were still on offer 5 minutes later. It runs here to measure, at one-minute resolution, how long
such prices last.
  reference    each book's quote before the trigger: points from the minute poll about 5 minutes earlier, other markets
               from the baseline
  movers       books whose quote for a teammate's prop differs from their reference quote (2+ needed); their de-vigged
               prices, moved to the stale line, averaged on the normal scale, give the fair price
  flag         a teammate's quote at a bet book still at its reference quote, worth 3%+ against that price, in the 30
               minutes after the trigger; one per teammate, prop and side
Every flagged price of either rule is then re-checked at each later poll for an hour (points every minute), recording
when it was last seen unchanged and when it first was not.

Strict trigger: MEASUREMENT ONLY (outputs/clean/feature_lab/protocol_strict_trigger.json, from 2026-11-15), recorded on the
trigger and flag rows; it never changes the rule above and uses no extra credits. In the first two live nights only 4 of 16
pulled players sat out (75-100% in the tests): the live trigger also fires when DraftKings briefly takes players down.
  confirmed    at the pull, or at the first minute poll 2 minutes or more after it, another book that had quoted the player's
               points no longer does (while quoting 3+ other players' points); or at that 2-minute poll DraftKings has still
               not re-posted him
  kept flag    from a confirmed trigger, on a player DraftKings still quotes (never one it has pulled too), and still on offer
               when the trigger was confirmed: at its own price if another book was already off at the pull, otherwise only if
               the 2-minute follow-up showed the same line and price

Polling, for games tipping within 3 hours (the events list is free):
  every minute            player_points at the ten books: 1 credit a game
  every 30 minutes        all five markets at the ten books, 5 credits a game: the baseline for the stale test and the
                          pre-news fair price; not refreshed while a key player is missing at a book that had him
  at a trigger            all five markets at once: the quotes the flags use; again about 2, 5 and 10 minutes later
                          (every trigger), to see how long flagged prices stay on offer and for the fast-book rule
  10 minutes before tip   all five markets once, for games with flags of either rule: the closing fair price used in grading
About 2,000-3,000 credits on a ten-game night; a daily cap (NEWS_MONITOR_DAILY_CAP, default 4,000) stops polling.
Prices are as the Odds API reports them; anyone acting on a flag must still see the price at the book.

    python news_monitor.py run [--minutes 240]     # poll locally (on AWS the web app calls tick() every minute)
    python news_monitor.py grade [--refresh-box]   # closing value and results for finished games
    python news_monitor.py report
    python news_monitor.py replay                  # offline check: this code over the bought 5-minute snapshots
"""

import argparse
import io
import json
import os
import sqlite3
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

BASE_DIR = Path(__file__).parent
_EFS = os.environ.get("EFS_MOUNT")
LOG_DB = Path(os.environ.get("NEWS_MONITOR_DB") or (Path(_EFS) / "news_monitor" / "news_monitor.db" if _EFS else BASE_DIR / "data" / "news_monitor.db"))
# Copies of the teammate model (teammate_model.py); /app/outputs is the EFS volume on AWS, so the lab copies are not in the image.
MODEL_FILE = BASE_DIR / "news_monitor_files" / "teammate_lgb.model"
META_FILE = BASE_DIR / "news_monitor_files" / "teammate_meta.json"
KEY_LINE = 15.5
EV_MIN = 0.03
MIN_OTHERS = 3
POLL_S = int(os.environ.get("NEWS_MONITOR_POLL_S", "60"))
WINDOW = timedelta(hours=float(os.environ.get("NEWS_MONITOR_WINDOW_H", "3")))
BASE_EVERY = timedelta(minutes=30)
STOP_BEFORE_TIP = timedelta(minutes=5)
CLOSE_BEFORE_TIP = timedelta(minutes=10)
FOLLOW_UPS = (2, 5, 10)
FF_MIN_MOVERS = 2
FF_HORIZON = timedelta(minutes=30)
FF_REF_LAG = timedelta(minutes=4, seconds=30)  # the minute poll about 5 minutes before the trigger (the tests' snapshot spacing)
TRACK_FOR = timedelta(minutes=60)
CHECK_AFTER = timedelta(minutes=2)  # strict trigger: the first minute poll this long after the pull (the 2-minute follow-up's tick)
STRICT_LABEL = {"book_at_pull": "another book off at the pull", "book_by_2": "another book off by 2 min",
                "dk_still_off_2": "still off DraftKings at 2 min", "no": "not confirmed"}
EVENTS_EVERY = timedelta(minutes=10)
HEARTBEAT = timedelta(minutes=30)
DAILY_CAP = int(os.environ.get("NEWS_MONITOR_DAILY_CAP", "4000"))
POINTS = ["player_points"]
COLS = ["book", "prop_type", "player", "line", "over_odds", "under_odds", "last_update"]

_LOCK = threading.Lock()
_OUT = threading.local()
_STATE = {"events": [], "events_at": None, "ev": {}, "box": None, "box_mtime": None, "model": None, "ids": {}, "beat": None}


def _say(*args):
    """print, except that inside scheduled() it writes to that call's buffer (only the calling thread's output)."""
    print(*args, file=getattr(_OUT, "buf", None) or sys.stdout)


def _stamp(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _db() -> sqlite3.Connection:
    LOG_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(LOG_DB), timeout=30)
    conn.executescript("""
        create table if not exists polls (poll_id integer primary key autoincrement, polled_at text, event_id text, kind text,
            quotes integer, credits integer, balance text);
        create table if not exists quotes (poll_id integer, event_id text, book text, prop_type text, player text, line real,
            over_odds integer, under_odds integer, last_update text);
        create index if not exists idx_quotes_poll on quotes(poll_id);
        create table if not exists triggers (trigger_id integer primary key autoincrement, event_id text, date text,
            home_team text, away_team text, commence_time text, key_player text, key_line real, dk_last_seen text,
            trigger_at text, baseline_at text, note text, teammate_quotes integer, stale_quotes integer, flags integer,
            reappeared_at text, key_played integer, graded_at text, unique (event_id, key_player));
        create table if not exists flags (event_id text, trigger_id integer, key_player text, player text, prop_type text,
            side text, book text, line real, over_odds integer, under_odds integer, fair_pre real, shift real,
            fair_pred real, value_pred real, quote_age_min real, flagged_at text, commence_time text, date text,
            still_2 text, still_5 text, close_fair real, value_close real, actual real, played integer, won integer,
            graded_at text, primary key (event_id, player, prop_type, side));
        create table if not exists ff_flags (event_id text, trigger_id integer, key_player text, player text, prop_type text,
            side text, book text, line real, over_odds integer, under_odds integer, fair_ff real, value_ff real,
            n_movers integer, movers text, quote_age_min real, flagged_at text, mins_after real, commence_time text, date text,
            still_2 text, still_5 text, still_10 text, last_same_at text, gone_at text, close_fair real, value_close real,
            actual real, played integer, won integer, graded_at text, primary key (event_id, player, prop_type, side));
    """)
    # Measurement columns, added to older databases (2026-11-13: price timing; 2026-11-15: strict trigger). The rule is unchanged.
    for table, cols in (("flags", ("still_10 text", "last_same_at text", "gone_at text", "mate_pulled integer")),
                        ("ff_flags", ("mate_pulled integer",)),
                        ("triggers", ("books_off_t0 text", "check_at text", "dk_back_2 integer", "books_off_2 text", "strict text"))):
        have = {r[1] for r in conn.execute(f"pragma table_info({table})")}
        for col in cols:
            if col.split()[0] not in have:
                conn.execute(f"alter table {table} add column {col}")
    conn.commit()
    return conn


# =============================================================================
# THE RULE (pure: used live and by the replay)
# =============================================================================

def identities(box: pd.DataFrame, date: str, teams, names) -> dict:
    """Feed name -> (player_id, team_id), for players whose latest game this season before `date` was for one of `teams`."""
    from clean_backtest import TEAM_ALIASES, Roster

    teams = tuple(sorted(TEAM_ALIASES.get(t, t) for t in teams))
    key = (date, teams)
    if key not in _STATE["ids"]:
        season = int(date[:4]) if int(date[5:7]) >= 8 else int(date[:4]) - 1
        team_id = dict(zip(box["team"], box["team_id"]))
        b = box[(box["season"] == season) & (box["date"] < date)]
        latest = b.sort_values("date").drop_duplicates("player_id", keep="last")
        latest = latest[latest["team_id"].isin({team_id.get(t) for t in teams})]
        _STATE["ids"][key] = (Roster(zip(latest["player_id"], latest["player"])), latest.set_index("player_id")["team_id"].to_dict())
    roster, team_of = _STATE["ids"][key]
    return {n: (int(pid), int(team_of[pid])) for n, (pid, _how) in roster.resolve(sorted(set(names))).items()}


def fair(rows: pd.DataFrame, ref: pd.DataFrame, date: str, event_id: str):
    """S1 fair over-probability at each row's line from the reference quotes (all ten books), and the source it used."""
    import market_lab as ml
    import paper_log
    from clean_backtest import norm_name

    cols = ["date", "player_id", "prop_type", "book", "line", "over_odds", "under_odds"]
    R = ref.assign(date=date, player_id=event_id + "|" + ref["player"].map(norm_name))[cols]
    P = rows.assign(date=date, player_id=event_id + "|" + rows["player"].map(norm_name), book="ref")[cols]
    Q = ml.add_consensus(pd.concat([R, P.assign(row_tag=np.arange(len(P)))], ignore_index=True), paper_log._sigma())
    Q = Q[Q["book"] == "ref"].sort_values("row_tag")
    return Q["p_s1"].to_numpy(dtype=float), Q["ref"].to_numpy()


def compute_flags(event_id: str, date: str, teams, keys: dict, full: pd.DataFrame, base: pd.DataFrame, box: pd.DataFrame,
                  model, now: datetime):
    """The frozen rule at one trigger. keys: {key player's feed name: DraftKings points line}; full: quotes at the trigger;
    base: the baseline quotes. Returns ({key: counts and note}, flags), flags deduplicated per teammate, prop and side."""
    from clean_backtest import SIX_BOOKS, _dec

    ids = identities(box, date, teams, set(full["player"]) | set(base["player"]) | set(keys))
    ref = base[base["book"].isin(SIX_BOOKS)].drop_duplicates(["player", "prop_type", "book"])
    notes, found = {}, []
    for key, kline in keys.items():
        if key not in ids:
            notes[key] = {"note": "key player not on either roster", "teammate_quotes": 0, "stale_quotes": 0, "flags": 0}
            continue
        kpid, kteam = ids[key]
        mates = {n for n, (pid, team) in ids.items() if team == kteam and pid != kpid}
        cur = full[full["book"].isin(SIX_BOOKS) & full["player"].isin(mates)]
        m = cur.merge(ref[["player", "prop_type", "book", "line", "over_odds", "under_odds"]], on=["player", "prop_type", "book"], suffixes=("", "_b"))
        St = m[(m["line"] == m["line_b"]) & (m["over_odds"] == m["over_odds_b"]) & (m["under_odds"] == m["under_odds_b"])][COLS].copy()
        note = {"note": None, "teammate_quotes": int(len(cur)), "stale_quotes": int(len(St)), "flags": 0}
        if len(St):
            St["fair_pre"], St["fair_src"] = fair(St, base, date, event_id)
            St = St[St["fair_pre"].notna()].copy()
        if len(St):
            St["date"], St["player_id"] = date, [ids[n][0] for n in St["player"]]
            St["key"], St["key_team"], St["key_line"], St["key_player"] = kpid, kteam, float(kline), key
            St["shift"] = model(St, box)
            St["fair_pred"] = ndtr(ndtri(np.clip(St["fair_pre"].to_numpy(float), 0.005, 0.995)) + St["shift"].to_numpy(float))
            vo = _dec(St["over_odds"]) * St["fair_pred"].to_numpy() - 1
            vu = _dec(St["under_odds"]) * (1 - St["fair_pred"].to_numpy()) - 1
            St["side"], St["value_pred"] = np.where(vo >= vu, "over", "under"), np.maximum(vo, vu)
            St["quote_age_min"] = (pd.Timestamp(now) - pd.to_datetime(St["last_update"], utc=True)).dt.total_seconds() / 60
            F = St[St["value_pred"] >= EV_MIN]
            note["flags"] = int(len(F.drop_duplicates(["player", "prop_type", "side"])))
            found.append(F)
        notes[key] = note
    F = pd.concat(found, ignore_index=True) if found else pd.DataFrame()
    if len(F):
        F = F.sort_values("value_pred", ascending=False).drop_duplicates(["player", "prop_type", "side"]).reset_index(drop=True)
    return notes, F


def fast_flags(cur: pd.DataFrame, ref: pd.DataFrame, mates: set, sig: dict) -> pd.DataFrame:
    """MEASUREMENT ONLY (did not pass its test: audit section 8m). Teammates' quotes at the six bet books still at their
    reference quote, priced against the books whose quote has changed since (FF_MIN_MOVERS or more): the movers' power-de-vigged
    probabilities, moved to the stale quote's line with the frozen spreads, averaged on the normal scale. Flags at 3%+ on the
    better side; one per player, prop and side, at the best book."""
    import market_lab as ml
    from scipy.stats import norm

    from clean_backtest import SIX_BOOKS, _dec

    key = ["player", "prop_type", "book"]
    c = cur[cur["player"].isin(mates)].drop_duplicates(key)
    r = ref[ref["player"].isin(mates)].drop_duplicates(key)[key + ["line", "over_odds", "under_odds"]]
    j = c.merge(r, on=key, suffixes=("", "_ref"))
    same = (j["line"] == j["line_ref"]) & (j["over_odds"] == j["over_odds_ref"]) & (j["under_odds"] == j["under_odds_ref"])
    movers, stale = j[~same], j[same & j["book"].isin(SIX_BOOKS)]
    if movers.empty or stale.empty:
        return pd.DataFrame()
    mv = movers[key + ["line"]].rename(columns={"book": "book_m", "line": "line_m"})
    mv["z_m"] = norm.ppf(np.clip(ml.devig_power(movers["over_odds"], movers["under_odds"]), 0.005, 0.995))
    m = stale[key + ["line"]].merge(mv, on=["player", "prop_type"])
    m["zc"] = m["z_m"] + (m["line_m"] - m["line"]) / ml.sigma_of(m["prop_type"], m["line"], sig)
    agg = m.groupby(key).agg(z=("zc", "mean"), n_movers=("zc", "size"), movers=("book_m", lambda s: ",".join(sorted(s)))).reset_index()
    F = stale[COLS].merge(agg, on=key)
    F = F[F["n_movers"] >= FF_MIN_MOVERS].copy()
    if F.empty:
        return pd.DataFrame()
    F["fair_ff"] = norm.cdf(F["z"].to_numpy(float))
    vo = _dec(F["over_odds"]) * F["fair_ff"].to_numpy() - 1
    vu = _dec(F["under_odds"]) * (1 - F["fair_ff"].to_numpy()) - 1
    F["side"], F["value_ff"] = np.where(vo >= vu, "over", "under"), np.maximum(vo, vu)
    F = F[F["value_ff"] >= EV_MIN]
    return F.sort_values("value_ff", ascending=False).drop_duplicates(["player", "prop_type", "side"]).reset_index(drop=True)


# =============================================================================
# LIVE
# =============================================================================

def _get(path: str, params: dict):
    from fetch_early_odds import API, _key, get
    return get(f"{API}{path}", {"apiKey": _key(), **params})


def _store(conn, poll_id: int, event_id: str, Q: pd.DataFrame):
    conn.executemany("insert into quotes values (?,?,?,?,?,?,?,?,?)", [(poll_id, event_id, *r) for r in Q[COLS].itertuples(index=False)])


def _fetch(conn, event_id: str, markets: list, kind: str, stamp: str, store: bool = False):
    """One live pull for one game at the ten books. Returns (quotes, or None if the request failed; poll_id)."""
    from clean_backtest import MARKET_TO_PROP
    from fetch_early_odds import BOOKS, SPORT
    from fetch_multibook_odds import parse_all_books

    status, body, charged, balance = _get(f"/sports/{SPORT}/events/{event_id}/odds",
                                          {"bookmakers": ",".join(BOOKS), "markets": ",".join(markets), "oddsFormat": "american"})
    rows = []
    if status == 200 and isinstance(body, dict):
        updated = {(bk.get("key"), m.get("key")): m.get("last_update") or bk.get("last_update")
                   for bk in body.get("bookmakers", []) for m in bk.get("markets", [])}
        rows = [(r[1], MARKET_TO_PROP.get(r[2]), r[3], float(r[4]), int(r[5]), int(r[6]), updated.get((r[1], r[2])))
                for r in parse_all_books(body, event_id, stamp)]
    Q = pd.DataFrame(rows, columns=COLS)
    Q = Q[Q["prop_type"].notna() & (Q["over_odds"].abs() >= 100) & (Q["under_odds"].abs() >= 100)]
    Q = Q.drop_duplicates(["player", "prop_type", "book"]).reset_index(drop=True)  # first quote per player, prop and book, as in the tests
    cur = conn.execute("insert into polls (polled_at, event_id, kind, quotes, credits, balance) values (?,?,?,?,?,?)",
                       (stamp, event_id, kind, len(Q), int(charged or 0), balance))
    if store:
        _store(conn, cur.lastrowid, event_id, Q)
    conn.commit()
    return (Q if status == 200 else None), cur.lastrowid


def _events(now: datetime) -> list:
    """Upcoming games from the free events list, refreshed every 10 minutes."""
    from fetch_early_odds import SPORT
    if _STATE["events_at"] is None or now - _STATE["events_at"] >= EVENTS_EVERY:
        status, body, _, _ = _get(f"/sports/{SPORT}/events", {})
        if status == 200 and isinstance(body, list):
            _STATE["events"] = [{"id": e["id"], "tip": datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")),
                                 "commence_time": e["commence_time"], "home": e.get("home_team"), "away": e.get("away_team")} for e in body]
            _STATE["events_at"] = now
    return _STATE["events"]


def _spent_today(conn, now: datetime) -> int:
    from fetch_early_odds import ET
    start = datetime.combine(now.astimezone(ET).date(), datetime.min.time(), tzinfo=ET).astimezone(timezone.utc)
    return int(conn.execute("select coalesce(sum(credits), 0) from polls where polled_at >= ?", (_stamp(start),)).fetchone()[0])


def _box():
    from clean_backtest import BOX_FILE, load_box
    if not BOX_FILE.exists():
        return None
    mtime = BOX_FILE.stat().st_mtime
    if _STATE["box_mtime"] != mtime:
        _STATE.update(box=load_box(), box_mtime=mtime, model=None, ids={})
    return _STATE["box"]


def _model():
    if _STATE["model"] is None:
        import teammate_model as tm
        _STATE["model"] = tm.load_predictor(MODEL_FILE, META_FILE)
    return _STATE["model"]


def _in_flight(st: dict) -> bool:
    """True when a key player (DraftKings line 15.5+ in the baseline) is missing now at a book that quoted him in the baseline."""
    if st["pts"] is None or st["base"] is None:
        return False
    b = st["base"][st["base"]["prop_type"] == "points"]
    keys = set(b.loc[(b["book"] == "draftkings") & (b["line"] >= KEY_LINE), "player"])
    had = set(zip(b.loc[b["player"].isin(keys), "player"], b.loc[b["player"].isin(keys), "book"]))
    return bool(had - set(zip(st["pts"]["player"], st["pts"]["book"])))


def _sig() -> dict:
    if "sig" not in _STATE:
        import paper_log
        _STATE["sig"] = paper_log._sigma()
    return _STATE["sig"]


def _ff_reference(st: dict, now: datetime):
    """Quotes before the trigger for the fast-book measurement: points from the minute poll about 5 minutes earlier (the
    oldest kept if there is less history), the other markets from the baseline."""
    hist = st.get("pts_hist") or []
    older = [p for t, p in hist if t <= now - FF_REF_LAG]
    pts = older[-1] if older else (hist[0][1] if hist else None)
    parts = []
    if st["base"] is not None:
        parts.append(st["base"] if pts is None else st["base"][st["base"]["prop_type"] != "points"])
    if pts is not None:
        parts.append(pts)
    return pd.concat(parts, ignore_index=True) if parts else None


def _ff_check(conn, e: dict, st: dict, Q: pd.DataFrame, now: datetime, stamp: str):
    """Fast-book flags (measurement only) for each trigger on this game still inside its 30 minutes, from the markets in Q."""
    pulled = _dk_pulled(_had_points(st), Q)
    for w in [w for w in st.get("ff", []) if now - w["t0"] <= FF_HORIZON]:
        ref = w["ref"][w["ref"]["prop_type"].isin(set(Q["prop_type"]))]
        F = fast_flags(Q, ref, w["mates"], _sig())
        mins = (now - w["t0"]).total_seconds() / 60
        for r in F.itertuples(index=False):
            age = (pd.Timestamp(now) - pd.Timestamp(r.last_update)).total_seconds() / 60 if isinstance(r.last_update, str) else None
            mate_pulled = None if pulled is None else int(r.player in pulled)
            cur = conn.execute("insert or ignore into ff_flags (event_id, trigger_id, key_player, player, prop_type, side, book, line, "
                               "over_odds, under_odds, fair_ff, value_ff, n_movers, movers, quote_age_min, flagged_at, mins_after, "
                               "commence_time, date, mate_pulled) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                               (e["id"], w["tid"], w["key"], r.player, r.prop_type, r.side, r.book, float(r.line), int(r.over_odds),
                                int(r.under_odds), float(r.fair_ff), float(r.value_ff), int(r.n_movers), r.movers, age, stamp, mins,
                                e["commence_time"], w["date"], mate_pulled))
            if cur.rowcount:
                price = r.over_odds if r.side == "over" else r.under_odds
                _say(f"FAST    {r.player} {r.prop_type} {r.side} {r.line:g} {price:+d} at {r.book}, {mins:.0f} min after the pull: "
                     f"{r.n_movers} books moved ({r.movers}), their fair {100 * r.fair_ff:.1f}% over; value {100 * r.value_ff:+.1f}% "
                     f"(measurement only, paper)" + (" [he is off DraftKings too]" if mate_pulled else ""))
    conn.commit()


def _track(conn, event_id: str, Q: pd.DataFrame, now: datetime, stamp: str):
    """How long flagged prices last (both rules): for flags made in the last hour on markets this poll covers, record the
    last time the same line and price was seen, or the first time it was not. Skipped where the book quoted no part of that
    market in this poll, since a missing feed says nothing about the price."""
    present = set(zip(Q["book"], Q["prop_type"]))
    idx = Q.drop_duplicates(["player", "prop_type", "book"]).set_index(["player", "prop_type", "book"])
    for table in ("flags", "ff_flags"):
        F = pd.read_sql(f"select rowid as rid, player, prop_type, side, book, line, over_odds, under_odds from {table} "
                        "where event_id=? and gone_at is null and flagged_at >= ?", conn, params=(event_id, _stamp(now - TRACK_FOR)))
        for r in F.itertuples(index=False):
            if (r.book, r.prop_type) not in present:
                continue
            k = (r.player, r.prop_type, r.book)
            same = False
            if k in idx.index:
                q = idx.loc[k]
                same = bool(q["line"] == r.line and (q["over_odds"] == r.over_odds if r.side == "over" else q["under_odds"] == r.under_odds))
            conn.execute(f"update {table} set {'last_same_at' if same else 'gone_at'}=? where rowid=?", (stamp, r.rid))
    conn.commit()


def _measure(what: str, fn, *args, default=None):
    """Run a measurement step; an error there is reported and never stops the frozen rule. Returns fn's result, or default."""
    try:
        return fn(*args)
    except Exception as ex:  # noqa: BLE001
        _say(f"measurement error ({what}): {type(ex).__name__}: {ex}")
        return default


def _had_points(st: dict) -> pd.DataFrame:
    """Points quotes from before this poll: the baseline's and the minute polls' of the last 15 minutes."""
    parts = [p for _, p in st.get("pts_hist") or []]
    if st.get("base") is not None:
        parts.append(st["base"][st["base"]["prop_type"] == "points"])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS)


def _dk_pulled(had: pd.DataFrame, Q: pd.DataFrame):
    """Players DraftKings quoted for points before (had) and does not quote in Q. None when Q holds no DraftKings points board
    (3+ players), since then nothing can be said."""
    now = set(Q.loc[(Q["book"] == "draftkings") & (Q["prop_type"] == "points"), "player"])
    if len(now) < MIN_OTHERS:
        return None
    return set(had.loc[(had["book"] == "draftkings") & (had["prop_type"] == "points"), "player"]) - now


def _off_books(key: str, had: pd.DataFrame, Q: pd.DataFrame) -> list:
    """Books other than DraftKings that quoted the key player's points before (had) and in Q quote 3+ other players' points
    but not his."""
    before = set(had.loc[(had["player"] == key) & (had["prop_type"] == "points") & (had["book"] != "draftkings"), "book"])
    pts = Q[Q["prop_type"] == "points"]
    out = []
    for book in sorted(before):
        names = set(pts.loc[pts["book"] == book, "player"])
        if key not in names and len(names) >= MIN_OTHERS:
            out.append(book)
    return out


def _strict_label(off0, off2: list, back: bool) -> str:
    return "book_at_pull" if off0 else "book_by_2" if off2 else "no" if back else "dk_still_off_2"


def _strict_check(conn, e: dict, st: dict, pts: pd.DataFrame, dk: dict, now: datetime, stamp: str):
    """Strict trigger (measurement only): at the first minute poll 2 minutes or more after each trigger, has DraftKings re-posted
    the key player at any poll since, and which other books have stopped quoting him? Records the result on the trigger row."""
    from fetch_early_odds import ET

    if len(dk) < MIN_OTHERS:
        return  # no DraftKings points board in this poll: wait for the next one
    had = _had_points(st)
    for c in [c for c in st["checks"] if now - c["t0"] >= CHECK_AFTER]:
        st["checks"].remove(c)
        back = c["key"] in st["reappeared"] or c["key"] in dk
        off2 = _off_books(c["key"], had, pts)
        strict = _strict_label(c["off0"], off2, back)
        conn.execute("update triggers set check_at=?, dk_back_2=?, books_off_2=?, strict=? where trigger_id=?",
                     (stamp, int(back), ",".join(off2), strict, c["tid"]))
        _say(f"CHECK   {now.astimezone(ET):%H:%M} ET  {e['away']} @ {e['home']}: {c['key']} {'back on' if back else 'still off'} DraftKings "
             f"{(now - c['t0']).total_seconds() / 60:.0f} min after the pull; off at {len(off2)} other book(s)"
             + (f" ({', '.join(off2)})" if off2 else "")
             + (" -> strict trigger NOT confirmed" if strict == "no" else f" -> strict trigger confirmed: {STRICT_LABEL[strict]}")
             + " (measurement only)")
    conn.commit()


def _trigger(conn, e: dict, st: dict, keys: dict, now: datetime, stamp: str, full=None, poll_id=None, pts_now=None):
    from fetch_early_odds import ET, MARKETS

    had = _had_points(st)  # before this poll: for the strict-trigger measurement only
    off0 = {k: (_measure("strict trigger", _off_books, k, had, pts_now) if pts_now is not None else None) for k in keys}
    if full is None:
        full, poll_id = _fetch(conn, e["id"], MARKETS, "trigger", stamp, store=True)
    else:  # the baseline pull itself showed the pull: keep it as the trigger quotes
        _store(conn, poll_id, e["id"], full)
        conn.execute("update polls set kind='trigger' where poll_id=?", (poll_id,))
    if st["base"] is not None and st.get("base_stored") != st["base_poll"]:
        _store(conn, st["base_poll"], e["id"], st["base"])  # keep the baseline the flags were judged against
        st["base_stored"] = st["base_poll"]
    conn.commit()
    date = e["tip"].astimezone(ET).strftime("%Y-%m-%d")
    box = _box()
    model = _model() if box is not None else None
    problem = ("the trigger pull failed" if full is None else "no baseline yet" if st["base"] is None
               else "no box scores (paper_log grade, or news_monitor grade --refresh-box)" if box is None
               else "teammate model files missing" if model is None else None)
    if problem:
        notes, F = {k: {"note": problem, "teammate_quotes": 0, "stale_quotes": 0, "flags": 0} for k in keys}, pd.DataFrame()
    else:
        notes, F = compute_flags(e["id"], date, (e["home"], e["away"]), keys, full, st["base"], box, model, now)
    tids = {}
    for key, line in keys.items():
        n = notes[key]
        conn.execute("insert or ignore into triggers (event_id, date, home_team, away_team, commence_time, key_player, key_line, dk_last_seen, "
                     "trigger_at, baseline_at, note, teammate_quotes, stale_quotes, flags, books_off_t0) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (e["id"], date, e["home"], e["away"], e["commence_time"], key, float(line), st["dk_at"], stamp,
                      _stamp(st["base_at"]) if st["base_at"] else None, n["note"], n["teammate_quotes"], n["stale_quotes"], n["flags"],
                      None if off0[key] is None else ",".join(off0[key])))
        tids[key] = conn.execute("select trigger_id from triggers where event_id=? and key_player=?", (e["id"], key)).fetchone()[0]
        _say(f"TRIGGER {now.astimezone(ET):%H:%M} ET  {e['away']} @ {e['home']}: DraftKings pulled {key} (points {line:g}); "
             f"{n['teammate_quotes']} teammate quotes at the six books, {n['stale_quotes']} still at the baseline price, {n['flags']} flagged"
             + (f" [{n['note']}]" if n["note"] else "")
             + f"; other books off at the pull: {'unknown' if off0[key] is None else ', '.join(off0[key]) or 'none'}")
        st.setdefault("checks", []).append({"tid": tids[key], "key": key, "t0": now, "off0": off0[key]})
    pulled = _measure("strict trigger", _dk_pulled, had, full) if len(F) else None
    for r in F.itertuples(index=False):
        price = r.over_odds if r.side == "over" else r.under_odds
        mate_pulled = None if pulled is None else int(r.player in pulled)
        conn.execute("insert or ignore into flags (event_id, trigger_id, key_player, player, prop_type, side, book, line, over_odds, under_odds, "
                     "fair_pre, shift, fair_pred, value_pred, quote_age_min, flagged_at, commence_time, date, mate_pulled) "
                     "values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (e["id"], tids[r.key_player], r.key_player, r.player, r.prop_type, r.side, r.book, float(r.line), int(r.over_odds),
                      int(r.under_odds), float(r.fair_pre), float(r.shift), float(r.fair_pred), float(r.value_pred),
                      None if pd.isna(r.quote_age_min) else float(r.quote_age_min), stamp, e["commence_time"], date, mate_pulled))
        _say(f"FLAG    {r.player} {r.prop_type} {r.side} {r.line:g} {price:+d} at {r.book}: fair before the news "
             f"{100 * r.fair_pre:.1f}% over, predicted after {100 * r.fair_pred:.1f}%; value {100 * r.value_pred:+.1f}% (paper only)"
             + (" [he is off DraftKings too]" if mate_pulled else ""))
    conn.commit()
    watches = len(st["ff"])
    if full is not None and box is not None:
        _measure("trigger", _ff_start, conn, e, st, keys, tids, full, box, date, now, stamp)
    if len(F) or len(st["ff"]) > watches:
        st["follow"] += [(now + timedelta(minutes=m), m, sorted(set(tids.values()))) for m in FOLLOW_UPS]


def _ff_start(conn, e: dict, st: dict, keys: dict, tids: dict, full: pd.DataFrame, box: pd.DataFrame, date: str, now: datetime, stamp: str):
    """Open the fast-book measurement for each pulled key player (teammates, reference quotes) and check the trigger pull."""
    ref = _ff_reference(st, now)
    if ref is None:
        return
    ids = identities(box, date, (e["home"], e["away"]), set(full["player"]) | set(ref["player"]) | set(keys))
    for key in keys:
        if key in ids:
            kpid, kteam = ids[key]
            mates = {n for n, (pid, team) in ids.items() if team == kteam and pid != kpid}
            st["ff"].append({"tid": tids[key], "key": key, "mates": mates, "ref": ref, "t0": now, "date": date})
    _ff_check(conn, e, st, full, now, stamp)
    _track(conn, e["id"], full, now, stamp)


def _still(conn, table: str, minutes: int, tids: list, idx: pd.DataFrame):
    F = pd.read_sql(f"select rowid as rid, * from {table} where trigger_id in ({','.join('?' * len(tids))})", conn, params=tids)
    for r in F.itertuples(index=False):
        k = (r.player, r.prop_type, r.book)
        if k not in idx.index:
            status = "gone"
        else:
            q = idx.loc[k]
            now_price, was = (q["over_odds"], r.over_odds) if r.side == "over" else (q["under_odds"], r.under_odds)
            status = ("same" if q["line"] == r.line and now_price == was else
                      f"price {int(now_price):+d}" if q["line"] == r.line else f"line {q['line']:g}")
        conn.execute(f"update {table} set still_{minutes}=? where rowid=?", (status, r.rid))
    conn.commit()


def _follow_up(conn, e: dict, st: dict, minutes: int, tids: list, now: datetime, stamp: str):
    """Is each flagged price still on offer at the same book, line and price? Also the fast-book measurement on these quotes."""
    from fetch_early_odds import MARKETS
    Q, _ = _fetch(conn, e["id"], MARKETS, f"follow_{minutes}", stamp, store=True)
    if Q is None:
        return
    idx = Q.drop_duplicates(["player", "prop_type", "book"]).set_index(["player", "prop_type", "book"])
    _still(conn, "flags", minutes, tids, idx)
    if st is not None:
        _measure(f"follow-up {minutes}", _still, conn, "ff_flags", minutes, tids, idx)
        _measure(f"follow-up {minutes}", _ff_check, conn, e, st, Q, now, stamp)
        _measure(f"follow-up {minutes}", _track, conn, e["id"], Q, now, stamp)


def _watch(conn, e: dict, st: dict, now: datetime, stamp: str):
    from fetch_early_odds import MARKETS
    refresh = st["base"] is None or (now - st["base_at"] >= BASE_EVERY and not _in_flight(st))
    Q, poll_id = _fetch(conn, e["id"], MARKETS if refresh else POINTS, "baseline" if refresh else "points", stamp)
    if Q is None:
        return  # a failed request says nothing about who is quoted
    pts = Q[Q["prop_type"] == "points"]
    dk = dict(zip(pts.loc[pts["book"] == "draftkings", "player"], pts.loc[pts["book"] == "draftkings", "line"]))
    gone = {}
    if st["dk"] is not None and len(dk) >= MIN_OTHERS:
        gone = {p: line for p, line in st["dk"].items() if line >= KEY_LINE and p not in dk and p not in st["keys_done"]}
    if refresh and not gone:
        st.update(base=Q, base_at=now, base_poll=poll_id)
    if gone:
        st["keys_done"].update(gone)
        _trigger(conn, e, st, gone, now, stamp, full=Q if refresh else None, poll_id=poll_id if refresh else None, pts_now=pts)
    for p in st["keys_done"] - st["reappeared"] - set(gone):
        if p in dk:
            st["reappeared"].add(p)
            conn.execute("update triggers set reappeared_at=? where event_id=? and key_player=?", (stamp, e["id"], p))
            conn.commit()
    if st.get("checks"):
        _measure("strict check", _strict_check, conn, e, st, pts, dk, now, stamp)
    if st["ff"]:
        _measure("poll", _ff_check, conn, e, st, Q, now, stamp)
    _measure("poll", _track, conn, e["id"], Q, now, stamp)
    st["pts_hist"] = [(t, p) for t, p in st["pts_hist"] if now - t <= timedelta(minutes=15)] + [(now, pts)]
    st.update(dk=dk, dk_at=stamp, pts=pts)


def tick(now: datetime = None):
    """One pass: poll the games within the window, act on triggers, follow-ups and closing pulls."""
    from fetch_early_odds import ET, MARKETS

    now = now or datetime.now(timezone.utc)
    stamp = _stamp(now)
    conn = _db()
    spent = _spent_today(conn, now)
    if spent >= DAILY_CAP:
        if _STATE.get("capped") != now.astimezone(ET).date():
            _STATE["capped"] = now.astimezone(ET).date()
            _say(f"daily cap reached: {spent:,} credits today (NEWS_MONITOR_DAILY_CAP {DAILY_CAP:,}); no polling until tomorrow, Eastern")
        return
    events = _events(now)
    watching = [e for e in events if now < e["tip"] - STOP_BEFORE_TIP and e["tip"] - now <= WINDOW]
    for e in watching:
        st = _STATE["ev"].setdefault(e["id"], {"dk": None, "dk_at": None, "pts": None, "base": None, "base_at": None, "base_poll": None,
                                               "keys_done": set(), "reappeared": set(), "follow": [], "pts_hist": [], "ff": [],
                                               "checks": []})
        _watch(conn, e, st, now, stamp)
    for e in events:
        st = _STATE["ev"].get(e["id"])
        for due in [f for f in (st["follow"] if st else []) if f[0] <= now]:
            st["follow"].remove(due)
            _follow_up(conn, e, st, due[1], due[2], now, stamp)
        flagged = conn.execute("select 1 from flags where event_id=? union all select 1 from ff_flags where event_id=? limit 1",
                               (e["id"], e["id"])).fetchone()
        if e["tip"] - CLOSE_BEFORE_TIP <= now < e["tip"] and flagged \
                and not conn.execute("select 1 from polls where event_id=? and kind='close' limit 1", (e["id"],)).fetchone():
            _fetch(conn, e["id"], MARKETS, "close", stamp, store=True)
    live = {e["id"] for e in events if e["tip"] > now - timedelta(minutes=30)}
    for eid in [k for k in _STATE["ev"] if k not in live]:
        del _STATE["ev"][eid]
    if watching and (_STATE["beat"] is None or now - _STATE["beat"] >= HEARTBEAT):
        _STATE["beat"] = now
        _say(f"watching {len(watching)} game(s) tipping within {WINDOW.total_seconds() / 3600:g} hours; {spent:,} credits used today")


# =============================================================================
# GRADE AND REPORT
# =============================================================================

def grade(refresh_box: bool = False):
    import market_lab as ml
    from clean_backtest import PROP_TO_STAT, Roster, _dec, _games_table, load_box

    if refresh_box:
        import paper_log
        from clean_backtest import fetch_box
        with paper_log._LOCK:  # the paper log refreshes the same files
            fetch_box(refresh=False)
    conn = _db()
    cutoff = _stamp(datetime.now(timezone.utc) - timedelta(hours=4))
    F = pd.read_sql("select rowid as rid, * from flags where graded_at is null and commence_time < ?", conn, params=(cutoff,))
    FF = pd.read_sql("select rowid as rid, * from ff_flags where graded_at is null and commence_time < ?", conn, params=(cutoff,))
    T = pd.read_sql("select * from triggers where graded_at is null and commence_time < ?", conn, params=(cutoff,))
    if F.empty and FF.empty and T.empty:
        _say("nothing to grade yet")
        return
    box = load_box()
    games = _games_table(box)
    E = pd.read_sql("select event_id, home_team, away_team, date from triggers", conn)
    ev = E[E["event_id"].isin(set(F["event_id"]) | set(FF["event_id"]) | set(T["event_id"]))].drop_duplicates("event_id").rename(columns={"date": "et_date"})
    link = ml._link_events(ev, games)
    graded_at = _stamp(datetime.now(timezone.utc))
    done = {"flags": 0, "ff_flags": 0}
    done_t = 0
    for eid in ev["event_id"]:
        gid = link.get(eid)
        if pd.isna(gid):
            continue
        day = box[box["game_id"] == int(gid)].drop_duplicates("player_id")
        roster = Roster(zip(day["player_id"], day["player"]))
        stats = day.set_index("player_id")
        parts = {"flags": F[F["event_id"] == eid], "ff_flags": FF[FF["event_id"] == eid]}
        t = T[T["event_id"] == eid]
        res = roster.resolve(set(parts["flags"]["player"]) | set(parts["ff_flags"]["player"]) | set(t["key_player"]))
        played = lambda name: bool(stats["played"].get(res[name][0], False)) if name in res else False  # noqa: E731
        for r in t.itertuples(index=False):
            conn.execute("update triggers set key_played=?, graded_at=? where trigger_id=?", (int(played(r.key_player)), graded_at, r.trigger_id))
            done_t += 1
        if parts["flags"].empty and parts["ff_flags"].empty:
            continue
        close = pd.read_sql("select q.* from quotes q where q.poll_id = (select max(poll_id) from polls where kind='close' and event_id=?)",
                            conn, params=(eid,))
        for table, f in parts.items():
            if f.empty:
                continue
            p_close = fair(f[["player", "prop_type", "line", "over_odds", "under_odds"]].assign(book="x"), close, f["date"].iloc[0], eid)[0] \
                if len(close) else np.full(len(f), np.nan)
            for r, pc in zip(f.itertuples(index=False), p_close):
                dec = float(_dec(r.over_odds if r.side == "over" else r.under_odds))
                value_close = None if np.isnan(pc) else dec * (pc if r.side == "over" else 1 - pc) - 1
                ok = played(r.player)
                actual = float(stats.loc[res[r.player][0], PROP_TO_STAT[r.prop_type]].sum()) if ok else None
                won = None if actual is None else int(actual > r.line if r.side == "over" else actual < r.line)
                conn.execute(f"update {table} set close_fair=?, value_close=?, actual=?, played=?, won=?, graded_at=? where rowid=?",
                             (None if np.isnan(pc) else float(pc), value_close, actual, int(ok), won, graded_at, r.rid))
                done[table] += 1
    conn.commit()
    _say(f"graded {done_t} trigger(s), {done['flags']} flag(s) and {done['ff_flags']} fast-book measurement flag(s)")


def _mean_ci(x: pd.Series, cluster: pd.Series) -> str:
    x = x.astype(float)
    if len(x) == 0:
        return "none"
    by = pd.DataFrame({"c": cluster.to_numpy(), "x": x.to_numpy()}).groupby("c")["x"].agg(["sum", "size"])
    g, m = len(by), x.mean()
    ci = 1.96 * np.sqrt(g / (g - 1) * ((by["sum"] - m * by["size"]) ** 2).sum()) / len(x) if g > 1 else float("nan")
    return f"{100 * m:+.1f}% [±{100 * ci:.1f}] on {len(x)}"


def report():
    conn = _db()
    P = pd.read_sql("select kind, count(*) n, coalesce(sum(credits), 0) c, min(polled_at) a, max(polled_at) b from polls group by kind", conn)
    T = pd.read_sql("select * from triggers", conn)
    F = pd.read_sql("select * from flags", conn)
    if P.empty:
        _say("no polls yet")
        return
    _say(f"polls {int(P['n'].sum()):,} ({', '.join(f'{r.kind} {r.n:,}' for r in P.itertuples())}), {P['a'].min()} to {P['b'].max()}; "
         f"credits {int(P['c'].sum()):,}")
    if T.empty:
        _say("no triggers yet")
        return
    g = T[T["key_played"].notna()]
    _say(f"triggers {len(T)} on {T['date'].nunique()} night(s); key player sat out {1 - g['key_played'].mean():.0%} of {len(g)} graded; "
         f"{int(T['reappeared_at'].notna().sum())} re-posted by DraftKings later")
    if F.empty:
        _say("no flags yet")
    else:
        _say(f"flags {len(F)} from {F['trigger_id'].nunique()} trigger(s); predicted value {100 * F['value_pred'].mean():+.1f}%; "
             f"overs {(F['side'] == 'over').mean():.0%}; quote age at the flag median {F['quote_age_min'].median():.1f} min")
        C = F[F["value_close"].notna()]
        _say(f"   value against the closing fair price: {_mean_ci(C['value_close'], C['trigger_id'])}")
        R = F[F["won"].notna()]
        if len(R):
            _say(f"   realised return per bet (voids excluded): {_mean_ci(_pnl(R), R['trigger_id'])}")
        for m in FOLLOW_UPS:
            s = F[f"still_{m}"].dropna()
            if len(s):
                _say(f"   still on offer at the same price about {m} min later: {(s == 'same').mean():.0%} of {len(s)}")
        lasting = _lasting(F)
        if lasting:
            _say(f"   {lasting}")
        for book, b in F.groupby("book"):
            c = b[b["value_close"].notna()]
            _say(f"      {book:<11} {len(b):>4} flags; value at close {_mean_ci(c['value_close'], c['trigger_id'])}")
        _report_nights(T, F)
    _report_fast(conn)
    _report_strict(conn, T)


def _pnl(R: pd.DataFrame) -> pd.Series:
    """Profit per 1-unit bet on graded rows (won 1 or 0)."""
    from clean_backtest import _dec
    return pd.Series(np.where(R["won"] == 1, _dec(np.where(R["side"] == "over", R["over_odds"], R["under_odds"])) - 1, -1.0), index=R.index)


def _record(R: pd.DataFrame) -> str:
    if R.empty:
        return "no graded bets"
    p = _pnl(R)
    return f"{int((R['won'] == 1).sum())}-{int((R['won'] == 0).sum())}, {p.sum():+.2f} units ({100 * p.mean():+.1f}% a bet)"


def _report_nights(T: pd.DataFrame, F: pd.DataFrame, last: int = 7):
    """Each recent game night: triggers, how many pulled players sat out, model flags and their results (1 unit a bet)."""
    _say(f"   by game night (last {last}; model flags, 1 unit a bet):")
    for d in sorted(set(T["date"].dropna()) | set(F["date"].dropna()))[-last:]:
        t, f = T[T["date"] == d], F[F["date"] == d]
        g = t[t["key_played"].notna()]
        c = f["value_close"].dropna()
        _say(f"      {d}  triggers {len(t)}, pulled player sat out {int((g['key_played'] == 0).sum())} of {len(g)} graded; flags {len(f)} "
             f"(void {int((f['played'] == 0).sum())}): {_record(f[f['won'].notna()])}; "
             f"value at close {f'{100 * c.mean():+.1f}% on {len(c)}' if len(c) else 'none'}")


def _lasting(F: pd.DataFrame) -> str:
    """From the minute-by-minute points polls: the share of flagged points prices still on offer, unchanged, k minutes after the flag."""
    P = F[(F["prop_type"] == "points")] if "gone_at" in F.columns else F.iloc[0:0]
    if P.empty:
        return ""
    at = pd.to_datetime(P["flagged_at"], utc=True)
    gone = (pd.to_datetime(P["gone_at"], utc=True) - at).dt.total_seconds() / 60
    seen = (pd.to_datetime(P["last_same_at"], utc=True) - at).dt.total_seconds() / 60
    parts = []
    for k in (1, 2, 3, 5, 10):
        known = gone.notna() | (seen >= k)
        if known.any():
            still = (gone > k) | (gone.isna() & (seen >= k))
            parts.append(f"+{k} min {still[known].mean():.0%} of {int(known.sum())}")
    return "points prices still on offer, minute by minute: " + ", ".join(parts) if parts else ""


def _report_fast(conn):
    FF = pd.read_sql("select * from ff_flags", conn)
    _say("fast-book rule (MEASUREMENT ONLY; did not pass its test, audit section 8m):")
    if FF.empty:
        _say("   no fast-book flags yet")
        return
    _say(f"   flags {len(FF)} from {FF['trigger_id'].nunique()} trigger(s); value against the movers' price {100 * FF['value_ff'].mean():+.1f}%; "
         f"overs {(FF['side'] == 'over').mean():.0%}; minutes after the pull median {FF['mins_after'].median():.0f}; "
         f"quote age median {FF['quote_age_min'].median():.1f} min")
    C = FF[FF["value_close"].notna()]
    _say(f"   value against the closing fair price: {_mean_ci(C['value_close'], C['trigger_id'])}")
    R = FF[FF["won"].notna()]
    if len(R):
        _say(f"   realised return per bet (voids excluded): {_mean_ci(_pnl(R), R['trigger_id'])}")
    for m in FOLLOW_UPS:
        s = FF[f"still_{m}"].dropna()
        if len(s):
            _say(f"   still on offer at the same price at the {m}-minute follow-up: {(s == 'same').mean():.0%} of {len(s)}")
    lasting = _lasting(FF)
    if lasting:
        _say(f"   {lasting}")


def _strict_keep(F: pd.DataFrame, T: pd.DataFrame, table: str) -> pd.DataFrame:
    """Flags the strict trigger keeps: a confirmed trigger; a player DraftKings still quotes; the price still on offer when the
    trigger was confirmed (at the pull if another book was already off, otherwise the 2-minute follow-up showed the same line
    and price; a fast-book flag made at or after the check also counts)."""
    m = F.merge(T[["trigger_id", "strict", "check_at"]], on="trigger_id", how="left")
    ok = m["strict"].isin([k for k in STRICT_LABEL if k != "no"]) & (m["mate_pulled"].fillna(0) != 1)
    on_offer = (m["strict"] == "book_at_pull") | (m["still_2"] == "same")
    if table == "ff_flags":
        on_offer = on_offer | (m["flagged_at"].fillna("") >= m["check_at"].fillna("~"))
    return m[ok & on_offer]


def _report_strict(conn, T: pd.DataFrame):
    _say("strict trigger (MEASUREMENT ONLY from 2026-11-15; the rule above is unchanged; protocol_strict_trigger.json):")
    C = T[T["check_at"].notna()] if len(T) else T
    if C.empty:
        _say(f"   no checked triggers yet ({len(T)} older or unchecked)")
        return
    conf = C["strict"] != "no"
    _say(f"   triggers checked {len(C)}: confirmed {int(conf.sum())} ("
         + ", ".join(f"{STRICT_LABEL[k]} {int((C['strict'] == k).sum())}" for k in STRICT_LABEL if k != "no")
         + f"); not confirmed {int((~conf).sum())}; not checked {len(T) - len(C)}")
    parts = []
    for lab, sub in [("confirmed", C[conf]), ("not confirmed", C[~conf])] + [(STRICT_LABEL[k], C[C["strict"] == k]) for k in STRICT_LABEL if k != "no"]:
        g = sub[sub["key_played"].notna()]
        parts.append(f"{lab} {int((g['key_played'] == 0).sum())} of {len(g)}")
    _say("   pulled player sat out (graded): " + "; ".join(parts))
    for table, label in (("flags", "model flags"), ("ff_flags", "fast-book flags (measurement only)")):
        F = pd.read_sql(f"select * from {table}", conn)
        F = F[F["trigger_id"].isin(set(C["trigger_id"]))]
        if F.empty:
            _say(f"   {label}: none from checked triggers")
            continue
        K = _strict_keep(F, C, table)
        m = F.merge(C[["trigger_id", "strict"]], on="trigger_id")
        unconf = m["strict"] == "no"
        mate = ~unconf & (m["mate_pulled"] == 1)
        _say(f"   {label}: kept {len(K)} of {len(F)} from checked triggers; dropped: trigger not confirmed {int(unconf.sum())}, "
             f"on a player DraftKings had pulled {int(mate.sum())}, price gone by the confirmation {len(F) - len(K) - int(unconf.sum()) - int(mate.sum())}")
        for lab, sub in (("kept", K), (f"all {len(F)}", F)):
            c = sub[sub["value_close"].notna()]
            _say(f"      {lab:<8} value against the closing fair price {_mean_ci(c['value_close'], c['trigger_id'])}; "
                 f"results {_record(sub[sub['won'].notna()])}")


def recent(n: int = 50) -> dict:
    """The latest triggers and flags, newest first, for the web app."""
    conn = _db()
    T = pd.read_sql("select trigger_id, date, trigger_at, home_team, away_team, key_player, key_line, stale_quotes, flags, note, reappeared_at, "
                    "key_played, books_off_t0, check_at, dk_back_2, books_off_2, strict from triggers order by trigger_at desc limit ?",
                    conn, params=(n,))
    F = pd.read_sql("select flagged_at, player, prop_type, side, book, line, over_odds, under_odds, fair_pre, fair_pred, value_pred, still_2, "
                    "still_5, still_10, gone_at, value_close, actual, played, won, key_player, mate_pulled from flags "
                    "order by flagged_at desc limit ?", conn, params=(n,))
    FF = pd.read_sql("select flagged_at, mins_after, player, prop_type, side, book, line, over_odds, under_odds, n_movers, movers, fair_ff, "
                     "value_ff, still_2, still_5, still_10, gone_at, value_close, actual, played, won, key_player, mate_pulled from ff_flags "
                     "order by flagged_at desc limit ?", conn, params=(n,))
    return {"triggers": json.loads(T.to_json(orient="records")), "flags": json.loads(F.to_json(orient="records")),
            "fast_book_measurement_flags": json.loads(FF.to_json(orient="records"))}


def scheduled(cmd: str) -> str:
    """Run one step and return what it printed (the web app's scheduler and /api/news-monitor use this). SystemExit from
    a failed API call or a missing key is caught: inside the web app it would otherwise shut the server down."""
    buf = io.StringIO()
    with _LOCK:
        _OUT.buf = buf
        try:
            if cmd == "tick":
                tick()
            elif cmd == "grade":  # the grade and then the full report, so the web app's log holds every result
                grade(False)
                report()
            elif cmd == "report":
                report()
        except SystemExit as e:
            _say(f"{cmd} stopped: {e}")
        finally:
            _OUT.buf = None
    return buf.getvalue()


# =============================================================================
# REPLAY (offline check against the bought snapshots)
# =============================================================================

def replay():
    """The live trigger and flag code over the bought 5-minute snapshots: the 2025-26 pilot (news_lab.py) and sample B, the 2025-26
    late scratches (news_confirm.py). Differences from those reports, all from running as live: the trigger is a key player quoted
    in the previous snapshot and missing now; the baseline is the window's first bought snapshot (not the hourly one 5 minutes
    earlier); rosters come from games before the date (not that game's box score). Flags are scored as in news_confirm.py,
    against the window's last bought snapshot and the box score."""
    import news_confirm as nc
    import news_lab as nl
    import teammate_model as tm
    from clean_backtest import MARKET_TO_PROP, PROP_TO_STAT, Roster, _dec, load_box

    box = load_box()
    model = tm.load_predictor(MODEL_FILE, META_FILE)
    team_name = dict(zip(box["team_id"], box["team"]))
    rng = np.random.default_rng(19)
    sig = _sig()

    def scored(F):
        F = F[F["fair_post"].notna()].copy()
        F["dec"] = np.where(F["side"] == "over", _dec(F["over_odds"]), _dec(F["under_odds"]))
        F["value"] = F["dec"] * np.where(F["side"] == "over", F["fair_post"], 1 - F["fair_post"]) - 1
        F["played"], F["won"] = False, False
        for gid, g in F.groupby("game_id"):
            day = box[box["game_id"] == gid].drop_duplicates("player_id")
            res = Roster(zip(day["player_id"], day["player"])).resolve(set(g["player"]))
            stats = day.set_index("player_id")
            for i, r in g.iterrows():
                pid = res.get(r["player"], (None,))[0]
                if pid is not None and bool(stats["played"].get(pid, False)):
                    actual = float(stats.loc[pid, PROP_TO_STAT[r["prop_type"]]].sum())
                    F.at[i, "played"], F.at[i, "won"] = True, (actual > r["line"]) if r["side"] == "over" else (actual < r["line"])
        return F

    for label, events_file, db_file, keep in [("pilot 2025-26", nl.EVENTS_FILE, nl.DB, lambda e: True),
                                              ("late scratches 2025-26 (sample B)", nc.EVENTS_FILE, nc.DB, lambda e: e.get("sample", "").startswith("B"))]:
        events = [e for e in json.loads(events_file.read_text()) if keep(e)]
        conn = sqlite3.connect(str(db_file))
        R = pd.read_sql("select event_id, requested, bookmaker as book, market, player_name as player, line, over_odds, under_odds from props", conn)
        conn.close()
        R["prop_type"] = R["market"].map(MARKET_TO_PROP)
        R = R[R["prop_type"].notna() & (R["over_odds"].abs() >= 100) & (R["under_odds"].abs() >= 100)].assign(last_update=None)
        out, trig, ff_out = [], [], []
        for e in events:
            G = R[R["event_id"] == e["event_id"]]
            day = box[box["game_id"] == e["game_id"]]
            snaps = sorted(G["requested"].unique())
            if len(snaps) < 3 or day.empty:
                continue
            date = day["date"].iloc[0]
            teams = (team_name[int(day["home_team_id"].iloc[0])], team_name[int(day["visitor_team_id"].iloc[0])])
            at = {s: G[G["requested"] == s].drop_duplicates(["player", "prop_type", "book"])[COLS] for s in snaps}
            base = at[snaps[0]]
            dk_prev = dict(zip(base.loc[(base["book"] == "draftkings") & (base["prop_type"] == "points"), "player"],
                               base.loc[(base["book"] == "draftkings") & (base["prop_type"] == "points"), "line"]))
            done = set()
            for s in snaps[1:]:
                Q = at[s]
                dk = dict(zip(Q.loc[(Q["book"] == "draftkings") & (Q["prop_type"] == "points"), "player"],
                              Q.loc[(Q["book"] == "draftkings") & (Q["prop_type"] == "points"), "line"]))
                gone = {p: v for p, v in dk_prev.items() if v >= KEY_LINE and p not in dk and p not in done} if len(dk) >= MIN_OTHERS else {}
                if gone:
                    done |= set(gone)
                    later = [at[t] for t in snaps if t > s]
                    back = {p: any(((L["book"] == "draftkings") & (L["prop_type"] == "points") & (L["player"] == p)).any() for L in later) for p in gone}
                    trig.extend(back.values())
                    _, F = compute_flags(e["event_id"], date, teams, gone, Q, base, box, model, pd.Timestamp(s).to_pydatetime())
                    post = at[snaps[-1]]
                    if len(F):
                        F["fair_post"] = fair(F, post, date, e["event_id"])[0]
                        F["game_id"], F["reposted"] = e["game_id"], F["key_player"].map(back)
                        out.append(F)
                    # fast-book measurement: reference = the snapshot before the trigger; checks for 30 minutes
                    i = snaps.index(s)
                    ref = at[snaps[i - 1]]
                    ids = identities(box, date, teams, set(Q["player"]) | set(ref["player"]) | set(gone))
                    found = []
                    for key in [k for k in gone if k in ids]:
                        kpid, kteam = ids[key]
                        mates = {n for n, (pid, team) in ids.items() if team == kteam and pid != kpid}
                        for j in range(i, len(snaps)):
                            mins = (pd.Timestamp(snaps[j]) - pd.Timestamp(s)).total_seconds() / 60
                            if mins > FF_HORIZON.total_seconds() / 60:
                                break
                            Fj = fast_flags(at[snaps[j]], ref, mates, sig)
                            if Fj.empty:
                                continue
                            if j + 1 < len(snaps) and pd.Timestamp(snaps[j + 1]) - pd.Timestamp(snaps[j]) == pd.Timedelta(minutes=5):
                                nxt = at[snaps[j + 1]][["player", "prop_type", "book", "line", "over_odds", "under_odds"]]
                                Fj = Fj.merge(nxt, on=["player", "prop_type", "book"], how="left", suffixes=("", "_n"))
                                Fj["still_5"] = ((Fj["line_n"] == Fj["line"]) & (Fj["over_odds_n"] == Fj["over_odds"])
                                                 & (Fj["under_odds_n"] == Fj["under_odds"])).astype(float)
                            else:
                                Fj["still_5"] = np.nan
                            found.append(Fj.assign(mins=mins, t=snaps[j], key_player=key, reposted=back[key]))
                    if found:
                        G2 = pd.concat(found, ignore_index=True).sort_values(["t", "value_ff"], ascending=[True, False])
                        G2 = G2.drop_duplicates(["player", "prop_type", "side"]).reset_index(drop=True)
                        G2["fair_post"], G2["game_id"] = fair(G2, post, date, e["event_id"])[0], e["game_id"]
                        ff_out.append(G2)
                dk_prev = dk
        print(f"\n{label}: {len(events)} windows, {len(trig)} triggers, of which DraftKings re-posted the player later in the window: {sum(trig)}")
        F = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
        if F.empty:
            print("   no flags")
        else:
            F = scored(F)
            print(f"   flagged at predicted value 3%+   {nc.fmt(nc.score(F, rng))}")
            for lab, sub in [("player stayed pulled", F[~F["reposted"]]), ("player re-posted later", F[F["reposted"]])]:
                print(f"      {lab:<28} {nc.fmt(nc.score(sub, rng))}")
            print(f"   by prop: " + ", ".join(f"{p} {len(g)} ({100 * g['value'].mean():+.1f}%)" for p, g in F.groupby("prop_type")))
            print(f"   by book: " + ", ".join(f"{b} {len(g)} ({100 * g['value'].mean():+.1f}%)" for b, g in F.groupby("book")))
        FF = pd.concat(ff_out, ignore_index=True) if ff_out else pd.DataFrame()
        if FF.empty:
            print("   fast-book measurement: no flags")
            continue
        FF = scored(FF.drop_duplicates(["game_id", "player", "prop_type", "side"]))
        st5 = FF["still_5"].dropna()
        print(f"   fast-book measurement (2+ movers, 3%+)  {nc.fmt(nc.score(FF, rng))}")
        print(f"      still offered 5 minutes later {st5.mean():.0%} of {len(st5)}; at the pull itself {int((FF['mins'] == 0).sum())} flags; "
              f"model flags at the same teammate, prop and side {int(FF.merge(F[['game_id', 'player', 'prop_type', 'side']], how='inner').shape[0]) if len(F) else 0}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--minutes", type=float, default=240)
    g = sub.add_parser("grade")
    g.add_argument("--refresh-box", action="store_true")
    sub.add_parser("report")
    sub.add_parser("replay")
    a = ap.parse_args()
    if a.cmd == "run":
        end = time.time() + a.minutes * 60
        while time.time() < end:
            t0 = time.time()
            out = scheduled("tick")
            if out.strip():
                print(out.strip(), flush=True)
            time.sleep(max(5.0, POLL_S - (time.time() - t0)))
    elif a.cmd == "grade":
        grade(a.refresh_box)
        report()
    elif a.cmd == "report":
        report()
    else:
        replay()
