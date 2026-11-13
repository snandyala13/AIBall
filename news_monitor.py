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

Polling, for games tipping within 3 hours (the events list is free):
  every minute            player_points at the ten books: 1 credit a game
  every 30 minutes        all five markets at the ten books, 5 credits a game: the baseline for the stale test and the
                          pre-news fair price; not refreshed while a key player is missing at a book that had him
  at a trigger            all five markets at once: the quotes the flags use; again about 2 and 5 minutes later, to see
                          how long each flagged price stays on offer
  10 minutes before tip   all five markets once, for games with flags: the closing fair price used in grading
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
FOLLOW_UPS = (2, 5)
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
    """)
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


def _trigger(conn, e: dict, st: dict, keys: dict, now: datetime, stamp: str, full=None, poll_id=None):
    from fetch_early_odds import ET, MARKETS

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
                     "trigger_at, baseline_at, note, teammate_quotes, stale_quotes, flags) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (e["id"], date, e["home"], e["away"], e["commence_time"], key, float(line), st["dk_at"], stamp,
                      _stamp(st["base_at"]) if st["base_at"] else None, n["note"], n["teammate_quotes"], n["stale_quotes"], n["flags"]))
        tids[key] = conn.execute("select trigger_id from triggers where event_id=? and key_player=?", (e["id"], key)).fetchone()[0]
        _say(f"TRIGGER {now.astimezone(ET):%H:%M} ET  {e['away']} @ {e['home']}: DraftKings pulled {key} (points {line:g}); "
             f"{n['teammate_quotes']} teammate quotes at the six books, {n['stale_quotes']} still at the baseline price, {n['flags']} flagged"
             + (f" [{n['note']}]" if n["note"] else ""))
    for r in F.itertuples(index=False):
        price = r.over_odds if r.side == "over" else r.under_odds
        conn.execute("insert or ignore into flags (event_id, trigger_id, key_player, player, prop_type, side, book, line, over_odds, under_odds, "
                     "fair_pre, shift, fair_pred, value_pred, quote_age_min, flagged_at, commence_time, date) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (e["id"], tids[r.key_player], r.key_player, r.player, r.prop_type, r.side, r.book, float(r.line), int(r.over_odds),
                      int(r.under_odds), float(r.fair_pre), float(r.shift), float(r.fair_pred), float(r.value_pred),
                      None if pd.isna(r.quote_age_min) else float(r.quote_age_min), stamp, e["commence_time"], date))
        _say(f"FLAG    {r.player} {r.prop_type} {r.side} {r.line:g} {price:+d} at {r.book}: fair before the news "
             f"{100 * r.fair_pre:.1f}% over, predicted after {100 * r.fair_pred:.1f}%; value {100 * r.value_pred:+.1f}% (paper only)")
    conn.commit()
    if len(F):
        st["follow"] += [(now + timedelta(minutes=m), m, sorted(set(tids.values()))) for m in FOLLOW_UPS]


def _follow_up(conn, e: dict, minutes: int, tids: list, stamp: str):
    """Is each flagged price still on offer at the same book, line and price?"""
    from fetch_early_odds import MARKETS
    Q, _ = _fetch(conn, e["id"], MARKETS, f"follow_{minutes}", stamp, store=True)
    if Q is None:
        return
    idx = Q.set_index(["player", "prop_type", "book"])
    F = pd.read_sql(f"select rowid as rid, * from flags where trigger_id in ({','.join('?' * len(tids))})", conn, params=tids)
    for r in F.itertuples(index=False):
        k = (r.player, r.prop_type, r.book)
        if k not in idx.index:
            status = "gone"
        else:
            q = idx.loc[k]
            now_price, was = (q["over_odds"], r.over_odds) if r.side == "over" else (q["under_odds"], r.under_odds)
            status = ("same" if q["line"] == r.line and now_price == was else
                      f"price {int(now_price):+d}" if q["line"] == r.line else f"line {q['line']:g}")
        conn.execute(f"update flags set still_{minutes}=? where rowid=?", (status, r.rid))
    conn.commit()


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
        _trigger(conn, e, st, gone, now, stamp, full=Q if refresh else None, poll_id=poll_id if refresh else None)
    for p in st["keys_done"] - st["reappeared"] - set(gone):
        if p in dk:
            st["reappeared"].add(p)
            conn.execute("update triggers set reappeared_at=? where event_id=? and key_player=?", (stamp, e["id"], p))
            conn.commit()
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
                                               "keys_done": set(), "reappeared": set(), "follow": []})
        _watch(conn, e, st, now, stamp)
    for e in events:
        st = _STATE["ev"].get(e["id"])
        for due in [f for f in (st["follow"] if st else []) if f[0] <= now]:
            st["follow"].remove(due)
            _follow_up(conn, e, due[1], due[2], stamp)
        if e["tip"] - CLOSE_BEFORE_TIP <= now < e["tip"] \
                and conn.execute("select 1 from flags where event_id=? limit 1", (e["id"],)).fetchone() \
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
    T = pd.read_sql("select * from triggers where graded_at is null and commence_time < ?", conn, params=(cutoff,))
    if F.empty and T.empty:
        _say("nothing to grade yet")
        return
    box = load_box()
    games = _games_table(box)
    E = pd.read_sql("select event_id, home_team, away_team, date from triggers", conn)
    ev = E[E["event_id"].isin(set(F["event_id"]) | set(T["event_id"]))].drop_duplicates("event_id").rename(columns={"date": "et_date"})
    link = ml._link_events(ev, games)
    graded_at = _stamp(datetime.now(timezone.utc))
    done_f = done_t = 0
    for eid in ev["event_id"]:
        gid = link.get(eid)
        if pd.isna(gid):
            continue
        day = box[box["game_id"] == int(gid)].drop_duplicates("player_id")
        roster = Roster(zip(day["player_id"], day["player"]))
        stats = day.set_index("player_id")
        f = F[F["event_id"] == eid]
        t = T[T["event_id"] == eid]
        res = roster.resolve(set(f["player"]) | set(t["key_player"]))
        played = lambda name: bool(stats["played"].get(res[name][0], False)) if name in res else False  # noqa: E731
        for r in t.itertuples(index=False):
            conn.execute("update triggers set key_played=?, graded_at=? where trigger_id=?", (int(played(r.key_player)), graded_at, r.trigger_id))
            done_t += 1
        if f.empty:
            continue
        close = pd.read_sql("select q.* from quotes q where q.poll_id = (select max(poll_id) from polls where kind='close' and event_id=?)",
                            conn, params=(eid,))
        p_close = fair(f[["player", "prop_type", "line", "over_odds", "under_odds"]].assign(book="x"), close, f["date"].iloc[0], eid)[0] \
            if len(close) else np.full(len(f), np.nan)
        for r, pc in zip(f.itertuples(index=False), p_close):
            dec = float(_dec(r.over_odds if r.side == "over" else r.under_odds))
            value_close = None if np.isnan(pc) else dec * (pc if r.side == "over" else 1 - pc) - 1
            ok = played(r.player)
            actual = float(stats.loc[res[r.player][0], PROP_TO_STAT[r.prop_type]].sum()) if ok else None
            won = None if actual is None else int(actual > r.line if r.side == "over" else actual < r.line)
            conn.execute("update flags set close_fair=?, value_close=?, actual=?, played=?, won=?, graded_at=? where rowid=?",
                         (None if np.isnan(pc) else float(pc), value_close, actual, int(ok), won, graded_at, r.rid))
            done_f += 1
    conn.commit()
    _say(f"graded {done_t} trigger(s) and {done_f} flag(s)")


def _mean_ci(x: pd.Series, cluster: pd.Series) -> str:
    x = x.astype(float)
    if len(x) == 0:
        return "none"
    by = pd.DataFrame({"c": cluster.to_numpy(), "x": x.to_numpy()}).groupby("c")["x"].agg(["sum", "size"])
    g, m = len(by), x.mean()
    ci = 1.96 * np.sqrt(g / (g - 1) * ((by["sum"] - m * by["size"]) ** 2).sum()) / len(x) if g > 1 else float("nan")
    return f"{100 * m:+.1f}% [±{100 * ci:.1f}] on {len(x)}"


def report():
    from clean_backtest import _dec

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
        return
    _say(f"flags {len(F)} from {F['trigger_id'].nunique()} trigger(s); predicted value {100 * F['value_pred'].mean():+.1f}%; "
         f"overs {(F['side'] == 'over').mean():.0%}; quote age at the flag median {F['quote_age_min'].median():.1f} min")
    C = F[F["value_close"].notna()]
    _say(f"   value against the closing fair price: {_mean_ci(C['value_close'], C['trigger_id'])}")
    R = F[F["won"].notna()].copy()
    if len(R):
        R["pnl"] = np.where(R["won"] == 1, _dec(np.where(R["side"] == "over", R["over_odds"], R["under_odds"])) - 1, -1.0)
        _say(f"   realised return per bet (voids excluded): {_mean_ci(R['pnl'], R['trigger_id'])}")
    for m in FOLLOW_UPS:
        s = F[f"still_{m}"].dropna()
        if len(s):
            _say(f"   still on offer at the same price about {m} min later: {(s == 'same').mean():.0%} of {len(s)}")
    for book, b in F.groupby("book"):
        c = b[b["value_close"].notna()]
        _say(f"      {book:<11} {len(b):>4} flags; value at close {_mean_ci(c['value_close'], c['trigger_id'])}")


def recent(n: int = 50) -> dict:
    """The latest triggers and flags, newest first, for the web app."""
    conn = _db()
    T = pd.read_sql("select trigger_id, date, trigger_at, home_team, away_team, key_player, key_line, stale_quotes, flags, note, reappeared_at, "
                    "key_played from triggers order by trigger_at desc limit ?", conn, params=(n,))
    F = pd.read_sql("select flagged_at, player, prop_type, side, book, line, over_odds, under_odds, fair_pre, fair_pred, value_pred, still_2, "
                    "still_5, value_close, actual, played, won, key_player from flags order by flagged_at desc limit ?", conn, params=(n,))
    return {"triggers": json.loads(T.to_json(orient="records")), "flags": json.loads(F.to_json(orient="records"))}


def scheduled(cmd: str) -> str:
    """Run one step and return what it printed (the web app's scheduler and /api/news-monitor use this). SystemExit from
    a failed API call or a missing key is caught: inside the web app it would otherwise shut the server down."""
    buf = io.StringIO()
    with _LOCK:
        _OUT.buf = buf
        try:
            if cmd == "tick":
                tick()
            elif cmd == "grade":
                grade(False)
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
    for label, events_file, db_file, keep in [("pilot 2025-26", nl.EVENTS_FILE, nl.DB, lambda e: True),
                                              ("late scratches 2025-26 (sample B)", nc.EVENTS_FILE, nc.DB, lambda e: e.get("sample", "").startswith("B"))]:
        events = [e for e in json.loads(events_file.read_text()) if keep(e)]
        conn = sqlite3.connect(str(db_file))
        R = pd.read_sql("select event_id, requested, bookmaker as book, market, player_name as player, line, over_odds, under_odds from props", conn)
        conn.close()
        R["prop_type"] = R["market"].map(MARKET_TO_PROP)
        R = R[R["prop_type"].notna() & (R["over_odds"].abs() >= 100) & (R["under_odds"].abs() >= 100)].assign(last_update=None)
        out, trig = [], []
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
                    if len(F):
                        post = at[snaps[-1]]
                        F["fair_post"] = fair(F, post, date, e["event_id"])[0]
                        F["game_id"], F["reposted"] = e["game_id"], F["key_player"].map(back)
                        out.append(F)
                dk_prev = dk
        F = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
        print(f"\n{label}: {len(events)} windows, {len(trig)} triggers, of which DraftKings re-posted the player later in the window: {sum(trig)}")
        if F.empty:
            print("   no flags")
            continue
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
        print(f"   flagged at predicted value 3%+   {nc.fmt(nc.score(F, rng))}")
        for lab, sub in [("player stayed pulled", F[~F["reposted"]]), ("player re-posted later", F[F["reposted"]])]:
            print(f"      {lab:<28} {nc.fmt(nc.score(sub, rng))}")
        print(f"   by prop: " + ", ".join(f"{p} {len(g)} ({100 * g['value'].mean():+.1f}%)" for p, g in F.groupby("prop_type")))
        print(f"   by book: " + ", ".join(f"{b} {len(g)} ({100 * g['value'].mean():+.1f}%)" for b, g in F.groupby("book")))


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
