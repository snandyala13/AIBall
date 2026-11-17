#!/usr/bin/env python3
"""
paper_log.py — forward paper log for the unders-versus-fair-price screen. No bets are placed.

Each poll pulls current prices for upcoming NBA games from The Odds API (ten books including Pinnacle, five
markets; 5 credits a game), flags unders at the six books priced 2% or more above the fair price, and records
each flag once, when it is first seen. The fair price is rule S1 from market_lab.py, unchanged: Pinnacle's
de-vigged probability moved to the book's line, otherwise the average of three or more other books.

    python paper_log.py poll [--hours-ahead 30]     # pull prices now and log new flags
    python paper_log.py grade [--refresh-box]       # after games finish: results, and the last price seen before tip
    python paper_log.py report

The report also measures, from the quotes already stored (no API calls), each graded flag's value against the fair price at
the last poll before tip, and whether the next poll still showed the flagged price (added 2026-11-17 for the reading rule).

On the deployed web app the same steps run on a schedule (web/app.py, _paper_log_scheduler): a poll every 30 minutes from
9:00 to 21:30 Eastern while the server is up, and a grade once a day. There the log lives on the shared EFS volume.
A local dev server does not run the schedule unless PAPER_LOG_ENABLED=1.
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

BASE_DIR = Path(__file__).parent
_EFS = os.environ.get("EFS_MOUNT")
LOG_DB = Path(os.environ.get("PAPER_LOG_DB") or (Path(_EFS) / "paper_log" / "paper_log.db" if _EFS else BASE_DIR / "data" / "paper_log.db"))
SIGMA_FILE = BASE_DIR / "paper_log_sigma.json"
_LOCK = threading.Lock()
_OUT = threading.local()
EV_MIN = 0.02
SIGMA_SEASONS = [2023, 2024, 2025]


def _say(*args):
    """print, except that inside scheduled() it writes to that call's buffer. Only the calling thread's output is captured;
    contextlib.redirect_stdout would swap stdout for every thread of the web app while a step runs."""
    print(*args, file=getattr(_OUT, "buf", None) or sys.stdout)


def _db() -> sqlite3.Connection:
    LOG_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(LOG_DB), timeout=30)
    conn.execute("create table if not exists polls (poll_id integer primary key autoincrement, polled_at text, events integer, "
                 "quotes integer, credits integer, balance text)")
    conn.execute("create table if not exists quotes (poll_id integer, event_id text, book text, prop_type text, player text, line real, "
                 "over_odds integer, under_odds integer, last_update text)")
    conn.execute("create index if not exists idx_quotes_event on quotes(event_id, player, prop_type, book)")
    conn.execute("""create table if not exists flags (
        event_id text, player text, prop_type text, date text, commence_time text, home_team text, away_team text,
        first_seen text, book text, line real, under_odds integer, fair_under real, ev real, ref text, quote_age_min real,
        last_seen text, times_seen integer, last_ev real,
        close_under_odds integer, close_seen text, actual real, played integer, won integer, graded_at text,
        primary key (event_id, player, prop_type))""")
    conn.commit()
    return conn


def _sigma() -> dict:
    """Line-conversion spreads per prop (market_lab.fit_sigma on 2023-24 to 2025-26), frozen so the job needs no history files."""
    return {k: tuple(v) for k, v in json.loads(SIGMA_FILE.read_text())["params"].items()}


def poll(hours_ahead: float):
    import market_lab as ml
    from clean_backtest import SIX_BOOKS, MARKET_TO_PROP, _dec, norm_name
    from fetch_early_odds import API, BOOKS, DELAY, ET, MARKETS, SPORT, _key, get
    from fetch_multibook_odds import parse_all_books

    key, conn = _key(), _db()
    now = datetime.now(timezone.utc)
    status, events, _, _ = get(f"{API}/sports/{SPORT}/events", {"apiKey": key})
    if status != 200 or not isinstance(events, list):
        sys.exit(f"could not list events (http {status})")
    upcoming = []
    for e in events:
        tip = datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
        if now + timedelta(minutes=10) < tip < now + timedelta(hours=hours_ahead):
            upcoming.append((e, tip))
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    rows, spent, balance = [], 0, None
    for e, tip in upcoming:
        status, body, charged, balance = get(f"{API}/sports/{SPORT}/events/{e['id']}/odds",
                                             {"apiKey": key, "bookmakers": ",".join(BOOKS), "markets": ",".join(MARKETS), "oddsFormat": "american"})
        spent += charged
        if status == 200 and isinstance(body, dict):
            updated = {(bk.get("key"), m.get("key")): m.get("last_update") or bk.get("last_update")
                       for bk in body.get("bookmakers", []) for m in bk.get("markets", [])}
            for r in parse_all_books(body, e["id"], stamp):
                rows.append((e["id"], r[1], MARKET_TO_PROP.get(r[2]), r[3], float(r[4]), int(r[5]), int(r[6]), updated.get((r[1], r[2]))))
        time.sleep(DELAY)
    Q = pd.DataFrame(rows, columns=["event_id", "book", "prop_type", "player", "line", "over_odds", "under_odds", "last_update"])
    Q = Q[Q["prop_type"].notna() & (Q["over_odds"].abs() >= 100) & (Q["under_odds"].abs() >= 100)]
    cur = conn.execute("insert into polls (polled_at, events, quotes, credits, balance) values (?,?,?,?,?)", (stamp, len(upcoming), len(Q), spent, balance))
    poll_id = cur.lastrowid
    conn.executemany("insert into quotes values (?,?,?,?,?,?,?,?,?)", [(poll_id, *r) for r in Q.itertuples(index=False)])
    _say(f"{stamp}: {len(upcoming)} games tipping within {hours_ahead:g} hours, {len(Q):,} quotes, {spent} credits (balance {balance})")
    if Q.empty:
        conn.commit()
        return

    info = {e["id"]: (e, tip) for e, tip in upcoming}
    Q["date"] = [info[i][1].astimezone(ET).strftime("%Y-%m-%d") for i in Q["event_id"]]
    Q["player_id"] = Q["event_id"] + "|" + Q["player"].map(norm_name)
    Q = Q.drop_duplicates(["player_id", "prop_type", "book"])
    C = ml.add_consensus(Q, _sigma())
    C = C[C["book"].isin(SIX_BOOKS) & C["p_s1"].notna()].copy()
    C["dec"] = _dec(C["under_odds"])
    C["ev"] = C["dec"] * (1 - C["p_s1"]) - 1
    C["age"] = (now - pd.to_datetime(C["last_update"], utc=True)).dt.total_seconds() / 60
    best = C.sort_values("ev", ascending=False).drop_duplicates(["player_id", "prop_type"])
    flagged = best[best["ev"] >= EV_MIN]
    new = 0
    for r in flagged.itertuples(index=False):
        e, tip = info[r.event_id]
        seen = conn.execute("select 1 from flags where event_id=? and player=? and prop_type=?", (r.event_id, r.player, r.prop_type)).fetchone()
        if seen:
            conn.execute("update flags set last_seen=?, times_seen=times_seen+1, last_ev=? where event_id=? and player=? and prop_type=?",
                         (stamp, float(r.ev), r.event_id, r.player, r.prop_type))
        else:
            new += 1
            conn.execute("insert into flags (event_id, player, prop_type, date, commence_time, home_team, away_team, first_seen, book, line, under_odds, "
                         "fair_under, ev, ref, quote_age_min, last_seen, times_seen, last_ev) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (r.event_id, r.player, r.prop_type, r.date, e["commence_time"], e.get("home_team"), e.get("away_team"), stamp, r.book,
                          float(r.line), int(r.under_odds), float(1 - r.p_s1), float(r.ev), r.ref, float(r.age) if pd.notna(r.age) else None,
                          stamp, 1, float(r.ev)))
    conn.commit()
    _say(f"unders at 2%+ above the fair price right now: {len(flagged)} ({new} new to the log)")
    show = flagged.sort_values("ev", ascending=False).head(25)
    for r in show.itertuples(index=False):
        e, tip = info[r.event_id]
        _say(f"   {r.player:<24} {r.prop_type:<14} under {r.line:<5} {r.under_odds:>+5} at {r.book:<11} fair {100 * (1 - r.p_s1):4.1f}%  value {100 * r.ev:+5.1f}%  "
              f"[{r.ref}]  {e.get('away_team', '')[:14]} @ {e.get('home_team', '')[:14]}, tips {tip.astimezone(ET):%a %H:%M} ET")
    if len(flagged) > len(show):
        _say(f"   ... and {len(flagged) - len(show)} more")


def grade(refresh_box: bool):
    import market_lab as ml
    from clean_backtest import PROP_TO_STAT, Roster, _games_table, load_box
    from fetch_early_odds import ET

    if refresh_box:
        from clean_backtest import fetch_box
        fetch_box(refresh=False)          # refetches the current month and any month not yet on disk
    conn = _db()
    F = pd.read_sql("select rowid as rid, * from flags where graded_at is null", conn)
    now = datetime.now(timezone.utc)
    F = F[pd.to_datetime(F["commence_time"], utc=True) < now - timedelta(hours=4)]
    if F.empty:
        _say("nothing to grade yet")
        return
    box = load_box()
    games = _games_table(box)
    ev = F.drop_duplicates("event_id")[["event_id", "home_team", "away_team", "commence_time"]].copy()
    ev["et_date"] = pd.to_datetime(ev["commence_time"], utc=True).dt.tz_convert(ET).dt.strftime("%Y-%m-%d")
    link = ml._link_events(ev, games)
    stats = box.set_index(["game_id", "player_id"])
    stats = stats[~stats.index.duplicated()]
    done = 0
    for eid, grp in F.groupby("event_id"):
        gid = link.get(eid)
        if pd.isna(gid):
            continue
        gid = int(gid)
        day = box[box["game_id"] == gid]
        res = Roster(zip(day["player_id"], day["player"])).resolve(grp["player"].unique())
        for r in grp.itertuples(index=False):
            close = conn.execute("select q.under_odds, p.polled_at from quotes q join polls p on p.poll_id = q.poll_id where q.event_id=? and q.player=? "
                                 "and q.prop_type=? and q.book=? and q.line=? and p.polled_at < ? order by p.polled_at desc limit 1",
                                 (eid, r.player, r.prop_type, r.book, r.line, r.commence_time)).fetchone()
            pid = res.get(r.player, (None,))[0]
            played = bool(stats["played"].get((gid, pid), False)) if pid is not None else False
            actual = float(stats.loc[(gid, pid), PROP_TO_STAT[r.prop_type]].sum()) if played else None
            won = None if actual is None else int(actual < r.line)
            conn.execute("update flags set close_under_odds=?, close_seen=?, actual=?, played=?, won=?, graded_at=? where rowid=?",
                         (close[0] if close else None, close[1] if close else None, actual, int(played), won, now.strftime("%Y-%m-%dT%H:%M:%SZ"), r.rid))
            done += 1
    conn.commit()
    _say(f"graded {done} flags")


def _mean_ci(x: pd.Series, cluster: pd.Series) -> tuple:
    """Mean and a 95% half-width with each cluster (here: each game) counted as one unit."""
    x = np.asarray(x, dtype=float)
    m, n = x.mean(), len(x)
    s = pd.Series(x - m).groupby(np.asarray(cluster)).sum()
    g = len(s)
    return m, (1.96 * np.sqrt(g / (g - 1) * (s ** 2).sum()) / n if g > 1 else float("nan"))


def _fair_under_at(conn, F: pd.DataFrame, poll_of: dict) -> pd.Series:
    """Rule S1's fair under-probability at each flag's own line, from the stored quotes of one poll per flag (poll_of: index of F ->
    poll_id), with the flag's own book left out, as poll() computes it at first sight. Each flag is priced in a group of its own,
    so flags never price one another."""
    import market_lab as ml
    from clean_backtest import norm_name

    want = pd.DataFrame({"event_id": F["event_id"], "poll_id": pd.Series(poll_of, dtype=float)}).dropna()
    parts = []
    for (eid, pid), w in want.groupby(["event_id", "poll_id"]):
        Q = pd.read_sql("select book, prop_type, player, line, over_odds, under_odds from quotes where event_id = ? and poll_id = ? order by rowid",
                        conn, params=(eid, int(pid)))
        Q["player_id"] = Q["player"].map(norm_name)
        for i in w.index:
            f = F.loc[i]
            pid_f = norm_name(f["player"])
            k = Q[(Q["player_id"] == pid_f) & (Q["prop_type"] == f["prop_type"])].drop_duplicates("book")   # poll() keeps a book's first quote
            k = k[k["book"] != f["book"]]
            if k.empty:
                continue
            parts.append(k.assign(date=f"#{i}", fid=np.nan))
            parts.append(pd.DataFrame([{"book": f["book"], "prop_type": f["prop_type"], "player": f["player"], "line": float(f["line"]),
                                        "over_odds": -110, "under_odds": -110, "player_id": pid_f, "date": f"#{i}", "fid": float(i)}]))
    if not parts:
        return pd.Series(dtype=float)
    C = ml.add_consensus(pd.concat(parts, ignore_index=True), _sigma())
    C = C[C["fid"].notna()]
    return pd.Series((1 - C["p_s1"]).to_numpy(dtype=float), index=C["fid"].astype(int).to_numpy())


def _after_first_sight(conn, G: pd.DataFrame) -> pd.DataFrame:
    """For graded flags (indexed like G), from the quotes already stored:
    fair_close  rule S1's fair under-probability at the flag's own line from the last poll before tip that priced the game;
    next_ok     1 if the next poll that priced the game (normally 30 minutes later) showed the flag's book at the same line with an
                under price as good or better, 0 if not (including when that book no longer quoted it), missing if no later poll did."""
    from clean_backtest import norm_name

    polls = pd.read_sql("select poll_id, polled_at from polls order by polled_at", conn)
    close_of, next_of = {}, {}
    for eid, grp in G.groupby("event_id"):
        have = pd.read_sql("select distinct poll_id from quotes where event_id = ?", conn, params=(eid,))["poll_id"]
        P = polls[polls["poll_id"].isin(have) & (polls["polled_at"] < grp["commence_time"].iloc[0])]
        if P.empty:
            continue
        for i, f in grp.iterrows():
            close_of[i] = int(P["poll_id"].iloc[-1])
            later = P[P["polled_at"] > f["first_seen"]]
            if len(later):
                next_of[i] = int(later["poll_id"].iloc[0])
    out = pd.DataFrame(index=G.index)
    out["fair_close"] = _fair_under_at(conn, G, close_of)
    out["next_ok"] = np.nan
    want = pd.DataFrame({"event_id": G["event_id"], "poll_id": pd.Series(next_of, dtype=float)}).dropna()
    for (eid, pid), w in want.groupby(["event_id", "poll_id"]):
        Q = pd.read_sql("select book, prop_type, player, line, under_odds from quotes where event_id = ? and poll_id = ?", conn, params=(eid, int(pid)))
        Q["player_id"] = Q["player"].map(norm_name)
        for i in w.index:
            f = G.loc[i]
            k = Q[(Q["player_id"] == norm_name(f["player"])) & (Q["prop_type"] == f["prop_type"]) & (Q["book"] == f["book"])]
            out.loc[i, "next_ok"] = float((((k["line"] - f["line"]).abs() < 1e-9) & (k["under_odds"] >= f["under_odds"])).any())
    return out


def _report_close(conn, G: pd.DataFrame):
    """Report lines for value at the closing fair price and for flags lasting to the next poll (added 2026-11-17; the logging rule
    is unchanged). Intervals count each game as one cluster."""
    from clean_backtest import _dec

    G = G.join(_after_first_sight(conn, G))
    V = G[G["fair_close"].notna()].copy()
    V["v_first"], V["v_close"] = V["ev"] * 100, (V["dec"] * V["fair_close"] - 1) * 100
    V["v_move"] = (V["fair_close"] - V["fair_under"]) * 100

    def val(v: pd.DataFrame) -> str:
        if v.empty:
            return "   0 bets"
        m, ci = _mean_ci(v["v_close"], v["event_id"])
        return (f"{len(v):>4} bets: at first sight {v['v_first'].mean():+5.2f}%, at close {m:+5.2f}% [±{ci:.2f}]; "
                f"fair under-probability {v['v_move'].mean():+.1f} pts since first sight")

    _say(f"   value against the closing fair price (rule S1 at the flag's own line, last poll before tip, the flag's book left out; "
         f"found for {len(V):,} of {len(G):,} bets; interval: each game one cluster):")
    _say(f"      all graded flags          {val(V)}")
    _say(f"      value 4%+ at first sight  {val(V[V['ev'] >= 0.04])}")
    for book, g in G.groupby("book"):
        c = g[g["close_under_odds"].notna()]
        same = f"{((_dec(c['under_odds']) / _dec(c['close_under_odds']) - 1) * 100).mean():+.2f}%" if len(c) else "none"
        nx = g["next_ok"].dropna()
        nxt = f"{nx.mean():.0%} of {len(nx)}" if len(nx) else "none"
        _say(f"      {book:<26}{val(V[V['book'] == book])}; same book's last price {same}; on offer at the next poll {nxt}")
    for d in sorted(V["date"].unique())[-7:]:
        _say(f"      {d}                {val(V[V['date'] == d])}")
    nx = G["next_ok"].dropna()
    if len(nx):
        _say(f"   still on offer at the next poll that priced the game (same book and line, under price as good or better; normally 30 min later): "
             f"{nx.mean():.0%} of {len(nx):,} bets ({int(G['next_ok'].isna().sum())} had no later poll before tip)")


def report():
    import market_lab as ml
    from clean_backtest import _dec

    conn = _db()
    F = pd.read_sql("select * from flags", conn)
    polls = pd.read_sql("select count(*) n, sum(credits) c, min(polled_at) a, max(polled_at) b from polls", conn).iloc[0]
    _say(f"polls {int(polls['n'])} ({polls['a']} to {polls['b']}), credits {int(polls['c'] or 0):,}; flags logged {len(F):,}; graded {int(F['graded_at'].notna().sum()):,}; "
          f"void (did not play) {int((F['played'] == 0).sum()):,}")
    G = F[F["won"].notna()].copy()
    if G.empty:
        return
    G["dec"], G["won"] = _dec(G["under_odds"]), G["won"].astype(bool)
    _say(f"   all graded flags, at the price first seen: {ml.fmt(ml.roi(G))}")
    for ref, g in G.groupby("ref"):
        _say(f"      {ref:<22} {ml.fmt(ml.roi(g))}")
    for book, g in G.groupby("book"):
        _say(f"      {book:<22} {ml.fmt(ml.roi(g))}")
    C = G[G["close_under_odds"].notna()]
    if len(C):
        move = (_dec(C["under_odds"]) / _dec(C["close_under_odds"]) - 1) * 100
        _say(f"   price first seen against the same book's last price before tip, same line ({len(C):,} flags): {move.mean():+.2f}% better on average; "
              f"still as good or better at the end for {(C['close_under_odds'] >= C['under_odds']).mean() * 100:.0f}%")
    # Volume views (added 2026-11-12 for the owner's 3-5 and 30-50 bets a day preferences; the logging rule itself is unchanged):
    # the 4%+ rule is usable live; the nightly top 30/40 by value uses the whole night's flags, so it is a hindsight cap.
    nights = sorted(F["date"].unique())
    _say(f"   flags per game night: mean {len(F) / max(1, len(nights)):.1f}; " + ", ".join(
        f"{lab} {share:.0%}" for lab, share in zip(("0", "1-2", "3-5", "6-29", "30-49", "50+"),
                                                   np.histogram(F.groupby("date").size().reindex(nights, fill_value=0), [0, 1, 3, 6, 30, 50, 10_000])[0] / max(1, len(nights)))))
    for name, sub in [("value 4%+ at first sight (live rule)", G[G["ev"] >= 0.04]),
                      ("each night's top 30 by value", G.sort_values("ev", ascending=False).groupby("date").head(30)),
                      ("each night's top 40 by value", G.sort_values("ev", ascending=False).groupby("date").head(40))]:
        _say(f"   {name:<40} {ml.fmt(ml.roi(sub))}")
    # Each graded night on its own (added 2026-11-15 so the logged report shows the latest night; the rule is unchanged).
    _say("   by game night (last 7 graded):")
    for d in sorted(G["date"].unique())[-7:]:
        g = G[G["date"] == d]
        c = g[g["close_under_odds"].notna()]
        move = f"{((_dec(c['under_odds']) / _dec(c['close_under_odds']) - 1) * 100).mean():+.2f}% on {len(c)}" if len(c) else "none"
        _say(f"      {d}  {int((F['date'] == d).sum()):>4} flags, void {int((F.loc[F['date'] == d, 'played'] == 0).sum())}; all {ml.fmt(ml.roi(g))}; "
             f"4%+ {ml.fmt(ml.roi(g[g['ev'] >= 0.04]))}; first seen vs last price {move}")
    # Value at the closing fair price, and flags lasting to the next poll (added 2026-11-17: the reading rule needs both).
    # Printed last and guarded, so a failure here never costs the lines above.
    try:
        _report_close(conn, G)
    except Exception as e:
        _say(f"   closing-price measures failed: {e!r}")


def recent_flags(n: int = 50) -> list:
    """The latest flags, newest first, for the web app."""
    conn = _db()
    F = pd.read_sql("select date, first_seen, player, prop_type, book, line, under_odds, ev, ref, home_team, away_team, actual, played, won "
                    "from flags order by first_seen desc limit ?", conn, params=(n,))
    return json.loads(F.to_json(orient="records"))


def scheduled(cmd: str) -> str:
    """Run one scheduled step and return what it printed (the web app's scheduler and /api/paper-log use this).
    The command-line steps stop with sys.exit when an API call fails or a key is missing. SystemExit is not an Exception,
    so inside the web app it would get past the scheduler's handler and shut the server down; it is caught here instead."""
    buf = io.StringIO()
    with _LOCK:
        _OUT.buf = buf
        try:
            if cmd == "poll":
                poll(12)
            elif cmd == "grade":
                grade(True)
                report()
            elif cmd == "report":
                report()
        except SystemExit as e:
            _say(f"{cmd} stopped: {e}")
        finally:
            _OUT.buf = None
    return buf.getvalue()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("poll")
    p.add_argument("--hours-ahead", type=float, default=30)
    g = sub.add_parser("grade")
    g.add_argument("--refresh-box", action="store_true")
    sub.add_parser("report")
    a = ap.parse_args()
    if a.cmd == "poll":
        poll(a.hours_ahead)
    elif a.cmd == "grade":
        grade(a.refresh_box)
    else:
        report()
