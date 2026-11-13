#!/usr/bin/env python3
"""
market_lab.py — tests that use every book's quotes at the pre-game snapshots.
Rules were fixed beforehand in outputs/clean/feature_lab/protocol_market_lab.json.

    python market_lab.py build [--hours 12 16 ...]   # every book's quotes and alternate lines, matched and graded
    python market_lab.py dev                         # development seasons: fit on 2023-24, check on 2024-25
    python market_lab.py test                        # 2025-26, read once with everything frozen
    python market_lab.py season                      # this season, a direction check
"""

import argparse
import gzip
import json
import sqlite3
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import norm

import feature_lab as fl
from clean_backtest import LEGS_FILE, MARKET_TO_PROP, PROP_TO_STAT, SIX_BOOKS, Roster, _dec, _games_table, _imp, load_box
from fetch_early_odds import db_path, raw_dir

HOURS = [12, 16, 18, 20, 22]
DIR = fl.LAB_DIR / "market"
PROTOCOL_FILE = fl.LAB_DIR / "protocol_market_lab.json"
RESULTS_FILE = fl.LAB_DIR / "market_lab_results.json"
EXTRA_BOOKS = ["williamhill_us", "hardrockbet", "fanatics"]
LADDER_BOOKS = ["betrivers", "betparx", "espnbet"]
PROP = ["date", "player_id", "prop_type"]
SEASON_NAME = {2023: "2023-24", 2024: "2024-25", 2025: "2025-26", 2026: "2026-27"}
THRESHOLDS = (0.02, 0.04)
KEY_POINTS_LINE = 15.5
LINE = "=" * 126


# =============================================================================
# BUILD: every book's quotes (and the ladder books' alternate lines), matched to players and graded
# =============================================================================

def _link_events(events: pd.DataFrame, games: pd.DataFrame) -> pd.Series:
    """event_id -> game_id by team pair and Eastern date (within a day), the rule clean_backtest.build_legs uses."""
    by_pair = defaultdict(list)
    for r in games.itertuples(index=False):
        by_pair[frozenset((r.home, r.away))].append((r.date, r.game_id))
    out = {}
    for r in events.itertuples(index=False):
        d0 = datetime.strptime(r.et_date, "%Y-%m-%d").date()
        best = None
        for gdate, gid in by_pair.get(frozenset((r.home_team, r.away_team)), []):
            gap = abs((datetime.strptime(gdate, "%Y-%m-%d").date() - d0).days)
            if gap <= 1 and (best is None or gap < best[0]):
                best = (gap, gid)
        if best:
            out[r.event_id] = best[1]
    return pd.Series(out, dtype="float64")


def _match_and_grade(P: pd.DataFrame, box: pd.DataFrame, games: pd.DataFrame, rosters: dict) -> pd.DataFrame:
    """Rows with event_id, book, market, name, line, prices -> player ids, results and the flags needed to void a bet."""
    P["prop_type"] = P["market"].map(MARKET_TO_PROP)
    P = P[P["prop_type"].notna() & P["game_id"].notna() & (P["over_odds"].abs() >= 100) & (P["under_odds"].abs() >= 100)].copy()
    P["game_id"] = P["game_id"].astype(int)
    resolved = {}
    for gid, g in P.groupby("game_id"):
        if gid in rosters:
            for n, (pid, _) in rosters[gid].resolve(g["name"].unique()).items():
                resolved[(gid, n)] = pid
    P["player_id"] = [resolved.get(k) for k in zip(P["game_id"], P["name"])]
    P = P[P["player_id"].notna()].copy()
    P["player_id"] = P["player_id"].astype(int)
    stats = box.set_index(["game_id", "player_id"])
    stats = stats[~stats.index.duplicated()]
    idx = pd.MultiIndex.from_arrays([P["game_id"], P["player_id"]])
    P["played"] = stats["played"].reindex(idx).to_numpy() == True  # noqa: E712
    P["team_id"] = stats["team_id"].reindex(idx).to_numpy()
    P["actual"] = np.nan
    for prop, cols in PROP_TO_STAT.items():
        m = (P["prop_type"] == prop).to_numpy()
        P.loc[m, "actual"] = stats[cols].sum(axis=1).reindex(idx[m]).to_numpy()
    g = games.set_index("game_id")
    P["date"] = P["game_id"].map(g["date"])
    P["season"] = P["game_id"].map(g["season"]).astype(int)
    P["adds_up"] = P["game_id"].map(g["box_adds_up"]).astype(bool)
    return P


def build(hours=None):
    DIR.mkdir(parents=True, exist_ok=True)
    box = load_box()
    games = _games_table(box)
    rosters = {gid: Roster(zip(g["player_id"], g["player"])) for gid, g in box.groupby("game_id")}
    cols = ["date", "season", "game_id", "player_id", "team_id", "prop_type", "book", "line", "over_odds", "under_odds", "actual", "played", "adds_up"]
    for h in hours or HOURS:
        conn = sqlite3.connect(f"file:{db_path(h)}?mode=ro", uri=True, timeout=60)
        events = pd.read_sql("select event_id, et_date, home_team, away_team from events where status = 'ok'", conn)
        P = pd.read_sql("select p.event_id, p.bookmaker as book, p.market, p.player_name as name, p.line, p.over_odds, p.under_odds "
                        "from props p join events e on e.event_id = p.event_id where e.status = 'ok'", conn)
        conn.close()
        link = _link_events(events, games)
        P["game_id"] = P["event_id"].map(link)
        Q = _match_and_grade(P, box, games, rosters).drop_duplicates(["game_id", "player_id", "prop_type", "book"])
        Q[cols].to_parquet(DIR / f"quotes_h{h:02d}.parquet", index=False)

        rows = []
        for path in raw_dir(h).glob("*/*.json.gz"):
            with gzip.open(path, "rt") as fh:
                data = json.load(fh).get("data") or {}
            eid = path.name.split(".")[0]
            for bk in data.get("bookmakers", []):
                if bk.get("key") not in LADDER_BOOKS:
                    continue
                for mk in bk.get("markets", []):
                    by_line = {}
                    for o in mk.get("outcomes", []):
                        if o.get("description") and o.get("name") in ("Over", "Under") and o.get("point") is not None:
                            by_line.setdefault((o["description"], float(o["point"])), {})[o["name"]] = o.get("price", 0)
                    for (name, point), pr in by_line.items():
                        if "Over" in pr and "Under" in pr:
                            rows.append((eid, bk["key"], mk.get("key"), name, point, pr["Over"], pr["Under"]))
        R = pd.DataFrame(rows, columns=["event_id", "book", "market", "name", "line", "over_odds", "under_odds"])
        R["game_id"] = R["event_id"].map(link)
        R = _match_and_grade(R, box, games, rosters).drop_duplicates(["game_id", "player_id", "prop_type", "book", "line"])
        R["balance"] = np.abs(_imp(R["over_odds"]) - _imp(R["under_odds"]))
        main = R.sort_values("balance").drop_duplicates(["game_id", "player_id", "prop_type", "book"])
        R = R.merge(main[["game_id", "player_id", "prop_type", "book", "line"]].rename(columns={"line": "main_line"}), on=["game_id", "player_id", "prop_type", "book"])
        R["n_lines"] = R.groupby(["game_id", "player_id", "prop_type", "book"])["line"].transform("size")
        R[cols + ["main_line", "n_lines"]].to_parquet(DIR / f"rungs_h{h:02d}.parquet", index=False)
        print(f"{h:02d}:00  events {len(events):,} (linked {int(events['event_id'].isin(link.index).sum()):,}); quotes {len(Q):,} "
              f"by season {Q.groupby('season').size().to_dict()}; ladder lines {len(R):,} ({int((R['line'] != R['main_line']).sum()):,} alternate)", flush=True)


# =============================================================================
# FAIR PRICE
# =============================================================================

def devig_power(over_odds, under_odds) -> np.ndarray:
    """Power-method fair over-probability, vectorised: the exponent c with io^c + iu^c = 1, found by bisection."""
    a, b = _imp(over_odds), _imp(under_odds)
    lo, hi = np.full(a.shape, 0.2), np.full(a.shape, 8.0)
    for _ in range(46):
        mid = (lo + hi) / 2
        above = a ** mid + b ** mid - 1 >= 0
        lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
    c = (lo + hi) / 2
    pa, pb = a ** c, b ** c
    return pa / (pa + pb)


def fit_sigma(seasons) -> dict:
    """Spread per prop and line level from results: how fast the share of results above a line falls as the line rises.
    ln(sigma) = a + b ln(line), fitted across DraftKings closing lines with 200+ graded bets."""
    L = pd.read_parquet(LEGS_FILE, columns=["season", "prop_type", "line", "actual"])
    L = L[L["season"].isin(seasons)]
    out = {}
    for prop, g in L.groupby("prop_type"):
        pts = []
        for line, h in g.groupby("line"):
            if len(h) < 200:
                continue
            lower, upper = (h["actual"] > line - 1).mean(), (h["actual"] > line + 1).mean()
            if 0.02 < upper < lower < 0.98:
                pts.append((line, 2.0 / (norm.ppf(lower) - norm.ppf(upper)), len(h)))
        pts = np.array(pts)
        b, a = np.polyfit(np.log(pts[:, 0]), np.log(pts[:, 1]), 1, w=np.sqrt(pts[:, 2]))
        out[prop] = (float(a), float(b))
    return out


def sigma_of(prop: pd.Series, line: pd.Series, sig: dict) -> np.ndarray:
    a = prop.map({k: v[0] for k, v in sig.items()}).to_numpy(dtype=float)
    b = prop.map({k: v[1] for k, v in sig.items()}).to_numpy(dtype=float)
    return np.exp(a + b * np.log(np.maximum(line.to_numpy(dtype=float), 0.5)))


def add_consensus(Q: pd.DataFrame, sig: dict) -> pd.DataFrame:
    """For every quote: the other books' fair probabilities moved to its line (Pinnacle's, and the mean of the rest,
    both on the normal scale), and the other books' median line."""
    Q = Q.reset_index(drop=True)
    Q["row"] = np.arange(len(Q))
    Q["p_own"] = devig_power(Q["over_odds"], Q["under_odds"])
    Q["z_own"] = norm.ppf(np.clip(Q["p_own"], 0.005, 0.995))
    Q["sigma"] = sigma_of(Q["prop_type"], Q["line"], sig)
    src = Q[PROP + ["book", "line", "z_own"]].rename(columns={"book": "book_s", "line": "line_s", "z_own": "z_s"})
    pairs = Q[PROP + ["row", "book", "line", "sigma"]].merge(src, on=PROP)
    pairs = pairs[pairs["book"] != pairs["book_s"]]
    pairs["zc"] = pairs["z_s"] + (pairs["line_s"] - pairs["line"]) / pairs["sigma"]
    is_pin = pairs["book_s"] == "pinnacle"
    Q["z_pin"] = Q["row"].map(pairs[is_pin].groupby("row")["zc"].first())
    pin_line = Q["row"].map(pairs[is_pin].groupby("row")["line_s"].first())
    oth = pairs[~is_pin].groupby("row")["zc"].agg(["mean", "size"])
    Q["z_oth"] = Q["row"].map(oth["mean"])
    Q["n_oth"] = Q["row"].map(oth["size"]).fillna(0).astype(int)
    med = pairs.groupby("row")["line_s"].agg(["median", "size"])
    Q["med_line"] = Q["row"].map(med["median"])
    Q["n_others"] = Q["row"].map(med["size"]).fillna(0).astype(int)
    Q["p_s1"] = np.where(Q["z_pin"].notna(), norm.cdf(Q["z_pin"]), np.where(Q["n_oth"] >= 3, norm.cdf(Q["z_oth"]), np.nan))
    Q["ref"] = np.where(pin_line.isna(), "other books only", np.where((pin_line - Q["line"]).abs() < 1e-6, "Pinnacle, same line", "Pinnacle, other line"))
    Q["case"] = np.where(Q["z_pin"].notna() & (Q["n_oth"] >= 1), "A", np.where(Q["z_pin"].isna() & (Q["n_oth"] >= 3), "B",
                         np.where(Q["z_pin"].notna(), "C", "")))
    return Q


CASE_INPUTS = {"A": ["z_own", "z_pin", "z_oth"], "B": ["z_own", "z_oth"], "C": ["z_own", "z_pin"]}
PROP_LEVELS = ["points", "rebounds", "assists", "threes", "pts_rebs_asts"]


def _design(F: pd.DataFrame, case: str) -> np.ndarray:
    return np.column_stack([F[c].to_numpy(dtype=float) for c in CASE_INPUTS[case]]
                           + [(F["prop_type"] == p).to_numpy(dtype=float) for p in PROP_LEVELS[1:]])


def fit_s2(F: pd.DataFrame) -> dict:
    """Logistic fits of the result on what the market says, per target book and per case (which inputs exist)."""
    from sklearn.linear_model import LogisticRegression

    F = F[F["played"] & F["adds_up"] & (F["case"] != "")]
    y = (F["actual"] > F["line"]).astype(int)
    models, pooled = {}, F[F["book"].isin(SIX_BOOKS)]
    for case in CASE_INPUTS:
        P = pooled[pooled["case"] == case]
        if len(P) >= 2000:
            models[("pooled", case)] = LogisticRegression(C=1e4, max_iter=3000).fit(_design(P, case), y.loc[P.index])
        for book, g in F[F["case"] == case].groupby("book"):
            if len(g) >= 20000:
                models[(book, case)] = LogisticRegression(C=1e4, max_iter=3000).fit(_design(g, case), y.loc[g.index])
    return models


def apply_s2(Q: pd.DataFrame, models: dict) -> np.ndarray:
    p = np.full(len(Q), np.nan)
    for case in CASE_INPUTS:
        for book in Q["book"].unique():
            m = ((Q["case"] == case) & (Q["book"] == book)).to_numpy()
            if not m.any():
                continue
            use = "betrivers" if book == "betparx" else book
            model = models.get((use, case)) or models.get(("pooled", case))
            if model is not None:
                p[m] = model.predict_proba(_design(Q[m], case))[:, 1]
    return p


def describe_s2(models: dict):
    print("   fitted weights (case A: Pinnacle and other books both quote), on the normal scale; a weight near 1 on a book's own price means it needs no correcting")
    print(f"   {'book':<16}{'own price':>11}{'Pinnacle':>10}{'other books':>13}{'intercept':>11}")
    for (book, case), m in sorted(models.items()):
        if case == "A":
            print(f"   {book:<16}{m.coef_[0][0]:>11.2f}{m.coef_[0][1]:>10.2f}{m.coef_[0][2]:>13.2f}{m.intercept_[0]:>11.3f}")


# =============================================================================
# SCORING
# =============================================================================

def roi(df: pd.DataFrame) -> dict:
    """Flat $100 bets on rows carrying `dec` (decimal odds) and `won`. Interval: each date one cluster."""
    n = len(df)
    if n == 0:
        return {"n": 0, "wr": float("nan"), "roi": float("nan"), "ci": float("nan")}
    pnl = np.where(df["won"], 100 * (df["dec"] - 1), -100.0)
    mean = pnl.mean()
    by = pd.DataFrame({"date": df["date"].to_numpy(), "pnl": pnl}).groupby("date")["pnl"].agg(["sum", "size"])
    g = len(by)
    ci = 1.96 * np.sqrt(g / (g - 1) * ((by["sum"] - mean * by["size"]) ** 2).sum()) / n if g > 1 else float("nan")
    return {"n": int(n), "wr": float(df["won"].mean() * 100), "roi": float(mean), "ci": float(ci)}


def fmt(s: dict) -> str:
    return "no bets" if not s["n"] else f"{s['n']:>7,} bets {s['wr']:5.1f}% won  {s['roi']:+6.2f}% [±{s['ci']:.2f}]"


def sides(Q: pd.DataFrame, pcol: str = None) -> pd.DataFrame:
    """Each quote as two possible bets (over, under) with decimal odds, result and, given a fair probability, expected value."""
    keep = PROP + ["season", "game_id", "team_id", "hour", "book", "line", "med_line", "n_others", "row", "ref"]
    frames = []
    for side in ("over", "under"):
        d = Q[keep].copy()
        d["side"] = side
        d["dec"] = _dec(Q["over_odds"] if side == "over" else Q["under_odds"])
        d["won"] = (Q["actual"] > Q["line"]).to_numpy() if side == "over" else (Q["actual"] < Q["line"]).to_numpy()
        if pcol:
            p = Q[pcol].to_numpy(dtype=float)
            d["ev"] = d["dec"] * (p if side == "over" else 1 - p) - 1
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def first_flag(F: pd.DataFrame, unit=("date", "player_id", "prop_type", "side")) -> pd.DataFrame:
    """One bet per unit: the first snapshot where anything qualifies, and there the highest expected value."""
    return F.sort_values(["hour", "ev"], ascending=[True, False]).drop_duplicates(list(unit))


def by_season(label: str, F: pd.DataFrame, out: dict, seasons):
    row = {}
    for s in seasons:
        row[SEASON_NAME[s]] = roi(F[F["season"] == s])
    out[label] = row
    print(f"   {label:<46}" + "   ".join(f"{SEASON_NAME[s]}: {fmt(row[SEASON_NAME[s]])}" for s in seasons))


# =============================================================================
# THE TESTS
# =============================================================================

def prepare(fit_seasons, eval_seasons, sigma_seasons, hours):
    sig = fit_sigma(sigma_seasons)
    frames = {}
    for h in hours:
        Q = pd.read_parquet(DIR / f"quotes_h{h:02d}.parquet")
        Q = Q[Q["season"].isin(set(fit_seasons) | set(eval_seasons))]
        Q = add_consensus(Q, sig)
        Q["hour"] = h
        frames[h] = Q
    models = fit_s2(pd.concat([q[q["season"].isin(fit_seasons)] for q in frames.values()], ignore_index=True))
    for h, Q in frames.items():
        Q["p_s2"] = apply_s2(Q, models)
        frames[h] = Q[Q["played"] & Q["adds_up"]].reset_index(drop=True)
    return sig, models, frames


def presence(hours):
    """DraftKings' quoted players at each hour, graded or not, with the points line: used to spot pulled props."""
    out = {}
    for h in hours:
        Q = pd.read_parquet(DIR / f"quotes_h{h:02d}.parquet", columns=["game_id", "player_id", "team_id", "prop_type", "book", "line"])
        dk = Q[Q["book"] == "draftkings"]
        pts = dk[dk["prop_type"] == "points"].set_index(["game_id", "player_id"])["line"]
        players = dk.drop_duplicates(["game_id", "player_id"]).set_index(["game_id", "player_id"])["team_id"]
        out[h] = (players, pts[~pts.index.duplicated()])
    return out


def run(mode: str):
    if not PROTOCOL_FILE.exists():
        raise SystemExit(f"{PROTOCOL_FILE.name} must be written before these tests are read")
    fit_seasons, eval_seasons, sigma_seasons, hours = {
        "dev": ([2023], [2023, 2024], [2023, 2024], HOURS),
        "test": ([2023, 2024], [2025], [2023, 2024], HOURS),
        "season": ([2023, 2024, 2025], [2026], [2023, 2024, 2025], [12, 18, 20, 22]),
    }[mode]
    show = eval_seasons
    report = {"mode": mode, "fit_seasons": fit_seasons, "eval_seasons": eval_seasons}
    print(LINE)
    print(f"MARKET LAB, {mode.upper()}: fair price fitted on {[SEASON_NAME[s] for s in fit_seasons]}, shown for {[SEASON_NAME[s] for s in show]}"
          + ("  (2023-24 is in-sample for the fitted price here)" if mode == "dev" else ""))
    print(LINE)
    sig, models, frames = prepare(fit_seasons, eval_seasons, sigma_seasons, hours)
    print("   line-conversion spread at typical lines: " + "; ".join(
        f"{p} {np.exp(a + b * np.log(l)):.1f} at {l}" for p, l in [("points", 20.5), ("rebounds", 6.5), ("assists", 4.5), ("threes", 2.5), ("pts_rebs_asts", 30.5)]
        for a, b in [sig[p]]))
    describe_s2(models)
    Q = pd.concat(frames.values(), ignore_index=True)
    Q["row"] = np.arange(len(Q))
    elig = Q["case"] != ""
    six = Q["book"].isin(SIX_BOOKS)
    print(f"   quotes at the six books: {int(six.sum()):,}; with a fair price: {int((six & elig).sum()):,} ({(six & elig).sum() / six.sum() * 100:.0f}%)")

    # ---- T1
    print("\n" + LINE)
    print("T1. FAIR-PRICE SCREEN ON ANY LINE  (bet a side when decimal odds x fair probability - 1 clears the threshold; one bet per date, player, prop and side)")
    print(LINE)
    report["T1"] = {}
    flagged = {}
    for pcol, name in [("p_s2", "S2 fitted"), ("p_s1", "S1 plain")]:
        S = sides(Q[six & Q[pcol].notna()], pcol)
        for thr in THRESHOLDS:
            F = first_flag(S[S["ev"] >= thr])
            flagged[(pcol, thr)] = F
            tag = f"{name}, {thr * 100:.0f}%+"
            by_season(f"{tag}, all", F, report["T1"], show)
            by_season(f"{tag}, unders", F[F["side"] == "under"], report["T1"], show)
            by_season(f"{tag}, overs", F[F["side"] == "over"], report["T1"], show)
    print("   unders at 2%+, by what the fair price rests on:")
    for pcol, name in [("p_s1", "S1 plain"), ("p_s2", "S2 fitted")]:
        F = flagged[(pcol, 0.02)]
        for ref, g in F[F["side"] == "under"].groupby("ref"):
            by_season(f"      {name}, unders, {ref}", g, report["T1"], show)
    print("   S2 fitted, 2%+, by snapshot (every qualifying side at that hour, best expected value per prop and side):")
    S2 = sides(Q[six & Q["p_s2"].notna()], "p_s2")
    for h in hours:
        by_season(f"      {h:02d}:00", first_flag(S2[(S2["ev"] >= 0.02) & (S2["hour"] == h)]), report["T1"], show)
    print("   S2 fitted, 2%+, first flag, by book:")
    for b, g in flagged[("p_s2", 0.02)].groupby("book"):
        by_season(f"      {b}", g, report["T1"], show)
    print("   the same screen aimed at books outside the six:")
    for b in EXTRA_BOOKS:
        Sx = sides(Q[(Q["book"] == b) & Q["p_s2"].notna()], "p_s2")
        by_season(f"      {b}, 2%+", first_flag(Sx[Sx["ev"] >= 0.02]), report["T1"], show)

    # ---- T2
    print("\n" + LINE)
    print("T2. OFF-MARKET LINES  (the book's line is 1.0 or more from the other books' median line, three or more others quoting)")
    print(LINE)
    report["T2"] = {}
    off = six & (Q["n_others"] >= 3) & ((Q["line"] - Q["med_line"]).abs() >= 1.0)
    on = six & (Q["n_others"] >= 3) & ((Q["line"] - Q["med_line"]).abs() < 1e-6)
    print(f"   off-market quotes at the six books: {int(off.sum()):,} of {int((six & (Q['n_others'] >= 3)).sum()):,} ({off.sum() / max(1, (six & (Q['n_others'] >= 3)).sum()) * 100:.1f}%)")
    A = sides(Q[off])
    A["gap"] = (A["line"] - A["med_line"]).abs()
    toward = A[((A["line"] > A["med_line"]) & (A["side"] == "under")) | ((A["line"] < A["med_line"]) & (A["side"] == "over"))].copy()
    toward["ev"] = toward["gap"] + toward["dec"] / 100
    T = first_flag(toward)
    by_season("a. toward the market, posted price, all", T, report["T2"], show)
    by_season("a. under where the book's line is higher", T[T["side"] == "under"], report["T2"], show)
    by_season("a. over where the book's line is lower", T[T["side"] == "over"], report["T2"], show)
    away = A[((A["line"] > A["med_line"]) & (A["side"] == "over")) | ((A["line"] < A["med_line"]) & (A["side"] == "under"))].copy()
    away["ev"] = away["gap"] + away["dec"] / 100
    by_season("   for scale: away from the market", first_flag(away), report["T2"], show)
    for pcol, name in [("p_s2", "S2 fitted"), ("p_s1", "S1 plain")]:
        for label, mask in [("off-market quotes", off), ("quotes on the market line", on)]:
            S = sides(Q[mask & Q[pcol].notna()], pcol)
            by_season(f"b. {name} 2%+ on {label}", first_flag(S[S["ev"] >= 0.02]), report["T2"], show)

    # ---- T3
    print("\n" + LINE)
    print("T3. ALTERNATE LINES  (BetRivers and BetParx: points, PRA; ESPN Bet: points, rebounds, assists, threes)")
    print(LINE)
    report["T3"] = {}
    rung_frames = []
    for h in hours:
        R = pd.read_parquet(DIR / f"rungs_h{h:02d}.parquet")
        R = R[R["season"].isin(show) & R["played"] & R["adds_up"] & (R["n_lines"] > 1) & (R["line"] != R["main_line"])].copy()
        main = frames[h][["date", "player_id", "prop_type", "book", "line", "p_s2", "p_s1", "sigma"]].rename(
            columns={"line": "q_line", "p_s2": "p_main_s2", "p_s1": "p_main_s1", "sigma": "sigma_main"})
        R = R.merge(main, on=["date", "player_id", "prop_type", "book"])
        R = R[(R["q_line"] - R["main_line"]).abs() < 1e-6]
        R["hour"] = h
        rung_frames.append(R)
    R = pd.concat(rung_frames, ignore_index=True)
    R["steps"] = np.clip(np.round(R["line"] - R["main_line"]), -3, 3).astype(int)
    for pc in ("s2", "s1"):
        z = norm.ppf(np.clip(R[f"p_main_{pc}"], 0.005, 0.995)) + (R["main_line"] - R["line"]) / R["sigma_main"]
        R[f"p_{pc}"] = np.where(R[f"p_main_{pc}"].notna(), norm.cdf(z), np.nan)
    R["med_line"], R["n_others"], R["row"], R["ref"] = R["main_line"], 0, np.arange(len(R)), ""
    print(f"   alternate lines graded: {len(R):,}; by book {R.groupby('book').size().to_dict()}")
    print("   a. blind bets by distance from the main line (one per date, player, prop, line and side: first snapshot, best price)")
    B = sides(R)
    B["steps"] = np.repeat(R["steps"].to_numpy()[None, :], 2, axis=0).ravel()
    B["ev"] = B["dec"]
    unit = ("date", "player_id", "prop_type", "line", "side")
    for st in (-3, -2, -1, 1, 2, 3):
        name = f"{abs(st)}{'+' if abs(st) == 3 else ''} step{'s' if abs(st) > 1 else ''} {'below' if st < 0 else 'above'} the main line"
        for side in ("over", "under"):
            by_season(f"   {name}, {side}", first_flag(B[(B["steps"] == st) & (B["side"] == side)], unit), report["T3"], show)
    for pc, name in [("p_s2", "S2 fitted"), ("p_s1", "S1 plain")]:
        S = sides(R[R[pc].notna()], pc)
        for thr in THRESHOLDS:
            F = first_flag(S[S["ev"] >= thr], unit)
            by_season(f"b. {name}, {thr * 100:.0f}%+, all", F, report["T3"], show)
            by_season(f"b. {name}, {thr * 100:.0f}%+, unders", F[F["side"] == "under"], report["T3"], show)
            by_season(f"b. {name}, {thr * 100:.0f}%+, overs", F[F["side"] == "over"], report["T3"], show)

    # ---- T4
    print("\n" + LINE)
    print(f"T4. PULLED PROPS  (DraftKings quoted the player at the previous snapshot and not now, game still quoted; key = points line {KEY_POINTS_LINE}+)")
    print(LINE)
    report["T4"] = {}
    pres = presence(hours)
    mates, mates_same = [], []
    n_pulled = n_key = 0
    for g, h in zip(hours[:-1], hours[1:]):
        prev_players, prev_pts = pres[g]
        now_players, _ = pres[h]
        games_now = set(now_players.index.get_level_values(0))
        pulled = [k for k in prev_players.index if k not in now_players.index and k[0] in games_now]
        key = [k for k in pulled if prev_pts.get(k, 0) >= KEY_POINTS_LINE]
        n_pulled += len(pulled)
        n_key += len(key)
        teams = {(k[0], prev_players.get(k)) for k in key}
        D = frames[h][frames[h]["book"] == "draftkings"]
        m = np.array([(gid, tid) in teams for gid, tid in zip(D["game_id"], D["team_id"])], dtype=bool)
        t = D[m].copy()
        prev_line = frames[g][frames[g]["book"] == "draftkings"].set_index(PROP)["line"] if g in frames else pd.Series(dtype=float)
        prev_line = prev_line[~prev_line.index.duplicated()]
        t["prev_line"] = prev_line.reindex(pd.MultiIndex.from_frame(t[PROP])).to_numpy()
        mates.append(t)
    Tm = pd.concat(mates, ignore_index=True) if mates else pd.DataFrame()
    Tm = Tm[Tm["season"].isin(show)]
    allDK = Q[(Q["book"] == "draftkings") & (Q["hour"] != hours[0])]
    print(f"   pulled players {n_pulled:,}, key ones {n_key:,} (all seasons loaded); teammates' DraftKings props at that snapshot, shown seasons: {len(Tm):,}")
    for side in ("over", "under"):
        S = sides(Tm)
        by_season(f"a. teammates, blind {side}", S[S["side"] == side], report["T4"], show)
        Sall = sides(allDK)
        by_season(f"   for scale: every DraftKings prop, blind {side}", Sall[Sall["side"] == side], report["T4"], show)
    same = Tm[(Tm["prev_line"] - Tm["line"]).abs() < 1e-6]
    S = sides(same)
    by_season("b. teammates whose line has not moved, over", S[S["side"] == "over"], report["T4"], show)
    by_season("b. teammates whose line has not moved, under", S[S["side"] == "under"], report["T4"], show)
    moved = Tm[Tm["prev_line"].notna() & ((Tm["prev_line"] - Tm["line"]).abs() > 1e-6)]
    S = sides(moved)
    by_season("   teammates whose line has moved, over", S[S["side"] == "over"], report["T4"], show)
    for prop in PROP_LEVELS:
        S = sides(Tm[Tm["prop_type"] == prop])
        by_season(f"   teammates, {prop}, over", S[S["side"] == "over"], report["T4"], show)
    tm_keys = set(zip(Tm["date"], Tm["player_id"], Tm["hour"]))
    S2t = S2[(S2["ev"] >= 0.02)]
    S2t = S2t[[k in tm_keys for k in zip(S2t["date"], S2t["player_id"], S2t["hour"])]]
    by_season("c. S2 2%+ screen among those teammates", first_flag(S2t), report["T4"], show)
    by_season("c. ... unders", first_flag(S2t[S2t["side"] == "under"]), report["T4"], show)
    by_season("c. ... overs", first_flag(S2t[S2t["side"] == "over"]), report["T4"], show)

    done = json.loads(RESULTS_FILE.read_text()) if RESULTS_FILE.exists() else {}
    done[mode] = {"finished": datetime.now().isoformat(timespec="seconds"), **report}
    RESULTS_FILE.write_text(json.dumps(done, indent=1, default=float))
    flagged[("p_s2", 0.02)].to_parquet(DIR / f"t1_s2_first_flag_{mode}.parquet", index=False)
    flagged[("p_s1", 0.02)].to_parquet(DIR / f"t1_s1_first_flag_{mode}.parquet", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--hours", type=int, nargs="*", default=None)
    for name in ("dev", "test", "season"):
        sub.add_parser(name)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.hours)
    else:
        run(a.cmd)
