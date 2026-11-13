#!/usr/bin/env python3
"""Feature experiments on the clean backtest: does a feature group add anything to the market price?"""

import argparse
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from clean_backtest import BOOK_LINES_FILE, CLEAN_DIR, LEGS_FILE, SIX_BOOKS, _dec, _imp, load_box

logger = logging.getLogger("feature_lab")

LAB_DIR = CLEAN_DIR / "feature_lab"
FEATURES_FILE = LAB_DIR / "features.parquet"
PROTOCOL_FILE = LAB_DIR / "protocol.json"
SCREEN_FILE = LAB_DIR / "screen_results.json"
FINAL_FILE = LAB_DIR / "final_results.json"

PROP_STAT = {"points": "pts", "rebounds": "reb", "assists": "ast", "threes": "fg3m", "pts_rebs_asts": "pra"}
TARGET_STATS = ["pts", "reb", "ast", "fg3m", "pra"]

# team -> (latitude, longitude, UTC offset in winter, altitude in feet) of the home arena
ARENAS = {
    "Atlanta Hawks": (33.757, -84.396, -5, 1050), "Boston Celtics": (42.366, -71.062, -5, 20),
    "Brooklyn Nets": (40.683, -73.975, -5, 30), "Charlotte Hornets": (35.225, -80.839, -5, 750),
    "Chicago Bulls": (41.881, -87.674, -6, 600), "Cleveland Cavaliers": (41.497, -81.688, -5, 650),
    "Dallas Mavericks": (32.790, -96.810, -6, 430), "Denver Nuggets": (39.749, -105.008, -7, 5280),
    "Detroit Pistons": (42.341, -83.055, -5, 600), "Golden State Warriors": (37.768, -122.388, -8, 10),
    "Houston Rockets": (29.751, -95.362, -6, 50), "Indiana Pacers": (39.764, -86.156, -5, 715),
    "Los Angeles Clippers": (33.990, -118.300, -8, 100), "Los Angeles Lakers": (34.043, -118.267, -8, 270),
    "Memphis Grizzlies": (35.138, -90.051, -6, 260), "Miami Heat": (25.781, -80.187, -5, 10),
    "Milwaukee Bucks": (43.045, -87.917, -6, 600), "Minnesota Timberwolves": (44.979, -93.276, -6, 830),
    "New Orleans Pelicans": (29.949, -90.082, -6, 5), "New York Knicks": (40.751, -73.994, -5, 30),
    "Oklahoma City Thunder": (35.463, -97.515, -6, 1200), "Orlando Magic": (28.539, -81.384, -5, 100),
    "Philadelphia 76ers": (39.901, -75.172, -5, 20), "Phoenix Suns": (33.446, -112.071, -7, 1090),
    "Portland Trail Blazers": (45.532, -122.667, -8, 50), "Sacramento Kings": (38.580, -121.500, -8, 30),
    "San Antonio Spurs": (29.427, -98.438, -6, 650), "Toronto Raptors": (43.643, -79.379, -5, 250),
    "Utah Jazz": (40.768, -111.901, -7, 4226), "Washington Wizards": (38.898, -77.021, -5, 30),
}
ROTATION_MINUTES = 18.0
ABSENCE_WINDOW = 10


# =============================================================================
# FEATURE ENGINE (everything is computed from games before the one being priced,
# except the availability group, which also uses who took the floor that night)
# =============================================================================

def _roll(df: pd.DataFrame, key: str, cols: list, window, how: str = "mean") -> pd.DataFrame:
    """Per-group statistic over previous rows only (the current row is shifted out)."""
    shifted = df.groupby(key, sort=False)[cols].shift(1)
    grouped = shifted.groupby(df[key], sort=False)
    if window == "all":
        r = grouped.expanding(min_periods=1)
    elif isinstance(window, str):
        r = grouped.ewm(halflife=float(window.replace("ewm", "")), min_periods=1)
    else:
        r = grouped.rolling(window, min_periods=1)
    out = getattr(r, how)()
    out.index = out.index.droplevel(0)
    return out.reindex(df.index)


def _haversine_miles(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def _pos_group(pos) -> str:
    pos = str(pos or "").strip().upper()
    return pos[0] if pos[:1] in ("G", "F", "C") else "U"


def build_team_games(box: pd.DataFrame) -> pd.DataFrame:
    sums = ["pts", "fga", "fgm", "fg3a", "fg3m", "fta", "ftm", "oreb", "dreb", "reb", "ast", "stl", "blk", "turnover", "pf", "minutes"]
    played = box[box["played"]]
    tg = played.groupby(["game_id", "team_id"], as_index=False)[sums].sum()
    meta = box.drop_duplicates("game_id")[["game_id", "date", "season", "postseason", "home_team_id", "visitor_team_id"]]
    tg = tg.merge(meta, on="game_id")
    tg = tg[(tg["team_id"] == tg["home_team_id"]) | (tg["team_id"] == tg["visitor_team_id"])].copy()
    tg["is_home"] = (tg["team_id"] == tg["home_team_id"]).astype(int)
    tg["opp_team_id"] = np.where(tg["is_home"] == 1, tg["visitor_team_id"], tg["home_team_id"])
    opp = tg[["game_id", "team_id"] + sums].rename(columns={c: f"opp_{c}" for c in sums}).rename(columns={"team_id": "opp_team_id"})
    tg = tg.merge(opp, on=["game_id", "opp_team_id"])
    tg["pra"] = tg["pts"] + tg["reb"] + tg["ast"]
    tg["opp_pra"] = tg["opp_pts"] + tg["opp_reb"] + tg["opp_ast"]

    poss = 0.5 * ((tg["fga"] + 0.44 * tg["fta"] - tg["oreb"] + tg["turnover"])
                  + (tg["opp_fga"] + 0.44 * tg["opp_fta"] - tg["opp_oreb"] + tg["opp_turnover"]))
    tg["poss"] = poss.clip(lower=50)
    tg["pace"] = 48 * tg["poss"] / (tg["minutes"].clip(lower=200) / 5)
    tg["off_rtg"] = 100 * tg["pts"] / tg["poss"]
    tg["def_rtg"] = 100 * tg["opp_pts"] / tg["poss"]
    tg["net_rtg"] = tg["off_rtg"] - tg["def_rtg"]
    tg["efg"] = (tg["fgm"] + 0.5 * tg["fg3m"]) / tg["fga"].clip(lower=1)
    tg["opp_efg"] = (tg["opp_fgm"] + 0.5 * tg["opp_fg3m"]) / tg["opp_fga"].clip(lower=1)
    tg["tov_pct"] = tg["turnover"] / tg["poss"]
    tg["opp_tov_pct"] = tg["opp_turnover"] / tg["poss"]
    tg["ft_rate"] = tg["fta"] / tg["fga"].clip(lower=1)
    tg["opp_ft_rate"] = tg["opp_fta"] / tg["opp_fga"].clip(lower=1)
    tg["fg3_rate"] = tg["fg3a"] / tg["fga"].clip(lower=1)
    tg["opp_fg3_rate"] = tg["opp_fg3a"] / tg["opp_fga"].clip(lower=1)
    tg["orb_pct"] = tg["oreb"] / (tg["oreb"] + tg["opp_dreb"]).clip(lower=1)
    tg["opp_orb_pct"] = tg["opp_oreb"] / (tg["opp_oreb"] + tg["dreb"]).clip(lower=1)
    tg["ast_rate"] = tg["ast"] / tg["fgm"].clip(lower=1)
    tg["margin"] = tg["pts"] - tg["opp_pts"]
    tg["blowout"] = (tg["margin"].abs() >= 15).astype(float)
    for s in ["pts", "reb", "ast", "fg3m", "pra"]:
        tg[f"allowed_{s}_per100"] = 100 * tg[f"opp_{s}"] / tg["poss"]

    tg = tg.sort_values(["team_id", "date", "game_id"]).reset_index(drop=True)
    tg["tkey"] = tg["team_id"].astype(str) + "_" + tg["season"].astype(str)
    form = ["pace", "off_rtg", "def_rtg", "net_rtg", "efg", "opp_efg", "tov_pct", "opp_tov_pct", "ft_rate", "opp_ft_rate",
            "fg3_rate", "opp_fg3_rate", "orb_pct", "opp_orb_pct", "ast_rate", "pts", "reb", "ast", "fg3m", "pra",
            "opp_pts", "opp_reb", "opp_ast", "opp_fg3m", "opp_pra", "stl", "blk", "margin",
            "allowed_pts_per100", "allowed_reb_per100", "allowed_ast_per100", "allowed_fg3m_per100", "allowed_pra_per100"]
    l10 = _roll(tg, "tkey", form, 10).add_suffix("_l10")
    l5 = _roll(tg, "tkey", ["pace", "off_rtg", "def_rtg", "opp_pts", "opp_reb", "opp_ast", "opp_fg3m", "opp_pra"], 5).add_suffix("_l5")
    l20 = _roll(tg, "tkey", ["blowout"], 20).add_suffix("_l20")
    tg = pd.concat([tg, l10, l5, l20], axis=1)

    team_name = box.drop_duplicates("team_id").set_index("team_id")["team"].to_dict()
    tg["d"] = pd.to_datetime(tg["date"])
    sched = {}
    for _, g in tg.groupby("tkey", sort=False):
        days = g["d"].values.astype("datetime64[D]").astype(int)
        n = len(g)
        idx = np.arange(n)
        home_arena = ARENAS[team_name[g["team_id"].iloc[0]]]
        where = [team_name[t] if h == 1 else team_name[o] for t, o, h in zip(g["team_id"], g["opp_team_id"], g["is_home"])]
        loc = np.array([ARENAS[w] for w in where])
        prev = np.vstack([np.array(home_arena)[None, :], loc[:-1]])
        away = (g["is_home"].values == 0).astype(int)
        road, stand = np.zeros(n), np.zeros(n)
        for i in range(n):
            if away[i]:
                road[i] = (road[i - 1] if i and away[i - 1] else 0) + 1
            else:
                stand[i] = (stand[i - 1] if i and not away[i - 1] else 0) + 1
        block = pd.DataFrame({
            "days_rest": np.minimum(np.diff(days, prepend=days[0] - 7), 10),
            "games_last_4": idx - np.searchsorted(days, days - 4, "left"),
            "games_last_7": idx - np.searchsorted(days, days - 7, "left"),
            "team_game_no": idx + 1,
            "travel_miles": _haversine_miles(prev[:, 0], prev[:, 1], loc[:, 0], loc[:, 1]),
            "tz_delta": loc[:, 2] - prev[:, 2],
            "altitude_ft": loc[:, 3],
            "altitude_delta": loc[:, 3] - prev[:, 3],
            "road_trip_len": road, "home_stand_len": stand,
        }, index=g.index)
        sched[_] = block
    tg = pd.concat([tg, pd.concat(sched.values()).reindex(tg.index)], axis=1)
    tg["is_b2b"] = (tg["days_rest"] <= 1).astype(int)

    # league baselines from the previous 30 days only
    daily = tg.groupby("d")[["poss", "minutes", "pts", "reb", "ast", "fg3m", "pra"]].sum().sort_index()
    past = daily.rolling("30D", closed="left").sum()
    league = pd.DataFrame({"league_pace": 48 * past["poss"] / (past["minutes"] / 5)}, index=daily.index)
    for s in ["pts", "reb", "ast", "fg3m", "pra"]:
        league[f"league_{s}_per100"] = 100 * past[s] / past["poss"]
    tg = tg.merge(league, left_on="d", right_index=True, how="left")
    return tg


def build_player_games(box: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    pg = box[box["played"]].copy()
    pg["pra"] = pg["pts"] + pg["reb"] + pg["ast"]
    tcols = ["minutes", "fga", "fta", "turnover", "reb", "opp_reb", "fgm", "fg3a", "pts", "ast", "fg3m", "pra"]
    t = tg[["game_id", "team_id", "opp_team_id", "is_home", "team_game_no", "tkey"] + tcols].rename(columns={c: f"t_{c}" for c in tcols})
    pg = pg.merge(t, on=["game_id", "team_id"], how="inner")
    m = pg["minutes"].clip(lower=1.0)
    tm5 = pg["t_minutes"].clip(lower=200) / 5
    poss_used = pg["fga"] + 0.44 * pg["fta"] + pg["turnover"]
    pg["usgposs"] = poss_used
    pg["usg"] = poss_used * tm5 / (m * (pg["t_fga"] + 0.44 * pg["t_fta"] + pg["t_turnover"]).clip(lower=1))
    pg["reb_rate"] = pg["reb"] * tm5 / (m * (pg["t_reb"] + pg["t_opp_reb"]).clip(lower=1))
    pg["ast_rate_p"] = pg["ast"] / ((m / tm5) * pg["t_fgm"] - pg["fgm"]).clip(lower=1)
    pg["min_share"] = pg["minutes"] / tm5
    pg["fga_share"] = pg["fga"] / pg["t_fga"].clip(lower=1)
    pg["fg3a_share"] = pg["fg3a"] / pg["t_fg3a"].clip(lower=1)
    pg["pts_share"] = pg["pts"] / pg["t_pts"].clip(lower=1)
    pg["pos"] = pg["position"].map(_pos_group)
    pg = pg.sort_values(["player_id", "date", "game_id"]).reset_index(drop=True)
    pg["d"] = pd.to_datetime(pg["date"])

    parts = [pg]
    base = TARGET_STATS + ["minutes"]
    for n in (3, 5, 10, 20):
        parts.append(_roll(pg, "player_id", base, n).add_suffix(f"_l{n}"))
    parts.append(_roll(pg, "player_id", base, "ewm5").add_suffix("_ewm"))
    parts.append(_roll(pg, "player_id", base, "all").add_suffix("_car"))
    parts.append(_roll(pg, "player_id", TARGET_STATS, "all", "std").add_suffix("_carsd"))
    parts.append(_roll(pg, "player_id", ["stl", "blk", "turnover"], 5).add_suffix("_l5"))
    parts.append(_roll(pg, "player_id", ["usg", "reb_rate", "ast_rate_p", "min_share", "fga_share", "fg3a_share", "pts_share"], 10).add_suffix("_l10"))
    parts.append(_roll(pg, "player_id", ["usg"], 3).add_suffix("_l3"))
    parts.append(_roll(pg, "player_id", ["usg"], 5).add_suffix("_l5"))
    sums10 = _roll(pg, "player_id", ["pts", "fga", "fgm", "fg3a", "fg3m", "fta", "pf", "turnover", "minutes"] + ["reb", "ast", "pra"], 10, "sum").add_suffix("_s10")
    parts.append(sums10)
    pg = pd.concat(parts, axis=1)
    mins10 = pg["minutes_s10"].clip(lower=1)
    pg["ts_l10"] = pg["pts_s10"] / (2 * (pg["fga_s10"] + 0.44 * pg["fta_s10"])).clip(lower=1)
    pg["efg_l10"] = (pg["fgm_s10"] + 0.5 * pg["fg3m_s10"]) / pg["fga_s10"].clip(lower=1)
    pg["fta_rate_l10"] = pg["fta_s10"] / pg["fga_s10"].clip(lower=1)
    pg["fg3a_rate_l10"] = pg["fg3a_s10"] / pg["fga_s10"].clip(lower=1)
    pg["pf_per_min_l10"] = pg["pf_s10"] / mins10
    pg["tov_per_min_l10"] = pg["turnover_s10"] / mins10
    for s in TARGET_STATS:
        pg[f"{s}_permin_l10"] = pg[f"{s}_s10"] / mins10

    # home/away and versus-opponent history, previous games only
    for s in TARGET_STATS:
        pg[f"{s}_ctx"] = pg.groupby(["player_id", "is_home"], sort=False)[s].transform(lambda x: x.shift(1).expanding().mean())
        pg[f"{s}_vsopp"] = pg.groupby(["player_id", "opp_team_id"], sort=False)[s].transform(lambda x: x.shift(1).expanding().mean())
    pg["ctx_n"] = pg.groupby(["player_id", "is_home"], sort=False).cumcount()
    pg["vsopp_n"] = pg.groupby(["player_id", "opp_team_id"], sort=False).cumcount()
    pg["career_games"] = pg.groupby("player_id", sort=False).cumcount()

    # previous season's averages
    prev = pg.groupby(["player_id", "season"])[TARGET_STATS + ["minutes"]].mean().add_suffix("_prevseason").reset_index()
    prev["season"] = prev["season"] + 1
    pg = pg.merge(prev, on=["player_id", "season"], how="left")
    pg = pg.sort_values(["player_id", "date", "game_id"]).reset_index(drop=True)

    # absences: team games missed right before this one, and in the team's last ten
    grp = pg.groupby("player_id", sort=False)
    prev_no, prev_key, prev_d = grp["team_game_no"].shift(1), grp["tkey"].shift(1), grp["d"].shift(1)
    same_stint = prev_key == pg["tkey"]
    pg["absent_streak"] = np.where(same_stint, pg["team_game_no"] - prev_no - 1, 0).clip(0, 20)
    pg["days_since_played"] = (pg["d"] - prev_d).dt.days.clip(upper=30).fillna(30)
    pg["min_last_vs_l10"] = grp["minutes"].shift(1) - pg["minutes_l10"]
    missed = np.zeros(len(pg))
    since_return = np.zeros(len(pg))
    for _, g in pg.groupby(["player_id", "tkey"], sort=False):
        nos = g["team_game_no"].to_numpy()
        i = np.arange(len(nos))
        played_in_window = i - np.searchsorted(nos, nos - ABSENCE_WINDOW, "left")
        missed[g.index] = np.minimum(nos - 1, ABSENCE_WINDOW) - played_in_window
        gaps = np.diff(nos, prepend=nos[0]) - 1
        last_gap = np.maximum.accumulate(np.where(gaps >= 2, i, -1))
        since_return[g.index] = np.where(last_gap >= 0, i - last_gap, 20).clip(0, 20)
    pg["missed_l10"] = missed
    pg["games_since_return"] = since_return

    rank = pg.groupby(["game_id", "team_id"])["minutes_l5"].rank(ascending=False, method="first")
    pg["min_rank_l5"] = rank
    return pg


def build_availability(pg: pd.DataFrame, tg: pd.DataFrame):
    """Per team-game: regular rotation players who are not playing tonight, and what they usually produce."""
    cols = ["minutes", "usgposs", "pts", "reb", "ast", "fg3m", "pra"]
    played = {}
    for (gid, tid), g in pg.groupby(["game_id", "team_id"], sort=False):
        played[(gid, tid)] = dict(zip(g["player_id"], g[cols].to_numpy()))
    rows, detail = [], {}
    order = tg.sort_values(["tkey", "date", "game_id"])
    for _, g in order.groupby("tkey", sort=False):
        hist = []
        for gid, tid in zip(g["game_id"], g["team_id"]):
            tonight = played.get((gid, tid), {})
            window = hist[-ABSENCE_WINDOW:]
            row = {"game_id": gid, "team_id": tid, "avail_ready": int(len(window) >= 3)}
            if len(window) >= 3:
                agg = defaultdict(lambda: [0, np.zeros(len(cols))])
                for gm in window:
                    for pid, v in gm.items():
                        a = agg[pid]
                        a[0] += 1
                        a[1] = a[1] + v
                per_game = {pid: a[1] / a[0] for pid, a in agg.items()}
                rotation = [pid for pid, a in agg.items() if a[0] >= max(3, len(window) / 2) and per_game[pid][0] >= ROTATION_MINUTES]
                out = [pid for pid in rotation if pid not in tonight]
                vac = np.sum([per_game[pid] for pid in out], axis=0) if out else np.zeros(len(cols))
                top = max(rotation, key=lambda pid: per_game[pid][1]) if rotation else None
                last = window[-1]
                back = [pid for pid in tonight if pid not in last and pid in agg and agg[pid][0] >= 2 and per_game[pid][0] >= ROTATION_MINUTES]
                row.update({
                    "n_out": len(out), "min_out": vac[0], "usgposs_out": vac[1], "pts_out": vac[2], "reb_out": vac[3],
                    "ast_out": vac[4], "fg3m_out": vac[5], "pra_out": vac[6],
                    "top_usg_out": int(top is not None and top in out),
                    "min_returning": float(sum(per_game[pid][0] for pid in back)),
                })
                detail[(gid, tid)] = [(pid, per_game[pid][0]) for pid in out]
            rows.append(row)
            hist.append(tonight)
    return pd.DataFrame(rows), detail, played


class _Columns(dict):
    """Feature columns collected as plain arrays and turned into one frame at the end."""

    def __setitem__(self, name, values):
        super().__setitem__(name, np.asarray(values) if name == "leg_id" else np.asarray(values, dtype=float))


def build_leg_features(legs: pd.DataFrame, book_lines: pd.DataFrame, box: pd.DataFrame) -> pd.DataFrame:
    """One row per leg; columns are named <group>__<feature>."""
    t0 = time.time()
    tg = build_team_games(box)
    pg = build_player_games(box, tg)
    avail, out_detail, played = build_availability(pg, tg)
    logger.info("tables built in %.0fs (team-games %d, player-games %d)", time.time() - t0, len(tg), len(pg))

    L = legs[["leg_id", "game_id", "player_id", "date", "prop_type", "line", "over_odds", "under_odds"]].copy()
    L["game_id"] = L["game_id"].astype(int)
    L = L.merge(pg.drop(columns=["date"]), on=["game_id", "player_id"], how="left")
    own = tg.drop(columns=["date", "season", "postseason", "d", "tkey", "is_home", "team_game_no", "opp_team_id"]).add_prefix("T_")
    L = L.merge(own, left_on=["game_id", "team_id"], right_on=["T_game_id", "T_team_id"], how="left")
    opp = tg.drop(columns=["date", "season", "d", "tkey", "opp_team_id"]).add_prefix("O_")
    L = L.merge(opp, left_on=["game_id", "opp_team_id"], right_on=["O_game_id", "O_team_id"], how="left")
    L = L.merge(avail.add_prefix("A_"), left_on=["game_id", "team_id"], right_on=["A_game_id", "A_team_id"], how="left")
    L = L.merge(avail.add_prefix("OA_"), left_on=["game_id", "opp_team_id"], right_on=["OA_game_id", "OA_team_id"], how="left")
    stat = L["prop_type"].map(PROP_STAT)

    def pick(fmt: str) -> np.ndarray:
        out = np.full(len(L), np.nan)
        for s in TARGET_STATS:
            m = (stat == s).to_numpy()
            out[m] = L.loc[m, fmt.format(s=s)].to_numpy(dtype=float)
        return out

    F = _Columns()
    F["leg_id"] = L["leg_id"]
    line = L["line"].to_numpy(dtype=float)
    d = pd.to_datetime(L["date"])

    # --- schedule
    F["sched__is_home"] = L["is_home"]
    F["sched__days_rest"] = L["T_days_rest"]
    F["sched__is_b2b"] = L["T_is_b2b"]
    F["sched__games_last_4"] = L["T_games_last_4"]
    F["sched__games_last_7"] = L["T_games_last_7"]
    F["sched__team_game_no"] = L["team_game_no"]
    F["sched__postseason"] = L["postseason"].astype(float)
    F["sched__dow"] = d.dt.dayofweek.to_numpy()
    F["sched__month"] = d.dt.month.to_numpy()
    F["sched__opp_days_rest"] = L["O_days_rest"]
    F["sched__opp_is_b2b"] = L["O_is_b2b"]
    F["sched__rest_advantage"] = L["T_days_rest"] - L["O_days_rest"]

    # --- travel
    for c in ["travel_miles", "tz_delta", "altitude_ft", "altitude_delta", "road_trip_len", "home_stand_len"]:
        F[f"travel__{c}"] = L[f"T_{c}"]
    F["travel__opp_travel_miles"] = L["O_travel_miles"]

    # --- own team form
    for c in ["pace", "off_rtg", "def_rtg", "net_rtg", "efg", "tov_pct", "ft_rate", "fg3_rate", "orb_pct", "ast_rate", "pts", "margin"]:
        F[f"team__{c}_l10"] = L[f"T_{c}_l10"]
    F["team__pace_trend"] = L["T_pace_l5"] - L["T_pace_l10"]
    F["team__off_rtg_trend"] = L["T_off_rtg_l5"] - L["T_off_rtg_l10"]
    team_tgt = pick("T_{s}_l10")
    F["team__target_l10"] = team_tgt
    F["team__player_share_of_target"] = pick("{s}_l10") / np.clip(team_tgt, 1, None)

    # --- opponent defence
    F["oppdef__def_rtg_l10"] = L["O_def_rtg_l10"]
    F["oppdef__net_rtg_l10"] = L["O_net_rtg_l10"]
    F["oppdef__pace_l10"] = L["O_pace_l10"]
    F["oppdef__efg_allowed_l10"] = L["O_opp_efg_l10"]
    F["oppdef__tov_forced_l10"] = L["O_opp_tov_pct_l10"]
    F["oppdef__ft_rate_allowed_l10"] = L["O_opp_ft_rate_l10"]
    F["oppdef__fg3_rate_allowed_l10"] = L["O_opp_fg3_rate_l10"]
    F["oppdef__orb_allowed_l10"] = L["O_opp_orb_pct_l10"]
    F["oppdef__stl_l10"] = L["O_stl_l10"]
    F["oppdef__blk_l10"] = L["O_blk_l10"]
    allowed = pick("O_opp_{s}_l10")
    F["oppdef__allowed_target_l10"] = allowed
    F["oppdef__allowed_target_trend"] = pick("O_opp_{s}_l5") - allowed
    F["oppdef__allowed_target_vs_league"] = pick("O_allowed_{s}_per100_l10") / np.clip(pick("T_league_{s}_per100"), 1e-6, None) - 1

    # allowed to the player's position group, relative to the league's last 30 days
    pos_tot = pg.groupby(["game_id", "opp_team_id", "pos", "date", "season"], as_index=False)[TARGET_STATS].sum()
    pos_tot = pos_tot.rename(columns={"opp_team_id": "def_team_id"}).sort_values(["def_team_id", "pos", "date", "game_id"]).reset_index(drop=True)
    pos_tot["pkey"] = pos_tot["def_team_id"].astype(str) + "_" + pos_tot["season"].astype(str) + "_" + pos_tot["pos"]
    pos_roll = _roll(pos_tot, "pkey", TARGET_STATS, 15).add_suffix("_posallowed")
    pos_tot = pd.concat([pos_tot, pos_roll], axis=1)
    pos_tot["d"] = pd.to_datetime(pos_tot["date"])
    lg = pos_tot.groupby(["pos", "d"])[TARGET_STATS].agg(["sum", "count"])
    lg_parts = []
    for p, g in lg.groupby(level=0):
        g = g.droplevel(0).sort_index()
        past = g.rolling("30D", closed="left").sum()
        frame = pd.DataFrame({f"{s}_posleague": past[(s, "sum")] / past[(s, "count")] for s in TARGET_STATS})
        frame["pos"] = p
        lg_parts.append(frame.reset_index())
    pos_tot = pos_tot.merge(pd.concat(lg_parts), on=["pos", "d"], how="left")
    keep = ["game_id", "def_team_id", "pos"] + [f"{s}_posallowed" for s in TARGET_STATS] + [f"{s}_posleague" for s in TARGET_STATS]
    L = L.merge(pos_tot[keep].add_prefix("P_"), left_on=["game_id", "opp_team_id", "pos"],
                right_on=["P_game_id", "P_def_team_id", "P_pos"], how="left")
    F["oppdef__allowed_target_to_position"] = pick("P_{s}_posallowed") / np.clip(pick("P_{s}_posleague"), 1e-6, None) - 1

    # --- game environment
    proj_pace = (L["T_pace_l10"] + L["O_pace_l10"]) / 2
    F["env__projected_pace"] = proj_pace
    F["env__pace_vs_league"] = proj_pace - L["T_league_pace"]
    exp_margin = (L["T_net_rtg_l10"] - L["O_net_rtg_l10"]) / 2 + np.where(L["is_home"] == 1, 2.5, -2.5)
    F["env__expected_margin"] = exp_margin
    F["env__abs_expected_margin"] = np.abs(exp_margin)
    F["env__blowout_rate_l20"] = L["T_blowout_l20"]
    F["env__opp_blowout_rate_l20"] = L["O_blowout_l20"]
    pts_legs = L[L["prop_type"] == "points"][["game_id", "team_id", "line", "pts_l10"]].copy()
    pts_legs["gap"] = pts_legs["line"] - pts_legs["pts_l10"]
    team_lines = pts_legs.groupby(["game_id", "team_id"]).agg(line_sum=("line", "sum"), gap=("gap", "sum"), n=("line", "size"))
    own_key = pd.MultiIndex.from_arrays([L["game_id"], L["team_id"]])
    opp_key = pd.MultiIndex.from_arrays([L["game_id"], L["opp_team_id"]])
    F["env__team_points_lines"] = team_lines["line_sum"].reindex(own_key).to_numpy()
    F["env__team_points_lines_n"] = team_lines["n"].reindex(own_key).to_numpy()
    F["env__team_lines_vs_form"] = team_lines["gap"].reindex(own_key).to_numpy()
    F["env__opp_lines_vs_form"] = team_lines["gap"].reindex(opp_key).to_numpy()

    # --- role
    for c in ["usg_l5", "usg_l10", "reb_rate_l10", "ast_rate_p_l10", "min_share_l10", "fga_share_l10", "fg3a_share_l10",
              "pts_share_l10", "ts_l10", "efg_l10", "fta_rate_l10", "fg3a_rate_l10", "pf_per_min_l10", "tov_per_min_l10", "min_rank_l5"]:
        F[f"role__{c}"] = L[c]
    F["role__usg_trend"] = L["usg_l3"] - L["usg_l10"]
    F["role__starter_proxy"] = (L["min_rank_l5"] <= 5).astype(float)

    # --- return from absence
    for c in ["absent_streak", "missed_l10", "days_since_played", "min_last_vs_l10", "games_since_return"]:
        F[f"ret__{c}"] = L[c]
    F["ret__early_season"] = (L["team_game_no"] <= 5).astype(float)

    # --- extended player form
    ewm, car = pick("{s}_ewm"), pick("{s}_car")
    F["form__target_ewm"] = ewm
    F["form__line_minus_ewm"] = line - ewm
    F["form__trend_l5_l20"] = pick("{s}_l5") - pick("{s}_l20")
    F["form__z_l5_vs_career"] = (pick("{s}_l5") - car) / np.clip(pick("{s}_carsd"), 0.5, None)
    ctx = np.where(L["ctx_n"] >= 3, pick("{s}_ctx") - car, 0.0)
    F["form__home_away_delta"] = ctx
    vs = np.where(L["vsopp_n"] >= 2, pick("{s}_vsopp") - car, 0.0)
    F["form__vs_opponent_delta"] = vs
    F["form__vs_opponent_n"] = L["vsopp_n"].clip(upper=12)
    prevseason = pick("{s}_prevseason")
    F["form__prev_season_avg"] = prevseason
    F["form__line_minus_prev_season"] = line - prevseason
    F["form__minutes_ewm"] = L["minutes_ewm"]
    F["form__minutes_vs_prev_season"] = L["minutes_l10"] - L["minutes_prevseason"]
    F["form__rate_times_minutes_ewm"] = pick("{s}_permin_l10") * L["minutes_ewm"].to_numpy()
    F["form__line_minus_rate_proj"] = line - F["form__rate_times_minutes_ewm"]
    F["form__career_games"] = L["career_games"].clip(upper=250)
    for c in ["pts_l5", "reb_l5", "ast_l5", "fg3m_l5", "stl_l5", "blk_l5", "turnover_l5"]:
        F[f"form__{c}"] = L[c]

    # --- availability (uses who played tonight)
    for c in ["n_out", "min_out", "usgposs_out", "top_usg_out", "min_returning"]:
        F[f"avail__{c}"] = L[f"A_{c}"]
    tgt_out = np.full(len(L), np.nan)
    for s in TARGET_STATS:
        m = (stat == s).to_numpy()
        tgt_out[m] = L.loc[m, f"A_{s}_out"].to_numpy(dtype=float)
    F["avail__target_out"] = tgt_out
    F["avail__opp_n_out"] = L["OA_n_out"]
    F["avail__opp_min_out"] = L["OA_min_out"]
    F["avail__opp_top_usg_out"] = L["OA_top_usg_out"]
    F["avail__min_out_x_share"] = L["A_min_out"] * L["min_share_l10"]
    F["avail__target_out_x_share"] = tgt_out * F["team__player_share_of_target"]

    pos_of = pg.drop_duplicates("player_id").set_index("player_id")["pos"].to_dict()
    hist = {}
    for pid, g in pg.groupby("player_id", sort=False):
        hist[pid] = (g["game_id"].to_numpy(), g["team_id"].to_numpy(), g["tkey"].to_numpy(), g["minutes"].to_numpy(),
                     {s: g[s].to_numpy() for s in TARGET_STATS})
    same_pos, wowy_n, wowy_delta, wowy_min = np.zeros(len(L)), np.zeros(len(L)), np.zeros(len(L)), np.zeros(len(L))
    gids, pids, tids, tkeys = L["game_id"].to_numpy(), L["player_id"].to_numpy(), L["team_id"].to_numpy(), L["tkey"].to_numpy()
    stats_arr = stat.to_numpy()
    for i in range(len(L)):
        outs = out_detail.get((gids[i], tids[i]))
        if not outs:
            continue
        me = pos_of.get(pids[i], "U")
        same_pos[i] = sum(mpg for q, mpg in outs if pos_of.get(q, "U") == me)
        q = max(outs, key=lambda x: x[1])[0]
        h_g, h_t, h_k, h_m, h_s = hist[pids[i]]
        cut = int(np.argmax(h_g == gids[i]))
        prior = np.arange(cut)[h_k[:cut] == tkeys[i]]
        if len(prior) < 4:
            continue
        with_q = np.array([q in played.get((h_g[j], h_t[j]), ()) for j in prior])
        n_wo = int((~with_q).sum())
        if n_wo == 0 or with_q.sum() == 0:
            continue
        vals, mins = h_s[stats_arr[i]][prior], h_m[prior]
        rate_wo = vals[~with_q].sum() / max(mins[~with_q].sum(), 1.0)
        rate_w = vals[with_q].sum() / max(mins[with_q].sum(), 1.0)
        shrink = n_wo / (n_wo + 3.0)
        wowy_n[i] = min(n_wo, 15)
        wowy_delta[i] = shrink * (rate_wo - rate_w) * 36
        wowy_min[i] = shrink * (mins[~with_q].mean() - mins[with_q].mean())
    F["avail__same_position_min_out"] = same_pos
    F["avail__without_teammate_n"] = wowy_n
    F["avail__without_teammate_target_per36"] = wowy_delta
    F["avail__without_teammate_minutes"] = wowy_min

    # --- market context beyond the price of this prop
    io, iu = _imp(L["over_odds"]), _imp(L["under_odds"])
    fair = io / (io + iu)
    F["mkt__dk_hold"] = io + iu - 1
    bl = book_lines.merge(L[["leg_id", "line"]].rename(columns={"line": "dk_line"}), on="leg_id")
    same = bl[(bl["line"] - bl["dk_line"]).abs() < 1e-6]
    six = same[same["book"].isin(SIX_BOOKS)].groupby("leg_id")[["over_odds", "under_odds"]].agg(["max", "size"])
    F["mkt__best_under_lift_six"] = _dec(L["leg_id"].map(six[("under_odds", "max")]).fillna(L["under_odds"])) / _dec(L["under_odds"]) - 1
    F["mkt__best_over_lift_six"] = _dec(L["leg_id"].map(six[("over_odds", "max")]).fillna(L["over_odds"])) / _dec(L["over_odds"]) - 1
    F["mkt__n_books_six"] = L["leg_id"].map(six[("over_odds", "size")]).fillna(1).to_numpy()
    pin = same[same["book"] == "pinnacle"].drop_duplicates("leg_id").set_index("leg_id")
    pio, piu = _imp(pin["over_odds"]), _imp(pin["under_odds"])
    pin_fair = pd.Series(pio / (pio + piu), index=pin.index)
    F["mkt__pinnacle_gap"] = (L["leg_id"].map(pin_fair) - fair).to_numpy()
    F["mkt__pinnacle_on_line"] = L["leg_id"].isin(pin.index).astype(float).to_numpy()
    L["_fair"] = fair
    by_player = L.groupby(["game_id", "player_id"])["_fair"].agg(["sum", "size"])
    pk = pd.MultiIndex.from_arrays([L["game_id"], L["player_id"]])
    tot, cnt = by_player["sum"].reindex(pk).to_numpy(), by_player["size"].reindex(pk).to_numpy()
    F["mkt__player_other_props_fair_over"] = np.where(cnt > 1, (tot - fair) / np.clip(cnt - 1, 1, None), np.nan)
    F["mkt__player_n_props"] = cnt
    by_team = L.groupby(["game_id", "team_id", "prop_type"])["_fair"].agg(["sum", "size"])
    tk = pd.MultiIndex.from_arrays([L["game_id"], L["team_id"], L["prop_type"]])
    tot, cnt = by_team["sum"].reindex(tk).to_numpy(), by_team["size"].reindex(tk).to_numpy()
    F["mkt__teammates_same_prop_fair_over"] = np.where(cnt > 1, (tot - fair) / np.clip(cnt - 1, 1, None), np.nan)

    F = pd.DataFrame(F)
    logger.info("leg features: %d rows x %d columns in %.0fs", len(F), F.shape[1] - 1, time.time() - t0)
    return F


def build():
    LAB_DIR.mkdir(parents=True, exist_ok=True)
    F = build_leg_features(pd.read_parquet(LEGS_FILE), pd.read_parquet(BOOK_LINES_FILE), load_box())
    F.to_parquet(FEATURES_FILE, index=False)
    groups = defaultdict(int)
    for c in F.columns[1:]:
        groups[c.split("__")[0]] += 1
    logger.info("saved %s; groups: %s", FEATURES_FILE, dict(groups))


def leak_test(seed: int = 7) -> bool:
    """Scramble every box score on one date; features for bets on that date must not move. One date per season."""
    box = load_box()
    legs, bl = pd.read_parquet(LEGS_FILE), pd.read_parquet(BOOK_LINES_FILE)
    rng = np.random.default_rng(seed)
    stats = ["pts", "reb", "ast", "fg3m", "fg3a", "fgm", "fga", "ftm", "fta", "stl", "blk", "turnover", "oreb", "dreb", "pf", "minutes"]
    base = build_leg_features(legs, bl, box).set_index("leg_id")
    clean = True
    for season, grp in legs.groupby("season"):
        dates = np.sort(grp["date"].unique())
        day = rng.choice(dates[len(dates) // 4: -len(dates) // 4])
        scrambled = box.copy()
        m = (scrambled["date"] == day) & scrambled["played"]
        scrambled.loc[m, stats] = np.maximum(rng.permutation(scrambled.loc[m, stats].to_numpy()) * 2 + 1, 1)
        alt = build_leg_features(legs, bl, scrambled).set_index("leg_id")
        same_day = legs.loc[legs["date"] == day, "leg_id"]
        a, b = base.loc[same_day], alt.loc[same_day]
        moved = [c for c in a.columns if not np.allclose(a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float), equal_nan=True, atol=1e-9)]
        later = legs.loc[(legs["date"] > day) & (legs["season"] == season), "leg_id"]
        changed = (~np.isclose(base.loc[later].to_numpy(dtype=float), alt.loc[later].to_numpy(dtype=float), equal_nan=True)).any(axis=1).mean()
        print(f"{day}: scrambled {int(m.sum())} box-score lines; {len(same_day)} bets that day; "
              f"features that moved for those bets: {moved if moved else 'none'}; later bets that season whose features changed: {changed * 100:.0f}%")
        clean = clean and not moved
    return clean


# =============================================================================
# EXPERIMENT HARNESS
# Every model starts from DraftKings' de-vigged closing probability and learns a correction,
# so a group of features only scores if it knows something the price does not.
# =============================================================================

GROUPS = ["sched", "travel", "team", "oppdef", "env", "role", "ret", "form", "avail", "mkt"]
HOLDOUT_START = "2026-02-01"
SEEDS = (11, 23)
VALID_SHARE = 0.15
MAX_ROUNDS, EARLY_STOP = 1500, 100
LGB_PARAMS = {"objective": "binary", "learning_rate": 0.04, "num_leaves": 15, "max_depth": 5, "min_data_in_leaf": 200,
              "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0, "verbose": -1}
PASS_Z = -2.0
V8_MARKET = {
    "line", "over_odds", "under_odds", "implied_over_prob", "implied_under_prob", "dk_devigged_over", "dk_devigged_under",
    "vig", "odds_skew", "consensus_over_prob", "consensus_under_prob", "dk_vs_consensus_prob", "consensus_line",
    "dk_vs_consensus_line", "line_range", "num_books", "num_sharp_books", "num_soft_books", "soft_book_score",
    "pinnacle_over_prob", "pinnacle_available", "sharp_consensus_over_prob", "best_over_odds_lift", "best_under_odds_lift",
    "odds_spread_width", "sharp_soft_disagreement", "market_efficiency", "num_books_total", "books_line_above",
    "books_line_below", "pinnacle_line_diff", "is_points", "is_rebounds", "is_assists", "is_threes", "is_pts_rebs_asts",
    "is_steals", "is_blocks",
}
PROTOCOL = {
    "question": "Does adding a feature group improve out-of-sample log-loss over a model that already has the market?",
    "start_point": "every model is a correction to DraftKings' de-vigged closing probability (LightGBM init_score)",
    "bases": {"market": "price and cross-book inputs only", "market+history": "market plus the player-history inputs v8 already uses"},
    "folds": {"2024-25": "train 2023-24, test 2024-25",
              "2025-26 to Jan": f"train 2023-24 and 2024-25, test 2025-10-21 to {HOLDOUT_START} (exclusive)"},
    "early_stopping": f"last {int(VALID_SHARE * 100)}% of each training window by date; never the test window",
    "metric": "mean log-loss difference versus the same base, paired per bet, standard error clustered by date",
    "pass_rule": f"difference below zero in both folds and pooled z at or below {PASS_Z}",
    "controls": "each group is also run with its rows shuffled within date and prop type; those must not pass",
    "holdout": f"bets from {HOLDOUT_START} to 2026-06-13 are not read during screening; the final step reads them once",
    "final_candidate": "market+history plus every group that passed on that base (none passing means no additions)",
    "seeds": list(SEEDS), "model_params": LGB_PARAMS, "max_rounds": MAX_ROUNDS, "early_stop_rounds": EARLY_STOP,
}


def load_matrix() -> pd.DataFrame:
    from train_v8 import DATASETS

    cache = DATASETS["clean"]["cache_dir"]
    X = pd.read_parquet(cache / "features.parquet")
    meta = pd.read_parquet(cache / "metadata.parquet")[["leg_id", "date", "player", "prop_type", "line", "actual", "over_odds", "under_odds"]]
    X = X.rename(columns={c: (f"v8m__{c}" if c in V8_MARKET else f"v8p__{c}") for c in X.columns})
    df = pd.concat([meta.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    df = df.merge(pd.read_parquet(FEATURES_FILE), on="leg_id")
    df = df.merge(pd.read_parquet(LEGS_FILE)[["leg_id", "season"]], on="leg_id")
    return _finish_matrix(df)


def _finish_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df["y"] = (df["actual"] > df["line"]).astype(int)
    p0 = df["v8m__dk_devigged_over"].clip(0.02, 0.98)
    df["base_logit"] = np.log(p0 / (1 - p0))
    return df.sort_values(["date", "leg_id"]).reset_index(drop=True)


def group_cols(df: pd.DataFrame, group: str) -> list:
    return [c for c in df.columns if c.startswith(group + "__")]


def fit_predict(X_train, y_train, base_train, X_test, base_test):
    """Train on the earlier part of the window, stop on its last slice, return test probabilities."""
    import lightgbm as lgb

    cut = int(len(X_train) * (1 - VALID_SHARE))
    raw, rounds = np.zeros(len(X_test)), []
    for seed in SEEDS:
        dtr = lgb.Dataset(X_train[:cut], y_train[:cut], init_score=base_train[:cut], free_raw_data=False)
        dva = lgb.Dataset(X_train[cut:], y_train[cut:], init_score=base_train[cut:], reference=dtr)
        booster = lgb.train({**LGB_PARAMS, "seed": seed}, dtr, MAX_ROUNDS, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        raw += booster.predict(X_test, raw_score=True, num_iteration=booster.best_iteration)
        rounds.append(booster.best_iteration)
    return 1 / (1 + np.exp(-(base_test + raw / len(SEEDS)))), rounds


def log_loss_rows(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired(diff, dates):
    """Mean of a per-bet difference and its standard error with each date as one cluster."""
    by = pd.DataFrame({"date": dates, "d": diff}).groupby("date")["d"].agg(["sum", "size"])
    n, g, mean = len(diff), len(by), float(np.mean(diff))
    se = float(np.sqrt(g / (g - 1) * ((by["sum"] - mean * by["size"]) ** 2).sum()) / n)
    return mean, se


def shuffle_within(df: pd.DataFrame, cols: list, rng) -> np.ndarray:
    """The same columns with whole rows swapped among bets of the same date and prop type."""
    key = (df["date"].astype(str) + "|" + df["prop_type"]).to_numpy()
    order = np.argsort(key, kind="stable")
    src = order[np.lexsort((rng.random(len(df)), key[order]))]
    values = df[cols].to_numpy(dtype=np.float32)
    out = np.empty_like(values)
    out[order] = values[src]
    return out


def screen():
    LAB_DIR.mkdir(parents=True, exist_ok=True)
    PROTOCOL_FILE.write_text(json.dumps({**PROTOCOL, "written": datetime.now().isoformat(timespec="seconds")}, indent=1))
    df = load_matrix()
    y, base, dates, props = df["y"].to_numpy(), df["base_logit"].to_numpy(), df["date"].to_numpy(), df["prop_type"].to_numpy()
    s = df["season"].to_numpy()
    early = (s == 2025) & (dates < HOLDOUT_START)
    folds = {"2024-25": (s == 2023, s == 2024), "2025-26 to Jan": ((s == 2023) | (s == 2024), early)}
    blocks = {g: df[group_cols(df, g)].to_numpy(dtype=np.float32) for g in GROUPS + ["v8m", "v8p"]}
    rng = np.random.default_rng(2026)
    shuffled = {g: shuffle_within(df, group_cols(df, g), rng) for g in GROUPS}
    bases = {"market": ["v8m"], "market+history": ["v8m", "v8p"]}

    def losses(parts, swap=None):
        X = np.hstack([shuffled[p] if p == swap else blocks[p] for p in parts])
        out, rounds = {}, {}
        for name, (tr, te) in folds.items():
            p, r = fit_predict(X[tr], y[tr], base[tr], X[te], base[te])
            out[name], rounds[name] = log_loss_rows(p, y[te]), r
        return out, rounds, X.shape[1]

    def compare(cand, ref):
        row, pooled_d, pooled_dates, pooled_props = {}, [], [], []
        for name, (_, te) in folds.items():
            d = cand[name] - ref[name]
            row[name] = float(d.mean()) * 1e4
            pooled_d.append(d); pooled_dates.append(dates[te]); pooled_props.append(props[te])
        d, dd, pp = np.concatenate(pooled_d), np.concatenate(pooled_dates), np.concatenate(pooled_props)
        mean, se = paired(d, dd)
        row.update({"pooled": mean * 1e4, "se": se * 1e4, "z": mean / se if se else 0.0,
                    "by_prop": {p: float(d[pp == p].mean()) * 1e4 for p in sorted(set(pp))}})
        row["pass"] = bool(all(row[f] < 0 for f in folds) and row["z"] <= PASS_Z)
        return row

    results, t0 = [], time.time()
    price = {name: log_loss_rows(1 / (1 + np.exp(-base[te])), y[te]) for name, (_, te) in folds.items()}
    base_loss = {}
    print(f"bets: {len(df):,}; screening folds: " + ", ".join(f"{k} ({int(te.sum()):,} test bets)" for k, (_, te) in folds.items()))
    print("log-loss differences are in units of 0.0001; negative is better\n")
    header = f"{'configuration':<44}{'inputs':>7}{'2024-25':>10}{'2025-26':>10}{'pooled':>9}{'±95%':>7}{'z':>7}  result"
    print(header)
    for bname, parts in bases.items():
        base_loss[bname], rounds, n = losses(parts)
        row = compare(base_loss[bname], price)
        results.append({"config": f"{bname} vs raw price", "base": "raw price", "group": bname, "kind": "base", "inputs": n, "rounds": rounds, **row})
        print(f"{bname + ' vs raw price':<44}{n:>7}{row['2024-25']:>+10.2f}{row['2025-26 to Jan']:>+10.2f}{row['pooled']:>+9.2f}{1.96 * row['se']:>7.2f}{row['z']:>+7.1f}")
    row = compare(base_loss["market+history"], base_loss["market"])
    results.append({"config": "history on top of market", "base": "market", "group": "v8p", "kind": "real", "inputs": blocks["v8p"].shape[1], **row})
    print(f"{'history on top of market':<44}{blocks['v8p'].shape[1]:>7}{row['2024-25']:>+10.2f}{row['2025-26 to Jan']:>+10.2f}{row['pooled']:>+9.2f}{1.96 * row['se']:>7.2f}{row['z']:>+7.1f}  {'PASS' if row['pass'] else ''}")
    for bname, parts in bases.items():
        print(f"\n--- each group added to: {bname} ---")
        for kind in ("real", "shuffled"):
            for g in GROUPS:
                cand, rounds, n = losses(parts + [g], swap=g if kind == "shuffled" else None)
                row = compare(cand, base_loss[bname])
                label = f"+ {g}" + (" (shuffled control)" if kind == "shuffled" else "")
                results.append({"config": f"{bname} {label}", "base": bname, "group": g, "kind": kind, "inputs": blocks[g].shape[1], "rounds": rounds, **row})
                print(f"{label:<44}{blocks[g].shape[1]:>7}{row['2024-25']:>+10.2f}{row['2025-26 to Jan']:>+10.2f}{row['pooled']:>+9.2f}{1.96 * row['se']:>7.2f}{row['z']:>+7.1f}  {'PASS' if row['pass'] else ''}", flush=True)
    cand, rounds, n = losses(["v8m", "v8p"] + GROUPS)
    row = compare(cand, base_loss["market+history"])
    results.append({"config": "market+history + all ten groups", "base": "market+history", "group": "ALL", "kind": "real", "inputs": n, "rounds": rounds, **row})
    print(f"\n{'+ all ten groups (on market+history)':<44}{n:>7}{row['2024-25']:>+10.2f}{row['2025-26 to Jan']:>+10.2f}{row['pooled']:>+9.2f}{1.96 * row['se']:>7.2f}{row['z']:>+7.1f}  {'PASS' if row['pass'] else ''}")
    SCREEN_FILE.write_text(json.dumps({"finished": datetime.now().isoformat(timespec="seconds"), "minutes": round((time.time() - t0) / 60, 1), "results": results}, indent=1))
    print(f"\nsaved {SCREEN_FILE} ({(time.time() - t0) / 60:.1f} min)")


# =============================================================================
# FEATURE-LEVEL SCREEN
# For each single feature: is what the market-only model gets wrong related to it, out of sample?
# =============================================================================

FEATURE_SCREEN_FILE = LAB_DIR / "feature_screen.parquet"
FEATURE_PROTOCOL_FILE = LAB_DIR / "protocol_feature_level.json"
FDR_Q = 0.10
FEATURE_PROTOCOL = {
    "question": "Does any single feature explain the errors of the market-only model on bets it was not trained on?",
    "statistic": "sum over test bets of (outcome - market-model probability) x standardized feature, both centred within "
                 "date and prop type; variance from per-date sums; pooled across the two screening folds",
    "standardization": "per prop type from the training window only, clipped at its 1st and 99th percentiles; missing values contribute nothing",
    "selection_rule": f"Benjamini-Hochberg at q={FDR_Q} on the pooled two-sided p-values, and the same sign in both folds",
    "control": "the same statistic with feature rows swapped among bets of the same date and prop type",
    "holdout": f"not read here; bets from {HOLDOUT_START}",
    "final_candidate": "market-only model as offset plus an L2 logistic correction (penalty 10, no tuning) on the selected features; "
                       "if nothing is selected there is no second candidate",
    "amendment_1": "First run centred within prop type only. Its shuffled control came out over-dispersed (z spread 1.21, "
                   "expected 1.0) because slow drift in the under rate within a season loads onto any feature that trends "
                   "over the season. Centring within date removes that and leaves only same-day comparisons, which is also "
                   "what picking bets uses. Changed before the holdout was read; features constant within a date are no longer testable here.",
}


def _standardize(train: np.ndarray, test: np.ndarray, prop_train: np.ndarray, prop_test: np.ndarray) -> np.ndarray:
    """Test-window features scaled per prop type by training-window statistics; missing stays missing."""
    out = np.full(test.shape, np.nan, dtype=np.float64)
    for p in np.unique(prop_test):
        a, b = train[prop_train == p].astype(np.float64), test[prop_test == p].astype(np.float64)
        lo, hi = np.nanpercentile(a, 1, axis=0), np.nanpercentile(a, 99, axis=0)
        a = np.clip(a, lo, hi)
        mu, sd = np.nanmean(a, axis=0), np.nanstd(a, axis=0)
        out[prop_test == p] = (np.clip(b, lo, hi) - mu) / np.where(sd > 1e-9, sd, np.nan)
    return out


def _center_within(values: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Subtract each group's mean (ignoring missing); missing values come back as zero."""
    values = values.reshape(len(values), -1).astype(np.float64)
    order = np.argsort(codes, kind="stable")
    v, k = values[order], codes[order]
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    valid = ~np.isnan(v)
    sums = np.add.reduceat(np.where(valid, v, 0.0), starts, axis=0)
    counts = np.add.reduceat(valid.astype(np.float64), starts, axis=0)
    group = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(k)]))
    out = np.empty_like(values)
    out[order] = np.where(valid, v - (sums / np.maximum(counts, 1))[group], 0.0)
    return out


def _score_stats(resid: np.ndarray, w: np.ndarray, Z: np.ndarray, prop: np.ndarray, dates: np.ndarray):
    """Score, its date-clustered variance and the information, per feature column, from same-day comparisons."""
    codes = pd.factorize(pd.Series(dates).astype(str) + "|" + pd.Series(prop))[0]
    r = _center_within(resid, codes)[:, 0]
    Zc = _center_within(Z, codes)
    contrib = r[:, None] * Zc
    starts = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1]])
    by_date = np.add.reduceat(contrib, starts, axis=0)
    return contrib.sum(axis=0), (by_date ** 2).sum(axis=0), (w[:, None] * Zc ** 2).sum(axis=0)


def _market_folds(df: pd.DataFrame):
    y, base, dates, s = df["y"].to_numpy(), df["base_logit"].to_numpy(), df["date"].to_numpy(), df["season"].to_numpy()
    early = (s == 2025) & (dates < HOLDOUT_START)
    return y, base, dates, {"2024-25": (s == 2023, s == 2024), "2025-26 to Jan": ((s == 2023) | (s == 2024), early)}


def feature_screen():
    from scipy.stats import norm

    LAB_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_PROTOCOL_FILE.write_text(json.dumps({**FEATURE_PROTOCOL, "written": datetime.now().isoformat(timespec="seconds")}, indent=1))
    df = load_matrix()
    y, base, dates, folds = _market_folds(df)
    prop = df["prop_type"].to_numpy()
    cols = [c for g in ["v8p"] + GROUPS for c in group_cols(df, g)]
    Xf = df[cols].to_numpy(dtype=np.float32)
    Xm = df[group_cols(df, "v8m")].to_numpy(dtype=np.float32)
    rng = np.random.default_rng(404)
    parts = {k: {"U": 0.0, "V": 0.0, "I": 0.0} for k in ("real", "null")}
    per_fold, per_prop = {}, defaultdict(lambda: {"U": 0.0, "V": 0.0})
    for name, (tr, te) in folds.items():
        p0, _ = fit_predict(Xm[tr], y[tr], base[tr], Xm[te], base[te])
        resid, w = y[te] - p0, p0 * (1 - p0)
        Z = _standardize(Xf[tr], Xf[te], prop[tr], prop[te])
        U, V, info = _score_stats(resid, w, Z, prop[te], dates[te])
        per_fold[name] = U / np.sqrt(V)
        for k, v in zip(("U", "V", "I"), (U, V, info)):
            parts["real"][k] = parts["real"][k] + v
        sub = df.loc[te, ["date", "prop_type"]].reset_index(drop=True)
        key = (sub["date"].astype(str) + "|" + sub["prop_type"]).to_numpy()
        order = np.argsort(key, kind="stable")
        Zs = np.empty_like(Z)
        Zs[order] = Z[order[np.lexsort((rng.random(len(sub)), key[order]))]]
        Un, Vn, In = _score_stats(resid, w, Zs, prop[te], dates[te])
        for k, v in zip(("U", "V", "I"), (Un, Vn, In)):
            parts["null"][k] = parts["null"][k] + v
        for p in np.unique(prop[te]):
            m = prop[te] == p
            Up, Vp, _ = _score_stats(resid[m], w[m], Z[m], prop[te][m], dates[te][m])
            per_prop[p]["U"] = per_prop[p]["U"] + Up
            per_prop[p]["V"] = per_prop[p]["V"] + Vp

    with np.errstate(divide="ignore", invalid="ignore"):
        z = parts["real"]["U"] / np.sqrt(parts["real"]["V"])
        z_null = parts["null"]["U"] / np.sqrt(parts["null"]["V"])
        out = pd.DataFrame({"feature": cols, "group": [c.split("__")[0] for c in cols], "z": z,
                            "pct_points_per_sd": 25 * parts["real"]["U"] / parts["real"]["I"], "z_shuffled": z_null})
        for name in folds:
            out[f"z {name}"] = per_fold[name]
        for p, d in per_prop.items():
            out[f"z {p}"] = d["U"] / np.sqrt(d["V"])
    # a feature that never varies among same-day bets cannot be tested here
    out = out[np.isfinite(out["z"]) & (parts["real"]["I"] > 1e-6)].reset_index(drop=True)
    z_null = out["z_shuffled"].to_numpy()
    out["p"] = 2 * norm.sf(np.abs(out["z"]))
    ranked = out["p"].rank(method="first")
    passed = out["p"] <= FDR_Q * ranked / len(out)
    cutoff = out.loc[passed, "p"].max() if passed.any() else -1.0
    fold_cols = [f"z {name}" for name in folds]
    out["same_sign"] = (out[fold_cols[0]] * out[fold_cols[1]]).fillna(0) > 0
    out["selected"] = (out["p"] <= cutoff) & out["same_sign"]
    out = out.sort_values("p").reset_index(drop=True)
    out.to_parquet(FEATURE_SCREEN_FILE, index=False)

    n_te = sum(int(te.sum()) for _, te in folds.values())
    print(f"features tested: {len(out)} on {n_te:,} out-of-sample bets (two screening folds)")
    print(f"shuffled controls: z has mean {z_null.mean():+.2f}, sd {z_null.std():.2f}, largest |z| {np.abs(z_null).max():.2f}"
          f"   (a sound test gives about 0, 1 and {norm.isf(0.5 / len(out)):.1f})")
    print(f"real features:     z has mean {out['z'].mean():+.2f}, sd {out['z'].std():.2f}, largest |z| {out['z'].abs().max():.2f};"
          f"  |z| >= 2: {int((out['z'].abs() >= 2).sum())} (chance alone: about {0.0455 * len(out):.0f})")
    print(f"selected by the pre-set rule (FDR {FDR_Q}, same sign in both folds): {int(out['selected'].sum())}")
    print(f"\n{'feature':<42}{'z':>7}{'2024-25':>9}{'2025-26':>9}{'pts/SD':>8}{'shuffled z':>12}  selected")
    for r in out.head(25).to_dict("records"):
        print(f"{r['feature']:<42}{r['z']:>+7.2f}{r[fold_cols[0]]:>+9.2f}{r[fold_cols[1]]:>+9.2f}{r['pct_points_per_sd']:>+8.2f}"
              f"{r['z_shuffled']:>+12.2f}  {'YES' if r['selected'] else ''}")
    prop_cols = [c for c in out.columns if c.startswith("z ") and c not in fold_cols]
    flat = out.melt(id_vars="feature", value_vars=prop_cols, var_name="prop", value_name="zp")
    print(f"\nper-prop tests: {len(flat)}; |z| >= 3: {int((flat['zp'].abs() >= 3).sum())} (chance alone: about {0.0027 * len(flat):.0f}); "
          f"largest |z| {flat['zp'].abs().max():.2f} (chance alone: about {norm.isf(0.5 / len(flat)):.1f})")
    for r in flat.reindex(flat["zp"].abs().sort_values(ascending=False).index).head(8).itertuples(index=False):
        print(f"   {r.feature:<42}{r.prop[2:]:<16}{r.zp:>+6.2f}")
    print(f"\nsaved {FEATURE_SCREEN_FILE}")


# =============================================================================
# FINAL: one read of the held-out bets
# =============================================================================

# Declared after the feature screen and before the holdout was read: the opponent-defence
# features that narrowly missed the selection rule (z 3.03, 2.94, 2.26). Exploratory.
SECONDARY_EXTRA = ["oppdef__allowed_target_l10", "oppdef__allowed_target_vs_league", "oppdef__allowed_target_to_position"]
RIDGE_PENALTY = 10.0


def _ridge_offset_logit(X: np.ndarray, y: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Logistic coefficients for X given a fixed offset, with an L2 penalty (Newton steps)."""
    beta = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-(offset + X @ beta)))
        grad = X.T @ (p - y) + RIDGE_PENALTY * beta
        hess = (X * (p * (1 - p))[:, None]).T @ X + RIDGE_PENALTY * np.eye(X.shape[1])
        step = np.linalg.solve(hess, grad)
        beta -= step
        if np.abs(step).max() < 1e-9:
            break
    return beta


def final():
    from clean_backtest import fmt, score

    screen_table = pd.read_parquet(FEATURE_SCREEN_FILE)
    primary = screen_table.loc[screen_table["selected"], "feature"].tolist()
    secondary = primary + [f for f in SECONDARY_EXTRA if f not in primary]
    (LAB_DIR / "protocol_final.json").write_text(json.dumps({
        "written": datetime.now().isoformat(timespec="seconds"),
        "holdout": f"bets from {HOLDOUT_START}; read once by this step",
        "candidates": {
            "market-only model": "trees on market inputs, trained on everything before the holdout",
            "market + player history": "the group screen's candidate (no new group passed, so nothing is added)",
            "market + selected features": {"features": primary, "rule": FEATURE_PROTOCOL["selection_rule"]},
            "... + opponent defence": {"features": secondary, "status": "exploratory; these narrowly missed the rule and were added before the holdout was read"},
        },
        "correction": f"L2 logistic on standardized features with the market-only model as offset, penalty {RIDGE_PENALTY}, "
                      "fitted on the two screening folds' out-of-sample predictions",
        "betting_rule": "bet the side whose model probability exceeds DraftKings' implied probability; thresholds 0, 1 and 2 points",
    }, indent=1))
    df = load_matrix()
    y, base, dates, folds = _market_folds(df)
    prop, s = df["prop_type"].to_numpy(), df["season"].to_numpy()
    hold = (s == 2025) & (dates >= HOLDOUT_START)
    pre = ~hold
    Xm = df[group_cols(df, "v8m")].to_numpy(dtype=np.float32)
    Xh = df[group_cols(df, "v8m") + group_cols(df, "v8p")].to_numpy(dtype=np.float32)
    logit = lambda p: np.log(p / (1 - p))  # noqa: E731

    # the linear correction is fitted on bets the market model had not seen (the two screening folds)
    rows, offs = [], []
    for _, (tr, te) in folds.items():
        p0, _ = fit_predict(Xm[tr], y[tr], base[tr], Xm[te], base[te])
        rows.append(np.flatnonzero(te))
        offs.append(logit(p0))
    rows, offs = np.concatenate(rows), np.concatenate(offs)
    p_mkt, _ = fit_predict(Xm[pre], y[pre], base[pre], Xm[hold], base[hold])
    p_hist, _ = fit_predict(Xh[pre], y[pre], base[pre], Xh[hold], base[hold])

    def corrected(features):
        raw = df[features].to_numpy(dtype=np.float32)
        Zfit = np.nan_to_num(_standardize(raw[pre], raw[rows], prop[pre], prop[rows]))
        Zh = np.nan_to_num(_standardize(raw[pre], raw[hold], prop[pre], prop[hold]))
        beta = _ridge_offset_logit(Zfit, y[rows].astype(float), offs)
        return 1 / (1 + np.exp(-(logit(p_mkt) + Zh @ beta))), dict(zip(features, np.round(beta, 4).tolist()))

    models = {"raw DraftKings price": 1 / (1 + np.exp(-base[hold])), "market-only model": p_mkt,
              "market + player history (v8 inputs)": p_hist}
    coefs = {}
    if primary:
        models["market + selected features"], coefs["primary"] = corrected(primary)
        models["  ... + opponent defence (exploratory)"], coefs["secondary"] = corrected(secondary)

    yh, dh = y[hold], dates[hold]
    ll = {k: log_loss_rows(p, yh) for k, p in models.items()}
    print(f"held-out bets: {int(hold.sum()):,} from {dh.min()} to {dh.max()}; read once")
    print(f"selected features: {primary}")
    print(f"coefficients (logit per SD): {coefs}\n")
    print(f"{'model':<42}{'log-loss':>10}{'vs raw price':>22}{'vs market-only':>22}")
    report = {"holdout_bets": int(hold.sum()), "primary": primary, "secondary": secondary, "coefficients": coefs, "log_loss": {}, "betting": {}}
    for k in models:
        a, sa = paired(ll[k] - ll["raw DraftKings price"], dh)
        b, sb = paired(ll[k] - ll["market-only model"], dh)
        report["log_loss"][k] = {"mean": float(ll[k].mean()), "vs_price": a * 1e4, "vs_price_se": sa * 1e4, "vs_market": b * 1e4, "vs_market_se": sb * 1e4}
        za = f"{a * 1e4:+6.2f} ±{1.96 * sa * 1e4:4.2f} (z {a / sa:+.1f})" if sa else "       —"
        zb = f"{b * 1e4:+6.2f} ±{1.96 * sb * 1e4:4.2f} (z {b / sb:+.1f})" if sb else "       —"
        print(f"{k:<42}{ll[k].mean():>10.5f}{za:>22}{zb:>22}")
    print("   (differences in units of 0.0001; negative is better)")

    H = df.loc[hold, ["leg_id", "date", "prop_type", "line", "actual", "over_odds", "under_odds"]].reset_index(drop=True)
    H["io"], H["iu"] = _imp(H["over_odds"]), _imp(H["under_odds"])
    H["od"], H["ud"] = _dec(H["over_odds"]), _dec(H["under_odds"])
    H["won_over"], H["won_under"] = H["actual"] > H["line"], H["actual"] < H["line"]
    bl = pd.read_parquet(BOOK_LINES_FILE).merge(H[["leg_id", "line"]].rename(columns={"line": "dk_line"}), on="leg_id")
    same = bl[((bl["line"] - bl["dk_line"]).abs() < 1e-6) & bl["book"].isin(SIX_BOOKS)].groupby("leg_id")[["over_odds", "under_odds"]].max()
    H["od_six"] = _dec(H["leg_id"].map(same["over_odds"]).fillna(H["over_odds"]))
    H["ud_six"] = _dec(H["leg_id"].map(same["under_odds"]).fillna(H["under_odds"]))
    print(f"\nbetting on the held-out bets (flat stakes; edge = model probability minus DraftKings' implied probability for that side)")
    print(f"   blind under: {fmt(score(H, 'under'))}   at best of six: ROI {score(H, 'under', 'od_six', 'ud_six')['roi']:+.2f}%")
    for k, p in models.items():
        if k == "raw DraftKings price":
            continue
        eo, eu = (p - H["io"]) * 100, ((1 - p) - H["iu"]) * 100
        side = np.where(eo >= eu, "over", "under")
        edge = np.maximum(eo, eu)
        print(f"   {k.strip()}")
        report["betting"][k] = {}
        for thr in (0, 1, 2):
            m = (edge > thr).to_numpy()
            a, b = score(H[m], side[m]), score(H[m], side[m], "od_six", "ud_six")
            report["betting"][k][f"edge>{thr}"] = {"dk": a, "six": b, "under_share": float((side[m] == "under").mean()) if m.any() else None}
            share = f"{(side[m] == 'under').mean() * 100:.0f}% unders" if m.any() else ""
            print(f"      edge above {thr}: {fmt(a)}   at best of six: ROI {b['roi']:+6.2f}% [±{b['ci']:.2f}]   {share}")
    FINAL_FILE.write_text(json.dumps({"finished": datetime.now().isoformat(timespec="seconds"), **report}, indent=1))
    print(f"\nsaved {FINAL_FILE}")


# =============================================================================
# EARLY PRICES: DraftKings at 18:00 UTC on game day, hours before the close
# =============================================================================

EARLY_FILE = LAB_DIR / "early.parquet"
EARLY_PROTOCOL_FILE = LAB_DIR / "protocol_early.json"
EARLY_RESULTS_FILE = LAB_DIR / "early_results.json"
# The 2024-25 file is a later reconstruction with no known capture time, so it is not used.
EARLY_SOURCES = ["backtest_2023_24_data.json", "backtest_oos_data.json"]
EARLY_SNAPSHOT_HOUR_UTC = 18
# Inputs a bettor has at the early snapshot. Two role inputs rank tonight's participants, so they are left out.
EARLY_CONTEXT_GROUPS = ["sched", "travel", "team", "oppdef", "env", "role", "ret", "form"]
NOT_KNOWN_EARLY = {"role__min_rank_l5", "role__starter_proxy"}
EARLY_PROTOCOL = {
    "question": "Does a model using only what is known at 18:00 UTC beat DraftKings' price at that hour?",
    "data": "DraftKings props captured at 18:00 UTC on game day for 2023-24 and Jan-Jun 2026; games tipping before 18:30 UTC excluded; "
            "players matched by full name; a bet is void if the player did not play; graded on clean box scores",
    "split": "train on 2023-24, test once on Jan-Jun 2026",
    "candidates": {
        "early market only": "trees correcting the early de-vigged price from DraftKings' own line and odds",
        "early market + history": "plus the player-history inputs v8 uses, computed against the early line",
        "early market + history + context": "plus schedule, travel, team form, opponent defence, game environment, role, return and form inputs",
        "oracle: + who actually played": "adds teammate-availability inputs, which are not known at 18:00 UTC; shows what the news is worth, not a bettable model",
    },
    "metrics": ["log-loss versus the early price, paired per bet, clustered by date",
                "return on bets where the model's probability beats DraftKings' early implied probability (thresholds 0, 1, 2 points)",
                "closing-line value on those bets: early decimal odds x closing fair probability - 1, same-line props only"],
    "no_selection": "all candidates are fixed here; nothing is chosen on the test season",
}


def _early_file_rows() -> pd.DataFrame:
    """DraftKings' 18:00 UTC rows from the two original early files."""
    from clean_backtest import BASE_DIR

    rows = []
    for name in EARLY_SOURCES:
        for g in json.loads((BASE_DIR / "outputs" / name).read_text())["games"]:
            for player, props in (g.get("props") or {}).items():
                for prop, v in props.items():
                    if prop in PROP_STAT:
                        rows.append((g["date"], player, prop, float(v["line"]), int(v["over_odds"]), int(v["under_odds"])))
    return pd.DataFrame(rows, columns=["date", "name", "prop_type", "line", "over_odds", "under_odds"])


def snapshot_rows(hour: int, windows: list = None, book: str = "draftkings") -> pd.DataFrame:
    """One book's rows from a bought pre-game snapshot (fetch_early_odds.py), laid out like the original early files.
    The date is the game's date in the clean legs where the event is known there, else its Eastern date."""
    import sqlite3

    from fetch_early_odds import db_path
    from fetch_multibook_odds import MARKET_TO_PROP

    conn = sqlite3.connect(f"file:{db_path(hour)}?mode=ro", uri=True, timeout=60)
    P = pd.read_sql("select p.event_id, e.et_date, p.market, p.player_name as name, p.line, p.over_odds, p.under_odds, p.snapshot_time "
                    "from props p join events e on e.event_id = p.event_id where p.bookmaker = ? and e.status = 'ok'", conn, params=(book,))
    conn.close()
    game_date = pd.read_parquet(LEGS_FILE, columns=["event_id", "date"]).drop_duplicates("event_id").set_index("event_id")["date"]
    P["date"] = P["event_id"].map(game_date).fillna(P["et_date"])
    P["prop_type"] = P["market"].map(MARKET_TO_PROP)
    keep = P["prop_type"].isin(PROP_STAT)
    if windows:
        inside = pd.Series(False, index=P.index)
        for first, last in windows:
            inside |= P["date"].between(first, last)
        keep &= inside
    P = P[keep].dropna(subset=["line", "over_odds", "under_odds"])
    return P[["date", "name", "prop_type", "line", "over_odds", "under_odds", "event_id", "snapshot_time"]].reset_index(drop=True)


def _early_frame(E: pd.DataFrame, first_leg_id: int = 0, snapshot_hour: int = EARLY_SNAPSHOT_HOUR_UTC) -> pd.DataFrame:
    """Grade raw pre-game rows on the clean box scores and attach DraftKings' closing price and every feature."""
    from clean_backtest import PROP_TO_STAT, Roster, player_keys
    from train_v8 import V8Trainer

    box = load_box()
    keys = player_keys(box)
    E = E[(E["over_odds"].abs() >= 100) & (E["under_odds"].abs() >= 100)].copy()
    n_raw = len(E)

    by_date = {d: g for d, g in box.groupby("date")}
    resolved = {}
    for d, g in E.groupby("date"):
        if d in by_date:
            for n, (pid, _) in Roster(zip(by_date[d]["player_id"], by_date[d]["player"])).resolve(g["name"].unique()).items():
                resolved[(d, n)] = pid
    E["player_id"] = [resolved.get(k) for k in zip(E["date"], E["name"])]
    n_unresolved = int(E["player_id"].isna().sum())
    E = E.dropna(subset=["player_id"]).copy()
    E["player_id"] = E["player_id"].astype(int)
    E = E.drop_duplicates(["date", "player_id", "prop_type"])

    stats = box.set_index(["date", "player_id"])
    stats = stats[~stats.index.duplicated()]
    idx = pd.MultiIndex.from_arrays([E["date"], E["player_id"]])
    E["game_id"] = stats["game_id"].reindex(idx).to_numpy()
    E["played"] = stats["played"].reindex(idx).to_numpy()
    E["actual"] = np.nan
    for prop, cols in PROP_TO_STAT.items():
        m = (E["prop_type"] == prop).to_numpy()
        E.loc[m, "actual"] = stats[cols].sum(axis=1).reindex(idx[m]).to_numpy()
    E["player"] = E["player_id"].map(keys)

    close = pd.read_parquet(LEGS_FILE)
    tip = close.groupby("game_id")["commence_time"].first()
    E["commence_time"] = E["game_id"].map(tip)
    cutoff = pd.to_datetime(E["date"] + "T00:00:00Z", utc=True) + pd.Timedelta(hours=snapshot_hour, minutes=30)
    pregame = pd.to_datetime(E["commence_time"], utc=True) >= cutoff
    n_no_tip, n_live = int(E["commence_time"].isna().sum()), int((~pregame & E["commence_time"].notna()).sum())
    E = E[pregame].copy()
    n_void = int((E["played"] != True).sum())  # noqa: E712
    E = E[E["played"] == True].copy()  # noqa: E712
    E["game_id"] = E["game_id"].astype(int)
    E = E.merge(close[["date", "player_id", "prop_type", "leg_id", "line", "over_odds", "under_odds"]].rename(
        columns={"leg_id": "close_leg_id", "line": "close_line", "over_odds": "close_over", "under_odds": "close_under"}),
        on=["date", "player_id", "prop_type"], how="left")
    E = E.sort_values(["date", "game_id", "player", "prop_type"]).reset_index(drop=True)
    E["leg_id"] = first_leg_id + np.arange(len(E))
    E["season"] = E["game_id"].map(box.drop_duplicates("game_id").set_index("game_id")["season"])

    trainer = V8Trainer(dataset="clean")
    trainer.feature_builder.load_games()
    trainer.feature_builder.build_player_history()
    legs_iter = ({"leg_id": int(r.leg_id), "player": r.player, "prop_type": r.prop_type, "date": r.date, "actual": float(r.actual),
                  "line": float(r.line), "over_odds": int(r.over_odds), "under_odds": int(r.under_odds), "all_books": {}}
                 for r in E.itertuples(index=False))
    X, _, meta = trainer._features_from_legs(legs_iter)
    X = X.rename(columns={c: (f"v8m__{c}" if c in V8_MARKET else f"v8p__{c}") for c in X.columns})
    X["leg_id"] = meta["leg_id"].to_numpy()
    dk_only = E[["leg_id", "line", "over_odds", "under_odds"]].assign(book="draftkings")
    lab = build_leg_features(E, dk_only, box)
    out = E.merge(X, on="leg_id").merge(lab, on="leg_id")
    print(f"early props read: {n_raw:,}; unresolved names {n_unresolved:,}; no tip time {n_no_tip:,}; already tipped at the snapshot {n_live:,}; "
          f"void (did not play) {n_void:,}")
    print(f"graded early bets: {len(E):,}; with enough history for features: {len(out):,}; matched to a DraftKings closing price: {int(out['close_leg_id'].notna().sum()):,}")
    print("by season:", out.groupby("season").size().to_dict())
    return out


def build_early():
    out = _early_frame(_early_file_rows())
    LAB_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(EARLY_FILE, index=False)
    print("->", EARLY_FILE)


# -----------------------------------------------------------------------------
# Confirmation: the same model on windows it has never seen (protocol_early_confirm.json, written first)
# -----------------------------------------------------------------------------

EARLY_CONFIRM_FILE = LAB_DIR / "early_confirm.parquet"
EARLY_CONFIRM_PROTOCOL_FILE = LAB_DIR / "protocol_early_confirm.json"
EARLY_CONFIRM_RESULTS_FILE = LAB_DIR / "early_confirm_results.json"
EARLY_CONFIRM_WINDOWS = [("2024-10-01", "2025-06-30"), ("2025-10-01", "2025-12-31")]


def build_early_confirm():
    """DraftKings' 18:00 UTC rows for the two windows the original early files do not cover, through the same build."""
    rows = snapshot_rows(EARLY_SNAPSHOT_HOUR_UTC, EARLY_CONFIRM_WINDOWS)
    print(f"snapshot rows: {len(rows):,} from {rows['event_id'].nunique():,} games, {rows['date'].min()} to {rows['date'].max()}")
    out = _early_frame(rows, first_leg_id=10_000_000)
    out.to_parquet(EARLY_CONFIRM_FILE, index=False)
    print("->", EARLY_CONFIRM_FILE)


def _price_columns(E: pd.DataFrame) -> pd.DataFrame:
    E["io"], E["iu"] = _imp(E["over_odds"]), _imp(E["under_odds"])
    E["od"], E["ud"] = _dec(E["over_odds"]), _dec(E["under_odds"])
    E["fair"] = E["io"] / (E["io"] + E["iu"])
    E["won_over"], E["won_under"] = E["actual"] > E["line"], E["actual"] < E["line"]
    matched = E["close_leg_id"].notna()
    cio, ciu = _imp(E["close_over"].fillna(-110)), _imp(E["close_under"].fillna(-110))
    E["close_fair"] = np.where(matched, cio / (cio + ciu), np.nan)
    E["same_line"] = matched & ((E["close_line"] - E["line"]).abs() < 1e-6)
    return E


def _model_picks(T: pd.DataFrame, p: np.ndarray):
    """Side and claimed edge (points of probability over the early implied probability) for each bet."""
    eo, eu = (p - T["io"].to_numpy()) * 100, ((1 - p) - T["iu"].to_numpy()) * 100
    return np.where(eo >= eu, "over", "under"), np.maximum(eo, eu)


def _pick_stats(T: pd.DataFrame, side: np.ndarray, m: np.ndarray) -> dict:
    from clean_backtest import score

    s = score(T[m], side[m])
    ms = m & T["same_line"].to_numpy()
    d = np.where(side[ms] == "over", T.loc[ms, "od"], T.loc[ms, "ud"])
    pc = np.where(side[ms] == "over", T.loc[ms, "close_fair"], 1 - T.loc[ms, "close_fair"])
    cm, cs = paired(d * pc - 1, T.loc[ms, "date"].to_numpy()) if ms.sum() > 1 else (float("nan"), float("nan"))
    return {**s, "under_share": float((side[m] == "under").mean()) if m.any() else None, "days": int(T.loc[m, "date"].nunique()),
            "clv_n": int(ms.sum()), "clv_pct": cm * 100, "clv_ci": 1.96 * cs * 100}


def early_confirm():
    """Read the pre-declared confirmation test once: the early model, trained as before, on bets it has never seen."""
    from clean_backtest import fmt, score

    if not EARLY_CONFIRM_PROTOCOL_FILE.exists():
        raise SystemExit(f"{EARLY_CONFIRM_PROTOCOL_FILE.name} must be written before this test is read")
    A = _price_columns(_finish_matrix(pd.read_parquet(EARLY_FILE)))
    N = _price_columns(_finish_matrix(pd.read_parquet(EARLY_CONFIRM_FILE)))
    cols = group_cols(A, "v8m") + group_cols(A, "v8p")
    absent = [c for c in cols if c not in N.columns]
    if absent:
        raise SystemExit(f"new rows lack model inputs: {absent[:5]}")
    tr = A[A["season"] == 2023]
    old = A[A["season"] == 2025].assign(window="Jan-Jun 2026")
    N["window"] = np.where(N["date"] <= "2025-06-30", "2024-25", "Oct-Dec 2025")
    T = pd.concat([old, N], ignore_index=True)
    line = "=" * 118
    print(line)
    print("CONFIRMATION TEST  (rules fixed in protocol_early_confirm.json before these bets were graded)")
    print(line)
    print(f"model: early market + history, {len(cols)} inputs, trained on {len(tr):,} bets from 2023-24 ({tr['date'].min()} to {tr['date'].max()})")
    p, rounds = fit_predict(tr[cols].to_numpy(dtype=np.float32), tr["y"].to_numpy(), tr["base_logit"].to_numpy(),
                            T[cols].to_numpy(dtype=np.float32), T["base_logit"].to_numpy())
    side, edge = _model_picks(T, p)
    T["p_model"], T["side"], T["edge"] = p, side, edge
    print(f"trees per seed: {rounds}")
    report = {"finished": datetime.now().isoformat(timespec="seconds"), "train_bets": int(len(tr)), "rounds": rounds, "windows": {}}

    def block(label, mask):
        sub = T[mask].reset_index(drop=True)
        sd, ed = sub["side"].to_numpy(), sub["edge"].to_numpy()
        a, sa = paired(log_loss_rows(sub["p_model"].to_numpy(), sub["y"].to_numpy()) - log_loss_rows(sub["fair"].to_numpy(), sub["y"].to_numpy()),
                       sub["date"].to_numpy())
        blind = score(sub, "under")
        res = {"bets": int(len(sub)), "days": int(sub["date"].nunique()), "blind_under": blind, "logloss_vs_price": a * 1e4, "logloss_ci": 1.96 * sa * 1e4}
        print(f"\n{label}: {len(sub):,} graded bets on {sub['date'].nunique()} nights ({sub['date'].min()} to {sub['date'].max()})")
        print(f"   blind under: {fmt(blind)}    model log-loss versus the early price: {a * 1e4:+.2f} ±{1.96 * sa * 1e4:.2f} (x0.0001; negative is better)")
        for thr in (2, 1, 0):
            st = _pick_stats(sub, sd, ed > thr)
            res[f"edge>{thr}"] = st
            tag = "PRIMARY " if thr == 2 else "        "
            print(f"   {tag}edge above {thr}: {fmt(st)}   {st['under_share'] * 100:.0f}% unders, {st['days']} nights; "
                  f"closing-line value {st['clv_pct']:+.2f}% ±{st['clv_ci']:.2f} on {st['clv_n']:,} same-line bets")
        report["windows"][label] = res
        return res

    new = (T["window"] != "Jan-Jun 2026").to_numpy()
    primary = block("BOTH NEW WINDOWS POOLED", new)
    block("2024-25 season", (T["window"] == "2024-25").to_numpy())
    block("Oct-Dec 2025", (T["window"] == "Oct-Dec 2025").to_numpy())
    block("reproduction check: Jan-Jun 2026, already read (earlier result: 3,616 bets +2.50% and 1,638 bets +5.80%)", ~new)

    s = primary["edge>2"]
    verdict = "confirmed" if s["roi"] - s["ci"] > 0 else ("supportive" if s["roi"] > 0 else "not_supported")
    report["primary"] = {"verdict": verdict, **s}
    print("\n" + line)
    print(f"PRIMARY ENDPOINT: {s['n']:,} bets, return {s['roi']:+.2f}% ±{s['ci']:.2f}  ->  {verdict.replace('_', ' ').upper()}")
    print(line)

    # Secondary: walk-forward for the second window, trained on 2023-24 plus the new 2024-25 bets.
    tr2 = pd.concat([tr, N[N["window"] == "2024-25"]], ignore_index=True).sort_values(["date", "leg_id"])
    W = N[N["window"] == "Oct-Dec 2025"].reset_index(drop=True)
    p2, rounds2 = fit_predict(tr2[cols].to_numpy(dtype=np.float32), tr2["y"].to_numpy(), tr2["base_logit"].to_numpy(),
                              W[cols].to_numpy(dtype=np.float32), W["base_logit"].to_numpy())
    sd2, ed2 = _model_picks(W, p2)
    print(f"\nsecondary, walk-forward: trained on 2023-24 and 2024-25 ({len(tr2):,} bets, trees {rounds2}), tested on Oct-Dec 2025")
    report["walk_forward_oct_dec_2025"] = {"train_bets": int(len(tr2)), "rounds": rounds2}
    for thr in (2, 1, 0):
        st = _pick_stats(W, sd2, ed2 > thr)
        report["walk_forward_oct_dec_2025"][f"edge>{thr}"] = st
        print(f"   edge above {thr}: {fmt(st)}   closing-line value {st['clv_pct']:+.2f}% ±{st['clv_ci']:.2f} on {st['clv_n']:,} same-line bets")

    T[["leg_id", "date", "window", "season", "player", "prop_type", "line", "over_odds", "under_odds", "actual", "close_line", "close_over",
       "close_under", "p_model", "side", "edge"]].to_parquet(LAB_DIR / "early_confirm_predictions.parquet", index=False)
    EARLY_CONFIRM_RESULTS_FILE.write_text(json.dumps(report, indent=1, default=float))
    print(f"\nsaved {EARLY_CONFIRM_RESULTS_FILE}")


def early_confirm_splits():
    """Exploratory splits of the confirmation picks, next to the same splits on the first forward test."""
    from clean_backtest import fmt, score

    P = pd.read_parquet(LAB_DIR / "early_confirm_predictions.parquet")
    P["od"], P["ud"] = _dec(P["over_odds"]), _dec(P["under_odds"])
    P["won_over"], P["won_under"] = P["actual"] > P["line"], P["actual"] < P["line"]
    new, old = P[P["window"] != "Jan-Jun 2026"], P[P["window"] == "Jan-Jun 2026"]
    out = {}

    def row(label, a, b):
        sa, sb = score(a, a["side"].to_numpy()), score(b, b["side"].to_numpy())
        out[label] = {"new_windows": sa, "jan_jun_2026": sb}
        print(f"   {label:<34}{fmt(sa)}      {fmt(sb)}")

    print("=" * 118)
    print("EXPLORATORY SPLITS OF THE CONFIRMATION PICKS   (left: the two new windows; right: Jan-Jun 2026, read earlier)")
    print("=" * 118)
    print("   by claimed edge")
    for lo, hi in [(-np.inf, 0), (0, 1), (1, 2), (2, 3), (3, np.inf)]:
        label = f"edge {lo:g} to {hi:g}" if np.isfinite(lo) and np.isfinite(hi) else (f"edge {lo:g} or more" if np.isfinite(lo) else "edge 0 or less")
        row(label, new[(new["edge"] > lo) & (new["edge"] <= hi)], old[(old["edge"] > lo) & (old["edge"] <= hi)])
    for thr in (1, 2):
        print(f"\n   picks with edge above {thr}, by prop")
        for prop in ["points", "pts_rebs_asts", "threes", "rebounds", "assists"]:
            row(f"{prop} (edge>{thr})", new[(new["edge"] > thr) & (new["prop_type"] == prop)], old[(old["edge"] > thr) & (old["prop_type"] == prop)])
        print(f"   picks with edge above {thr}, by side")
        for sd in ("under", "over"):
            row(f"{sd}s (edge>{thr})", new[(new["edge"] > thr) & (new["side"] == sd)], old[(old["edge"] > thr) & (old["side"] == sd)])
        print(f"   picks with edge above {thr}, one bet per player per night (largest edge kept)")
        one = lambda d: d[d["edge"] > thr].sort_values("edge", ascending=False).drop_duplicates(["date", "player"])  # noqa: E731
        row(f"one per player-night (edge>{thr})", one(new), one(old))
    print("\n   picks with edge above 1, by half-season of the new windows")
    for label, a, b in [("Oct-Dec 2024", "2024-10-01", "2024-12-31"), ("Jan-Mar 2025", "2025-01-01", "2025-03-31"),
                        ("Apr-Jun 2025", "2025-04-01", "2025-06-30"), ("Oct-Dec 2025", "2025-10-01", "2025-12-31")]:
        g = new[(new["edge"] > 1) & new["date"].between(a, b)]
        s1 = score(g, g["side"].to_numpy())
        g2 = g[g["edge"] > 2]
        s2 = score(g2, g2["side"].to_numpy())
        out[f"period {label}"] = {"edge>1": s1, "edge>2": s2}
        print(f"   {label:<16} edge>1: {fmt(s1)}      edge>2: {fmt(s2)}")
    (LAB_DIR / "early_confirm_splits.json").write_text(json.dumps(out, indent=1, default=float))


def early_experiment():
    from scipy.stats import norm

    from clean_backtest import fmt, score

    LAB_DIR.mkdir(parents=True, exist_ok=True)
    EARLY_PROTOCOL_FILE.write_text(json.dumps({**EARLY_PROTOCOL, "written": datetime.now().isoformat(timespec="seconds")}, indent=1))
    E = _finish_matrix(pd.read_parquet(EARLY_FILE))
    E["io"], E["iu"] = _imp(E["over_odds"]), _imp(E["under_odds"])
    E["od"], E["ud"] = _dec(E["over_odds"]), _dec(E["under_odds"])
    E["fair"] = E["io"] / (E["io"] + E["iu"])
    E["won_over"], E["won_under"] = E["actual"] > E["line"], E["actual"] < E["line"]
    matched = E["close_leg_id"].notna()
    cio, ciu = _imp(E["close_over"].fillna(-110)), _imp(E["close_under"].fillna(-110))
    E["close_fair"] = np.where(matched, cio / (cio + ciu), np.nan)
    E["same_line"] = matched & ((E["close_line"] - E["line"]).abs() < 1e-6)
    report = {}
    line = "=" * 112

    print(line)
    print("1. WHAT HAPPENS BETWEEN 18:00 UTC AND THE CLOSE")
    print(line)
    print(f"   early bets {len(E):,} ({E['date'].min()} to {E['date'].max()}); hold at the early snapshot {(E['io'] + E['iu'] - 1).mean() * 100:.2f}%")
    M = E[matched]
    moved = M[~M["same_line"]]
    up = (moved["close_line"] > moved["line"]).mean() * 100
    print(f"   still offered at the close: {len(M):,} ({len(M) / len(E) * 100:.1f}%); line changed by the close: {len(moved) / len(M) * 100:.1f}% "
          f"(of those, {up:.0f}% up and {100 - up:.0f}% down)")
    S = E[E["same_line"]]
    shift = (S["close_fair"] - S["fair"]) * 100
    print(f"   same line at the close ({len(S):,}): the over's fair probability moved by {shift.mean():+.2f} points on average; "
          f"it fell on {(shift < -0.25).mean() * 100:.0f}% of props and rose on {(shift > 0.25).mean() * 100:.0f}%")
    a, sa = paired(log_loss_rows(S["close_fair"].to_numpy(), S["y"].to_numpy()) - log_loss_rows(S["fair"].to_numpy(), S["y"].to_numpy()), S["date"].to_numpy())
    print(f"   accuracy on those props: closing price versus early price, log-loss difference {a * 1e4:+.2f} ±{1.96 * sa * 1e4:.2f} (x0.0001; negative means the close knows more)")
    report["movement"] = {"n": int(len(E)), "matched": int(len(M)), "line_changed_share": float(len(moved) / len(M)), "moved_up_share": float(up / 100),
                          "same_line_mean_shift_pts": float(shift.mean()), "close_vs_early_logloss": a * 1e4, "close_vs_early_se": sa * 1e4}

    print("\n" + line)
    print("2. BLIND BETS AT THE EARLY PRICE")
    print(line)
    report["blind"] = {}
    for label, sub in [("all early bets", E)] + [(f"season starting {s}", g) for s, g in E.groupby("season")]:
        u, o = score(sub, "under"), score(sub, "over")
        report["blind"][label] = {"under": u, "over": o}
        print(f"   {label:<22} blind UNDER {fmt(u)}    blind OVER ROI {o['roi']:+6.2f}%")
    C = M.assign(od=_dec(M["close_over"]), ud=_dec(M["close_under"]), won_over=M["actual"] > M["close_line"], won_under=M["actual"] < M["close_line"])
    print(f"   same {len(M):,} props, blind UNDER:  at the early line and price {score(M, 'under')['roi']:+.2f}%   at the closing line and price {score(C, 'under')['roi']:+.2f}%")
    report["blind"]["matched_under_early"] = score(M, "under")
    report["blind"]["matched_under_close"] = score(C, "under")

    print("\n" + line)
    print("3. MODELS THAT ONLY SEE WHAT IS KNOWN AT 18:00 UTC   (train 2023-24, test Jan-Jun 2026, read once)")
    print(line)
    tr, te = (E["season"] == 2023).to_numpy(), (E["season"] == 2025).to_numpy()
    y, base, dates = E["y"].to_numpy(), E["base_logit"].to_numpy(), E["date"].to_numpy()
    market, history = group_cols(E, "v8m"), group_cols(E, "v8p")
    context = [c for g in EARLY_CONTEXT_GROUPS for c in group_cols(E, g) if c not in NOT_KNOWN_EARLY]
    candidates = {"early market only": market, "early market + history": market + history,
                  "early market + history + context": market + history + context,
                  "oracle: + who actually played": market + history + context + group_cols(E, "avail")}
    T = E[te].reset_index(drop=True)
    preds = {"raw early price": T["fair"].to_numpy()}
    for name, cols in candidates.items():
        X = E[cols].to_numpy(dtype=np.float32)
        preds[name], _ = fit_predict(X[tr], y[tr], base[tr], X[te], base[te])
    ll = {k: log_loss_rows(p, y[te]) for k, p in preds.items()}
    print(f"   train {int(tr.sum()):,} bets, test {int(te.sum()):,} bets")
    print(f"   {'model':<36}{'inputs':>7}{'log-loss':>10}{'vs early price':>24}{'vs early market only':>24}")
    report["models"] = {}
    for k in preds:
        a, sa = paired(ll[k] - ll["raw early price"], dates[te])
        b, sb = paired(ll[k] - ll["early market only"], dates[te])
        report["models"][k] = {"log_loss": float(ll[k].mean()), "vs_price": a * 1e4, "vs_price_se": sa * 1e4, "vs_market": b * 1e4, "vs_market_se": sb * 1e4}
        za = f"{a * 1e4:+6.2f} ±{1.96 * sa * 1e4:4.2f} (z {a / sa:+.1f})" if sa else "—"
        zb = f"{b * 1e4:+6.2f} ±{1.96 * sb * 1e4:4.2f} (z {b / sb:+.1f})" if sb else "—"
        n_in = len(candidates[k]) if k in candidates else 0
        print(f"   {k:<36}{n_in:>7}{ll[k].mean():>10.5f}{za:>24}{zb:>24}")

    print("\n" + line)
    print("4. BETTING THE MODEL'S PICKS AT THE EARLY PRICE   (test season; edge = model probability minus early implied probability)")
    print(line)
    blind_u = score(T, "under")
    print(f"   blind under on the test season: {fmt(blind_u)}")
    report["betting"] = {"blind_under": blind_u}
    same = T["same_line"].to_numpy()
    for k, p in preds.items():
        if k == "raw early price":
            continue
        eo, eu = (p - T["io"]) * 100, ((1 - p) - T["iu"]) * 100
        side = np.where(eo >= eu, "over", "under")
        edge = np.maximum(eo, eu).to_numpy()
        print(f"   {k}")
        report["betting"][k] = {}
        for thr in (0, 1, 2):
            m = edge > thr
            s = score(T[m], side[m])
            ms = m & same
            d = np.where(side[ms] == "over", T.loc[ms, "od"], T.loc[ms, "ud"])
            pc = np.where(side[ms] == "over", T.loc[ms, "close_fair"], 1 - T.loc[ms, "close_fair"])
            clv = d * pc - 1
            cm, cs = paired(clv, T.loc[ms, "date"].to_numpy()) if ms.sum() > 1 else (float("nan"), float("nan"))
            report["betting"][k][f"edge>{thr}"] = {**s, "under_share": float((side[m] == "under").mean()) if m.any() else None,
                                                   "clv_n": int(ms.sum()), "clv_pct": cm * 100, "clv_se": cs * 100}
            share = f"{(side[m] == 'under').mean() * 100:.0f}% unders" if m.any() else ""
            print(f"      edge above {thr}: {fmt(s)}   {share:<11} closing-line value {cm * 100:+5.2f}% ±{1.96 * cs * 100:.2f} on {int(ms.sum()):,} same-line bets")
    u_same = T[same]
    clv_u = u_same["ud"] * (1 - u_same["close_fair"]) - 1
    cm, cs = paired(clv_u.to_numpy(), u_same["date"].to_numpy())
    print(f"   for scale, a blind under's closing-line value on same-line props: {cm * 100:+.2f}% ±{1.96 * cs * 100:.2f}  (a random side's is about {-(T['io'] + T['iu'] - 1).mean() / 2 * 100:.1f}%)")
    report["betting"]["blind_under_clv"] = {"pct": cm * 100, "se": cs * 100}
    EARLY_RESULTS_FILE.write_text(json.dumps({"finished": datetime.now().isoformat(timespec="seconds"), **report}, indent=1))
    print(f"\nsaved {EARLY_RESULTS_FILE}")


def early_followup():
    """Did the close move toward the early models' picks, counting line moves; and does the reverse split agree?"""
    from scipy.stats import norm

    from clean_backtest import fmt, score

    E = _finish_matrix(pd.read_parquet(EARLY_FILE))
    E["io"], E["iu"] = _imp(E["over_odds"]), _imp(E["under_odds"])
    E["od"], E["ud"] = _dec(E["over_odds"]), _dec(E["under_odds"])
    E["fair"] = E["io"] / (E["io"] + E["iu"])
    E["won_over"], E["won_under"] = E["actual"] > E["line"], E["actual"] < E["line"]
    matched = E["close_leg_id"].notna().to_numpy()
    cio, ciu = _imp(E["close_over"].fillna(-110)), _imp(E["close_under"].fillna(-110))
    close_fair = np.clip(cio / (cio + ciu), 0.02, 0.98)
    # A moved line is converted to a probability at the early line using the player's own game-to-game spread.
    spread = np.maximum(E["v8p__target_std_season"].to_numpy(dtype=float), 0.75)
    shift = (E["close_line"].fillna(E["line"]) - E["line"]).to_numpy() / spread
    E["close_at_early_line"] = np.where(matched, norm.cdf(norm.ppf(close_fair) + shift), np.nan)
    E["line_move"] = np.where(matched, np.sign(E["close_line"].fillna(E["line"]) - E["line"]), np.nan)

    y, base, dates, season = E["y"].to_numpy(), E["base_logit"].to_numpy(), E["date"].to_numpy(), E["season"].to_numpy()
    market, history = group_cols(E, "v8m"), group_cols(E, "v8p")
    context = [c for g in EARLY_CONTEXT_GROUPS for c in group_cols(E, g) if c not in NOT_KNOWN_EARLY]
    candidates = {"early market only": market, "early market + history": market + history,
                  "early market + history + context": market + history + context,
                  "oracle: + who actually played": market + history + context + group_cols(E, "avail")}

    def gain(frame, side):
        """Return implied by the close minus return implied by the early fair price, per $1, for the chosen side."""
        f = frame[frame["close_at_early_line"].notna().to_numpy()]
        s = np.asarray(side)[frame["close_at_early_line"].notna().to_numpy()]
        d = np.where(s == "over", f["od"], f["ud"])
        moved = np.where(s == "over", f["close_at_early_line"] - f["fair"], f["fair"] - f["close_at_early_line"])
        g = d * moved
        mean, se = paired(g, f["date"].to_numpy()) if len(g) > 1 else (float("nan"), float("nan"))
        chg = f["line_move"].to_numpy() != 0
        fav = np.where(s == "over", f["line_move"] > 0, f["line_move"] < 0)[chg]
        return mean * 100, se * 100, (fav.mean() * 100 if chg.any() else float("nan")), int(chg.sum())

    results = {}
    for label, tr, te in [("forward: train 2023-24, test Jan-Jun 2026", season == 2023, season == 2025),
                          ("reverse: train Jan-Jun 2026, test 2023-24", season == 2025, season == 2023)]:
        T = E[te].reset_index(drop=True)
        print("=" * 118)
        print(label + f"   ({int(te.sum()):,} test bets)")
        print("=" * 118)
        g, se, fav, n_chg = gain(T, np.full(len(T), "under"))
        print(f"   blind under: {fmt(score(T, 'under'))}   close moved toward it by {g:+.2f}% ±{1.96 * se:.2f}; "
              f"of {n_chg:,} line changes {fav:.0f}% went its way")
        results[label] = {}
        for name, cols in candidates.items():
            X = E[cols].to_numpy(dtype=np.float32)
            p, _ = fit_predict(X[tr], y[tr], base[tr], X[te], base[te])
            eo, eu = (p - T["io"]) * 100, ((1 - p) - T["iu"]) * 100
            side = np.where(eo >= eu, "over", "under")
            edge = np.maximum(eo, eu).to_numpy()
            print(f"   {name}")
            results[label][name] = {}
            for thr in (0, 1, 2):
                m = edge > thr
                if m.sum() < 20:
                    print(f"      edge above {thr}: {int(m.sum())} bets")
                    continue
                s = score(T[m], side[m])
                g, se, fav, n_chg = gain(T[m], side[m])
                results[label][name][f"edge>{thr}"] = {**s, "close_gain_pct": g, "close_gain_se": se, "line_moves_favourable_pct": fav, "line_moves": n_chg}
                print(f"      edge above {thr}: {fmt(s)}   close moved toward the pick by {g:+.2f}% ±{1.96 * se:.2f}; "
                      f"of {n_chg:,} line changes {fav:.0f}% went its way")
    (LAB_DIR / "early_followup.json").write_text(json.dumps({"finished": datetime.now().isoformat(timespec="seconds"), **results}, indent=1))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("early-followup", help="Closing-line check including line moves, and the reverse split")
    sub.add_parser("build", help="Compute every feature group for the clean legs")
    sub.add_parser("leak-test", help="Check that no feature uses the game being priced")
    sub.add_parser("screen", help="Test each feature group against the market, walk-forward, with shuffled controls")
    sub.add_parser("feature-screen", help="Test each single feature against the market-only model's errors")
    sub.add_parser("final", help="Read the held-out bets once for the candidates the screens produced")
    sub.add_parser("build-early", help="Join DraftKings' 18:00 UTC prices to clean results and features")
    sub.add_parser("early", help="Test models that only see what is known at 18:00 UTC against that price")
    sub.add_parser("build-early-confirm", help="The same build for the newly bought 18:00 UTC snapshots (2024-25, Oct-Dec 2025)")
    sub.add_parser("early-confirm", help="Read the pre-declared confirmation test of the early model on those windows")
    sub.add_parser("early-confirm-splits", help="Exploratory splits of the confirmation picks")
    args = parser.parse_args()
    if args.cmd == "build":
        build()
    elif args.cmd == "leak-test":
        raise SystemExit(0 if leak_test() else 1)
    elif args.cmd == "screen":
        screen()
    elif args.cmd == "feature-screen":
        feature_screen()
    elif args.cmd == "final":
        final()
    elif args.cmd == "build-early":
        build_early()
    elif args.cmd == "early":
        early_experiment()
    elif args.cmd == "early-followup":
        early_followup()
    elif args.cmd == "build-early-confirm":
        build_early_confirm()
    elif args.cmd == "early-confirm":
        early_confirm()
    elif args.cmd == "early-confirm-splits":
        early_confirm_splits()


if __name__ == "__main__":
    main()
