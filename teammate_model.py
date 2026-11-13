#!/usr/bin/env python3
"""
teammate_model.py — step 1 of the news-speed work: when a key player is pulled, how far does each teammate prop's fair
price move? Trained on the hourly snapshots of 2023-24 and 2024-25, tested on 2025-26.
Rules: outputs/clean/feature_lab/protocol_news_confirm.json (step_1_teammate_model).

    python teammate_model.py run       # build targets and features, fit, test accuracy, save the model
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.special import ndtri

import feature_lab as fl
import market_lab as ml
from clean_backtest import load_box

DIR = fl.LAB_DIR / "teammate"
MODEL_FILE = DIR / "teammate_lgb.txt"
META_FILE = DIR / "teammate_meta.json"
RESULTS_FILE = fl.LAB_DIR / "teammate_model_results.json"
PROPS = ["points", "rebounds", "assists", "threes", "pts_rebs_asts"]
STATS = ["minutes", "pts", "reb", "ast", "fg3m", "fga"]
PARAMS = {"objective": "regression", "learning_rate": 0.04, "num_leaves": 15, "max_depth": 5, "min_data_in_leaf": 50,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0, "verbose": -1, "seed": 11}
LINE = "=" * 126


# =============================================================================
# FORM FROM BOX SCORES (games before the date only)
# =============================================================================

def _season_of(date: pd.Series) -> pd.Series:
    d = pd.to_datetime(date)
    return np.where(d.dt.month >= 8, d.dt.year, d.dt.year - 1)


def form_tables(box: pd.DataFrame):
    b = box[box["played"] == True].copy()  # noqa: E712
    b["date"] = pd.to_datetime(b["date"])
    b = b.sort_values(["player_id", "date"])
    g = b.groupby(["player_id", "season"])
    n = g.cumcount() + 1
    P = b[["player_id", "season", "date", "position"]].copy()
    for c in STATS:
        P[f"avg_{c}"] = g[c].cumsum() / n
    P["n_games"] = n
    P["pos"] = P["position"].astype(str).str[:1]
    team = b.groupby(["team_id", "season", "date"])[["pts", "reb", "ast"]].sum().reset_index().sort_values(["team_id", "date"])
    tg = team.groupby(["team_id", "season"])
    nt = tg.cumcount() + 1
    T = team[["team_id", "season", "date"]].copy()
    for c in ("pts", "reb", "ast"):
        T[f"team_{c}"] = tg[c].cumsum() / nt
    return P.sort_values("date"), T.sort_values("date")


def _asof(df: pd.DataFrame, table: pd.DataFrame, by: str) -> pd.DataFrame:
    """Latest row of `table` strictly before each row's date, same `by` id and same season."""
    left = df.assign(_d=pd.to_datetime(df["date"]), _s=_season_of(df["date"])).reset_index().sort_values("_d")
    right = table.rename(columns={"date": "_d", "season": "_ts"})
    m = pd.merge_asof(left, right, on="_d", by=by, allow_exact_matches=False, direction="backward")
    m.loc[m["_ts"] != m["_s"], [c for c in right.columns if c not in (by, "_d")]] = np.nan
    return m.sort_values("index").set_index("index")


def features(rows: pd.DataFrame, P: pd.DataFrame, T: pd.DataFrame) -> pd.DataFrame:
    """rows: date, player_id (teammate), team_id, prop_type, line, z_pre, keys (list of key player ids), key_lines (list)."""
    tm = _asof(rows[["date", "player_id"]], P, "player_id")
    X = pd.DataFrame(index=rows.index)
    X["prop"] = rows["prop_type"].map({p: i for i, p in enumerate(PROPS)}).astype(float)
    X["z_pre"] = rows["z_pre"].astype(float)
    X["line"] = rows["line"].astype(float)
    for c in STATS:
        X[f"tm_{c}"] = tm[f"avg_{c}"].to_numpy()
    X["tm_games"] = tm["n_games"].to_numpy()
    X["tm_starter"] = (X["tm_minutes"] >= 24).astype(float)
    team = _asof(rows[["date", "team_id"]], T, "team_id")
    for c in ("pts", "reb", "ast"):
        X[f"tm_share_{c}"] = X[f"tm_{c}"] / team[f"team_{c}"].to_numpy()
    key_rows = rows[["date", "keys"]].explode("keys").rename(columns={"keys": "player_id"}).dropna()
    key_rows["player_id"] = key_rows["player_id"].astype(int)
    kf = _asof(key_rows.reset_index().rename(columns={"index": "row"}).set_index("row")[["date", "player_id"]].reset_index(), P, "player_id")
    kf["row"] = key_rows.index.to_numpy()
    agg = kf.groupby("row")[[f"avg_{c}" for c in STATS]].sum()
    for c in STATS:
        X[f"key_{c}"] = agg[f"avg_{c}"].reindex(rows.index).to_numpy()
    kpos = kf.groupby("row")["pos"].apply(set)
    X["same_pos"] = [float(p in kpos.get(i, set())) for i, p in zip(rows.index, tm["pos"].to_numpy())]
    X["n_keys"] = rows["keys"].apply(len).astype(float)
    X["key_lines"] = rows["key_lines"].apply(sum).astype(float)
    return X


# =============================================================================
# TARGETS FROM THE HOURLY SNAPSHOTS
# =============================================================================

def _fair(pseudo: pd.DataFrame, quotes: pd.DataFrame, sig: dict) -> np.ndarray:
    ps = pseudo.assign(book="ref", over_odds=-110, under_odds=-110)
    cols = ["date", "season", "game_id", "player_id", "team_id", "prop_type", "book", "line", "over_odds", "under_odds"]
    Q = pd.concat([quotes[cols], ps[cols].assign(row_tag=np.arange(len(ps)))], ignore_index=True)
    Q = ml.add_consensus(Q, sig)
    return Q[Q["book"] == "ref"].sort_values("row_tag")["p_s1"].to_numpy()


def build() -> pd.DataFrame:
    E = pd.read_parquet(ml.DIR / "news_candidates.parquet")
    hours = sorted(set(E["from"]) | set(E["to"]))
    quotes = {h: pd.read_parquet(ml.DIR / f"quotes_h{h:02d}.parquet") for h in hours}
    sig = ml.fit_sigma([2023, 2024])
    rows = []
    for (gid, g, h, team), ev in E.groupby(["game_id", "from", "to", "team_id"]):
        keys = ev["player_id"].astype(int).tolist()
        pre = quotes[g]
        dk = pre[(pre["game_id"] == gid) & (pre["book"] == "draftkings") & (pre["team_id"] == team) & ~pre["player_id"].isin(keys)]
        for r in dk.itertuples(index=False):
            rows.append({"season": int(ev["season"].iloc[0]), "date": ev["date"].iloc[0], "game_id": int(gid), "team_id": int(team), "from": int(g), "to": int(h),
                         "player_id": int(r.player_id), "prop_type": r.prop_type, "line": float(r.line), "keys": keys, "key_lines": ev["pts_line"].tolist()})
    D = pd.DataFrame(rows)
    D["fair_pre"], D["fair_post"] = np.nan, np.nan
    for g in hours:
        m = (D["from"] == g).to_numpy()
        if m.any():
            Q = quotes[g][quotes[g]["game_id"].isin(D.loc[m, "game_id"])]
            D.loc[m, "fair_pre"] = _fair(D[m], Q, sig)
        m = (D["to"] == g).to_numpy()
        if m.any():
            Q = quotes[g][quotes[g]["game_id"].isin(D.loc[m, "game_id"])]
            D.loc[m, "fair_post"] = _fair(D[m], Q, sig)
    D = D[D["fair_pre"].notna() & D["fair_post"].notna()].copy()
    D["z_pre"] = ndtri(np.clip(D["fair_pre"], 0.005, 0.995))
    D["shift"] = ndtri(np.clip(D["fair_post"], 0.005, 0.995)) - D["z_pre"]
    return D.reset_index(drop=True)


def run():
    import lightgbm as lgb

    DIR.mkdir(parents=True, exist_ok=True)
    box = load_box()
    P, T = form_tables(box)
    D = build()
    X = features(D, P, T)
    cols = list(X.columns)
    tr, te = (D["season"] <= 2024).to_numpy(), (D["season"] == 2025).to_numpy()
    print(LINE)
    print(f"TEAMMATE MODEL: training rows {int(tr.sum()):,} (2023-24, 2024-25), test rows {int(te.sum()):,} (2025-26); features {len(cols)}")
    print(LINE)
    order = np.argsort(pd.to_datetime(D.loc[tr, "date"]).to_numpy(), kind="stable")
    Xtr, ytr = X[tr].iloc[order], D.loc[tr, "shift"].iloc[order]
    cut = int(len(Xtr) * (1 - fl.VALID_SHARE))
    dtr = lgb.Dataset(Xtr.iloc[:cut], ytr.iloc[:cut])
    dva = lgb.Dataset(Xtr.iloc[cut:], ytr.iloc[cut:], reference=dtr)
    booster = lgb.train(PARAMS, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    booster.save_model(str(MODEL_FILE), num_iteration=booster.best_iteration)
    base = D[tr].groupby("prop_type")["shift"].mean()
    META_FILE.write_text(json.dumps({"columns": cols, "base_shift": base.to_dict(), "trees": int(booster.best_iteration)}))
    y = D.loc[te, "shift"].to_numpy()
    pred = booster.predict(X[te], num_iteration=booster.best_iteration)
    pbase = D.loc[te, "prop_type"].map(base).to_numpy()
    sst = ((y - y.mean()) ** 2).sum()
    out = {"train_rows": int(tr.sum()), "test_rows": int(te.sum()), "trees": int(booster.best_iteration)}
    print(f"   trees {booster.best_iteration}; actual shift on 2025-26: mean {y.mean():+.3f}, spread {y.std():.3f} (probit units; 0.1 is about 4 points of probability near 50%)")
    for name, p in [("model", pred), ("mean shift per prop", pbase)]:
        r2 = 1 - ((y - p) ** 2).sum() / sst
        out[name] = {"r2": float(r2), "corr": float(np.corrcoef(p, y)[0, 1])}
        print(f"   {name:<22} R squared {r2:+.3f}; correlation {np.corrcoef(p, y)[0, 1]:+.3f}")
    for prop in PROPS:
        m = (D.loc[te, "prop_type"] == prop).to_numpy()
        if m.sum() > 50:
            r2 = 1 - ((y[m] - pred[m]) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()
            print(f"      {prop:<14} rows {int(m.sum()):>5}; actual mean shift {y[m].mean():+.3f}; model R squared {r2:+.3f}; correlation {np.corrcoef(pred[m], y[m])[0, 1]:+.3f}")
    imp = pd.Series(booster.feature_importance("gain"), index=cols).sort_values(ascending=False)
    print("   most useful inputs:", ", ".join(f"{k} {v / imp.sum():.0%}" for k, v in imp.head(8).items()))
    RESULTS_FILE.write_text(json.dumps({"finished": datetime.now().isoformat(timespec="seconds"), **out}, indent=1, default=float))


def load_predictor(model_file=None, meta_file=None):
    """A function (stale rows, box) -> predicted probit shift, for rows with date, player_id, key, key_team, key_line, prop_type, line, fair_pre.
    model_file and meta_file default to the lab copies; news_monitor.py passes its own (outputs/ is the EFS volume on AWS)."""
    import lightgbm as lgb

    model_file, meta_file = model_file or MODEL_FILE, meta_file or META_FILE
    if not model_file.exists():
        return None
    booster = lgb.Booster(model_file=str(model_file))
    meta = json.loads(meta_file.read_text())
    cache = {}

    def predict(St: pd.DataFrame, box: pd.DataFrame) -> np.ndarray:
        if "tables" not in cache:
            cache["tables"] = form_tables(box)
        P, T = cache["tables"]
        rows = pd.DataFrame({"date": St["date"].to_numpy(), "player_id": St["player_id"].astype(int).to_numpy(), "team_id": St["key_team"].astype(int).to_numpy(),
                             "prop_type": St["prop_type"].to_numpy(), "line": St["line"].astype(float).to_numpy(),
                             "z_pre": ndtri(np.clip(St["fair_pre"].astype(float).to_numpy(), 0.005, 0.995))})
        rows["keys"] = [[int(k)] for k in St["key"]]
        rows["key_lines"] = [[float(v)] for v in St["key_line"]]
        X = features(rows, P, T)[meta["columns"]]
        return booster.predict(X)

    return predict


if __name__ == "__main__":
    run()
