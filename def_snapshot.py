"""Defense breakdown snapshot (2026) from nflverse play-by-play + FTN charting. Embedded in both tools; rebuilt weekly with the analyzer."""
import pandas as pd, numpy as np, json, datetime, os
SEASON = int(os.environ.get("SEASON", "2026")); REL = "https://github.com/nflverse/nflverse-data/releases/download/"
pbp = pd.read_csv(REL + f"pbp/play_by_play_{SEASON}.csv.gz", low_memory=False, compression="gzip")
pbp = pbp[(pbp.season_type == "REG") & pbp.defteam.notna() & ((pbp["pass"] == 1) | (pbp.rush == 1)) & (pbp.qb_spike != 1) & (pbp.qb_kneel != 1)].copy()
pbp["db"] = (pbp.qb_dropback == 1).astype(int); pbp["ru"] = ((pbp.rush == 1) & (pbp.qb_dropback != 1)).astype(int)
pbp["expl"] = (((pbp.db == 1) & (pbp.yards_gained >= 20)) | ((pbp.ru == 1) & (pbp.yards_gained >= 10))).astype(int)
pbp["tk"] = (pbp.interception.fillna(0) + pbp.fumble_lost.fillna(0)).clip(0, 1)
ftn = pd.read_csv(REL + f"ftn_charting/ftn_charting_{SEASON}.csv", low_memory=False)
f = pbp.merge(ftn[["nflverse_game_id", "nflverse_play_id", "n_blitzers", "n_pass_rushers", "n_defense_box", "is_play_action", "is_screen_pass", "is_interception_worthy"]],
              left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "nflverse_play_id"], how="inner")
def metrics(d, fd):
    db, ru = d[d.db == 1], d[d.ru == 1]; att = d[d.pass_attempt == 1]; cmpd = att[att.complete_pass == 1]
    third = d[d.down == 3]; deep = att[att.air_yards >= 20]
    drv = d.groupby(["game_id", "drive"]).agg(rz=("yardline_100", lambda s: (s <= 20).any()), res=("fixed_drive_result", "first")).reset_index()
    rz = drv[drv.rz]
    m = dict(n=int(d.game_id.nunique()), plays=len(d),
        epa=d.epa.mean(), epa_p=db.epa.mean(), epa_r=ru.epa.mean(), sr=d.success.mean(), ypp=d.yards_gained.mean(),
        ypd=db.yards_gained.mean(), ypc=ru.yards_gained.mean(), expl=d.expl.mean(),
        sack=db.sack.mean(), hit=db.qb_hit.mean(), take=d.tk.sum() / max(1, d.game_id.nunique()),
        third=(third.first_down.fillna(0) + third.touchdown.fillna(0)).clip(0, 1).mean() if len(third) else None,
        rztd=(rz.res == "Touchdown").mean() if len(rz) else None,
        cmp=att.complete_pass.mean(), cpoe=att.cpoe.mean(), adot=att.air_yards.mean(), yac=cmpd.yards_after_catch.mean(),
        deep=len(deep) / max(1, len(att)), deep_cmp=deep.complete_pass.mean() if len(deep) else None, pass_rate=d.db.mean())
    fdb, fru = fd[fd.db == 1], fd[fd.ru == 1]
    if len(fdb) >= 15:
        bl = fdb[fdb.n_blitzers > 0]; nb = fdb[fdb.n_blitzers == 0]; pa = fdb[fdb.is_play_action == 1]
        m.update(blitz=(fdb.n_blitzers > 0).mean(), rush=fdb.n_pass_rushers.mean(), 
                 epa_bl=bl.epa.mean() if len(bl) >= 8 else None, epa_nb=nb.epa.mean() if len(nb) >= 8 else None,
                 sack_bl=bl.sack.mean() if len(bl) >= 8 else None, pa=fdb.is_play_action.mean(), epa_pa=pa.epa.mean() if len(pa) >= 8 else None,
                 screen=fdb.is_screen_pass.mean(), ftn_n=int(fd.game_id.nunique()))
    if len(fru) >= 10: m.update(box8=(fru.n_defense_box >= 8).mean(), ypc8=fru[fru.n_defense_box >= 8].yards_gained.mean() if (fru.n_defense_box >= 8).sum() >= 5 else None)
    return {k: (None if v is None or (isinstance(v, float) and not np.isfinite(v)) else (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)) for k, v in m.items()}
out = {}
for t, d in pbp.groupby("defteam"):
    fd = f[f.defteam == t]
    s = metrics(d, fd); g = {}
    for w, dw in d.groupby("week"):
        fw = fd[fd.week == w]; x = metrics(dw, fw)
        g[int(w)] = {k: x.get(k) for k in ["epa", "epa_p", "epa_r", "sr", "sack", "take", "expl", "blitz", "rush", "box8", "cmp", "adot"]}
        g[int(w)]["opp"] = dw.posteam.iloc[0]; g[int(w)]["sk"] = int(dw.sack.sum()); g[int(w)]["db"] = int(dw.db.sum())
    out[t] = {"s": s, "g": g}
# ranks of 32: 1 = best defense for outcome stats; 1 = most for style stats
LOWBEST = ["epa", "epa_p", "epa_r", "sr", "ypp", "ypd", "ypc", "expl", "third", "rztd", "cmp", "cpoe", "yac", "deep_cmp", "epa_bl", "epa_nb", "epa_pa", "ypc8"]
HIGHBEST = ["sack", "hit", "take"]
STYLE = ["blitz", "rush", "rush5", "box8", "pa", "screen", "adot", "deep", "pass_rate"]
for k in LOWBEST + HIGHBEST + STYLE:
    vals = {t: o["s"].get(k) for t, o in out.items() if o["s"].get(k) is not None}
    if len(vals) < 20: continue
    sr = pd.Series(vals).rank(ascending=k in LOWBEST, method="min")
    for t, r in sr.items(): out[t].setdefault("r", {})[k] = int(r)
tw = pbp.groupby("week").posteam.nunique(); thr = int(max([w for w, n in tw.items() if n >= 28]))
ftw = f.groupby("week").posteam.nunique(); fthr = int(max([w for w, n in ftw.items() if n >= 28] or [0]))
meta = {"built": datetime.date.today().isoformat(), "season": SEASON, "through": thr, "ftn_through": fthr, "pbp_max": int(pbp.week.max())}
M = {"LA": "LAR"}; out = {M.get(t, t): o for t, o in out.items()}
for o in out.values():
    for g in o["g"].values(): g["opp"] = M.get(g["opp"], g["opp"])
json.dump({"meta": meta, "t": out}, open("def_snapshot.json", "w"), separators=(",", ":"))
import os; print(len(out), "teams", os.path.getsize("def_snapshot.json") // 1024, "KB", meta)
print({k: out["PIT"]["s"].get(k) for k in ["n", "epa", "sr", "sack", "blitz", "rush", "box8", "rztd", "third", "ftn_n"]}, out["PIT"]["r"].get("epa"))
