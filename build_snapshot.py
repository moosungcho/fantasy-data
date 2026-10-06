"""Nightly builder for the fantasy tools' Analyzer data.
Downloads nflverse data (play-by-play, Next Gen Stats, FTN charting via nflverse, participation) and writes analyzer_snapshot.json,
keyed by Sleeper player id. Data: nflverse (CC-BY-SA 4.0); FTN Data via nflverse. Requires: pandas, numpy."""
import pandas as pd, numpy as np, json, datetime, os
REL = "https://github.com/nflverse/nflverse-data/releases/download/"
TODAY = datetime.date.today()
SEASON = int(os.environ.get("SEASON") or (TODAY.year if TODAY.month >= 8 else TODAY.year - 1))
def rd(path, **kw): return pd.read_csv(REL + path, low_memory=False, **kw)
R = lambda x, d=1: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), d)

ids = pd.read_csv("https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_playerids.csv", low_memory=False)
ids = ids[ids.sleeper_id.notna() & ids.gsis_id.notna()]
G2S = {g: str(int(s)) for g, s in zip(ids.gsis_id, ids.sleeper_id)}

pbp = rd(f"pbp/play_by_play_{SEASON}.csv.gz")
pbp = pbp[(pbp.season_type == "REG") & (pbp.play_type.isin(["pass", "run"])) & (pbp.two_point_attempt != 1)].copy()
ftn = rd(f"ftn_charting/ftn_charting_{SEASON}.csv")
pbp = pbp.merge(ftn[["nflverse_game_id", "nflverse_play_id", "read_thrown", "is_play_action", "is_screen_pass", "is_catchable_ball", "is_contested_ball", "is_drop", "n_defense_box", "n_blitzers", "qb_location", "is_qb_out_of_pocket"]],
                left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "nflverse_play_id"], how="left")
# label = last week that was played in full (28+ teams with a game), so a Thursday-only week isn't called "through week N"
_tw = pbp.groupby("week").posteam.nunique(); weeks = int(max([w for w, n in _tw.items() if n >= 28] or [int(pbp.week.max())]))
partial = int(pbp.week.max()) if int(pbp.week.max()) > weeks else None
P = pbp[(pbp["pass"] == 1) & pbp.receiver_player_id.notna()].copy()  # targets
P["rec"] = P.complete_pass.fillna(0); P["yds"] = P.receiving_yards.fillna(0); P["ay"] = P.air_yards
P["td"] = ((P.touchdown == 1) & (P.complete_pass == 1)).astype(int)
U = pbp[(pbp.rush == 1) & pbp.rusher_player_id.notna() & (pbp.qb_scramble != 1)].copy()  # designed carries
U["yds"] = U.yards_gained.fillna(0)
team_tg = P.groupby("posteam").size(); team_ay = P.groupby("posteam").ay.sum()
rz_tg = P[P.yardline_100 <= 20].groupby("posteam").size(); i10_tg = P[P.yardline_100 <= 10].groupby("posteam").size()
first_tg = P[P.read_thrown == "1"].groupby("posteam").size()
team_car = U.groupby("posteam").size(); rz_car = U[U.yardline_100 <= 20].groupby("posteam").size(); i5_car = U[U.yardline_100 <= 5].groupby("posteam").size()
drop = pbp[pbp.qb_dropback == 1].groupby("posteam").size()
def state(d): return np.where(d.score_differential >= 8, "lead", np.where(d.score_differential <= -8, "trail", "neutral"))
P["st"] = state(P); U["st"] = state(U)
st_tg = P.groupby(["posteam", "st"]).size()
# snaps (PFR) -> estimated routes = offense snap share x team dropbacks, per game
sn = rd(f"snap_counts/snap_counts_{SEASON}.csv"); sn = sn[sn.game_type == "REG"]
pfr2s = {p: str(int(s)) for p, s in zip(ids.pfr_id, ids.sleeper_id) if isinstance(p, str)}
dbw = pbp[pbp.qb_dropback == 1].groupby(["posteam", "week"]).size()

out = {}
def put(sid, k, v):
    if sid: out.setdefault(sid, {})[k] = v

# receivers
for gid, d in P.groupby("receiver_player_id"):
    sid = G2S.get(gid); tm = d.posteam.mode()[0]; n = len(d)
    if not sid or n < 4: continue
    dep = []
    for lo, hi in [(-99, 0), (0, 10), (10, 20), (20, 99)]:
        x = d[(d.ay >= lo) & (d.ay < hi)] if lo > -99 else d[d.ay < 0]
        dep.append([int(len(x)), int(x.rec.sum()), int(x.yds.sum())])
    tgt_team = int(team_tg.get(tm, 0))
    def stshare(s):
        t = st_tg.get((tm, s), 0); m = int((d.st == s).sum()); return [m, R(100 * m / t, 0) if t else None]
    fr = d[d.read_thrown.isin(["1", "2", "CHK", "SD", "DES"])]
    pa = d[d.is_play_action == True]; npa = d[d.is_play_action == False]
    put(sid, "r", {
        "tg": n, "rec": int(d.rec.sum()), "yds": int(d.yds.sum()), "td": int(d.td.sum()),
        "ts": R(100 * n / tgt_team, 1) if tgt_team else None, "ays": R(100 * d.ay.sum() / team_ay.get(tm, np.nan), 1), "adot": R(d.ay.mean(), 1),
        "yac": R(d.yards_after_catch[d.rec == 1].mean(), 1),
        "dep": dep,
        "rz": [int((d.yardline_100 <= 20).sum()), R(100 * (d.yardline_100 <= 20).sum() / rz_tg.get(tm, np.nan), 0)],
        "i10": [int((d.yardline_100 <= 10).sum()), R(100 * (d.yardline_100 <= 10).sum() / i10_tg.get(tm, np.nan), 0)],
        "d3": [int(d.down.isin([3, 4]).sum()), int(d[d.down.isin([3, 4])].rec.sum())],
        "st": {s: stshare(s) for s in ["lead", "neutral", "trail"]},
        "fr": [int((d.read_thrown == "1").sum()), R(100 * (d.read_thrown == "1").sum() / first_tg.get(tm, np.nan), 0), R(100 * (d.read_thrown == "1").sum() / len(fr), 0) if len(fr) else None],
        "cat": [int(d.is_catchable_ball.fillna(False).sum()), int(d.is_drop.fillna(False).sum()), int(d.is_contested_ball.fillna(False).sum()), int(d[d.is_contested_ball == True].rec.sum())],
        "pa": [int(len(pa)), R(pa.yds.sum() / len(pa), 1) if len(pa) else None, int(len(npa)), R(npa.yds.sum() / len(npa), 1) if len(npa) else None],
        "scr": int((d.is_screen_pass == True).sum()),
    })
    # estimated routes from snap share x team dropbacks (per game)
# rushers
for gid, d in U.groupby("rusher_player_id"):
    sid = G2S.get(gid); tm = d.posteam.mode()[0]; n = len(d)
    if not sid or n < 8: continue
    def ypc(x): return R(x.yds.mean(), 1) if len(x) else None
    gun = d[d.shotgun == 1]; uc = d[d.shotgun == 0]; b8 = d[d.n_defense_box >= 8]; lb = d[(d.n_defense_box > 0) & (d.n_defense_box <= 6)]
    put(sid, "u", {"car": n, "yds": int(d.yds.sum()), "td": int((d.touchdown == 1).sum()), "cs": R(100 * n / team_car.get(tm, np.nan), 1),
        "rz": [int((d.yardline_100 <= 20).sum()), R(100 * (d.yardline_100 <= 20).sum() / rz_car.get(tm, np.nan), 0)],
        "i5": [int((d.yardline_100 <= 5).sum()), R(100 * (d.yardline_100 <= 5).sum() / i5_car.get(tm, np.nan), 0)],
        "gun": [len(gun), ypc(gun)], "uc": [len(uc), ypc(uc)], "b8": [len(b8), ypc(b8)], "lb": [len(lb), ypc(lb)],
        "st": {s: [int((d.st == s).sum()), ypc(d[d.st == s])] for s in ["lead", "neutral", "trail"]},
        "exp": [int((d.yds >= 10).sum()), int((d.yds <= 0).sum())]})
# passers
Q = pbp[(pbp.qb_dropback == 1) & pbp.passer_player_id.notna() & (pbp.sack != 1) & (pbp.qb_scramble != 1)].copy()
Q["yds"] = Q.passing_yards.fillna(0) if "passing_yards" in Q else Q.yards_gained.fillna(0)
for gid, d in Q.groupby("passer_player_id"):
    sid = G2S.get(gid); n = len(d)
    if not sid or n < 25: continue
    def ypa(x): return R(x.yds.sum() / len(x), 1) if len(x) else None
    pa = d[d.is_play_action == True]; npa = d[d.is_play_action == False]; bl = d[d.n_blitzers > 0]; nb = d[d.n_blitzers == 0]; oop = d[d.is_qb_out_of_pocket == True]
    deep = d[d.air_yards >= 20]
    put(sid, "q", {"att": n, "pa": [len(pa), ypa(pa)], "npa": [len(npa), ypa(npa)], "bl": [len(bl), ypa(bl)], "nb": [len(nb), ypa(nb)], "oop": [len(oop), ypa(oop)],
        "deep": [len(deep), R(100 * len(deep) / n, 0), int(deep.complete_pass.sum())], "rz": [int((d.yardline_100 <= 20).sum()), int(((d.yardline_100 <= 20) & (d.touchdown == 1)).sum())],
        "st": {s: [int((state(d) == s).sum()), ypa(d[state(d) == s])] for s in ["lead", "neutral", "trail"]}})

# Next Gen Stats, 2026 season-level rows (week 0 = season to date)
def ngs(file, key, cols):
    d = rd("nextgen_stats/" + file); d = d[(d.season == SEASON) & (d.season_type == "REG")]
    w0 = d[d.week == 0]; d = w0 if len(w0) else d.sort_values("week").groupby("player_gsis_id").tail(1)
    for r in d.itertuples():
        sid = G2S.get(r.player_gsis_id)
        if sid: put(sid, key, {k: R(getattr(r, c), 2 if k in ("cpoe",) else 1) for k, c in cols.items()})
ngs("ngs_receiving.csv.gz", "nr", {"sep": "avg_separation", "cush": "avg_cushion", "iay": "avg_intended_air_yards", "ays": "percent_share_of_intended_air_yards", "yac": "avg_yac", "xyac": "avg_expected_yac", "yacoe": "avg_yac_above_expectation", "cp": "catch_percentage"})
ngs("ngs_rushing.csv.gz", "nu", {"eff": "efficiency", "b8": "percent_attempts_gte_eight_defenders", "ttl": "avg_time_to_los", "ryoe": "rush_yards_over_expected_per_att", "roe": "rush_pct_over_expected"})
ngs("ngs_passing.csv.gz", "nq", {"ttt": "avg_time_to_throw", "cpoe": "completion_percentage_above_expectation", "agg": "aggressiveness", "iay": "avg_intended_air_yards", "cay": "avg_completed_air_yards", "ayd": "avg_air_yards_differential"})

# routes run (latest season with participation data) (on the field on a dropback) and targeted-route profile (participation, FTN via nflverse)

PREV = SEASON
try: rd(f"pbp_participation/pbp_participation_{SEASON}.csv", nrows=1)
except Exception: PREV = SEASON - 1
par = rd(f"pbp_participation/pbp_participation_{PREV}.csv", usecols=["nflverse_game_id", "play_id", "offense_players", "route", "possession_team"])
p25 = rd(f"pbp/play_by_play_{PREV}.csv.gz",
                  usecols=["game_id", "play_id", "season_type", "qb_dropback", "receiver_player_id", "complete_pass", "receiving_yards", "pass", "position" if False else "posteam", "two_point_attempt"])
p25 = p25[(p25.season_type == "REG") & (p25.two_point_attempt != 1)]
m = p25.merge(par, left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "play_id"], how="inner")
db = m[m.qb_dropback == 1]
routes = {}
for pl in db.offense_players.dropna():
    for g in pl.split(";"): routes[g] = routes.get(g, 0) + 1
tg25 = m[(m["pass"] == 1) & m.receiver_player_id.notna()].copy(); tg25["yds"] = tg25.receiving_yards.fillna(0)
pos = {g: p for g, p in zip(ids.gsis_id, ids.position)}
agg = tg25.groupby("receiver_player_id").agg(tg=("yds", "size"), yds=("yds", "sum"))
for gid, r in agg.iterrows():
    sid = G2S.get(gid); rr = routes.get(gid, 0)
    if not sid or r.tg < 20 or rr < 100: continue
    put(sid, "y25", {"rt": int(rr), "tg": int(r.tg), "yds": int(r.yds), "tprr": R(100 * r.tg / rr, 1), "yprr": R(r.yds / rr, 2)})
rt = tg25[tg25.route.notna()].copy(); rt["pos"] = rt.receiver_player_id.map(pos)
avg = rt.groupby(["pos", "route"]).agg(n=("yds", "size"), c=("complete_pass", "mean"), y=("yds", "mean"))
AVG = {}
for (p, ro), r in avg.iterrows():
    if r.n >= 30: AVG.setdefault(p, {})[ro] = [R(100 * r.c, 0), R(r.y, 1)]
for gid, d in rt.groupby("receiver_player_id"):
    sid = G2S.get(gid)
    if not sid or len(d) < 20: continue
    g = d.groupby("route").agg(n=("yds", "size"), c=("complete_pass", "sum"), y=("yds", "sum")).sort_values("n", ascending=False)
    put(sid, "rt25", [[ro, int(r.n), int(r.c), int(r.y)] for ro, r in g.iterrows()])

meta = {"built": datetime.date.today().isoformat(), "season": SEASON, "routes_season": PREV, "through": weeks, "partial": partial, "avg25": AVG}
json.dump({"meta": meta, "p": out}, open("analyzer_snapshot.json", "w"), separators=(",", ":"))
import os; print(len(out), "players", os.path.getsize("analyzer_snapshot.json") // 1024, "KB", "through week", weeks)
