# Fantasy tool data

This repository rebuilds the advanced-stats data behind two fantasy football tools:
- **Player Full breakdown** (Next Gen Stats, target and rushing profiles, routes): `analyzer_snapshot.json`
- **Defense Full breakdown** (EPA, success rate, blitz rate, pass rush, run defense): `def_snapshot.json`

A free GitHub Actions job (`.github/workflows/analyzer-snapshot.yml`) runs `build_snapshot.py` and `def_snapshot.py`
twice a day (about 6:30 a.m. and 6:30 p.m. Eastern), plus midday Tuesday and Wednesday when each week's charting usually lands.
It only commits when the data changed. The tools read these files on every **Refresh**.

To update by hand: **Actions → Analyzer data → Run workflow**.

## Data credits
nflverse (play-by-play with the nflfastR EPA model, Next Gen Stats, snap counts, participation; CC-BY-SA 4.0) and FTN Data charting via nflverse.
Player ids: DynastyProcess.
