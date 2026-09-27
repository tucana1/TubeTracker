"""Length errors of prediction tags in my folder: per movie, traces within tolerance, too long (> tol), too short
(< -tol), original truth; per grain lines where the count within tolerance changes (--grains).

    python lenerr.py TAG_A TAG_B MOVIE[,MOVIE...] [--grains]
"""
import json
import sys
from pathlib import Path

ME = Path("/tmp/claude-0/-home-user-TubeTracker/21fce780-cdc3-5467-a04a-73e34cbe3068/scratchpad/agents/faintdata")
sys.path.insert(0, str(ME.parent.parent / "round5"))
import score5  # noqa: E402
from sparsetrack.evaluate import score  # noqa: E402

a, b, movies = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
show = "--grains" in sys.argv


def split(r):
    ok = long = short = 0
    for f in r.get("full", []):
        tol = max(2.0, 0.1 * f["human"])
        if abs(f["error"]) <= tol:
            ok += 1
        elif f["error"] > 0:
            long += 1
        else:
            short += 1
    return ok, long, short


for mv in movies:
    doc = score5.docs(mv)["orig"]
    ra = {r["grain"]: r for r in score(doc, json.loads((ME / "preds" / mv / f"{a}.json").read_text()), onset_tol=50)["rows"]}
    rb = {r["grain"]: r for r in score(doc, json.loads((ME / "preds" / mv / f"{b}.json").read_text()), onset_tol=50)["rows"]}
    ta, tb = [0, 0, 0], [0, 0, 0]
    lines = []
    for gid in sorted(ra):
        sa, sb = split(ra[gid]), split(rb[gid])
        ta = [x + y for x, y in zip(ta, sa)]
        tb = [x + y for x, y in zip(tb, sb)]
        if sa[0] != sb[0]:
            fa = {f["frame"]: f for f in ra[gid].get("full", [])}
            fb = {f["frame"]: f for f in rb[gid].get("full", [])}
            fr = sorted(fa)
            pick = fr[len(fr) // 4], fr[len(fr) // 2], fr[-1]
            ex = " ".join(f"b{k // 25}:{fa[k]['human']:.0f}/{fa[k]['pred']:.0f}/{fb[k]['pred']:.0f}" for k in pick if k in fb)
            lines.append(f"   {gid}: ok/long/short {sa} -> {sb}  (bin: truth/{a}/{b}) {ex}")
    print(f"{mv:12s} {a}: ok {ta[0]} long {ta[1]} short {ta[2]} | {b}: ok {tb[0]} long {tb[1]} short {tb[2]}")
    if show:
        print("\n".join(lines))
