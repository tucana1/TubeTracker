"""Where the app draws a reviewed tube: the person's route at every bin (the app until 3 Oct 2026) against
``tubetracker.app.overlay.route_at`` (their route near the bins they traced, the model's own per-bin route
elsewhere), scored by length-and-tip on the simulated reviews (prototypes/review_fill's protocols).

The app's own code draws the tube: each labelled movie is opened as the app opens an analysis (``model.RunData`` on
0.8.8's predictions, copied to a scratch analysis folder), the person's traces are put in the grain's record as the
app keeps them (in the grain's own frame: reference coordinates less the analysis' drift), its curve is the reviewed
curve (``sparsetrack.review.reviewed_curve``) with the model's own kept as ``Lm``, and the tip is the end of
``overlay.tube_at``. Lengths are the same for every route policy; only the tips differ.

``--grid`` also ranks variants of the rule on ld (the window round a person's traces; a check that the model's route
leaves the grain where the person's does, at the bin drawn or at theirs; the model's route only where it read a
tube), implemented here.

    python -m prototypes.review_curve.routes [--grid]
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
from pathlib import Path

import numpy as np

from prototypes.review_fill.build import BASE
from prototypes.review_fill.evaluate import load_movie, paired, tip_tol, tol
from sparsetrack.review import reviewed_curve
from sparsetrack.routes import resample
from tubetracker.app import overlay
from tubetracker.app.model import RunData
from tubetracker.app.runfolder import RunFolder

SCRATCH = Path("/private/tmp/claude-501/-Users-joshjiang-Documents-TubeTracker/eaa2b708-9f78-4565-aa0c-9af3763da8fd/"
               "scratchpad/rf/runs")


def old_route_at(g: dict, b: int) -> list:
    """The app's route until 3 Oct 2026: the first route a person traced at or after b; else their last one,
    carried on along the model's route; else the model's route."""
    model = overlay.model_route(g, b)
    drawn = [t for t in g.get("human") or [] if t.get("pts")]
    if not drawn:
        return model
    later = [t for t in drawn if t["bin"] >= b]
    return later[0]["pts"] if later else overlay.continued(drawn[-1]["pts"], model)


def base_dist(pts, model, upto: float) -> float:
    """Mean distance of a traced route's first ``upto`` px (every px, never past either route's end) to a route."""
    q, s = resample(pts, 1.0)
    q = q[s <= min(s[-1], overlay.path_length(model), upto)]
    m = np.asarray(model, float)
    a, d = m[:-1], np.diff(m, axis=0)
    seg2 = np.maximum((d ** 2).sum(1), 1e-12)
    t = np.clip(((q[:, None, :] - a[None]) * d[None]).sum(2) / seg2[None], 0.0, 1.0)
    return float(np.hypot(*(a[None] + t[..., None] * d[None] - q[:, None, :]).transpose(2, 0, 1)).min(axis=1).mean())


def variant_route_at(g: dict, b: int, near_bins: int, agree: float | None, at: str, base: float, read: bool) -> list:
    """overlay.route_at with its variants: ``agree`` px (None: no check) between the person's route's base and the
    model's route at the bin drawn (``at="here"``) or at the person's bin (``"theirs"``); ``read``: the model's route
    only where the model read a tube."""
    model = overlay.model_route(g, b)
    drawn = [t for t in g.get("human") or [] if t.get("pts")]
    if not drawn:
        return model
    near = [t for t in drawn if abs(t["bin"] - b) <= near_bins]
    if not near and len(model) >= 2 and (not read or overlay.read_by_model(g, b)):
        later = [t for t in drawn if t["bin"] >= b]
        ref = later[0] if later else drawn[-1]
        other = model if at == "here" else overlay.model_route(g, ref["bin"])
        if agree is None or (len(other) >= 2 and base_dist(ref["pts"], other, base) <= agree):
            route = overlay.along(model, ref["pts"], float(g["L"][b]))
            if route is not None:
                return route
    pool = near or drawn
    later = [t for t in pool if t["bin"] >= b]
    return later[0]["pts"] if later else overlay.continued(pool[-1]["pts"], model)


def open_movie(movie: str) -> RunData:
    folder = SCRATCH / movie
    folder.mkdir(parents=True, exist_ok=True)
    pred = folder / "predictions.json"
    if not pred.exists():
        shutil.copy(BASE / f"{movie}_real_0" / "predictions.json", pred)
    return RunData(RunFolder(folder))


def cases(movie: str) -> list[dict]:
    """Every simulated review of a movie: the grain's record as the app would hold it after the person's traces,
    and the targets to score."""
    data = open_movie(movie)
    out = []
    for g in load_movie(movie):
        if g.gid not in data.model:
            continue
        drift = data._static[g.gid]["drift"]
        for proto in g.protocols():
            targets = g.targets(proto)
            if not targets:
                continue
            anchors = g.anchors(proto)
            la, fv, _ = g.review_onset(min(b for b, _, _ in anchors), "review")
            L = reviewed_curve(g.model_px, fv, [(b, h) for b, h, _ in anchors])
            rec = dict(data.grain(g.gid))
            rec["Lm"] = list(rec["L"])  # the model's own curve, as RunData keeps it once a person's answers change it
            rec["L"] = [float(v) for v in L]
            rec["human"] = []
            for b, h, t in anchors:
                d = np.asarray(drift[b] if drift else (0.0, 0.0), float)
                rec["human"].append({"bin": b, "state": "full", "L": h,
                                     "pts": (np.asarray(t["path_xy_ref"], float) - d).tolist()})
            rec["human"].sort(key=lambda t: t["bin"])
            out.append({"gid": g.gid, "proto": proto, "rec": rec, "targets": targets})
    return out


def score(cs: list[dict], policy) -> dict:
    """rows[proto] = per-target rows (gid, hit, both, tip error) with tips from ``policy(rec, b)`` -> tip (ref)."""
    rows = {p: [] for p in ("P1", "P2", "P3")}
    for c in cs:
        rec = c["rec"]
        for b, h, apex, t in c["targets"]:
            L = rec["L"][b]
            hit = abs(L - h) <= tol(h)
            tip = policy(rec, b) if L >= overlay.MIN_TUBE_PX else None
            te = float(np.hypot(*(np.asarray(tip) - apex))) if tip is not None else None
            rows[c["proto"]].append({"gid": c["gid"], "bin": b, "hit": bool(hit), "tip_err": te,
                                     "both": bool(hit and te is not None and te <= tip_tol(h))})
    return rows


def tip_on(route_fn):
    def tip(rec: dict, b: int):
        route = route_fn(rec, b)
        if len(route) < 2:
            return None
        return overlay.shift(overlay.to_length(route, rec["L"][b]), overlay.drift_at(rec, b))[-1]
    return tip


def new_tip(rec: dict, b: int):
    tube = overlay.tube_at(rec, b)
    return tube[-1] if tube else None


def both(rows) -> int:
    return sum(r["both"] for p in rows for r in rows[p])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", default="ld")
    ap.add_argument("--grid", action="store_true", help="rank route variants on the tuning movie")
    a = ap.parse_args(argv)
    data = {m: cases(m) for m in ("ld", "m2", "m1")}
    summary = {"near_bins": overlay.NEAR_BINS, "join_px": overlay.JOIN_PX, "movies": {}}
    if a.grid:
        ranked = []
        for read, near, agree, at in itertools.product((True, False), (0, 20, 40), (None, 3.0, 8.0, 12.0),
                                                       ("here", "theirs")):
            if agree is None and at != "here":
                continue
            fn = tip_on(lambda g, b, n=near, ag=agree, w=at, r=read: variant_route_at(g, b, n, ag, w, 10.0, r))
            r = {m: score(cs, fn) for m, cs in data.items()}
            ranked.append((both(r[a.tune]), near, agree, at, read, {m: both(x) for m, x in r.items()}))
        ranked.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        print(f"route variants ranked on {a.tune} by length-and-tip over P1-P3 (agree None: no check; base 10 px):")
        for tot, near, agree, at, read, per in ranked:
            print(f"   read-only {str(read):5s} near {near:3d} agree {str(agree):5s} at {at:6s}: "
                  + " ".join(f"{m} {v}" for m, v in per.items()))
        summary["grid"] = ranked
    print(f"overlay.route_at: near {overlay.NEAR_BINS} bins, join {overlay.JOIN_PX} px")
    print("| movie | protocol | targets | lengths | len & tip, person's route (old) | len & tip, new | new - old "
          "[95% CI] | median tip error old / new (px) |")
    print("|---|---|---|---|---|---|---|---|")
    old_tip = tip_on(old_route_at)
    for m, cs in data.items():
        ro, rn = score(cs, old_tip), score(cs, new_tip)
        summary["movies"][m] = {"old": ro, "new": rn}
        for p in ("P1", "P2", "P3"):
            if not ro[p]:
                continue
            pc = paired([dict(r, hit=r["both"]) for r in ro[p]], [dict(r, hit=r["both"]) for r in rn[p]])
            med = lambda rows: np.median([r["tip_err"] for r in rows if r["tip_err"] is not None])
            print(f"| {m} | {p} | {len(ro[p])} | {sum(r['hit'] for r in ro[p])} | {sum(r['both'] for r in ro[p])} | "
                  f"{sum(r['both'] for r in rn[p])} | {pc['lengths'][0]:+d} [{pc['lengths'][1]:+.0f}, "
                  f"{pc['lengths'][2]:+.0f}] | {med(ro[p]):.1f} / {med(rn[p]):.1f} |")
    out = Path(__file__).resolve().parents[2] / "runs/research/review_curve/routes"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, default=float))


if __name__ == "__main__":
    main()
