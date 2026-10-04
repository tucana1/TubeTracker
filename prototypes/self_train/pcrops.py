"""Training crops from pseudo-labels (``select.py``), as ``prototypes/tube_net/realdata.py`` (version 3, movie 1's
options) builds them from a person's traces, with wider unscored margins for a reading's uncertain apex:

- per pseudo-trace (the centred route cut 3 px short of the reading's apex): ``along`` crops centred along it and
  ``exits`` at its exit (jittered by up to a third of the crop), and the same route at bins b +/- 1 where the grain
  did not move (the tracker's own drift); tube within 3 px of the route, scored background 6..14 px off it (not on
  routes whose band is marked on the map: ``contact``), the grain's inside background (unscored on a route that
  starts over the grain: a face-on pore), nothing within ``tip_blind`` px of the reading's apex scored (the tube may
  go on, or end short of it);
- negatives and clean grains if the pseudo-labels have them (``select.Rules.negatives``; off by default);
- ``--prop``: also ``prototypes/tube_adapt/propagate.py``'s crops from the pseudo-traces (the tube between two
  pseudo-traces across bins that were not confident readings, and after the last one while the optical flow carries
  it reliably), with the tracker's own drift as the grain offsets; no negatives unless the pseudo-labels have them.

    python -m prototypes.self_train.pcrops runs/research/self_train/pseudo/m1_r1.json --movie m1 [--prop]

Shards: runs/research/self_train/shards/{trace,prop}_<pseudo name>.npz.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from prototypes.learned_flood.data import CacheView
from prototypes.self_train.select import OUT
from prototypes.tube_adapt.common import MOVIES, REPO
from prototypes.tube_net.realdata import _targets


def build(doc_path: str | Path, movie: str, out: str | Path, half: int = 48, along: int = 4, exits: int = 1,
          neighbours: tuple = (-1, 1), tip_blind: float = 14.0, over_grain: float = 0.6, max_move: float = 1.5,
          seed: int = 0, log=print) -> Path:
    doc = json.loads(Path(doc_path).read_text())
    tip_blind = float((doc.get("rules") or {}).get("tip_blind", tip_blind))
    view = CacheView(REPO / MOVIES[movie][0])
    rng = np.random.default_rng(seed)
    lo, hi = view.rs + 3, view.n_bins - 2
    xs, bodies, ws, info, gids, kinds = [], [], [], [], [], []
    n = {"trace": 0, "exit": 0, "neighbour": 0, "neighbour_skipped": 0, "negative": 0, "clean": 0, "nan": 0}

    def add(b, cx, cy, body_w, kind, gid):
        cx = float(np.clip(round(cx), half + 4, view.r.width - half - 4))
        cy = float(np.clip(round(cy), half + 4, view.r.height - half - 4))
        x = view.sample(b, cx, cy, half)
        if not np.isfinite(x).all():
            n["nan"] += 1
            return
        body, w = body_w(cx, cy)
        xs.append(x.astype(np.float16)); bodies.append(body); ws.append(w)
        info.append((b, cx, cy, int(kind in ("negative", "clean")))); gids.append(gid); kinds.append(kind)
        n[kind] += 1

    jit = lambda: rng.uniform(-half / 3, half / 3)
    for gid, lab in doc["labels"].items():
        g = doc["grains"][gid]
        ps = lab["pseudo"]
        drift = np.asarray(ps["drift"], float)
        for bs, t in sorted(lab["traces"].items(), key=lambda kv: int(kv[0])):
            b = int(bs)
            path = np.asarray(t["path_xy_ref"], float)
            apex = np.asarray(t["apex_xy_ref"], float)
            off = t.get("view_offset") or [0.0, 0.0]
            gxy = (g["x"] + off[0], g["y"] + off[1])
            band = not t.get("contact")
            over = bool(over_grain) and np.hypot(path[0][0] - gxy[0], path[0][1] - gxy[1]) < over_grain * g["r"]
            s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])

            def tw(blind_r):
                return lambda cx, cy: _targets(path, cx, cy, half, gxy, g["r"], False, band,
                                               [(apex[0], apex[1], blind_r)], False, over)
            for _ in range(along):
                u = rng.uniform(0, s[-1])
                add(b, np.interp(u, s, path[:, 0]) + jit(), np.interp(u, s, path[:, 1]) + jit(), tw(tip_blind),
                    "trace", gid)
            for _ in range(exits):
                add(b, path[0][0] + jit(), path[0][1] + jit(), tw(tip_blind), "exit", gid)
            for k in neighbours:
                if not lo <= b + k <= hi:
                    continue
                if np.hypot(*(drift[b + k] - drift[b])) > max_move:
                    n["neighbour_skipped"] += 1
                    continue
                u = rng.uniform(0, s[-1])
                add(b + k, np.interp(u, s, path[:, 0]) + jit(), np.interp(u, s, path[:, 1]) + jit(),
                    tw(tip_blind + 1.5 * abs(k)), "neighbour", gid)
        if ps["kind"] == "germinated" and ps.get("neg_bins"):
            r0 = np.asarray(ps["neg_route"], float)
            b0 = int(ps["neg_route_bin"])
            s0 = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(r0, axis=0).T))])
            for b in ps["neg_bins"]:
                sh = drift[b] - drift[b0]
                path, gxy = r0 + sh, (g["x"] + drift[b][0], g["y"] + drift[b][1])
                for k in range(2):
                    u = 0.0 if k == 0 else rng.uniform(0, s0[-1])
                    add(b, np.interp(u, s0, path[:, 0]) + jit(), np.interp(u, s0, path[:, 1]) + jit(),
                        lambda cx, cy, path=path, gxy=gxy: _targets(path, cx, cy, half, gxy, g["r"], True, True, [],
                                                                    False, False), "negative", gid)
        if ps["kind"] == "clean":
            for b in ps["clean_bins"]:
                c = np.array([g["x"], g["y"]]) + drift[b]

                def ring(cx, cy, c=c):
                    jj, ii = np.meshgrid(np.arange(2 * half), np.arange(2 * half))
                    d = np.hypot(jj - (c[0] - cx + half - 0.5), ii - (c[1] - cy + half - 0.5))
                    return np.zeros((2 * half, 2 * half), np.uint8), (d <= g["r"] + 12.0).astype(np.uint8)
                add(b, c[0] + jit(), c[1] + jit(), ring, "clean", gid)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    B, W = np.stack(bodies), np.stack(ws)
    np.savez_compressed(out, x=np.stack(xs), body=B, tip=np.zeros(B.shape, np.float16), w=W,
                        info=np.array(info, np.float32), grain=np.array(gids), kind=np.array(kinds), movie=movie)
    log(f"{out.name}: {len(xs)} crops {n}; scored {100 * W.mean():.1f}%, tube among them "
        f"{100 * B.sum() / max(W.sum(), 1):.1f}%")
    return out


def build_prop(doc_path: str | Path, movie: str, out: str | Path, log=print) -> Path:
    """``propagate.build`` on the pseudo-traces (its labels and grain offsets replaced by the pseudo-labels' and the
    tracker's drift; the pseudo onsets dropped unless negatives were selected)."""
    from prototypes.tube_adapt import propagate
    doc = json.loads(Path(doc_path).read_text())
    d2 = copy.deepcopy(doc)
    for lab in d2["labels"].values():
        if not lab["pseudo"].get("neg_bins"):
            lab["onset"] = {}
    off = {gid: np.asarray(lab["pseudo"]["drift"], float) for gid, lab in doc["labels"].items()}
    propagate.labels = lambda m: d2
    propagate.offsets = lambda m: off
    grains = [gid for gid, lab in d2["labels"].items() if lab["traces"]]
    return propagate.build(out, grains, movie=movie, log=log)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pseudo")
    ap.add_argument("--movie", default="m1")
    ap.add_argument("--prop", action="store_true")
    a = ap.parse_args(argv)
    name = Path(a.pseudo).stem
    build(a.pseudo, a.movie, OUT / "shards" / f"trace_{name}.npz")
    if a.prop:
        build_prop(a.pseudo, a.movie, OUT / "shards" / f"prop_{name}.npz")


if __name__ == "__main__":
    main()
