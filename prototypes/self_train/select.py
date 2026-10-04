"""Pseudo-labels for label-free per-movie adaptation: the tracker's own confident readings of a movie, written as
traces in the labelling tool's format, so the crop builders take them as they take a person's traces.

    python -m prototypes.self_train.select PRED.json --movie m1 --name r1 [--maps CACHE_DIR | --model NET.pt]
        [--diagnose]

``PRED.json``: SparseTrack's predictions for the movie (e.g. 0.8.8's). The maps are the network's tube probability
that produced them (the shipped network's cache on disk for movie 1, read only; or any network computed in memory).
Writes runs/research/self_train/pseudo/<name>.json (labels format, ``review_origin: model``) and <name>_stats.json.
``--diagnose`` then compares the selection with the human labels (only reported; nothing is chosen with them).

Rules (fixed before any result was looked at; the human labels are never read by the selection):

- a germinated grain is used only if no flag says its reading is unsafe (``touches:``, ``drift_rejected``,
  ``shared_change_split``, ``rotates:``, ``onset_at_focus_change``); a grain lost partway is used up to
  ``lost_margin`` bins before it was lost;
- a bin is a confident reading if the length there is >= ``min_len`` px, ``review.trace_confidence`` >= ``min_conf``
  (the tube grew within the last few bins and is long enough), the drawn tube (as the app draws it: bent, turned,
  cut to the length, moved by the grain's drift) lies on the map (P >= 0.5) for >= ``min_share`` of its length
  beyond the rim (``learned.drawn_check``'s measure, at every bin), and no point of it is within ``edge_px`` of the
  frame edge;
- pseudo-traces: every ``spacing``-th confident bin, at most ``cap`` per grain spread over its confident bins; each
  is the drawn tube cut ``apex_cut`` px short of the reading's apex (state ``partial``; the crop builder leaves a
  ``tip_blind`` px disc round the reading's apex unscored); its background band is dropped (``contact``) where more
  than ``band_max`` of it is marked on the map (another tube may run there);
- negatives (no tube at the exit): bins up to ``guard`` bins before the earlier of the tracker's onset and the
  emergence extrapolated back from the first confident reading (length / growth over the next 10 bins, growth
  clipped to 0.2..5 px per bin), every ``neg_every`` bins, only on grains with at least one pseudo-trace;
- grains read as never germinated, not flagged, whose surroundings stay clean on the map (at >= ``clean_frac`` of
  every 10th bin no more than ``clean_max_px`` px of P >= 0.5 within r+1..r+``clean_ring`` px): no tube at any bin
  (``clean_every``).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from prototypes.tube_adapt.common import MOVIES, REPO
from sparsetrack import routes
from sparsetrack.review import lengths_by_bin, trace_confidence

OUT = REPO / "runs/research/self_train"
P_SCALE = 250.0
UNSAFE = ("touches:", "drift_rejected", "shared_change_split", "rotates:", "onset_at_focus_change")


@dataclass
class Rules:
    min_len: float = 8.0
    min_conf: float = 0.7
    min_share: float = 0.8
    lost_margin: int = 10
    edge_px: float = 10.0
    spacing: int = 3
    cap: int = 40
    apex_cut: float = 3.0
    tip_blind: float = 14.0    # crop builder: unscored disc round the reading's apex
    band_max: float = 0.05
    guard: int = 10
    neg_every: int = 4
    clean_ring: float = 20.0
    clean_max_px: int = 5
    clean_frac: float = 0.95
    clean_every: int = 8
    centre: bool = True       # move each pseudo-route onto the middle of its band on the map (bins b-1..b+1)
    negatives: bool = False   # pseudo-negatives before the onset (pre-registered on; off after the diagnosis: see README)


class Maps:
    """A probability movie (uint8 P x 250) in reference coordinates: the network's cache on disk (memory-mapped,
    read only) or computed in memory for a network."""

    def __init__(self, movie: str, cache: str | None = None, model: str | None = None, log=print):
        if cache:
            self.arr = np.load(Path(cache) / "bins.npy", mmap_mode="r")
            meta = json.loads((Path(cache) / "meta.json").read_text())
            self.shifts = np.asarray(meta["shifts"], float)
        else:
            from prototypes.tube_net.pixels import prob_movie
            from sparsetrack import stack
            _, meta = stack.load(REPO / MOVIES[movie][0])
            self.arr, m = prob_movie(model, movie, list(range(int(meta["n_bins"]))), log=log)
            self.shifts = np.asarray(m["shifts"], float)
        self.h, self.w = self.arr.shape[1:]

    def at(self, b: int, pts: np.ndarray) -> np.ndarray:
        q = (np.asarray(pts, float) + self.shifts[b] - 0.5).astype(np.float32)
        v = cv2.remap(np.asarray(self.arr[b]), q[:, 0][None], q[:, 1][None], cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return v.ravel() / P_SCALE

    def window(self, b: int, lo: np.ndarray, hi: np.ndarray):
        """(P window, its reference origin) over [lo, hi) (reference px, clipped to the frame)."""
        sx, sy = self.shifts[b]
        x0, y0 = int(max(0, math.floor(lo[0] + sx))), int(max(0, math.floor(lo[1] + sy)))
        x1, y1 = int(min(self.w, math.ceil(hi[0] + sx))), int(min(self.h, math.ceil(hi[1] + sy)))
        return np.asarray(self.arr[b, y0:y1, x0:x1], np.float32) / P_SCALE, np.array([x0 - sx, y0 - sy])


def series(res: dict, fpb: int, nb: int):
    """Per bin from bin 0: length (review.lengths_by_bin), drift, rotation (degrees), and the series index."""
    frames = np.asarray((res.get("length") or {}).get("frames") or [], int)
    L = lengths_by_bin(res, fpb, nb)
    start = int(frames[0]) // fpb if len(frames) else 0
    n = len(frames)
    drift = np.zeros((nb, 2))
    if res.get("drift"):
        d = np.asarray(res["drift"]["xy"], float)
        drift[start:start + len(d)] = d[: nb - start]
        drift[start + len(d):] = d[-1] if len(d) else 0.0
    rot = np.zeros(nb)
    r = np.asarray(res.get("rotation_deg") or [], float)
    if len(r) == n:
        rot[start:start + n] = r[: nb - start]
    return L, drift, rot, start


def drawn(res: dict, i: int, length: float, drift: np.ndarray, rot: float, pivot: np.ndarray) -> np.ndarray:
    """The tube the app draws at series index ``i`` in reference coordinates (``learned.drawn_check``)."""
    th = math.radians(float(rot))
    turn = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    return (routes.cut(routes.route_at(res, i), length) - pivot) @ turn.T + pivot + drift


def on_map(maps: Maps, b: int, route: np.ndarray, centre: np.ndarray, r: float) -> float:
    pts, s = routes.resample(route, 1.0)
    keep = (np.hypot(*(pts - centre).T) > r + 3.0) & (s < s[-1] - 2.0)
    if keep.sum() < 4:
        return float("nan")
    return float(np.mean(maps.at(b, pts[keep]) >= 0.5))


def band_marked(maps: Maps, b: int, route: np.ndarray, apex: np.ndarray, centre: np.ndarray, r: float,
                gap: float = 6.0, band: float = 14.0, blind: float = 10.0) -> float:
    """Share of the background band (gap..band px off the route; not within ``blind`` of the apex, not on the
    grain) marked on the map."""
    lo, hi = route.min(axis=0) - band - 3, route.max(axis=0) + band + 3
    P, org = maps.window(b, lo, hi)
    if P.size == 0:
        return 0.0
    q = route - org - 0.5
    line = np.zeros(P.shape, np.uint8)
    cv2.polylines(line, [np.round(q * 4).astype(np.int32).reshape(-1, 1, 2)], False, 1, 1, cv2.LINE_8, 2)
    d = cv2.distanceTransform(1 - line, cv2.DIST_L2, 5)
    jj, ii = np.meshgrid(np.arange(P.shape[1]), np.arange(P.shape[0]))
    a, c = apex - org - 0.5, centre - org - 0.5
    ring = (d >= gap) & (d <= band) & (np.hypot(jj - a[0], ii - a[1]) > blind) & (np.hypot(jj - c[0], ii - c[1]) > r + 2)
    return float(np.mean(P[ring] >= 0.5)) if ring.any() else 0.0


def centred(maps: Maps, b: int, route: np.ndarray, nb: int) -> tuple[np.ndarray, float]:
    """The route moved along its normals onto the middle of its band on the map (averaged over bins b-1..b+1), as
    ``learned.centre_route`` centres a reading's route (Params' centring defaults); (route every px, median shift)."""
    from sparsetrack.analyze import Params
    from sparsetrack.learned import _across, _centres_along
    p = Params()
    pts, _ = routes.resample(route, 1.0)
    if len(pts) < 3:
        return pts, 0.0
    nrm = routes.normals(pts)
    offs = np.arange(-10.0, 10.25, 0.5)

    class _R:
        bins, shifts = maps.arr, maps.shifts
    prof = _across(_R, [(x, pts, nrm) for x in (b - 1, b, b + 1) if 0 <= x < nb], offs)
    ci = _centres_along(prof, offs, p)
    if ci is None:
        return pts, 0.0
    return pts + ci[:, None] * nrm, float(np.median(np.abs(ci)))


def ring_marks(maps: Maps, b: int, centre: np.ndarray, r: float, out: float) -> int:
    P, org = maps.window(b, centre - r - out - 2, centre + r + out + 2)
    if P.size == 0:
        return 0
    jj, ii = np.meshgrid(np.arange(P.shape[1]), np.arange(P.shape[0]))
    d = np.hypot(jj + org[0] + 0.5 - centre[0], ii + org[1] + 0.5 - centre[1])
    return int(np.sum((P >= 0.5) & (d > r + 1) & (d <= r + out)))


def select(pred: dict, census: dict, maps: Maps, nb: int, rules: Rules, width: int, height: int, log=print):
    fpb = int(pred.get("frames_per_bin", 300))
    exit_pivot = (pred.get("params") or {}).get("rot_pivot") == "exit"
    rs = int(pred["grains"][0]["length"]["frames"][0]) // fpb if pred["grains"] else 0
    lo, hi = rs + 3, nb - 2
    labels, stats = {}, {"grains": {}, "rules": asdict(rules)}
    for res in pred["grains"]:
        gid = res["id"]
        g = census.get(gid)
        if g is None or g.get("excluded"):
            continue
        st = {"status": res.get("status"), "flags": [f for f in res.get("flags", []) if not f.startswith("reader")]}
        stats["grains"][gid] = st
        L, drift, rot, start = series(res, fpb, nb)
        centre0 = np.array([res["x"], res["y"]], float)
        unsafe = [f for f in res.get("flags", []) if f.startswith(UNSAFE)]
        if unsafe:
            st["skipped"] = "unsafe flags " + ",".join(unsafe)
            continue
        status = res.get("status") or ""
        if status.startswith("emerged"):
            pivot = np.asarray(res["exit_xy"] if exit_pivot and res.get("exit_xy") else centre0, float)
            onset_bin = int(res["onset_frame"]) // fpb if res.get("onset_frame") is not None else rs
            until = res.get("observed_until_frame")
            until_bin = int(until) // fpb - rules.lost_margin if until is not None else nb
            elig, n_reason = [], {"short": 0, "conf": 0, "edge": 0, "share": 0, "lost": 0}
            for b in range(max(lo, onset_bin), hi + 1):
                if L[b] < rules.min_len:
                    n_reason["short"] += 1
                    continue
                if b > until_bin:
                    n_reason["lost"] += 1
                    continue
                conf = trace_confidence(L, b)
                if conf < rules.min_conf:
                    n_reason["conf"] += 1
                    continue
                i = b - start
                route = drawn(res, i, L[b], drift[b], rot[b], pivot)
                if (route[:, 0].min() < rules.edge_px or route[:, 1].min() < rules.edge_px
                        or route[:, 0].max() > width - rules.edge_px or route[:, 1].max() > height - rules.edge_px):
                    n_reason["edge"] += 1
                    continue
                share = on_map(maps, b, route, centre0 + drift[b], res["r"])
                if not share >= rules.min_share:
                    n_reason["share"] += 1
                    continue
                elig.append((b, i, conf, share, route))
            st["confident_bins"] = len(elig)
            st["rejected_bins"] = n_reason
            if not elig:
                st["skipped"] = "no confident bin"
                continue
            chosen = elig[:: rules.spacing]
            if len(chosen) > rules.cap:
                chosen = [chosen[k] for k in np.round(np.linspace(0, len(chosen) - 1, rules.cap)).astype(int)]
            traces = {}
            shifts = []
            for b, i, conf, share, route in chosen:
                if rules.centre:
                    route, sh = centred(maps, b, route, nb)
                    shifts.append(sh)
                apex = route[-1]
                cutr = routes.cut(route, max(routes.arc(route)[-1] - rules.apex_cut, 2.0))
                bm = band_marked(maps, b, route, apex, centre0 + drift[b], res["r"])
                traces[str(b)] = {"bin": b, "state": "partial", "path_xy_ref": np.round(cutr, 2).tolist(),
                                  "apex_xy_ref": np.round(apex, 2).tolist(),
                                  "view_offset": np.round(drift[b], 2).tolist(), "contact": bool(bm > rules.band_max),
                                  "length_px": round(float(L[b]), 2), "model_confidence": round(conf, 3),
                                  "on_map": round(share, 3), "band_marked": round(bm, 3), "review_origin": "model"}
            b1, i1 = elig[0][0], elig[0][1]
            j = min(b1 + 10, hi)
            v = (L[j] - L[b1]) / (j - b1) if j > b1 else float("nan")
            v = float(np.clip(v, 0.2, 5.0)) if np.isfinite(v) else 0.2
            emerge = b1 - L[b1] / v
            neg_until = int(math.floor(min(onset_bin, emerge))) - rules.guard
            neg_bins = list(range(lo, neg_until + 1, rules.neg_every)) if rules.negatives else []
            st["neg_bins_rule"] = len(range(lo, neg_until + 1, rules.neg_every))
            st["centring_shift_median"] = round(float(np.median(shifts)), 2) if shifts else None
            neg_route = elig[0][4]
            labels[gid] = {"onset": {"verdict": "emerged_within", "last_absent_bin": neg_until + 1,
                                     "first_visible_bin": onset_bin, "review_origin": "model"},
                           "traces": traces,
                           "pseudo": {"kind": "germinated", "neg_bins": neg_bins, "neg_route": np.round(neg_route, 2).tolist(),
                                      "neg_route_bin": b1, "drift": np.round(drift, 2).tolist(),
                                      "emerge_est": round(float(emerge), 1), "growth": round(v, 3),
                                      "onset_bin": onset_bin, "confident": [e[0] for e in elig],
                                      "neg_bins_rule": list(range(lo, neg_until + 1, rules.neg_every))}}
            st.update(pseudo_traces=len(traces), negatives=len(neg_bins), first_confident=b1,
                      emerge_est=round(float(emerge), 1), onset_bin=onset_bin,
                      contact=sum(t["contact"] for t in traces.values()))
        elif status == "no_emergence_by_end":
            checked = list(range(lo, hi + 1, 10))
            marks = [ring_marks(maps, b, centre0 + drift[b], res["r"], rules.clean_ring) for b in checked]
            clean = float(np.mean([m <= rules.clean_max_px for m in marks])) if marks else 0.0
            st.update(clean_share=round(clean, 3), ring_marks_max=int(max(marks) if marks else 0))
            if clean < rules.clean_frac:
                st["skipped"] = "surroundings marked"
                continue
            labels[gid] = {"onset": {"verdict": "no_emergence_by_end", "review_origin": "model"}, "traces": {},
                           "pseudo": {"kind": "clean", "clean_bins": list(range(lo, hi + 1, rules.clean_every)),
                                      "drift": np.round(drift, 2).tolist()}}
            st["clean_bins"] = len(labels[gid]["pseudo"]["clean_bins"])
        else:
            st["skipped"] = f"status {status}"
    tot = {"grains_read": len(stats["grains"]),
           "germinated_used": sum(1 for v in labels.values() if v["pseudo"]["kind"] == "germinated"),
           "clean_used": sum(1 for v in labels.values() if v["pseudo"]["kind"] == "clean"),
           "pseudo_traces": sum(len(v["traces"]) for v in labels.values()),
           "confident_bins": sum(s.get("confident_bins", 0) for s in stats["grains"].values()),
           "negative_bins": sum(len(v["pseudo"].get("neg_bins", [])) for v in labels.values()),
           "clean_bins": sum(len(v["pseudo"].get("clean_bins", [])) for v in labels.values()),
           "skipped": {k: s["skipped"] for k, s in stats["grains"].items() if "skipped" in s}}
    stats["totals"] = tot
    return labels, stats


def write_doc(movie: str, labels: dict, out: Path, source: str) -> Path:
    from prototypes.tube_adapt.common import labels as human
    H = human(movie)
    doc = {k: H[k] for k in ("schema", "frames_per_bin", "n_bins", "coordinates", "bin_semantics", "grains", "movie")
           if k in H}  # the census and movie metadata only: no human answer is copied
    doc.update({"labels": labels, "pseudo_source": source})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc))
    return out


# ---------------------------------------------------------------------------------------------- diagnosis only
def diagnose(movie: str, doc: dict, pred: dict, maps: Maps | None = None) -> dict:
    """How right the pseudo-labels are, judged on the human labels (reported, never used to select):

    - pseudo-traces: the tracker's reading at each human FULL trace's bin (not touching), for traces whose bin the
      rules would accept (a confident bin of a used grain) against the rest - length within max(2 px, 10%), apex
      within max(5 px, 10%), and the share of the reading's route within 3 px of the human route (as far as the
      shorter of the two reaches); PARTIAL human traces (lower bounds): the reading not beyond them;
    - negatives: pseudo-negative bins at or after the human first visible bin (a tube was there) - wrong;
    - clean grains: the human verdict."""
    from prototypes.tube_adapt.common import labels as human
    H = human(movie)
    fpb = int(pred.get("frames_per_bin", 300))
    nb = int(H["n_bins"])
    preds = {r["id"]: r for r in pred["grains"]}
    exit_pivot = (pred.get("params") or {}).get("rot_pivot") == "exit"
    out = {"accepted": [], "other": [], "partial_accepted": [], "negatives": {}, "clean": {}}
    for gid, lab in H["labels"].items():
        if H["grains"][gid].get("excluded") or gid not in preds:
            continue
        res = preds[gid]
        L, drift, rot, start = series(res, fpb, nb)
        pl = doc["labels"].get(gid)
        conf_bins = set()
        if pl and pl["pseudo"]["kind"] == "germinated":
            conf_bins = set(pl["pseudo"].get("confident", []))
        pivot = np.asarray(res["exit_xy"] if exit_pivot and res.get("exit_xy") else [res["x"], res["y"]], float)
        for b, t in (lab.get("traces") or {}).items():
            b = int(b)
            if t["state"] not in ("full", "partial") or len(t.get("path_xy_ref") or []) < 2 or t.get("contact"):
                continue
            h = float(t["length_px"])
            hp = np.asarray(t["path_xy_ref"], float)
            m = float(L[b]) if b < nb else 0.0
            row = {"grain": gid, "bin": b, "human": round(h, 1), "model": round(m, 1), "state": t["state"]}
            if m >= 2 and res.get("path"):
                route = drawn(res, b - start, m, drift[b], rot[b], pivot)
                row["tip_err"] = round(float(np.hypot(*(route[-1] - hp[-1]))), 1)
                hq, _ = routes.resample(hp, 1.0)

                def near(rt):
                    pts, _ = routes.resample(rt, 1.0)
                    d = np.hypot(pts[:, None, 0] - hq[None, :, 0], pts[:, None, 1] - hq[None, :, 1]).min(axis=1)
                    n = min(len(pts), len(hq))
                    return round(float(np.mean(d[:n] <= 3.0)), 2), round(float(np.median(d[:n])), 2)
                row["on_human"], row["dist_med"] = near(route)
                if maps is not None and b in conf_bins:
                    rc, _ = centred(maps, b, route, nb)
                    row["on_human_centred"], row["dist_med_centred"] = near(rc)
                    row["tip_err_centred"] = round(float(np.hypot(*(rc[-1] - hp[-1]))), 1)
            row["len_ok"] = abs(m - h) <= max(2.0, 0.1 * h) if t["state"] == "full" else m <= h + max(2.0, 0.1 * h)
            row["tip_ok"] = row.get("tip_err", 1e9) <= max(5.0, 0.1 * h)
            key = ("accepted" if b in conf_bins else "other") if t["state"] == "full" else (
                "partial_accepted" if b in conf_bins else None)
            if key:
                out[key].append(row)
        if pl and pl["pseudo"]["kind"] == "germinated":
            on = lab.get("onset") or {}
            negs = pl["pseudo"].get("neg_bins_rule", pl["pseudo"]["neg_bins"])  # the pre-registered rule's
            if on.get("verdict") == "emerged_within":
                wrong = [b for b in negs if b >= int(on["first_visible_bin"])]
                out["negatives"][gid] = {"n": len(negs), "wrong": len(wrong), "human_first_visible": on["first_visible_bin"],
                                         "last_neg": max(negs) if negs else None}
            elif on.get("verdict") == "emerged_at_start":
                out["negatives"][gid] = {"n": len(negs), "wrong": len(negs), "human_first_visible": 0}
            else:
                out["negatives"][gid] = {"n": len(negs), "wrong": 0, "human": on.get("verdict")}
        if pl and pl["pseudo"]["kind"] == "clean":
            on = lab.get("onset") or {}
            out["clean"][gid] = on.get("verdict")

    def summ(rows):
        if not rows:
            return {"n": 0}
        return {"n": len(rows), "len_ok": sum(r["len_ok"] for r in rows), "tip_ok": sum(r["tip_ok"] for r in rows),
                "both_ok": sum(r["len_ok"] and r["tip_ok"] for r in rows),
                "on_human_median": float(np.median([r.get("on_human", 0.0) for r in rows])),
                "dist_median": float(np.median([r.get("dist_med", 99.0) for r in rows])),
                "on_human_centred_median": float(np.median([r.get("on_human_centred", np.nan) for r in rows])),
                "dist_centred_median": float(np.median([r.get("dist_med_centred", np.nan) for r in rows])),
                "median_err": float(np.median([r["model"] - r["human"] for r in rows]))}
    out["summary"] = {"accepted": summ(out["accepted"]), "other": summ(out["other"]),
                      "partial_accepted": summ(out["partial_accepted"]),
                      "negatives": {"n": sum(v["n"] for v in out["negatives"].values()),
                                    "wrong": sum(v["wrong"] for v in out["negatives"].values()),
                                    "grains_with_wrong": sum(v["wrong"] > 0 for v in out["negatives"].values())},
                      "clean": out["clean"]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pred")
    ap.add_argument("--movie", default="m1")
    ap.add_argument("--name", required=True)
    ap.add_argument("--maps", help="probability cache directory of the network that made PRED (read only)")
    ap.add_argument("--model", help="network that made PRED (its maps computed in memory)")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="rule overrides key=value")
    a = ap.parse_args(argv)
    rules = Rules()
    for kv in a.set:
        k, v = kv.split("=", 1)
        setattr(rules, k, type(getattr(rules, k))(v))
    pred = json.loads(Path(a.pred).read_text())
    from prototypes.tube_adapt.common import labels as human
    census = human(a.movie)["grains"]  # the census only (positions, radii, exclusions)
    nb = int(human(a.movie)["n_bins"])
    maps = Maps(a.movie, cache=a.maps, model=a.model)
    labels, stats = select(pred, census, maps, nb, rules, maps.w, maps.h)
    out = OUT / "pseudo" / f"{a.name}.json"
    write_doc(a.movie, labels, out, str(a.pred))
    doc = json.loads(out.read_text())
    doc["rules"] = asdict(rules)
    out.write_text(json.dumps(doc))
    (OUT / "pseudo" / f"{a.name}_stats.json").write_text(json.dumps(stats, indent=1, default=float))
    t = stats["totals"]
    print(f"{a.name}: {t['grains_read']} grains read; used {t['germinated_used']} germinated + {t['clean_used']} "
          f"clean; {t['confident_bins']} confident bins -> {t['pseudo_traces']} pseudo-traces; "
          f"{t['negative_bins']} negative bins; {t['clean_bins']} clean bins", flush=True)
    for gid, s in stats["grains"].items():
        print(f"  {gid}: {s.get('status')} {s.get('skipped') or ''} conf {s.get('confident_bins', '-')} "
              f"traces {s.get('pseudo_traces', '-')} neg {s.get('negatives', '-')} first {s.get('first_confident', '-')} "
              f"onset {s.get('onset_bin', '-')} emerge {s.get('emerge_est', '-')} rej {s.get('rejected_bins', '')} "
              f"{s.get('flags')}", flush=True)
    if a.diagnose:
        report(a.movie, out, pred, maps)


def report(movie: str, doc_path: Path, pred: dict, maps: Maps | None = None) -> dict:
    doc = json.loads(Path(doc_path).read_text())
    d = diagnose(movie, doc, pred, maps)
    s = d["summary"]
    print(f"diagnosis on the human labels ({movie}):")
    for k in ("accepted", "other", "partial_accepted"):
        print(f"  {k}: {s[k]}")
    print(f"  negatives: {s['negatives']}")
    print(f"  clean grains (human verdict): {s['clean']}")
    for r in d["accepted"]:
        print(f"    accepted {r}")
    for gid, v in d["negatives"].items():
        if v["wrong"]:
            print(f"    negatives wrong {gid}: {v}")
    Path(str(doc_path).replace(".json", "_diagnosis.json")).write_text(json.dumps(d, indent=1, default=float))
    return d


if __name__ == "__main__":
    main()
