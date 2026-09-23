"""Offline route evaluation with a SAMPLING-INVARIANT metric (rev8 step 2).

Supersedes the rev7 evaluator, whose `coverage()` counted stored
candidate *vertices* near the gold path and called 50% sufficient — a
two-point line could pass on one good endpoint (0.50 as vertices vs
0.158 when sampled every pixel; the rev8 review's counterexample).

What this does now:

1. `path_metrics(pred, gold)` resamples BOTH polylines at ~1 px
   arclength and reports two-sided precision/recall, root and terminal
   error, length ratio, a self-intersection (loop) flag, and the joint
   diagnostic (>=80% precision AND recall within 5 px, both endpoints
   within 5 px, length within 20%) — the same limits the review used,
   labelled diagnostic, not accepted product thresholds.
2. Every ranking rule is evaluated through `select.select_candidate`,
   the SAME function the movie runner calls, so an offline winner is
   the winner the pipeline would produce.
3. `--parity-run DIR` replays a saved run's candidates and checks the
   reproduced winners against that run's recorded selection decisions
   (exact parity is a precondition for quoting any offline number).

Usage:
    .venv/bin/python prototypes/v30_video_apex/eval_route_evidence.py \
        --candidates runs/prototypes/v30/movie_v17/candidates.json \
        --snapshot runs/prototypes/v30/snapshots/snap16 \
        --movie "/path/to/lowdens.mp4" \
        --out runs/prototypes/v30/route_evidence_eval_rev8.json \
        [--parity-run runs/prototypes/v30/movie_v18_evgate]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "prototypes" / "timesfm_tip_forecast"))

from prototypes.v30_video_apex.candidates import (  # noqa: E402
    read_candidates)
from prototypes.v30_video_apex.selection import (  # noqa: E402
    SelectionPolicy, select_candidate)

TOL_PX = 5.0
RESAMPLE_PX = 1.0
HALF = 14.0
N_OFF = 29


def resample(poly, step: float = RESAMPLE_PX) -> np.ndarray:
    pts = np.asarray(poly, float).reshape(-1, 2)
    if len(pts) < 2:
        return pts
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        d = float(np.hypot(*(b - a)))
        n = max(1, int(d // step))
        for k in range(1, n + 1):
            out.append(a + (b - a) * (k / n))
    return np.asarray(out, float)


def _min_dist_to_path(q: np.ndarray, path: np.ndarray) -> np.ndarray:
    seg = path[1:] - path[:-1]
    seglen = np.hypot(seg[:, 0], seg[:, 1]).clip(min=1e-9)
    q = np.atleast_2d(q)
    w = (((q[:, None, :] - path[None, :-1, :]) * seg[None, :, :]
          ).sum(axis=2) / (seglen ** 2)[None, :])
    proj = path[None, :-1, :] + np.clip(w, 0, 1)[..., None] * seg[None, :, :]
    return np.hypot(proj[..., 0] - q[:, 0, None],
                    proj[..., 1] - q[:, 1, None]).min(axis=1)


def path_metrics(pred, gold, tol: float = TOL_PX) -> dict:
    """Two-sided, sampling-invariant agreement between two polylines."""
    p = resample(pred)
    g = resample(gold)
    out: dict = {"n_pred": int(len(p)), "n_gold": int(len(g))}
    if len(p) < 2 or len(g) < 2:
        out.update({"precision": 0.0, "recall": 0.0, "f1": 0.0,
                    "root_err_px": float("inf"),
                    "terminal_err_px": float("inf"),
                    "length_ratio": 0.0, "loop": False, "joint_pass": False})
        return out
    dp = _min_dist_to_path(p, g)
    dg = _min_dist_to_path(g, p)
    precision = float((dp <= tol).mean())
    recall = float((dg <= tol).mean())
    lp = float(np.hypot(*(p[1:] - p[:-1]).T).sum())
    lg = float(np.hypot(*(g[1:] - g[:-1]).T).sum())
    # self-intersection: two points more than 20 samples apart (>= ~20 px
    # of arclength) closer than 2 px is a loop around something
    loop = False
    if len(p) > 30:
        idx = np.arange(len(p))
        d_pair = np.hypot(p[:, 0][:, None] - p[:, 0][None, :],
                          p[:, 1][:, None] - p[:, 1][None, :])
        far = np.abs(idx[:, None] - idx[None, :]) > 20
        loop = bool(np.min(np.where(far, d_pair, np.inf)) < 2.0)
    root_err = float(np.hypot(*(p[0] - g[0])))
    term_err = float(np.hypot(*(p[-1] - g[-1])))
    ratio = lp / lg if lg > 0 else 0.0
    joint = bool(precision >= 0.80 and recall >= 0.80
                 and root_err <= 5.0 and term_err <= 5.0
                 and 0.80 <= ratio <= 1.20 and not loop)
    out.update({"precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(2 * precision * recall /
                            max(precision + recall, 1e-9), 3),
                "root_err_px": round(root_err, 2),
                "terminal_err_px": round(term_err, 2),
                "length_px": round(lp, 1), "gold_length_px": round(lg, 1),
                "length_ratio": round(ratio, 3),
                "loop": loop, "joint_pass": joint})
    return out


def wall_energy(gray, pts, half: float = HALF, n_off: int = N_OFF):
    if len(pts) < 2:
        return np.zeros(0)
    g = np.asarray(gray, dtype=np.float32)
    h, w = g.shape
    tang = np.gradient(pts, axis=0)
    tang = tang / (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9)
    perp = np.column_stack([-tang[:, 1], tang[:, 0]])
    offs = np.linspace(-half, half, n_off)
    out = np.empty(len(pts))
    for j, (p, nv) in enumerate(zip(pts, perp)):
        xs = np.clip(p[0] + nv[0] * offs, 0, w - 1).astype(np.float32)[None, :]
        ys = np.clip(p[1] + nv[1] * offs, 0, h - 1).astype(np.float32)[None, :]
        pr = cv2.remap(g, xs, ys, cv2.INTER_LINEAR)[0]
        out[j] = float(np.abs(np.diff(pr)).max())
    return out


def load_gold(snapshot: Path) -> dict[str, list]:
    obs = json.loads((snapshot / "observations.json").read_text())
    return {str(o.get("obs_uuid", "")): [[float(q[0]), float(q[1])]
                                         for q in o["path_xy"]]
            for o in obs
            if o.get("path_xy") and len(o["path_xy"]) >= 2}


RULES = {
    # name: (rank, evidence_gate)
    "shipped_sel_gate0": ("sel", 0.0),
    "runtime_sel_gate0.6": ("sel", 0.6),
    "route_p_gate0.6": ("route_p", 0.6),
    "q_max_gate0.6": ("q_max", 0.6),
    "route_p_only": ("route_p", 0.0),
    "evidence_only": ("ev", 0.0),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--movie", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parity-run", default="",
                    help="saved run dir whose recorded winners must be "
                         "reproduced exactly by the shared selector")
    ap.add_argument("--measure-support-fallback", action="store_true",
                    help="DIAGNOSTIC ONLY (rev11): measure candidates "
                         "that lack a current path on their support "
                         "polyline, recording the fallback per row. The "
                         "default reports them unavailable — a support "
                         "line is not a measured path.")
    a = ap.parse_args()

    from tubetracker.annotation_frames import FrameReader

    gold = load_gold(Path(a.snapshot))
    # rev9 WP-A.5: read through the one candidate record, so a legacy
    # export (tip_score + nested evidence.wall_ev) is mapped once and
    # ranks identically to the runtime that produced it.
    cands = read_candidates(a.candidates)
    by_event: dict[str, list] = {}
    for c in cands:
        by_event.setdefault(
            str(c["candidate_id"]).rsplit(":", 1)[0], []).append(c)

    reader = FrameReader(a.movie)
    report: dict = {"events": {}, "rules": {}, "parity": {}}
    rule_stats: dict[str, dict] = {k: {"n": 0, "joint": 0, "sel": 0,
                                       "prec": [], "rec": []}
                                   for k in RULES}
    n_events = 0
    for key in sorted(by_event):
        rows = by_event[key]
        obs_uuid = str(rows[0].get("owner_id", ""))
        truth = gold.get(obs_uuid)
        frame = int(rows[0]["source_frame"])
        if truth is None:
            report["events"][key] = {"skipped": f"no human route {obs_uuid}"}
            continue
        n_events += 1
        res = reader.read(frame)
        gray = (res.frame if res.frame.ndim == 2
                else cv2.cvtColor(res.frame, cv2.COLOR_BGR2GRAY))
        gy, gx = np.gradient(gray.astype(np.float32))
        norm95 = float(np.percentile(np.abs(gx) + np.abs(gy), 95)) or 1.0
        for r in rows:
            # rev10 WP-C / rev11: score the MEASURED path (truncated at
            # the front head's cutoff). A missing current path is
            # reported UNAVAILABLE — it is never silently measured on the
            # support polyline. The declared --measure-support-fallback
            # flag restores the old behaviour for continuity
            # reproductions only, and the field used is recorded per row.
            _field = "current_path"
            _path = r.get("current_path")
            if not _path and a.measure_support_fallback:
                _field = ("support_path" if r.get("support_path")
                          else "polyline_native")
                _path = r.get(_field)
                _field = f"{_field} (declared fallback)"
            if not _path:
                r["measured_field"] = "unavailable"
                r["measured_points"] = 0
                r["ev"] = 0.0
                r["_m"] = None
                continue
            _path = np.asarray(_path, float)
            r["measured_field"] = _field
            r["measured_points"] = int(len(_path))
            pts = resample(_path, step=4.0)
            e = wall_energy(gray, pts)
            r["ev"] = float(np.median(e)) / norm95 if len(e) else 0.0
            r["_m"] = path_metrics(_path, truth)
        ev_rows: dict = {}
        for name, (rank, gate) in RULES.items():
            dec = select_candidate(rows, SelectionPolicy(
                rank=rank, evidence_gate=gate))
            m = (dec.winner or {}).get("_m") if dec.winner else None
            ev_rows[name] = {"route_id": (dec.winner or {}).get("route_id"),
                             "refused": dec.refused,
                             "reason": dec.reason, "metrics": m,
                             "measured_field": (dec.winner or {}).get(
                                 "measured_field")}
            st = rule_stats[name]
            st["n"] += 1
            if dec.winner and not dec.refused:
                st["sel"] += 1
                st["joint"] += int(bool(m and m["joint_pass"]))
                if m:
                    st["prec"].append(m["precision"])
                    st["rec"].append(m["recall"])
        # oracle: best candidate by whatever geometry the gold supports
        # (only candidates with a measured path can be scored; a run
        # where none is measurable reports None rather than crashing).
        _measurable = [r for r in rows if r.get("_m")]
        best = (max(_measurable, key=lambda r: r["_m"]["f1"])
                if _measurable else None)
        report["events"][key] = {
            "obs_uuid": obs_uuid, "frame": frame, "n_cands": len(rows),
            "oracle_route": (best or {}).get("route_id"),
            "oracle_metrics": (best or {}).get("_m"),
            "n_joint_pass_candidates": sum(
                1 for r in rows
                if r.get("_m") and r["_m"]["joint_pass"]),
            "n_unmeasured_candidates": sum(
                1 for r in rows if not r.get("_m")),
            "rules": ev_rows,
        }
    report["rules"] = {
        k: {"n_events": v["n"], "selected": v["sel"],
            "joint_pass": v["joint"],
            "mean_precision": round(float(np.mean(v["prec"])), 3)
            if v["prec"] else None,
            "mean_recall": round(float(np.mean(v["rec"])), 3)
            if v["rec"] else None}
        for k, v in rule_stats.items()}

    # parity: the shared selector must reproduce a saved run exactly
    if a.parity_run:
        run = Path(a.parity_run)
        sel_path = run / "selection.json"
        cand_path = run / "candidates.json"
        if sel_path.exists() and cand_path.exists():
            saved = {str(s.get("event")): s
                     for s in json.loads(sel_path.read_text())}
            saved_c = read_candidates(cand_path)
            by_ev: dict[str, list] = {}
            for c in saved_c:
                by_ev.setdefault(str(c["candidate_id"]).rsplit(":", 1)[0],
                                 []).append(c)
            n_match = n_tot = 0
            mism = []
            for k2, rows2 in by_ev.items():
                obs2 = str(rows2[0].get("owner_id", ""))
                if obs2 not in saved:
                    continue
                n_tot += 1
                dec2 = select_candidate(
                    rows2, SelectionPolicy(
                        rank=saved[obs2]["policy"]["rank"],
                        evidence_gate=saved[obs2]["policy"][
                            "evidence_gate"],
                        min_score=saved[obs2]["policy"]["min_score"],
                        min_evidence=saved[obs2]["policy"][
                            "min_evidence"]))
                want = saved[obs2].get("winner")
                got = (dec2.winner or {}).get("route_id")
                if want == got and bool(saved[obs2].get("refused")) == \
                        dec2.refused:
                    n_match += 1
                else:
                    mism.append({"event": obs2, "saved": want,
                                 "replayed": got})
            report["parity"] = {"n": n_tot, "matched": n_match,
                                "mismatches": mism}
            print(f"parity vs {run.name}: {n_match}/{n_tot} winners match")
        elif cand_path.exists():
            # rev9 WP-A.5: a run that predates selection.json still
            # carries its saved winner per event in the export's
            # `selected` flag, and its declared policy in the candidates'
            # `evidence.gate`. This is the replay the rev9 audit used.
            saved_c = read_candidates(cand_path)
            by_ev: dict[str, list] = {}
            for c in saved_c:
                by_ev.setdefault(str(c["candidate_id"]).rsplit(":", 1)[0],
                                 []).append(c)
            n_match = n_tot = 0
            mism = []
            for k2, rows2 in by_ev.items():
                obs2 = str(rows2[0].get("owner_id", ""))
                winners = [r for r in rows2 if r.get("selected")]
                if len(winners) != 1:
                    continue
                n_tot += 1
                gate = float((winners[0].get("evidence") or {}).get(
                    "gate", 0.0) or 0.0)
                dec2 = select_candidate(
                    rows2, SelectionPolicy(rank="sel", evidence_gate=gate,
                                           evidence_key="ev"))
                want = str(winners[0].get("route_id"))
                got = str((dec2.winner or {}).get("route_id") or "")
                if want == got:
                    n_match += 1
                else:
                    mism.append({"event": obs2, "saved": want,
                                 "replayed": got,
                                 "n_kept": dec2.n_kept,
                                 "n_candidates": dec2.n_candidates})
            report["parity"] = {"source": "selected-flags",
                                "n": n_tot, "matched": n_match,
                                "mismatches": mism}
            print(f"parity vs {run.name} (selected flags): "
                  f"{n_match}/{n_tot} winners match")
        else:
            report["parity"] = {"error": "run has neither selection.json "
                                         "nor candidates.json"}

    Path(a.out).write_text(json.dumps(report, indent=1, default=str))
    print(f"events with human routes: {n_events}")
    for k, v in report["rules"].items():
        print(f"  {k:<22} selected={v['selected']}/{v['n_events']}  "
              f"joint-pass={v['joint_pass']}  "
              f"prec={v['mean_precision']} rec={v['mean_recall']}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
