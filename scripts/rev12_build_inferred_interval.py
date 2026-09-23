"""rev12 P1.1: the INFERRED contiguous interval — real model calls.

Replaces the annotation-only interval demo. For every frame of the
banked 15-frame crossing interval (51030..51450, step 30) and every
relevant physical owner, this script:

1. generates routes with the DEPLOYMENT proposer (routes.py FRST
   fans + walkers) seeded from the previous accepted cap (or the
   owner's registered attachment at the first frame);
2. scores each route with the real model forward (front presence x
   cap distribution; route_p recorded);
3. associates hypotheses across frames with the ACTUAL association
   module (beam_search_per_owner) — alternatives retained;
4. treats the human anchor lanes as IDENTITY CONSTRAINTS and blinded
   truth at the three anchor frames only — never as measurements;
5. exports measurement semantics: full length ONLY for a
   root-to-current-cap route with declared completeness; partial
   crossing lanes stay partial_path with NULL full length; unresolved
   owners produce no measured record (withheld).

Corrections come from the crossing store (the same store the
annotation UI writes); a correction bump invalidates the pixel-inference
cache and regenerates the export. Acquisition cadence is NULL unless
explicitly supplied with source metadata — no hardcoded 3 s.

Outputs (in --out-dir): export_inferred.json, montage_inferred.png,
verification.txt, cache_key.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

X_DB = Path("/Users/joshjiang/Documents/TubeTracker-annotator-projects/"
            "rev10round/annotations.db")
REG = REPO / "runs/prototypes/v30/rev11own_verify"
SNAP = REPO / "runs/prototypes/v30/snap25_plus_rev11own"
CKPT = (REPO / "runs/prototypes/v30/rev11_front_tailjitter"
        / "best_front_ep15.pt")
V1 = REPO / "runs/prototypes/timesfm/tip_cnn_v1/best-point-heatmap-model.pt"
OUT = REPO / "runs/prototypes/v30/rev12_interval"
MOVIE = "/Users/joshjiang/Downloads/test1lowdensjoshua-28c-hz.mp4 .mp4"
F0, F1, STEP = 51030, 51450, 30
CS = 288
SEPARATION_PX = 12.0
ROOT_TOL_PX = 6.0
# rev13 W2: the accepted cap must be interior to the support with at
# least this much context beyond it, else the route's own length budget
# may have truncated the tube: the path is named PARTIAL.
MIN_TAIL_PX = 3.0


def _sha16(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _corrections_from_store(db: Path) -> dict:
    """Corrections = the crossing-store lane edits (the UI path).

    Returns {revision_total, units: {uuid: {revision, lanes}}, digest}.
    """
    import sqlite3
    if not db.exists():
        return {"revision_total": 0, "units": {}, "digest": "no-store"}
    con = sqlite3.connect(db)
    units = {}
    for u, data, rev in con.execute(
            "select uuid, data, revision from entities "
            "where kind='crossing'"):
        d = json.loads(data)
        units[u] = {"revision": int(rev),
                    "lanes": {k: [list(map(float, q)) for q in v]
                              for k, v in d.get("lanes", {}).items()}}
    con.close()
    digest = hashlib.sha256(json.dumps(
        units, sort_keys=True).encode()).hexdigest()[:16]
    return {"revision_total": sum(u["revision"] for u in units.values()),
            "units": units, "digest": digest}


def _render_and_verify(expf: Path, out: Path, frames, movie: str,
                       cache_key: str) -> None:
    """Montage + verification rendered from the EXPORT (cache-hit safe)."""
    import cv2
    from tubetracker.annotation_frames import FrameReader
    _exp = json.loads(expf.read_text())
    _rows_m = _exp["rows"]
    _anchors_m = _exp["anchor_agreement"]
    r2 = FrameReader(movie)
    try:
        tiles = []
        cols = {"own-ld-0001": (0, 220, 0), "own-ld-0002": (0, 160, 255)}
        for fi, f in enumerate(frames):
            img = r2.read(int(f)).frame[:, :, :3].copy()
            vis = img.copy()
            for row in _rows_m:
                if row["frame"] != int(f):
                    continue
                # rev13: a corrected row may carry no path (a bare tip
                # correction) — draw the point only, never crash the
                # montage on an empty path (caught by the W4 receipt).
                _raw = np.asarray(row.get("path") or [], float)
                col = cols.get(row["owner"], (200, 200, 200))
                solid = row["state"] == "model-inferred"
                if len(_raw) >= 2:
                    pts = _raw.astype(int)
                    if solid:
                        cv2.polylines(vis, [pts], False, col, 1)
                    else:
                        for q in range(0, len(pts) - 1, 2):
                            cv2.line(vis, tuple(pts[q]), tuple(pts[q + 1]),
                                     col, 1)
                    cv2.circle(vis, tuple(pts[-1]), 3, col, -1)
                else:
                    _tip = row.get("tip") or []
                    if len(_tip) == 2:
                        cv2.circle(vis, (int(_tip[0]), int(_tip[1])), 3,
                                   col, -1)
            for ar in _anchors_m:
                if ar.get("frame") != int(f) or not ar.get("human_tip"):
                    continue
                cv2.drawMarker(vis, (int(ar["human_tip"][0]),
                                     int(ar["human_tip"][1])),
                               (255, 255, 255), cv2.MARKER_CROSS, 8, 1)
            cv2.putText(vis, f"f{f}", (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)
            tiles.append(vis[200:640, 760:1240])
        half = len(tiles) // 2
        g1 = np.concatenate(tiles[:half], axis=1)
        g2 = np.concatenate(tiles[half:], axis=1)
        if g1.shape[1] != g2.shape[1]:
            pad = np.zeros((tiles[0].shape[0],
                            g2.shape[1] - g1.shape[1], 3),
                           dtype=g1.dtype)
            g1 = np.concatenate([g1, pad], axis=1)
        grid = np.concatenate([g1, g2], axis=0)
        cv2.imwrite(str(out / "montage_inferred.png"), grid)
    finally:
        r2.close()
    cov = _exp["coverage"]
    with (out / "verification.txt").open("w") as f:
        f.write("rev12 P1.1 inferred interval verification\n")
        f.write(f"cache_key {cache_key}\n")
        f.write(f"rows {cov['n_rows']}/{cov['n_rows_expected']} "
                f"(coverage {cov['coverage_frac']}); withheld "
                f"{cov['n_withheld']}; full-length {cov['n_full_length']}"
                f"; partial {cov['n_partial']}\n")
        for ar in _anchors_m:
            f.write(f"anchor {ar['frame']}/{ar.get('lane')}: "
                    f"{ar.get('endpoint_err_px')}\n")
    print(f"-> {out}/export_inferred.json, montage_inferred.png, "
          f"verification.txt")


def cache_keys(*, frames, owners, ckpt_sha, corrections_digest,
               ui_corrections_digest, acquisition_cadence_s,
               cadence_source, calibration="") -> tuple[str, str]:
    """rev13 W4: per-stage cache keys.

    The pixel-inference key covers everything inference depends on;
    the measurement/export key adds acquisition/calibration
    metadata, so a cadence-only change can never reuse a stale
    export (time/rate fields must regenerate).
    """
    _pix = hashlib.sha256(json.dumps({
        "frames": frames, "owners": owners, "ckpt": ckpt_sha,
        "corrections": corrections_digest,
        "ui_corrections": ui_corrections_digest,
        "sep": SEPARATION_PX,
        "root_tol": ROOT_TOL_PX, "v": 3},
        sort_keys=True).encode()).hexdigest()[:16]
    _meas = hashlib.sha256(json.dumps({
        "pixel": _pix,
        "acquisition_cadence_s": acquisition_cadence_s,
        "cadence_source": cadence_source,
        "calibration": calibration, "v": 3},
        sort_keys=True).encode()).hexdigest()[:16]
    return _pix, _meas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=str(SNAP))
    ap.add_argument("--checkpoint", default=str(CKPT))
    ap.add_argument("--v1-weights", default=str(V1))
    ap.add_argument("--owners",
                    default="own-ld-0001,own-ld-0002,own-ld-0003",
                    help="rev13 W2.5: all THREE neighboring crossing "
                         "owners participate (the old default omitted "
                         "the third)")
    ap.add_argument("--frames", default=f"{F0}:{F1}:{STEP}")
    ap.add_argument("--movie", default=MOVIE)
    ap.add_argument("--crossing-store", default=str(X_DB))
    ap.add_argument("--corrections-file", default="",
                    help="JSON {owner: {frame: {tip, actor, revision}}} "
                         "— the UI correction bridge; a revision bump "
                         "invalidates the inference cache")
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--acquisition-cadence-s", type=float, default=None)
    ap.add_argument("--cadence-source", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-owners", type=int, default=7)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if a.acquisition_cadence_s is not None and not a.cadence_source:
        print("--acquisition-cadence-s requires --cadence-source "
              "(no unprovenanced timing)", file=sys.stderr)
        return 2

    import torch
    from prototypes.v30_video_apex import routes as R
    from prototypes.v30_video_apex.inference import build_typed_prompt
    from prototypes.v30_video_apex.model_factory import (
        build_model_from_checkpoint)
    from prototypes.v30_video_apex.association import (
        beam_search_per_owner, TrackState)
    from prototypes.v30_video_apex.targets import load_clip_pixels
    from tubetracker.annotation_frames import FrameReader

    f0, f1, step = (int(v) for v in a.frames.split(":"))
    frames = list(range(f0, f1 + 1, step))
    owners_all = {o["id"]: o for o in json.loads(
        (REG / "owners.json").read_text())["owners"]}
    want = ([w.strip() for w in a.owners.split(",") if w.strip()]
            or list(owners_all))
    owners = [owners_all[w] for w in want if w in owners_all]
    links = json.loads((REG / "crossing_owner_links.json").read_text())
    lane_owner = {(int(lk["frame"]), lk["lane"]): lk.get("owner")
                  for lk in links}
    corr = _corrections_from_store(Path(a.crossing_store))
    # UI correction bridge: corrections JSON entries override inference
    # at their (owner, frame) and participate in the cache key.
    corr_rows: dict = {}
    corr_digest = "none"
    if a.corrections_file and Path(a.corrections_file).exists():
        corr_rows = json.loads(Path(a.corrections_file).read_text())
        # rev13 W4: corrections are TYPED. Only an explicit `tip` field
        # creates a corrected row; a lane or path endpoint is never
        # silently promoted to tip truth, and an entry without a tip
        # is rejected loudly (it is not a correction this runner can
        # apply).
        _bad = []
        for _o, _per in (corr_rows or {}).items():
            if not isinstance(_per, dict):
                _bad.append(_o)
                continue
            for _f2, _cr in _per.items():
                _typed = isinstance(_cr, dict) and (
                    "tip" in _cr or "state" in _cr)
                if not _typed:
                    _bad.append(f"{_o}|{_f2}")
        if _bad:
            print(f"corrections: refusing {len(_bad)} entr(ies) with "
                  f"no typed `tip` field: {_bad[:5]}", file=sys.stderr)
            return 2
        corr_digest = hashlib.sha256(json.dumps(
            corr_rows, sort_keys=True).encode()).hexdigest()[:16]
    ckpt_sha = _sha16(Path(a.checkpoint))
    # rev13 W4: the cache is keyed per STAGE. The pixel-inference key
    # covers everything the inference depends on (movie frames, owners,
    # model, corrections, proposal geometry). The measurement/export
    # key adds the acquisition/calibration metadata: a cadence-only
    # change must regenerate time/rate fields even when pixel inference
    # would be reusable — it can never hit a stale export.
    pixel_key, measure_key = cache_keys(
        frames=frames, owners=want, ckpt_sha=ckpt_sha,
        corrections_digest=corr["digest"],
        ui_corrections_digest=corr_digest,
        acquisition_cadence_s=a.acquisition_cadence_s,
        cadence_source=a.cadence_source,
        calibration=getattr(a, "calibration", ""))
    cache_key = measure_key  # the export's identity is the full key
    keyf = out / "cache_key.json"
    expf = out / "export_inferred.json"
    if keyf.exists() and expf.exists() and not a.force:
        prev = json.loads(keyf.read_text())
        if prev.get("measure_key", prev.get("key")) == measure_key:
            print(f"cache HIT ({measure_key}); export reused")
            _render_and_verify(expf, out, frames, a.movie, cache_key)
            return 0
        if prev.get("pixel_key") == pixel_key:
            print(f"cache MISS ({measure_key}) — MEASUREMENT-ONLY "
                  f"change (cadence/calibration): pixel inference "
                  f"({pixel_key}) was reusable, but time/rate fields "
                  f"must regenerate; rerunning the measurement stage")
    print(f"cache MISS ({measure_key}) — running inference")

    # rev13 W1: the metadata-driven STRICT factory. The declared
    # architecture — including the presence head this checkpoint
    # trained — is rebuilt from the checkpoint's own config and the
    # weights are loaded strictly (missing/unexpected tensors refuse).
    # Evaluation must not add an untrained random head: the audit's
    # counterexample had a freshly-initialized presence head scoring
    # every route.
    from prototypes.v30_video_apex.model_factory import (
        build_model_from_checkpoint)
    model, _minfo = build_model_from_checkpoint(a.checkpoint)
    print(f"checkpoint {a.checkpoint}: strict metadata load, "
          f"semantics={_minfo['semantics']!r}, schema="
          f"{_minfo['model_schema']!r}, post-load parameter hash "
          f"{_minfo['parameter_hash']}, kwargs={_minfo['model_kwargs']}")
    model.eval()
    operative = {
        "checkpoint": str(a.checkpoint),
        "checkpoint_sha16": _sha16(Path(a.checkpoint)),
        "parameter_hash": _minfo["parameter_hash"],
        "model_schema": _minfo["model_schema"],
        "activation": _minfo["activation"],
        "preprocessing": _minfo["preprocessing"],
        "config_hash": _minfo["config_hash"],
        "model_kwargs": _minfo["model_kwargs"],
    }
    v1, dev = R.load_v1(a.v1_weights)

    # anchor lanes (identity constraints + blinded truth AT anchors)
    anchors: dict[int, dict] = {}
    for u, unit in corr["units"].items():
        f = int(u.rsplit("-", 1)[-1]) if "-" in u else -1
        # store uuid convention: cross-rev10x-000 carries its own
        # source_frame in the data; recover it below when available
    import sqlite3
    con = sqlite3.connect(Path(a.crossing_store))
    for u, data in con.execute(
            "select uuid, data from entities where kind='crossing'"):
        d = json.loads(data)
        anchors[int(d["source_frame"])] = {
            lab: [list(map(float, q)) for q in pts]
            for lab, pts in d.get("lanes", {}).items()}
    con.close()
    print(f"anchor frames: {sorted(anchors)}")

    reader = FrameReader(a.movie)
    try:
        gray_cache: dict[int, np.ndarray] = {}
        clip_cache: dict = {}

        def _gray(frame):
            if frame not in gray_cache:
                fr = reader.read(int(frame)).frame
                gray_cache[frame] = np.asarray(
                    fr[:, :, 0] if fr.ndim == 3 else fr)
            return gray_cache[frame]

        def _clip(frame, crop):
            key = (int(frame), tuple(int(v) for v in crop))
            if key not in clip_cache:
                # the SAME temporal construction the trainer/evaluator
                # use (QUERY_OFFSETS around the source frame)
                from prototypes.v30_video_apex.dataset import (
                    QUERY_OFFSETS)
                fr = tuple(max(0, int(frame) + o) for o in QUERY_OFFSETS)
                miss = tuple(1 if int(frame) + o < 0 else 0
                             for o in QUERY_OFFSETS)
                clip_cache[key] = load_clip_pixels(
                    reader, fr, (crop[0], crop[1], CS, CS), miss)
            return clip_cache[key]

        # per-owner crop: centered on the registered grain (fixed over
        # the interval so pixel evidence is comparable across frames)
        owner_crop = {}
        for o in owners:
            gx, gy = o["grain_native"]
            H, W = _gray(frames[0]).shape
            ox = int(min(max(gx - CS / 2, 0), max(0, W - CS)))
            oy = int(min(max(gy - CS / 2, 0), max(0, H - CS)))
            owner_crop[o["id"]] = (ox, oy)

        cands_by_owner: dict[str, list[list[dict]]] = {
            o["id"]: [] for o in owners}
        # rev13 W2.6: DECLARED fixed-root policy. The candidate pass is
        # per-frame and independent; proposals anchor at the registered
        # attachment. A previous accepted cap is never used as an
        # undeclared anchor (the old `accepted_prev` was always empty —
        # a claim of previous-state anchoring the code never made).
        accepted_prev: dict[str, tuple] = {}  # retained only for the
        # export note; never consulted for anchoring.
        frame_rows = []
        for fi, f in enumerate(frames):
            gray = _gray(f)
            for o in owners:
                oid = o["id"]
                crop = owner_crop[oid]
                ox, oy = crop
                clip_np = _clip(f, crop)
                clip = torch.from_numpy(clip_np).unsqueeze(0).unsqueeze(2)
                prompt, _ = build_typed_prompt(
                    oid, crop_wh=(CS, CS), crop_origin=(ox, oy),
                    human_grain_xy=tuple(o["grain_native"]),
                    human_grain_radius_px=13.0)
                # declared fixed-root policy (rev13 W2.6): the seed is
                # the registered attachment — not an undeclared
                # previous-state anchor.
                seed = tuple(o.get("attachment_native")
                             or o["grain_native"])
                # rev13 W2.6: the ONE proposal callable — shared with
                # the proposal builder/fit and the movie runner. The
                # family (FRST fans, curved walkers, v1 peaks, model
                # body walks) and the evidence actually used are
                # recorded by the callable itself.
                from prototypes.v30_video_apex.proposals import (
                    generate_route_hypotheses)
                body_np = None
                try:
                    with torch.no_grad():
                        pb = model.forward(clip, prompt)
                    body_np = torch.sigmoid(
                        pb.body[0, 0]).detach().numpy()
                except Exception:  # noqa: BLE001
                    body_np = None
                props, _ginfo = generate_route_hypotheses(
                    gray, seed, v1=v1, dev=dev, body_prob=body_np,
                    body_prob_origin=(ox, oy))
                root_xy = tuple(_ginfo["root_xy"])
                root_source = _ginfo["root_source"]
                scored = []
                for pr in props:
                    pts = np.asarray(pr["polyline"], float)
                    if len(pts) < 2:
                        continue
                    rc = (pts - np.array([ox, oy])).tolist()
                    with torch.no_grad():
                        p = model.forward(clip, prompt, route_xy=rc)
                        # rev13 W1: the presence factor comes from the
                        # checkpoint's DECLARED architecture. When the
                        # declared model has no presence head, a
                        # documented neutral factor of one is used —
                        # never an untrained random head (the audit's
                        # counterexample scored every route with one).
                        if getattr(p, "front_present_logit",
                                   None) is not None:
                            pres = float(torch.sigmoid(
                                p.front_present_logit.reshape(-1))[0])
                            pres_source = "presence-head"
                        else:
                            pres = 1.0
                            pres_source = ("neutral-1.0 (checkpoint "
                                           "declares no presence head)")
                        q = torch.softmax(p.front_logits, -1)[0].numpy()
                        sg = p.front_s.numpy()
                        rp = (float(torch.sigmoid(
                            p.route_logit.reshape(-1))[0])
                            if p.route_logit is not None else 1.0)
                    sc = pres * q
                    idx = [i for i in range(1, len(sc) - 1)
                           if sc[i] >= sc[i - 1] and sc[i] >= sc[i + 1]]
                    idx.sort(key=lambda i: -sc[i])
                    if not idx:
                        continue
                    i = idx[0]
                    s_i = float(sg[i])
                    seg = np.diff(pts, axis=0)
                    cum = np.concatenate([[0.0], np.cumsum(np.hypot(
                        seg[:, 0], seg[:, 1]))])
                    xy = np.array([np.interp(s_i, cum, pts[:, 0]),
                                   np.interp(s_i, cum, pts[:, 1])])
                    # rev13 W2: support vs CURRENT geometry are
                    # separate. The current path is the support
                    # truncated at the accepted front arclength with
                    # its final segment interpolated ONCE; its last
                    # point IS the reported tip and its arclength IS
                    # the full length when the row qualifies.
                    from prototypes.v30_video_apex.ownership import (
                        current_path_from_support)
                    _cur, _curlen = current_path_from_support(pts, s_i)
                    scored.append({
                        "candidate_id": f"{oid}|{f}|"
                                        f"{pr.get('route_id', 'r')}",
                        "route_id": str(pr.get("route_id", "r")),
                        "x_native": float(_cur[-1][0]),
                        "y_native": float(_cur[-1][1]),
                        "tip_score": float(sc[i]),
                        "front_present": pres, "route_p": rp,
                        "pres_source": pres_source,
                        "s_cap_px": s_i,
                        "current_len_px": _curlen,
                        "support_len_px": float(cum[-1]),
                        "path_complete": bool(
                            float(cum[-1]) - s_i >= MIN_TAIL_PX),
                        "root_source": root_source,
                        "path": _cur,
                        "support_xy": pts.tolist(),
                        "path_start": _cur[0],
                        "observation": "observed",
                        "ambiguous": bool(pres < 0.5),
                    })
                # separated top-2 (alternatives retained)
                scored.sort(key=lambda c: -c["tip_score"])
                kept = []
                for c in scored:
                    if all(np.hypot(c["x_native"] - k["x_native"],
                                    c["y_native"] - k["y_native"])
                           >= SEPARATION_PX for k in kept):
                        kept.append(c)
                    if len(kept) >= 2:
                        break
                # rev13 W4: annotation revisions are consumed as
                # inference CONSTRAINTS before association. The
                # human tip enters this frame's candidate pool
                # with explicit human provenance (never model
                # scores), so the joint selection and continuity
                # see it instead of it being applied afterwards.
                _cr_c = (corr_rows.get(oid, {}) or {}).get(
                    str(int(f)))
                if _cr_c is not None and _cr_c.get("tip") is not None:
                    _hx = [float(_cr_c["tip"][0]),
                           float(_cr_c["tip"][1])]
                    _hpath = _cr_c.get("path") or [list(seed),
                                                    _hx]
                    _arr = np.asarray(_hpath, float)
                    _al = (float(np.hypot(*(
                        np.diff(_arr, axis=0).T)).sum())
                        if len(_arr) > 1 else 0.0)
                    kept.insert(0, {
                        "candidate_id":
                            f"{oid}|{f}|human-correction",
                        "route_id": "human-correction",
                        "x_native": _hx[0], "y_native": _hx[1],
                        "tip_score": 1.0, "front_present": 1.0,
                        "route_p": 1.0, "pres_source": "human",
                        "s_cap_px": _al, "current_len_px": _al,
                        "support_len_px": _al,
                        "path_complete": False,
                        "root_source": "human",
                        "path": _hpath, "support_xy": _hpath,
                        "provenance": "human-correction",
                        "correction": {
                            "actor": _cr_c.get("actor", ""),
                            "revision": _cr_c.get("revision")}})
                cands_by_owner[oid].append(kept)
                frame_rows.append({
                    "frame": int(f), "analysis_index": fi, "owner": oid,
                    "n_candidates": len(kept),
                    "candidates": [
                        {k: c[k] for k in ("candidate_id", "route_id",
                                           "x_native", "y_native",
                                           "tip_score", "front_present",
                                           "route_p", "s_cap_px",
                                           "current_len_px",
                                           "support_len_px",
                                           "path_complete",
                                           "root_source")}
                        for c in kept]})
            print(f"frame {f}: " + " ".join(
                f"{r['owner']}×{r['n_candidates']}" for r in frame_rows
                if r["frame"] == f))

        # ---- association (the actual module) -------------------------
        assoc: dict[str, list] = {}
        for o in owners:
            oid = o["id"]
            hyps = beam_search_per_owner(cands_by_owner[oid], oid,
                                         beam_width=4)
            assoc[oid] = hyps

        # accepted cap per owner per frame from the best hypothesis
        # lineage: the beam returns end states; rebuild the accepted
        # chain from the best lineage ids
        accepted: dict[tuple, dict] = {}
        for o in owners:
            oid = o["id"]
            best = assoc[oid][0] if assoc[oid] else None
            if best is None:
                continue
            ids = set(best.lineage)
            for fi, f in enumerate(frames):
                cands = [c for c in cands_by_owner[oid][fi]
                         if c["candidate_id"] in ids]
                if cands:
                    accepted[(oid, int(f))] = cands[0]

        # ---- joint selection over owner x path/cap x TIME (rev13 W2) --
        # Replaces per-frame greedy pairwise demotion: exact enumeration
        # over the joint state space with an explicit unresolved option,
        # the cross-owner same-physical-cap constraint, and temporal
        # consistency from actual source frames. Alternatives, score
        # components, re-acquisition flags and future-assisted identity
        # are recorded per row.
        from prototypes.v30_video_apex.ownership import (
            RouteHypothesis, joint_select_over_time)
        _jmeta: dict[tuple, dict] = {}
        cands_j: dict[str, dict[int, list]] = {}
        for o in owners:
            oid = o["id"]
            attach = tuple(o.get("attachment_native")
                           or o["grain_native"])
            per: dict[int, list] = {}
            for fi, f in enumerate(frames):
                hs = []
                for c in cands_by_owner[oid][fi]:
                    hs.append(RouteHypothesis(
                        owner_id=oid, frame=int(f),
                        route_id=c["route_id"], attachment=attach,
                        support_xy=c["support_xy"],
                        current_prefix_len_px=c["s_cap_px"],
                        local_cap_score=c["tip_score"],
                        whole_route_score=c["route_p"],
                        current_path_xy=c["path"],
                        tip_xy=tuple(c["path"][-1]),
                        length_px=c["current_len_px"],
                        cap_candidate_id=c["candidate_id"],
                        cap_clear=bool(c["tip_score"] >= 0.5
                                       and c["front_present"] >= 0.5)))
                per[int(f)] = hs
            cands_j[oid] = per
        jsel = joint_select_over_time(
            cands_j, [int(f) for f in frames])
        n_joint_unresolved = 0
        for o in owners:
            oid = o["id"]
            for fi, f in enumerate(frames):
                row = (jsel["rows"].get(oid) or {}).get(int(f))
                if row is None or row["state"] != "present":
                    if (oid, int(f)) in accepted:
                        n_joint_unresolved += 1
                    accepted.pop((oid, int(f)), None)
                    continue
                cands = cands_by_owner[oid][fi]
                idx = int(row.get("cand_index", -1))
                if not (0 <= idx < len(cands)):
                    accepted.pop((oid, int(f)), None)
                    continue
                accepted[(oid, int(f))] = cands[idx]
                # future-assisted identity: the chosen hypothesis was
                # NOT this frame's locally-best candidate — the choice
                # was made with temporal/future evidence. Stored
                # separately from the current visible extent.
                _best = max(range(len(cands)),
                            key=lambda k: cands[k]["tip_score"]
                            + cands[k]["route_p"]) if cands else -1
                _jmeta[(oid, int(f))] = {
                    "score_components": row["score_components"],
                    "reacquisition": row["reacquisition"],
                    "cap_step_px": row["cap_step_px"],
                    "future_assisted": bool(idx != _best),
                    "alternatives": (jsel["alternatives_by_row"]
                                     .get(oid, {}).get(int(f), []))}
        print(f"joint selection: total {jsel['total']:.2f}, "
              f"{jsel['n_states_visited']} states visited, "
              f"{n_joint_unresolved} accepted row(s) became unresolved")

        # ---- rows with measurement semantics -------------------------
        rows = []
        withheld = []
        n_corrected = 0
        for o in owners:
            oid = o["id"]
            attach = np.asarray(o.get("attachment_native")
                                or o["grain_native"], float)
            for fi, f in enumerate(frames):
                # UI correction: overrides inference at this (owner,
                # frame); continuity downstream still uses inference
                _cr = (corr_rows.get(oid, {}) or {}).get(str(int(f)))
                if _cr is not None and _cr.get("tip") is None:
                    # rev13 W4: a scoped absence/visibility
                    # correction removes an inappropriate owned
                    # path — the row is withheld with the human
                    # reason, never turned into a tip.
                    withheld.append({
                        "owner": oid, "frame": int(f),
                        "reason": "human-scoped correction: "
                                  + str(_cr.get("state",
                                                "not_visible"))
                                  + " (owned path removed)",
                        "correction": {
                            "actor": _cr.get("actor", ""),
                            "revision": _cr.get("revision")},
                        "as_inference_constraint": True})
                    continue
                if _cr is not None:
                    n_corrected += 1
                    rows.append({
                        "owner": oid, "frame": int(f),
                        "analysis_index": fi, "source_frame": int(f),
                        "state": "human-corrected",
                        "tip": [float(_cr["tip"][0]),
                                float(_cr["tip"][1])],
                        "path": _cr.get("path", []),
                        "path_source": "corrected",
                        # rev13 W4: typed correction fields — lane,
                        # current tip, path completeness and ownership
                        # are separate fields; a lane is never promoted
                        # to tip truth and ownership is explicit.
                        "correction_type": "current_tip",
                        "lane": _cr.get("lane"),
                        "path_completeness": str(
                            _cr.get("path_completeness", "unknown")),
                        "ownership": str(_cr.get("owner", oid)),
                        "length_px": None,
                        "length_kind": "partial",
                        "support_len_px": None,
                        "root_link": "root-link hypothesis",
                        "root_ok": False,
                        "front_present": None, "route_p": None,
                        "correction": {"actor": _cr.get("actor", ""),
                                       "revision": _cr.get("revision")},
                        "completeness": {
                            "root_within_6px": False,
                            "cap_is_accepted_current": False,
                            "declared": "human measurement "
                                        "override, consumed as an "
                                        "inference constraint "
                                        "before association; it "
                                        "does not masquerade as "
                                        "model propagation; "
                                        "completeness and "
                                        "ownership are typed "
                                        "fields"},
                        "as_inference_constraint": True,
                        "model_candidate_for_frame": (
                            (accepted.get((oid, int(f))) or {})
                            .get("candidate_id")),
                        "correction_changed_outcome": bool(
                            (accepted.get((oid, int(f))) or {})
                            .get("candidate_id")
                            != f"{oid}|{f}|human-correction"),
                    })
                    continue
                acc = accepted.get((oid, int(f)))
                if acc is None:
                    withheld.append({"owner": oid, "frame": int(f),
                                     "reason": "no accepted hypothesis "
                                               "(association lost / "
                                               "unresolved)"})
                    continue
                pts = np.asarray(acc["path"], float)
                root_ok = (np.hypot(*(pts[0] - attach)) <= ROOT_TOL_PX)
                # rev13 W2: full length requires a verified grain
                # exit/root link, a COMPLETE current owned path and the
                # accepted current cap. Root proximity alone cannot
                # certify completeness; partial lengths are named
                # separately.
                root_verified = bool(
                    root_ok and acc["root_source"] in (
                        "human", "registry", "frst"))
                path_complete = bool(acc.get("path_complete"))
                full = bool(root_verified and path_complete)
                from prototypes.v30_video_apex.ownership import (
                    geometry_consistent)
                geom = geometry_consistent(
                    acc["path"], (acc["x_native"], acc["y_native"]),
                    acc["current_len_px"])
                rows.append({
                    "owner": oid, "frame": int(f), "analysis_index": fi,
                    "source_frame": int(f),
                    "state": "model-inferred",
                    "tip": [acc["path"][-1][0], acc["path"][-1][1]],
                    "path": acc["path"],
                    "support_xy": acc["support_xy"],
                    "path_source": "inferred",
                    "length_px": (acc["current_len_px"] if full
                                  else None),
                    "length_kind": "full" if full else "partial",
                    "current_len_px": acc["current_len_px"],
                    "support_len_px": acc["support_len_px"],
                    "root_link": "root-link hypothesis",
                    "root_ok": bool(root_ok),
                    "root_verified": root_verified,
                    "path_complete": path_complete,
                    "geometry_check": geom,
                    "front_present": acc["front_present"],
                    "route_p": acc["route_p"],
                    "joint": _jmeta.get((oid, int(f))),
                    "future_assisted": bool(
                        (_jmeta.get((oid, int(f))) or {}).get(
                            "future_assisted", False)),
                    "completeness": {
                        "root_within_6px": bool(root_ok),
                        "root_verified": root_verified,
                        "path_complete": path_complete,
                        "cap_is_accepted_current": True,
                        "declared": "full only when the route starts at "
                                    "a VERIFIED root link, the current "
                                    "owned path is complete (cap "
                                    "interior to the support) and ends "
                                    "at the accepted current cap; "
                                    "length_px = arclength(current "
                                    "path), support length is named "
                                    "separately"},
                })

        # ---- anchor agreement (blinded truth AT anchors only) --------
        anchor_rows = []
        for f in sorted(anchors):
            for lab, pts in anchors[f].items():
                oid = lane_owner.get((f, lab))
                if oid is None or oid not in {o["id"] for o in owners}:
                    anchor_rows.append({
                        "frame": f, "lane": lab, "owner": oid,
                        "note": "lane end unresolved in the registry "
                                "ray test — identity withheld"})
                    continue
                tip_h = np.asarray(pts[-1], float)
                acc = accepted.get((oid, f))
                err = (float(np.hypot(*(np.asarray(
                    [acc["x_native"], acc["y_native"]], float) - tip_h)))
                    if acc else None)
                anchor_rows.append({
                    "frame": f, "lane": lab, "owner": oid,
                    "human_tip": [float(tip_h[0]), float(tip_h[1])],
                    "inferred_tip": ([acc["x_native"], acc["y_native"]]
                                     if acc else None),
                    "endpoint_err_px": err,
                    "within_5px": bool(err is not None and err <= 5.0)})

        # ---- cadence / time provenance -------------------------------
        time_info = {
            "acquisition_cadence_s": a.acquisition_cadence_s,
            "cadence_source": (a.cadence_source
                               if a.acquisition_cadence_s is not None
                               else ""),
            "time_fields": ("derived from the supplied cadence"
                            if a.acquisition_cadence_s is not None
                            else "NULL — no acquisition provenance; "
                                 "frame-index quantities only"),
            "source_frame_map": {int(f): fi
                                 for fi, f in enumerate(frames)},
        }

        export = {
            "interval": [int(f0), int(f1)], "step": int(step),
            "frames": [int(f) for f in frames],
            "owners": [o["id"] for o in owners],
            "checkpoint": a.checkpoint, "checkpoint_sha16": ckpt_sha,
            # rev13 W1: the complete operative state after load —
            # post-load parameter hash + declared schema/activation/
            # preprocessing + the config hash the model was built from.
            "operative_state": operative,
            "cache_key": cache_key,
            "corrections": {"digest": corr["digest"],
                            "revision_total": corr["revision_total"],
                            "ui_corrections_digest": corr_digest,
                            "n_ui_corrected_rows": n_corrected},
            "measurement_semantics": {
                "states": ["model-inferred", "human-corrected",
                           "context-only", "withheld"],
                "full_length_rule": "length_px is exported ONLY when the "
                                    "route starts at a VERIFIED root "
                                    "link, the current owned path is "
                                    "complete (cap interior to the "
                                    "support) and ends at the accepted "
                                    "current cap; length_px = "
                                    "arclength(current path); support "
                                    "length is named separately as "
                                    "support_len_px",
                "geometry_contract": "current_path[-1] == tip and "
                                     "arclength(current_path) == "
                                     "length_px within floating-point "
                                     "tolerance (per-row geometry_check)",
                "owner_unresolved_rule": "no measured record; the row "
                                         "is withheld with a reason",
                "context_rendering": "dashed, with its source frame "
                                     "time; context never becomes "
                                     "measurement"},
            "time": time_info,
            "rows": rows, "withheld": withheld,
            "anchor_agreement": anchor_rows,
            "joint_selection": {
                "total": jsel["total"],
                "n_states_visited": jsel["n_states_visited"],
                "n_became_unresolved": n_joint_unresolved,
                "temporal_semantics": jsel.get("temporal_semantics"),
                "declared": "exact joint selection over owner x "
                            "path/cap x time with an explicit "
                            "unresolved option; per-row components, "
                            "re-acquisition and future-assisted flags "
                            "are recorded on the rows"},
            "coverage": {
                "n_rows_expected": len(frames) * len(owners),
                "n_rows": len(rows),
                "n_withheld": len(withheld),
                "coverage_frac": round(
                    len(rows) / max(1, len(frames) * len(owners)), 4),
                "n_full_length": sum(1 for r in rows
                                     if r["length_kind"] == "full"),
                "n_partial": sum(1 for r in rows
                                 if r["length_kind"] == "partial"),
            },
            "note": "REAL model inference on every frame (front "
                    "presence x cap distribution; association via "
                    "beam_search_per_owner). Human anchors constrain "
                    "identity and are blinded truth AT anchor frames "
                    "only; they never substitute for inference on "
                    "other frames. The six legacy anchor lanes are NOT "
                    "counted as six whole-tube measurements.",
        }
        expf.write_text(json.dumps(export, indent=1) + "\n")
        keyf.write_text(json.dumps({
            "pixel_key": pixel_key, "measure_key": measure_key,
            "key": cache_key}) + "\n")
    finally:
        reader.close()

    # ---- montage + verification (from the export; cache-hit safe) ----
    _render_and_verify(expf, out, frames, a.movie, cache_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
