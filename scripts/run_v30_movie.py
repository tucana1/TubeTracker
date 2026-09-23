"""v30 automatic movie workflow (rev5 #4): pixels -> routes -> front -> measurements.

Per event (movie, query frame, gaze anchor):
  1. auto_root: nearest FRST grain ('frst') or anchor fallback ('assisted').
  2. Frozen v1 heat -> apex peaks (rim-excluded) -> propose_routes.
  3. Trained front head scores every fitting proposal; top q_max wins.
  4. Joint rule: two owners taking tips within 6px -> higher score keeps
     it, the loser falls back (recorded, never averaged).
  5. adapter_v29 exports ONE curve truncated at the selected front:
     tip + path + length agree by construction; tip-only outputs
     withhold length instead of inventing it.

--review applies a human correction file and replays the event; those
rows are marked assisted and scored separately from automatic rows.
Gold comes from the snapshot and is used ONLY in scores.json, never in
proposals or selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from prototypes.v30_video_apex.eval_route_evidence import (  # noqa: E402
    resample as _resample_route,
    wall_energy as _wall_energy,
)


def resample(p, step: float = 4.0):  # alias used by the evidence gate
    return _resample_route(p, step=step)


def wall_energy(g, p):
    return _wall_energy(g, p)


def deployment_prompt(owner_id, ox: int, oy: int, cs: int,
                      grain_c, radius_px):
    """The runtime's owner prompt for a candidate crop — one call site.

    rev11: routes through the typed constructor; the disc comes from the
    DETECTOR's center and radius (never a gaze/focus point).
    """
    from prototypes.v30_video_apex.inference import build_typed_prompt
    return build_typed_prompt(
        str(owner_id), crop_wh=(int(cs), int(cs)),
        crop_origin=(int(ox), int(oy)),
        detected_grain_xy=(float(grain_c[0]), float(grain_c[1])),
        detected_radius_px=float(radius_px))


CONFLICT_PX = 6.0
RIM_EXCL_PX = 26.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--front-checkpoint", required=True)
    ap.add_argument("--v1-weights", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--movie", required=True,
                    help="key=path of the movie to run")
    ap.add_argument("--event-ref", action="append", default=[],
                    help="snapshot obs_uuid substring (repeatable)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--review", default="",
                    help="json {obs_substr: {tip_xy: [x, y]}} corrections")
    ap.add_argument("--crop", type=int, default=288)
    ap.add_argument("--select-rank", default="sel",
                    help="which candidate field the selection policy "
                         "ranks by (sel = q_max x route_p, the shipped "
                         "behaviour; front_peak / front_peak_prob = the "
                         "learned current-cap evidence). Declared per "
                         "run so rankers can be compared at equal data.")
    ap.add_argument("--body-ablation", default="none",
                    choices=["none", "zero", "swap"],
                    help="none = use pred.body as-is; zero = remove it; "
                         "swap = replace its spatial field with a constant "
                         "(a body evidence ablation, not a renormalization)")
    ap.add_argument("--body-proposals", default="none",
                    choices=["none", "walk"],
                    help="walk = add root-connected routes traced from "
                         "the owner's own body probability map (rev11 "
                         "item 6), retaining crossing alternatives; "
                         "none = the historic families only")
    ap.add_argument("--acquisition-cadence-s", type=float, default=0.0,
                    help="seconds per source frame, as DECLARED by the "
                         "acquisition. 0 = unknown, and then records carry "
                         "source_time_s: null rather than a guess "
                         "(playback FPS does not establish growth rate).")
    ap.add_argument("--allow-legacy-migration", action="store_true",
                    help="explicitly accept a checkpoint that predates "
                         "the complete-config/semantics contract: it is "
                         "replayed under the CLI-declared architecture "
                         "(--base, multiscale=False) and the migration "
                         "is recorded in the report. Only legacy-"
                         "declaration failures are eligible; every other "
                         "contract failure is final (rev11).")
    ap.add_argument("--allow-route-head-gap", action="store_true",
                    help="declared exception for a checkpoint whose "
                         "route head is FULLY absent (pre-route-head): "
                         "zero only that absent head. A present trained "
                         "head is always preserved (rev11).")
    ap.add_argument("--stop-after-load", action="store_true",
                    help="diagnostic: load the front model, write "
                         "load_provenance.json and exit before any "
                         "image inference (exercises the real loader "
                         "path of this entry point)")
    ap.add_argument("--base", type=int, default=16,
                    help="front-model width (must match checkpoint)")
    ap.add_argument("--min-q", type=float, default=0.0,
                    help="withhold events whose selection score "
                    "(route_p x q_max) is below this (0 = disabled)")
    ap.add_argument("--select-evidence-gate", type=float, default=0.0,
                    help="relative image-evidence gate before ranking: "
                    "a candidate whose frame-normalized wall support "
                    "along its own route is below FLOAT x the event's "
                    "best is ineligible to win (0 = off, current "
                    "behavior; 0.6 per eval_route_evidence.py)")
    ap.add_argument("--select-min-evidence", type=float, default=0.0,
                    help="absolute floor on the event's best evidence: "
                    "below it the event is REFUSED (a null decision) "
                    "instead of forcing a winner")
    a = ap.parse_args(argv)

    out = Path(a.out_dir)
    if out.exists():
        print(f"refusing to overwrite {out}")
        return 1
    out.mkdir(parents=True)
    movie_key, _, movie_path = a.movie.partition("=")
    if not movie_key or not movie_path:
        print("need --movie key=path")
        return 1

    import numpy as np
    import torch

    from prototypes.v30_video_apex import routes as R
    from prototypes.v30_video_apex.adapter_v29 import export_tip_path_length
    from prototypes.v30_video_apex.association import beam_search_per_owner
    from prototypes.v30_video_apex.dataset import QUERY_OFFSETS
    from prototypes.v30_video_apex.evaluate import (
        confidence_gate, score_measurements, score_probes,
        substitute_oracle_routes)
    from prototypes.v30_video_apex.candidates import (
        export_candidates, normalize_candidate)
    from prototypes.v30_video_apex.model import build_model, \
        build_owner_prompt
    from prototypes.v30_video_apex.targets import (
        load_clip_pixels, project_to_arclength, resample_polyline)
    from prototypes.v30_video_apex.train import load_checkpoint
    from tubetracker.annotation_frames import FrameReader

    snap = Path(a.snapshot)
    obs = json.loads((snap / "observations.json").read_text())
    review = json.loads(Path(a.review).read_text()) if a.review else {}

    # Gold index: (movie, frame) -> list of precise tips. Scoring only.
    gold: dict[tuple, list] = {}
    want = []
    for o in obs:
        if str(o.get("movie", "")) != movie_key:
            continue
        if not any(r in str(o.get("obs_uuid", "")) for r in a.event_ref):
            continue
        want.append(o)
        if o.get("direct_xy") and str(o.get("direct_state", "")) in (
                "direct_visible", "visible_imprecise"):
            gold.setdefault((movie_key, int(o["source_frame"])), []).append(o)
    if not want:
        print("no events matched --event-ref")
        return 1

    v1, dev = R.load_v1(a.v1_weights)
    # rev9 WP-A.4 / rev11: the runner builds what the CHECKPOINT
    # declares, STRICTLY, by default. The trained parameters are
    # preserved exactly — the audit found the allowed-gap branch zeroing
    # a PRESENT trained route head (abs sum 1.28663 -> 0.0, runtime
    # probability pinned to 0.5). A legacy migration is performed only
    # when explicitly requested AND applicable (a legacy-declaration
    # failure); every other contract failure is final. No silent
    # architecture fallback remains.
    from prototypes.v30_video_apex.model_factory import (
        CheckpointContractError, LegacyCheckpointError,
        build_model_from_checkpoint, optimizer_groups_summary,
        parameter_hash)
    finfo: dict = {}
    meta: dict = {}
    front_load: dict = {}
    try:
        fmodel, finfo = build_model_from_checkpoint(
            a.front_checkpoint,
            allow_route_head_gap=bool(a.allow_route_head_gap),
            allow_legacy_semantics=False)
        print(f"model from checkpoint: {finfo['model_kwargs']} "
              f"| groups: {optimizer_groups_summary(finfo['config'])}")
        if finfo.get("route_head_fallback"):
            print(f"note: route head fallback — "
                  f"{finfo['route_head_fallback']} (declared)")
        else:
            print(f"route head preserved from checkpoint "
                  f"(post-load params sha256 "
                  f"{finfo['parameter_hash'][:12]})")
        front_load = {
            "mode": "strict-current",
            "checkpoint": str(a.front_checkpoint),
            "config_hash": finfo.get("config_hash", ""),
            "parameter_hash_post_load": finfo.get("parameter_hash"),
            "manifest": finfo.get("manifest", {}),
            "model_kwargs": finfo.get("model_kwargs"),
            "semantics": finfo.get("semantics"),
            "missing_keys": finfo.get("missing_keys"),
            "unexpected_keys": finfo.get("unexpected_keys"),
            "route_head_fallback": finfo.get("route_head_fallback"),
        }
    except LegacyCheckpointError as e:
        if not a.allow_legacy_migration:
            print(f"REFUSE: {e}")
            print("this is a pre-contract checkpoint; rerun with "
                  "--allow-legacy-migration to replay it under its own "
                  "declared (or the CLI-declared) architecture — the "
                  "migration is then recorded in the report).")
            return 1
        # Legacy checkpoint and the migration was explicitly requested.
        # Stage 1: the checkpoint's OWN declared architecture, replayed
        # under an explicitly declared semantics. Stage 2 (only when the
        # config itself predates the contract): the CLI-declared
        # architecture. A failure of either stage that is not itself a
        # legacy declaration is FINAL.
        print(f"legacy migration (explicitly requested): {e}")
        try:
            fmodel, finfo = build_model_from_checkpoint(
                a.front_checkpoint,
                allow_route_head_gap=bool(a.allow_route_head_gap),
                allow_legacy_semantics=True)
            print(f"migrated under the checkpoint's own declared "
                  f"architecture: {finfo['model_kwargs']} "
                  f"(semantics {finfo['semantics']})")
            front_load = {
                "mode": "legacy-migration",
                "checkpoint": str(a.front_checkpoint),
                "config_hash": finfo.get("config_hash", ""),
                "parameter_hash_post_load": finfo.get("parameter_hash"),
                "manifest": finfo.get("manifest", {}),
                "model_kwargs": finfo.get("model_kwargs"),
                "semantics": finfo.get("semantics"),
                "missing_keys": finfo.get("missing_keys"),
                "unexpected_keys": finfo.get("unexpected_keys"),
                "route_head_fallback": finfo.get("route_head_fallback"),
            }
        except LegacyCheckpointError as e2:
            # Stage 2: the config itself predates the contract — the
            # architecture exists only on the command line.
            print(f"checkpoint config predates the contract ({e2}); "
                  f"replaying under the CLI-declared architecture: "
                  f"base={a.base}, multiscale=False")
            try:
                fmodel = build_model("temporal", base=a.base)
                meta = load_checkpoint(a.front_checkpoint, fmodel,
                                       strict=False)
            except RuntimeError as e3:
                print(f"REFUSE: the CLI-declared architecture does not "
                      f"fit these weights ({e3})")
                return 1
            if meta.get("missing_keys"):
                print(f"note: checkpoint gaps {meta['missing_keys']} "
                      f"(route_p falls back to 0.5)")
            _zeroed = any("route_head" in k
                          for k in meta.get("missing_keys", []))
            if _zeroed:
                with torch.no_grad():
                    for p in fmodel.route_head.parameters():  # type: ignore[union-attr]
                        p.zero_()
            front_load = {
                "mode": "legacy-migration",
                "checkpoint": str(a.front_checkpoint),
                "config_hash": meta.get("config_hash", ""),
                "parameter_hash_post_load": parameter_hash(fmodel),
                "manifest": meta.get("manifest", {}),
                "model_kwargs": None,
                "semantics": "legacy-declared-replay (CLI architecture)",
                "missing_keys": meta.get("missing_keys"),
                "unexpected_keys": meta.get("unexpected_keys"),
                "route_head_fallback": ("zeroed-absent-legacy-head"
                                        if _zeroed else None),
            }
        except CheckpointContractError as e2:
            print(f"REFUSE: {e2}")
            print("no fallback: contract failures other than a legacy "
                  "declaration are final (rev11 loader rule)")
            return 1
    except CheckpointContractError as e:
        print(f"REFUSE: {e}")
        print("no fallback: contract failures other than a legacy "
              "declaration are final (rev11 loader rule)")
        return 1
    fmodel.eval()
    if a.stop_after_load:
        (out / "load_provenance.json").write_text(json.dumps({
            "front_load": front_load,
            "v1_weights": str(a.v1_weights),
            "snapshot": str(a.snapshot),
            "movie": str(a.movie),
            "event_ref": list(a.event_ref),
        }, indent=2, default=str))
        print(f"stop-after-load: provenance -> "
              f"{out / 'load_provenance.json'}")
        return 0
    # Dedup pixel-exact doubles (span confirmations re-click the same tip).
    deduped, seen_keys, n_dropped_doubles = [], set(), 0
    for o in sorted(want, key=lambda d: str(d.get("obs_uuid", ""))):
        tip = o.get("direct_xy")
        key = (movie_key, int(o.get("source_frame", -1)),
               round(float(tip[0]), 1) if tip else None,
               round(float(tip[1]), 1) if tip else None)
        if key in seen_keys:
            n_dropped_doubles += 1
            continue
        seen_keys.add(key)
        deduped.append(o)
    want = deduped
    # Gold must match the deduped events: a span double's gold row would
    # otherwise be an unscorable miss (different owner key, no predictions).
    kept_uuids = {str(o.get("obs_uuid", "")) for o in want}
    kept_tasks = {str(o.get("task_uuid", "")) for o in want}
    for key in list(gold):
        gold[key] = [g for g in gold[key]
                     if str(g.get("task_uuid", "")) in kept_tasks]
        if not gold[key]:
            del gold[key]
    reader = FrameReader(movie_path)
    H, W = reader.native_size[1], reader.native_size[0]

    CS = a.crop
    measurements, candidates, auto_rows, assisted_rows = [], [], [], []
    selection_rows: list[dict] = []  # rev8: per-event selection decisions
    gold_pred_rows, gold_gold_rows = [], []

    for o in want:
        frame = int(o["source_frame"])
        # Root prompt anchor: the review-confirmed path start when the
        # event has one (a prompt, not geometry — proposals still come
        # from pixels); otherwise the task gaze point.
        if o.get("path_xy") and len(o["path_xy"]) >= 2:
            anchor = (float(o["path_xy"][0][0]),
                      float(o["path_xy"][0][1]))
            anchor_source = "human-root"
        else:
            anchor = tuple(float(v) for v in (
                o.get("focus_xy") or o.get("direct_xy") or (W / 2, H / 2)))
            anchor_source = "gaze"
        gray = reader.read(frame).frame
        gray = gray[:, :, 0] if gray.ndim == 3 else gray
        gray = np.asarray(gray)
        # Relative image-evidence gate (eval_route_evidence.py). The
        # straight-fan family (`a*`: 2-point radial lines) can sit at
        # the top of the front softmax while lying on no tube; wall
        # support along a route, normalized per frame (H148 lesson),
        # gates such candidates out before the trained ranker votes.
        # rev10 WP-C: the frame-normalised image support is a MEASUREMENT
        # and must be recorded whether or not a gate consumes it. It used
        # to be computed only when --select-evidence-gate > 0, so every
        # exported candidate carried ev=0.0 on runs without the flag --
        # a silently-zero field (H344 family) that would mislead any
        # consumer ranking or auditing by image support.
        _gy, _gx = np.gradient(gray.astype(np.float32))
        _ev_norm = float(np.percentile(
            np.abs(_gx) + np.abs(_gy), 95)) or 1.0
        root_xy, root_source = R.auto_root(gray, anchor)
        # rev6: attachment fans (boundary points, not the center) carry
        # availability; v1-peak routes add apex-seeded alternatives.
        # A 12px center error collapses fan coverage 1.00->0.00.
        (grain_c, radius_px, grain_source) = R.auto_grain(gray, anchor)
        if root_source == "frst":
            proposals = R.propose_from_attachments(
                root_xy, radius_px, dirs_per_attach=1,
                lengths=(120.0,), radii=(8.0, 13.0, 18.0))
            root_source = f"frst-attach({grain_source})"
            # rev7: curved walkers join the set (union availability:
            # walkers win curves 0.6-0.9, fans hold straights 0.5).
            try:
                proposals += R.propose_curved_walkers(
                    gray, root_xy, radius_px, n_attach=8,
                    dirs_per_attach=1, lengths=(120.0,),
                    radii=(8.0, 13.0, 18.0))
            except Exception as e:
                print(f"note: curved walkers failed ({e}); fans only")
        else:
            proposals = []
        heat = R.v1_tip_heat(v1, dev, gray)
        peaks = [p for p in R.heat_peaks(heat)
                 if np.hypot(p[0] - root_xy[0], p[1] - root_xy[1])
                 > RIM_EXCL_PX]
        for i, (x, y, h) in enumerate(sorted(
                peaks, key=lambda p: -p[2])[:3]):
            proposals.append({"route_id": f"p{i}s",
                              "polyline": [list(root_xy), [x, y]],
                              "seed": f"v1-peak-{i}", "seed_heat": h})
            proposals.append({"route_id": f"p{i}c+",
                              "polyline": R._curve(root_xy, (x, y), 12.0),
                              "seed": f"v1-peak-{i}", "seed_heat": h})
            proposals.append({"route_id": f"p{i}c-",
                              "polyline": R._curve(root_xy, (x, y), -12.0),
                              "seed": f"v1-peak-{i}", "seed_heat": h})

        # rev11 item 6: root-connected routes from the owner's OWN body
        # probabilities (declared arm; default off so old runs stay
        # reproducible). One anchor-crop forward with the deployment
        # prompt gives pred.body; routes are traced from it retaining
        # crossing alternatives. The map itself is exported per event.
        body_map_row = None
        if a.body_proposals == "walk":
            try:
                _bw = CS // 2
                bx0 = int(min(max(root_xy[0] - _bw, 0), max(0, W - CS)))
                by0 = int(min(max(root_xy[1] - _bw, 0), max(0, H - CS)))
                _bframes = tuple(max(0, frame + off)
                                 for off in QUERY_OFFSETS)
                _bmissing = tuple(1 if frame + off < 0 else 0
                                  for off in QUERY_OFFSETS)
                bclip_np = load_clip_pixels(
                    reader, _bframes, (bx0, by0, CS, CS), _bmissing)
                bclip = (torch.from_numpy(bclip_np)
                         .unsqueeze(0).unsqueeze(2))
                bprompt, _ = deployment_prompt(
                    str(o.get("obs_uuid", "")), bx0, by0, CS, grain_c,
                    radius_px)
                with torch.no_grad():
                    bp = fmodel.forward(bclip, bprompt)
                bprob = torch.sigmoid(bp.body[0, 0]).numpy()
                root_c = (float(root_xy[0] - bx0), float(root_xy[1] - by0))
                walks = R.propose_body_walks(bprob, root_c,
                                             min_len_px=30.0)
                for w in walks:
                    proposals.append({
                        "route_id": w["route_id"],
                        "polyline": [[float(x + bx0), float(y + by0)]
                                     for x, y in w["polyline"]],
                        "seed": w["seed"], "seed_heat": 0.0,
                        "origin": "body-walk"})
                body_map_row = {
                    "crop_xywh": [bx0, by0, CS, CS],
                    "n_walks": len(walks),
                    "walk_ids": [w["route_id"] for w in walks],
                    "min_prob": 0.5,
                }
            except Exception as e:
                print(f"note: body walks failed ({e}); other families only")
                body_map_row = {"crop_xywh": None, "n_walks": 0,
                                "error": str(e)}

        frames = tuple(max(0, frame + off) for off in QUERY_OFFSETS)
        missing = tuple(1 if frame + off < 0 else 0
                        for off in QUERY_OFFSETS)
        scored = []
        for pr in proposals:
            pts = np.asarray(pr["polyline"], float)
            cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
            ox = int(min(max(cx - CS / 2, 0), max(0, W - CS)))
            oy = int(min(max(cy - CS / 2, 0), max(0, H - CS)))
            route_crop = (pts - np.array([ox, oy])).tolist()
            try:
                rpts, _ = resample_polyline(route_crop)
            except ValueError:
                continue
            if not (rpts[:, 0].min() >= 0 and rpts[:, 0].max() <= CS and
                    rpts[:, 1].min() >= 0 and rpts[:, 1].max() <= CS):
                pr["crop_miss"] = True
                continue
            clip_np = load_clip_pixels(reader, frames, (ox, oy, CS, CS),
                                       missing)
            clip = torch.from_numpy(clip_np).unsqueeze(0).unsqueeze(2)
            # rev7/rev11: deployment prompt carries AUTO geometry only
            # (the DETECTOR's grain disc, never human path geometry,
            # never anything derived from the candidate being scored),
            # built by the one typed constructor. The provenance dict
            # records the detector center in both frames, its radius and
            # the coordinate transform.
            prompt, _pprov = deployment_prompt(
                str(o.get("obs_uuid", "")), ox, oy, CS, grain_c, radius_px)
            with torch.no_grad():
                pred = fmodel.forward(  # type: ignore[call-arg]
                    clip, prompt, route_xy=route_crop)
                q = torch.softmax(pred.front_logits, -1)[0].numpy()
                s_grid = pred.front_s.detach().numpy()
            j = int(q.argmax())
            # rev10 WP-C: the LEARNED current-cap evidence for this
            # candidate. front_logits is the front head's score over
            # arclength; q is its softmax (used to pick the tip). Both
            # the peak logit (is there a cap here at all) and the peak
            # probability (how sharp the choice is along this route)
            # are recorded, and the export carries them.
            with torch.no_grad():
                _fl = pred.front_logits[0]
                front_peak = float(_fl.max())
                front_peak_prob = float(q.max())
                front_margin = float(_fl.max() - _fl.mean())
                # rev11: "save all cap distributions" — the full q(s)
                # over the route's arclength grid, per candidate, not
                # only its peak stats.
                front_q = [round(float(v), 6) for v in
                           torch.softmax(_fl, -1).tolist()]
                front_s_grid = [round(float(v), 3)
                                for v in pred.front_s.tolist()]
            # rev10 WP-C: the model's OWN body evidence for this
            # candidate. Nothing consumed pred.body before; the review
            # requires it exposed AND integrated.
            with torch.no_grad():
                _bp = torch.sigmoid(pred.body[0, 0]).numpy()
            if a.body_ablation == "zero":
                _bp = np.zeros_like(_bp)
            elif a.body_ablation == "swap":
                _bp = np.full_like(_bp, float(_bp.mean()))
            # Every scored proposal exports a tip (route truncated at
            # ITS front): these are the candidates; oracle coverage =
            # best of them, selected = the winner. H164 separation.
            try:
                pexp = export_tip_path_length(
                    pr["polyline"], None, status="direct")
                plen = float(pexp["length_px"] or 0.0)
                # rev10 WP-C: keep the MEASURED portion, not only its
                # endpoint. The export had zero `current_path` fields.
                _pcut = export_tip_path_length(
                    pr["polyline"], min(float(s_grid[j]), plen),
                    status="direct")
                ptip = _pcut["tip"]
                pcur = _pcut.get("path")
            except ValueError:
                ptip = None
                pcur = None
            # rev11: body evidence is scored over the candidate's
            # MEASURED CURRENT PREFIX (the support genuinely owned up to
            # the front cutoff) — the old sampling averaged over the full
            # support, including arbitrary post-tip background that no
            # owner claim covers. A candidate without a current path
            # reports the domain as unavailable rather than pretending.
            _bdomain = "current_prefix"
            _cpts = np.asarray(pcur or [], float)
            if _cpts.shape[0] >= 2:
                _crpts, _ = resample_polyline(_cpts.tolist())
                _ry = np.clip(_crpts[:, 1].round().astype(int), 0, CS - 1)
                _rx = np.clip(_crpts[:, 0].round().astype(int), 0, CS - 1)
                _vals = _bp[_ry, _rx]
            else:
                _bdomain = "unavailable"
                _vals = np.zeros(0, np.float32)
            body_support_mean = float(_vals.mean()) if _vals.size else 0.0
            body_support_max = float(_vals.max()) if _vals.size else 0.0
            body_support_frac = float((_vals > 0.5).mean()) if _vals.size \
                else 0.0
            # Every scored proposal exports a tip (route truncated at
            # ITS front): these are the candidates; oracle coverage =
            # best of them, selected = the winner. H164 separation.
            try:
                pexp = export_tip_path_length(
                    pr["polyline"], None, status="direct")
                plen = float(pexp["length_px"] or 0.0)
                # rev10 WP-C: keep the MEASURED portion, not only its
                # endpoint. The export had zero `current_path` fields.
                _pcut = export_tip_path_length(
                    pr["polyline"], min(float(s_grid[j]), plen),
                    status="direct")
                ptip = _pcut["tip"]
                pcur = _pcut.get("path")
            except ValueError:
                ptip = None
                pcur = None
            import math as _m

            route_p = 1.0 / (1.0 + _m.exp(
                -float(pred.route_logit.detach()[0]))) \
                if pred.route_logit is not None else 0.5
            _ev = 0.0
            if _ev_norm:
                try:
                    _e = wall_energy(gray, resample(
                        np.asarray(pr["polyline"], float), step=4.0))
                    _ev = float(np.median(_e)) / _ev_norm if len(_e) else 0.0
                except Exception:
                    _ev = 0.0
            scored.append({
                "route_id": pr["route_id"], "seed": pr["seed"],
                "seed_heat": pr.get("seed_heat", 0.0),
                "q_max": float(q[j]), "route_p": route_p,
                "sel": float(q[j]) * route_p,
                "ev": _ev,
                "front_peak": front_peak,
                "front_peak_prob": front_peak_prob,
                "front_margin": front_margin,
                "front_q": front_q,
                "front_s_grid": front_s_grid,
                "body_support_mean": body_support_mean,
                "body_support_max": body_support_max,
                "body_support_frac": body_support_frac,
                "body_support_domain": _bdomain,
                "body_ablation": str(a.body_ablation),
                "score_body": float(q[j]) * route_p * body_support_mean,
                "prompt_provenance": _pprov["prompt_kind"],
                "prompt_geometry": _pprov,
                "anchor_source": anchor_source,
                "root_source": root_source,
                "s_hat": float(s_grid[j]),
                "support": float(s_grid[-1]),
                "tip": [float(ptip[0]), float(ptip[1])]
                if ptip else None,
                "current_path": pcur,
                "ox": ox, "oy": oy, "polyline_native": pr["polyline"],
                "s_grid": s_grid.tolist(), "q": q.tolist()})
        scored.sort(key=lambda d: -d["q_max"])
        n_crop_miss = sum(1 for pr in proposals if pr.get("crop_miss"))
        # Selection: route-correctness x front confidence. The heat
        # rerank was tried and hurt (0/8): the apex head fires on every
        # real apex including foreign ones, so heat cannot arbitrate
        # owners. heat_at_tip stays a recorded diagnostic only.
        # heat_at_tip: raw-score readout (NOT a probability; the heat
        # head is a regression score, so no sigmoid). One forward per
        # DISTINCT crop, indexed in that crop's own frame (rev6: the old
        # code read every candidate through candidate-0's crop origin).
        heat_cache: dict[tuple, Any] = {}

        def heat_for(ox: int, oy: int) -> Any:
            key = (int(ox), int(oy))
            arr = heat_cache.get(key)
            if arr is None:
                clip_np = load_clip_pixels(
                    reader, frames, (int(ox), int(oy), CS, CS), missing)
                clip = torch.from_numpy(clip_np).unsqueeze(0).unsqueeze(2)
                with torch.no_grad():
                    hpred = fmodel.forward(  # type: ignore[call-arg]
                        clip, build_owner_prompt(
                            str(o.get("obs_uuid", "")),
                            provenance="none"))
                    arr = hpred.heat[0, 0].detach().numpy()
                heat_cache[key] = arr
            return arr

        if scored:
            for srow in scored:
                if srow["tip"]:
                    hm = heat_for(srow["ox"], srow["oy"])
                    j = int(round(srow["tip"][0] - srow["ox"]))
                    i = int(round(srow["tip"][1] - srow["oy"]))
                    srow["heat_at_tip"] = float(hm[i, j]) \
                        if 0 <= i < hm.shape[0] and 0 <= j < hm.shape[1] \
                        else 0.0
                else:
                    srow["heat_at_tip"] = 0.0
            scored.sort(key=lambda d: -d["sel"])
        # Relative evidence gate (optional): keep the ranker's vote, but
        # only among candidates that actually lie on image structure.
        sel_pool = scored
        n_gated_out = 0
        _best_ev_all = max((float(s.get("ev") or 0.0) for s in scored),
                           default=0.0)
        # rev8: ONE selection policy, shared with the offline evaluator.
        # The runner gates evidence and ranks; refusals carry a reason.
        from prototypes.v30_video_apex.selection import (
            SelectionPolicy, select_candidate)
        _policy = SelectionPolicy(rank=str(a.select_rank),
                                  evidence_gate=float(
                                      a.select_evidence_gate),
                                  min_score=float(a.min_q),
                                  min_evidence=float(
                                      getattr(a, "select_min_evidence",
                                              0.0)))
        _dec = select_candidate(scored, _policy)
        selection_rows.append({**_dec.as_dict(),
                               "event": str(o.get("obs_uuid", "")),
                               "frame": int(frame)})
        winner = None
        if _dec.winner is not None:
            # restore the real q_max on the selected row for reporting
            real = next((s for s in scored
                         if s["route_id"] == _dec.winner["route_id"]), None)
            winner = dict(real) if real is not None else dict(_dec.winner)
            n_gated_out = _dec.n_candidates - _dec.n_kept
        guard_withheld = bool(scored) and winner is None

        assisted = None
        for key, corr in review.items():
            if key in str(o.get("obs_uuid", "")):
                assisted = corr
        if assisted is not None:
            # Human tip correction: export the corrected TIP ONLY with
            # length withheld (rev6). A straight root->tip segment is
            # not the tube and must never be reported as its path.
            try:
                exp = export_tip_path_length(
                    None,
                    None,
                    tip_xy=[float(assisted["tip_xy"][0]),
                            float(assisted["tip_xy"][1])],
                    status="assisted")
            except ValueError as e:
                exp = {"tip": None, "path": None, "length_px": None,
                       "status": "withheld", "withheld": str(e)[:120]}
            assisted_rows.append({"event": str(o.get("obs_uuid", ""))})
        elif winner is not None:
            poly = winner["polyline_native"]
            try:
                exp = export_tip_path_length(
                    poly, min(winner["s_hat"],
                              float(np.sum(np.hypot(
                                  *np.diff(np.asarray(poly, float),
                                           axis=0).T)))),
                    status="direct")
            except ValueError as e:
                exp = {"tip": None, "path": None, "length_px": None,
                       "status": "withheld", "withheld": str(e)[:120]}
            auto_rows.append({"event": str(o.get("obs_uuid", ""))})
        else:
            exp = {"tip": None, "path": None, "length_px": None,
                   "status": "withheld", "withheld": "no-fitting-proposal"}
        proposal_miss = (winner is None and n_crop_miss == 0 and
                         not [p for p in peaks])

        measurements.append({
            "event": str(o.get("obs_uuid", "")), "movie": movie_key,
            "source_frame": frame, "root_xy": list(root_xy),
            "root_source": root_source,
            "body_proposals": body_map_row,
            "anchor": list(anchor), "anchor_source": anchor_source,
            "n_proposals": len(proposals), "n_scored": len(scored),
            "n_crop_miss": n_crop_miss,
            "select_evidence_gate": float(a.select_evidence_gate),
            "n_gated_out": int(n_gated_out),
            "proposal_miss": bool(proposal_miss),
            "guard_withheld": bool(guard_withheld),
            "winner": ({k: winner[k] for k in (
                "route_id", "seed", "q_max", "route_p", "sel",
                "s_hat", "support", "prompt_provenance")}
                if winner else None),
            "prompt_provenance": (winner.get("prompt_provenance")
                                  if winner else None),
            "anchor_source": anchor_source,
            "tip": exp["tip"], "path": exp["path"],
            "length_px": exp["length_px"], "status": exp["status"],
            "assisted": assisted is not None})
        for srow in scored:
            if srow["tip"] is None:
                continue
            candidates.append({
                "candidate_id": f"{movie_key}:{frame}:"
                f"{o.get('obs_uuid')}:{srow['route_id']}",
                "movie_id": movie_key,
                "owner_id": str(o.get("obs_uuid", "")),
                "source_frame": frame,
                # rev10 WP-C: frame index AND acquisition time together.
                # The cadence is DECLARED (--acquisition-cadence-s); when
                # it is not declared this stays null, because playback
                # FPS does not establish biological growth rate.
                "source_time_s": (round(float(frame) * float(
                    a.acquisition_cadence_s), 3)
                    if float(a.acquisition_cadence_s or 0.0) > 0.0
                    else None),
                "acquisition_cadence_s": (float(a.acquisition_cadence_s)
                                          if float(a.acquisition_cadence_s
                                                   or 0.0) > 0.0 else None),
                "x_native": float(srow["tip"][0]),
                "y_native": float(srow["tip"][1]),
                "route_id": srow["route_id"],
                "polyline_native": srow["polyline_native"],
                # rev10 WP-C: one candidate contract everywhere —
                # support path, front arclength cutoff, measured portion,
                # current tip, origins, evidence, status, owner, frame
                # and the coordinate transform.
                "support_path": srow.get("polyline_native"),
                "front_s": srow.get("s_hat"),
                "support_s": srow.get("support"),
                "current_path": srow.get("current_path"),
                "current_tip": ([float(srow["tip"][0]),
                                 float(srow["tip"][1])]
                                if srow.get("tip") else None),
                "coord_transform": {
                    "crop_origin_native": [int(srow["ox"]),
                                            int(srow["oy"])],
                    "scale": 1.0,
                    "note": "native frame -> crop: subtract the origin; "
                            "no resampling"},
                "physical_origin": {
                    # rev11: the DETECTED center (routes.auto_grain's
                    # output), not the search anchor. The old value
                    # recorded the anchor the detector was pointed from
                    # — an audit could not separate the two.
                    "grain_center_native": [float(grain_c[0]),
                                            float(grain_c[1])],
                    "grain_center_source": str(grain_source),
                    "grain_center_radius_px": float(radius_px),
                    "search_anchor_native": [float(anchor[0]),
                                             float(anchor[1])],
                    "search_anchor_source": str(anchor_source),
                    "grain_center_use": "prompt disc identity only; "
                                        "does NOT enter the length",
                    "length_origin": "proximal_path_end",
                    "length_origin_status": "partial-origin (no verified "
                                            "grain exit known yet)",
                    "note": "the review requires a grain-boundary exit "
                            "or first visible proximal point for the "
                            "physical length origin; these labels do not "
                            "carry it yet, so the measured length starts "
                            "at the proximal end of the proposed support "
                            "path and is exported as a partial-origin "
                            "measurement",
                },
                "visibility": "observed",
                "status": "observed",
                "front_evidence": {
                    "peak_logit": round(float(srow.get("front_peak") or 0),
                                        4),
                    "peak_prob": round(float(
                        srow.get("front_peak_prob") or 0), 6),
                    "margin": round(float(srow.get("front_margin") or 0), 4),
                    "q_s": srow.get("front_q"),
                    "s_grid": srow.get("front_s_grid"),
                    "note": "the front head's own score along this "
                            "candidate's route; the review's 'learned "
                            "current-cap evidence'. q_s = the FULL cap "
                            "distribution over the route's arclength "
                            "grid (s_grid) — every candidate's "
                            "distribution is saved, not only its peak.",
                },
                "body_support": {
                    "mean": round(float(srow.get("body_support_mean") or 0),
                                  4),
                    "max": round(float(srow.get("body_support_max") or 0), 4),
                    "frac_above_half": round(float(
                        srow.get("body_support_frac") or 0), 4),
                    "domain": srow.get("body_support_domain",
                                       "current_prefix"),
                    "ablation": srow.get("body_ablation", "none"),
                    "note": "pred.body sampled along this candidate's "
                            "MEASURED CURRENT PREFIX (rev11); "
                            "'unavailable' means the candidate had no "
                            "current path, never a silent fallback. "
                            "ablation != none means the signal was "
                            "deliberately destroyed for comparison",
                },
                "score_body": srow.get("score_body"),
                "tip_score": srow["q_max"],
                "route_p": srow.get("route_p"),
                "prompt_provenance": srow.get("prompt_provenance"),
                "anchor_source": anchor_source,
                "root_source": root_source,
                "observation": "observed",
                "evidence": {"seed": srow["seed"],
                             "wall_ev": round(float(
                                 srow.get("ev") or 0.0), 4),
                             "gate": float(a.select_evidence_gate),
                             "gate_kept": bool(
                                 a.select_evidence_gate <= 0.0
                                 or _best_ev_all <= 0.0
                                 or float(srow.get("ev") or 0.0)
                                 >= a.select_evidence_gate * _best_ev_all)},
                # rev6: the exported winner is flagged so the
                # evaluator scores the actual choice, not top-q_max.
                "selected": bool(
                    winner is not None and assisted is None and
                    srow["route_id"] == winner["route_id"])})
        # Selection scoring: every proposal tip is a candidate; the
        # top-q winner is the model's actual choice (score_probes
        # separates best-of-set oracle from top-score selected).
        for cnd in candidates:
            if (cnd["movie_id"] == movie_key and
                    cnd["source_frame"] == frame and
                    cnd["owner_id"] == str(o.get("obs_uuid", ""))):
                gold_pred_rows.append(cnd)
    reader.close()

    # Joint conflict rule across same-frame events (kept separate owners).
    by_frame: dict[tuple, list] = {}
    for m in measurements:
        if m["tip"] is not None:
            by_frame.setdefault((m["movie"], m["source_frame"]),
                                []).append(m)
    conflicts = 0
    for key, ms in by_frame.items():
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                d = np.hypot(ms[i]["tip"][0] - ms[j]["tip"][0],
                             ms[i]["tip"][1] - ms[j]["tip"][1])
                if d < CONFLICT_PX:
                    conflicts += 1
                    ms[i].setdefault("conflicts", []).append(ms[j]["event"])
                    ms[j].setdefault("conflicts", []).append(ms[i]["event"])

    for tips in gold.values():
        for g in tips:
            gold_gold_rows.append({
                "movie_id": movie_key,
                "owner_id": str(g.get("obs_uuid", "")),
                "source_frame": int(g["source_frame"]),
                "x_native": float(g["direct_xy"][0]),
                "y_native": float(g["direct_xy"][1]),
                "observation": "observed", "stratum": "pilot"})
    report = score_probes(gold_pred_rows, gold_gold_rows, tolerance_px=5.0)
    # rev8: refusals and negatives are first-class in the same report —
    # a system that withholds everything must not read as good, and a
    # system that emits on every grain must not read as accurate.
    if isinstance(report, dict):
        _n_ref = sum(1 for r in selection_rows if r.get("refused"))
        report["n_refused_opportunities"] = _n_ref
        report["refusal_rate"] = (
            _n_ref / len(selection_rows)) if selection_rows else 0.0
        _den = max(1, int(report.get("n_opportunities", 0)) - _n_ref)
        report["usable_coverage"] = (
            int(report.get("selected_hits", 0)) / _den)
    # rev6: score the FINAL exported measurements (gating, assisted
    # tips, refusals) alongside the candidate-set metrics.
    final_report = score_measurements(
        measurements, gold_gold_rows, tolerance_px=5.0)

    # Oracle-route substitution hook (diagnostic, reported separately).
    oracle_routes = {}
    for o in want:
        if o.get("path_xy") and len(o["path_xy"]) >= 2:
            oracle_routes[(movie_key, str(o.get("obs_uuid", "")))] = \
                o["path_xy"]
    oracle_cands = substitute_oracle_routes(candidates, {
        (movie_key, str(o.get("obs_uuid", ""))): {"polyline": p}
        for (mk, ou), p in oracle_routes.items()})

    beam_demo = []
    if candidates:
        by_owner: dict[str, list] = {}
        for cnd in candidates:
            by_owner.setdefault(cnd["owner_id"], []).append(cnd)
        owner0 = sorted(by_owner)[0]
        beam_demo = [{
            "owner": h.owner_id, "route": h.route_id, "frame": h.frame,
            "x": h.x, "y": h.y, "score": h.score,
            "state": str(h.state)}
            for h in beam_search_per_owner([by_owner[owner0][:4]], owner0)]

    # P1 route diagnostic: is the true tip even ON any proposal
    # (availability) vs did the winner pick the wrong one (ranking)?
    # Gold is used here as a measuring stick, never as an input.
    # rev11: THREE distinct quantities per event, each measured on its
    # own geometry — support proximity (whole proposal polyline),
    # measured-path deviation (current_path, the truncated portion), and
    # endpoint error (the exported current_tip). A support line passing
    # near the gold tip is NOT a detected front; a value must never be
    # silently read from one domain under another's name.
    route_recall = []
    for m in measurements:
        g = next((g for g in gold_gold_rows
                  if g["owner_id"] == m["event"] and
                  g["source_frame"] == m["source_frame"]), None)
        if g is None:
            continue
        gx, gy = g["x_native"], g["y_native"]
        from prototypes.v30_video_apex.evaluate import candidate_metrics

        rows_d = []
        for cnd in candidates:
            if cnd["owner_id"] != m["event"]:
                continue
            _cm = candidate_metrics((gx, gy), cnd.get("polyline_native"),
                                    cnd.get("current_path"),
                                    cnd.get("current_tip"))
            rows_d.append({
                "route_id": cnd.get("route_id", ""),
                "support_px": _cm["support_px"],
                "current_px": _cm["current_px"],
                "endpoint_px": _cm["endpoint_px"],
                "is_winner": bool(
                    (m.get("winner") or {}).get("route_id") ==
                    cnd.get("route_id"))})

        def _best_of(key):
            vals = [d[key] for d in rows_d if d[key] is not None]
            return min(vals) if vals else None

        def _winner_of(key):
            for d in rows_d:
                if d["is_winner"]:
                    return d[key]
            return None

        _b_sup, _w_sup = _best_of("support_px"), _winner_of("support_px")
        route_recall.append({
            "event": m["event"], "source_frame": m["source_frame"],
            "n_candidates": len(rows_d),
            "support": {"best_px": _b_sup, "winner_px": _w_sup,
                        "winner_is_closest": (
                            None if _b_sup is None or _w_sup is None
                            else bool(_w_sup <= _b_sup + 1e-9))},
            "current_path": {"best_px": _best_of("current_px"),
                             "winner_px": _winner_of("current_px"),
                             "n_with_current_path": sum(
                                 1 for d in rows_d
                                 if d["current_px"] is not None)},
            "endpoint": {"best_px": _best_of("endpoint_px"),
                         "winner_px": _winner_of("endpoint_px"),
                         "winner_route_id": next(
                             (d["route_id"] for d in rows_d
                              if d["is_winner"]), None),
                         "n_with_endpoint": sum(
                             1 for d in rows_d
                             if d["endpoint_px"] is not None)},
        })

    (out / "measurements.json").write_text(json.dumps(
        measurements, indent=2))
    (out / "candidates.json").write_text(
        json.dumps(export_candidates(candidates), indent=2))
    (out / "selection.json").write_text(json.dumps(
        selection_rows, indent=2))
    # rev10 WP-C: proposal coverage, selection accuracy and final
    # measurement error reported SEPARATELY — "Do not optimize selection
    # alone when most correct fronts are absent from the candidate set."
    # Availability is DERIVED from the diagnostic above (closest
    # PROPOSED candidate to the gold tip), never read off a constant:
    # the first version of this block reported a snapshot flag that is
    # False every run, which looks like "no coverage" and means nothing.
    # rev10 WP-C / rev11: proposal coverage, selection accuracy and
    # final measurement error reported SEPARATELY — "Do not optimize
    # selection alone when most correct fronts are absent from the
    # candidate set." rev11: each event reports THREE quantities with
    # their own tolerance BESIDE them (support proximity / measured-path
    # proximity / endpoint error). The old `front_available` (support
    # proximity at 15 px) is gone — it was read as if it counted real
    # fronts; support distance is not an endpoint measurement. A missing
    # current path or endpoint reports unavailable (null), never a
    # silent fallback to support geometry.
    _rr_by_event = {rr.get("event"): rr for rr in route_recall}
    _TOL_PX = 5.0
    _coverage = []
    for m in measurements:
        _ev = m.get("event")
        _rr = _rr_by_event.get(_ev) or {}
        _sup = _rr.get("support") or {}
        _cur = _rr.get("current_path") or {}
        _end = _rr.get("endpoint") or {}
        _b_sup = _sup.get("best_px")
        _b_cur = _cur.get("best_px")
        _b_end = _end.get("best_px")
        _w_end = _end.get("winner_px")
        _coverage.append({
            "event": _ev, "source_frame": m.get("source_frame"),
            "n_proposals": int(m.get("n_proposals") or 0),
            "n_scored": int(m.get("n_scored") or 0),
            "support": {
                "best_distance_px": _b_sup, "tol_px": _TOL_PX,
                "within_tol": (None if _b_sup is None
                               else bool(_b_sup <= _TOL_PX))},
            "current_path": {
                "best_distance_px": _b_cur, "tol_px": _TOL_PX,
                "within_tol": (None if _b_cur is None
                               else bool(_b_cur <= _TOL_PX)),
                "available": _b_cur is not None},
            "endpoint_oracle": {
                "best_endpoint_error_px": _b_end, "tol_px": _TOL_PX,
                "within_tol": (None if _b_end is None
                               else bool(_b_end <= _TOL_PX))},
            "selected_endpoint": {
                "endpoint_error_px": _w_end, "tol_px": _TOL_PX,
                "within_tol": (None if _w_end is None
                               else bool(_w_end <= _TOL_PX))},
            "n_oracle_substituted_this_event": int(m.get(
                "n_oracle_substituted") or 0),
        })

    def _cov_count(section: str, key: str) -> int:
        return sum(1 for c in _coverage
                   if (c.get(section) or {}).get(key) is True)

    _coverage_summary = {
        "n_events": len(_coverage), "tol_px": _TOL_PX,
        "n_support_within_tol": _cov_count("support", "within_tol"),
        "n_current_path_within_tol": _cov_count("current_path",
                                                "within_tol"),
        "n_endpoint_oracle_within_tol": _cov_count("endpoint_oracle",
                                                   "within_tol"),
        "n_selected_endpoint_within_tol": _cov_count("selected_endpoint",
                                                     "within_tol"),
        "note": "each quantity carries its own tolerance; support "
                "proximity here is the candidate's FULL proposal "
                "polyline and is not an endpoint measurement. The "
                "previous report's 15 px support tolerance is not the "
                "5 px endpoint criterion.",
    }
    _src_counts: dict[str, int] = {}
    for _c in export_candidates(candidates):
        _k = str((_c.get("evidence") or {}).get("route_source"))
        _src_counts[_k] = _src_counts.get(_k, 0) + 1
    _sel = [{"event": r.get("event"), "winner": r.get("winner"),
             "refused": r.get("refused"), "reason": r.get("reason"),
             "n_candidates": r.get("n_candidates"),
             "n_kept": r.get("n_kept")} for r in selection_rows]
    _meas_rows = [{"event": rr.get("event"),
                   "selected_route": (rr.get("endpoint") or {}).get(
                       "winner_route_id"),
                   "support_distance_px": (rr.get("support") or {}).get(
                       "winner_px"),
                   "current_path_distance_px": (rr.get("current_path")
                                                or {}).get("winner_px"),
                   "endpoint_error_px": (rr.get("endpoint") or {}).get(
                       "winner_px"),
                   "tol_px": _TOL_PX}
                  for rr in route_recall]
    from prototypes.v30_video_apex.evaluate import (
        MEDIAN_CONVENTION as _MEDIAN_CONVENTION,
        conventional_median as _conventional_median,
        upper_middle as _upper_middle)
    _end_errs = [r["endpoint_error_px"] for r in _meas_rows
                 if r["endpoint_error_px"] is not None]
    _meas_summary = {
        "n_selected": len(_end_errs), "tol_px": _TOL_PX,
        "n_selected_endpoint_within_tol": sum(
            1 for e in _end_errs if e <= _TOL_PX),
        "median_endpoint_error_px": _conventional_median(_end_errs),
        "median_convention": _MEDIAN_CONVENTION,
        "upper_middle_endpoint_error_px": _upper_middle(_end_errs),
    }
    (out / "report.json").write_text(json.dumps({
        "selected_vs_oracle": report,
        "proposal_coverage": _coverage,
        "proposal_coverage_summary": _coverage_summary,
        "selection_accuracy": _sel,
        # rev11: the selected ENDPOINT error (rows carry the three
        # quantities separately; the summary's median is the documented
        # conventional one). The old `final_measurement_error` repeated
        # support-polyline distances under a measurement name.
        "final_selection_error": {"rows": _meas_rows,
                                  "summary": _meas_summary},
        "measurement_domain": {
            "path_field": "current_path — the candidate's own support "
                          "path truncated at ITS front arclength cutoff "
                          "(front_s). support_path is used ONLY for "
                          "support-proximity diagnostics (proposal "
                          "coverage), never as a measurement.",
            "fallback": "none — a candidate without a current path (or "
                        "without an endpoint) is reported unavailable "
                        "(null) rather than silently measured on "
                        "support geometry",
            "length": "measured from the proximal end of the proposed "
                      "support path; grain centre is identity only and "
                      "does NOT enter the length",
            "time": "source_frame plus a DECLARED acquisition cadence; "
                    "null when undeclared",
            "calibration": "native pixels, no resampling",
        },
        "candidate_route_sources": _src_counts,
        "selection_decisions": selection_rows,
        "n_refused": sum(1 for r in selection_rows if r.get("refused")),
        "route_recall": route_recall,
        "n_conflicts": conflicts,
        "n_dropped_doubles": n_dropped_doubles,
        "n_auto": len(auto_rows), "n_assisted": len(assisted_rows),
        "n_oracle_routes_available": len(oracle_routes),
        "n_oracle_substituted": sum(
            1 for c in oracle_cands
            if c.get("evidence", {}).get("route_source") == "oracle"),
        "front_checkpoint": str(a.front_checkpoint),
        "select_evidence_gate": float(a.select_evidence_gate),
        # rev11: provenance from the SUCCESSFULLY loaded checkpoint —
        # mode, manifest, config hash and the POST-LOAD parameter hash
        # (the audit verifies the runner's loaded tensors against the
        # checkpoint; the old report's front_manifest was empty because
        # it read a key the factory info did not carry).
        "front_load": front_load,
        "front_manifest": front_load.get("manifest", {}),
        "front_model_kwargs": front_load.get("model_kwargs"),
        "beam_demo": beam_demo,
    }, indent=2, default=str))
    report_path = out / "report.json"
    rep = json.loads(report_path.read_text())
    rep["final_measurements"] = final_report
    rep["conflicts_resolved"] = False  # rev6: conflicts are recorded
    report_path.write_text(json.dumps(rep, indent=2, default=str))
    # A vacuous 0.00 (empty gold set) must not read like a failed 0.00.
    _n_gold = int(report.get("n_opportunities") or 0)
    print(f"movie run -> {out}: {len(measurements)} events, "
          f"coverage={report['correct_owner_coverage']:.2f} "
          f"oracle={report['oracle_coverage']:.2f} "
          f"(selected/oracle over n_gold={_n_gold}"
          f"{' -- VACUOUS, no gold rows for these events' if _n_gold == 0 else ''}) "
          f"conflicts={conflicts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
