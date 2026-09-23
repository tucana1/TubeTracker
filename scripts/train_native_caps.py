"""Prepare, inspect, fit and evaluate licensed native cap supervision.

The six reviewed distal-end crops are a plumbing/fit panel, not a claim
of whole-movie or held-out accuracy. Runtime uses the identical scorer.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prototypes.v30_video_apex.cap_evidence import CapLabelIndex
from prototypes.v30_video_apex.native_caps import (
    NativeCapNet, OFFSETS, QUERY_INDEX, TILE, VALID_MARGIN, extract_tile,
    file_hash, load_native_checkpoint, native_cap_loss, save_native_checkpoint,
    score_native_tile)

PAD = 16
STORED_TILE = TILE + 2 * PAD


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def reviewed_corpus_cases(index, fit_movies, reserved_intervals, *, sampling_radius=2.0):
    """Group nearby clicks for sampling; never merge physical grain identities."""
    if index.training_scope is None:
        index = index.for_training(context_offsets=OFFSETS, reserved_intervals=reserved_intervals)
    fit_movies = set(fit_movies)
    if not fit_movies:
        raise ValueError("reviewed-corpus requires an explicit --fit-movies declaration")
    reserved = [(str(m), int(first), int(last)) for m, first, last in reserved_intervals]
    if any(first > last for _, first, last in reserved):
        raise ValueError("reserved interval endpoints are reversed")
    def exclusion(movie, frame):
        if movie not in fit_movies:
            return 'reserved_movie'
        if any(movie == m and any(first <= frame+d <= last for d in OFFSETS)
               for m, first, last in reserved):
            return 'reserved_query_or_context_frame'
        return None
    cases, excluded = [], copy.deepcopy(index.training_scope['excluded'])
    for (movie, frame), points in sorted(index.points.items()):
        reason = exclusion(movie, frame)
        if reason:
            excluded.extend({'movie': movie, 'frame': frame, 'obs_uuid': p['obs_uuid'],
                             'project': p['project'], 'revision': p['revision'], 'reason': reason} for p in points)
            continue
        def priority(p):
            observation = index.resolved_observations[(movie, frame, p['owner'])]
            return (observation.get('task_type') != 'review_tip', -len(p['lineage']),
                    -p['revision'], p['project'], p['obs_uuid'])
        groups = []
        for point in sorted(points, key=priority):
            group = next((g for g in groups if all(
                np.linalg.norm(np.asarray(point['xy'])-p['xy']) <= sampling_radius for p in g)), None)
            if group is None:
                groups.append([point])
            else:
                group.append(point)
        for group in groups:
            canonical = group[0]
            cases.append({'id': canonical['obs_uuid'], 'kind': 'positive', 'movie': movie,
                          'frame': frame, 'centre': canonical['xy'], 'truth': copy.deepcopy(canonical),
                          'sampling_group': {'radius_px': sampling_radius, 'members': copy.deepcopy(group),
                              'owner_aliases': sorted({p['owner'] for p in group}),
                              'meaning': 'nearby query sampling only; all original labels and separate owner aliases retained'}})
    for (movie, frame), regions in sorted(index.negatives.items()):
        for region in regions:
            reason = exclusion(movie, frame)
            if reason:
                excluded.append({'movie': movie, 'frame': frame, 'region_uuid': region['_region_uuid'], 'reason': reason})
                continue
            polygon = np.asarray(region['polygon_xy'])
            cases.append({'id': region['_region_uuid'], 'kind': 'negative_region', 'movie': movie,
                          'frame': frame, 'centre': polygon.mean(0).tolist(),
                          'truth': {'polygon_xy': region['polygon_xy'], 'revision': region['_region_revision'],
                                    'project': region['_project']}})
    audit = {'fit_movies': sorted(fit_movies), 'reserved_intervals': reserved,
             'context_offsets_checked': list(OFFSETS), 'excluded': excluded,
             'positive_point_records_used': sum(len(c['sampling_group']['members']) for c in cases if c['kind'] == 'positive'),
             'positive_sampling_groups': sum(c['kind'] == 'positive' for c in cases),
             'negative_regions': sum(c['kind'] == 'negative_region' for c in cases),
             'identity_limit': 'Some owner IDs are observation aliases. Sampling groups are not independent biological grains; frame exclusions do not establish owner-independent generalization.'}
    return cases, audit


def hard_negative_spot_cases(path):
    """Sampling locations from measured peaks; reviewed polygons license loss."""
    doc = json.loads(Path(path).read_text())
    cases = []
    for k, s in enumerate(doc["spots"]):
        cases.append({"id": f"negspot-{k:02d}-{s['case']}", "kind": "negative_spot",
                      "movie": s["movie"], "frame": int(s["frame"]),
                      "centre": [float(s["x"]), float(s["y"])],
                      "radius_px": float(s.get("radius_px", 8.0)),
                      "measured_probability": s.get("probability"),
                      "source_case": s["case"], "source_peak_kind": s.get("kind"),
                      "truth": {"note": "measured peak selects a crop and focus disk; "
                                        "only intersection with existing reviewed negatives is supervised"}})
    return cases, {"source": str(path), "sha256": file_hash(Path(path)), "n_spots": len(cases)}


def scoped_negative_spot(licensed, origin, centre, radius):
    """Concentrate loss near a hard example without expanding its license."""
    if not np.isfinite(radius) or radius <= 0 or not np.isfinite(centre).all():
        raise ValueError('hard-negative focus requires finite geometry and a positive radius')
    yy, xx = np.mgrid[:licensed.shape[0], :licensed.shape[1]]
    cc = np.asarray(centre, float) - np.asarray(origin)
    disk = (xx-cc[0])**2 + (yy-cc[1])**2 <= radius**2
    mask = np.asarray(licensed, bool) & disk
    return mask, {'requested_spot_pixels': int(disk.sum()),
                  'licensed_spot_pixels': int(mask.sum()),
                  'unlicensed_spot_pixels_excluded': int((disk & ~licensed).sum())}


def prepare(args):
    from tubetracker.annotation_frames import FrameReader
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    snap = Path(args.snapshot).resolve()
    sm = json.loads((snap / "snapshot_manifest.json").read_text())
    intervals = []
    for item in args.reserve_interval:
        movie, first, last = item.rsplit(':', 2)
        intervals.append((movie, int(first), int(last)))
    index = CapLabelIndex.from_snapshot(snap).for_training(
        context_offsets=OFFSETS, reserved_intervals=intervals)
    readers, frame_cache = {}, {}
    membership = None
    if args.positive_source == 'reviewed-corpus':
        cases, membership = reviewed_corpus_cases(index,
            [m.strip() for m in args.fit_movies.split(',') if m.strip()], intervals)
    else:
        cases = []
        for (movie, frame), points in index.points.items():
            for p in points:
                if str(p["task_uuid"]).startswith("rev13tip-"):
                    cases.append({"id": p["obs_uuid"], "kind": "positive", "movie": movie,
                                  "frame": frame, "centre": p["xy"], "truth": p})
        if len(cases) != 6:
            raise ValueError(f"expected six explicit reviewed fit tips, found {len(cases)}")
        for (movie, frame), regions in index.negatives.items():
            for r in regions:
                polygon = np.asarray(r["polygon_xy"])
                cases.append({"id": r["_region_uuid"], "kind": "negative_region", "movie": movie,
                              "frame": frame, "centre": polygon.mean(0).tolist(),
                              "truth": {"polygon_xy": r["polygon_xy"],
                                        "revision": r["_region_revision"], "project": r["_project"]}})
    spot_audit = None
    if getattr(args, 'hard_negative_spots', ''):
        spot_cases, spot_audit = hard_negative_spot_cases(args.hard_negative_spots)
        allowed = {(c['movie'], c['frame']) for c in cases}
        kept, skipped = [], []
        for c in spot_cases:
            (kept if (c['movie'], c['frame']) in allowed else skipped).append(c)
        spot_audit['kept'] = len(kept)
        spot_audit['skipped_frames_outside_panel'] = [c['id'] for c in skipped]
        cases += kept
    pixels, positives, negatives, offsets = [], [], [], []
    observed_owners = set()
    fig, axes = plt.subplots((len(cases) + 3) // 4, 4, figsize=(12, 3 * ((len(cases) + 3) // 4)))
    try:
        for j, case in enumerate(cases):
            movie, frame = case["movie"], case["frame"]
            reader = readers.setdefault(movie, None)
            if reader is None:
                reader = readers[movie] = FrameReader(sm["movies"][movie]["path"])
            fids = [frame + d for d in OFFSETS]
            if min(fids) < 0 or max(fids) >= len(reader):
                raise ValueError("fit context must contain actual source frames")
            key = movie, frame
            if key not in frame_cache:
                fs = []
                for fid in fids:
                    result = reader.read(fid)
                    if not result.exact:
                        raise ValueError("source-frame verification failed")
                    fs.append(cv2.cvtColor(result.frame, cv2.COLOR_BGR2GRAY))
                frame_cache[key] = np.stack(fs)
            frames = frame_cache[key]
            origin = np.floor(np.asarray(case["centre"]) - STORED_TILE / 2).astype(int)
            arr = extract_tile(frames, origin, STORED_TILE)
            t = index.spatial(movie, frame, origin, (STORED_TILE, STORED_TILE))
            yy, xx = np.mgrid[:STORED_TILE, :STORED_TILE]
            if case.get('kind') == 'negative_spot':
                t['negative'], focus_audit = scoped_negative_spot(
                    t['negative'], origin, case['centre'], float(case.get('radius_px', 8.0)))
                case['negative_scope_audit'] = focus_audit
            h, w = frames.shape[-2:]
            native_margin = 32 if args.normalization == 'pixel' else VALID_MARGIN
            real = ((xx + origin[0] >= native_margin) & (xx + origin[0] < w - native_margin)
                    & (yy + origin[1] >= native_margin) & (yy + origin[1] < h - native_margin))
            t["positive"] &= real
            t["negative"] &= real
            case.update(origin=origin.tolist(), image_size=[w, h], source_frames=fids,
                        positive_pixels=int(t["positive"].sum()), negative_pixels=int(t["negative"].sum()))
            case['supervised_points'] = [copy.deepcopy(p) for p in index.points.get(key, [])
                if all(origin[i]-6 <= p['xy'][i] <= origin[i]+STORED_TILE+6 for i in (0,1))]
            for p in index.points.get(key, []):
                if all(origin[i] - 6 <= p["xy"][i] <= origin[i] + STORED_TILE + 6 for i in (0, 1)):
                    observed_owners.add(movie + "|" + p["owner"])
            pixels.append(arr)
            positives.append(t["positive"])
            negatives.append(t["negative"])
            offsets.append(t["offset_xy"].transpose(2, 0, 1))
            ax = axes.flat[j]
            ax.imshow(arr[QUERY_INDEX], cmap="gray", vmin=0, vmax=255)
            overlay = np.zeros((STORED_TILE, STORED_TILE, 4))
            overlay[t["positive"]] = [0, 1, 0, 0.4]
            overlay[t["negative"]] = [1, 0.5, 0, 0.25]
            ax.imshow(overlay)
            ax.set_title(f"{case['id']} / {movie}:{frame}\n+{case['positive_pixels']} -{case['negative_pixels']}", fontsize=8)
            ax.axis("off")
        for ax in axes.flat[len(cases):]:
            ax.axis("off")
        fig.suptitle("Native query images: green = all known caps; orange = reviewed non-cap; elsewhere unknown")
        fig.tight_layout()
        fig.savefig(out / "input-target-sheet.png", dpi=140)
        plt.close(fig)
    finally:
        for reader in readers.values():
            if reader:
                reader.close()
    np.savez_compressed(out / "panel.npz", pixels=np.stack(pixels), positive=np.stack(positives),
                        negative=np.stack(negatives), offset=np.stack(offsets))
    manifest = {"schema": "tubetracker.native_cap_panel.v1", "snapshot": str(snap),
                 "hard_negative_spots": spot_audit,
                 "hard_negative_scope_policy": "reviewed_negative_intersection_v1",
                "snapshot_hashes": {p.name: file_hash(p) for p in snap.glob("*.json")},
                "label_scope": index.summary(), "cases": cases,
                "dataset_sha256": file_hash(out / "panel.npz"),
                "supervised_owner_aliases": sorted(observed_owners),
                "scope": "six familiar distal-end fit crops plus explicit local noncap regions; no generalization claim",
                "input_offsets": list(OFFSETS), "query_index": QUERY_INDEX,
                "positive_radius_px": 6, "negative_exclusion_px": 10,
                "new_interval_observations_reserved_for_checks": True}
    manifest['frozen_utc'] = datetime.now(timezone.utc).isoformat()
    manifest['native_valid_margin_px'] = native_margin
    if membership is not None:
        manifest.update(scope='broader reviewed development corpus; geometric sampling groups, not independent grains; no whole-movie accuracy claim',
                        membership=membership, selection_metric='proposal_recall',
                        new_interval_observations_reserved_for_checks=None)
    dump(out / "panel.json", manifest)
    print(json.dumps({"panel": str(out), "cases": len(cases), "labels": index.summary()}), flush=True)
    return 0


def negative_response_counts(out, negative, origin, *, threshold=.5):
    """Keep raw rejection separate from refined detections leaving reviewed scope."""
    import cv2
    probability = out["probability"]
    valid = out["valid"]
    yy, xx = np.mgrid[:probability.shape[0], :probability.shape[1]]
    locations = np.stack([xx + origin[0], yy + origin[1]], -1)
    from prototypes.v30_video_apex.cap_evidence import emit_cap_candidates
    peaks = valid & (probability >= cv2.dilate(probability, np.ones((7, 7), np.uint8)))
    raw = emit_cap_candidates(probability, locations, peaks, threshold=threshold)
    def inside(xy):
        q = np.rint(np.asarray(xy) - origin).astype(int)
        return (0 <= q[0] < negative.shape[1] and 0 <= q[1] < negative.shape[0]
                and bool(negative[q[1], q[0]]))
    raw_negative = [c for c in raw if inside(c["tip_xy"])]
    refined_negative = [c for c in out["caps"] if inside(c["tip_xy"])]
    escapes = [c for c in out["caps"]
               if inside(c.get("unrefined_xy", c["location_xy"])) and not inside(c["tip_xy"])]
    return {"raw_negative_peaks": len(raw_negative),
            "licensed_negative_peaks": len(refined_negative),
            "negative_to_unknown_escapes": len(escapes),
            "licensed_negative_peak_locations": [{"xy": [round(float(v), 2) for v in c["tip_xy"]],
                "probability": round(float(c["probability"]), 4)} for c in refined_negative],
            "raw_negative_peak_locations": [{"xy": [round(float(v), 2) for v in c["tip_xy"]],
                "probability": round(float(c["probability"]), 4)} for c in raw_negative],
            "negative_escape_examples": [{"unrefined_xy": c.get("unrefined_xy", c["location_xy"]),
                "tip_xy": c["tip_xy"], "probability": c["probability"]} for c in escapes]}


def evaluate(model, arrays, manifest, *, case_indices=None, render_path=None):
    cases = manifest["cases"]
    indices = list(range(len(cases))) if case_indices is None else list(case_indices)
    rows, images = [], []
    for i in indices:
        case = cases[i]
        clip = arrays["pixels"][i, :, PAD:PAD + TILE, PAD:PAD + TILE]
        origin = np.asarray(case["origin"]) + PAD
        out = score_native_tile(model, clip, origin=origin, image_size=case["image_size"],
                                movie=case["movie"], source_frame=case["frame"])
        top = out["caps"][0] if out["caps"] else None
        neg = arrays["negative"][i, PAD:PAD + TILE, PAD:PAD + TILE] & out["valid"]
        rejection = negative_response_counts(out, neg, origin)
        err, nearest_error = None, None
        if case["kind"] == "positive" and top:
            err = float(np.linalg.norm(np.asarray(top["tip_xy"]) - case["truth"]["xy"]))
            nearest_error = min(float(np.linalg.norm(np.asarray(c['tip_xy'])-case['truth']['xy'])) for c in out['caps'])
        rows.append({"id": case["id"], "kind": case["kind"], "top1": top,
                     "top1_error_px": err, "within_5px": err is not None and err <= 5,
                     "nearest_proposal_error_px": nearest_error,
                     "proposal_within_5px": nearest_error is not None and nearest_error <= 5,
                     **rejection, "negative_pixels": int(neg.sum()),
                     "negative_max_probability": float(out["probability"][neg].max()) if neg.any() else None,
                     "emitted": len(out["caps"]), "input_sha256": out["input_sha256"]})
        images.append((clip[QUERY_INDEX], origin, case, out))
    positive = [r for r in rows if r["kind"] == "positive"]
    result = {"scope": manifest["scope"], "threshold": 0.5,
              "n_positive_cases": len(positive), "selected_tips_within_5px": sum(r["within_5px"] for r in positive),
              "target_proposals_within_5px": sum(r['proposal_within_5px'] for r in positive),
               **{k: sum(r[k] for r in rows) for k in (
                   "licensed_negative_peaks", "raw_negative_peaks", "negative_to_unknown_escapes")},
               "cases": rows}
    result['selection_metric'] = manifest.get('selection_metric', 'top1')
    matched = result['target_proposals_within_5px'] if result['selection_metric'] == 'proposal_recall' else result['selected_tips_within_5px']
    result["output_scope_gate_passed"] = (bool(positive) and matched == len(positive)
                                   and result["licensed_negative_peaks"] == 0
                                   and sum(r["negative_pixels"] for r in rows) > 0)
    result["fit_gate_passed"] = result["output_scope_gate_passed"] and result["raw_negative_peaks"] == 0
    result["rejection_rule"] = "no raw native peaks or refined detections in reviewed negatives; outside remains unknown"
    if render_path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots((len(images) + 3) // 4, 4, figsize=(12, 3 * ((len(images) + 3) // 4)), squeeze=False)
        for j, (im, origin, case, out) in enumerate(images):
            ax = axes.flat[j]
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            for k, c in enumerate(out["caps"][:20]):
                xy = np.asarray(c["tip_xy"]) - origin
                ax.plot(*xy, "x", color="red" if k else "cyan", markersize=5)
            if case["kind"] == "positive":
                xy = np.asarray(case["truth"]["xy"]) - origin
                ax.plot(*xy, "o", markerfacecolor="none", markeredgecolor="lime", markersize=9)
            row = rows[j]
            ax.set_title(f"{case['id']}\nerror={row['top1_error_px']} neg-peaks={row['licensed_negative_peaks']}", fontsize=7)
            ax.axis("off")
        for ax in axes.flat[len(images):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(render_path, dpi=140)
        plt.close(fig)
    return result


def read_panel(path):
    panel = Path(path).resolve()
    manifest = json.loads((panel / "panel.json").read_text())
    if file_hash(panel / "panel.npz") != manifest["dataset_sha256"]:
        raise ValueError("panel changed after preparation")
    with np.load(panel / "panel.npz", allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    if any(len(v) != len(manifest["cases"]) for v in arrays.values()):
        raise ValueError("panel cases and arrays disagree")
    return panel, manifest, arrays


def evaluate_checkpoint(args):
    """Re-evaluate fixed weights without creating an optimizer or a checkpoint."""
    import torch
    torch.set_num_threads(2)
    if not args.checkpoint:
        raise ValueError("eval requires --checkpoint")
    panel, manifest, arrays = read_panel(args.panel)
    model, checkpoint = load_native_checkpoint(args.checkpoint, args.device)
    training = checkpoint["manifest"].get("inputs", {})
    inputs = {"checkpoint": checkpoint["checkpoint"], "checkpoint_sha256": checkpoint["sha256"],
              "panel": str(panel), "panel_sha256": file_hash(panel / "panel.json"),
              "dataset_sha256": manifest["dataset_sha256"],
              "source": {p: file_hash(ROOT / p) for p in ["scripts/train_native_caps.py",
                  "prototypes/v30_video_apex/native_caps.py", "prototypes/v30_video_apex/cap_evidence.py"]}}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    report = evaluate(model, arrays, manifest, render_path=out / "predictions.png")
    if file_hash(args.checkpoint) != checkpoint["sha256"]:
        raise RuntimeError("checkpoint changed during evaluation")
    report.update(checkpoint=checkpoint["checkpoint"], inputs=inputs,
                  evaluated_training_panel=all(inputs[k] == training.get(k)
                                               for k in ("panel_sha256", "dataset_sha256")))
    dump(out / "evaluation.json", report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("cases", "inputs")}), flush=True)
    return 0


def sample_cap_batch(rng, positives, negatives, focused=()):
    """Keep positive/ordinary replay while actually exposing licensed hard spots."""
    if not positives or not negatives:
        raise ValueError('cap fitting requires positive and reviewed-negative cases')
    ids = list(rng.choice(positives, 2))
    if len(focused):
        ids += list(rng.choice(negatives, 1)) + list(rng.choice(focused, 1))
    else:
        ids += list(rng.choice(negatives, 2))
    return ids


def fit(args):
    import torch
    from prototypes.v30_video_apex.model_factory import parameter_hash
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    panel, manifest, arrays = read_panel(args.panel)
    if (any(c['kind'] == 'negative_spot' for c in manifest['cases'])
            and manifest.get('hard_negative_scope_policy') != 'reviewed_negative_intersection_v1'):
        raise ValueError('Rebuild hard-negative panel: focus disks must stay inside reviewed negative masks')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    initial = None
    if getattr(args, 'init', None):
        initial, _ = load_native_checkpoint(args.init)
    if getattr(args, 'image_gain', None) is None:
        # rev14 P1: never silently train/continue with a different input
        # gain than the checkpoint: inherit it unless explicitly overridden
        # (the effective value is recorded in the manifest).
        args.image_gain = float(getattr(initial, 'image_gain', 1.0)) if initial else 1.0
    model = NativeCapNet(base=args.base, temporal=args.temporal, normalization=args.normalization,
                        image_gain=args.image_gain)
    if args.init:
        previous, _ = load_native_checkpoint(args.init)
        model.load_state_dict(previous.state_dict(), strict=True)
    initial_hash = parameter_hash(model)
    config = {"seed": args.seed, "updates": args.updates, "lr": args.lr,
              "batch_size": 4, "augmentation": "translated crop and dihedral geometry" if not args.canary else "fixed inputs",
              "negative_sampling": getattr(args, 'negative_sampling', 'focused'),
              "sampling_policy": "two_positive_one_region_one_licensed_spot_v1; two regions when no eligible spot, legacy control or canary",
              "loss": {"positive_bce": 1, "negative_bce": 1, "offset_smooth_l1": .25, "offset_nll": .01,
                       "hard_negative_weight": args.hard_negative_weight, "hard_negative_k": args.hard_negative_k}}
    inputs = {"panel": str(panel.resolve()), "panel_sha256": file_hash(panel / "panel.json"),
              "dataset_sha256": manifest["dataset_sha256"], "model": model.config(),
              "initial_parameter_hash": initial_hash, "training": config,
              "initial_checkpoint": str(Path(args.init).resolve()) if args.init else None,
              "initial_checkpoint_sha256": file_hash(args.init) if args.init else None,
              "source": {p: file_hash(ROOT / p) for p in ["scripts/train_native_caps.py",
                         "prototypes/v30_video_apex/native_caps.py", "prototypes/v30_video_apex/cap_evidence.py"]}}
    save_native_checkpoint(out / "init.pt", model, manifest={"inputs": inputs})
    model.to(args.device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    pos = [i for i, c in enumerate(manifest["cases"]) if c["kind"] == "positive"]
    neg = [i for i, c in enumerate(manifest["cases"]) if c["kind"] == "negative_region"]
    focused = [i for i, c in enumerate(manifest['cases'])
               if c['kind'] == 'negative_spot' and c.get('negative_pixels', 0) > 0]
    if getattr(args, 'negative_sampling', 'focused') == 'legacy':
        focused = []
    if args.canary:
        pos, neg = pos[:1], neg[:1]
        focused = []
    exposure = {str(i): {'id':c['id'], 'kind':c['kind'], 'draws':0, 'positive_pixels':0, 'negative_pixels':0}
                for i,c in enumerate(manifest['cases'])}
    history, mass = [], {"positive": 0, "negative": 0}
    started = time.monotonic()
    best_rank, best = None, None
    for step in range(1, args.updates + 1):
        ids = sample_cap_batch(rng, pos, neg, focused)
        batch = {k: [] for k in ("pixels", "positive", "negative", "offset")}
        for i in ids:
            dx, dy = (PAD, PAD) if args.canary else rng.integers(0, 2 * PAD + 1, 2)
            sample = {k: v[i, ..., dy:dy + TILE, dx:dx + TILE].copy() for k, v in arrays.items()}
            if not args.canary:
                rotation = int(rng.integers(0, 4))
                for k in sample:
                    sample[k] = np.rot90(sample[k], rotation, axes=(-2, -1)).copy()
                for _ in range(rotation):
                    a, b = sample["offset"][0].copy(), sample["offset"][1].copy()
                    sample["offset"][0], sample["offset"][1] = b, -a
                if rng.random() < .5:
                    for k in sample:
                        sample[k] = sample[k][..., ::-1].copy()
                    sample["offset"][0] *= -1
            for k in batch:
                batch[k].append(sample[k])
        x = torch.as_tensor(np.stack(batch["pixels"]), device=args.device, dtype=torch.float32)[:, :, None] / 255
        pm = torch.as_tensor(np.stack(batch["positive"]), device=args.device)[:, None]
        nm = torch.as_tensor(np.stack(batch["negative"]), device=args.device)[:, None]
        for mask in (pm, nm):
            margin = model.valid_margin
            mask[:, :, :margin] = mask[:, :, -margin:] = False
            mask[:, :, :, :margin] = mask[:, :, :, -margin:] = False
        positive_counts = pm.sum(dim=(1,2,3)).cpu().tolist()
        negative_counts = nm.sum(dim=(1,2,3)).cpu().tolist()
        for i, positive_count, negative_count in zip(ids, positive_counts, negative_counts):
            record = exposure[str(i)]
            record['draws'] += 1
            record['positive_pixels'] += int(positive_count)
            record['negative_pixels'] += int(negative_count)
        ot = torch.as_tensor(np.stack(batch["offset"]), device=args.device, dtype=torch.float32)
        pred = model(x)
        loss, terms = native_cap_loss(pred, pm, nm, ot,
            hard_negative_weight=args.hard_negative_weight, hard_negative_k=args.hard_negative_k)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = {name: float(head.weight.grad.norm()) for name, head in (("cap", model.cap), ("offset", model.offset), ("variance", model.logvar))}
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        opt.step()
        mass["positive"] += terms["positive_mass"]
        mass["negative"] += terms["negative_mass"]
        if step == 1 or step % args.block == 0 or step == args.updates:
            model.eval()
            report = evaluate(model, arrays, manifest, case_indices=pos + neg if args.canary else None)
            model.train()
            row = {"update": step, "loss": float(loss.detach()), "terms": terms,
                   "gradient_norms": grad, "selected_within_5": report["selected_tips_within_5px"],
                   "negative_peaks": report["licensed_negative_peaks"],
                   "raw_negative_peaks": report["raw_negative_peaks"], "elapsed_s": time.monotonic() - started}
            history.append(row)
            print(json.dumps(row), flush=True)
            metric = 'nearest_proposal_error_px' if manifest.get('selection_metric') == 'proposal_recall' else 'top1_error_px'
            errors = [r[metric] if r[metric] is not None else 1000 for r in report["cases"] if r["kind"] == "positive"]
            matched = report['target_proposals_within_5px'] if manifest.get('selection_metric') == 'proposal_recall' else report['selected_tips_within_5px']
            rank = (matched,
                    -report["raw_negative_peaks"] - report["licensed_negative_peaks"], -float(np.mean(errors)))
            ck = out / f"step-{step:04d}.pt"
            save_native_checkpoint(ck, model, manifest={"inputs": inputs, "update": step})
            dump(out / f"step-{step:04d}.eval.json", report)
            if best_rank is None or rank > best_rank:
                best_rank, best = rank, str(ck)
            dump(out / "run.json", {"manifest_schema": "tubetracker.run.v1", "inputs": inputs,
                                    "results": {"history": history, "loss_mass": mass,
                                                "case_exposure": exposure, "best_checkpoint": best}})
    chosen, _ = load_native_checkpoint(best, args.device)
    result = evaluate(chosen, arrays, manifest, case_indices=pos + neg if args.canary else None,
                      render_path=out / "predictions.png")
    result["checkpoint"] = best
    dump(out / "evaluation.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}), flush=True)
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["prepare", "fit", "eval"])
    p.add_argument("--snapshot", default="runs/prototypes/v30/snap26_rev14")
    p.add_argument("--panel", default="runs/prototypes/v30/rev14_native_panel")
    p.add_argument("--positive-source", choices=['six-review', 'reviewed-corpus'], default='six-review')
    p.add_argument("--fit-movies", default='', help='Explicit comma-separated movie IDs for reviewed-corpus preparation')
    p.add_argument("--reserve-interval", action='append', default=[], help='Exclude query/context frames in movie:first:last; repeatable')
    p.add_argument("--out", required=True)
    p.add_argument("--base", type=int, default=12)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--updates", type=int, default=600)
    p.add_argument("--block", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hard-negative-weight", type=float, default=0.)
    p.add_argument("--hard-negative-spots", default="",
                   help="JSON of MEASURED false peaks of a previous arm; prepare appends "
                        "zero-target disk cases at those locations (licenses nothing new)")
    p.add_argument("--hard-negative-k", type=int, default=32)
    p.add_argument("--negative-sampling", choices=['focused', 'legacy'], default='focused',
                   help="Expose one licensed hard spot per batch while replaying ordinary negatives; legacy is an explicit comparison")
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    p.add_argument("--init", default="")
    p.add_argument("--checkpoint", default="", help="Fixed checkpoint for eval; never trains or rewrites weights")
    p.add_argument("--canary", action="store_true")
    p.add_argument("--temporal", action="store_true")
    p.add_argument("--normalization", choices=["spatial", "pixel"], default="spatial")
    p.add_argument("--image-gain", type=float, default=None,
                   help="input contrast gain; default: inherit from --init (recorded)")
    a = p.parse_args()
    return {"prepare": prepare, "fit": fit, "eval": evaluate_checkpoint}[a.mode](a)


if __name__ == "__main__":
    raise SystemExit(main())
