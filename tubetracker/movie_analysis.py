"""The shared native movie workflow used by the annotation app and CLI."""
from __future__ import annotations

import copy
import csv
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from .analysis_dependencies import (FileFingerprinter, assert_runtime_code_current,
    dependency_fingerprint, require_same_dependencies)
from .analysis_contracts import (AnalysisRequest, measurement_rows,
    resolve_review_constraints, snapshot_review_records, stable_hash,
    store_review_records,
    snapshot_tombstones, route_model_bindings)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "tubetracker.movie_analysis.v1"


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix='.' + path.name + '-', suffix='.tmp', delete=False) as stream:
        tmp = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush(); os.fsync(stream.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _code_hashes(names):
    from prototypes.v30_video_apex.native_caps import file_hash
    return {n: file_hash(ROOT / n) for n in names}



def frame_root_evidence(constraints):
    """Frame-scoped verified roots from human FULL traces (rev14 W2).

    Returns {frame: {owner_id: evidence}}; a root is only present when the
    resolved constraint carries one (complete human geometry for that
    frame). Nothing is transported across frames here."""
    roots = {}
    for (owner_id, frame), c in (constraints or {}).items():
        vr = (c or {}).get("verified_root")
        if vr and int(vr.get("scope_frame", frame)) == int(frame):
            roots.setdefault(int(frame), {})[owner_id] = dict(vr)
    return roots


def route_owner_for(owner, frame_roots, frame):
    """The owner mapping used by the query/route step at one frame.

    A verified frame root overrides the static attachment for THAT frame
    only; other frames keep the static attachment unchanged, and the root
    evidence travels with the mapping so provenance can be recorded."""
    ev = ((frame_roots or {}).get(int(frame)) or {}).get(owner["id"])
    if not ev:
        return owner, None
    return ({**owner, "attachment_native": list(ev["xy"]),
             "attachment_verified": True,
             "attachment_source": {k: ev.get(k) for k in
                 ("observation_id", "revision", "basis", "scope_frame")}}, ev)


def body_root_for(owner):
    """Resolve a declared, verified root without promoting a guessed attachment."""
    if owner.get("attachment_verified") and owner.get("attachment_native") is not None:
        xy, source = owner["attachment_native"], owner.get("attachment_source")
    elif owner.get("root_verified") and owner.get("root_native") is not None:
        xy, source = owner["root_native"], owner.get("root_source")
    else:
        return None, None
    xy = np.asarray(xy, dtype=float)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError("verified body root must be a finite native point")
    return xy.tolist(), source or {"basis": "declared_verified_attachment"}


class MovieAnalysisService:
    def __init__(self, cap_checkpoint, *, body_checkpoint="", cache_dir,
                 device="cpu", pixel_provider=None, registry_path=None):
        self.cap_checkpoint = str(Path(cap_checkpoint).resolve())
        self.body_checkpoint = str(Path(body_checkpoint).resolve()) if body_checkpoint else ""
        self.cache_dir = Path(cache_dir)
        self.registry_path = Path(registry_path) if registry_path else self.cache_dir.parent/"grain_registry.sqlite"
        self.device = device
        self.pixel_provider = pixel_provider
        self._cap_model = None
        self._body_model = None
        self._body_kind = None
        self._loaded_model_hashes = {}
        self._grain_detector = None
        self._grain_detector_key = None
        self._fingerprinter = FileFingerprinter()

    def _discover(self, request, movie_hash):
        from .grain_detection import GrainDetector, GrainDetectionConfig
        from .annotation_frames import FrameReader
        import cv2
        reader = FrameReader(request.movie_path)
        try:
            read = reader.read(request.frames[0])
            if not read.exact:
                raise ValueError("inexact discovery frame")
            gray = cv2.cvtColor(read.frame, cv2.COLOR_BGR2GRAY)
        finally:
            reader.close()
        cfg = GrainDetectionConfig.from_dict(request.grain_detection)
        checkpoint = self._fingerprinter.file(cfg.checkpoint) if cfg.backend == 'cpdino' else None
        key = stable_hash({'config': cfg.to_dict(), 'checkpoint': checkpoint, 'device': self.device})
        if key != self._grain_detector_key:
            self._grain_detector = GrainDetector(cfg.to_dict(), device=self.device)
            self._grain_detector_key = key
        proposals, detector_report = self._grain_detector.detect(gray, request.roi_xyxy)
        detector_report['checkpoint'] = checkpoint
        from .grain_registry import DiscoveryRegistry
        owners, report = DiscoveryRegistry(self.registry_path).reconcile(
            request.movie_id, movie_hash, request.frames[0], request.owners, proposals)
        # Proximity defines a review group, never a many-grains-to-one merge.
        for i, owner in enumerate(owners):
            nearby = [o["id"] for j, o in enumerate(owners) if j != i
                      and np.linalg.norm(np.asarray(o["grain_native"]) - owner["grain_native"])
                      < float(o.get("grain_radius_px", 13)) + float(owner.get("grain_radius_px", 13)) + 3]
            if nearby:
                owner["clump_review_group"] = sorted([owner["id"]] + nearby)
        report.update(automatic_proposals=[o["id"] for o in owners
                      if o["id"] not in {seed["id"] for seed in request.owners}],
                       matched_existing=len(report["matched_existing_ids"]), missed_grains=None,
                       detector=detector_report, proposals=proposals)
        return owners, report

    def _load_models(self, expected=None):
        """Tie live model objects to the exact checkpoint bytes in the run key."""
        import torch
        from prototypes.v30_video_apex.native_caps import load_native_checkpoint
        from prototypes.v30_video_apex.native_body import load_body_checkpoint
        from prototypes.v30_video_apex.model_factory import build_model_from_checkpoint
        if not self.body_checkpoint:
            raise ValueError("whole-tube analysis requires an explicitly configured body checkpoint")
        paths = {"cap": self.cap_checkpoint, "body": self.body_checkpoint}
        actual = {k: self._fingerprinter.file(p).get("sha256") for k, p in paths.items()}
        if expected is not None:
            require_same_dependencies(expected, actual)
        for name, path in paths.items():
            if actual[name] is None:
                raise FileNotFoundError(path)
            if self._loaded_model_hashes.get(name) == actual[name]:
                continue
            if name == "cap":
                model, _ = load_native_checkpoint(path, self.device)
                kind = None
            else:
                header = torch.load(path, map_location="cpu", weights_only=False)
                if str(header.get("schema", "")).startswith("tubetracker.native_owned_body."):
                    model, _ = load_body_checkpoint(path, self.device)
                    kind = header["schema"]
                else:
                    model, _ = build_model_from_checkpoint(path)
                    model.to(self.device).eval()
                    kind = "strict-multiscale"
            require_same_dependencies(actual[name], self._fingerprinter.file(path).get("sha256"))
            setattr(self, "_" + name + "_model", model)
            self._loaded_model_hashes[name] = actual[name]
            if name == "body":
                self._body_kind = kind
        return actual

    def _body_pixel_contract(self, expected):
        # Providers have the existing two-argument, review-independent contract.
        if self.pixel_provider:
            return {"schema": "pixel_provider", "uses_root": False}
        self._load_models(expected)
        native = self._body_kind.startswith("tubetracker.native_owned_body.")
        return {"schema": self._body_kind,
                "uses_root": bool(self._body_model.root_channel) if native else True,
                "review_modes": "separate_body_arrays_v1"}

    def _pixels(self, request, owners, progress, array_store=None, model_hashes=None, frame_roots=None,
                grain_poses=None):
        if self.pixel_provider:
            return self.pixel_provider(request, owners)
        import cv2
        import torch
        from .annotation_frames import FrameReader
        from .frame_geometry import owner_at_frame
        from prototypes.v30_video_apex.native_caps import (
            detect_caps, OFFSETS)
        from prototypes.v30_video_apex.dataset import QUERY_OFFSETS
        from prototypes.v30_video_apex.inference import build_typed_prompt
        from prototypes.v30_video_apex.targets import load_clip_pixels
        from prototypes.v30_video_apex.native_body import predict_owned_body, predict_owned_body_region
        body_contract = self._body_pixel_contract(model_hashes)
        reader = FrameReader(request.movie_path)
        cache, evidence = {}, {}
        arrays = array_store if array_store is not None else {}
        width, height = reader.native_size
        if request.roi_xyxy and (request.roi_xyxy[2] > width or request.roi_xyxy[3] > height):
            reader.close()
            raise ValueError("ROI exceeds the native movie dimensions")

        class CachedReader:
            def read(self, frame):
                if frame not in cache:
                    value = reader.read(frame)
                    if not value.exact:
                        raise ValueError(f"inexact source frame {frame}")
                    cache[frame] = value
                return cache[frame]

        cached = CachedReader()
        try:
            for i, frame in enumerate(request.frames):
                if frame >= len(reader):
                    raise ValueError("requested source frame exceeds movie")
                fids = [frame + delta for delta in OFFSETS]
                # At the actual movie edge a temporal detector cannot claim
                # a complete context window. A single-frame model only uses
                # the query; repeated edge context is explicitly recorded.
                clipped = [min(max(f, 0), len(reader) - 1) for f in fids]
                if self._cap_model.temporal and clipped != fids:
                    raise ValueError("temporal cap model requires actual context frames; use single-frame at movie edges")
                frames = [cv2.cvtColor(cached.read(f).frame, cv2.COLOR_BGR2GRAY) for f in clipped]
                caps = detect_caps(self._cap_model, frames, movie=request.movie_id,
                                   source_frame=frame, roi=request.roi_xyxy,
                                   threshold=request.cap_threshold)
                def owned_pixels(query_owner, root_ev):
                    root_xy, root_source = body_root_for(query_owner) if body_contract["uses_root"] else (None, None)
                    # Native body prediction uses root_native; route generation
                    # uses attachment_native. Resolve that distinction explicitly.
                    query_owner = dict(query_owner, root_native=root_xy)
                    ox, oy = np.floor(np.asarray(query_owner["grain_native"]) - 144).astype(int)
                    extent = {'kind': 'grain_crop', 'shape': [288, 288],
                              'origin': [int(ox), int(oy)]}
                    if self._body_kind.startswith("tubetracker.native_owned_body."):
                        if request.body_extent == 'analysis_roi':
                            body, (ox, oy), extent = predict_owned_body_region(
                                self._body_model, frames[1], query_owner, request.roi_xyxy,
                                spatial_context=request.body_normalization_context, root_xy=root_xy)
                        else:
                            body, (ox, oy) = predict_owned_body(self._body_model, frames[1], query_owner, root_xy=root_xy)
                            extent.update(origin=[int(ox), int(oy)],
                                          crop_origin_lattice_px=self._body_model.pooling_lattice,
                                          normalization=self._body_model.normalization)
                        body_fids, missing = [frame], (0,)
                        provenance = {"query": "grain centre/radius; current native frame",
                                      "attachment_supplied": root_xy is not None,
                                      "root_channel": body_contract["uses_root"],
                                      "root_xy": root_xy, "root_source": root_source,
                                      "schema": self._body_kind}
                        # WO1-solver: scored model no-emergence evidence from the
                        # body presence head (posed tile + emergence ring pool).
                        # A logit is evidence, not a verdict; the solver
                        # decides against route candidates. No head => absent key.
                        presence_logit = None
                        if getattr(self._body_model, "presence_head", False):
                            from prototypes.v30_video_apex.native_body import body_input as _body_input
                            from prototypes.v30_video_apex.native_caps import extract_tile as _extract_tile
                            _tile = _extract_tile([frames[1]], (ox, oy), 288)[0]
                            _gx, _gy = (np.asarray(query_owner["grain_native"], float)
                                        - np.asarray([ox, oy], float))
                            _rr = float(query_owner.get("grain_radius_px", 13.0))
                            _yy, _xx = np.mgrid[:288, :288]
                            _ring = ((np.hypot(_xx - _gx, _yy - _gy) >= _rr)
                                     & (np.hypot(_xx - _gx, _yy - _gy) <= 2.5 * _rr))
                            _x = _body_input(_tile, np.asarray([ox, oy]),
                                             query_owner["grain_native"],
                                             query_owner.get("grain_radius_px", 13.0),
                                             image_gain=self._body_model.image_gain)
                            with torch.no_grad():
                                presence_logit = float(self._body_model.presence_logit(
                                    torch.from_numpy(_x)[None].to(self.device),
                                    torch.as_tensor(_ring, dtype=torch.bool,
                                                    device=self.device))[0].cpu())
                            provenance["presence_head"] = True
                            provenance["presence_pool"] = "emergence ring 1r..2.5r (posed query)"
                    else:
                        if request.body_extent != 'grain_crop':
                            raise ValueError('region body tiling requires a native owned-body checkpoint')
                        presence_logit = None  # non-native body models carry no presence head
                        body_fids = [frame + delta for delta in QUERY_OFFSETS]
                        missing = tuple(int(f < 0 or f >= len(reader)) for f in body_fids)
                        clip = load_clip_pixels(cached, body_fids, (ox, oy, 288, 288), missing)
                        kwargs = ({"human_grain_xy": query_owner["grain_native"], "human_grain_radius_px": query_owner.get("grain_radius_px", 13)}
                                  if query_owner.get("identity_verified", bool(query_owner.get("source_task"))) else
                                  {"detected_grain_xy": query_owner["grain_native"], "detected_radius_px": query_owner.get("grain_radius_px", 13)})
                        prompt, provenance = build_typed_prompt(query_owner["id"], crop_wh=(288, 288),
                            crop_origin=(ox, oy), attachment_xy=root_xy, **kwargs)
                        provenance.update(root_xy=root_xy, root_source=root_source)
                        with torch.no_grad():
                            pred = self._body_model(torch.from_numpy(clip)[None, :, None].to(self.device), prompt)
                        if pred.body is None:
                            raise ValueError("configured checkpoint has no body evidence")
                        body = torch.sigmoid(pred.body[0, 0]).cpu().numpy().astype(np.float32)
                    if query_owner.get('grain_pose'):
                        provenance['grain_pose'] = copy.deepcopy(query_owner['grain_pose'])
                    y, x = np.mgrid[:body.shape[0], :body.shape[1]]
                    body[(x+ox < 0) | (x+ox >= width) | (y+oy < 0) | (y+oy >= height)] = 0
                    key = f"f{frame}_" + stable_hash(query_owner["id"])[:12]
                    if root_ev is not None:
                        key += "_root_assisted"
                    arrays[key] = body
                    return {"body_key": key, "origin": [int(ox), int(oy)],
                            "prompt": provenance, "source_frames": body_fids,
                            "missing_frames": list(missing), 'extent': extent,
                            "attachment_source": root_ev, "presence_logit": presence_logit}

                def _event_evidence(base_owner, frame, cached, grain_poses, schedule):
                    """Baseline emergence score for one owner/frame (WO4).

                    Reference = median of the first two scheduled frames
                    (unreviewed; left-censor warning in provenance). The
                    scored tile uses the frame-specific pose, never the
                    root-shifted query owner.
                    """
                    from prototypes.v30_video_apex.event_evidence import (
                        EVENT_CONTRACT, emergence_score, score_from_parts)
                    from prototypes.v30_video_apex.native_caps import extract_tile as _et
                    posed_now = owner_at_frame(base_owner, frame, grain_poses)
                    grain = np.asarray(posed_now["grain_native"], float)
                    radius = float(posed_now.get("grain_radius_px", 13.0))
                    ox, oy = int(np.floor(grain[0] - 144)), int(np.floor(grain[1] - 144))
                    gray_now = cv2.cvtColor(cached.read(frame).frame, cv2.COLOR_BGR2GRAY)
                    tile_now = _et([gray_now], (ox, oy), 288)[0].astype(np.float32)
                    yy, xx = np.mgrid[:288, :288]
                    dist = np.hypot(xx - (grain[0] - ox), yy - (grain[1] - oy))
                    ref_frames = [int(f) for f in schedule[:2]]
                    ref_tiles = []
                    for rf in ref_frames:
                        posed_ref = owner_at_frame(base_owner, rf, grain_poses)
                        gr = np.asarray(posed_ref["grain_native"], float)
                        rr = float(posed_ref.get("grain_radius_px", 13.0))
                        oxr, oyr = int(np.floor(gr[0] - 144)), int(np.floor(gr[1] - 144))
                        gray_ref = cv2.cvtColor(cached.read(rf).frame, cv2.COLOR_BGR2GRAY)
                        ref_tiles.append(_et([gray_ref], (oxr, oyr), 288)[0].astype(np.float32))
                    ref = np.median(np.stack(ref_tiles), axis=0)
                    frac, noise, conc = emergence_score(tile_now, dist, radius, ref)
                    return {"event_score": score_from_parts(frac, conc),
                            "event_frac": frac, "event_concentration": conc,
                            "event_noise": noise,
                            "event_reference_frames": ref_frames,
                            "event_reference_basis": "unreviewed_earliest_schedule",
                            "event_left_censor_warning":
                                "grain emerged at schedule start would blind its own reference",
                            "event_contract": dict(EVENT_CONTRACT)}

                by_owner = {}
                for base_owner in owners:
                    owner = owner_at_frame(base_owner, frame, grain_poses)
                    info = owned_pixels(owner, None)
                    query_owner, root_ev = route_owner_for(owner, frame_roots, frame)
                    if body_contract["uses_root"] and root_ev is not None:
                        info["root_assisted"] = owned_pixels(query_owner, root_ev)
                    # WO4: installed event model on the shared posed clips.
                    info["event"] = _event_evidence(
                        base_owner, frame, cached, grain_poses, request.frames)
                    by_owner[owner["id"]] = info
                evidence[str(frame)] = {"caps": caps["caps"], "cap_tile_hashes": caps["tile_hashes"],
                                         "cap_source_frames": clipped, "owners": by_owner}
                progress({"stage": "pixel evidence", "completed": i + 1, "total": len(request.frames)})
                # Keep only the current neighbourhood in memory.
                for key in list(cache):
                    if key < frame - 10:
                        del cache[key]
        finally:
            reader.close()
        return {"frames": evidence, "image_size": [width, height], "body_schema": self._body_kind}, arrays

    def analyze(self, request, *, review_entities=(), review_source="", review_tombstones=(), progress=None):
        from .evidence_cache import BodyArrayStore
        from .population import (census_review_records, apply_reviewed_inventory,
                                 analysis_population, resolve_body_presence)
        from .seeded_body import prepare_body_assistance, apply_body_assistance
        from .frame_geometry import load_grain_poses, pose_index, owner_at_frame
        from .grain_identity import grain_identity_reviews
        from prototypes.v30_video_apex.native_caps import file_hash
        from prototypes.v30_video_apex.cap_evidence import emit_cap_candidates
        from prototypes.v30_video_apex.route_evidence import generate_owned_routes
        from prototypes.v30_video_apex.route_quality import (
            RoutePolicy, geometry_fingerprint, load_route_validation)
        from prototypes.v30_video_apex.state_solver import SolverConfig, solve_joint_states
        if isinstance(request, dict):
            request = AnalysisRequest(**request)
        request.validate()
        assert_runtime_code_current()
        dependency_config = {"request": request.to_dict(), "cap_checkpoint": self.cap_checkpoint,
                             "body_checkpoint": self.body_checkpoint}
        dependencies = dependency_fingerprint(dependency_config, self._fingerprinter)
        def check_dependencies():
            assert_runtime_code_current()
            require_same_dependencies(dependencies, dependency_fingerprint(dependency_config, self._fingerprinter))
        started = time.monotonic()
        progress = progress or (lambda _event: None)
        movie_hash = file_hash(request.movie_path)
        if request.discover_grains:
            owners, discovery = self._discover(request, movie_hash)
        else:
            owners, discovery = copy.deepcopy(request.owners), None
        census = census_review_records(review_entities, request.snapshot)
        owners, inventory_review = apply_reviewed_inventory(owners, census, movie=request.movie_id,
            roi_xyxy=request.roi_xyxy, selected_owner_ids=request.selected_owner_ids)
        if not owners and not request.discover_grains and not inventory_review["applied"]:
            raise ValueError("no grains to analyze; enable discovery or provide a reviewed inventory")
        records = snapshot_review_records(request.snapshot)
        records += store_review_records(review_entities, movie_id=request.movie_id, source=review_source)
        tombstones = list(snapshot_tombstones(request.snapshot)) + list(review_tombstones)
        identity_reviews, identity_audit = grain_identity_reviews(review_entities, request, owners,
            source=review_source, tombstones=tombstones)
        pose_path, motion_receipt = request.grain_pose_path, None
        if request.grain_motion:
            from .grain_motion import generate_grain_motion
            pose_path, motion_receipt = generate_grain_motion(request, owners, identity_reviews,
                movie_hash, self.cache_dir, progress=progress)
        grain_geometry = load_grain_poses(pose_path, request, owners, movie_hash,
            fingerprinter=self._fingerprinter, identity_reviews=identity_reviews)
        grain_poses = pose_index(grain_geometry)
        constraints, ignored = resolve_review_constraints(records, request, owners,
                                                          tombstones=tombstones)
        assistance = prepare_body_assistance(request, owners, review_entities,
            source=review_source, tombstones=tombstones, fingerprinter=self._fingerprinter,
            device=self.device)
        assisted_ids = {s['owner_id'] for s in assistance['seeds']} if assistance else set()
        frame_roots = frame_root_evidence(constraints)
        pixel_roots = {frame: {oid: ev for oid, ev in roots.items() if oid not in assisted_ids}
                       for frame, roots in frame_roots.items()}
        pixel_roots = {frame: roots for frame, roots in pixel_roots.items() if roots}
        model_hashes = {"cap": file_hash(self.cap_checkpoint),
                        "body": file_hash(self.body_checkpoint) if self.body_checkpoint else None}
        body_contract = self._body_pixel_contract(model_hashes)
        code_pixels = _code_hashes([
            "prototypes/v30_video_apex/native_caps.py", "prototypes/v30_video_apex/cap_evidence.py",
            "prototypes/v30_video_apex/native_body.py",
            "prototypes/v30_video_apex/model.py", "prototypes/v30_video_apex/model_factory.py",
            "prototypes/v30_video_apex/dataset.py", "prototypes/v30_video_apex/targets.py",
            "prototypes/v30_video_apex/inference.py", "tubetracker/annotation_frames.py",
            "tubetracker/movie_analysis.py", "tubetracker/evidence_cache.py",
            "tubetracker/analysis_dependencies.py",
            "tubetracker/seeded_body.py", "tubetracker/population.py",
            "tubetracker/review_semantics.py", "tubetracker/frame_geometry.py"])
        query_owners = [{"id": o["id"], "grain_native": o["grain_native"],
            "grain_radius_px": o.get("grain_radius_px", 13),
            "attachment_native": o.get("attachment_native"),
            "attachment_verified": bool(o.get("attachment_verified")),
            "body_root": body_root_for(o) if body_contract["uses_root"] else None,
            "identity_verified": o.get("identity_verified", bool(o.get("source_task")))} for o in owners]
        pixel_inputs = {"schema": SCHEMA, "movie_id": request.movie_id, "movie_sha256": movie_hash,
                        "frames": request.frames, "roi": request.roi_xyxy, "owners": query_owners,
                        "body_extent": request.body_extent,
                        "body_normalization_context": request.body_normalization_context,
                        "cap_checkpoint_sha256": model_hashes["cap"],
                        "body_checkpoint_sha256": model_hashes["body"],
                        "body_prompt_contract": body_contract,
                        "cap_threshold": request.cap_threshold, "device": self.device, "code": code_pixels,
                        "frame_roots": pixel_roots if body_contract["uses_root"] else None,
                        "pixel_provider": (f"{self.pixel_provider.__module__}.{self.pixel_provider.__qualname__}"
                                            if self.pixel_provider else "native_models")}
        if assistance:
            pixel_inputs['body_assistance'] = assistance
        if grain_geometry:
            pixel_inputs['grain_geometry'] = grain_geometry
        pixel_key = stable_hash(pixel_inputs)
        route_policy = RoutePolicy(**request.route_policy)
        route_validation = load_route_validation(request.route_validation,
            model_hashes=route_model_bindings(pixel_inputs),
            geometry_hash=geometry_fingerprint(owners, request.roi_xyxy, request.body_extent,
                                              request.body_normalization_context,
                                              grain_geometry=grain_geometry),
            movie_sha256=movie_hash, policy=route_policy, review_records=records)
        solver_config = SolverConfig(**request.solver)
        corrections = [{"owner": k[0], "frame": k[1], **v} for k, v in sorted(constraints.items())]
        inference_inputs = {"pixel_key": pixel_key, "corrections": corrections,
                            'grain_geometry_complete': {o['id']: o.get('grain_geometry_complete', True) for o in owners},
                            'route_policy': vars(route_policy),
                            'route_validation_sha256': route_validation['validation_sha256'] if route_validation else None,
                            "solver": vars(solver_config),
                            "code": _code_hashes(["prototypes/v30_video_apex/route_evidence.py",
                                    'prototypes/v30_video_apex/route_quality.py',
                                    "prototypes/v30_video_apex/state_solver.py",
                                    "tubetracker/analysis_contracts.py", "tubetracker/review_semantics.py",
                                    "tubetracker/movie_analysis.py"])}
        inference_key = stable_hash(inference_inputs)
        measurement_key = stable_hash({"inference_key": inference_key, "acquisition": request.acquisition,
                                      "clock_code": _code_hashes(["tubetracker/acquisition.py"])})
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        pixel_json = self.cache_dir / f"pixels-{pixel_key}.json"
        pixel_arrays = self.cache_dir / f"pixels-{pixel_key}.arrays"
        inferred_path = self.cache_dir / f"inference-{inference_key}.json"
        measured_path = self.cache_dir / f"measurements-{measurement_key}.json"
        cache_report = {"pixel_hit": pixel_json.exists() and pixel_arrays.is_dir(),
                        "inference_hit": inferred_path.exists(), "measurement_hit": measured_path.exists(),
                        "pixel_key": pixel_key, "inference_key": inference_key, "measurement_key": measurement_key}
        if measured_path.exists():
            result = json.loads(measured_path.read_text())
        else:
            if inferred_path.exists():
                inferred = json.loads(inferred_path.read_text())
            else:
                if cache_report["pixel_hit"]:
                    pixels = json.loads(pixel_json.read_text())
                    if pixels["body_arrays"]["schema"] != BodyArrayStore.schema:
                        raise ValueError("unsupported body evidence cache schema")
                    arrays = BodyArrayStore(pixel_arrays, pixels["body_arrays"]["entries"])
                else:
                    arrays = BodyArrayStore(pixel_arrays)
                    pixels, produced = self._pixels(request, owners, progress, arrays, model_hashes={
                        "cap": pixel_inputs["cap_checkpoint_sha256"],
                        "body": pixel_inputs["body_checkpoint_sha256"]},
                        frame_roots=pixel_roots, grain_poses=grain_poses)
                    check_dependencies()
                    if produced is not arrays:
                        for key, value in produced.items():
                            arrays[key] = value
                        del produced
                    if grain_geometry and list(pixels['image_size']) != grain_geometry['image_size']:
                        raise ValueError('grain pose dimensions differ from the actual source movie')
                    apply_body_assistance(request, pixels, arrays, assistance,
                                          device=self.device, progress=progress)
                    check_dependencies()
                    pixels["body_arrays"] = arrays.manifest()
                    pixels["inputs"] = pixel_inputs
                    _write_json(pixel_json, pixels)
                # rev14 P1: three declared evaluation modes --
                #   baseline      : declared grain inputs only, no review geometry
                #   root-assisted : frame-scoped reviewed roots, no constraints
                #   corrected     : roots + solver constraints (human-constrained)
                original = {o["id"]: {} for o in owners}
                rooted = {o["id"]: {} for o in owners}
                corrected = {o["id"]: {} for o in owners}
                for frame in request.frames:
                    progress({"stage": "owned routes", "completed": request.frames.index(frame),
                              "total": len(request.frames)})
                    evidence = pixels["frames"][str(frame)]
                    current_owners = [owner_at_frame(o, frame, grain_poses) for o in owners]
                    pool = evidence["caps"]
                    augmented = copy.deepcopy(pool)
                    for (oid, fid), constraint in constraints.items():
                        if fid == frame and constraint.get("tip_xy") is not None:
                            tip = constraint["tip_xy"]
                            nearest = min(augmented, key=lambda c: np.linalg.norm(np.asarray(c["tip_xy"]) - tip), default=None)
                            if nearest is None or np.linalg.norm(np.asarray(nearest["tip_xy"]) - tip) > 5:
                                augmented += emit_cap_candidates([1], [tip], [True], movie=request.movie_id, source_frame=frame)
                    for owner in current_owners:
                        info = evidence["owners"][owner["id"]]
                        body = arrays[info["body_key"]]
                        assisted_info = info.get("root_assisted", info)
                        assisted_body = arrays[assisted_info["body_key"]]
                        route_owner, _root_ev = route_owner_for(owner, frame_roots, frame)
                        original[owner["id"]][frame] = generate_owned_routes(
                            body, info["origin"], owner, pool, image_size=pixels["image_size"],
                            other_owners=current_owners, policy=route_policy, validation=route_validation,
                            movie=request.movie_id, source_frame=frame, body_evidence=info.get('prompt'))
                        rooted[owner["id"]][frame] = (
                            generate_owned_routes(assisted_body, assisted_info["origin"], route_owner, pool,
                                image_size=pixels["image_size"], other_owners=current_owners,
                                policy=route_policy, validation=route_validation,
                                movie=request.movie_id, source_frame=frame,
                                body_evidence=assisted_info.get('prompt'))
                            if route_owner is not owner else original[owner["id"]][frame])
                        corrected[owner["id"]][frame] = (
                            generate_owned_routes(assisted_body, assisted_info["origin"], route_owner, augmented,
                                image_size=pixels["image_size"], other_owners=current_owners,
                                policy=route_policy, validation=route_validation, movie=request.movie_id,
                                source_frame=frame, body_evidence=assisted_info.get('prompt'))
                            if len(augmented) != len(pool) else rooted[owner["id"]][frame])
                        # WO1-solver: the frame's scored no-emergence evidence
                        # rides on every candidate; the solver decides.
                        for _cand_list in (original[owner["id"]][frame],
                                           rooted[owner["id"]][frame],
                                           corrected[owner["id"]][frame]):
                            for _cand in (_cand_list or []):
                                if isinstance(_cand, dict) and _cand.get("presence_logit") is None:
                                    _cand["presence_logit"] = info.get("presence_logit")
                                # WO4: installed event-model evidence rides along.
                                if isinstance(_cand, dict) and _cand.get("event_score") is None:
                                    _cand["event_score"] = (info.get("event") or {}).get("event_score")
                progress({"stage": "joint selection", "completed": 0, "total": 2})
                baseline = solve_joint_states(original, request.frames, movie=request.movie_id, config=solver_config)
                root_assisted = (solve_joint_states(rooted, request.frames, movie=request.movie_id,
                                  config=solver_config) if frame_roots else baseline)
                inferred = (solve_joint_states(corrected, request.frames, constraints=constraints,
                            movie=request.movie_id, config=solver_config) if constraints else root_assisted)
                if grain_geometry:
                    reference_owners = {o['id']: o for o in owners}
                    for mode_rows in (baseline['rows'], root_assisted['rows'], inferred['rows']):
                        for row in mode_rows:
                            current_owner = owner_at_frame(reference_owners[row['owner_id']],
                                                           row['source_frame'], grain_poses)
                            row['grain_pose'] = dict(current_owner['grain_pose'],
                                geometry_identity=grain_geometry['identity'],
                                provider=grain_geometry['provider'])
                inferred.update(model_without_reviews=baseline["rows"],
                                root_assisted=root_assisted["rows"],
                                modes={"baseline": (
                                           "reviewed-mask-assisted model body proposals; no frame-review roots or solver constraints"
                                           if assistance else
                                           "declared grain inputs and verified static attachments; no frame-review roots or constraints"),
                                       "root_assisted": "frame-scoped reviewed roots; no solver constraints",
                                       "corrected": "reviewed roots plus solver constraints (human-constrained)"},
                                inputs=inference_inputs,
                                body_assistance_receipt=pixels.get('body_assistance'),
                                grain_geometry=grain_geometry,
                                image_size=pixels["image_size"], owners=owners,
                                discovery=discovery, ignored_reviews=ignored)
                inferred["review_comparison"] = [
                    {"owner_id": a["owner_id"], "source_frame": a["source_frame"],
                     "changed": any(a.get(k) != b.get(k) for k in ("state", "tip_xy", "current_path_xy")),
                     "human_constraint": bool(b.get("as_inference_constraint"))}
                    for a, b in zip(baseline["rows"], inferred["rows"])]
                check_dependencies()
                _write_json(inferred_path, inferred)
            rows = measurement_rows(inferred["rows"], request.acquisition, first_frame=request.frames[0])
            result = {"schema": SCHEMA, "movie_id": request.movie_id,
                      "movie_path": str(Path(request.movie_path).resolve()), "movie_sha256": movie_hash,
                      "request": request.to_dict(), "owners": owners, "rows": rows,
                      "model_without_reviews": inferred["model_without_reviews"],
                      "root_assisted": inferred.get("root_assisted"),
                      "modes": inferred.get("modes"),
                      "body_assistance_receipt": inferred.get('body_assistance_receipt'),
                      "grain_geometry": inferred.get('grain_geometry'),
                      "discovery": discovery, "ignored_reviews": ignored,
                      "review_comparison": inferred.get("review_comparison", []),
                      "inference": {k: inferred.get(k) for k in ["objective", "n_transitions_evaluated",
                                    "n_assignment_expansions", "components", "component_search",
                                    "candidate_pool_truncated", "assignment_search_truncated",
                                    "temporal_search_truncated", "search_truncated", "beam_width",
                                    "score_note", "memory_note", "config"]},
                      "inputs": {"pixel": pixel_inputs, "inference": inference_inputs, "acquisition": request.acquisition},
                      "image_size": inferred["image_size"],
                      "measurement_note": "full length requires a current cap and a certified complete owned path; missing acquisition metadata leaves physical rates null"}
            check_dependencies()
            _write_json(measured_path, result)
        result["cache"] = cache_report
        result["elapsed_wall_s"] = time.monotonic() - started
        result["artifacts"] = {"pixels": str(pixel_json), "inference": str(inferred_path),
                               "measurements": str(measured_path)}
        # Population is rebuilt from current census revisions even when pixel/route caches are reusable.
        body_presence, mask_audit = resolve_body_presence(review_entities, request, owners,
            source=review_source, tombstones=tombstones, image_size=result['image_size'])
        from tubetracker.population import germination_event_records
        result["population"] = analysis_population(owners, result["rows"], constraints, census,
            movie=request.movie_id, scope=request.census_scope, acquisition=request.acquisition,
            model_rows=result.get('model_without_reviews', []), body_presence=body_presence,
            selected_owner_ids=request.selected_owner_ids,
            germination_events=germination_event_records(review_entities))
        result['population']['body_presence_audit'] = mask_audit
        if assistance:
            result['population']['model_body_assistance'] = {
                'identity': assistance['identity'], 'classification': assistance['classification'],
                'seeds': [{k: s[k] for k in ('owner_id', 'frame', 'source_id', 'revision', 'source_project')}
                          for s in assistance['seeds']]}
        result["inventory_review"] = inventory_review
        result['grain_identity_review_audit'] = identity_audit
        result['grain_motion_receipt'] = motion_receipt
        population_inputs = {"inference_key": inference_key, "scope": request.census_scope,
            'selected_owner_ids': request.selected_owner_ids,
            "census": census, "body_presence": body_presence, 'body_presence_audit': mask_audit,
            "acquisition": request.acquisition,
            "code": _code_hashes(["tubetracker/population.py", "prototypes/v30_video_apex/census.py",
                                  "tubetracker/acquisition.py"])}
        population_key = stable_hash(population_inputs)
        result["cache"].update(population_key=population_key,
            export_key=stable_hash([measurement_key, population_key]))
        result["inputs"]["population"] = population_inputs
        result["request"] = request.to_dict()
        result["owners"] = owners
        result["discovery"] = discovery
        check_dependencies()
        result["dependencies"] = dependencies
        result["elapsed_wall_s"] = time.monotonic() - started
        return result


def export_analysis(result, directory):
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": str(out / "analysis.json"), "csv": str(out / "measurements.csv")}
    _write_json(out / "analysis.json", result)
    if "population" in result:
        _write_json(out / "population.json", result["population"])
        grain_fields = ['grain_id', 'movie', 'grain_x', 'grain_y', 'membership_resolved',
            'biological_class', 'classification_rule', 'classification_source', 'classification_revision',
            'timing_status', 'onset_after_frame', 'onset_by_frame', 'onset_after_time_s', 'onset_by_time_s',
            'timing_source', 'complete_measurement_count', 'last_complete_length_frame',
            'last_complete_length_px', 'last_complete_length_um', 'last_complete_length_provenance',
            'last_complete_length_certificate', 'latest_observation_frame', 'latest_observation_state',
            'latest_length_withheld_reasons',
            'model_first_supported_presence_frame', 'model_first_supported_presence_time_s',
            'model_first_sustained_presence_frame', 'model_first_sustained_presence_time_s',
            'model_last_supported_path_frame', 'model_last_supported_path_length_px',
            'model_presence_support_rule', 'model_timing_note',
            'model_event_bracket', 'model_latest_event_score',
            'human_event_verdict', 'human_event_bracket', 'human_event_source']
        with (out / 'grains.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=grain_fields)
            writer.writeheader()
            for row in result['population'].get('grain_summaries', []):
                writer.writerow({k: (json.dumps(row[k], sort_keys=True)
                                     if isinstance(row.get(k), (dict, list)) else row.get(k))
                                 for k in grain_fields})
        paths.update(grains_csv=str(out / 'grains.csv'), population_json=str(out / 'population.json'))
    fields = ["owner_id", "source_frame", "state", "tip_x", "tip_y", "length_px", "length_um",
              "presence_logit", "event_score", "exit_x", "exit_y", "exit_support", "exit_contract",
              "source_time_s", "elapsed_time_s", "growth_px_per_s", "growth_um_per_s",
              "growth_interval_s", "measurement_domain", "provenance", "as_inference_constraint",
              "model_body_provider", "body_assistance_identity", "body_seed_mask_id", "body_seed_frame",
              "grain_x", "grain_y", "grain_pose_status", "grain_identity_status", "grain_pose_origin",
              "grain_geometry_identity"]
    with (out / "measurements.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in result["rows"]:
            tip = row.get("tip_xy")
            d = {k: row.get(k) for k in fields}
            d.update(tip_x=tip[0] if tip else None, tip_y=tip[1] if tip else None)
            ex = row.get("exit_xy")
            d.update(exit_x=ex[0] if ex else None, exit_y=ex[1] if ex else None,
                     exit_support=row.get("exit_support"), exit_contract=row.get("exit_contract"))
            provider = (row.get('route_evidence') or {}).get('body_provider') or {}
            seed = provider.get('seed') or {}
            d.update(model_body_provider=provider.get('backend'),
                     body_assistance_identity=provider.get('assistance_identity'),
                     body_seed_mask_id=seed.get('source_id'), body_seed_frame=seed.get('frame'))
            pose = row.get('grain_pose') or {}
            center = pose.get('grain_native')
            d.update(grain_x=center[0] if center is not None else None,
                     grain_y=center[1] if center is not None else None,
                     grain_pose_status=pose.get('pose_status'),
                     grain_identity_status=pose.get('identity_status'),
                     grain_pose_origin=pose.get('review_origin'),
                     grain_geometry_identity=pose.get('geometry_identity'))
            writer.writerow(d)
    _write_json(out / "run.json", {"manifest_schema": "tubetracker.run.v1",
                                  "inputs": result["inputs"],
                                  "results": {"cache": result["cache"], "inference": result["inference"],
                                              "elapsed_wall_s": result["elapsed_wall_s"]}})
    return paths
