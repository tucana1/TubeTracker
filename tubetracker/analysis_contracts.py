"""Movie requests, typed reviews and measurements shared by app and CLI."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from .acquisition import AcquisitionClock, validate_acquisition
from .review_semantics import is_workflow_measurement


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()


def route_model_bindings(pixel_inputs):
    """Bind route validation to every pixel provider actually used."""
    bindings = {name: pixel_inputs[name + '_checkpoint_sha256'] for name in ('cap', 'body')}
    plan = pixel_inputs.get('body_assistance')
    if plan:
        unhashed = {k: v for k, v in plan.items() if k != 'identity'}
        if plan.get('identity') != stable_hash(unhashed):
            raise ValueError('reviewed-mask assistance identity does not match its inputs')
        bindings['body_assistance'] = plan['identity']
    geometry = pixel_inputs.get('grain_geometry')
    if geometry:
        unhashed = {k: v for k, v in geometry.items() if k != 'identity'}
        if geometry.get('identity') != stable_hash(unhashed):
            raise ValueError('frame grain geometry identity does not match its inputs')
        bindings['grain_geometry'] = geometry['identity']
    return bindings


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class AnalysisRequest:
    movie_path: str
    movie_id: str
    frames: list[int]
    owners: list[dict] = field(default_factory=list)
    roi_xyxy: list[int] | None = None
    discover_grains: bool = False
    grain_detection: dict = field(default_factory=dict)
    snapshot: str = ""
    cap_threshold: float = .5
    acquisition: dict = field(default_factory=dict)
    owner_aliases: dict = field(default_factory=dict)
    solver: dict = field(default_factory=dict)
    census_scope: dict | None = None
    body_extent: str = "grain_crop"
    body_normalization_context: str = "per_tile"
    route_policy: dict = field(default_factory=dict)
    route_validation: str = ""
    body_assistance: dict = field(default_factory=dict)
    grain_pose_path: str = ''
    selected_owner_ids: list[str] | None = None
    grain_motion: dict = field(default_factory=dict)

    def validate(self):
        if not self.movie_id or not Path(self.movie_path).is_file():
            raise ValueError("analysis requires an existing movie and its identity")
        if not self.frames or any(int(f) != f or f < 0 for f in self.frames):
            raise ValueError("analysis requires nonnegative source-frame indices")
        if self.frames != sorted(set(self.frames)):
            raise ValueError("source frames must be unique and increasing")
        ids = [o["id"] for o in self.owners]
        if len(ids) != len(set(ids)):
            raise ValueError("one analysis owner per physical grain ID")
        if self.selected_owner_ids is not None:
            selected = self.selected_owner_ids
            if (not isinstance(selected, list) or not selected
                    or any(not isinstance(v, str) or not v for v in selected)
                    or len(selected) != len(set(selected))):
                raise ValueError('selected_owner_ids requires unique nonempty physical grain IDs')
        for owner in self.owners:
            if owner.get("movie", self.movie_id) != self.movie_id:
                raise ValueError("owner belongs to a different movie")
            if len(owner.get("grain_native", [])) != 2 or not np.isfinite(owner["grain_native"]).all():
                raise ValueError("owner requires native grain geometry")
        if self.roi_xyxy:
            if len(self.roi_xyxy) != 4 or any(not math.isfinite(v) or int(v) != v for v in self.roi_xyxy):
                raise ValueError("ROI requires four native integer coordinates")
            x0, y0, x1, y1 = self.roi_xyxy
            if min(x0, y0) < 0 or not x0 < x1 or not y0 < y1:
                raise ValueError("ROI is an exact nonempty native rectangle")
        if not 0 < self.cap_threshold < 1:
            raise ValueError("cap threshold must be a probability in (0, 1)")
        if self.discover_grains:
            from .grain_detection import GrainDetectionConfig
            GrainDetectionConfig.from_dict(self.grain_detection)
        if self.body_extent not in ('grain_crop', 'analysis_roi'):
            raise ValueError('body_extent must be grain_crop or analysis_roi')
        if self.body_extent == 'analysis_roi' and not self.roi_xyxy:
            raise ValueError('body tiling requires an explicit analysis ROI')
        if self.body_normalization_context not in ('per_tile', 'grain_reference'):
            raise ValueError('body_normalization_context must be per_tile or grain_reference')
        if self.body_normalization_context == 'grain_reference' and self.body_extent != 'analysis_roi':
            raise ValueError('grain-reference normalization requires analysis_roi body extent')
        if self.body_assistance:
            from .seeded_body import SeededBodyConfig
            SeededBodyConfig.from_dict(self.body_assistance)
            if not self.roi_xyxy:
                raise ValueError('reviewed-mask tracking requires an explicit analysis ROI')
        if self.grain_pose_path and not Path(self.grain_pose_path).is_file():
            raise ValueError('grain pose file does not exist')
        if self.grain_motion:
            if self.grain_pose_path:
                raise ValueError('choose automatic grain motion or an external pose artifact')
            from .grain_motion import GrainMotionConfig
            GrainMotionConfig.from_dict(self.grain_motion)
        from prototypes.v30_video_apex.route_quality import RoutePolicy
        RoutePolicy(**self.route_policy)
        if self.route_validation and not Path(self.route_validation).is_file():
            raise ValueError('route validation file does not exist')
        validate_acquisition(self.acquisition)
        if self.census_scope:
            from prototypes.v30_video_apex.census import CensusScope
            scope = CensusScope(**self.census_scope)
            scope.validate()
            if scope.movie != self.movie_id or not set(scope.frames).issubset(self.frames):
                raise ValueError("census movie/frames must belong to this analysis")
            if self.roi_xyxy:
                x, y, w, h = scope.roi_xywh
                x0, y0, x1, y1 = self.roi_xyxy
                if not (x0 <= x and y0 <= y and x+w <= x1 and y+h <= y1):
                    raise ValueError("census scope must lie within the analysis field")

    def to_dict(self):
        return asdict(self)


def measurement_rows(rows, metadata, *, first_frame):
    clock = AcquisitionClock.from_metadata(metadata)
    scale = metadata.get("micrometres_per_pixel")
    origin_time = clock.at(int(first_frame))
    out, previous = [], {}
    for raw in sorted(rows, key=lambda r: (r["owner_id"], r["source_frame"])):
        r = copy.deepcopy(raw)
        frame, owner = int(r["source_frame"]), r["owner_id"]
        t = clock.at(frame)
        length = r.get("length_px")
        workflow = is_workflow_measurement(r)
        if workflow or r.get("state") != "present" or not r.get("path_complete"):
            length = None
        if workflow:
            r['measurement_domain'] = 'workflow_verification'
        r["length_px"] = length
        r["source_time_s"] = t
        r["elapsed_time_s"] = t - origin_time if t is not None and origin_time is not None else None
        r["length_um"] = float(length) * float(scale) if length is not None and scale is not None else None
        r["growth_px_per_s"] = r["growth_um_per_s"] = None
        r["growth_interval_s"] = None
        prior = previous.get(owner)
        if (length is not None and prior and prior.get("length_px") is not None
                and t is not None and prior["source_time_s"] is not None):
            dt = t - prior["source_time_s"]
            if dt > 0:
                r["growth_interval_s"] = dt
                r["growth_px_per_s"] = (float(length) - float(prior["length_px"])) / dt
                r["growth_um_per_s"] = r["growth_px_per_s"] * float(scale) if scale is not None else None
        # A workflow, hidden, absent, partial or uncertain interval breaks the rate
        # chain; no interpolated path or future extent becomes a measure.
        previous[owner] = r
        r["measurement_provenance"] = {
            "length": "current supported projected path" if length is not None else "withheld",
            "exclusion": "workflow_verification" if workflow else None,
            "cadence": metadata.get("cadence_source") if t is not None else None,
            "calibration": metadata.get("calibration_source") if scale is not None else None,
            "rate": "average change in projected current length over the reported interval"}
        out.append(r)
    return out


PROJECT_IDENTITY_FILE = 'project-identity.json'


def project_identity(source):
    """Stable logical project identity that survives backup/staging.

    A project may declare its identity in project-identity.json (written at
    bootstrap); the storage location is then recorded separately by callers
    (source/source_project paths). Without a declaration the normalized
    directory path is the identity, so genuinely different sources stay
    independent even when observation UUIDs collide.
    """
    if not source:
        return ''
    if isinstance(source, str) and source.startswith('project-id:'):
        return source          # already an identity; never re-resolve
    p = Path(source).expanduser()
    root = p.parent if p.suffix == '.db' else p
    declared = None
    identity = root / PROJECT_IDENTITY_FILE
    if identity.exists():
        try:
            declared = json.loads(identity.read_text()).get('project_id')
        except (json.JSONDecodeError, OSError):
            declared = None
    if declared:
        return f'project-id:{declared}'
    return str(root.resolve())


def _review_project(source):
    if isinstance(source, str) and source.startswith('project-id:'):
        return source          # already an identity; never re-resolve
    return project_identity(source)


def snapshot_review_records(directory):
    if not directory:
        return []
    p = Path(directory)
    records = json.loads((p / "observations.json").read_text())
    return [dict(r, source=str(p), source_id=r.get("obs_uuid"),
                 source_project=_review_project(r.get('project') or str(p)),
                 source_revision=int(r.get("obs_revision", 1))) for r in records]


def store_review_records(entities, *, movie_id, source):
    from .review_semantics import is_withdrawn
    tasks = {r["uuid"]: r["data"] for r in entities if r["kind"] == "task"}
    records = []
    for entity in entities:
        if entity["kind"] != "observation":
            continue
        d = dict(entity["data"])
        task = tasks.get(d.get("task_uuid"), {})
        d.update(movie=task.get("movie") or movie_id,
                 owner_uuid=d.get("owner_uuid") or task.get("owner_uuid"),
                 task_type=task.get("task_type", ""), source=str(source),
                 source_project=_review_project(source),
                 source_id=entity["uuid"], source_revision=int(entity["revision"]),
                 obs_uuid=entity["uuid"], live_review=True,
                 updated_utc=entity.get("updated_utc", ""))
        if task.get("reassign_from_owner"):
            d["reassign_from_owner"] = task["reassign_from_owner"]
        if "distinct_cap_evidence" in task:
            d["distinct_cap_evidence"] = task["distinct_cap_evidence"]
        if task.get("review_origin"):
            d["review_origin"] = task["review_origin"]
        if is_withdrawn(task):
            d['review_status'] = 'withdrawn'
        records.append(d)
    return sorted(records, key=lambda r: (r["updated_utc"], r["source_id"]))


def snapshot_tombstones(directory):
    """Deletions exported with a snapshot, as (project, uuid) identities."""
    if not directory:
        return []
    p = Path(directory)
    path = p / 'tombstones.json'
    if not path.exists():
        return []
    return [dict(r, source=str(p), source_project=project_identity(r.get('project') or str(p)))
            for r in json.loads(path.read_text())]


def resolve_review_constraints(records, request, owners, tombstones=()):
    from .annotation_schema import DIRECT_STATES
    from .review_semantics import is_withdrawn, precise_tip
    allowed = {o["id"] for o in owners}
    # A deletion (Undo included) is a tombstone: it must shadow an older
    # snapshot copy of the same logical project and observation, even when
    # the live database no longer holds a record at all.
    tomb = {(r.get('source_project') or project_identity(r.get('project') or r.get('source') or ''),
             r.get('uuid')) for r in tombstones}
    dropped = []
    if tomb:
        kept, dropped = [], []
        for r in records:
            key = (r.get('source_project') or project_identity(r.get('source') or ''),
                   r.get('source_id') or r.get('obs_uuid'))
            if key in tomb:
                dropped.append({'observation': key[1], 'source': r.get('source'),
                                'revision': r.get('source_revision'),
                                'reason': 'review deleted (tombstone)'})
            else:
                kept.append(r)
        records = kept
    aliases = request.owner_aliases
    groups = {}
    ignored = []
    # A live withdrawal must shadow the same observation in an older
    # immutable snapshot. UUIDs may collide across projects/movies.
    latest = {}
    for i, r in enumerate(records):
        uid = r.get('source_id') or r.get('obs_uuid')
        key = (_review_project(r.get('source_project') or r.get('project') or r.get('source')),
               r.get('movie'), uid if uid is not None else ('unidentified', i))
        rank = (int(r.get('source_revision', r.get('obs_revision', 1))),
                bool(r.get('live_review')), r.get('updated_utc', ''), i)
        if key not in latest or rank > latest[key][0]:
            latest[key] = rank, r
    for _, record in sorted(latest.values(), key=lambda pair: pair[0][-1]):
        r = dict(record)
        if r.get("movie") != request.movie_id or int(r.get("source_frame", -1)) not in request.frames:
            continue
        if is_withdrawn(r):
            ignored.append({'observation': r.get('source_id'), 'source': r.get('source'),
                            'revision': r.get('source_revision'), 'reason': 'review withdrawn'})
            continue
        raw_owner = str(r.get("owner_uuid") or r.get("tube_uuid") or r.get("obs_uuid", ""))
        if raw_owner == "unassigned":
            raw_owner = str(r.get("obs_uuid", ""))
        owner = aliases.get(raw_owner, raw_owner)
        if owner not in allowed:
            ignored.append({"observation": r.get("source_id"), "reason": "owner not in analysis"})
            continue
        key = owner, int(r["source_frame"])
        groups.setdefault(key, []).append(r)
    constraints = {}
    selected = []
    ignored.extend(dropped)
    for key, group in groups.items():
        tip_reviews = [g for g in group if g.get("tip_source") != "unobserved"]
        path_reviews = [g for g in group if len(g.get("path_xy") or []) >= 2]
        if not tip_reviews and not path_reviews:
            ignored.append({"observation": group[-1].get("source_id"),
                            "reason": "review contains no licensed tip or supported partial path"})
            continue
        # Live edits explicitly supersede the immutable snapshot. Within a
        # source, dedicated tip reviews take precedence over path drafts.
        r = max(enumerate(tip_reviews or path_reviews), key=lambda it: (bool(it[1].get("live_review")),
                 it[1].get("updated_utc", "") if it[1].get("live_review") else "",
                 it[1].get("task_type") == "review_tip", it[0]))[1]
        state, tip = (r.get("direct_state"), precise_tip(r)) if tip_reviews else ("unreviewed_tip", None)
        if state not in DIRECT_STATES and state != "unreviewed_tip":
            ignored.append({"observation": r.get("source_id"), "reason": "unsupported observation state"})
            continue
        if state == "direct_visible" and tip is None:
            region = np.asarray(r.get("direct_region") or [], dtype=float)
            if (region.ndim == 2 and region.shape[1] == 2 and len(region) >= 3
                    and np.isfinite(region).all()
                    and abs(np.dot(region[:, 0], np.roll(region[:, 1], 1))
                            - np.dot(region[:, 1], np.roll(region[:, 0], 1))) > 1e-6):
                state = "visible_imprecise"
            else:
                ignored.append({"observation": r.get("source_id"), "reason": "no separately licensed precise tip"})
                continue
        c = {"state": state, "tip_xy": tip, "source": r.get("source"),
             "observation_id": r.get("source_id"), "revision": r.get("source_revision", 1),
             "path_xy": r.get("path_xy") or [], "path_complete": bool(r.get("path_complete")),
             "tip_region_xy": r.get("direct_region") or [],
             "distinct_cap_evidence": r.get("distinct_cap_evidence"),
             "review_origin": r.get("review_origin", "human"),
             "lineage": [{"source": g.get("source"), "id": g.get("source_id"),
                          "revision": g.get("source_revision", 1),
                          "review_origin": g.get("review_origin", "human")} for g in group]}
        def provenance(record):
            return {"source": record.get("source"), "id": record.get("source_id"),
                    "revision": record.get("source_revision", 1),
                    "review_origin": record.get("review_origin", "human")}
        c["tip_evidence"] = provenance(r) if tip_reviews else None
        if c["path_xy"]:
            c["geometry_source"] = provenance(r)
        # Preserve a separately reviewed full route when a later point
        # confirmation refines its endpoint within the 5 px review gate.
        if state == "direct_visible" and tip and not c["path_xy"]:
            compatible = [g for g in group if g.get("path_complete") and len(g.get("path_xy") or []) >= 2
                          and np.linalg.norm(np.asarray(g["path_xy"][-1]) - tip) <= 5]
            if compatible:
                g = compatible[-1]
                c.update(path_xy=g["path_xy"], path_complete=True,
                         geometry_source=provenance(g))
            else:
                partial = [g for g in group if g.get("path_xy")]
                if partial:
                    c.update(path_xy=partial[-1]["path_xy"], path_complete=False,
                             geometry_source=provenance(partial[-1]))
        if (c.get("geometry_source") or {}).get("review_origin") == "workflow_test":
            c["review_origin"] = "workflow_test"
        if state != "direct_visible":
            c["tip_xy"] = None
            c["path_complete"] = False
        # rev14 W2: a human FULL trace licenses a FRAME-SPECIFIC root.
        # Point 1 of the reviewed centreline is the reviewed grain exit.
        # The root is scoped to this observation's frame and is never
        # transported to other frames without an explicit motion rule.
        # A PARTIAL or hidden trace can never yield a root.
        c["verified_root"] = None
        _gs = c.get("geometry_source") or {}
        if (c["path_complete"] and len(c["path_xy"]) >= 2
                and _gs.get("review_origin") != "workflow_test"
                and c.get("review_origin") != "workflow_test"):
            c["verified_root"] = {
                "xy": list(c["path_xy"][0]),
                "basis": "reviewed FULL path exit (point 1 of the human trace)",
                "scope_frame": int(r["source_frame"]),
                "observation_id": _gs.get("id"), "revision": _gs.get("revision"),
                "source": _gs.get("source"),
                "review_origin": _gs.get("review_origin", "human")}
        constraints[key] = c
        selected.append((key, r))
    # An explicit reassignment releases an older point constraint on the
    # previous owner only when it refers to that same reviewed location.
    for (new_owner, frame), r in selected:
        if r.get("live_review") and r.get("reassign_from_owner") and precise_tip(r) is not None:
            old = aliases.get(r["reassign_from_owner"], r["reassign_from_owner"])
            if old == new_owner:
                continue
            key = old, frame
            prior = constraints.get(key)
            if prior and prior.get("tip_xy") and np.linalg.norm(np.asarray(prior["tip_xy"]) - r["direct_xy"]) <= 5:
                del constraints[key]
    return constraints, ignored
