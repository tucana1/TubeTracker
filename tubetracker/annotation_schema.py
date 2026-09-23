"""Versioned v2 annotation entities (P0A foundation, Qt-free).

Pure typed dataclasses + validation shared by the workbench, training,
and evaluation. No napari/Qt import allowed here so headless GPU hosts
can use this module. Schema version is explicit; legacy v1 imports go
through an adapter (not this file).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

SCHEMA_VERSION = "v2.1"

# Observation states (typed missingness — never impute coordinates).
DIRECT_VISIBLE = "direct_visible"
VISIBLE_IMPRECISE = "visible_imprecise"
NOT_DIRECTLY_VISIBLE = "not_directly_visible"
NO_TUBE_VISIBLE = "no_tube_visible"
OWNER_UNCERTAIN = "owner_uncertain"
OUT_OF_FIELD = "out_of_field"
LOST_IDENTITY = "lost_identity"
UNRESOLVED_OVERLAP = "unresolved_overlap"

DIRECT_STATES = frozenset({
    DIRECT_VISIBLE, VISIBLE_IMPRECISE, NOT_DIRECTLY_VISIBLE, NO_TUBE_VISIBLE,
    OWNER_UNCERTAIN, OUT_OF_FIELD, LOST_IDENTITY, UNRESOLVED_OVERLAP,
})


def new_uuid() -> str:
    return uuid.uuid4().hex


@dataclass
class Movie:
    uuid: str
    content_hash: str
    location: str
    acquisition_group: str
    specimen_id: str
    native_width: int
    native_height: int
    calibration_px_per_um: float | None = None
    calibration_note: str = "unknown"

    def validate(self) -> list[str]:
        errs = []
        if not self.content_hash:
            errs.append("movie content_hash required")
        if self.native_width <= 0 or self.native_height <= 0:
            errs.append("movie native size must be positive")
        return errs


@dataclass
class Owner:
    uuid: str
    movie_uuid: str
    grain_xy: tuple[float, float] | None = None
    root_xy: tuple[float, float] | None = None
    evidence_source_frames: list[int] = field(default_factory=list)
    pipeline_aliases: list[int] = field(default_factory=list)

    def validate(self) -> list[str]:
        errs = []
        if not self.movie_uuid:
            errs.append("owner movie_uuid required")
        return errs


@dataclass
class EventTask:
    uuid: str
    movie_uuid: str
    owner_uuid: str
    source_start: int
    source_end: int
    query_frames: list[int] = field(default_factory=list)
    stratum: str = ""
    role: str = "training"  # training | development | reference
    task_type: str = "apex"  # apex | centerline | crossing | owner | germination_event ...
    completed: bool = False
    # P0A annotation-extent contract: what was requested, where, and
    # how complete the review claims to be. Export must state these.
    geometry_type: str = "point"  # point | path | region | mask | census
    review_region: list[tuple[float, float]] | None = None
    class_scope: str = "apex"  # apex | tube | grain | crossing | census
    instance_scope: str = ""  # owner/tube uuid under review, "" = unassigned
    completeness: str = "partial"  # partial | complete | unreviewed

    def validate(self) -> list[str]:
        errs = []
        if self.source_end < self.source_start:
            errs.append("event source_end < source_start")
        if self.role not in ("training", "development", "reference"):
            errs.append(f"bad role {self.role!r}")
        if self.completeness not in ("partial", "complete", "unreviewed"):
            errs.append(f"bad completeness {self.completeness!r}")
        return errs


@dataclass
class Observation:
    uuid: str
    owner_uuid: str
    source_frame: int
    direct_state: str = NOT_DIRECTLY_VISIBLE
    direct_xy: tuple[float, float] | None = None
    direct_region: list[tuple[float, float]] | None = None
    context_xy: tuple[float, float] | None = None
    context_frames: list[int] = field(default_factory=list)
    owner_decision: str = "undecided"  # confirmed | corrected | undecided
    # P0A identity contract: every observation names its task, movie,
    # and tube ("" = provisional/unassigned — never fabricate).
    task_uuid: str = ""
    movie_uuid: str = ""
    tube_uuid: str = ""
    # P0A path support: ordered root-to-tip polyline with per-vertex
    # visibility (True = directly seen, False = hidden/inferred span).
    path_xy: list[tuple[float, float]] = field(default_factory=list)
    path_visible: list[bool] = field(default_factory=list)
    path_complete: bool = False  # full root-to-tip vs partial span
    annotator: str = ""
    revision: int = 1

    def validate(self) -> list[str]:
        errs = []
        if self.direct_state not in DIRECT_STATES:
            errs.append(f"bad direct_state {self.direct_state!r}")
        if self.direct_state in (DIRECT_VISIBLE, VISIBLE_IMPRECISE):
            if self.direct_xy is None and not self.direct_region:
                errs.append("visible apex needs point or region")
        if self.direct_state == NOT_DIRECTLY_VISIBLE and self.direct_xy is not None:
            errs.append("hidden apex must not carry an exact point")
        if self.path_xy and len(self.path_visible) not in (0, len(self.path_xy)):
            errs.append("path_visible must be empty or match path_xy")
        return errs


@dataclass
class SupervisionRegion:
    uuid: str
    movie_uuid: str
    source_frame: int
    polygon_xy: list[tuple[float, float]] = field(default_factory=list)
    kind: str = "ignore"  # positive | verified_negative | ignore
    class_scope: str = "apex"
    owner_uuid: str | None = None
    confirmed: bool = False

    def validate(self) -> list[str]:
        errs = []
        if self.kind not in ("positive", "verified_negative", "ignore"):
            errs.append(f"bad kind {self.kind!r}")
        if len(self.polygon_xy) < 3:
            errs.append("region needs >=3 vertices")
        return errs


GERMINATION_VERDICTS = ("emerged_at_start", "emerged_within",
                        "no_emergence_by_end", "unobservable")


@dataclass
class GerminationEvent:
    """One physical grain's whole-interval emergence answer (WO2).

    The unit is one physical grain through one reviewed movie interval;
    identity stays fixed. Interval/censoring semantics are stored, never
    a bare timestamp: bracket endpoints may each be null (left/right
    censored), and 'no_emergence_by_end' is censored, not a biological
    never-germinates claim. 'unobservable' keeps unknown frames unknown.
    """
    uuid: str
    movie_uuid: str
    movie_content_hash: str = ""
    owner_uuid: str = ""
    task_uuid: str = ""
    window_start: int = 0
    window_end: int = 0
    verdict: str = "unobservable"  # GERMINATION_VERDICTS
    last_absent_frame: int | None = None
    first_visible_frame: int | None = None
    consulted_frames: list[int] = field(default_factory=list)
    absence_region_xy: list[tuple[float, float]] | None = None
    grain_mask_uuid: str | None = None
    tube_mask_uuid: str | None = None
    exit_xy: tuple[float, float] | None = None
    apex_xy: tuple[float, float] | None = None
    annotator: str = ""
    revision: int = 1
    lineage: list[str] = field(default_factory=list)

    def validate(self) -> list[str]:
        errs = []
        if not self.movie_uuid:
            errs.append("germination event movie_uuid required")
        if not self.owner_uuid:
            errs.append("germination event owner_uuid required")
        if self.window_end < self.window_start:
            errs.append("germination event window_end < window_start")
        if self.verdict not in GERMINATION_VERDICTS:
            errs.append(f"bad verdict {self.verdict!r}")
        if self.verdict == "emerged_within":
            if self.last_absent_frame is None and self.first_visible_frame is None:
                errs.append("emerged_within needs at least one bracket endpoint")
            if (self.last_absent_frame is not None and self.first_visible_frame is not None
                    and self.first_visible_frame <= self.last_absent_frame):
                errs.append("bracket endpoints do not order")
        if self.verdict == "emerged_at_start" and self.last_absent_frame is not None:
            errs.append("emerged_at_start cannot carry a last-absent frame")
        if self.verdict == "no_emergence_by_end" and self.first_visible_frame is not None:
            errs.append("no_emergence_by_end cannot carry a first-visible frame")
        return errs


@dataclass
class Crossing:
    uuid: str
    movie_uuid: str
    owner_uuids: list[str] = field(default_factory=list)
    entry_segments: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    exit_segments: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    continuation: dict[str, str] = field(default_factory=dict)
    unresolved: bool = True
    adjudicated: bool = False

    def validate(self) -> list[str]:
        errs = []
        lanes = set(self.entry_segments) & set(self.exit_segments)
        if len(lanes) < 2 and len(self.owner_uuids) < 2:
            errs.append("crossing needs >=2 traced lanes")
        if lanes and not self.unresolved and set(self.continuation) != lanes:
            errs.append("resolved crossing needs an assignment for every lane")
        return errs


@dataclass
class SnapshotLock:
    uuid: str
    schema_version: str
    member_uuids: list[str] = field(default_factory=list)
    content_hash: str = ""
    split: str = "development"
    finalized: bool = False

    def validate(self) -> list[str]:
        errs = []
        if self.schema_version != SCHEMA_VERSION:
            errs.append("snapshot schema mismatch")
        if self.finalized and not self.content_hash:
            errs.append("finalized snapshot needs content hash")
        return errs

    def migrate_to_current(self) -> "SnapshotLock":
        """Re-stamp a v2.0 lock as v2.1 (H260 contract).

        v2.0 -> v2.1 changed task/observation extent semantics, not
        snapshot membership: member UUIDs stay valid, so migration
        re-stamps the version and records provenance. Anything older
        than v2.0 is rejected, not guessed.
        """
        if self.schema_version == SCHEMA_VERSION:
            return self
        if self.schema_version != "v2.0":
            raise ValueError(
                f"cannot migrate snapshot schema {self.schema_version!r}")
        return SnapshotLock(
            uuid=self.uuid,
            schema_version=SCHEMA_VERSION,
            member_uuids=list(self.member_uuids),
            content_hash=self.content_hash,
            split=self.split,
            finalized=False,  # re-finalize (and re-hash) under v2.1
        )
