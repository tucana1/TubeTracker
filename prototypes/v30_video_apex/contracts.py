"""v30 typed contracts: IDs, geometry, candidates, visibility.

Headless: stdlib + numpy only. No Qt/napari, no torch required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math
from typing import Any, Optional, Sequence


# ---- Typed IDs (opaque strings; owner UUID is a lookup key, never a feature) ----

MovieId = str
OwnerId = str
GrainId = str
TubeId = str
CandidateId = str


class Visibility(str, Enum):
    DIRECTLY_VISIBLE = "directly_visible"
    IMPRECISE = "visible_imprecise"
    HIDDEN = "not_directly_visible"
    NO_TUBE = "no_tube_visible"
    OUT_OF_FIELD = "out_of_field"
    LOST_IDENTITY = "lost_identity"
    UNRESOLVED = "unresolved_overlap"


class ObservationState(str, Enum):
    OBSERVED = "observed"
    NOT_OBSERVED = "not_observed"
    RECONSTRUCTED = "reconstructed"
    IDENTITY_AMBIGUOUS = "identity_ambiguous"
    CENSORED = "censored"


class AcceptanceState(str, Enum):
    ACCEPTED = "accepted"
    WITHHELD = "withheld"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class FrameRef:
    movie_id: MovieId
    source_frame: int
    t_seconds: float = 0.0

    def validate(self) -> None:
        if not self.movie_id:
            raise ValueError("FrameRef.movie_id must be non-empty")
        if self.source_frame < 0:
            raise ValueError("FrameRef.source_frame must be >= 0")
        if not math.isfinite(self.t_seconds) or self.t_seconds < 0:
            raise ValueError("FrameRef.t_seconds must be finite and >= 0")


@dataclass(frozen=True)
class NativeTransform:
    """Crop/pad mapping between crop pixels and native source pixels.

    native = (crop - offset) / scale. scale > 0 preserves aspect.
    """

    scale: float
    offset_x: float
    offset_y: float
    crop_w: int
    crop_h: int

    def validate(self) -> None:
        if not (math.isfinite(self.scale) and self.scale > 0):
            raise ValueError("NativeTransform.scale must be finite and > 0")
        for v in (self.offset_x, self.offset_y):
            if not math.isfinite(v):
                raise ValueError("NativeTransform offsets must be finite")
        if self.crop_w <= 0 or self.crop_h <= 0:
            raise ValueError("NativeTransform crop dims must be > 0")

    def to_native(self, x: float, y: float) -> tuple[float, float]:
        self.validate()
        return ((x - self.offset_x) / self.scale, (y - self.offset_y) / self.scale)

    def from_native(self, xn: float, yn: float) -> tuple[float, float]:
        self.validate()
        return (xn * self.scale + self.offset_x, yn * self.scale + self.offset_y)


@dataclass
class SupervisionMask:
    """Per-head positive/negative/unknown scoping. Unknown (0) gets no gradient."""

    positive: Any = None  # np.ndarray bool
    negative: Any = None
    unknown: Any = None

    def validate(self) -> None:
        import numpy as np

        for name in ("positive", "negative", "unknown"):
            arr = getattr(self, name)
            if arr is None:
                raise ValueError(f"SupervisionMask.{name} must be set")
            a = np.asarray(arr, dtype=bool)
            if a.ndim != 2:
                raise ValueError(f"SupervisionMask.{name} must be 2D")
        p = self.positive
        import numpy as _np

        p, n, u = (getattr(self, k) for k in ("positive", "negative", "unknown"))
        p, n, u = _np.asarray(p, bool), _np.asarray(n, bool), _np.asarray(u, bool)
        if p.shape != n.shape or p.shape != u.shape:
            raise ValueError("SupervisionMask heads must share shape")
        if bool((p & n).any()) or bool((p & u).any()) or bool((n & u).any()):
            raise ValueError("SupervisionMask heads must be mutually exclusive")

    @property
    def shape(self):  # type: ignore[no-untyped-def]
        return self.positive.shape


@dataclass
class RouteCandidate:
    candidate_id: CandidateId
    movie_id: MovieId
    owner_id: OwnerId
    source_frame: int
    polyline_native: Any  # (N,2) float array in native coords
    route_score: float = 0.0
    owner_compat: float = 0.0
    config_hash: str = ""
    model_hash: str = ""

    def validate(self) -> None:
        import numpy as np

        if not self.candidate_id or not self.movie_id or not self.owner_id:
            raise ValueError("RouteCandidate ids must be non-empty")
        if self.source_frame < 0:
            raise ValueError("RouteCandidate.source_frame must be >= 0")
        pl = np.asarray(self.polyline_native, dtype=float)
        if pl.ndim != 2 or pl.shape[0] < 2 or pl.shape[1] != 2:
            raise ValueError("RouteCandidate.polyline_native must be (N>=2, 2)")
        if not bool(np.isfinite(pl).all()):
            raise ValueError("RouteCandidate.polyline_native must be finite")
        for v in (self.route_score, self.owner_compat):
            if not math.isfinite(v):
                raise ValueError("RouteCandidate scores must be finite")


@dataclass
class FrontDistribution:
    """Per-tube-per-route front distribution q(s) over arclength grid."""

    candidate_id: CandidateId
    s_grid: Any  # (S,) arclength samples
    probs: Any  # (S,) normalized
    observed: bool = True

    def validate(self) -> None:
        import numpy as np

        s = np.asarray(self.s_grid, dtype=float)
        p = np.asarray(self.probs, dtype=float)
        if s.ndim != 1 or p.ndim != 1 or s.shape != p.shape or s.shape[0] < 1:
            raise ValueError("FrontDistribution s_grid/probs must be matching 1D arrays")
        if not bool(np.isfinite(p).all()):
            raise ValueError("FrontDistribution.probs must be finite")
        if (p < 0).any():
            raise ValueError("FrontDistribution.probs must be >= 0")
        total = float(p.sum())
        if not math.isfinite(total) or total <= 0:
            raise ValueError("FrontDistribution.probs must sum to > 0")
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"FrontDistribution.probs must be normalized (sum={total})")


@dataclass
class ApexCandidate:
    """The one candidate interface shared by model/infer/association/evaluate."""

    candidate_id: CandidateId
    movie_id: MovieId
    owner_id: OwnerId
    source_frame: int
    x_native: Optional[float] = None
    y_native: Optional[float] = None
    mode: int = 0
    tip_score: float = 0.0
    owner_compat: float = 0.0
    visibility: Visibility = Visibility.DIRECTLY_VISIBLE
    observation: ObservationState = ObservationState.OBSERVED
    route: Optional[RouteCandidate] = None
    front: Optional[FrontDistribution] = None
    evidence: dict = field(default_factory=dict)

    def validate(self) -> None:
        if not self.candidate_id or not self.movie_id or not self.owner_id:
            raise ValueError("ApexCandidate ids must be non-empty")
        if self.source_frame < 0:
            raise ValueError("ApexCandidate.source_frame must be >= 0")
        if not isinstance(self.visibility, Visibility):
            raise ValueError("ApexCandidate.visibility must be a Visibility")
        if not isinstance(self.observation, ObservationState):
            raise ValueError("ApexCandidate.observation must be an ObservationState")
        if self.observation == ObservationState.NOT_OBSERVED:
            if self.x_native is not None or self.y_native is not None:
                raise ValueError("NOT_OBSERVED candidate must carry no coordinates")
        else:
            if self.x_native is None or self.y_native is None:
                raise ValueError("Observed candidate must carry coordinates")
            if not (math.isfinite(self.x_native) and math.isfinite(self.y_native)):
                raise ValueError("ApexCandidate coordinates must be finite")
        for v in (self.tip_score, self.owner_compat):
            if not math.isfinite(v):
                raise ValueError("ApexCandidate scores must be finite")
        if self.route is not None:
            self.route.validate()
        if self.front is not None:
            self.front.validate()


def validate_candidate_schema(d: dict) -> ApexCandidate:
    """Build + validate an ApexCandidate from a plain dict; reject bad joins."""
    required = ("candidate_id", "movie_id", "owner_id", "source_frame")
    for key in required:
        if key not in d:
            raise ValueError(f"candidate dict missing required key: {key}")
    vis = d.get("visibility", Visibility.DIRECTLY_VISIBLE)
    obs = d.get("observation", ObservationState.OBSERVED)
    if isinstance(vis, str):
        vis = Visibility(vis)
    if isinstance(obs, str):
        obs = ObservationState(obs)
    cand = ApexCandidate(
        candidate_id=str(d["candidate_id"]),
        movie_id=str(d["movie_id"]),
        owner_id=str(d["owner_id"]),
        source_frame=int(d["source_frame"]),
        x_native=d.get("x_native"),
        y_native=d.get("y_native"),
        mode=int(d.get("mode", 0)),
        tip_score=float(d.get("tip_score", 0.0)),
        owner_compat=float(d.get("owner_compat", 0.0)),
        visibility=vis,
        observation=obs,
        route=d.get("route"),
        front=d.get("front"),
        evidence=dict(d.get("evidence", {})),
    )
    cand.validate()
    return cand


def stable_hash(parts: Sequence[str]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def reject_unsupported_join(left_movie: MovieId, right_movie: MovieId) -> None:
    if left_movie != right_movie:
        raise ValueError(f"unsupported cross-movie join: {left_movie!r} vs {right_movie!r}")
