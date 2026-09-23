"""v30 association: event-level beam search with explicit unresolved state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence


class TrackState(str, Enum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    LOST = "lost_identity"
    NO_OBSERVATION = "no_observation"


@dataclass
class Hypothesis:
    owner_id: str
    route_id: str
    frame: int
    x: float | None
    y: float | None
    score: float
    state: TrackState = TrackState.RESOLVED
    lineage: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.owner_id or not self.route_id:
            raise ValueError("Hypothesis owner/route ids must be non-empty")
        if not isinstance(self.state, TrackState):
            raise ValueError("Hypothesis.state must be a TrackState")
        if self.state in (TrackState.RESOLVED, TrackState.UNRESOLVED):
            if self.x is None or self.y is None:
                raise ValueError(f"{self.state} hypothesis must carry coordinates")
        if self.state in (TrackState.LOST, TrackState.NO_OBSERVATION):
            pass  # coordinates optional (held prediction, not fresh observation)


def _transitions(prev: Hypothesis, cand: dict, frame: int) -> Hypothesis:
    import math

    ambiguous = cand.get("ambiguous", False) or cand.get("owner_compat", 1.0) < 0.3
    obs = str(cand.get("observation", "observed"))
    if obs == "not_observed" or cand.get("x_native") is None:
        return Hypothesis(prev.owner_id, prev.route_id, frame, None, None,
                          prev.score + float(cand.get("tip_score", 0.0)) - 1.0,
                          TrackState.NO_OBSERVATION, prev.lineage + [cand["candidate_id"]])
    state = TrackState.UNRESOLVED if ambiguous else TrackState.RESOLVED
    dx = (cand["x_native"] - (prev.x or 0.0)) if prev.x is not None else 0.0
    dy = (cand["y_native"] - (prev.y or 0.0)) if prev.y is not None else 0.0
    motion = math.hypot(dx, dy)
    score = prev.score + float(cand.get("tip_score", 0.0)) - 0.01 * motion
    return Hypothesis(prev.owner_id, str(cand.get("route_id", prev.route_id)), frame,
                      float(cand["x_native"]), float(cand["y_native"]), score, state,
                      prev.lineage + [cand["candidate_id"]])


def beam_search_per_owner(candidates_by_frame: Sequence[Sequence[dict]], owner_id: str,
                          beam_width: int = 4) -> list[Hypothesis]:
    """Beam over frames; keeps UNRESOLVED alternatives instead of forcing a winner."""
    if not candidates_by_frame:
        return [Hypothesis(owner_id, "none", -1, None, None, 0.0, TrackState.LOST, [])]
    beam = [Hypothesis(owner_id, f"r{k}", 0, None, None, 0.0, TrackState.NO_OBSERVATION, [])
            for k in range(min(beam_width, 2))]
    for t, cands in enumerate(candidates_by_frame):
        expanded: list[Hypothesis] = []
        pool = list(cands) or [{"candidate_id": f"{owner_id}:{t}:empty", "observation": "not_observed",
                                "x_native": None, "y_native": None, "tip_score": 0.0}]
        for hyp in beam:
            for cand in pool:
                expanded.append(_transitions(hyp, cand, t))
        # keep top by score but always retain one UNRESOLVED and one NO_OBSERVATION alt
        expanded.sort(key=lambda h: h.score, reverse=True)
        kept = expanded[:beam_width]
        for want in (TrackState.UNRESOLVED, TrackState.NO_OBSERVATION):
            if not any(h.state == want for h in kept):
                alt = next((h for h in expanded if h.state == want), None)
                if alt is not None:
                    kept[-1] = alt
        beam = kept
    for h in beam:
        h.validate()
    return sorted(beam, key=lambda h: h.score, reverse=True)
