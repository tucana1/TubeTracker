"""rev12 P1.2: local perception vs joint physical-owner assignment.

The four required interfaces, focused in one module (they did not fit
anywhere else; no isolated demo):

- FrameEvidence   — native image/transform/validity, grain instances,
                    tube foreground/boundaries, local current-cap
                    likelihood and uncertainty, ordered temporal context.
- Owner           — durable grain ID, grain geometry, attachment
                    hypotheses, annotation links, identity confidence;
                    at most one biological tube.
- RouteHypothesis — owner + attachment, support geometry, current
                    prefix/front, local-cap evidence, whole-route
                    consistency, missing/ambiguous segments, provenance.
- OwnerFrameState — present / verified-absent / occluded / out-of-field
                    / identity-uncertain, accepted hypothesis or
                    alternatives, measurement domain and time provenance.

Plus the JOINT assignment step: per-owner beams (association module)
followed by cross-owner constraints — a clearly identified cap cannot
be owned by two grains; overlapping pixels and uncertain co-located
tips remain permitted; alternatives are retained when evidence is
insufficient. Future observations may establish identity, never
historical visible length.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

OWNER_STATES = ("present", "verified_absent", "occluded", "out_of_field",
                "identity_uncertain")


@dataclass
class FrameEvidence:
    """Everything observable at ONE source frame for one owner."""

    movie: str
    source_frame: int
    crop_xywh: tuple
    validity_frac: float           # fraction of pixels actually read
    grain_instances: list = field(default_factory=list)
    tube_fg: Any = None            # native tube foreground probabilities
    tube_boundaries: Any = None
    local_cap_logits: Any = None   # raw (calibration-independent)
    local_cap_uncertainty: float = 1.0
    temporal_context: list = field(default_factory=list)  # ordered frames

    def validate(self) -> None:
        if not self.movie or self.source_frame < 0:
            raise ValueError("FrameEvidence needs movie/frame")
        if len(self.crop_xywh) != 4:
            raise ValueError("FrameEvidence needs a crop box")
        if not (0.0 <= self.validity_frac <= 1.0):
            raise ValueError("validity_frac must be in [0, 1]")


@dataclass
class Owner:
    """A durable physical grain; at most one biological tube."""

    owner_id: str                  # durable movie-qualified id
    grain_id: str
    movie: str
    grain_native: tuple
    grain_radius_px: float
    attachments: list = field(default_factory=list)  # hypotheses
    annotation_links: list = field(default_factory=list)
    identity_confidence: float = 1.0

    def validate(self) -> None:
        if not self.owner_id or not self.grain_id:
            raise ValueError("Owner needs owner_id and grain_id")
        if len(self.grain_native) != 2:
            raise ValueError("Owner needs grain_native xy")


@dataclass
class RouteHypothesis:
    """One root-to-cap candidate for one owner at one frame.

    rev13 W2: support and current geometry are SEPARATE. `support_xy`
    is the context polyline (may extend past the cap); `current_path_xy`
    is the owned root-to-cap path whose final point IS the reported tip
    (`tip_xy`). For valid full geometry, `current_path_xy[-1] ==
    tip_xy` and `arclength(current_path_xy) == length_px` within
    tolerance.
    """

    owner_id: str
    frame: int
    route_id: str
    attachment: tuple
    support_xy: list               # native polyline (context)
    current_prefix_len_px: float   # prefix up to the current front
    local_cap_score: float
    whole_route_score: float
    # rev13 W2: current geometry + evidence identity
    current_path_xy: list | None = None   # owned root-to-cap path
    tip_xy: tuple | None = None           # == current_path_xy[-1]
    length_px: float | None = None        # full length only when valid
    cap_candidate_id: str = ""            # deployed pool's cap id
    cap_clear: bool = False               # local evidence is decisive
    temporal_score: float = 0.0
    score_components: dict = field(default_factory=dict)
    missing_segments: list = field(default_factory=list)
    ambiguous_segments: list = field(default_factory=list)
    provenance: str = "runtime-proposal"
    alternatives: list = field(default_factory=list)

    def validate(self) -> None:
        if not self.owner_id or len(self.support_xy) < 2:
            raise ValueError("RouteHypothesis needs owner + support")
        if self.current_path_xy is not None and len(
                self.current_path_xy) < 2:
            raise ValueError("current_path_xy needs >= 2 points")

    def cap_xy(self) -> tuple | None:
        """The CURRENT CAP this hypothesis claims (never the support
        end): tip_xy when set, else the current path's final point."""
        if self.tip_xy is not None:
            return (float(self.tip_xy[0]), float(self.tip_xy[1]))
        if self.current_path_xy:
            p = self.current_path_xy[-1]
            return (float(p[0]), float(p[1]))
        return None


@dataclass
class OwnerFrameState:
    """The joint result for one owner at one frame."""

    owner_id: str
    frame: int
    state: str                     # OWNER_STATES
    accepted: RouteHypothesis | None = None
    alternatives: list = field(default_factory=list)
    measurement_domain: str = ""   # full | partial | none
    time_provenance: str = ""      # cadence source or "none (frames only)"
    note: str = ""

    def validate(self) -> None:
        if self.state not in OWNER_STATES:
            raise ValueError(f"bad owner state {self.state!r}")
        if self.state == "present" and self.accepted is None:
            raise ValueError("present state needs an accepted hypothesis")


def current_path_from_support(support_xy, s_cap_px,
                              *, tol_px: float = 1e-9):
    """rev13 W2 shared measurement contract: truncate the support at
    the accepted front arclength and interpolate its final segment
    ONCE. Returns (current_path, length_px) where current_path[-1] is
    the cap point and length_px == arclength(current_path) within
    floating-point tolerance. The current path — never the full
    support — is what ranking, collision, rendering, correction
    storage and export must use.
    """
    import numpy as _np
    pts = _np.asarray(support_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2:
        raise ValueError("support needs >= 2 points")
    seg = _np.diff(pts, axis=0)
    cum = _np.concatenate(
        [[0.0], _np.cumsum(_np.hypot(seg[:, 0], seg[:, 1]))])
    total = float(cum[-1])
    s = min(max(float(s_cap_px), 0.0), total)
    cap = _np.array([_np.interp(s, cum, pts[:, 0]),
                     _np.interp(s, cum, pts[:, 1])])
    keep = pts[cum <= s + tol_px]
    if len(keep) == 0:
        path = _np.vstack([pts[0], cap])
    elif _np.hypot(*(keep[-1] - cap)) > tol_px:
        path = _np.vstack([keep, cap])
    else:
        path = keep
    d = _np.diff(path, axis=0)
    length = float(_np.hypot(d[:, 0], d[:, 1]).sum())
    return path.tolist(), length


def geometry_consistent(path_xy, tip_xy, length_px,
                        *, tol_px: float = 1e-6,
                        tol_len: float = 1e-6) -> dict:
    """Check the declared full-geometry contract: current_path[-1] ==
    tip and arclength(current_path) == length_px within tolerance."""
    import numpy as _np
    p = _np.asarray(path_xy, dtype=float)
    d = _np.diff(p, axis=0)
    alen = float(_np.hypot(d[:, 0], d[:, 1]).sum())
    e_tip = (float(_np.hypot(*(p[-1] - _np.asarray(tip_xy, float))))
             if tip_xy is not None else None)
    e_len = abs(alen - float(length_px)) if length_px is not None \
        else None
    return {"endpoint_err_px": e_tip, "length_err_px": e_len,
            "endpoint_ok": e_tip is not None and e_tip <= tol_px,
            "length_ok": e_len is not None and e_len <= tol_len}


def _same_physical_cap(a: RouteHypothesis, b: RouteHypothesis,
                       merge_tol_px: float) -> tuple[bool, float]:
    """Does the EVIDENCE identify the same physical cap?

    rev13 W2: current caps (never support ends) within merge_tol_px is
    necessary but NOT sufficient — XY coincidence alone cannot exclude a
    claim. The caps must also be identified as the same object: the
    deployed pool's cap candidate id agrees (when both sides carry
    one), or both sides' local cap evidence is decisive (`cap_clear`).
    """
    pa, pb = a.cap_xy(), b.cap_xy()
    if pa is None or pb is None:
        return False, float("inf")
    d = ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5
    if d > merge_tol_px:
        return False, d
    if a.cap_candidate_id and b.cap_candidate_id:
        return a.cap_candidate_id == b.cap_candidate_id, d
    return bool(a.cap_clear and b.cap_clear), d


def joint_assign(states_by_owner: dict[str, list[OwnerFrameState]],
                 merge_tol_px: float = 6.0,
                 clear_margin: float = 0.25) -> dict:
    """Cross-owner constraint pass over per-owner frame states.

    rev13 W2 repair of the audit's counterexamples:
    - comparison uses the CURRENT CAP (tip / current path end), not the
      support end: same cap + different support ends demotes; different
      caps + same support end does NOT;
    - duplicate exclusion requires evidence of the SAME physical cap
      (`_same_physical_cap`), not XY coincidence;
    - no assertion on malformed entries (a demoted or accepted-less
      entry is skipped, reported) — the three-owner collision no longer
      crashes;
    - ties break deterministically by owner_id, so owner ordering never
      changes the outcome.

    Returns {swaps, n_duplicate_resolved, skipped, overlaps}.
    """
    by_frame: dict[int, list[tuple[str, OwnerFrameState]]] = {}
    for oid, states in states_by_owner.items():
        for st in states:
            if st.state == "present" and st.accepted is not None:
                by_frame.setdefault(st.frame, []).append((oid, st))
    swaps = []
    skipped = []
    overlaps = []
    for frame, entries in sorted(by_frame.items()):
        # deterministic order: owner id, then route id
        entries = sorted(entries, key=lambda e: (
            e[0], getattr(e[1].accepted, "route_id", "")))
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                oi, si = entries[i]
                oj, sj = entries[j]
                ai, aj = si.accepted, sj.accepted
                if ai is None or aj is None:
                    # demoted by an earlier pair (or malformed input):
                    # never an assertion — report and continue.
                    skipped.append({"frame": frame, "owner": oi if ai is
                                    None else oj,
                                    "reason": "no accepted hypothesis"})
                    continue
                same, d = _same_physical_cap(ai, aj, merge_tol_px)
                if d > merge_tol_px:
                    continue
                mi = (ai.local_cap_score
                      - (si.alternatives[0].local_cap_score
                         if si.alternatives else 0.0))
                mj = (aj.local_cap_score
                      - (sj.alternatives[0].local_cap_score
                         if sj.alternatives else 0.0))
                if same and mi >= clear_margin and mj >= clear_margin:
                    # one clearly identified physical cap, claimed by
                    # two owners: duplicate ownership. The lower-scoring
                    # claim is demoted; deterministic tie-break.
                    _sa = (ai.local_cap_score, ai.owner_id)
                    _sb = (aj.local_cap_score, aj.owner_id)
                    loser, keeper = ((si, oj) if _sa < _sb else (sj, oi))
                    loser.state = "identity_uncertain"
                    loser.alternatives = ([loser.accepted]
                                          + loser.alternatives)
                    loser.accepted = None
                    loser.note = (f"duplicate cap ownership with "
                                  f"{keeper} at frame {frame} "
                                  f"(d={d:.1f} px); retained as "
                                  f"alternative")
                    swaps.append({"frame": frame, "kept": keeper,
                                  "demoted": loser.owner_id,
                                  "distance_px": round(d, 2)})
                else:
                    for _oid, st in ((oi, si), (oj, sj)):
                        if st.state == "present" and not st.note:
                            st.note = (f"co-located current caps with "
                                       f"another owner at frame {frame} "
                                       f"(d={d:.1f} px) — permitted; "
                                       f"evidence insufficient for "
                                       f"duplicate exclusion")
                    overlaps.append({"frame": frame, "owners": [oi, oj],
                                     "distance_px": round(d, 2),
                                     "same_cap_evidence": bool(same)})
    return {"swaps": swaps, "n_duplicate_resolved": len(swaps),
            "skipped": skipped, "overlaps": overlaps}


def joint_select_over_time(
        candidates_by_owner: dict[str, dict[int, list[RouteHypothesis]]],
        frames: list[int], *,
        merge_tol_px: float = 6.0, clear_margin: float = 0.25,
        temporal_w: float = 1.0, jump_px: float = 7.5,
        switch_cost: float = 0.05,
        top_alternatives: int = 2) -> dict:
    """rev13 W2: JOINT selection over owner x path/cap x TIME.

    Replaces per-frame greedy pairwise demotion for the small
    three-owner interval. Each owner picks, per frame, ONE candidate
    hypothesis or the explicit UNRESOLVED option. The total score is

        sum_frames sum_owners [ local (cap) + whole (route) ]
        + temporal_w * sum_frames sum_owners [ -jump_penalty ]

    where the temporal term measures current-cap motion against the
    previous frame's accepted cap of the SAME owner using the ACTUAL
    source-frame gap: the budget is jump_px per source frame times the
    frame difference, and a larger motion costs its excess in px;
    presence/absence switches cost switch_cost.
    Abstention is a real option but never a free reset: an owner that
    abstains while candidates exist forfeits that frame's best
    available evidence score (declared). The cross-owner constraint (a
    clearly identified physical cap cannot be claimed by two owners in
    one frame) is enforced on every joint state. Exact enumeration over
    the joint state space (small: a few owners x <= K candidates) —
    auditable, no greedy steps.

    Returns {rows, total, n_states_visited, alternatives_by_row}.
    """
    import itertools

    owners = sorted(candidates_by_owner)
    if not owners:
        return {"rows": {}, "total": 0.0, "n_states_visited": 0,
                "alternatives_by_row": {}}
    # candidate list per owner/frame; index -1 == unresolved
    def _cands(oid, f):
        return list(candidates_by_owner[oid].get(f) or [])

    # prune per-frame states that violate cap-uniqueness
    def _legal(choice):
        claimed = []
        for oid, idx in zip(owners, choice):
            if idx < 0:
                continue
            h = _cands(oid, f)[idx]
            for oh, _oid2 in claimed:
                same, d = _same_physical_cap(h, oh, merge_tol_px)
                if d <= merge_tol_px and same:
                    return False
            claimed.append((h, oid))
        return True

    def _frame_states(f):
        opts = [list(range(-1, len(_cands(oid, f)))) for oid in owners]
        states = []
        for choice in itertools.product(*opts):
            if not _legal(choice):
                continue
            s = 0.0
            for oid, idx in zip(owners, choice):
                cands = _cands(oid, f)
                if idx < 0:
                    # rev13 W2: abstention forfeits the frame's best
                    # available evidence — it is a real choice, never a
                    # free reset of the motion history.
                    if cands:
                        s -= max(h.local_cap_score + h.whole_route_score
                                 for h in cands)
                    continue
                h = cands[idx]
                s += h.local_cap_score + h.whole_route_score
            states.append((choice, s))
        return states

    # DP over frames: state -> (cum_score, path)
    n_visited = 0
    prev: dict[tuple, tuple[float, list]] = {tuple([-1] * len(owners)):
                                             (0.0, [])}
    for fi, f in enumerate(frames):
        cur: dict[tuple, tuple[float, list]] = {}
        states = _frame_states(f)
        n_visited += len(states)
        for choice, s in states:
            for pchoice, (pscore, ppath) in prev.items():
                t = 0.0
                for oi_, oid in enumerate(owners):
                    idx, pidx = choice[oi_], pchoice[oi_]
                    if idx < 0 and pidx < 0:
                        continue
                    if (idx < 0) != (pidx < 0):
                        t -= switch_cost
                        continue
                    ca = _cands(oid, f)[idx].cap_xy()
                    pf = frames[fi - 1] if fi > 0 else None
                    cb = _cands(oid, pf)[pidx].cap_xy() if pf is not None \
                        else None
                    if ca is None or cb is None:
                        continue
                    d = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
                    # budget scales with the ACTUAL source-frame gap
                    _gap = max(1, int(f) - int(pf))
                    if d > jump_px * _gap:
                        t -= temporal_w * (d - jump_px * _gap)
                tot = pscore + s + t
                old = cur.get(choice)
                if old is None or tot > old[0]:
                    cur[choice] = (tot, ppath + [choice])
        prev = cur
        if not prev:
            break
    if not prev:
        return {"rows": {}, "total": 0.0, "n_states_visited": n_visited,
                "alternatives_by_row": {}}
    best = max(prev.items(), key=lambda kv: kv[1][0])
    bchoice, (btotal, bpath) = best
    rows: dict[str, dict[int, dict]] = {}
    alts: dict[str, dict[int, list]] = {}
    for fi, f in enumerate(frames):
        choice = bpath[fi] if fi < len(bpath) else bchoice
        for oi_, oid in enumerate(owners):
            idx = choice[oi_]
            cands = _cands(oid, f)
            # rev13 W2: the temporal component is recomputed along the
            # chosen trajectory and recorded per row. It compares the
            # current cap against the owner's most recent PRESENT frame
            # (consecutive-frame motion; an acquisition after an
            # unresolved gap is charged switch_cost, flagged as
            # re-acquisition — never an invented speed).
            t_term, cap_step, reacq = 0.0, None, False
            prev_present = None
            for f2 in range(fi - 1, -1, -1):
                c2 = bpath[f2] if f2 < len(bpath) else None
                if c2 is None:
                    continue
                i2 = c2[oi_]
                if i2 >= 0:
                    prev_present = (f2, _cands(oid, frames[f2])[i2])
                    break
            if idx < 0:
                if prev_present is not None:
                    t_term = -switch_cost  # lost presence this frame
            elif prev_present is None:
                t_term = -switch_cost
                reacq = fi > 0
            else:
                f2, h2 = prev_present
                ca = cands[idx].cap_xy()
                cb = h2.cap_xy()
                if ca is not None and cb is not None:
                    d = ((ca[0] - cb[0]) ** 2
                         + (ca[1] - cb[1]) ** 2) ** 0.5
                    cap_step = round(d, 2)
                    if f2 != fi - 1:
                        reacq = True
                        t_term = -switch_cost
                    else:
                        _gap = max(1, int(f) - int(frames[f2]))
                        if d > jump_px * _gap:
                            t_term = -temporal_w * (d - jump_px * _gap)
            row = {"owner": oid, "frame": f,
                   "state": "present" if idx >= 0 else "identity_uncertain",
                   "route_id": cands[idx].route_id if idx >= 0 else "",
                   "cand_index": int(idx),
                   "cap_xy": cands[idx].cap_xy() if idx >= 0 else None,
                   "reacquisition": bool(reacq),
                   "cap_step_px": cap_step,
                   "score_components": {
                       "local": cands[idx].local_cap_score if idx >= 0
                       else None,
                       "whole": cands[idx].whole_route_score if idx >= 0
                       else None,
                       "temporal": round(t_term, 4)}}
            rows.setdefault(oid, {})[f] = row
            others = [cands[k].route_id for k in range(len(cands))
                      if k != idx][:top_alternatives]
            alts.setdefault(oid, {})[f] = others
    return {"rows": rows, "total": btotal,
            "n_states_visited": n_visited,
            "alternatives_by_row": alts,
            "owners": owners, "frames": list(frames),
            "temporal_semantics": {
                "unit": "consecutive source frames; jump penalty is "
                        "excess px beyond jump_px",
                "reacquisition": "presence acquired after a gap: "
                                 "switch_cost, no invented speed"}}
