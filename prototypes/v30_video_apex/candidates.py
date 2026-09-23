"""One versioned candidate record (rev9 WP-A.5).

The rev9 audit replayed the eight saved gated events through the shared
selector and got **5/8** saved winners: the exported candidates carry
`tip_score` and nested `evidence.wall_ev`, while the selector ranks on
`sel`/`q_max` and gates on a flat `ev`. Nothing was wrong with the
candidates — the *serialization boundary* dropped the scores, so the
ranking fields read as zero and ties resolved by existing order.

This module is that boundary, once:

* `normalize_candidate(row)` maps a legacy row **once**, idempotently,
  stamps `candidate_schema`, and REJECTS a row that carries no score or
  no evidence (silence is not a valid candidate).
* `export_candidates` / `read_candidates` use the same object, so the
  runtime, the JSON export and every replay see identical rows.
* The path contract is explicit: `support_path` (the proposal), its
  truncation at `front_s` (`current_path`), the `current_tip`, and
  `length_px` == arclength of the accepted path including the declared
  root vertex. `path_consistency` checks that claim.

Nothing here rescores or retrains: it is schema, not modelling.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

CANDIDATE_SCHEMA_VERSION = 2

REQUIRED_IDENTITY_KEYS = ("candidate_id", "movie_id", "owner_id",
                          "source_frame")
SCORE_SOURCES = ("q_max", "tip_score")
EVIDENCE_SOURCES = ("ev", "wall_ev")


class CandidateSchemaError(ValueError):
    """A candidate row cannot be represented without guessing."""


def _first_present(row: dict, keys) -> tuple[str, object] | tuple[None, None]:
    for k in keys:
        if row.get(k) is not None:
            return k, row[k]
    return None, None


def _nested_evidence(row: dict) -> dict:
    ev = row.get("evidence")
    return ev if isinstance(ev, dict) else {}


def normalize_candidate(row: dict) -> dict:
    """Return the v2 form of a candidate row (idempotent).

    Legacy -> v2 mapping happens here and only here:
      * `tip_score`      -> `q_max`   (kept, provenance preserved)
      * `evidence.wall_ev` / `wall_ev` -> `ev`
      * `sel` = `q_max * route_p` when not already present

    Raises CandidateSchemaError when the row has no score or no evidence:
    a candidate whose support cannot be read must fail loudly rather
    than silently rank as zero.
    """
    if not isinstance(row, dict):
        raise CandidateSchemaError(f"candidate must be a dict, got {type(row)}")
    out = dict(row)
    for key in REQUIRED_IDENTITY_KEYS:
        if out.get(key) in (None, ""):
            raise CandidateSchemaError(f"candidate missing required key: {key}")

    score_key, score = _first_present(out, SCORE_SOURCES)
    if score_key is None:
        raise CandidateSchemaError(
            "candidate carries no score (looked for "
            f"{list(SCORE_SOURCES)})")
    try:
        q = float(score)
    except (TypeError, ValueError):
        raise CandidateSchemaError(f"candidate score {score!r} is not numeric")
    if not math.isfinite(q):
        raise CandidateSchemaError(f"candidate score is not finite: {q}")
    out["q_max"] = q

    ev_key, ev = _first_present(out, EVIDENCE_SOURCES)
    if ev_key is None:
        ev_key, ev = _first_present(_nested_evidence(out), ("wall_ev", "ev"))
    if ev_key is None:
        raise CandidateSchemaError(
            "candidate carries no evidence (looked for "
            f"{list(EVIDENCE_SOURCES)} and evidence.wall_ev)")
    try:
        out["ev"] = float(ev)
    except (TypeError, ValueError):
        raise CandidateSchemaError(f"candidate evidence {ev!r} not numeric")

    if out.get("sel") is None:
        route_p = out.get("route_p")
        out["sel"] = (q * float(route_p)) if route_p is not None else q
    out["candidate_schema"] = CANDIDATE_SCHEMA_VERSION
    return out


def normalize_rows(rows: list[dict]) -> list[dict]:
    return [normalize_candidate(r) for r in rows]


# ---------------------------------------------------------------- paths
def polyline_length(pts) -> float:
    """Arclength of a polyline in whatever units its vertices carry."""
    if not pts or len(pts) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        total += math.hypot(float(b[0]) - float(a[0]),
                            float(b[1]) - float(a[1]))
    return total


def accepted_path(row: dict, *, root_xy=None) -> dict:
    """Split support / front / current-path for one candidate.

    `support_path` is the proposal the scorer considered; `front_s` is
    where the accepted front sits along it; `current_path` is the
    support truncated there (plus the declared root vertex, so the
    attachment connector is included when the body route begins
    downstream); `length_px` is the arclength of that accepted path.
    """
    row = normalize_candidate(row) if "q_max" not in row else row
    support = [list(map(float, q)) for q in (row.get("polyline_native")
                                             or [])]
    front_s = row.get("front_s", row.get("s"))
    front_s = float(front_s) if front_s is not None else None
    root = list(map(float, root_xy)) if root_xy else None
    root_source = str(row.get("root_source", "") or "")
    cur = support
    if front_s is not None and len(support) >= 2:
        cur, acc = [support[0]], 0.0
        for a, b in zip(support[:-1], support[1:]):
            seg = math.hypot(float(b[0]) - float(a[0]),
                             float(b[1]) - float(a[1]))
            if acc + seg > front_s:
                # truncate inside this segment at exactly front_s
                t = 0.0 if seg == 0 else (front_s - acc) / seg
                cur.append([float(a[0]) + t * (float(b[0]) - float(a[0])),
                            float(a[1]) + t * (float(b[1]) - float(a[1]))])
                break
            cur.append(b)
            acc += seg
    if root is not None and (not cur or cur[0] != root):
        cur = [root] + cur
    tip = cur[-1] if cur else None
    return {"support_path": support, "front_s": front_s,
            "current_path": cur, "current_tip": tip,
            "root_xy": root, "root_source": root_source,
            "length_px": polyline_length(cur),
            "support_length_px": polyline_length(support)}


def path_consistency(row: dict, *, root_xy=None, tol_px: float = 1.0
                     ) -> tuple[bool, list[str]]:
    """Does the row's declared length/tip agree with its path?

    Returns (ok, problems). A row that carries `length_px` or
    `current_tip` must have them match the accepted path — the audit
    found consumers comparing a length taken from a different curve
    than the one exported.
    """
    problems: list[str] = []
    ap = accepted_path(row, root_xy=root_xy)
    if not ap["current_path"]:
        problems.append("empty accepted path")
        return False, problems
    declared_len = row.get("length_px")
    if declared_len is not None:
        if abs(float(declared_len) - ap["length_px"]) > tol_px:
            problems.append(
                f"declared length {float(declared_len):.2f} != arclength "
                f"{ap['length_px']:.2f}")
    declared_tip = row.get("current_tip")
    if declared_tip is not None and ap["current_tip"] is not None:
        d = math.hypot(float(declared_tip[0]) - ap["current_tip"][0],
                       float(declared_tip[1]) - ap["current_tip"][1])
        if d > tol_px:
            problems.append(
                f"declared tip is {d:.2f} px from the accepted path end")
    if ap["front_s"] is not None:
        if ap["front_s"] < -tol_px or (ap["support_length_px"]
                                       and ap["front_s"]
                                       > ap["support_length_px"] + tol_px):
            problems.append(
                f"front_s {ap['front_s']:.2f} outside the support "
                f"(len {ap['support_length_px']:.2f})")
    return (not problems), problems


# ---------------------------------------------------------------- io
def export_candidates(rows: list[dict]) -> list[dict]:
    """The v2 JSON form (same objects the runtime selects on).

    rev10 WP-C: every exported candidate states its PROVENANCE. A
    downstream consumer must be able to separate automatic proposals
    from oracle-substituted diagnostics without guessing: the interval
    run's first export had `route_source: None` on all 220 rows, so a
    coverage number could not be attributed to the proposer at all.
    """
    out = normalize_rows(rows)
    for r in out:
        ev = r.get("evidence")
        if isinstance(ev, dict):
            ev.setdefault("route_source", "proposed")
        else:
            r["evidence"] = {"route_source": "proposed"}
    return out


def read_candidates(path: str | Path) -> list[dict]:
    """Read candidates from disk, normalized (legacy files included)."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):       # tolerate {"candidates": [...]}
        data = data.get("candidates", [])
    if not isinstance(data, list):
        raise CandidateSchemaError(
            f"{path}: expected a list of candidates")
    return normalize_rows(data)
