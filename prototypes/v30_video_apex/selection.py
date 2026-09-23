"""The ONE selection policy (rev8).

The rev7 review found the offline evaluator and the runtime runner
disagreed: the evaluator gated evidence then maximized `route_p`; the
runner gated evidence then kept the `q_max x route_p` ordering. That
produced different winners in four of eight events and made offline
conclusions inapplicable to the pipeline.

Everything that picks a winner now calls `select_candidate`. The
decision (including refusals) is returned as data so both paths can
print and store exactly what happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SelectionPolicy:
    """Declared, serializable selection policy."""

    rank: str = "sel"            # sel | route_p | q_max  (score to maximize)
    evidence_gate: float = 0.0   # relative gate; 0 disables
    evidence_key: str = "ev"     # per-candidate evidence field
    min_score: float = 0.0       # absolute floor on the ranked score
    min_evidence: float = 0.0    # absolute floor on the event's best evidence

    def as_dict(self) -> dict:
        return {"rank": self.rank, "evidence_gate": self.evidence_gate,
                "evidence_key": self.evidence_key,
                "min_score": self.min_score,
                "min_evidence": self.min_evidence}


@dataclass
class SelectionDecision:
    winner: dict | None = None
    refused: bool = False
    reason: str = ""
    n_candidates: int = 0
    n_kept: int = 0
    best_evidence: float = 0.0
    ranked_by: str = ""
    policy: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"winner": (self.winner or {}).get("route_id"),
                "refused": self.refused, "reason": self.reason,
                "n_candidates": self.n_candidates,
                "n_kept": self.n_kept,
                "best_evidence": self.best_evidence,
                "ranked_by": self.ranked_by, "policy": self.policy}


def _score(row: dict, policy: SelectionPolicy) -> float:
    if policy.rank == "sel":
        s = row.get("sel")
        if s is None:
            s = float(row.get("q_max") or 0.0) * float(
                row.get("route_p") or 0.0)
        return float(s)
    return float(row.get(policy.rank) or 0.0)


def select_candidate(rows: list[dict],
                     policy: SelectionPolicy) -> SelectionDecision:
    """Pick a winner from candidate rows, or refuse with a reason.

    Rules (all recorded, none silent):
      * no rows                      -> refused, "no-candidates"
      * relative evidence gate       -> candidates below
        `evidence_gate x best` cannot win
      * absolute evidence floor      -> if the best evidence is itself
        below `min_evidence`, nothing can win ("evidence-below-floor":
        zero support everywhere is a refusal, not a winner)
      * absolute score floor         -> a winner below `min_score` is
        refused ("score-below-floor")
    """
    pol = policy.as_dict()
    dec = SelectionDecision(policy=pol)
    if not rows:
        dec.refused, dec.reason = True, "no-candidates"
        return dec
    dec.n_candidates = len(rows)
    ev_key = policy.evidence_key
    evs = [float(r.get(ev_key) or 0.0) for r in rows]
    dec.best_evidence = max(evs)
    pool = rows
    if policy.evidence_gate > 0.0:
        best = dec.best_evidence
        if best > 0.0:
            pool = [r for r in rows
                    if float(r.get(ev_key) or 0.0)
                    >= policy.evidence_gate * best]
            if not pool:
                dec.refused, dec.reason = True, "evidence-gate-empty"
                dec.n_kept = 0
                return dec
    dec.n_kept = len(pool)
    if policy.min_evidence > 0.0 and dec.best_evidence < policy.min_evidence:
        dec.refused = True
        dec.reason = "evidence-below-floor"
        return dec
    dec.ranked_by = policy.rank
    winner = max(pool, key=lambda r: _score(r, policy))
    if _score(winner, policy) < policy.min_score:
        dec.refused = True
        dec.reason = "score-below-floor"
        return dec
    dec.winner = winner
    dec.reason = "selected"
    return dec
