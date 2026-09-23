"""Annotation task scheduling (P0A, Qt-free).

Pure logic over task dicts (as returned by `AnnotationStore` or the
legacy importer): calibration batching, next-unfinished queue, training
mixture sampling (40% disagreement/failure, 40% random visible, 20%
anchor/control), per-owner/movie caps, and progress by stratum.

Disagreement ranking consumes caller-supplied scores (model/route
disagreement, visibility uncertainty, proposal absence) — this module
ranks, it never runs models.
"""

from __future__ import annotations

import random

MIX_FAILURE = 0.4
MIX_RANDOM = 0.4
MIX_ANCHOR = 0.2


def next_unfinished(tasks: list[dict]) -> dict | None:
    """First unfinished task in queue order (stable by uuid)."""
    pending = [t for t in tasks if not t.get("completed")]
    if not pending:
        return None
    return sorted(pending, key=lambda t: t.get("uuid", ""))[0]


def calibration_batch(tasks: list[dict], n: int = 12,
                      seed: int = 0) -> list[dict]:
    """Stratified starter set: spread across strata, deterministic."""
    rng = random.Random(seed)
    by_stratum: dict[str, list[dict]] = {}
    for t in tasks:
        by_stratum.setdefault(str(t.get("stratum", "")), []).append(t)
    batch: list[dict] = []
    strata = sorted(by_stratum)
    i = 0
    while len(batch) < n and any(by_stratum.values()):
        for s in strata:
            if len(batch) >= n:
                break
            pool = by_stratum[s]
            if pool:
                batch.append(pool.pop(rng.randrange(len(pool))))
        i += 1
        if i > n + 1:
            break
    return batch


def mixture_batch(
    tasks: list[dict],
    scores: dict[str, float],
    n: int,
    per_owner_cap: int = 3,
    seed: int = 0,
) -> list[dict]:
    """40/40/20 failure/random/anchor sample with per-owner caps."""
    rng = random.Random(seed)
    pool = [t for t in tasks if not t.get("completed")]
    if not pool:
        return []
    ranked = sorted(pool, key=lambda t: -scores.get(t.get("uuid", ""), 0.0))
    n_fail = int(n * MIX_FAILURE)
    n_rand = int(n * MIX_RANDOM)
    picked: list[dict] = []
    counts: dict[str, int] = {}

    def take(cands: list[dict], k: int) -> None:
        for t in cands:
            if len(picked) >= k:
                return
            o = str(t.get("owner_uuid", ""))
            if counts.get(o, 0) >= per_owner_cap:
                continue
            if t["uuid"] in {p["uuid"] for p in picked}:
                continue
            picked.append(t)
            counts[o] = counts.get(o, 0) + 1

    take(ranked, n_fail)
    rest = [t for t in pool if t["uuid"] not in {p["uuid"] for p in picked}]
    rng.shuffle(rest)
    take(rest, n_fail + n_rand)
    anchors = [t for t in pool if t.get("stratum") in ("clean", "control")
               and t["uuid"] not in {p["uuid"] for p in picked}]
    rng.shuffle(anchors)
    take(anchors + rest, n)
    return picked


def progress_by_stratum(tasks: list[dict]) -> dict[str, tuple[int, int]]:
    """stratum -> (completed, total)."""
    out: dict[str, tuple[int, int]] = {}
    for t in tasks:
        s = str(t.get("stratum", ""))
        done, total = out.get(s, (0, 0))
        out[s] = (done + int(bool(t.get("completed"))), total + 1)
    return out


GERMINATION_EPISODE_FIELDS = (
    "grain_id", "movie_id", "movie_content_hash", "window_start", "window_end",
    "unresolved_error", "why_existing_insufficient", "cheapest_answer",
    "consuming_loss_or_eval", "role", "before_after_comparison", "stratum",
)


def germination_episode(**fields) -> dict:
    """Propose one grain-episode annotation task (WO2).

    Every field in GERMINATION_EPISODE_FIELDS is required: a task with no
    consumer or no actionable comparison must not enter the queue. Returns
    a task dict with task_type 'germination_event'; the caller persists it.
    """
    missing = [f for f in GERMINATION_EPISODE_FIELDS
               if f not in fields or fields[f] is None or fields[f] == ""]
    if missing:
        raise ValueError(f"germination episode missing justification: {missing}")
    if fields["window_end"] < fields["window_start"]:
        raise ValueError("episode window_end < window_start")
    if fields["role"] not in ("training", "calibration", "heldout"):
        raise ValueError(f"bad episode role {fields['role']!r}")
    task = {"uuid": fields.get("uuid") or __import__("uuid").uuid4().hex,
            "task_type": "germination_event",
            "movie_uuid": fields["movie_id"],
            "owner_uuid": fields["grain_id"],
            "source_start": fields["window_start"],
            "source_end": fields["window_end"],
            "stratum": fields["stratum"],
            "role": {"training": "training", "calibration": "development",
                     "heldout": "reference"}[fields["role"]],
            "geometry_type": "event",
            "class_scope": "grain",
            "instance_scope": fields["grain_id"],
            "completeness": "unreviewed",
            "completed": False,
            "episode_justification": {f: fields[f] for f in GERMINATION_EPISODE_FIELDS}}
    return task
