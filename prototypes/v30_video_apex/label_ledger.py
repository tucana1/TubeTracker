"""Per-label / per-head consumption table (rev9 WP-A.6).

The reviewer's rule: "Do not infer consumed supervision from sample
counts." Eleven `comparison` samples in the snapshot are not eleven
consumed route judgments; a label that is quarantined must contribute
ZERO to every head, and the run must be able to show exactly which
unique labels fed which head, with their pixel/route counts and the
reason any label was excluded.

`build_consumption_table` turns the trainer's `note()` bookkeeping into
that table and *fails* when a quarantined sample appears among consumed
keys — the failure mode the reviewer called out (default visibility
supervision leaking to quarantined comparisons).
"""
from __future__ import annotations

from collections import Counter

# Sample kinds and the heads they are allowed to feed. A head outside
# this map is a design error, not a warning.
# Head keys are the target-mask keys the trainer notes: apex_pos,
# apex_neg, vis_valid, body_valid, body_confusable, front, route.
CONSUMER_MAP = {
    # rev10: a path sample that carries a LINKED body mask is genuinely
    # body-supervised, so the mask's own region channels are sanctioned
    # here. A ribbon-only path (no mask) emits none of them -- the
    # trainer builds the band inside the linked-mask branch -- so this
    # cannot launder an unsupervised region into the loss.
    "path_tip": ("apex_pos", "apex_neg", "vis_valid", "body_valid",
                 "body_confusable", "front", "route",
                 "body_band", "body_bg_reviewed", "body_overlap"),
    "tip_only": ("apex_pos", "vis_valid"),
    "no_tube": ("vis_valid",),                # owned presence/visibility
    "neg_region": ("apex_neg",),              # verified cap-free regions
    # a body-mask crop may also carry an explicitly reviewed cap-free
    # region at its frame (frame-global truth), but never visibility
    "body_mask": ("body_valid", "body_confusable", "apex_neg",
                  "body_band", "body_bg_reviewed", "body_overlap"),
    "comparison": ("route",),                 # pairwise lane preference
    "census_tile": ("apex_pos", "vis_valid"),  # discovery over a tile
}


def build_consumption_table(samples, usage: dict[str, set],
                            counts: dict | None = None,
                            kind_of: dict | None = None,
                            strict: bool = True) -> dict:
    """Return the consumption table; raise on quarantined consumption.

    samples   : the SampleDef list the run was built from
    usage     : head -> set(sample_key) recorded by the trainer's note()
    counts    : optional head -> {key: {metric: value}} (pixels, routes)
    kind_of   : optional key -> kind (defaults to the sample's own kind)
    """
    by_key = {}
    for s in samples:
        key = getattr(s, "sample_key", "") or getattr(s, "entry_id", "")
        by_key[key] = s
    kind_of = dict(kind_of or {})
    for key, s in by_key.items():
        kind_of.setdefault(key, str(getattr(s, "kind", "")))

    quarantined = {k: s for k, s in by_key.items()
                   if str(getattr(s, "quarantine_reason", "") or "")}
    leaked = sorted({k for v in usage.values() for k in v
                     if k in quarantined})
    if leaked:
        reasons = {k: quarantined[k].quarantine_reason for k in leaked[:5]}
        raise ValueError(
            f"quarantined labels contributed supervision: {reasons}")

    heads: dict[str, dict] = {}
    for head, keys in sorted(usage.items()):
        entry = {"n_unique": len(keys), "keys": sorted(keys)[:2000],
                 "kinds": dict(Counter(kind_of.get(str(k), "?")
                                       for k in keys)),
                 "counts": {}}
        if counts:
            for k in sorted(keys)[:2000]:
                if k in counts.get(head, {}):
                    entry["counts"][str(k)] = counts[head][k]
        allowed = Counter()
        off_map = []
        for k in keys:
            kind = kind_of.get(str(k), "?")
            allowed[kind] += 1
            if kind in CONSUMER_MAP and head not in CONSUMER_MAP[kind]:
                off_map.append(f"{kind}->{head}")
        entry["kinds"] = dict(allowed)
        if off_map:
            entry["off_consumer_map"] = sorted(set(off_map))
            # rev10 WP-A: an off-map consumer is a design error, not a
            # warning. The previous form recorded it and let the run
            # continue ("records an off-map band consumer without
            # failing" — the review).
            if strict:
                raise ValueError(
                    f"off-map label consumers: {sorted(set(off_map))} "
                    f"(head {head!r} may not consume those kinds; see "
                    f"CONSUMER_MAP)")
        heads[head] = entry

    return {
        "heads": heads,
        "unique_total": len({k for v in usage.values() for k in v}),
        "quarantined_labels": len(quarantined),
        "quarantined_consumed": [],
        "consumer_map": {k: list(v) for k, v in sorted(CONSUMER_MAP.items())},
    }
