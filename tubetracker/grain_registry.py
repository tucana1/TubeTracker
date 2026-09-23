"""Durable grain identities, explicit alias redirects and discovery history."""
from __future__ import annotations

import copy
import json
import math
import sqlite3
from pathlib import Path


def canonical_id(registry, grain_id):
    visited = set()
    while grain_id in registry.get("tombstones", {}):
        if grain_id in visited:
            raise ValueError("grain alias redirect cycle")
        visited.add(grain_id)
        grain_id = registry["tombstones"][grain_id]["canonical"]
    return grain_id


def _absorb(registry, old, target, evidence):
    old, target = canonical_id(registry, old), canonical_id(registry, target)
    if old == target:
        return
    grains = registry["grains"]
    prior = grains.get(old)
    current = grains[target]
    if prior and prior.get("movie") != current.get("movie"):
        raise ValueError("reviewed grain merges cannot cross movies")
    if prior:
        if current.get("grain_native") is None:
            current["grain_native"] = prior.get("grain_native")
            current["first_seen_frame"] = prior.get("first_seen_frame", -1)
        sources = current.setdefault("provenance_sources", [])
        for source in [prior.get("provenance"), *prior.get("provenance_sources", [])]:
            if source and source not in sources:
                sources.append(copy.deepcopy(source))
        registry.setdefault("tombstones", {})[old] = {
            "canonical": target, "record": copy.deepcopy(prior), "evidence": copy.deepcopy(evidence)}
        del grains[old]
    for alias, gid in list(registry["aliases"].items()):
        if gid == old:
            registry["aliases"][alias] = target


def apply_reviewed_links(registry, links):
    """Explicit biological identity links outrank every historical alias."""
    grains = registry.setdefault("grains", {})
    aliases = registry.setdefault("aliases", {})
    explicit = {}
    for link in links:
        target = str(link.get("canonical") or "")
        if not target:
            continue
        target = canonical_id(registry, target)
        obs = link.get("observations") or {}
        project, label = str(obs.get("project", "")), str(obs.get("label", ""))
        keys = [f"emerge|{project}|{int(frame)}|{label}" for frame in obs.get("frames", [])]
        if not link.get("basis") or not keys:
            raise ValueError("grain merge requires its reviewed basis and observation keys")
        grains.setdefault(target, {"movie": link.get("movie", ""), "grain_native": None,
            "first_seen_frame": -1, "linkage": "owner-linked",
            "provenance": {"project": "reviewed-links", "task": "", "label": "",
                           "kind": "explicit-merge"}})
        event = {"revision": int(link.get("revision", 1)), "absorbed_aliases": sorted(keys),
                 "basis": link["basis"]}
        history = grains[target].setdefault("merge_history", [])
        if event not in history:
            history.append(event)
        for key in keys:
            if key in explicit and explicit[key] != target:
                raise ValueError("contradictory reviewed grain identity links")
            explicit[key] = target
            if key in aliases:
                _absorb(registry, aliases[key], target, event)
            aliases[key] = target
    return explicit


def register_alias_record(registry, record, alias_keys, explicit=None):
    """One source grain may have several aliases; preserve every redirect."""
    explicit = explicit or {}
    # project/frame/label alone is ambiguous: several tasks can label
    # different grains "A" on the same frame. Only an explicit reviewed
    # link gives an emergence alias physical identity authority.
    alias_keys = [k for k in alias_keys if not k.startswith("emerge|") or k in explicit]
    declared = {canonical_id(registry, explicit[k]) for k in alias_keys if k in explicit}
    if len(declared) > 1:
        raise ValueError("one source observation maps to multiple reviewed grains")
    existing = {canonical_id(registry, registry["aliases"][k]) for k in alias_keys
                if k in registry["aliases"]}
    if len(existing) > 1 and not declared:
        raise ValueError("conflicting source-grain aliases require an explicit reviewed merge")
    target = next(iter(declared)) if declared else min(existing) if existing else record["grain_id"]
    current = registry["grains"].setdefault(target, {k: copy.deepcopy(v) for k, v in record.items()
                                                    if k != "grain_id"})
    for old in existing:
        _absorb(registry, old, target, {"basis": "same explicit source grain aliases",
                                      "aliases": list(alias_keys), "provenance": record["provenance"]})
    if current.get("movie") != record.get("movie"):
        raise ValueError("grain alias points into another movie")
    if current.get("grain_native") is None:
        current["grain_native"] = copy.deepcopy(record.get("grain_native"))
    frame = int(record.get("first_seen_frame", -1))
    if frame >= 0 and (current.get("first_seen_frame", -1) < 0 or frame < current["first_seen_frame"]):
        current["first_seen_frame"] = frame
    if record.get("linkage") == "owner-linked":
        current["linkage"] = "owner-linked"
    sources = current.setdefault("provenance_sources", [])
    if record["provenance"] not in sources:
        sources.append(copy.deepcopy(record["provenance"]))
    for key in alias_keys:
        registry["aliases"][key] = target
    return target


class DiscoveryRegistry:
    """A separate transactional store; pixel-cache removal cannot erase IDs."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def reconcile(self, movie, movie_hash, frame, seeds, detections):
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            connection.execute("CREATE TABLE IF NOT EXISTS grains "
                               "(movie_hash TEXT, grain_id TEXT, data TEXT, "
                               "PRIMARY KEY(movie_hash,grain_id))")
            connection.execute("BEGIN IMMEDIATE")
            known = {gid: json.loads(data) for gid, data in connection.execute(
                "SELECT grain_id,data FROM grains WHERE movie_hash=?", (movie_hash,))}
            owners = {o["id"]: copy.deepcopy(o) for o in seeds}
            for oid, owner in owners.items():
                if owner.get("movie", movie) != movie:
                    raise ValueError("discovery seed belongs to a different movie")
                old = known.get(oid, {})
                known[oid] = {**old, **copy.deepcopy(owner)}
                known[oid].setdefault("positions", {})
            assigned, proposed, reused, ambiguous, duplicates = set(), [], [], [], []
            for detection in sorted(detections, key=lambda d: (-d["score"], d["xy"])):
                xy, radius = detection["xy"], float(detection["radius"])
                if len(xy) != 2 or not all(math.isfinite(float(v)) for v in xy) or radius <= 0:
                    raise ValueError("discovery requires finite grain geometry")
                distances = sorted((math.dist(xy, o["grain_native"]), oid)
                    for oid, o in known.items() if o.get("grain_native") is not None)
                nearby = [(d, oid) for d, oid in distances if d <= max(7., .6*radius)]
                if nearby and nearby[0][1] in assigned:
                    duplicates.append({"xy": xy, "grain_id": nearby[0][1]})
                    continue
                if len(nearby) > 1 and nearby[1][0]-nearby[0][0] < 2.:
                    ambiguous.append({"xy": xy, "candidate_ids": [oid for _, oid in nearby],
                                      "reason": "physical grain match ambiguous"})
                    continue
                if nearby:
                    oid = nearby[0][1]
                    reused.append(oid)
                    if oid not in owners:
                        owner = {k: copy.deepcopy(v) for k, v in known[oid].items() if k != "positions"}
                        owner["grain_native"] = list(xy)
                        owner["source_frame"] = int(frame)
                        if math.dist(xy, known[oid]["grain_native"]) > 1:
                            owner["attachment_verified"] = False
                        owners[oid] = owner
                else:
                    serial = 1
                    prefix = f"{movie}|{movie_hash[:10]}|grain-"
                    while prefix+f"{serial:05d}" in known:
                        serial += 1
                    oid = prefix+f"{serial:05d}"
                    owners[oid] = {"id": oid, "movie": movie, "grain_native": list(xy),
                        "grain_radius_px": radius, "attachment_native": None,
                        "attachment_verified": False, "identity_verified": False,
                        "grain_geometry_complete": detection.get("geometry_complete", True),
                        "source_frame": int(frame),
                        "source": detection.get("source", "automatic radial-symmetry proposal")}
                    known[oid] = {**copy.deepcopy(owners[oid]), "positions": {}}
                    proposed.append(oid)
                assigned.add(oid)
                if oid not in {o["id"] for o in seeds}:
                    known[oid]["grain_native"] = list(xy)
                known[oid]["positions"][str(int(frame))] = {"xy": list(xy), "score": float(detection["score"])}
                owners[oid]["discovery_score"] = float(detection["score"])
            for oid, record in known.items():
                connection.execute("INSERT OR REPLACE INTO grains VALUES (?,?,?)",
                    (movie_hash, oid, json.dumps(record, sort_keys=True, allow_nan=False)))
            connection.commit()
            result = [owners[k] for k in sorted(owners)]
            return result, {"registry_path": str(self.path.resolve()), "reference_frame": int(frame),
                "new_ids": proposed, "matched_existing_ids": reused, "ambiguous": ambiguous,
                "duplicate_detections": duplicates, "completeness_certified": False}
        finally:
            connection.close()
