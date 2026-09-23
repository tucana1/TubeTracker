"""rev12 P0.2: durable physical-grain IDs (append-only migration map).

Every CLICKED grain — owner-task grains (the balls the user marked) and
owned-absence grains — gets one durable, movie-qualified physical ID
(`<movie>|grain-NNNN`). Old identifiers (task-local labels, region
uuids, the legacy `own-ld-XXXX` owner ids) are preserved as ALIASES;
nothing in the source databases is rewritten.

Rules:
- IDs are assigned once and never reused; re-running the builder keeps
  existing assignments and only appends newly seen grains.
- Matching is by declared provenance keys (project|task|label and
  region uuid), never by row order; geometry is recorded for audit.
- `linkage`: "owner-linked" when the source task carries a real owner
  observation; otherwise "pending" (a local grain query whose owner
  identity is not yet established). The six rev11o-002 grains get six
  different IDs; their owner link stays whatever the annotation says.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tubetracker.grain_registry import apply_reviewed_links, register_alias_record
from tubetracker.review_semantics import is_workflow_record

OWNERS = REPO / "runs/prototypes/v30/rev11own_verify/owners.json"
SNAP = REPO / "runs/prototypes/v30/snap26_rev14"
OUT = REPO / "runs/prototypes/v30/grain_ids.json"
# rev13 W5.1: explicit many-observations-to-one-grain links (applied
# before the rebuild; the registry stays append-only).
LINKS = REPO / "runs/prototypes/v30/rev13_identity_links.json"

SOURCE_DIRS = [
    REPO / "runs/prototypes/v30/snap25_sources",
    Path("/Users/joshjiang/Documents/TubeTracker-annotator-projects"),
]


def _scan_owner_tasks():
    """Yield (project, movie, task_uuid, owner_uuid, frame, label, xy,
    no_tube) for every owner-task grain in every scanned database."""
    for root in SOURCE_DIRS:
        if not root.exists():
            continue
        for db in sorted(root.glob("*/annotations.db")):
            try:
                con = sqlite3.connect(db)
                rows = con.execute(
                    "select uuid, data from entities where kind='task'"
                ).fetchall()
                con.close()
            except sqlite3.Error:
                continue
            proj = db.parent.name
            for _u, d in rows:
                try:
                    t = json.loads(d)
                except (TypeError, ValueError):
                    continue
                if str(t.get("task_type", "")) != "owner":
                    continue
                if is_workflow_record(t):
                    continue
                grains = t.get("grains") or {}
                if not isinstance(grains, dict):
                    continue
                fr = int((t.get("query_frames") or [-1])[0])
                for label, g in sorted(grains.items()):
                    xy = (g or {}).get("xy") or []
                    if len(xy) != 2:
                        continue
                    yield {
                        "project": proj,
                        "movie": str(t.get("movie", "") or ""),
                        "task_uuid": str(t.get("uuid", "") or _u),
                        "owner_uuid": str(t.get("owner_uuid", "") or ""),
                        "frame": fr, "label": str(label),
                        "xy": [float(xy[0]), float(xy[1])],
                        "no_tube": bool((g or {}).get("no_tube")),
                    }


def _scan_absence_regions():
    reg = SNAP / "regions.json"
    if not reg.exists():
        return []
    rows = json.loads(reg.read_text())
    out = []
    for r in rows:
        if r.get("kind") != "owned_absence":
            continue
        grain = r.get("grain_xy") or []
        if len(grain) != 2:
            continue
        out.append({
            # normalize to the project DIRECTORY NAME so the region's
            # provenance key matches the task-scan key exactly
            "project": Path(str(r.get("_project", ""))).name,
            "movie": str(r.get("movie_uuid", "") or ""),
            "task_uuid": str(r.get("task_uuid", "")),
            "owner_uuid": str(r.get("owner_uuid", "") or ""),
            "frame": int(r.get("source_frame", -1)),
            "label": str(r.get("_region_uuid", "")).rsplit("-", 1)[-1],
            "region_uuid": str(r.get("_region_uuid", "")),
            "xy": [float(grain[0]), float(grain[1])],
        })
    return out


def main(argv=None) -> int:
    global OUT, SNAP, LINKS, OWNERS, SOURCE_DIRS
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--init-registry", type=Path)
    parser.add_argument("--snapshot", type=Path, default=SNAP)
    parser.add_argument("--links", type=Path, default=LINKS)
    parser.add_argument("--owners", type=Path, default=OWNERS)
    parser.add_argument("--source-root", type=Path, action="append")
    args = parser.parse_args(argv)
    OUT, SNAP, LINKS, OWNERS = args.out, args.snapshot, args.links, args.owners
    if args.source_root is not None:
        SOURCE_DIRS = args.source_root
    prior = OUT if OUT.exists() else args.init_registry
    existing = json.loads(prior.read_text()) if prior and prior.exists() else {}
    grains: dict[str, dict] = existing.get("grains", {})
    aliases: dict[str, str] = existing.get("aliases", {})

    links_doc = json.loads(LINKS.read_text()) if LINKS.exists() else {"links": []}
    existing["grains"], existing["aliases"] = grains, aliases
    explicit = apply_reviewed_links(existing, links_doc.get("links", []))

    def _register(rec, alias_keys):
        return register_alias_record(existing, rec, alias_keys, explicit)

    # collect candidates with their provenance keys
    cands: list[tuple[dict, list[str]]] = []
    for o in _scan_owner_tasks():
        rec = {"movie": o["movie"], "grain_native": o["xy"],
               "first_seen_frame": o["frame"],
               "no_tube": o["no_tube"],
               "linkage": ("owner-linked" if o["owner_uuid"]
                           not in ("", "unassigned") else "pending"),
               "provenance": {"project": o["project"],
                              "task": o["task_uuid"],
                              "label": o["label"], "kind": "owner-task"}}
        keys = [f"task|{o['project']}|{o['task_uuid']}|{o['label']}",
                f"emerge|{o['project']}|{o['frame']}|{o['label']}"]
        cands.append((rec, keys))
    for r in _scan_absence_regions():
        rec = {"movie": r["movie"], "grain_native": r["xy"],
               "first_seen_frame": r["frame"], "no_tube": True,
               "linkage": ("owner-linked" if r["owner_uuid"]
                           not in ("", "unassigned") else "pending"),
               "provenance": {"project": r["project"],
                              "task": r["task_uuid"], "label": r["label"],
                              "kind": "owned-absence-region"}}
        keys = [f"region|{r['region_uuid']}"]
        if r["project"] and r["task_uuid"] and r["label"]:
            # the region was built FROM the task's grain click: the
            # task-provenance key unifies the two records as one grain
            keys.append(f"task|{r['project']}|{r['task_uuid']}"
                        f"|{r['label']}")
        cands.append((rec, keys))
    # the legacy owner registry: same grains, their own IDs as aliases
    if OWNERS.exists():
        for o in json.loads(OWNERS.read_text())["owners"]:
            rec = {"movie": o.get("movie", "ld"),
                   "grain_native": o["grain_native"],
                   "first_seen_frame": int(o.get("identified_at_frame",
                                                 -1)),
                   "no_tube": bool(o.get("no_tube")),
                   "linkage": "owner-linked",
                   "provenance": {"project": "rev11own",
                                  "task": o.get("source_task", ""),
                                  "label": o.get("source_label", ""),
                                  "kind": "owner-registry"}}
            keys = [f"owner|{o['id']}"]
            if o.get("source_task") and o.get("source_label"):
                keys.append(f"task|rev11own|{o['source_task']}"
                            f"|{o['source_label']}")
            cands.append((rec, keys))

    # assign IDs in a stable geometric order (movie, frame, y, x); the
    # append-only alias map keeps previously assigned IDs fixed.
    taken = set(grains) | set(existing.get("tombstones", {}))
    n = 0
    for rec, keys in sorted(cands, key=lambda c: (
            c[0]["movie"], c[0]["first_seen_frame"],
            round(c[0]["grain_native"][1], 1),
            round(c[0]["grain_native"][0], 1))):
        keys = [k for k in keys if not k.startswith("emerge|") or k in explicit]
        if any(k in aliases for k in keys):
            _register(rec, keys)
            continue
        n += 1
        while f"{rec['movie']}|grain-{n:04d}" in taken:
            n += 1
        rec = dict(rec, grain_id=f"{rec['movie']}|grain-{n:04d}")
        _register(rec, keys)

    doc = {
        "generated_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "note": "Durable physical grains; explicit reviewed merges redirect aliases and retain absorbed records as tombstones.",
        "n_grains": len(grains),
        "n_aliases": len(aliases),
        "grains": grains,
        "aliases": aliases,
        "tombstones": existing.get("tombstones", {}),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"grains: {len(grains)} | aliases: {len(aliases)} -> {OUT}")
    for gid, g in sorted(grains.items()):
        _gn = g.get("grain_native")
        _pos = (f"{[round(v, 1) for v in _gn]}" if _gn
                else "<pending: no direct click; linked observations "
                     "carry the geometry>")
        print(f"  {gid}: {g['movie']} f{g['first_seen_frame']} "
              f"{_pos} "
              f"({g['linkage']}, {g['provenance']['kind']}"
              f"{', no_tube' if g.get('no_tube') else ''})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
