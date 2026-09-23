"""Entity-level diff between two v30 snapshots (rev8 step 1).

Versioned, immutable snapshots are only durable if loss is DETECTABLE.
The rev7→rev8 loss (snap13 -> snap15: ten paths, two absence rows, six
masks, eight comparisons, four census tiles) was found by hand; this
tool makes it mechanical.

Usage:
    .venv/bin/python scripts/snapshot_diff.py OLD_DIR NEW_DIR \
        [--json out.json] [--fail-on-removal]

Exit codes: 0 = no removals (or not requested), 2 = removals detected
with --fail-on-removal, 1 = usage/IO error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ENTITY_FILES = [
    ("observations", "observations.json", ("obs_uuid", "uuid")),
    ("regions", "regions.json", ("_region_uuid", "region_uuid", "uuid")),
    ("body_masks", "body_masks.json", ("mask_uuid", "_mask_uuid", "uuid")),
    ("duels", "duels.json", ("_duel_uuid", "duel_uuid", "uuid")),
    ("crossings", "crossings.json", ("_xing_uuid", "xing_uuid", "uuid")),
    ("census", "census.json", ("task_uuid", "uuid")),
    ("tubes", "tubes.json", ("task_uuid", "uuid")),
]


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise SystemExit(f"unreadable {path}: {e}")
    if not isinstance(rows, list):
        raise SystemExit(f"{path} is not a JSON list")
    return rows


def _key(row: dict, candidates: tuple[str, ...]) -> str:
    for k in candidates:
        v = row.get(k)
        if v not in (None, ""):
            return f"{k}={v}"
    # fall back to a content key so anonymous rows still diff by content
    return "content:" + json.dumps(row, sort_keys=True)[:64]


def _entity_diff(old: list[dict], new: list[dict],
                 candidates: tuple[str, ...]) -> dict:
    o = {_key(r, candidates): r for r in old}
    n = {_key(r, candidates): r for r in new}
    added = sorted(set(n) - set(o))
    removed = sorted(set(o) - set(n))
    changed = []
    for k in sorted(set(o) & set(n)):
        if json.dumps(o[k], sort_keys=True) != json.dumps(n[k], sort_keys=True):
            fields = sorted(
                f for f in set(o[k]) | set(n[k])
                if json.dumps(o[k].get(f), sort_keys=True)
                != json.dumps(n[k].get(f), sort_keys=True))
            changed.append({"key": k, "fields": fields})
    return {"n_old": len(old), "n_new": len(new),
            "added": added, "removed": removed, "changed": changed}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--json", default="")
    ap.add_argument("--fail-on-removal", action="store_true")
    a = ap.parse_args()

    old_dir, new_dir = Path(a.old), Path(a.new)
    for d in (old_dir, new_dir):
        if not d.is_dir():
            print(f"not a directory: {d}")
            return 1

    report: dict = {"old": str(old_dir), "new": str(new_dir), "entities": {}}
    total_removed = 0
    print(f"snapshot diff\n  OLD {old_dir}\n  NEW {new_dir}\n")
    for name, fname, keys in ENTITY_FILES:
        d = _entity_diff(_load(old_dir / fname), _load(new_dir / fname), keys)
        report["entities"][name] = d
        total_removed += len(d["removed"])
        flag = ""
        if d["removed"]:
            flag = f"  <-- {len(d['removed'])} REMOVED"
        elif d["added"] and not d["n_old"]:
            flag = "  (first record)"
        print(f"  {name:<12} {d['n_old']:>4} -> {d['n_new']:>4}   "
              f"+{len(d['added'])} -{len(d['removed'])} "
              f"~{len(d['changed'])}{flag}")
        for k in d["removed"][:10]:
            print(f"        removed: {k}")
        for c in d["changed"][:5]:
            print(f"        changed: {c['key']} fields={c['fields']}")
    # label-consumption-relevant summary (the numbers a reviewer asks for)
    obs_new = _load(new_dir / "observations.json")
    report["summary"] = {
        "n_observations": len(obs_new),
        "n_path_samples": sum(
            1 for o in obs_new if o.get("path_complete")
            and len(o.get("path_xy") or []) >= 2),
        "n_masks": len(_load(new_dir / "body_masks.json")),
        "n_masks_linked": sum(
            1 for m in _load(new_dir / "body_masks.json")
            if m.get("source_obs_uuid") or m.get("linked_obs_uuid")),
        "n_duels": len(_load(new_dir / "duels.json")),
        "n_census": len(_load(new_dir / "census.json")),
    }
    print("\n  summary (NEW):", json.dumps(report["summary"]))
    print(f"\n  total removed rows: {total_removed}")
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1, default=str))
        print(f"  wrote {a.json}")
    if total_removed and a.fail_on_removal:
        print("FAIL: removals detected (--fail-on-removal)")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
