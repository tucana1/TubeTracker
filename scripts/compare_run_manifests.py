"""rev12 P0.4 → rev13 W1: mechanical run-manifest comparator.

A proposed "one-variable" comparison is REJECTED when any undeclared
setting, initialization, data or update budget differs. The verdict is
computed from the serialized manifests — never from prose.

rev13 W1: the comparison is RECURSIVE over the canonical input
manifest (snapshot content hashes, split IDs, owner/annotation
versions, initialization, architecture, losses/domains, sampler,
update budget, preprocessing). Input checkpoint hashes are compared by
schema — a field named `checkpoint`/`weights` is an INPUT unless it is
an output-artifact path — instead of ignoring every field with those
names. Identical runs report as `identical`. Only exact declared paths
(and their descendants) can absorb a difference.

Usage:
  python scripts/compare_run_manifests.py A.json B.json \
      --variable body.bg_region
  python scripts/compare_run_manifests.py A.json B.json \
      --variable updates --out comparison.json

Exit code 0 = the manifests differ ONLY in the declared variable
(and/or its direct consequences); 1 = the claim does not hold (or the
runs are identical and no variable changed); 2 = unusable input. The
full field diff is always printed and written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Fields that are OUTPUTS (results, timestamps, paths of results), not
# configuration: they never enter the comparison — at any depth. Note
# that `checkpoint`, `weights` and `param_hash` are deliberately NOT
# here: an input checkpoint/weights/hash is configuration and must
# invalidate a claimed comparison (rev13 audit: a changed input was
# invisible). Output-artifact paths are named `*_path`/`out`/`report`.
_RESULT_KEYS = {
    "history", "events", "rows", "cases", "visibility_confusion",
    "samples", "label_usage", "run_manifest", "report", "results",
    "out", "out_path", "log", "notes", "note", "duration_s",
    "elapsed_s", "wall_s", "finished_utc", "started_utc",
    "best_epoch", "best_mean_dev_front_px", "checkpoint_path",
    "checkpoint_out", "resolved", "verification", "gate",
    "gate_passed", "absence_gate_passed", "skipped_fit", "fit_gate",
    "owned_absence_case_pre", "owned_absence_case_post", "timings",
    "wall_clock", "per_epoch_s", "epoch_seconds",
}

# Long non-result lists are compared by content hash, not elementwise.
_LIST_HASH_THRESHOLD = 200


def _flatten(prefix: str, obj, out: dict) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(obj, (list, tuple)):
        if len(obj) > _LIST_HASH_THRESHOLD:
            blob = json.dumps(obj, sort_keys=True, default=str)
            out[prefix] = {"__len": len(obj),
                           "__sha256_12": hashlib.sha256(
                               blob.encode()).hexdigest()[:12]}
        else:
            out[prefix] = json.loads(
                json.dumps(obj, sort_keys=True, default=str))
    else:
        out[prefix] = obj


def load_config(path: Path) -> dict:
    """All CONFIGURATION fields of a run manifest, flattened
    RECURSIVELY (nested input structures — snapshot content hashes,
    split IDs, owner versions, initialization — are compared in full,
    not only their scalar members)."""
    raw = json.loads(path.read_text())
    flat: dict = {}
    # Canonical manifests explicitly distinguish immutable inputs from
    # outputs. Never filter keys by their spelling within the input tree.
    if raw.get("manifest_schema") == "tubetracker.run.v1":
        if not isinstance(raw.get("inputs"), dict):
            raise ValueError("canonical run manifest requires inputs")
        _flatten("inputs", raw["inputs"], flat)
        return flat
    for key, val in raw.items():
        if key in _RESULT_KEYS:
            continue
        _flatten(key, val, flat)
    return flat


def _is_declared(field: str, variable: str) -> bool:
    """Only the exact declared path or its descendants (or the
    declared path's own container) may absorb a difference."""
    v = variable.strip()
    if not v:
        return False
    return (field == v or field.startswith(v + ".")
            or v.startswith(field + "."))


def compare(a_path: Path, b_path: Path, variable: str) -> dict:
    a, b = load_config(a_path), load_config(b_path)
    diffs = []
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k, "<absent>"), b.get(k, "<absent>")
        if va != vb:
            diffs.append({"field": k, "a": va, "b": vb})
    declared = [d for d in diffs if _is_declared(d["field"], variable)]
    undeclared = [d for d in diffs if d not in declared]
    if not diffs:
        verdict = "identical"
        note = ("the manifests are identical; the declared variable "
                f"{variable!r} did not differ — a one-variable "
                "experiment needs the intended changed variable")
    elif undeclared:
        verdict = "NOT-single-variable"
        note = ("undeclared differences confound the comparison")
    else:
        verdict = "single-variable"
        note = (f"every differing configuration field is {variable!r} "
                "or a descendant of it")
    return {"a": str(a_path), "b": str(b_path),
            "variable": variable, "verdict": verdict, "note": note,
            "declared_diffs": declared, "undeclared_diffs": undeclared,
            "n_diffs": len(diffs),
            "rule": ("a one-variable claim requires that EVERY "
                     "differing configuration field is the declared "
                     "variable (or a subfield of it); anything else "
                     "confounds the comparison")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--variable", required=True,
                    help="the single declared varying field")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    for p in (a.a, a.b):
        if not Path(p).exists():
            print(f"compare_run_manifests: missing input {p}",
                  file=sys.stderr)
            return 2
    res = compare(Path(a.a), Path(a.b), a.variable)
    txt = json.dumps(res, indent=1)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n")
    return 0 if res["verdict"] == "single-variable" else 1


if __name__ == "__main__":
    raise SystemExit(main())
