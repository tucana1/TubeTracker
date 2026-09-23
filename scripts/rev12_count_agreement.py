"""rev12 P0.4: count agreement across status/manifest/builder/gate.

Recomputes the headline counts from their SOURCES and compares them
with the numbers recorded in the artifacts. Exits nonzero when any
unit disagrees — counts must agree across the target builder, the
manifest, the gate and the status report; prose is not a source.

Checks (all recomputed, never copied from a report):
- no_tube inventory: 42 = 35 no_tube_visible + 6 not_directly_visible
  + 1 owner_uncertain; 41 unique (movie, frame, focus) triples;
  9 certified grain-scoped records (one per physical grain ID).
- proposal census: 277 rows = 23 present + 212 absent + 42 unknown;
  the vertex-geometry census 20/234/23 and the 24-change ledger.
- absence gate: cases == certified records == 9.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SNAP = REPO / "runs/prototypes/v30/snap25_plus_rev11own"
PROPS = REPO / "runs/prototypes/v30/rev12_proposals.json"
GATE = REPO / "runs/prototypes/v30/rev12_absence_gate_absdiet_ms.json"
OUT = REPO / "runs/prototypes/v30/rev12_count_agreement.json"

EXPECT = {
    "no_tube_records": 42,
    "no_tube_visible": 35,
    "not_directly_visible": 6,
    "owner_uncertain": 1,
    "unique_triples": 41,
    "certified": 9,
    "gate_cases": 9,
    "proposals": 277,
    "present": 23,
    "absent": 212,
    "unknown": 42,
    "vertex_present": 20,
    "vertex_absent": 234,
    "vertex_unknown": 23,
    "label_changes": 24,
}


def main() -> int:
    from prototypes.v30_video_apex.targets import samples_from_snapshot
    got: dict = {}
    ss = samples_from_snapshot(str(SNAP))
    nt = [s for s in ss if s.kind == "no_tube"]
    got["no_tube_records"] = len(nt)
    from collections import Counter
    st = Counter(str(s.direct_state) for s in nt)
    got["no_tube_visible"] = int(st.get("no_tube_visible", 0))
    got["not_directly_visible"] = int(st.get("not_directly_visible", 0))
    got["owner_uncertain"] = int(st.get("owner_uncertain", 0))
    got["unique_triples"] = len({
        (s.movie, s.source_frame,
         (round(s.focus_xy[0], 1), round(s.focus_xy[1], 1)))
        for s in nt if s.focus_xy})
    cert = [s for s in nt if getattr(s, "certified", False)]
    got["certified"] = len(cert)
    got["certified_unique_grains"] = len({
        str(getattr(s, "grain_id", "")) for s in cert})
    # rev13 W1: every proposal count is RECOMPUTED from the per-row
    # labels and the change ledger. The manifest's headline aggregates
    # are then cross-checked against the recomputation — a headline
    # that cannot be regenerated from its own rows is not evidence.
    man = json.loads(PROPS.read_text())
    rows = man["rows"]
    got["proposals"] = len(rows)
    got["present"] = sum(1 for r in rows if r.get("present") == 1)
    got["absent"] = sum(1 for r in rows if r.get("present") == 0)
    got["unknown"] = sum(1 for r in rows if r.get("present") is None)
    got["vertex_present"] = sum(
        1 for r in rows if r.get("present_vertex_geometry") == 1)
    got["vertex_absent"] = sum(
        1 for r in rows if r.get("present_vertex_geometry") == 0)
    got["vertex_unknown"] = sum(
        1 for r in rows if r.get("present_vertex_geometry") is None)
    got["label_changes"] = sum(1 for r in rows if r.get("label_changed"))
    got["label_change_ledger"] = len(man.get("label_change_ledger") or [])
    head_mismatch = []
    for hk, hv in (("n_proposals", got["proposals"]),
                   ("n_present", got["present"]),
                   ("n_absent", got["absent"]),
                   ("n_uncertain", got["unknown"]),
                   ("label_changes", got["label_changes"])):
        if man.get(hk) != hv:
            head_mismatch.append({"headline": hk,
                                  "manifest": man.get(hk),
                                  "from_rows": hv})
    for vk, hv in (("present", got["vertex_present"]),
                   ("absent", got["vertex_absent"]),
                   ("unknown", got["vertex_unknown"])):
        if (man.get("census_vertex_geometry") or {}).get(vk) != hv:
            head_mismatch.append({
                "headline": f"census_vertex_geometry.{vk}",
                "manifest": (man.get("census_vertex_geometry")
                             or {}).get(vk), "from_rows": hv})
    if got["label_change_ledger"] != got["label_changes"]:
        head_mismatch.append({
            "headline": "label_change_ledger length",
            "manifest": got["label_change_ledger"],
            "from_rows": got["label_changes"]})
    got["headline_mismatches"] = head_mismatch
    gate = json.loads(GATE.read_text())
    got["gate_cases"] = gate["owned_absence_case_pre"]["n_cases"]
    got["gate_inventory"] = gate["owned_absence_case_pre"]["inventory"]

    fails = []
    for k, want in EXPECT.items():
        if got.get(k) != want:
            fails.append({"field": k, "expected": want,
                          "got": got.get(k)})
    if got["gate_inventory"] != {"no_tube_visible": 35,
                                 "not_directly_visible": 6,
                                 "owner_uncertain": 1}:
        fails.append({"field": "gate_inventory",
                      "expected": {"no_tube_visible": 35,
                                   "not_directly_visible": 6,
                                   "owner_uncertain": 1},
                      "got": got["gate_inventory"]})
    if got.get("certified_unique_grains") != got.get("certified"):
        fails.append({"field": "certified_unique_grains",
                      "expected": got.get("certified"),
                      "got": got.get("certified_unique_grains")})
    if got["headline_mismatches"]:
        fails.append({"field": "manifest_headlines_vs_rows",
                      "expected": "every headline regenerable from rows",
                      "got": got["headline_mismatches"]})
    res = {"recomputed": got, "expected": EXPECT,
           "fails": fails, "ok": not fails,
           "note": "counts recomputed from the target builder, the "
                   "proposal ROWS + change ledger and the gate "
                   "artifact; manifest headlines are cross-checked "
                   "against the recomputation — never used as sources"}
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    print(f"count agreement: {'OK' if not fails else 'FAIL'}"
          f" -> {OUT}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
