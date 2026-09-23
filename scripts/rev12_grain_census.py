"""Build a scoped, canonical grain report from preserved human evidence."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tubetracker.analysis_contracts import file_hash
from tubetracker.population import banked_grains, build_population_report, census_tiles, snapshot_data


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry", type=Path, default=REPO / "runs/prototypes/v30/rev14_field_population/grain_ids.json")
    p.add_argument("--snapshot", type=Path, default=REPO / "runs/prototypes/v30/snap27_rev14")
    p.add_argument("--owners", type=Path, default=REPO / "runs/prototypes/v30/rev11own_verify/owners.json")
    p.add_argument("--emergence-db", type=Path, default=Path(
        "/Users/joshjiang/Documents/TubeTracker-annotator-projects/rev12emergence/annotations.db"))
    p.add_argument("--movie", default="ld")
    p.add_argument("--roi", type=float, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    p.add_argument("--frames", type=int, nargs="+")
    p.add_argument("--classes", nargs="+", default=["grains"])
    p.add_argument("--out-dir", type=Path, default=REPO / "runs/prototypes/v30/rev14_field_population/report")
    args = p.parse_args(argv)
    if bool(args.roi) != bool(args.frames):
        p.error("--roi and --frames are required together for an exact census scope")
    registry = json.loads(args.registry.read_text())
    data = snapshot_data(args.snapshot)
    owners = json.loads(args.owners.read_text())["owners"]
    rulings = list(data.get("rulings", []))
    if not rulings and args.emergence_db.exists():
        con = sqlite3.connect(args.emergence_db.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            for uid, value, revision in con.execute(
                    "select uuid,data,revision from entities where kind='ruling'"):
                rulings.append({"uuid": uid, "data": json.loads(value), "revision": revision,
                                "source": str(args.emergence_db.resolve()) + "#" + uid})
        finally:
            con.close()
    grains, audit = banked_grains(registry, data, owners, rulings)
    scope = {"movie": args.movie, "roi_xywh": args.roi, "frames": args.frames,
             "class_scopes": args.classes} if args.roi else None
    report = build_population_report(grains, census_tiles(data["census"]), movie=args.movie, scope=scope)
    report["audit"] = audit
    report["provenance"] = {"registry": str(args.registry.resolve()),
        "registry_sha256": file_hash(args.registry), "snapshot": str(args.snapshot.resolve()),
        "snapshot_manifest_sha256": file_hash(args.snapshot / "snapshot_manifest.json"),
        "rulings": [{"id": r["uuid"], "revision": r["revision"], "source": r["source"]} for r in rulings]}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "emergence_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(path), "grains": report["census"]["n_grains"],
        "denominator": report["germination"]["denominator_grains"],
        "classified_germinated": report["germination"]["numerator_germinated"],
        "complete": report["census"]["census_completeness_certified"],
        "decoded_masks": len(audit["mask_decoding"]),
        "unresolved_evidence": len(audit["unresolved_evidence"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
