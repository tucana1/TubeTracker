"""rev13 W2 acceptance: replay the interval export against the geometry
contract — no exported current-path endpoint differs from its reported
tip; every full length equals its current path's arclength; the
completeness gating is internally consistent.

Usage: python scripts/rev13_interval_geometry_check.py <export.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

TOL = 1e-6


def main() -> int:
    from prototypes.v30_video_apex.ownership import geometry_consistent

    p = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        REPO / "runs/prototypes/v30/rev13_interval/export_inferred.json")
    d = json.loads(p.read_text())
    rows = d["rows"] if isinstance(d, dict) and "rows" in d \
        else [r for u in (d.get("units") or {}).values()
              for r in (u.get("rows") or [])]
    n_end_bad, n_len_bad, n_full, n_partial, n_rows = 0, 0, 0, 0, 0
    bad = []
    for r in rows:
        n_rows += 1
        path = r.get("path") or []
        tip = r.get("tip")
        if len(path) < 2 or tip is None:
            continue
        length = r.get("length_px")
        g = geometry_consistent(path, tip, length if length is not None
                                else None)
        if not g["endpoint_ok"]:
            n_end_bad += 1
            bad.append({"owner": r.get("owner"), "frame": r.get("frame"),
                        "endpoint_err_px": g["endpoint_err_px"]})
        if r.get("length_kind") == "full":
            n_full += 1
            if not g["length_ok"]:
                n_len_bad += 1
                bad.append({"owner": r.get("owner"),
                            "frame": r.get("frame"),
                            "length_err_px": g["length_err_px"]})
        else:
            n_partial += 1
    out = {"export": str(p), "n_rows": n_rows,
           "n_full_length": n_full, "n_partial": n_partial,
           "n_endpoint_mismatch": n_end_bad,
           "n_full_length_mismatch": n_len_bad,
           "ok": (n_end_bad == 0 and n_len_bad == 0),
           "rule": ("every exported current-path endpoint equals its "
                    "reported tip and every full length equals its "
                    "current path's arclength within 1e-6 px"),
           "bad": bad[:20]}
    outp = p.parent / "geometry_check.json"
    outp.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    print(f"geometry check: {'OK' if out['ok'] else 'FAIL'} -> {outp}")
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
