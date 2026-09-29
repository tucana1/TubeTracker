"""Markdown tables from the protocol results (runs/kymo_reader/results/protocol_*.json).

    python -m prototypes.kymo_reader.tables A B C
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "runs/kymo_reader/results"


def ci(v: dict, k: str) -> str:
    x = v[k]
    return f"{x['delta']:+d} [{x['ci'][0]:+.0f}, {x['ci'][1]:+.0f}]"


def main(argv=None):
    tags = argv or sys.argv[1:]
    lines = ["| model | test | reader / path | onsets | lengths | length + tip | vs st053: onsets | lengths | length + tip |",
             "|---|---|---|---|---|---|---|---|---|"]
    oracle = ["| model | test | lengths: kymo vs SparseTrack, both on the human route | onsets | length + tip |",
              "|---|---|---|---|---|"]
    for tag in tags:
        p = RES / f"protocol_{tag}.json"
        if not p.exists():
            continue
        res = json.loads(p.read_text())
        ch = res["choice"]
        for mv in ("ld", "m2"):
            if mv not in res:
                continue
            out = res[mv]
            b = out["st053"]
            lines.append(f"| {tag} | {mv} | SparseTrack 0.5.3 (st053) | {b['onset']} | {b['lengths']} | {b['length_and_tip']} | | | |")
            rows = [(f"KymoReader on st053's path", out[f"{tag}_st"])]
            if f"{mv}_stonset" in res:
                rows.append(("KymoReader lengths, st053's calls", res[f"{mv}_stonset"][f"{tag}_stonset_st"]))
            rows += [("KymoReader on the human route (oracle)", out[f"{tag}_route"]),
                     ("SparseTrack on the human route (oracle)", out["st053_route"])]
            for name, r in rows:
                v = r["vs_st053"]
                lines.append(f"| {tag} | {mv} | {name} | {r['onset']} | {r['lengths']} | {r['length_and_tip']} | "
                             f"{ci(v, 'onset_hit')} | {ci(v, 'len_hit')} | {ci(v, 'both_hit')} |")
            v = out[f"{tag}_route"]["vs_st053_route"]
            oracle.append(f"| {tag} | {mv} | {ci(v, 'len_hit')} | {ci(v, 'onset_hit')} | {ci(v, 'both_hit')} |")
        lines.append(f"| {tag} | | choices: {json.dumps({k: ch[k] for k in ch if k in ('start', 'onset_source', 'onset_px', 'min_len')})} | | | | | | |")
    print("\n".join(lines))
    print()
    print("\n".join(oracle))


if __name__ == "__main__":
    main()
