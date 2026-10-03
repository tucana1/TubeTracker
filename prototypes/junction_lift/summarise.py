import json
import sys

import numpy as np

SP = "/private/tmp/claude-501/-Users-joshjiang-Documents-TubeTracker/eaa2b708-9f78-4565-aa0c-9af3763da8fd/scratchpad/"
res = json.load(open(SP + "jt/route_test.json"))
TAKE = {"takeover", "takeover_start", "overshoot_onto_tube"}
EXTRA = {"m2 g038@244", "m2 g038@349", "m2 g005@349", "m2 g011@349"}
groups = {
    "takeover traces": [a for a in res if a["cls"] in TAKE or a["case"] in EXTRA],
    "near another tube (all)": [a for a in res if a["kind"] == "near" or a["case"] in EXTRA or a["cls"] in TAKE],
    "near, length hit (controls)": [a for a in res if a["len_hit"]],
}
variants = [k for k in res[0] if isinstance(res[0][k], dict) and "cov" in res[0][k]]
for gname, rows in groups.items():
    print(f"\n{gname}: {len(rows)} traces")
    print(f"{'variant':12s} {'cov<=2px':>9s} {'cov<=4px':>9s} {'junction ok':>12s} {'global ok':>10s}")
    for v in variants:
        cov = np.mean([a[v]["cov"] for a in rows])
        cov4 = np.mean([a[v]["cov4"] for a in rows])
        dec = [a[v]["dec_ok"] for a in rows if a[v]["dec_ok"] is not None]
        glob = [a[v]["global_ok"] for a in rows]
        print(f"{v:12s} {cov:9.2f} {cov4:9.2f} {sum(dec):>5d}/{len(dec):<6d} {sum(glob):>4d}/{len(glob):<5d}")
print()
for a in res:
    line = f"{a['case']:13s} {a['cls']:20s} {'C' if a['contact'] else ' '} {'end@J' if a['ends_at_junction'] else '     '}"
    for v in variants:
        d = a[v]["dec_ok"]
        line += f" | {v}: {a[v]['cov']:.2f}/{a[v]['cov4']:.2f} {'-' if d is None else ('ok' if d else 'X')}"
    print(line)
