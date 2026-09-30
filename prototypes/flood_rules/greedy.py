"""Stage 2 of the tuning, on ONE movie: the stage-1 settings that gained there, combined greedily.

    python -m prototypes.flood_rules.greedy QUEUE PREFIX BASE_NAME [--reader flood] NAME=key=value[,key=value] ...

Candidates are given best first (by their stage-1 gain on this movie). Starting from the default rules (the dump
BASE_NAME in QUEUE/done), each candidate is added to the rules kept so far and run (QUEUE/todo, served by
``bench serve``); it is kept if the movie's FULL-length hits rise by at least one (ties: length and tip). The rule set
kept at the end is printed and written to QUEUE/done/PREFIX_final.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def wait(queue: Path, name: str) -> dict:
    while True:
        f, e = queue / "done" / f"{name}.json", queue / "done" / f"{name}.err"
        if f.exists():
            return json.loads(f.read_text())
        if e.exists():
            raise RuntimeError(e.read_text())
        time.sleep(5)


def main(argv=None):
    from .bench import submit
    args = list(argv if argv is not None else sys.argv[1:])
    queue, prefix, base = Path(args.pop(0)), args.pop(0), args.pop(0)
    extra = {}
    if args and args[0] == "--reader":
        args.pop(0)
        extra["reader"] = args.pop(0)
    cands = []
    for a in args:
        name, kvs = a.split("=", 1)
        cands.append((name, dict(kv.split("=", 1) for kv in kvs.split(","))))
    cur = wait(queue, base)
    movie = next(iter(cur))
    kept: dict = {}
    log = []
    for k, (name, sets) in enumerate(cands, 1):
        trial = {**kept, **sets}
        run = f"{prefix}_{k:02d}_{name}"
        submit(queue, run, [f"{a}={b}" for a, b in {**extra, **trial}.items()])
        res = wait(queue, run)[movie]
        c = cur[movie]
        better = (res["len_hit"], res["both"]) > (c["len_hit"], c["both"]) and res["len_hit"] >= c["len_hit"]
        log.append({"run": run, "set": trial, "len": res["len_hit"], "both": res["both"], "on": res["on_hit"],
                    "kept": bool(better)})
        print(f"{run}: len {res['len_hit']} both {res['both']} on {res['on_hit']} "
              f"(so far {c['len_hit']}/{c['both']}) -> {'KEEP' if better else 'drop'}", flush=True)
        if better:
            kept, cur = trial, {movie: res}
    out = {"movie": movie, "prefix": prefix, "base": base, "reader": extra.get("reader"), "kept": kept, "log": log}
    (queue / "done" / f"{prefix}_final.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["kept"]))


if __name__ == "__main__":
    main()
