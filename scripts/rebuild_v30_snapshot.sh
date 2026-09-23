#!/usr/bin/env bash
# rev8: rebuild the v30 snapshot from the SAME surviving sources as
# snap16 plus one newly-annotated project. Sources are taken from
# snap16's own manifest, so the list can never drift by hand.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
PREV=${PREV:-runs/prototypes/v30/snapshots/snap16}
OUT=${OUT:-runs/prototypes/v30/snapshots/snap17}
# extras are folded into the project list above (deduped)
# projects = predecessor's list UNION the extras, each exactly once
# (passing a project the predecessor already lists is the easy mistake)
projects=()
while IFS= read -r p; do
  [ -n "$p" ] && projects+=("$p")
done < <("$PY" -c "
import json
m=json.loads(open('$PREV/snapshot_manifest.json').read())
extra=[q for q in '$*'.split('|') if q]
seen=set(); out=[]
for p in list(m['projects'])+extra:
    r=p.rstrip('/')
    if r not in seen:
        seen.add(r); out.append(p)
for p in out: print(p)
")
args=()
for p in "${projects[@]}"; do args+=("--project-dir" "$p"); done
"$PY" scripts/build_v30_snapshot.py "${args[@]}" \
  --movie "ld=/Users/joshjiang/Downloads/test1lowdensjoshua-28c-hz.mp4 .mp4" \
  --movie "m1=/Users/joshjiang/Downloads/Pollen tube movie 2 7-14-26.mp4" \
  --movie "m2=/Users/joshjiang/Downloads/Pollen tube movie 2 7-14-26.mp4" \
  --default-movie ld \
  --out "$OUT"
echo "built $OUT"
