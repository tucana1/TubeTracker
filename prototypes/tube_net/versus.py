"""Paired comparison of two bench dumps (``synth_bench.paired``: onsets, lengths, length and tip; 95% bootstrap
intervals over grains), e.g. a new leave-one-out recipe against the unet_d recipe on the movie neither saw.

    python -m prototypes.tube_net.versus BASE.json NEW.json [MOVIE ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

if __name__ == "__main__":
    import synth_bench as sb
    base, new = (json.loads(Path(p).read_text()) for p in sys.argv[1:3])
    movies = sys.argv[3:] or sorted(set(base) & set(new))
    print(sb.paired({m: base[m] for m in movies}, {m: new[m] for m in movies}))
