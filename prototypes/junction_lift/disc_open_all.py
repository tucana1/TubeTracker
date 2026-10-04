"""Diagnostic only: would opening other grains' discs at all (every pixel the map ever marks inside them, no band
test) let the flood follow the m2 disc-crossing tubes? Monkeypatches learned.disc_passages."""
import functools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reflood  # noqa: E402
from sparsetrack import learned  # noqa: E402

learned.disc_passages = functools.partial(learned.disc_passages, min_apart_deg=0.0, max_edge_deg=360.0)
sys.argv = [sys.argv[0], "m2", "g005", "g052", "g038", "g092", "g048", "g064", "flood_disc_pass=true",
            "flood_disc_width=1000.0"]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "opt_check.py")).read())
