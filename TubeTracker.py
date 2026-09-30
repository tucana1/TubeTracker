#!/usr/bin/env python3
"""Open TubeTracker (``python TubeTracker.py [MOVIE_OR_ANALYSIS_FOLDER]``; ``--legacy`` for the old manual pipeline)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tubetracker.app.window import main  # noqa: E402


if __name__ == "__main__":
    main()
