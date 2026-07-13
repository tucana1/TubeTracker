# TubeTracker

TubeTracker is a wxPython desktop application and reusable analysis engine for
measuring pollen germination, tube-tip movement, growth, and reviewed rupture
candidates from microscopy videos. It is currently developed and tested on
macOS.

## Local setup

From the repository root:

```bash
./Start_TubeTracker_local
```

The launcher creates a local `.venv` and installs the project and its pinned
dependencies from `pyproject.toml` on first use. It also reconciles dependency
changes on later launches. The original conda-based installer remains available
for compatibility.

## Code architecture

The application is split by responsibility so analysis code can be tested and
used without constructing the desktop interface:

- `tubetracker/models.py` contains the `Point`, `ROI`, and `Track` domain objects.
- `tubetracker/analysis.py` contains frame preprocessing, detection, LapTrack
  association, germination and rupture scoring, and result export.
- `tubetracker/views.py` contains the reusable wxPython image canvas controls.
- `tubetracker/gui.py` contains the desktop workflow and wxPython event handlers.
- `tubetracker_resources.py` validates and loads packaged image resources.
- `scripts/` contains headless video inspection, pilot analysis, and regression
  workflows.
- `TubeTracker.py` is intentionally only a compatibility facade for older code
  that imports `TubeTracker`; new analysis code should import from `tubetracker`.

Keep new biological state and trajectory behavior in `models.py`, image and
tracking algorithms in `analysis.py`, and user interaction in `gui.py` or
`views.py`. Analysis modules must not import wxPython. Runtime resources belong
under `tubetracker_assets/`, not as arrays or paths embedded in Python source.
Architecture tests enforce these boundaries and prevent opaque numbered helper
methods from being reintroduced.

## Pilot analysis workflow

Tools under `scripts/` make an incoming laboratory video inspectable and
reproducible before GUI tuning:

```bash
.venv/bin/python scripts/inspect_video.py data/raw/example.avi --output-dir runs/intake/example

.venv/bin/python scripts/run_pilot.py data/raw/example.avi \
  --sample-id example \
  --genotype WT \
  --biological-replicate plant-1 \
  --time-per-frame 30 \
  --pixel-size 0.8 \
  --distance-unit um
```

The pilot runner writes the exact parameters, source metadata, grain-level
germination results, tip trajectories, growth rates, annotated videos, and QC
flags into a timestamped directory under `runs/`.

TubeTracker uses LapTrack for deterministic microscopy-oriented grain and tip
association. OpenCV performs video decoding, segmentation, morphology, grain
detection, and tip detection. Pilot manifests record both package versions;
dependency upgrades must be treated as analysis changes and validated against
the reference annotations before combining results across versions.

## Versioned analysis resources

TubeTracker does not embed learned weights or opaque image data in its Python
source. The optional template-matching detector loads its 23 legacy grayscale
reference images from `tubetracker_assets/tip_templates/`. A versioned manifest
fixes their order and verifies each image's dimensions and decoded-pixel checksum
before an analysis starts. These templates are algorithm inputs, so replacements
should be reviewed and regression-tested like code changes.

Run the bundled regression check with:

```bash
.venv/bin/python scripts/smoke_test.py
```

For development, install the test and build tools declared in `pyproject.toml`:

```bash
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
```

See `docs/pilot-guide.md` for the incoming data requirements and complete
analysis/QC sequence.

License:

Copyright 2024, Brown University, Providence, RI.

                        All Rights Reserved

Permission to use, copy, modify, and distribute this software and
its documentation for any purpose other than its incorporation into a
commercial product or service is hereby granted without fee, provided
that the above copyright notice appear in all copies and that both
that copyright notice and this permission notice appear in supporting
documentation, and that the name of Brown University not be used in
advertising or publicity pertaining to distribution of the software
without specific, written prior permission.

BROWN UNIVERSITY DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY
PARTICULAR PURPOSE.  IN NO EVENT SHALL BROWN UNIVERSITY BE LIABLE FOR
ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
