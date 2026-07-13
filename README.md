# TubeTracker
Please refer to the provided manual. 

Code has been tested and built on Mac

Enjoy

## Local setup

From the repository root:

```bash
./Start_TubeTracker_local
```

The launcher creates a local `.venv` and installs the pinned dependencies in
`requirements.txt` on first use. The original conda-based installer remains
available for compatibility.

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

## Versioned analysis resources

TubeTracker does not embed learned weights or opaque image data in its Python
source. The optional template-matching detector loads its 23 legacy grayscale
reference images from `assets/tip_templates/`. A versioned manifest fixes their
order and verifies each image's dimensions and decoded-pixel checksum before an
analysis starts. These templates are algorithm inputs, so replacements should
be reviewed and regression-tested like code changes.

Run the bundled regression check with:

```bash
.venv/bin/python scripts/smoke_test.py
```

See `docs/pilot-data-request.md` for the information needed with new videos and
`docs/pilot-analysis-protocol.md` for the analysis/QC sequence.

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
