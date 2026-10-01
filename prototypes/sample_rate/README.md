# Higher sample rates (1 Oct 2026, SparseTrack 0.8.5)

The sparse movie analysed with every frame, or with keyframes in 150- and 100-frame bins, against keyframes in
300-frame bins (`compare.md`; `docs/status-2026-09-29.md`). `run_rate.py NAME SAMPLE FPB` builds the cache, its tube
maps and the analysis of the labelled grains; `run_scaled.py` the 150-frame bins with the bin-counted settings
doubled; `ref_offset.py` the offset of a cache's reference frame from the original's. Nothing beat keyframes in
300-frame bins: the frames between keyframes in these x264 movies are near-copies of the keyframe before them.
