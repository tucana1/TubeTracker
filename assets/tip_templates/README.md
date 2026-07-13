# Tip-matching templates

These 23 lossless grayscale PNGs are the legacy, hand-authored image templates
used by TubeTracker's optional template-matching tip detector. They are static
reference images, not learned model weights.

The images were extracted losslessly from the original integer arrays. The
manifest records their order, dimensions, and decoded-pixel SHA-256 checksums.
TubeTracker validates all three before analysis so missing, reordered, or
silently changed assets fail with a clear error.

Changes to these images should be reviewed as algorithm changes and accompanied
by an intentional manifest update and regression tests against representative
microscopy videos.
