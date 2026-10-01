"""Offset (dx, dy) of a cache's reference frame from the original sparse-movie cache's (phase correlation of the
two registered reference images, high-passed): add it to the labels' coordinates to put them in the new frame."""
import sys
import cv2
import numpy as np
sys.path.insert(0, "/Users/joshjiang/Documents/TubeTracker")
from sparsetrack import stack
from tubetracker.app.imaging import FrameSource


def ref_image(cache):
    F = FrameSource(cache, fields=4)
    rs = F.ref_start
    img = np.mean([F.registered(b) for b in range(rs, rs + 3)], axis=0)
    return img - cv2.GaussianBlur(img, (0, 0), 8)


a = ref_image("/Users/joshjiang/Documents/TubeTracker/runs/sparsetrack/ld")
for c in sys.argv[1:]:
    b = ref_image(c)
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32), win)
    meta = stack.load(c)[1]
    print(c.split("/")[-2], "ref bins", meta["ref_start"], "- frames", meta["ref_start"] * meta["frames_per_bin"],
          f"offset ({dx:+.2f}, {dy:+.2f}) px, response {resp:.2f}")
