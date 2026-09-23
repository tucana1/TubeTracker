"""P22-clean autopsy: same crop through frozen v1 vs E2 (v4events) eyes."""
import cv2
import numpy as np
import torch

from tubetracker.annotation_frames import FrameReader
from tubetracker.cnn_prototype import (
    extract_heatmap_points,
    load_cnn_checkpoint,
    predict_heatmaps_tiled,
)

V1 = "runs/prototypes/timesfm/tip_cnn_v1/best-point-heatmap-model.pt"
V4 = "runs/prototypes/timesfm/tip_cnn_v4events/best-point-heatmap-model.pt"
MOVIE = "/Users/joshjiang/Downloads/test1lowdensjoshua-28c-hz.mp4 .mp4"
FRAME, TX, TY = 42000, 1004.5786, 308.586  # P22-clean probe
HALF = 150

device = torch.device("cpu")
models = {}
for name, ckpt in (("v1", V1), ("v4", V4)):
    m, _ = load_cnn_checkpoint(ckpt, device=device)
    m.eval()
    models[name] = m

rdr = FrameReader(MOVIE)
frame = rdr.read(FRAME).frame
rdr.close()
if frame.ndim == 3:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
else:
    gray = frame
H, W = gray.shape
x0, x1 = max(0, int(TX) - HALF), min(W, int(TX) + HALF)
y0, y1 = max(0, int(TY) - HALF), min(H, int(TY) + HALF)

panels = []
with torch.no_grad():
    hms = {n: np.asarray(predict_heatmaps_tiled(m, frame, device)[1])
           for n, m in models.items()}
for name in ("v1", "v4"):
    hm = hms[name]
    crop = gray[y0:y1, x0:x1]
    rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    hcrop = hm[y0:y1, x0:x1]
    hot = cv2.applyColorMap((np.clip(hcrop, 0, 1) * 255).astype(np.uint8),
                            cv2.COLORMAP_JET)
    blend = cv2.addWeighted(rgb, 0.45, hot, 0.55, 0)
    cv2.drawMarker(blend, (int(TX) - x0, int(TY) - y0), (0, 255, 0),
                   cv2.MARKER_CROSS, 24, 2)  # truth = green cross
    for i, (px, py, conf) in enumerate(
            extract_heatmap_points(hm, threshold=0.35)[:3]):
        if x0 <= px < x1 and y0 <= py < y1:
            cv2.circle(blend, (int(px) - x0, int(py) - y0), 8,
                       (255, 0, 255) if i == 0 else (255, 255, 0), 2)
            cv2.putText(blend, f"{conf:.2f}", (int(px) - x0 + 10,
                        int(py) - y0), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)
    cv2.putText(blend, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    panels.append(blend)

raw = cv2.cvtColor(gray[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
cv2.drawMarker(raw, (int(TX) - x0, int(TY) - y0), (0, 255, 0),
               cv2.MARKER_CROSS, 24, 2)
cv2.putText(raw, "raw + truth", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2)
sheet = np.hstack([raw] + panels)
cv2.imwrite("/tmp/p22_autopsy.png", sheet)
print("wrote /tmp/p22_autopsy.png", sheet.shape)
