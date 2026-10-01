"""Contact sheet of the 15 poorly covered m2 traces: the registered field at the traced bin, the annotator's trace
(green), the model's tube as the app draws it (magenta, dot at its end), the rest of the model's route (thin dashed
pink), labelled gid@bin, the primary category and traced / model lengths."""
import cv2
import numpy as np

from common import BAD, GREEN, MAGENTA, OUT, PINK_DIM, Tile, drawn, length_at, trace

CAT = {
    ("g005", 349): "(b) stops at a crossing",
    ("g009", 140): "(c) old route, exit off (+e)",
    ("g016", 140): "(c) 3 px off, bend capped",
    ("g038", 68): "(c) on dark wall, 3 px",
    ("g038", 244): "(a) took a passing tube",
    ("g038", 349): "(a) took a passing tube",
    ("g048", 349): "(e) trace doubtful (+d,f)",
    ("g052", 140): "(c) end route 7 px off",
    ("g052", 244): "(a) turns onto crossing tube",
    ("g052", 349): "(a) other tubes after 20 px",
    ("g054", 349): "(b) froze at bin 250",
    ("g064", 349): "(c) tube reshaped (+b)",
    ("g069", 244): "(c) tube swung; U-turn",
    ("g092", 349): "(a) took tube from left (+c)",
    ("g106", 140): "(b) stub under-read",
}
SIZE = 400
tiles = []
for gid, b, f in BAD:
    g = __import__("common").G[gid]
    tr, t = trace(gid, b)
    d = drawn(g, b)
    full = drawn(g, b, cut=False)
    pts = np.vstack([tr] + ([d] if d is not None else []))
    lo, hi = pts.min(0), pts.max(0)
    cx, cy = (lo + hi) / 2
    half = int(max((hi - lo).max() / 2 + 18, 26))
    T = Tile(b, cx, cy, half, SIZE)
    T.dashed(full, PINK_DIM, 1)
    T.line(tr, GREEN, 2)
    if d is not None:
        T.line(d, MAGENTA, 2)
        T.dot(d[-1], MAGENTA, 3)
    T.dot(tr[-1], GREEN, 3, filled=False)
    cv2.rectangle(T.im, (0, 0), (SIZE, 44), (30, 30, 30), -1)
    cv2.putText(T.im, f"{gid}@{b}  {CAT[(gid, b)]}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)
    cv2.putText(T.im, f"traced {t['length_px']:.0f} px  model {length_at(g, b):.0f} px  cover {f:.2f}  (crop {2 * half} px)",
                (6, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    tiles.append(T.im)
# legend tile
leg = np.full((SIZE, SIZE, 3), 255, np.uint8)
items = [(GREEN, "annotator trace (circle = apex)", 2, False), (MAGENTA, "model tube as drawn (dot = end)", 2, False),
         (PINK_DIM, "rest of model route (uncut)", 1, True)]
for k, (col, txt, th, dash) in enumerate(items):
    y = 60 + 40 * k
    if dash:
        for x0 in range(20, 80, 10):
            cv2.line(leg, (x0, y), (x0 + 6, y), col, th, cv2.LINE_AA)
    else:
        cv2.line(leg, (20, y), (80, y), col, th + 1, cv2.LINE_AA)
    cv2.putText(leg, txt, (92, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
lines = ["m2, SparseTrack 0.8.4 predictions", "traces with < 50% of points within 2 px",
         "(a) another tube  (b) stopped early", "(c) route elsewhere at that bin",
         "(d) crossing/clump  (e) trace doubtful", "(f) other"]
for k, s in enumerate(lines):
    cv2.putText(leg, s, (20, 210 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
tiles.append(leg)
tiles = [np.pad(im, ((3, 3), (3, 3), (0, 0)), constant_values=255) for im in tiles]
rows = [np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)]
sheet = np.vstack(rows)
cv2.imwrite(OUT + "sheet.png", sheet)
print("wrote", OUT + "sheet.png", sheet.shape)
