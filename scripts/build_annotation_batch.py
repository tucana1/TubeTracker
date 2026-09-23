"""Round-2 annotation batch: value-maximizing task queue (H239).

Scores every candidate (frame, grain) by model UNCERTAINTY so human
clicks land where they teach the most, and skips cases the machine
already owns:
  - certain: strong single tip-heat peak near an isolated grain -> SKIP
    (a few kept as calibration controls)
  - faint: no tip-heat peak anywhere near the grain -> LABEL (the open
    faint-apex class, H208 moratorium evidence)
  - ambiguous: 2+ strong tip peaks near one grain (crossings, neighbor
    tubes) -> LABEL (identity disambiguation)
  - crowded: 2+ grains within 60px -> LABEL (crossing-prone)

Also spreads frames across the movie, caps tasks per frame region,
and dedupes against already-completed tasks in the project db.

Runs in the main .venv (torch + cv2); writes task rows into the
project's annotations.db, which the annotator app then serves.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "prototypes" / "timesfm_tip_forecast"))

from tubetracker.annotation_store import AnnotationStore  # noqa: E402
from tubetracker.annotation_frames import FrameReader  # noqa: E402
from grain_detect import detect_grains  # noqa: E402

TIP_CHANNEL = 1  # CNN_LABELS = ("grain", "tip")
GRAIN_R = 16     # nominal grain radius px (lowdens scale)
RIM_EXCL = 26    # tip-heat fires on bright grain rims (H239 eye-check):
                 # peaks inside this are the ball itself, not its tube
SEARCH_R = 130   # tip search radius around grain center
NEAR_R = 70      # peaks within this of grain center = this ball's tube
CROWD_R = 60     # grains within this = crowded
EDGE_M = 40      # skip grains closer than this to the frame edge
DUP_R = 30       # within this of a completed task = duplicate


def load_tip_model(weights: Path):
    import torch

    from tubetracker.cnn_prototype import PointHeatmapNet
    device = torch.device("cpu")
    payload = torch.load(weights, map_location=device, weights_only=False)
    state = payload["state_dict"] if isinstance(payload, dict) \
        and "state_dict" in payload else payload
    model = PointHeatmapNet(input_channels=2, output_channels=2,
                            base_channels=16)
    model.load_state_dict(state)
    model.eval()
    return model, device


def tip_heat_for_frame(model, device, gray: np.ndarray) -> np.ndarray:
    import torch

    from tubetracker.cnn_prototype import predict_heatmaps_tiled
    with torch.no_grad():
        out = predict_heatmaps_tiled(model, gray, device)
    # NOTE (P0 audit fix): predict_heatmaps_tiled already returns
    # sigmoid probabilities. A second sigmoid here was monotone but
    # moved every threshold (0.6 after two sigmoids ~= 0.405 true).
    return np.asarray(out[TIP_CHANNEL]).astype(np.float32)


def peak_count(patch: np.ndarray, thresh: float,
               footprint: int = 9) -> int:
    """Count significant local-maxima peaks above thresh."""
    return len(significant_peaks(patch, thresh, footprint))


def significant_peaks(patch: np.ndarray, thresh: float,
                      footprint: int = 9) -> list[tuple[float, float]]:
    """Centroids (x, y in patch coords) of significant peak components.

    The tip heat is spiky (single-pixel maxima, H239 probe) — blob-area
    filtering deletes every real peak. Instead, maxima are tested
    against a wide dilation footprint so only peaks separated by
    >=footprint//2 px survive; each is one pixel.
    """
    if patch.max() < thresh:
        return []
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (footprint, footprint))
    dil = cv2.dilate(patch, k)
    peaks = (patch >= dil) & (patch >= thresh)
    n, _, _, centroids = cv2.connectedComponentsWithStats(
        peaks.astype(np.uint8), connectivity=8)
    return [(float(centroids[i][0]), float(centroids[i][1]))
            for i in range(1, n)]


def completed_foci(db_path: Path) -> list[tuple[int, float, float]]:
    """(frame, x, y) of already-completed tasks + their marks."""
    if not db_path.exists():
        return []
    con = sqlite3.connect(str(db_path))
    try:
        rows = con.execute(
            "SELECT data FROM entities WHERE kind='task'").fetchall()
    except sqlite3.OperationalError:
        return []  # fresh project, no tables yet
    finally:
        con.close()
    out = []
    for (data,) in rows:
        d = json.loads(data)
        if not d.get("completed"):
            continue
        qf = d.get("query_frames", [0])
        for m in d.get("marks", []) or []:
            out.append((int(qf[0]), float(m[0]), float(m[1])))
        if d.get("focus_xy"):
            out.append((int(qf[0]), float(d["focus_xy"][0]),
                        float(d["focus_xy"][1])))
    return out


def build_batch(movie: str, project_dir: Path, weights: Path,
                n_frames: int = 40, per_frame: int = 10,
                batch_size: int = 80, calib_frac: float = 0.1,
                tag_prefix: str = "r2", movie_key: str = "",
                mode: str = "uncertainty",
                keep_classes: tuple[str, ...] = (),
                avoid: tuple[tuple[int, float, float], ...] = ()) -> list[dict]:
    reader = FrameReader(movie)
    try:
        n = len(reader)
        fids = [min(int(i * n / n_frames), n - 1) for i in range(n_frames)]
        model, device = (None, None)
        if mode == "uncertainty":
            model, device = load_tip_model(weights)
        done = completed_foci(project_dir / "annotations.db")
        cands: list[dict] = []
        for fid in fids:
            res = reader.read(fid)
            gray = (res.frame if res.frame.ndim == 2 else
                    cv2.cvtColor(res.frame, cv2.COLOR_BGR2GRAY))
            h, w = gray.shape
            grains = [g for g in detect_grains(gray)[::2]
                      if EDGE_M < g[0] < w - EDGE_M
                      and EDGE_M < g[1] < h - EDGE_M][:per_frame * 2]
            if not grains:
                continue
            if mode == "random":
                # P0A random controls: population sampling independent
                # of model uncertainty and the existing census rank.
                import random as _random
                rng = _random.Random(1000 + fid)
                for gx, gy, _s in rng.sample(
                        grains, min(len(grains), max(1, per_frame // 2))):
                    if any(f == fid and abs(x - gx) < DUP_R
                           and abs(y - gy) < DUP_R for f, x, y in done):
                        continue
                    cands.append({
                        "fid": fid, "x": float(gx), "y": float(gy),
                        "class": "random", "value": 0.0, "pmax": -1.0,
                        "npeaks": -1, "crowd": -1,
                    })
                continue
            heat = tip_heat_for_frame(model, device, gray)
            for gx, gy, _s in grains:
                x0, x1 = max(0, int(gx - SEARCH_R)), min(w, int(gx + SEARCH_R))
                y0, y1 = max(0, int(gy - SEARCH_R)), min(h, int(gy + SEARCH_R))
                patch = heat[y0:y1, x0:x1]
                yy, xx = np.mgrid[y0:y1, x0:x1]
                dist = np.hypot(xx - gx, yy - gy)
                ring = patch[(dist > RIM_EXCL) & (dist < SEARCH_R)]
                pmax = float(ring.max()) if ring.size else 0.0
                npeaks = peak_count(np.where(
                    ((dist > GRAIN_R) & (dist < SEARCH_R)), patch, 0.0), 0.6)
                # Near-peak analysis (H239): a peak 100px away belongs to
                # a neighbor's tube, not this grain. Only peaks whose
                # centroid falls within NEAR_R of the grain center speak
                # for THIS ball's tube: 0 near = faint-or-tubeless,
                # 2+ near = competing hypotheses on this ball.
                masked = np.where(
                    ((dist > RIM_EXCL) & (dist < SEARCH_R)), patch, 0.0)
                ox0 = max(0, int(gx - SEARCH_R))
                oy0 = max(0, int(gy - SEARCH_R))
                near = [math.hypot((ox0 + px) - gx, (oy0 + py) - gy)
                        for px, py in significant_peaks(masked, 0.6)]
                near = [d for d in near if d < NEAR_R]
                n_near = len(near)
                crowd = sum(1 for ox, oy, _ in grains
                            if abs(ox - gx) + abs(oy - gy) > 1e-6
                            and abs(ox - gx) < CROWD_R
                            and abs(oy - gy) < CROWD_R)
                if any(f == fid and abs(x - gx) < DUP_R and abs(y - gy) < DUP_R
                       for f, x, y in done):
                    continue
                if n_near >= 2:
                    cls, val = "ambiguous", 3.0 + n_near
                elif crowd >= 1:
                    cls, val = "crowded", 2.0 + crowd
                elif n_near == 0:
                    cls, val = ("faint", 1.0 + min(0.6 - pmax, 0.6)
                                if pmax < 0.6 else 1.0)
                else:
                    cls, val = "certain", 0.0
                if keep_classes and cls not in keep_classes:
                    continue  # targeted batch: only the wanted classes
                if any(abs(f - fid) < 6000 and math.hypot(x - gx, y - gy) < 25
                       for f, x, y in avoid):
                    continue  # already labeled in a previous dataset
                cands.append({
                    "fid": fid, "x": float(gx), "y": float(gy),
                    "class": cls, "value": val, "pmax": pmax,
                    "npeaks": n_near, "crowd": crowd,
                })
        # Quotas per class (H239): global value-ranking starves the
        # faint class (1.0-1.6) behind ambiguous/crowded (2-5) — yet
        # faint is the moratorium evidence we most need. Stratify.
        by_class: dict[str, list[dict]] = {}
        for c in cands:
            by_class.setdefault(c["class"], []).append(c)
        for v in by_class.values():
            v.sort(key=lambda c: -c["value"])
        quotas = [("ambiguous", 0.40), ("faint", 0.30), ("crowded", 0.20)]
        if mode == "random":
            quotas = [("random", 1.0)]
        picked: list[dict] = []
        for cls, frac in quotas:
            picked += by_class.get(cls, [])[:int(batch_size * frac)]
        n_cal = min(len(by_class.get("certain", [])),
                    int(batch_size * calib_frac))
        picked += by_class.get("certain", [])[:n_cal]
        # Fill leftovers with the best remaining, any class.
        if len(picked) < batch_size:
            rest = [c for v in by_class.values() for c in v
                    if c not in picked]
            rest.sort(key=lambda c: -c["value"])
            picked += rest[:batch_size - len(picked)]
        # Spread: cap 3 tasks per frame so one crowded frame can't eat
        # the batch. The field is STATIC (H239 fix): the same ball sits
        # at the same pixels in every frame, so a global one-per-ball
        # rule collapses 24 frames to ~15 tasks. Instead each ball may
        # appear up to 3 times, only if appearances are >=8000 frames
        # apart — tubes grow over time, so spaced repeats teach.
        spread: list[dict] = []
        per_fid: dict[int, int] = {}
        seen: list[dict] = []
        overflow: list[dict] = []
        def _dup(c):
            near = [s for s in seen if abs(c["x"] - s["x"]) < 25
                    and abs(c["y"] - s["y"]) < 25]
            if len(near) >= 4:
                return True
            return any(abs(c["fid"] - s["fid"]) < 6000 for s in near)
        for c in sorted(picked, key=lambda c: -c["value"]):
            if _dup(c):
                continue
            if per_fid.get(c["fid"], 0) < 3:
                spread.append(c)
                seen.append(c)
                per_fid[c["fid"]] = per_fid.get(c["fid"], 0) + 1
            else:
                overflow.append(c)
        for c in overflow:
            if len(spread) >= batch_size:
                break
            if _dup(c):
                continue
            spread.append(c)
            seen.append(c)
        tasks = [{
            "uuid": f"{tag_prefix}-{i:03d}",
            "owner_uuid": "unassigned",
            "movie": movie_key,
            "query_frames": [c["fid"]],
            "focus_xy": [c["x"], c["y"]],
            "task_type": "apex",
            "stratum": f"{tag_prefix}-{c['class']}",
            "priority": round(c["value"], 3),
            "why": (f"{c['class']}: tip-heat max {c['pmax']:.2f}, "
                    f"{c['npeaks']} peaks, {c['crowd']} neighbors"),
            "completed": False,
        } for i, c in enumerate(spread[:batch_size])]
        return tasks
    finally:
        reader.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie", required=True)
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--weights", default=str(
        REPO / "runs/prototypes/timesfm/tip_cnn_v1/best-point-heatmap-model.pt"))
    ap.add_argument("--batch-size", type=int, default=80)
    ap.add_argument("--actor", default="builder")
    ap.add_argument("--tag-prefix", default="r2")
    ap.add_argument("--movie-key", default="")
    ap.add_argument("--mode", default="uncertainty",
                    choices=["uncertainty", "random"])
    ap.add_argument("--keep-classes", default="",
                    help="comma list: only emit these classes "
                         "(e.g. faint for a targeted dim-tail batch)")
    ap.add_argument("--avoid-labels", default="",
                    help="labels.csv whose (frame,x,y) foci must not be "
                         "re-mined (cross-dataset dedupe)")
    ap.add_argument("--avoid-tags", default="",
                    help="comma list of image-name prefixes the avoid rows "
                         "must start with (frame numbers collide across "
                         "movies, e.g. 'ld_,lowdens_')")
    a = ap.parse_args()
    proj = Path(a.project_dir)
    proj.mkdir(parents=True, exist_ok=True)
    avoid: list[tuple[int, float, float]] = []
    if a.avoid_labels:
        import csv as _csv
        import re as _re
        tags = [s for s in a.avoid_tags.split(",") if s]
        with open(a.avoid_labels) as f:
            for row in _csv.DictReader(f):
                name = row.get("image_name", "")
                if tags and not any(name.startswith(t) for t in tags):
                    continue  # another movie's frames: numbers collide
                m = _re.search(r"_(\d+)\.png$", name)
                if not m:
                    continue
                try:
                    avoid.append((int(m.group(1)), float(row["x"]),
                                  float(row["y"])))
                except (KeyError, ValueError):
                    continue
    keep = tuple(s for s in a.keep_classes.split(",") if s)
    tasks = build_batch(a.movie, proj, Path(a.weights),
                        batch_size=a.batch_size, tag_prefix=a.tag_prefix,
                        movie_key=a.movie_key, mode=a.mode,
                        keep_classes=keep, avoid=tuple(avoid))
    store = AnnotationStore(proj / "annotations.db")
    try:
        by_cls: dict[str, int] = {}
        for t in tasks:
            store.save("task", t["uuid"], t, actor=a.actor)
            by_cls[t["stratum"]] = by_cls.get(t["stratum"], 0) + 1
        print(f"wrote {len(tasks)} tasks: {by_cls}", flush=True)
    finally:
        store.close()


if __name__ == "__main__":
    main()
