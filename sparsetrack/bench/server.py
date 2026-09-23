"""Local HTTP server for benchmark labelling (standard library only; binds to 127.0.0.1).

Labels use the vocabulary of ``tubetracker.annotation_schema.GerminationEvent``
(verdicts ``emerged_at_start`` / ``emerged_within`` / ``no_emergence_by_end`` /
``unobservable``; bracket endpoints ``last_absent_frame`` / ``first_visible_frame``)
and of the centerline observations (``path_xy``, ``path_complete``, ``direct_state``),
so answers can later be imported into an annotation project.

Every judgement is made on a registered bin average; a bin decision is stored both
as the bin index and as the bin-centre source frame.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from .. import __version__, stack
from ..grains import annotate_layout
from ..render import Renderer, png

SCHEMA = "sparsetrack.bench.v1"
VERDICTS = ("emerged_at_start", "emerged_within", "no_emergence_by_end", "unobservable")
TRACE_STATES = ("full", "partial", "no_tube", "unsure")
EXCLUDE_REASONS = ("not_a_grain", "clump", "edge", "out_of_focus", "other")
STATIC = Path(__file__).with_name("static")

# display layouts (CSS px): coarse = whole movie, fine = single bins around a transition
COARSE = {"bins_per_tile": 4, "half": 28, "zoom": 2.0, "cols": 11, "header": 16, "gap": 2}
FINE = {"n_tiles": 18, "half": 24, "zoom": 4.0, "cols": 6, "header": 16, "gap": 2}
TRACE_VIEWS = {"near": {"half": 64, "zoom": 5.0}, "wide": {"half": 128, "zoom": 2.5}}
RETEST_SIZE = 8


def trace_bins(first_visible_bin: int, n_bins: int) -> list[int]:
    """Bins at which a germinated grain's tube is traced: 6 bins after onset, then fixed times.

    Every trace is at least 6 bins after the first visible bin, so no trace asks for a
    tube only a pixel long.
    """
    last = n_bins - 2  # the final bin is usually partial
    early = first_visible_bin + 6
    fixed = [round(0.4 * last), round(0.7 * last), last]
    wanted = sorted({b for b in [early, *fixed] if early <= b <= last})
    merged: list[int] = []
    for b in wanted:
        if not merged or b - merged[-1] >= 8:
            merged.append(b)
        elif b == last:
            merged[-1] = b
    return merged


class Bench:
    def __init__(self, cache_dir: str | Path, labels_path: str | Path, annotator: str = "investigator"):
        self.cache_dir = Path(cache_dir)
        self.bins, self.meta = stack.load(self.cache_dir)
        self.renderer = Renderer(self.bins, self.meta)
        self.fpb = int(self.meta["frames_per_bin"])
        self.n_bins = int(self.meta["n_bins"])
        self.labels_path = Path(labels_path)
        self.journal_path = self.labels_path.with_suffix(".journal.jsonl")
        self.annotator = annotator
        self.lock = threading.Lock()
        if self.labels_path.exists():
            self.doc = json.loads(self.labels_path.read_text())
            self._check_movie()
        else:
            self.doc = self._new_doc()
            self.save("create")

    # ---- document -----------------------------------------------------------------
    def _new_doc(self) -> dict:
        grains = json.loads((self.cache_dir / "grains.json").read_text())["grains"]
        return {
            "schema": SCHEMA,
            "sparsetrack_version": __version__,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "movie": self.meta["movie"],
            "frames_per_bin": self.fpb,
            "n_bins": self.n_bins,
            "coordinates": ("reference: registered to the mean of bins 0-2; raw source "
                            "coordinates at a bin = reference + shift[bin]"),
            "bin_semantics": ("bin b averages the keyframes of source frames [b*fpb, (b+1)*fpb); "
                              "*_frame fields hold the bin-centre frame b*fpb + fpb//2"),
            "grains": {g["id"]: g for g in grains},
            "labels": {},
            "retest": {"grains": [], "labels": {}},
        }

    def _check_movie(self) -> None:
        a, b = self.doc.get("movie", {}), self.meta["movie"]
        if (a.get("name"), a.get("size_bytes")) != (b.get("name"), b.get("size_bytes")):
            raise SystemExit(f"labels {self.labels_path} belong to {a.get('name')!r}, "
                             f"cache is {b.get('name')!r}; refusing to mix them")

    def save(self, event: str, payload: dict | None = None) -> None:
        self.doc["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.labels_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.labels_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.doc, indent=1))
        os.replace(tmp, self.labels_path)
        with open(self.journal_path, "a") as fh:
            fh.write(json.dumps({"t": self.doc["updated"], "event": event, "payload": payload}) + "\n")

    def bin_centre(self, b: int) -> int:
        return int(b) * self.fpb + self.fpb // 2

    def grain(self, gid: str) -> dict:
        if gid not in self.doc["grains"]:
            raise KeyError(gid)
        return self.doc["grains"][gid]

    def order(self) -> list[str]:
        """Isolated, included grains first; then other included grains; excluded last."""
        def key(item):
            gid, g = item
            return (bool(g.get("excluded")), not g.get("isolated", False), gid)
        return [gid for gid, _ in sorted(self.doc["grains"].items(), key=key)]

    def state(self) -> dict:
        labels = self.doc["labels"]
        grains = self.doc["grains"]
        todo = [gid for gid in self.order() if not grains[gid].get("excluded")]
        onset_done = sum(1 for gid in todo if labels.get(gid, {}).get("onset"))
        traces_needed = traces_done = 0
        plan = {}
        for gid in todo:
            onset = labels.get(gid, {}).get("onset") or {}
            if onset.get("verdict") in ("emerged_within", "emerged_at_start"):
                fv = onset.get("first_visible_bin")
                fv = 0 if fv is None else fv
                plan[gid] = trace_bins(fv, self.n_bins)
                traces_needed += len(plan[gid])
                traces_done += sum(1 for b in plan[gid] if str(b) in labels[gid].get("traces", {}))
        return {
            "movie": self.doc["movie"], "n_bins": self.n_bins, "frames_per_bin": self.fpb,
            "shifts": self.meta["shifts"], "order": self.order(), "grains": grains,
            "labels": labels, "retest": self.doc["retest"], "trace_plan": plan,
            "layout": {"coarse": COARSE, "fine": FINE, "trace": TRACE_VIEWS},
            "verdicts": VERDICTS, "trace_states": TRACE_STATES, "exclude_reasons": EXCLUDE_REASONS,
            "progress": {"grains": len(todo), "onset_done": onset_done,
                         "traces_needed": traces_needed, "traces_done": traces_done},
            "labels_path": str(self.labels_path),
        }

    # ---- updates --------------------------------------------------------------------
    def set_onset(self, gid: str, body: dict, retest: bool = False) -> dict:
        verdict = body.get("verdict")
        if verdict not in VERDICTS:
            raise ValueError(f"bad verdict {verdict!r}")
        la, fv = body.get("last_absent_bin"), body.get("first_visible_bin")
        la = None if la is None else int(la)
        fv = None if fv is None else int(fv)
        if verdict == "emerged_within":
            if fv is None:
                raise ValueError("emerged_within needs first_visible_bin")
            if la is None:
                la = fv - 1 if fv > 0 else None
            if la is not None and la >= fv:
                raise ValueError("last_absent_bin must precede first_visible_bin")
        elif verdict == "emerged_at_start":
            la, fv = None, 0
        else:
            la = fv = None
        record = {
            "verdict": verdict,
            "last_absent_bin": la, "first_visible_bin": fv,
            "last_absent_frame": None if la is None else self.bin_centre(la),
            "first_visible_frame": None if fv is None else self.bin_centre(fv),
            "window_start": 0, "window_end": int(self.meta["movie"]["n_frames"]) - 1,
            "coarse_tile": body.get("coarse_tile"),
            "consulted_frames": sorted({self.bin_centre(b) for b in body.get("consulted_bins", [])}),
            "annotator": self.annotator, "review_origin": "human",
            "view": "registered bin averages", "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with self.lock:
            self.grain(gid)
            target = self.doc["retest"]["labels"] if retest else self.doc["labels"]
            entry = target.setdefault(gid, {})
            entry["onset"] = record
            if "time_spent_s" in body:
                entry["time_spent_s"] = round(float(body["time_spent_s"]), 1)
            self.save("retest_onset" if retest else "onset", {"grain": gid, **record})
        return record

    def set_trace(self, gid: str, body: dict) -> dict:
        state = body.get("state")
        if state not in TRACE_STATES:
            raise ValueError(f"bad trace state {state!r}")
        b = int(body["bin"])
        pts = [[round(float(x), 2), round(float(y), 2)] for x, y in body.get("points", [])]
        if state in ("full", "partial") and len(pts) < 2:
            raise ValueError("a traced tube needs at least an exit and an apex point")
        if state in ("no_tube", "unsure"):
            pts = pts if state == "unsure" else []
        dx, dy = self.meta["shifts"][b]
        length = float(np.sum(np.hypot(*np.diff(np.array(pts), axis=0).T))) if len(pts) > 1 else 0.0
        record = {
            "bin": b, "source_frame": self.bin_centre(b), "state": state,
            "path_xy_ref": pts,
            "path_xy": [[round(x + dx, 2), round(y + dy, 2)] for x, y in pts],
            "path_complete": state == "full",
            "direct_state": {"full": "direct_visible", "partial": "direct_visible",
                             "no_tube": "no_tube_visible", "unsure": "not_directly_visible"}[state],
            "length_px": round(length, 2),
            "contact": bool(body.get("contact", False)),
            "view": body.get("view", "near"),
            "annotator": self.annotator, "review_origin": "human",
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with self.lock:
            self.grain(gid)
            entry = self.doc["labels"].setdefault(gid, {})
            entry.setdefault("traces", {})[str(b)] = record
            if "time_spent_s" in body:
                entry["time_spent_s"] = round(float(body["time_spent_s"]), 1)
            self.save("trace", {"grain": gid, **record})
        return record

    def set_exclusion(self, gid: str, body: dict) -> dict:
        excluded = bool(body.get("excluded"))
        reason = body.get("reason") if excluded else None
        if excluded and reason not in EXCLUDE_REASONS:
            raise ValueError(f"bad exclusion reason {reason!r}")
        with self.lock:
            g = self.grain(gid)
            g["excluded"], g["exclude_reason"] = excluded, reason
            self.save("exclude", {"grain": gid, "excluded": excluded, "reason": reason})
        return g

    def add_grain(self, body: dict) -> dict:
        x, y = float(body["x"]), float(body["y"])
        r = float(body.get("r", 13.0))
        with self.lock:
            n = 1 + sum(1 for gid in self.doc["grains"] if gid.startswith("u"))
            gid = f"u{n:03d}"
            grains = list(self.doc["grains"].values()) + [
                {"id": gid, "x": x, "y": y, "r": r, "source": "user", "ring_contrast": None,
                 "body_contrast": None}]
            laid = annotate_layout(grains, (self.renderer.height, self.renderer.width))
            self.doc["grains"] = {g["id"]: {**self.doc["grains"].get(g["id"], {}), **g} for g in laid}
            self.save("add_grain", {"grain": gid, "x": x, "y": y, "r": r})
        return self.doc["grains"][gid]

    def pick_retest(self) -> list[str]:
        with self.lock:
            if not self.doc["retest"]["grains"]:
                done = sorted(gid for gid, lab in self.doc["labels"].items()
                              if (lab.get("onset") or {}).get("verdict") == "emerged_within")
                random.Random(20260923).shuffle(done)
                self.doc["retest"]["grains"] = done[:RETEST_SIZE]
                self.save("retest_pick", {"grains": self.doc["retest"]["grains"]})
        return self.doc["retest"]["grains"]

    # ---- images ---------------------------------------------------------------------
    def coarse_png(self, gid: str, mode: str) -> bytes:
        g = self.grain(gid)
        k = COARSE["bins_per_tile"]
        ranges = [(b, min(b + k - 1, self.n_bins - 1)) for b in range(0, self.n_bins, k)]
        labels = [f"{b0 * self.fpb}" for b0, _ in ranges]
        return png(self.renderer.strip(gid, g["x"], g["y"], ranges, labels, COARSE["half"], COARSE["zoom"],
                                       COARSE["cols"], mode, COARSE["header"], COARSE["gap"]))

    def fine_png(self, gid: str, start: int, mode: str) -> bytes:
        g = self.grain(gid)
        start = max(0, min(int(start), self.n_bins - FINE["n_tiles"]))
        ranges = [(b, b) for b in range(start, min(self.n_bins, start + FINE["n_tiles"]))]
        labels = [f"bin {b}  f{self.bin_centre(b)}" for b, _ in ranges]
        return png(self.renderer.strip(gid, g["x"], g["y"], ranges, labels, FINE["half"], FINE["zoom"],
                                       FINE["cols"], mode, FINE["header"], FINE["gap"]))

    def frame_png(self, gid: str, b: int, view: str, mode: str, smooth: int) -> bytes:
        g = self.grain(gid)
        v = TRACE_VIEWS[view]
        b0, b1 = int(b) - smooth, int(b) + smooth
        img = self.renderer.mean_crop(b0, b1, g["x"], g["y"], v["half"])
        window = self.renderer.contrast(gid, g["x"], g["y"], v["half"], mode)
        return png(self.renderer.to_display(img, window, v["zoom"]))

    def field_png(self, which: str) -> bytes:
        key = f"field_{which}"
        cached = self.cache_dir / f"{key}.png"
        if not cached.exists():
            cached.write_bytes(png(self.renderer.field(which, 0.68)))
        return cached.read_bytes()


def make_handler(bench: Bench):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the terminal quiet
            pass

        def _send(self, status, body: bytes, ctype: str):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status=HTTPStatus.OK):
            self._send(status, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[-1] for k, v in parse_qs(url.query).items()}
            parts = [p for p in url.path.split("/") if p]
            try:
                if url.path in ("/", "/index.html"):
                    return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                if parts[:1] == ["static"] and len(parts) == 2:
                    f = STATIC / parts[1]
                    ctype = {"js": "text/javascript", "css": "text/css"}.get(f.suffix[1:], "text/plain")
                    return self._send(200, f.read_bytes(), ctype + "; charset=utf-8")
                if parts == ["api", "state"]:
                    return self._json(bench.state())
                if parts == ["api", "retest"]:
                    return self._json({"grains": bench.pick_retest()})
                if parts[:2] == ["api", "img"]:
                    kind, mode = parts[2], q.get("contrast", "n")
                    if kind == "field":
                        return self._send(200, bench.field_png(q.get("which", "early")), "image/png")
                    gid = parts[3]
                    if kind == "coarse":
                        return self._send(200, bench.coarse_png(gid, mode), "image/png")
                    if kind == "fine":
                        return self._send(200, bench.fine_png(gid, int(q.get("start", 0)), mode), "image/png")
                    if kind == "frame":
                        return self._send(200, bench.frame_png(gid, int(q["bin"]), q.get("view", "near"), mode,
                                                               int(q.get("smooth", 0))), "image/png")
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (KeyError, ValueError, IndexError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def do_POST(self):
            parts = [p for p in urlparse(self.path).path.split("/") if p]
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                if parts[:2] == ["api", "onset"]:
                    return self._json(bench.set_onset(parts[2], body))
                if parts[:2] == ["api", "retest"]:
                    return self._json(bench.set_onset(parts[2], body, retest=True))
                if parts[:2] == ["api", "trace"]:
                    return self._json(bench.set_trace(parts[2], body))
                if parts[:2] == ["api", "exclude"]:
                    return self._json(bench.set_exclusion(parts[2], body))
                if parts == ["api", "grain"]:
                    return self._json(bench.add_grain(body))
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    return Handler


def serve(cache_dir, labels_path, port: int = 8765, open_browser: bool = True, annotator: str = "investigator"):
    bench = Bench(cache_dir, labels_path, annotator)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(bench))
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Benchmark labelling at {url}\nLabels: {bench.labels_path}\nStop with Ctrl-C.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
