"""A movie's run folder: where its cache, analysis, review and setup live.

The layout is SparseTrack's (``sparsetrack.cli.run_folder``)::

    runs/sparsetrack/<movie name>/
        setup.json            the app's setup: movie, how long it took, pixel size, sample metadata
        cache/                keyframe bins (bins.npy, meta.json) and the grain census (grains.json)
        analysis/             predictions.json, grains.csv, growth.csv, population.png, index.html, ...
        review/               review_labels.json: corrections (the labelling tool's format)
        results/              the app's exported tables and figures

An analysis folder made by ``sparsetrack analyze CACHE --out FOLDER`` (predictions.json directly inside, the
cache elsewhere) opens too: its cache is the one its predictions name.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_ROOT = REPO / "runs" / "sparsetrack"
MOVIE_SUFFIXES = (".mp4", ".avi", ".mov", ".m4v", ".mkv", ".mpg", ".mpeg", ".wmv")
SETUP_SCHEMA = "tubetracker.setup.v1"
SETUP_FIELDS = ("duration_s", "s_per_frame", "um_per_px", "sample_id", "genotype", "replicate", "notes", "flatfield",
                "setup_done")


def safe_name(stem: str) -> str:
    """A movie's run-folder name, as ``sparsetrack.cli.run_folder`` makes it."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_.") or "movie"


def is_movie(path: str | Path) -> bool:
    return Path(path).suffix.lower() in MOVIE_SUFFIXES


@dataclass(frozen=True)
class RunFolder:
    root: Path

    def __post_init__(self):
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())

    # ---- where things are --------------------------------------------------------
    @property
    def name(self) -> str:
        return self.root.name

    @property
    def analysis(self) -> Path:
        if not (self.root / "analysis").is_dir() and (self.root / "predictions.json").exists():
            return self.root  # a bare analysis folder (sparsetrack analyze --out)
        return self.root / "analysis"

    @property
    def predictions(self) -> Path:
        return self.analysis / "predictions.json"

    @cached_property
    def cache(self) -> Path:
        own = self.root / "cache"
        if (own / "meta.json").exists() or not self.predictions.exists():
            return own
        named = None  # a bare analysis folder: the cache its predictions were made from (named near the start)
        try:
            with open(self.predictions) as fh:
                m = re.search(r'"cache":\s*"((?:[^"\\]|\\.)*)"', fh.read(8192))
            named = json.loads(f'"{m.group(1)}"') if m else json.loads(self.predictions.read_text()).get("cache")
        except (OSError, ValueError):
            pass
        return Path(named) if named and (Path(named) / "meta.json").exists() else own

    @property
    def review_labels(self) -> Path:
        return self.root / "review" / "review_labels.json"

    @property
    def setup_path(self) -> Path:
        return self.root / "setup.json"

    @property
    def results(self) -> Path:
        return self.root / "results"

    def has_cache(self) -> bool:
        return (self.cache / "meta.json").exists() and (self.cache / "grains.json").exists()

    def has_analysis(self) -> bool:
        return self.predictions.exists()

    # ---- setup ---------------------------------------------------------------------
    def load_setup(self) -> dict:
        try:
            return json.loads(self.setup_path.read_text())
        except (OSError, ValueError):
            return {}

    def save_setup(self, values: dict) -> dict:
        """Merge ``values`` into the setup (unknown keys are ignored) and write it; returns the setup."""
        setup = self.load_setup()
        for key in ("movie", "movie_name", "n_frames", *SETUP_FIELDS):
            if key in values:
                setup[key] = values[key]
        setup.update(schema=SETUP_SCHEMA, updated=time.strftime("%Y-%m-%dT%H:%M:%S"))
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.setup_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(setup, indent=1))
        os.replace(tmp, self.setup_path)
        return setup

    def apply_setup(self, values: dict) -> dict:
        """Save what a person entered in the setup dialog: how long the movie ran (``hours``, ``minutes``) or else
        its frame interval (``s_per_frame``), the pixel size (``um_per_px``), the sample's metadata and the
        flat-field choice. Blank numbers are unknown. Returns the setup."""
        from .units import duration_seconds

        def number(key):
            v = values.get(key)
            if v in (None, ""):
                return None
            v = float(str(v).replace(",", "."))
            if not v > 0:
                raise ValueError(f"{key.replace('_', ' ')} must be a positive number")
            return v

        out = {"setup_done": True, "duration_s": None, "s_per_frame": None}
        duration = duration_seconds(values.get("hours"), values.get("minutes"), values.get("seconds"))
        spf = number("s_per_frame")
        if duration:
            out["duration_s"] = duration
        elif spf:
            n = self.n_frames()
            out["s_per_frame"] = spf
            out["duration_s"] = spf * n if n else None
        if "um_per_px" in values:
            out["um_per_px"] = number("um_per_px")
        for key in ("sample_id", "genotype", "replicate", "notes"):
            if key in values:
                out[key] = str(values.get(key) or "").strip()
        if "flatfield" in values:
            out["flatfield"] = bool(values["flatfield"])
        return self.save_setup(out)

    def cache_meta(self) -> dict:
        try:
            return json.loads((self.cache / "meta.json").read_text())
        except (OSError, ValueError):
            return {}

    def movie_path(self) -> Path | None:
        """The movie this folder was made from (setup, else the cache's record of it)."""
        path = self.load_setup().get("movie") or (self.cache_meta().get("movie") or {}).get("path")
        return Path(path) if path else None

    def n_frames(self) -> int | None:
        n = (self.cache_meta().get("movie") or {}).get("n_frames") or self.load_setup().get("n_frames")
        return int(n) if n else None

    def reviewed_at(self) -> float | None:
        """When a person last answered in this movie's review (the labels file written after its pre-fill), if ever."""
        labels, model = self.review_labels, self.review_labels.with_suffix(".model.json")
        if not labels.exists():
            return None
        t = labels.stat().st_mtime
        return t if not model.exists() or t > model.stat().st_mtime + 1.0 else None

    def movie_name(self) -> str:
        return (self.load_setup().get("movie_name") or (self.cache_meta().get("movie") or {}).get("name")
                or self.name)

    def is_setup(self) -> bool:
        """Whether a person saved this movie's setup (it may leave the duration blank: times then stay in frames)."""
        return bool(self.load_setup().get("setup_done"))

    # ---- listing ---------------------------------------------------------------------
    def listing(self) -> dict:
        """What the start screen lists for this folder, from small files only (no predictions are read)."""
        setup = self.load_setup()
        meta = self.cache_meta()
        movie = meta.get("movie") or {}
        analysed = self.predictions.stat().st_mtime if self.has_analysis() else None
        reviewed = self.reviewed_at()
        return {"folder": str(self.root), "name": self.name, "movie": self.movie_name(),
                "movie_path": str(self.movie_path() or ""),
                "sample_id": setup.get("sample_id") or "", "genotype": setup.get("genotype") or "",
                "replicate": setup.get("replicate") or "",
                "duration_s": setup.get("duration_s"), "um_per_px": setup.get("um_per_px"),
                "prepared": self.has_cache(), "analysed": analysed, "reviewed": reviewed,
                "updated": max(t for t in (analysed, reviewed, self.setup_path.stat().st_mtime
                                           if self.setup_path.exists() else None, 0.0) if t is not None)}


def folder_for(path: str | Path, runs_root: str | Path = DEFAULT_RUNS_ROOT) -> RunFolder:
    """The run folder for ``path``: a movie's own folder under ``runs_root``, or ``path`` itself when it is a run
    folder (or a bare analysis folder, or the ``analysis`` folder inside one)."""
    p = Path(path).expanduser()
    if p.is_file() and p.name == "predictions.json":
        p = p.parent
    if p.is_dir():
        if p.name == "analysis" and (p / "predictions.json").exists():
            return RunFolder(p.parent)
        return RunFolder(p)
    return RunFolder(Path(runs_root) / safe_name(p.stem))


def looks_like_run(p: Path) -> bool:
    return any((p / f).exists() for f in ("setup.json", "cache/meta.json", "analysis/predictions.json",
                                          "predictions.json"))


def find_runs(runs_root: str | Path = DEFAULT_RUNS_ROOT) -> list[RunFolder]:
    """Run folders directly under ``runs_root``, most recently changed first."""
    root = Path(runs_root)
    if not root.is_dir():
        return []
    found = [RunFolder(p) for p in root.iterdir() if p.is_dir() and not p.name.startswith(".") and looks_like_run(p)]
    return sorted(found, key=lambda f: -f.listing()["updated"])


def movie_frame_count(path: str | Path) -> int | None:
    """A movie's number of frames (ffprobe's stream count, else OpenCV's), without decoding it."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                                  "stream=nb_frames", "-of", "csv=p=0", str(path)],
                                 capture_output=True, text=True, timeout=30).stdout.strip().split(",")[0]
            if out.isdigit() and int(out) > 0:
                return int(out)
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n if n > 0 else None
    except Exception:  # noqa: BLE001 - a movie OpenCV cannot read: the count stays unknown
        return None


# ---- the lab's defaults, remembered across movies --------------------------------------
def load_prefs(runs_root: str | Path) -> dict:
    """The last pixel size used (and similar), else the lab's calibration.json (the command-line launcher's)."""
    prefs = {}
    cal = REPO / "calibration.json"
    if cal.exists():
        try:
            c = json.loads(cal.read_text())
            prefs.update({k: c[k] for k in ("um_per_px", "s_per_frame") if c.get(k)})
        except ValueError:
            pass
    try:
        prefs.update(json.loads((Path(runs_root) / "app_prefs.json").read_text()))
    except (OSError, ValueError):
        pass
    return prefs


def save_prefs(runs_root: str | Path, **values) -> None:
    path = Path(runs_root) / "app_prefs.json"
    prefs = {}
    try:
        prefs = json.loads(path.read_text())
    except (OSError, ValueError):
        pass
    prefs.update({k: v for k, v in values.items() if v not in (None, "")})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prefs, indent=1))
