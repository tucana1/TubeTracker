"""Content identities for an analysis, independent of Qt and model runtimes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class FileFingerprinter:
    """Rehash changed files; UI refreshes only stat unchanged movie files."""
    def __init__(self):
        self._cache = {}

    @staticmethod
    def _identity(stat):
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def file(self, path):
        path = Path(path).resolve()
        try:
            before = self._identity(path.stat())
        except FileNotFoundError:
            self._cache.pop(str(path), None)
            return {"path": str(path), "missing": True}
        cached = self._cache.get(str(path))
        if cached is not None and cached[0] == before:
            return dict(cached[1])
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if self._identity(path.stat()) != before:
            raise RuntimeError(f"Analysis input changed while being read: {path.name}")
        result = {"path": str(path), "sha256": digest.hexdigest()}
        self._cache[str(path)] = (before, result)
        return dict(result)


def code_files():
    paths = [p for directory in (ROOT / "tubetracker", ROOT / "prototypes/v30_video_apex")
             for p in directory.rglob("*.py") if "__pycache__" not in p.parts]
    paths += [ROOT / "prototypes/timesfm_tip_forecast/grain_detect.py",
              ROOT / "scripts/analyze_movie.py"]
    return sorted(set(paths))


def code_fingerprint(fingerprinter):
    return {str(p.relative_to(ROOT)): fingerprinter.file(p)["sha256"] for p in code_files()}


_PROCESS_FILES = FileFingerprinter()
_PROCESS_CODE = code_fingerprint(_PROCESS_FILES)


def assert_runtime_code_current():
    if code_fingerprint(_PROCESS_FILES) != _PROCESS_CODE:
        raise RuntimeError("Analysis source changed in this process. Restart the analysis worker before recomputing.")


def dependency_fingerprint(config, fingerprinter=None):
    fp = fingerprinter or FileFingerprinter()
    request = config["request"]
    files = {"movie": fp.file(request["movie_path"])}
    for name in ("cap_checkpoint", "body_checkpoint"):
        if config.get(name):
            files[name] = fp.file(config[name])
    if request.get('body_assistance'):
        from .seeded_body import assistance_files
        files.update({'body_assistance/' + name: value
                      for name, value in assistance_files(request['body_assistance'], fp).items()})
    if request.get('grain_pose_path'):
        from .frame_geometry import pose_dependencies
        files.update({'grain_geometry/' + name: value
                      for name, value in pose_dependencies(request['grain_pose_path'], fp).items()})
    if request.get('grain_motion'):
        from .grain_motion import motion_dependencies
        files.update({'grain_motion/' + name: value
                      for name, value in motion_dependencies(request['grain_motion'],fp).items()})
    if request.get('discover_grains'):
        from .grain_detection import GrainDetectionConfig
        detector = GrainDetectionConfig.from_dict(request.get('grain_detection'))
        if detector.backend == 'cpdino':
            files['grain_checkpoint'] = fp.file(detector.checkpoint)
    snapshot = request.get("snapshot")
    if snapshot:
        directory = Path(snapshot)
        # Include actual content as well as the manifest. An accidental edit
        # without a manifest update must still invalidate a saved result.
        paths = set(directory.glob("*.json")) | {directory / "snapshot_manifest.json"}
        files.update({"snapshot/" + p.name: fp.file(p) for p in sorted(paths)})
    validation = request.get('route_validation')
    if validation:
        files['route_validation'] = fp.file(validation)
        if not files['route_validation'].get('missing'):
            record = json.loads(Path(validation).read_text())
            for name, evidence in record.get('evidence_files', {}).items():
                files['route_validation/' + name] = fp.file(evidence['path'])
    return {"schema": "tubetracker.analysis_dependencies.v1", "files": files,
            "code": code_fingerprint(fp)}


def require_same_dependencies(expected, actual):
    if expected != actual:
        raise RuntimeError("Movie, model, snapshot or source changed during analysis. Recompute from the current inputs.")
