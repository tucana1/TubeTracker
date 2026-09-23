"""Immutable per-body arrays, loaded lazily instead of retaining a movie in RAM."""
from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from pathlib import Path

import numpy as np

from .analysis_contracts import file_hash


class BodyArrayStore:
    schema = "tubetracker.body_arrays.v1"

    def __init__(self, directory, entries=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.entries = dict(entries or {})
        self._verified = set()
        self._write_lock = threading.Lock()

    def __setitem__(self, key, array):
        with self._write_lock:
            self._publish(key, array)

    def _publish(self, key, array):
        if key in self.entries:
            raise ValueError("body evidence keys are immutable")
        array = np.asarray(array, dtype=np.float32)
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError("body evidence must be a finite 2-D array")
        # Distinct workers can compute the same key concurrently. Publish
        # complete content-addressed files; one worker cannot replace an
        # array already referenced by another worker's immutable manifest.
        with tempfile.NamedTemporaryFile(dir=self.directory, prefix=".body-",
                                         suffix=".tmp", delete=False) as stream:
            tmp = Path(stream.name)
            try:
                np.save(stream, array, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
        try:
            digest = file_hash(tmp)
            name = hashlib.sha256(str(key).encode()).hexdigest() + "-" + digest + ".npy"
            path = self.directory / name
            try:
                os.link(tmp, path)
            except FileExistsError:
                if file_hash(path) != digest:
                    raise ValueError("immutable body cache content is corrupt")
            self.entries[key] = {"file": name, "sha256": digest,
                                 "shape": list(array.shape), "dtype": "float32"}
        finally:
            tmp.unlink(missing_ok=True)

    def __getitem__(self, key):
        entry = self.entries[key]
        name = entry["file"]
        if Path(name).name != name:
            raise ValueError("body cache entry must be a local filename")
        path = self.directory / name
        if key not in self._verified:
            if file_hash(path) != entry["sha256"]:
                raise ValueError("cached body evidence checksum mismatch")
            self._verified.add(key)
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != entry["shape"] or str(array.dtype) != entry["dtype"]:
            raise ValueError("cached body evidence shape/dtype mismatch")
        return array

    def __len__(self):
        return len(self.entries)

    def manifest(self):
        return {"schema": self.schema, "entries": self.entries,
                "storage": "one native owner/frame array per file; read-only memory mapping"}
