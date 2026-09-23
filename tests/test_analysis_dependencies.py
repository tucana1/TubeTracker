import copy
import json
import os

import pytest
import torch

from prototypes.v30_video_apex.native_caps import NativeCapNet, save_native_checkpoint, file_hash
from prototypes.v30_video_apex.native_body import NativeBodyNet, save_body_checkpoint
from tubetracker import analysis_dependencies as dependencies
from tubetracker.analysis_workbench import AnalysisSession
from tubetracker.annotation_store import AnnotationStore
from tubetracker.movie_analysis import MovieAnalysisService
from test_movie_analysis import service_fixture


def test_same_path_model_replacement_reloads_actual_weights(tmp_path):
    cap, body = NativeCapNet(base=4), NativeBodyNet(base=4)
    cp, bp = tmp_path / "cap.pt", tmp_path / "body.pt"
    save_native_checkpoint(cp, cap, manifest={})
    save_body_checkpoint(bp, body, {})
    service = MovieAnalysisService(cp, body_checkpoint=bp, cache_dir=tmp_path / "cache")
    original_hashes = service._load_models()
    old_cap, old_body = service._cap_model, service._body_model
    with torch.no_grad():
        cap.cap.bias.fill_(3.25)
        body.body.bias.fill_(-4.75)
    save_native_checkpoint(tmp_path / "new-cap.pt", cap, manifest={})
    save_body_checkpoint(tmp_path / "new-body.pt", body, {})
    os.replace(tmp_path / "new-cap.pt", cp)
    os.replace(tmp_path / "new-body.pt", bp)
    with pytest.raises(RuntimeError, match="changed"):
        service._load_models(original_hashes)
    actual = service._load_models()
    assert service._cap_model is not old_cap and service._body_model is not old_body
    assert actual == {"cap": file_hash(cp), "body": file_hash(bp)} != original_hashes
    assert service._cap_model.cap.bias.item() == 3.25
    assert service._body_model.body.bias.item() == -4.75
    loaded = service._body_model
    service._load_models(actual)
    assert service._body_model is loaded


@pytest.mark.parametrize("changed", ["movie.bin", "cap.pt", "body.pt", "snapshot_manifest.json", "census.json"])
def test_changed_input_invalidates_export_even_after_restart(tmp_path, changed):
    service, request, _ = service_fixture(tmp_path)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name in ("snapshot_manifest.json", "census.json", "observations.json"):
        (snapshot / name).write_text("{}" if name.startswith("snapshot") else "[]")
    request.snapshot = str(snapshot)
    config = {"request": request.to_dict(), "cap_checkpoint": service.cap_checkpoint,
              "body_checkpoint": service.body_checkpoint}
    store = AnnotationStore(tmp_path / "annotations.db")
    session = AnalysisSession(store, config, service=service)
    session.recompute()
    assert not session.stale
    target = (snapshot if changed.endswith(".json") else tmp_path) / changed
    target.write_bytes(target.read_bytes() + b"\n")
    assert session.stale
    with pytest.raises(ValueError, match="Recompute"):
        session.export()
    store.close()
    store = AnnotationStore(tmp_path / "annotations.db")
    resumed = AnalysisSession(store, service=service)
    assert resumed.stale
    resumed.recompute()
    assert not resumed.stale
    store.close()


def test_source_change_requires_fresh_worker_and_stales_export(tmp_path, monkeypatch):
    source = tmp_path / "worker.py"
    source.write_text("version = 1\n")
    monkeypatch.setattr(dependencies, "ROOT", tmp_path)
    monkeypatch.setattr(dependencies, "code_files", lambda: [source])
    monkeypatch.setattr(dependencies, "_PROCESS_CODE", dependencies.code_fingerprint(dependencies.FileFingerprinter()))
    service, request, _ = service_fixture(tmp_path)
    store = AnnotationStore(tmp_path / "annotations.db")
    config = {"request": request.to_dict(), "cap_checkpoint": service.cap_checkpoint,
              "body_checkpoint": service.body_checkpoint}
    session = AnalysisSession(store, config, service=service)
    session.recompute()
    source.write_text("version = 2\n")
    assert session.stale
    with pytest.raises(RuntimeError, match="Restart the analysis worker"):
        session.recompute()
    store.close()


def test_changed_input_during_run_cannot_publish_pixel_cache(tmp_path):
    service, request, _ = service_fixture(tmp_path)
    original = service.pixel_provider
    def changing_provider(req, owners):
        result = original(req, owners)
        (tmp_path / "body.pt").write_bytes(b"changed during inference")
        return result
    service.pixel_provider = changing_provider
    with pytest.raises(RuntimeError, match="changed during analysis"):
        service.analyze(request)
    assert not list((tmp_path / "cache").glob("pixels-*.json"))


def test_unchanged_large_file_is_not_reopened_on_ui_refresh(tmp_path, monkeypatch):
    from pathlib import Path
    p = tmp_path / "movie.bin"
    p.write_bytes(b"a" * 4096)
    fp = dependencies.FileFingerprinter()
    first = fp.file(p)
    original_open, reads = Path.open, []
    def counted(path, *args, **kwargs):
        reads.append(str(path))
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", counted)
    assert fp.file(p) == fp.file(p) == first
    assert reads == []
