"""The window itself on a tiny synthetic analysis: open, look, check, correct, export (skipped without a display)."""

import os
import sys

import pytest

wx = pytest.importorskip("wx")

from app_fixture import make_run  # noqa: E402


@pytest.fixture(scope="module")
def app():
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        pytest.skip("no display")
    a = wx.App.Get() or wx.App(False)
    if not wx.App.IsDisplayAvailable():
        pytest.skip("no display")
    yield a


def _yield(n=5):
    for _ in range(n):
        wx.GetApp().Yield(True)


def test_the_window_opens_an_analysis_and_takes_a_correction(app, tmp_path, monkeypatch):
    from tubetracker.app import window
    from tubetracker.app.window import MainFrame

    monkeypatch.setattr(window, "reveal", lambda path: None)  # no Finder windows from tests

    run = make_run(tmp_path / "runs" / "tiny", {"duration_s": 24000.0, "um_per_px": 0.5, "sample_id": "tiny"})
    frame = MainFrame(tmp_path / "runs")
    try:
        frame.Show()
        _yield()
        assert frame.start.IsShown() and frame.start.list.GetItemCount() == 1
        frame.open_path(str(run))
        frame.reviewer.wait(30)
        _yield()
        assert frame.movie.IsShown() and frame.data is not None and len(frame.grains) == 4
        assert "tiny" in frame.title.GetLabel() and "T50" in frame.line.GetLabel()
        frame.goto_check(1)
        assert frame.sel is not None and frame.side.gid.GetLabel() == frame.sel
        frame.select("g001", zoom=True)
        frame.set_bin(30)
        for mode in ("g", "h", "n"):
            frame.set_mode(mode)
        _yield()
        frame.snapshot(tmp_path / "window.png")
        assert (tmp_path / "window.png").stat().st_size > 1000
        frame.on_action("onset")
        assert frame._by_id["g001"]["onset"] == 30 and frame._by_id["g001"]["review"]["state"] == "corrected"
        frame.on_action("undo")
        assert frame._by_id["g001"]["onset"] == 10
        frame.set_tool("path")
        frame.on_click((68.0, 50.0), None)
        frame.on_click((98.0, 50.0), None)
        frame.finish_path()
        assert any(t["bin"] == 30 for t in frame._by_id["g001"]["human"])
        frame.on_results()
        _yield()
        assert frame.results_win.table.GetItemCount() == 4
        frame.on_export()
        assert (run / "results" / "grains.csv").exists()
        frame.close_movie()
        assert frame.start.IsShown()
    finally:
        frame.Destroy()
        _yield()


def test_a_finished_analysis_opens_by_itself(app, tmp_path):
    """Analyse (here a stand-in worker that puts an analysis in place): progress shows, then the movie opens."""
    import time
    from tubetracker.app.window import MainFrame

    run = make_run(tmp_path / "runs" / "tiny", {"duration_s": 24000.0})
    (run / "analysis").rename(run / "ready")
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import json, os, sys, time\n"
        "def emit(**k): print('@@ ' + json.dumps(k), flush=True)\n"
        "emit(phase='grains', label='Reading the grains', k=1, n=4); time.sleep(0.3)\n"
        f"os.rename({str(run / 'ready')!r}, {str(run / 'analysis')!r})\n"
        "emit(phase='done', label='Analysis complete')\n")
    frame = MainFrame(tmp_path / "runs")
    try:
        frame.jobs.command = lambda job: [sys.executable, "-u", str(worker)]
        frame.open_path(str(run))
        assert frame.analysis.IsShown() and frame.data is None
        frame.on_analyse()
        t0 = time.time()
        while frame.data is None and time.time() - t0 < 20:
            _yield(2)
            time.sleep(0.05)
        assert frame.data is not None and frame.movie.IsShown()
    finally:
        frame.Destroy()
        _yield()


def test_the_launcher_opens_the_window_on_an_analysis_folder(app, tmp_path):
    from tubetracker.app.window import open_window, parse_args

    run = make_run(tmp_path / "runs" / "tiny", {"duration_s": 24000.0})
    frame = open_window(parse_args(["--runs", str(tmp_path / "runs"), str(run)]))
    try:
        _yield(10)
        assert frame.data is not None and frame.movie.IsShown()
    finally:
        frame.Destroy()
        _yield()
