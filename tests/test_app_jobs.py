"""The analysis in a child process: progress read from its output, failure, cancelling, earlier results kept."""

import sys
import time

from tubetracker.app.jobs import JobManager
from tubetracker.app.worker import GRAIN_LINE, keep_earlier

FAKE = r"""
import json, sys, time
def emit(**k): print("@@ " + json.dumps(k), flush=True)
emit(phase="prepare", label="Movie already prepared", skip=True)
emit(phase="speed", label="Measuring")
print("growth scale 1.0 px/bin", flush=True)
for k in range(1, 4):
    emit(phase="grains", label="Reading the grains", k=k, n=3, gid=f"g00{k}")
    print(f"g00{k}: emerged_within onset 100", flush=True)
if sys.argv[1] == "fail":
    emit(phase="error", label="RuntimeError: no grains")
    sys.exit(1)
if sys.argv[1] == "slow":
    time.sleep(30)
emit(phase="finish", label="Saving")
emit(phase="done", label="Analysis complete")
"""


def _manager(tmp_path, mode):
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE)
    return JobManager(command=lambda job: [sys.executable, "-u", str(script), mode])


def _wait(job, states=("done", "failed", "cancelled"), timeout=20.0):
    t0 = time.time()
    while job.state not in states and time.time() - t0 < timeout:
        time.sleep(0.05)
    return job


def test_progress_is_read_from_the_workers_output(tmp_path):
    jobs = _manager(tmp_path, "ok")
    job = _wait(jobs.submit(str(tmp_path / "run"), None, "tiny"))
    assert job.state == "done" and job.n == 0 and not job.error
    states = {p["phase"]: p["state"] for p in job.to_json()["phases"]}
    assert states["prepare"] == "skipped" and states["grains"] == "done" and states["done"] == "done"
    assert any(line.startswith("g003:") for line in job.log)
    assert (tmp_path / "run" / "analysis_log.txt").exists()
    assert jobs.job_for(str(tmp_path / "run")) is job and not jobs.status()["active"]


def test_a_failing_analysis_says_why(tmp_path):
    job = _wait(_manager(tmp_path, "fail").submit(str(tmp_path / "run"), None, "tiny"))
    assert job.state == "failed" and job.error == "RuntimeError: no grains"


def test_an_analysis_can_be_cancelled(tmp_path):
    jobs = _manager(tmp_path, "slow")
    job = jobs.submit(str(tmp_path / "run"), None, "tiny")
    _wait(job, states=("running",))
    t0 = time.time()
    while job.k < 3 and time.time() - t0 < 10:
        time.sleep(0.05)
    jobs.cancel(job.id)
    assert _wait(job).state == "cancelled"
    assert jobs.submit(str(tmp_path / "run"), None, "tiny") is not job  # a new one can start


def test_the_workers_grain_lines_count_progress():
    assert GRAIN_LINE.match("g012: emerged_within       onset   7950  final   89.2 px").group(1) == "g012"
    assert GRAIN_LINE.match("u001: no_emergence_by_end  onset   None").group(1) == "u001"
    assert not GRAIN_LINE.match("growth scale 1.08 px/bin")


def test_an_earlier_analysis_is_kept_whole(tmp_path):
    for sub in ("analysis", "review", "results"):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "x.txt").write_text(sub)
    kept = keep_earlier(tmp_path)
    assert kept.parent == tmp_path / "earlier" and not (tmp_path / "analysis").exists()
    assert (kept / "review" / "x.txt").read_text() == "review"
    assert keep_earlier(tmp_path) is None
