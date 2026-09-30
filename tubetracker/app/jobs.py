"""Analyses running in the background: one movie at a time, each in its own process (``worker.py``), in order.

The window polls ``status()``: the running job's phase and progress (keyframes read, tube maps built, grain k of
n), how long it has taken, the log's last lines, and the queue. ``cancel`` stops a job (its whole process group,
so ffmpeg goes too); a stopped analysis leaves the earlier one untouched.
"""

from __future__ import annotations

import itertools
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from .runfolder import REPO

PHASES = ("probe", "prepare", "register", "census", "maps", "speed", "grains", "finish", "done")
PHASE_LABELS = {"probe": "Read the movie", "prepare": "Read its keyframes", "register": "Register the frames",
                "census": "Find the grains", "maps": "Build the tube maps", "speed": "Measure growth speed",
                "grains": "Read every grain", "finish": "Save results", "done": "Done"}


class Job:
    _ids = itertools.count(1)

    def __init__(self, folder: str, movie: str | None, name: str, flatfield: bool = False):
        self.id = next(self._ids)
        self.folder, self.movie, self.name, self.flatfield = folder, movie, name, flatfield
        self.state = "queued"
        self.phase, self.label, self.k, self.n = None, "Waiting for the analysis before it", 0, 0
        self.phases: dict[str, dict] = {}
        self.queued_at, self.started, self.finished = time.time(), None, None
        self.error: str | None = None
        self.log: deque[str] = deque(maxlen=300)
        self.proc: subprocess.Popen | None = None
        self.cancelled = False
        self.grain_times: list[float] = []

    def update(self, msg: dict) -> None:
        phase = msg.get("phase")
        if phase == "error":
            self.error = msg.get("label")
            return
        if phase not in PHASES:
            return
        for p in PHASES[: PHASES.index(phase)]:  # earlier phases are over
            if p in self.phases and self.phases[p]["state"] == "running":
                self.phases[p]["state"] = "done"
        rec = self.phases.setdefault(phase, {"state": "running", "label": msg.get("label") or PHASE_LABELS[phase]})
        if msg.get("skip"):
            rec["state"] = "skipped"
            rec["label"] = msg.get("label") or rec["label"]
            return
        rec["state"] = "done" if phase == "done" else "running"
        self.phase, self.label = phase, msg.get("label") or self.label
        self.k, self.n = int(msg.get("k") or 0), int(msg.get("n") or 0)
        if phase == "grains" and self.k:
            self.grain_times.append(time.time())

    def eta(self) -> float | None:
        """Seconds left for the grains, from how fast the last ones went."""
        if self.phase != "grains" or self.k < 3 or not self.n:
            return None
        recent = self.grain_times[-10:]
        per = (recent[-1] - recent[0]) / max(len(recent) - 1, 1)
        return max(0.0, per * (self.n - self.k))

    def to_json(self) -> dict:
        now = time.time()
        return {"id": self.id, "folder": self.folder, "movie": self.movie, "name": self.name, "state": self.state,
                "phase": self.phase, "label": self.label, "k": self.k, "n": self.n,
                "phases": [{"phase": p, "title": PHASE_LABELS[p], **self.phases[p]} for p in PHASES if p in self.phases],
                "elapsed": (self.finished or now) - self.started if self.started else 0.0, "eta": self.eta(),
                "error": self.error, "log": list(self.log)[-40:]}


class JobManager:
    """Runs submitted analyses one after another. ``command(job)`` gives the process to run (the worker)."""

    def __init__(self, command=None, on_finish=None):
        self.command = command or self.worker_command
        self.on_finish = on_finish
        self.jobs: list[Job] = []
        self.lock = threading.Lock()
        self._runner: threading.Thread | None = None

    @staticmethod
    def worker_command(job: Job) -> list[str]:
        cmd = [sys.executable, "-u", "-m", "tubetracker.app.worker", "--run", job.folder]
        if job.movie:
            cmd += ["--movie", job.movie]
        if job.flatfield:
            cmd.append("--flatfield")
        return cmd

    def submit(self, folder: str, movie: str | None, name: str, flatfield: bool = False) -> Job:
        with self.lock:
            for j in self.jobs:
                if j.folder == folder and j.state in ("queued", "running"):
                    return j  # already on its way
            job = Job(folder, movie, name, flatfield)
            self.jobs.append(job)
            if self._runner is None or not self._runner.is_alive():
                self._runner = threading.Thread(target=self._run_queue, name="analyses", daemon=True)
                self._runner.start()
        return job

    def _next(self) -> Job | None:
        with self.lock:
            return next((j for j in self.jobs if j.state == "queued"), None)

    def _run_queue(self) -> None:
        while True:
            job = self._next()
            if job is None:
                return
            self._run(job)

    def _run(self, job: Job) -> None:
        job.state, job.started, job.label = "running", time.time(), "Starting"
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        try:
            job.proc = subprocess.Popen(self.command(job), cwd=str(REPO), stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, bufsize=1, env=env,
                                        start_new_session=True)
        except OSError as exc:
            job.state, job.error, job.finished = "failed", f"could not start the analysis: {exc}", time.time()
            return
        try:
            path = worker_log_path(job.folder)
            path.parent.mkdir(parents=True, exist_ok=True)
            logfile = open(path, "a")
            logfile.write(f"\n=== analysis started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        except OSError:
            logfile = None
        for line in job.proc.stdout:
            line = line.rstrip("\n")
            if logfile:
                logfile.write(line + "\n")
                logfile.flush()
            if line.startswith("@@ "):
                try:
                    job.update(json.loads(line[3:]))
                except ValueError:
                    job.log.append(line)
            elif line.strip():
                job.log.append(line)
        code = job.proc.wait()
        if logfile:
            logfile.close()
        job.finished = time.time()
        if job.cancelled:
            job.state, job.label = "cancelled", "Stopped"
        elif code == 0 and job.phases.get("done"):
            job.state, job.label = "done", "Analysis complete"
        else:
            job.state = "failed"
            job.error = job.error or f"the analysis stopped (exit code {code}); see its log"
        if self.on_finish:
            try:
                self.on_finish(job)
            except Exception as exc:  # noqa: BLE001 - never let a callback kill the queue
                job.log.append(f"after the analysis: {exc}")

    def cancel(self, job_id: int) -> Job | None:
        with self.lock:
            job = next((j for j in self.jobs if j.id == int(job_id)), None)
        if job is None:
            return None
        if job.state == "queued":
            job.state, job.cancelled, job.finished = "cancelled", True, time.time()
        elif job.state == "running" and job.proc is not None:
            job.cancelled = True
            job.label = "Stopping"
            stop(job.proc)
        return job

    def status(self) -> dict:
        with self.lock:
            jobs = list(self.jobs)
        active = [j for j in jobs if j.state in ("queued", "running")]
        recent = [j for j in jobs if j.state not in ("queued", "running")][-5:]
        return {"active": [j.to_json() for j in active], "recent": [j.to_json() for j in recent],
                "busy": any(j.state == "running" for j in jobs)}

    def job_for(self, folder: str) -> Job | None:
        with self.lock:
            for j in reversed(self.jobs):
                if j.folder == folder:
                    return j
        return None

    def shutdown(self) -> None:
        for j in list(self.jobs):
            if j.state == "running" and j.proc is not None:
                j.cancelled = True
                stop(j.proc)


def stop(proc: subprocess.Popen, grace: float = 5.0) -> None:
    """Stop a worker and everything it started (its process group): politely, then for good."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    def later():
        try:
            proc.wait(grace)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    threading.Thread(target=later, daemon=True).start()


def worker_log_path(folder: str | Path) -> Path:
    return Path(folder) / "analysis_log.txt"
