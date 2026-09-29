"""The double-click launchers' own logic (zsh), with a stand-in for the Python steps that records how it is called:
Adapt offers to switch back to the plain reading when the summary says the labels find the fused one worse, and to
redo every step when labels or the reading changed; Analyze follows the reading Adapt kept."""

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
HERE = Path("prototypes/learned_evidence/launchers")  # where the launchers live, relative to the repository
pytestmark = pytest.mark.skipif(shutil.which("zsh") is None, reason="the launchers are zsh scripts")

FAKE_PYTHON = """#!/bin/bash
echo "$*" >> "$CALLS"
if [ "$1" = "-m" ]; then
  if [ "$2" = "prototypes.learned_evidence.adapt" ]; then
    mkdir -p runs/learned_evidence
    if [ "$(grep -c 'learned_evidence.adapt' "$CALLS")" = 1 ]; then
      printf '%s\\n' "$FIRST_SUMMARY" > runs/learned_evidence/SUMMARY.md
    else
      echo "all current" > runs/learned_evidence/SUMMARY.md
    fi
  fi
  exit 0
fi
exec "$REAL_PYTHON" "$@"
"""


def _exe(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.fixture
def lab(tmp_path):
    (tmp_path / HERE).mkdir(parents=True)
    for name in ("Adapt_Learned_To_Dev_Movie.command", "Analyze_Movie_Learned.command"):
        shutil.copy(REPO / HERE / name, tmp_path / HERE / name)
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    _exe(tmp_path / ".venv" / "bin" / "python", FAKE_PYTHON)
    (tmp_path / "bin").mkdir()
    _exe(tmp_path / "bin" / "open", "#!/bin/sh\nexit 0\n")
    _exe(tmp_path / "bin" / "osascript", "#!/bin/sh\necho /Users/lab/movies/pollen_day3.avi\n")
    (tmp_path / "runs" / "sparsetrack" / "ld").mkdir(parents=True)
    (tmp_path / "runs" / "sparsetrack" / "ld" / "meta.json").write_text("{}")
    cache = tmp_path / "runs" / "sparsetrack" / "pollen_day3" / "cache"
    cache.mkdir(parents=True)
    (cache / "meta.json").write_text("{}")
    (cache / "grains.json").write_text("{}")
    (tmp_path / "benchmark" / "labels").mkdir(parents=True)
    (tmp_path / "benchmark" / "labels" / "ld_v1.json").write_text("{}")
    return tmp_path


def _run(lab: Path, script: str, answer: str = "", first_summary: str = "") -> list[str]:
    import sys
    env = {**os.environ, "PATH": f"{lab / 'bin'}:{os.environ['PATH']}", "CALLS": str(lab / "calls.txt"),
           "FIRST_SUMMARY": first_summary, "REAL_PYTHON": sys.executable}
    subprocess.run(["zsh", str(HERE / script)], cwd=lab, input=answer, text=True, env=env, check=True, capture_output=True,
                   timeout=120)
    return [c for c in (lab / "calls.txt").read_text().splitlines() if c.startswith("-m ")]


def test_adapt_offers_the_switch_back_when_the_labels_find_the_fused_reading_worse(lab):
    summary = "**Your labels find the fused reading worse than the plain one (lengths -9 [-16, -3]).**"
    calls = _run(lab, "Adapt_Learned_To_Dev_Movie.command", "y\n", summary)
    assert len(calls) == 2 and "--reading" not in calls[0]
    assert calls[1].endswith("--reading plain --redo")


def test_adapt_offers_a_redo_when_a_step_read_the_movie_otherwise(lab):
    summary = "**calibrate read the movie otherwise than now (fused evidence with continuity): run this again**"
    calls = _run(lab, "Adapt_Learned_To_Dev_Movie.command", "y\n", summary)
    assert len(calls) == 2 and calls[1].endswith("--redo") and "--reading" not in calls[1]


def test_adapt_asks_nothing_when_the_fused_reading_is_not_worse(lab):
    summary = "On your labels the fused reading is not clearly worse than the plain one (lengths +2 [-4, +8]), so it stays."
    assert len(_run(lab, "Adapt_Learned_To_Dev_Movie.command", "y\n", summary)) == 1


def test_analyze_follows_the_reading_adapt_kept(lab):
    calls = _run(lab, "Analyze_Movie_Learned.command")
    assert "prototypes.learned_evidence.pipeline" in calls[0] and "--no-thick-model" not in calls[0]
    (lab / "calls.txt").unlink()
    (lab / "runs" / "learned_evidence").mkdir(parents=True)
    (lab / "runs" / "learned_evidence" / "reading.json").write_text(
        '{"reading": "plain", "thick_model": null, "continuity": null}')
    calls = _run(lab, "Analyze_Movie_Learned.command")
    assert calls[0].endswith("--no-thick-model --no-continuity")
