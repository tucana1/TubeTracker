"""The one-command adaptation: steps in order, finished steps skipped, and a summary of what is used now."""

import pytest

pytest.importorskip("torch")

from prototypes.learned_evidence import adapt  # noqa: E402


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs/sparsetrack/ld").mkdir(parents=True)
    (tmp_path / "runs/sparsetrack/ld/meta.json").write_text("{}")
    (tmp_path / "benchmark/labels").mkdir(parents=True)
    (tmp_path / "benchmark/labels/ld_v1.json").write_text('{"labels": {}}')
    return tmp_path


def _steps(calls, adopt_decoder=True, adopt_model=False):
    def step(name, extra=None):
        def run(argv):
            calls.append((name, argv))
            work = adapt.Path(argv[argv.index("--work") + 1])
            work.mkdir(parents=True, exist_ok=True)
            (work / "report.txt").write_text(f"{name} report")
            if extra:
                (work / extra).write_text("{}")
        return run
    return {"dev": step("dev"), "calibrate": step("calibrate", "decoder.json" if adopt_decoder else None),
            "finetune": step("finetune", "unet_ft.pt" if adopt_model else None), "once": step("once")}


def test_steps_run_in_order_and_the_summary_says_what_is_used(repo):
    calls = []
    out = adapt.main([], steps=_steps(calls))
    assert [c[0] for c in calls] == ["dev", "calibrate", "finetune", "once"]
    assert "--model" not in calls[0][1]  # the full dev test trains on the dev field
    text = out.read_text()
    assert "dev report" in text and "calibrate report" in text and "finetune report" in text and "once report" in text
    assert "## 2. Decoder calibration: adopted" in text and "## 3. Fine-tuning: not adopted" in text
    assert "--decoder runs/learned_evidence/ld_cal/decoder.json" in text and "--prefix --heldout-once" in text
    assert "unet_v2_sample_field.pt" in text  # no model trained or tuned here: the shipped one is used


def test_finished_steps_are_skipped_and_redo_runs_them_again(repo):
    calls = []
    adapt.main([], steps=_steps(calls))
    calls.clear()
    adapt.main([], steps=_steps(calls))
    assert calls == []
    adapt.main(["--redo"], steps=_steps(calls, adopt_decoder=False, adopt_model=True))
    assert [c[0] for c in calls] == ["dev", "calibrate", "finetune", "once"]  # the dev test is scored again
    assert (repo / "runs/learned_evidence/ld_cal_old/decoder.json").exists()
    assert (repo / "runs/learned_evidence/ld_once_old/report.txt").exists()
    assert (repo / "runs/learned_evidence/ld/report_old.txt").exists()
    text = (repo / "runs/learned_evidence/SUMMARY.md").read_text()
    assert "Model: `runs/learned_evidence/ld_ft/unet_ft.pt`" in text and "--decoder" not in text


def test_steps_run_on_labels_since_changed_are_said_so(repo, capsys):
    adapt.main([], steps=_steps([]))
    (repo / "benchmark/labels/ld_v1.json").write_text('{"labels": {"g1": {}}}')  # more labels since
    calls = []
    out = adapt.main([], steps=_steps(calls))
    assert calls == [] and "run again with --redo" in capsys.readouterr().out
    assert "The labels have changed since dev and calibrate and finetune and once ran" in out.read_text()
    out = adapt.main(["--redo"], steps=_steps(calls))
    assert "have changed" not in out.read_text()


def test_quick_uses_the_shipped_model_and_finetune_can_be_skipped(repo):
    calls = []
    adapt.main(["--quick", "--skip-finetune"], steps=_steps(calls))
    assert [c[0] for c in calls] == ["dev", "calibrate", "once"]
    assert calls[0][1][calls[0][1].index("--model") + 1].endswith("unet_v2_sample_field.pt")


def test_movie_2_and_a_missing_cache_are_refused(repo):
    with pytest.raises(SystemExit, match="held-out"):
        adapt.main(["--labels", "benchmark/labels/m2_v1.json"], steps={})
    with pytest.raises(SystemExit, match="no prepared cache"):
        adapt.main(["--field", "runs/sparsetrack/nothing"], steps={})
    with pytest.raises(SystemExit, match="no labels"):
        adapt.main(["--labels", "benchmark/labels/nothing.json"], steps={})


def test_paths_in_the_summary_stay_inside_the_repository_through_links(repo, tmp_path_factory):
    elsewhere = tmp_path_factory.mktemp("main_checkout")
    (elsewhere / "learned_evidence").mkdir()
    (repo / "runs" / "learned_evidence").symlink_to(elsewhere / "learned_evidence")  # runs/ linked to the main checkout
    assert adapt._here(repo / "runs/learned_evidence/ld/unet.pt") == adapt.Path("runs/learned_evidence/ld/unet.pt")
    assert adapt._here(adapt.Path("runs/learned_evidence/ld/unet.pt")) == adapt.Path("runs/learned_evidence/ld/unet.pt")
    assert adapt._here(elsewhere / "x.pt") == elsewhere / "x.pt"  # outside the repository: left as it is


def test_a_failing_trace_once_step_does_not_keep_the_summary_from_being_written(repo):
    calls = []
    steps = _steps(calls)

    def broken(argv):
        calls.append(("once", argv))
        raise SystemExit("no FULL trace with a drawn path to anchor on")

    steps["once"] = broken
    out = adapt.main([], steps=steps)
    assert [c[0] for c in calls] == ["dev", "calibrate", "finetune", "once"]
    text = out.read_text()
    assert "step 4 failed" in text and "no FULL trace" in text
    assert not (repo / "runs/learned_evidence/ld_once/labels.sha1").exists()
    calls.clear()
    adapt.main([], steps=_steps(calls))  # tried again next time, and the others are not
    assert [c[0] for c in calls] == ["once"]


def test_trace_once_runs_again_when_the_model_in_use_changes(repo):
    import json
    calls = []
    adapt.main(["--skip-finetune"], steps=_steps(calls))
    model, decoder = adapt.in_use()
    (repo / "runs/learned_evidence/ld_once/used.json").write_text(
        json.dumps({"in_use_model": str(model), "in_use_decoder": str(decoder)}))
    calls.clear()
    adapt.main([], steps=_steps(calls, adopt_model=True))  # fine-tuning now runs and is adopted
    assert [c[0] for c in calls] == ["finetune", "once"]
    assert (repo / "runs/learned_evidence/ld_once/report_old.txt").exists()
