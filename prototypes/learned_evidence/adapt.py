"""Adapt the learned pipeline to your dev movie in one go: the dev test, then calibration, then fine-tuning.

1. ``pipeline.py`` with the dev labels: trains the model on the dev movie's field (``--quick``: the shipped
   model instead) and scores its runs on the labels. The per-bin decoder is also scored as it read movies before
   27 Sep 2026, and the summary says whether your labels find the reading in use worse.
2. ``calibrate.py``: fits the per-bin decoder's end offset on the same traces; kept only if its check adopts it.
3. ``finetune.py``: tunes the network on the traces, judged with the decoder it will be used with; kept only
   if its check adopts it.
4. ``trace_once.py``: with the model and decoder now in use, how well one traced tube per grain gives the rest of
   your traces (the prefix decoder anchored on your latest trace), against the per-bin decoder.

Every step reads the movie one way, ``--reading fused`` (the default since 27 Sep 2026: the model's evidence
fused with the thick-tube network's, with tip-growth continuity) or ``--reading plain`` (the model's evidence
alone, without continuity, as before). The choice is kept in ``runs/learned_evidence/reading.json``: later runs,
``Analyze_Movie_Learned.command`` and the movie-2 command in ``SUMMARY.md`` follow it until another is given.

Each step is skipped when its report is already there, so the command can be stopped and started again;
a step that ran on labels since changed, or read the movie otherwise than the reading in use, is said so.
``--redo`` runs every step again on the current labels (calibration's and fine-tuning's earlier outputs are
moved aside to ``*_old``); the dev test keeps the model it trained, the longest part, and is only scored again. ``SUMMARY.md`` then says what each step found, which
model and decoder the launcher (``Analyze_Movie_Learned.command``) now uses, and the one command that
scores movie 2, once.

    python -m prototypes.learned_evidence.adapt --field runs/sparsetrack/ld --labels benchmark/labels/ld_v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import time
from pathlib import Path

ROOT = Path("runs/learned_evidence")
DEV, CAL, FT, ONCE = ROOT / "ld", ROOT / "ld_cal", ROOT / "ld_ft", ROOT / "ld_once"
READING = ROOT / "reading.json"  # the reading every step, the launcher and the movie-2 command use
FLAGS = {"fused": [], "plain": ["--no-thick-model", "--no-continuity"]}  # for pipeline, calibrate, finetune, trace-once


def _here(path: Path) -> Path:
    """``path`` relative to the working directory (the repository) when it lies inside it. Links are
    not followed: ``runs/`` in a worktree is a link to the main checkout's, and stays ``runs/...``."""
    try:
        return Path(os.path.abspath(path)).relative_to(os.path.abspath(Path.cwd()))
    except ValueError:
        return path


def in_use() -> tuple[Path, Path | None]:
    """The model and decoder settings the launcher picks, in its order."""
    from .finetune import SHIPPED
    model = next(p for p in (FT / "unet_ft.pt", DEV / "unet.pt", SHIPPED) if p.exists())
    decoder = CAL / "decoder.json"
    return _here(model), (decoder if decoder.exists() else None)


def reading_in_use() -> str:
    """The reading the last run chose ("fused" unless ``--reading plain`` was)."""
    import json
    try:
        return json.loads(READING.read_text()).get("reading", "fused")
    except (OSError, ValueError):
        return "fused"


def _want(reading: str) -> dict:
    from .fuse import THICK
    from .fuse import reading as rec
    return rec(THICK, True) if reading == "fused" else rec(None, False)


def _stale_note(stale, other=(), reading: str = "fused") -> str:
    from .fuse import describe
    notes = ([f"The labels have changed since {' and '.join(stale)} ran"] if stale else []) + (
        [f"{' and '.join(other)} read the movie otherwise than now ({describe(_want(reading))})"] if other else [])
    if not notes:
        return ""
    return (f"\n**{'. '.join(notes)}: run this again with `--redo` for results on the current labels and "
            "reading.**\n")


def _plain_verdict(reading: str) -> str:
    """What the dev test's labels say about the fused reading against the plain one (``pipeline.py``'s paired
    comparison). The fused reading stays unless the labels find it clearly worse: a 95% interval below zero for
    lengths or for onsets."""
    import json
    path = DEV / "scores.json"
    if reading != "fused" or not path.exists():
        return ""
    p = json.loads(path.read_text()).get("paired_perbin_plain")
    if not p:
        return ""
    txt = (f"lengths {p['length_diff']:+.0f} [{p['length_ci'][0]:+.0f}, {p['length_ci'][1]:+.0f}], onsets "
           f"{p['onset_diff']:+.0f} [{p['onset_ci'][0]:+.0f}, {p['onset_ci'][1]:+.0f}] over {p['grains']} grains")
    if p["length_ci"][1] < 0 or p["onset_ci"][1] < 0:
        return (f"\n**Your labels find the fused reading worse than the plain one ({txt}). Switch back with "
                "`--reading plain --redo`; the Adapt launcher offers it.**\n")
    if p["length_ci"][0] > 0 or p["onset_ci"][0] > 0:
        return f"\nOn your labels the fused reading does better than the plain one ({txt}), so it stays.\n"
    return f"\nOn your labels the fused reading is not clearly worse than the plain one ({txt}), so it stays.\n"


def _read_before(name: str, work: Path) -> dict | None:
    """How a finished step read the movie, as it recorded it (``fuse.reading``; records from before 27 Sep 2026
    carry nothing: the model's evidence alone, without continuity). None where the step's result does not depend
    on it or it left nothing to tell."""
    import json
    if name == "dev":
        path = work / "perbin" / "predictions.json"
        return (json.loads(path.read_text()).get("decoder") or {}) if path.exists() else None
    if name == "calibrate":
        path = next((p for p in (work / "decoder.json", work / "decoder_not_adopted.json") if p.exists()), None)
        return json.loads(path.read_text()) if path else None
    if name == "finetune":  # the check's verdict depends on it (no check: nothing to tell)
        path = work / "scores.json"
        return ((json.loads(path.read_text()).get("settings") or {}).get("reading") or {}) if path.exists() else None
    if name == "once":
        path = work / "used.json"
        return (json.loads(path.read_text()).get("reading") or {}) if path.exists() else None
    return None


def _read_otherwise(name: str, work: Path, want: dict) -> bool:
    """Whether a finished step read the movie otherwise than ``want`` (``fuse.reading``)."""
    from .fuse import same_reading
    before = _read_before(name, work)
    return before is not None and not same_reading(before, want)


def _report(path: Path) -> str:
    return path.read_text().strip() if path.exists() else "(not run)"


def _checked_other(work: Path, want: dict) -> bool:
    """Whether step 4 ran for another model or decoder than the ones in use now, or read the movie otherwise than
    ``want`` (``trace_once.py`` records them)."""
    used = work / "used.json"
    if not used.exists():
        return False
    import json
    rec = json.loads(used.read_text())
    model, decoder = in_use()
    return ((rec.get("in_use_model"), rec.get("in_use_decoder")) != (str(model), str(decoder))
            or _read_otherwise("once", work, want))


def summary(labels: Path, seconds: float, stale: tuple = (), failed: dict | None = None, other: tuple = (),
            reading: str = "fused") -> str:
    from .fuse import describe
    model, decoder = in_use()
    m2 = (f".venv/bin/python -m prototypes.learned_evidence.pipeline --field runs/sparsetrack/m2 \\\n"
          f"    --labels benchmark/labels/m2_v1.json --model {model} \\\n"
          + (f"    --decoder {decoder} \\\n" if decoder else "")
          + (f"    {' '.join(FLAGS[reading])} \\\n" if FLAGS[reading] else "")
          + "    --work runs/learned_evidence/m2 --prefix --heldout-once")
    cal = ("adopted: `" + str(CAL / "decoder.json") + "`" if (CAL / "decoder.json").exists()
           else "not adopted: the default end offset stays" if (CAL / "report.txt").exists() else "not run")
    ft = ("adopted: `" + str(FT / "unet_ft.pt") + "`" if (FT / "unet_ft.pt").exists()
          else "not adopted: the model from step 1 stays" if (FT / "report.txt").exists() else "not run")
    return f"""# Learned pipeline adapted to {labels.name}

Written {time.strftime("%Y-%m-%d %H:%M")} ({seconds / 60:.0f} min this run).
{_stale_note(stale, other, reading)}
## 1. Dev test: the runs scored on your labels

```
{_report(DEV / "report.txt")}
```
{_plain_verdict(reading)}
## 2. Decoder calibration: {cal}

```
{_report(CAL / "report.txt")}
```

## 3. Fine-tuning: {ft}

```
{_report(FT / "report.txt")}
```

## 4. Trace once: one traced tube per grain, the rest decoded

```
{(failed or {}).get("once") or _report(ONCE / "report.txt")}
```

## What is used now

- Model: `{model}`
- Per-bin decoder: {"`" + str(decoder) + "`" if decoder else "default settings"}
- Reading: {reading} ({describe(_want(reading))}), kept in `{READING}`
- `Analyze_Movie_Learned.command` picks all three up by itself.
- A fine-tuned model belongs to the imaging conditions of the movie it was tuned on.

## Movie 2, once, when its labels are in

This scores the model, decoder and reading above on movie 2. Run it once. Changing any of them afterwards and
scoring again would turn movie 2 into a development set. The per-bin decoder is the one this summary recommends; the prefix
decoder is scored alongside it in the same run, to tell whether its synthetic lead carries over to a real movie,
not to pick between the two afterwards.

```bash
{m2}
```
"""


def main(argv=None, steps=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--field", default="runs/sparsetrack/ld", help="the dev movie's prepared cache")
    ap.add_argument("--labels", default="benchmark/labels/ld_v1.json", help="the dev movie's labels")
    ap.add_argument("--quick", action="store_true",
                    help="use the shipped model instead of training on your field (minutes, not 1.5 hours)")
    ap.add_argument("--skip-finetune", action="store_true")
    ap.add_argument("--skip-trace-once", action="store_true", help="leave out step 4")
    ap.add_argument("--redo", action="store_true", help="run every step again on the current labels (calibration's "
                                                        "and fine-tuning's earlier outputs are moved aside to *_old); "
                                                        "the dev test keeps its trained model and is scored again")
    ap.add_argument("--reading", choices=tuple(FLAGS), default=None,
                    help="how every step reads the movie: fused (the default since 27 Sep 2026: the model's evidence "
                         "fused with the thick-tube network's, with tip-growth continuity) or plain (the model's "
                         "evidence alone, without continuity); kept for later runs and the launcher (default: the "
                         "one kept, else fused)")
    args = ap.parse_args(argv)
    field, labels = Path(args.field), Path(args.labels)
    if "m2" in labels.name:
        raise SystemExit("movie 2 is the held-out benchmark: adapt to the dev movie, then score movie 2 once")
    if not (field / "meta.json").exists():
        raise SystemExit(f"no prepared cache at {field}: prepare the dev movie first (Label_Sparse_Benchmark.command "
                         "does, or `python -m sparsetrack prepare MOVIE --out runs/sparsetrack/ld`)")
    if not labels.exists():
        raise SystemExit(f"no labels at {labels}")
    if steps is None:
        from . import calibrate, finetune, pipeline, trace_once
        steps = {"dev": pipeline.main, "calibrate": calibrate.main, "finetune": finetune.main, "once": trace_once.main}
    from .finetune import SHIPPED
    started = time.time()
    import json
    reading = args.reading or reading_in_use()
    want, flags = _want(reading), FLAGS[reading]
    ROOT.mkdir(parents=True, exist_ok=True)
    READING.write_text(json.dumps({"reading": reading, **want}))  # first: a stopped run leaves the choice in force
    if args.redo:
        for d in (CAL, FT, ONCE):
            if d.exists():
                old = d.with_name(d.name + "_old")
                shutil.rmtree(old, ignore_errors=True)
                d.rename(old)
        if (DEV / "report.txt").exists():  # scored again; the model it trained stays and is not trained again
            os.replace(DEV / "report.txt", DEV / "report_old.txt")
    stamp = hashlib.sha1(labels.read_bytes()).hexdigest()
    stale, other = [], []
    common = ["--field", str(field), "--labels", str(labels)]
    plan = [("dev", DEV, common + ["--work", str(DEV)] + (["--model", str(SHIPPED)] if args.quick else []) + flags),
            ("calibrate", CAL, common + ["--work", str(CAL)] + flags)]
    if not args.skip_finetune:
        plan.append(("finetune", FT, common + ["--work", str(FT)] + flags))
    if not args.skip_trace_once:  # with the model and decoder the steps before leave in use
        plan.append(("once", ONCE, common + ["--work", str(ONCE)] + flags))
    failed = {}

    def flag(name: str, work: Path) -> None:  # a finished step on other labels or another reading
        seen = work / "labels.sha1"
        changed = seen.exists() and seen.read_text().strip() != stamp
        otherwise = _read_otherwise(name, work, want)
        stale.extend([name] if changed else [])
        other.extend([name] if otherwise else [])
        print(f"== {name}: done before ({work / 'report.txt'}), skipped"
              + ("; the labels have changed since" if changed else "")
              + ("; it read the movie otherwise than now" if otherwise else ""), flush=True)

    for name, work, cmd in plan:
        if name == "once" and (work / "report.txt").exists() and _checked_other(work, want):
            if stale or other:  # the steps before run again first (--redo): running this now would be wasted
                print("== once: done before for another model, decoder or reading, skipped until those are redone",
                      flush=True)
                other.append(name)
                continue
            os.replace(work / "report.txt", work / "report_old.txt")
            print("== once: the model, decoder or reading in use has changed since it ran: running it again",
                  flush=True)
        if (work / "report.txt").exists():
            flag(name, work)
            continue
        print(f"== {name}: {' '.join(cmd)}", flush=True)
        try:
            steps[name](cmd)
        except (Exception, SystemExit) as e:  # a check that fails must not keep the summary from being written
            if name != "once":
                raise
            failed[name] = f"step 4 failed, and runs again next time: {type(e).__name__}: {e}"
            print(f"== {name}: {failed[name]}", flush=True)
            continue
        work.mkdir(parents=True, exist_ok=True)
        (work / "labels.sha1").write_text(stamp + "\n")  # the labels this step's results are on
    if args.skip_finetune and (FT / "report.txt").exists():  # skipped, but a model it adopted is still in use
        flag("finetune", FT)
    out = ROOT / "SUMMARY.md"
    out.write_text(summary(labels, time.time() - started, tuple(stale), failed, tuple(other), reading))
    if stale:
        print(f"== the labels have changed since {' and '.join(stale)} ran: run again with --redo", flush=True)
    if other:
        print(f"== {' and '.join(other)} read the movie otherwise than now: run again with --redo", flush=True)
    print(f"== summary: {out}")
    return out


if __name__ == "__main__":
    main()
