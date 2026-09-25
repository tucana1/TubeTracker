"""Adapt the learned pipeline to your dev movie in one go: the dev test, then calibration, then fine-tuning.

1. ``pipeline.py`` with the dev labels: trains the model on the dev movie's field (``--quick``: the shipped
   model instead) and scores the three runs on the labels.
2. ``calibrate.py``: fits the per-bin decoder's end offset on the same traces; kept only if its check adopts it.
3. ``finetune.py``: tunes the network on the traces, judged with the decoder it will be used with; kept only
   if its check adopts it.

Each step is skipped when its report is already there, so the command can be stopped and started again;
a step that ran on labels since changed is said so. ``--redo`` runs every step again on the current labels
(calibration's and fine-tuning's earlier outputs are moved aside to ``*_old``); the dev test keeps the model
it trained, the longest part, and is only scored again. ``SUMMARY.md`` then says what each step found, which
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
DEV, CAL, FT = ROOT / "ld", ROOT / "ld_cal", ROOT / "ld_ft"


def _here(path: Path) -> Path:
    """``path`` relative to the working directory (the repository) when it lies inside it."""
    try:
        return path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        return path


def in_use() -> tuple[Path, Path | None]:
    """The model and decoder settings the launcher picks, in its order."""
    from .finetune import SHIPPED
    model = next(p for p in (FT / "unet_ft.pt", DEV / "unet.pt", SHIPPED) if p.exists())
    decoder = CAL / "decoder.json"
    return _here(model), (decoder if decoder.exists() else None)


def _stale_note(stale) -> str:
    if not stale:
        return ""
    return (f"\n**The labels have changed since {' and '.join(stale)} ran: run this again with `--redo` for results "
            "on the current labels.**\n")


def _report(path: Path) -> str:
    return path.read_text().strip() if path.exists() else "(not run)"


def summary(labels: Path, seconds: float, stale: tuple = ()) -> str:
    model, decoder = in_use()
    m2 = (f".venv/bin/python -m prototypes.learned_evidence.pipeline --field runs/sparsetrack/m2 \\\n"
          f"    --labels benchmark/labels/m2_v1.json --model {model} \\\n"
          + (f"    --decoder {decoder} \\\n" if decoder else "")
          + "    --work runs/learned_evidence/m2 --heldout-once")
    cal = ("adopted: `" + str(CAL / "decoder.json") + "`" if (CAL / "decoder.json").exists()
           else "not adopted: the default end offset stays" if (CAL / "report.txt").exists() else "not run")
    ft = ("adopted: `" + str(FT / "unet_ft.pt") + "`" if (FT / "unet_ft.pt").exists()
          else "not adopted: the model from step 1 stays" if (FT / "report.txt").exists() else "not run")
    return f"""# Learned pipeline adapted to {labels.name}

Written {time.strftime("%Y-%m-%d %H:%M")} ({seconds / 60:.0f} min this run).
{_stale_note(stale)}
## 1. Dev test: three runs scored on your labels

```
{_report(DEV / "report.txt")}
```

## 2. Decoder calibration: {cal}

```
{_report(CAL / "report.txt")}
```

## 3. Fine-tuning: {ft}

```
{_report(FT / "report.txt")}
```

## What is used now

- Model: `{model}`
- Per-bin decoder: {"`" + str(decoder) + "`" if decoder else "default settings"}
- `Analyze_Movie_Learned.command` picks both up by itself.
- A fine-tuned model belongs to the imaging conditions of the movie it was tuned on.

## Movie 2, once, when its labels are in

This scores the model and decoder above on movie 2. Run it once. Changing either afterwards and scoring again
would turn movie 2 into a development set.

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
    ap.add_argument("--redo", action="store_true", help="run every step again on the current labels (calibration's "
                                                        "and fine-tuning's earlier outputs are moved aside to *_old); "
                                                        "the dev test keeps its trained model and is scored again")
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
        from . import calibrate, finetune, pipeline
        steps = {"dev": pipeline.main, "calibrate": calibrate.main, "finetune": finetune.main}
    from .finetune import SHIPPED
    started = time.time()
    if args.redo:
        for d in (CAL, FT):
            if d.exists():
                old = d.with_name(d.name + "_old")
                shutil.rmtree(old, ignore_errors=True)
                d.rename(old)
        if (DEV / "report.txt").exists():  # scored again; the model it trained stays and is not trained again
            os.replace(DEV / "report.txt", DEV / "report_old.txt")
    stamp = hashlib.sha1(labels.read_bytes()).hexdigest()
    stale = []
    plan = [("dev", DEV, ["--field", str(field), "--labels", str(labels), "--work", str(DEV)]
             + (["--model", str(SHIPPED)] if args.quick else [])),
            ("calibrate", CAL, ["--field", str(field), "--labels", str(labels), "--work", str(CAL)])]
    if not args.skip_finetune:
        plan.append(("finetune", FT, ["--field", str(field), "--labels", str(labels), "--work", str(FT)]))
    for name, work, cmd in plan:
        if (work / "report.txt").exists():
            seen = work / "labels.sha1"
            changed = seen.exists() and seen.read_text().strip() != stamp
            stale += [name] if changed else []
            print(f"== {name}: done before ({work / 'report.txt'}), skipped"
                  + ("; the labels have changed since" if changed else ""), flush=True)
            continue
        print(f"== {name}: {' '.join(cmd)}", flush=True)
        steps[name](cmd)
        work.mkdir(parents=True, exist_ok=True)
        (work / "labels.sha1").write_text(stamp + "\n")  # the labels this step's results are on
    ROOT.mkdir(parents=True, exist_ok=True)
    out = ROOT / "SUMMARY.md"
    out.write_text(summary(labels, time.time() - started, tuple(stale)))
    if stale:
        print(f"== the labels have changed since {' and '.join(stale)} ran: run again with --redo", flush=True)
    print(f"== summary: {out}")
    return out


if __name__ == "__main__":
    main()
