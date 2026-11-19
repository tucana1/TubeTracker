"""How much of "tip right, length off" is where the trace starts: the model's reading re-measured from the annotator's
exit (oracle exit), to the annotator's apex (oracle tip), or both (the route's shape alone), per FULL trace.

    python -m prototypes.four_movies.exit_oracle NAME CACHE PREDICTIONS LABELS

Lengths along the model's own route at that bin (as the app draws it, in the field): from the route point nearest the
start to the route point nearest the end, plus the two gaps. A hit is |length - traced| <= max(2 px, 10%)."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tubetracker.app import overlay  # noqa: E402
from tubetracker.app.model import RunData  # noqa: E402
from tubetracker.app.runfolder import RunFolder  # noqa: E402


def run_data(name: str, cache: Path, pred: Path) -> RunData:
    f = REPO / "runs/research/four_movies/rf" / name
    shutil.rmtree(f, ignore_errors=True)
    (f / "analysis").mkdir(parents=True)
    (f / "cache").symlink_to(cache.resolve())
    (f / "analysis" / "predictions.json").symlink_to(pred.resolve())
    return RunData(RunFolder(f))


def along(route: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Arc length along ``route`` between its points nearest ``a`` and ``b``, plus the gaps to them."""
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(route, axis=0).T))])
    dense = np.arange(0.0, s[-1] + 1e-9, 0.25)
    pts = np.stack([np.interp(dense, s, route[:, 0]), np.interp(dense, s, route[:, 1])], 1)
    ia, ib = (int(np.argmin(np.hypot(*(pts - q).T))) for q in (a, b))
    return float(abs(dense[ib] - dense[ia]) + np.hypot(*(pts[ia] - a)) + np.hypot(*(pts[ib] - b)))


def main(name, cache, pred, labels):
    data = RunData.__new__(RunData) if False else run_data(name, Path(cache), Path(pred))
    lab = json.loads(Path(labels).read_text())
    by = {g["id"]: g for g in json.loads(Path(pred).read_text())["grains"]}
    fpb = data.fpb
    rows = []
    for gid, gl in lab["grains"].items():
        if gl.get("excluded") or gid not in by or not gl.get("isolated", True):
            continue
        g = {**data.grain(gid), "human": []}
        res = by[gid]
        frames = np.asarray(res["length"]["frames"])
        for b, t in (lab["labels"][gid].get("traces") or {}).items():
            if t["state"] != "full" or t.get("contact"):
                continue
            b = int(b)
            H = np.asarray(t["path_xy_ref"], float)
            h = float(t["length_px"])
            i = int(np.argmin(np.abs(frames - (t.get("source_frame") or b * fpb + fpb // 2))))
            L = float(res["length"]["px"][i])
            tol = max(2.0, 0.1 * h)
            row = {"gid": gid, "bin": b, "h": h, "L": L, "hit": abs(L - h) <= tol, "young": h < 15}
            route, _ = overlay.route_at(g, b)
            tips = (res.get("tip") or {}).get("xy")
            if L > 0 and len(route) >= 2 and tips and tips[i]:
                d = np.asarray(overlay.drift_at(g, b))
                R = np.asarray(route, float) + d
                tip = np.asarray(tips[i], float) + d
                row["tip_err"] = float(np.hypot(*(tip - H[-1])))
                row["exit_err"] = float(np.hypot(*(R[0] - H[0])))
                gc = np.asarray(overlay.pos_at(g, b))
                row["exit_dr"] = float(np.hypot(*(R[0] - gc)) - np.hypot(*(H[0] - gc)))  # model exit radius - annotator's
                a_m, a_h = (np.arctan2(*(q - gc)[::-1]) for q in (R[0], H[0]))
                row["exit_da"] = float(np.degrees(np.angle(np.exp(1j * (a_m - a_h)))))
                # the model's own measure along its route (start to its tip), and the three oracles
                base = along(R, R[0], tip)
                off = L - base  # the reader's constant corrections (edge, tip offset, band beyond the tip, middle)
                row["oe"] = along(R, H[0], tip) + off * 0  # from the annotator's exit to the model's tip
                row["ot"] = along(R, R[0], H[-1]) + off     # the model's start to the annotator's apex
                row["oet"] = along(R, H[0], H[-1])          # the route's shape alone
                for k in ("oe", "ot", "oet"):
                    row[k + "_hit"] = abs(row[k] - h) <= tol
            rows.append(row)
    n = len(rows)
    have = [r for r in rows if "oe" in r]
    print(f"== {name}: {n} FULL traces, {sum(r['hit'] for r in rows)} hits; with a route and tip: {len(have)}")
    for label, sel in (("young (< 15 px)", lambda r: r["young"]), ("older", lambda r: not r["young"])):
        s = [r for r in have if sel(r)]
        if not s:
            continue
        tip_ok = [r for r in s if r["tip_err"] <= max(5.0, 0.1 * r["h"])]
        print(f"   {label}: {len(s)} traces, {sum(r['hit'] for r in s)} hits | tip within tolerance: {len(tip_ok)} "
              f"(of them {sum(r['hit'] for r in tip_ok)} length hits) | oracle exit {sum(r['oe_hit'] for r in s)}, "
              f"oracle tip {sum(r['ot_hit'] for r in s)}, both {sum(r['oet_hit'] for r in s)} | among tip-ok: oracle exit "
              f"{sum(r['oe_hit'] for r in tip_ok)}")
        print(f"      model exit vs annotator's: median distance {np.median([r['exit_err'] for r in s]):.1f} px; radius "
              f"difference median {np.median([r['exit_dr'] for r in s]):+.1f} (IQR {np.percentile([r['exit_dr'] for r in s], 25):+.1f}.."
              f"{np.percentile([r['exit_dr'] for r in s], 75):+.1f}); |angle| median {np.median(np.abs([r['exit_da'] for r in s])):.0f} deg, "
              f"> 20 deg in {sum(abs(r['exit_da']) > 20 for r in s)}; tip error median {np.median([r['tip_err'] for r in s]):.1f} px")
    return rows


if __name__ == "__main__":
    main(*sys.argv[1:5])
