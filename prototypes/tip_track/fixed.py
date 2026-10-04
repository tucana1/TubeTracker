"""The implemented onset rules (fixed settings, as in analyze.Params.tipdet_*) on all three movies, offline: hits vs
0.8.8 with paired bootstrap intervals, early/late counts, the m1 grains whose floods never start properly, and how
often the detector's onset fires on grains that never germinated and on census debris.

    python -m prototypes.tip_track.fixed
"""
from __future__ import annotations

import numpy as np

from .onsets import STUCK, TOL, Movie, boot, hits

LEVEL = dict(thr=0.3, W=3, frac=1.0, thr_lo=None, back=6, ang_tol=None)


def rule_frames(mv: Movie, g: str, rule: str, N: int = 10):
    fb = mv.base_frame(g)
    fd, _ = mv.det(g, LEVEL)
    if rule == "detector":
        return fd
    if fd is None:
        return fb
    if fb is None:
        return fd if rule == "later_or_missing" else None
    return fd if fb - fd > N * mv.fpb else fb


def main(log=print):
    for rule in ("detector", "later", "later_or_missing"):
        log(f"\n{rule} (thr {LEVEL['thr']}, {LEVEL['W']} bins{', N 10' if rule != 'detector' else ''})")
        for m in ("ld", "m2", "m1"):
            mv = Movie(m)
            new = {g: rule_frames(mv, g, rule) for g in mv.emerged}
            base = {g: mv.base_frame(g) for g in mv.emerged}
            hn, hb = hits(mv, new), hits(mv, base)
            d = np.array([int(hn[g]) - int(hb[g]) for g in mv.emerged])
            lo, hi = boot(d)
            early = lambda fr: sum(1 for g in mv.emerged if (e := mv.err(g, fr[g])) is not None and e < -TOL)
            late = lambda fr: sum(1 for g in mv.emerged if (e := mv.err(g, fr[g])) is not None and e > TOL)
            none = lambda fr: sum(1 for g in mv.emerged if fr[g] is None)
            new_early = [g for g in mv.emerged if (e := mv.err(g, new[g])) is not None and e < -TOL
                         and not ((eb := mv.err(g, base[g])) is not None and eb < -TOL)]
            never = sum(rule_frames(mv, g, rule) is not None for g in mv.never)
            deb = sum(mv.det(g, LEVEL)[0] is not None for g in mv.debris)
            stuck = {g: (None if new[g] is None else round(mv.err(g, new[g]) / mv.fpb, 1),
                         None if base[g] is None else round(mv.err(g, base[g]) / mv.fpb, 1))
                     for g in STUCK.get(m, []) if g in new}
            log(f"  {m}: {sum(hn.values())}/{len(hn)} vs 0.8.8 {sum(hb.values())} ({d.sum():+d}, 95% CI {lo:+.0f} to "
                f"{hi:+.0f}); early {early(new)} (0.8.8 {early(base)}), late {late(new)} ({late(base)}), none "
                f"{none(new)} ({none(base)}); newly early: {new_early}; never-germinated called: {never}/{len(mv.never)}; "
                f"debris with a detector onset: {deb}/{len(mv.debris)}; gained {[g for g in mv.emerged if hn[g] and not hb[g]]}"
                f" lost {[g for g in mv.emerged if hb[g] and not hn[g]]}" + (f"; stuck (err bins new, 0.8.8): {stuck}"
                                                                              if stuck else ""))


if __name__ == "__main__":
    main()
