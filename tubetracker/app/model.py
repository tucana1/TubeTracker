"""An analysis as the app shows it: every grain with its tube at every time, the movie's events, the check list.

Read from a run folder: SparseTrack's predictions (``analysis/predictions.json``), the cache's grain census and
metadata, the movie's setup (units) and, once checking has started, the review labels file
(``review/review_labels.json``: the labelling tool's format, pre-filled with the model's answers). Where a person
answered, the display follows them: their onset, and the model's growth curve pinned to the lengths they checked
(``sparsetrack.review.reviewed_series``, as in the reviewed exports).

Time is in bins: bin ``b`` averages source frames ``[b * fpb, (b + 1) * fpb)`` and stands at its centre frame
``b * fpb + fpb // 2``. SparseTrack's per-grain series start at the movie's reference bin; here every series is
by bin from bin 0. Coordinates are the cache's reference coordinates; a grain the analysis followed as it moved is
at its census place plus its drift, and its tube route is in its own frame (``overlay``).

Texts are short phrases (``text``) with a sentence for details on demand (``detail``).
"""

from __future__ import annotations

import json
import math

import numpy as np

from sparsetrack import report
from sparsetrack.bench.server import trace_bins
from sparsetrack.review import changed_trace, human_burst_bin, lengths_by_bin, reviewed_series, trace_confidence

from . import overlay
from .overlay import EMERGED
from .runfolder import RunFolder
from .units import Units

UNSURE = 0.5          # the model is less sure than this of a tube's length: worth a look
STALL_BINS = 15       # a tube that has not grown for this many bins before the end (or its loss) has stopped
STALL_MIN_PX = 8.0    # ...if it is at least this long
GROWTH_EPS = 0.25     # px per bin: less than this is standing still (as review.trace_confidence counts it)
# the order a grain's reasons are listed in (the first one sets the time the check list shows it at)
REASON_ORDER = ("lost", "no_grain", "unsure", "focus", "onset", "contact", "unfollowed", "coverage", "path", "short")


def by_bin(values: list, res: dict, fpb: int, n_bins: int) -> list:
    """A per-reading series of ``res`` (aligned with its length frames) by bin from 0: the first value before the
    reference bin, the last after the series end."""
    frames = (res.get("length") or {}).get("frames") or []
    if not values or not frames:
        return []
    start = int(frames[0]) // fpb
    out = [values[0]] * start + list(values[: max(n_bins - start, 0)])
    return out + [out[-1]] * (n_bins - len(out))


def _r(v, nd=1):
    return round(float(v), nd)


def stall_bin(L, last: int) -> int | None:
    """The bin a tube stopped growing at, if it then stood still for ``STALL_BINS`` bins or more before ``last``
    (its last observed bin): a real stop, a burst tip, or the reader losing the tube."""
    L = np.asarray(L[: last + 1], float)
    if len(L) < 2 or L[-1] < STALL_MIN_PX:
        return None
    grew = np.flatnonzero(np.diff(L) > GROWTH_EPS)
    if not len(grew):
        return None
    stop = int(grew[-1]) + 1
    return stop if last - stop >= STALL_BINS else None


# a model flag as a short phrase (None: not shown as a phrase) and in a sentence
PHRASES = {
    "reader": "read from tube maps", "shared_change_split": "shares change region", "drift_rejected": "not followed",
    "onset_at_focus_change": "onset at focus change", "onset_moved_to_front": "onset moved back",
    "onset_from_front": "onset from growth front", "contact_censored": "reaches another grain",
    "no_grain": "no grain rim", "front_too_short": "too short for a tube", "degenerate_path": "no usable route",
    "tube_map_without_onset": "change without onset", "second_attached_component": "second change region",
    "path_by_growth": "route chosen by growth", "flood_found_no_tube": "no tube in maps", "route_given": "route given",
    "grain_lost_after": "lost partway", "settled_from_bin": "settling at start",
    "touches": "touches {}", "rotates": "turns {}", "tip_continued": "tip continued {}", "onset_lookback": "onset back {} bins",
}


def flag_phrase(flag: str) -> str:
    name, _, arg = flag.partition(":")
    if name == "path_coverage":
        return f"route covers {100 * float(arg or 0):.0f}%"
    if name == "rotates":
        arg = arg.replace("deg", "°")
    return PHRASES.get(name, flag.replace("_", " ")).format(arg)


FLAG_DETAIL = {
    "reader": "Its surroundings are crowded or noisy, so it was read from the tube-probability maps.",
    "shared_change_split": "It shares a region of change with a neighbour; the region was split between them.",
    "drift_rejected": "The grain moved in a way that could not be followed; it was read at its first place.",
    "onset_at_focus_change": "The movie's focus changed then: the tube may have emerged earlier, unseen.",
    "onset_moved_to_front": "The onset was moved back to where the growth began.",
    "onset_from_front": "No clear stub at the exit: the onset comes from the growth front.",
    "contact_censored": "The tube reaches another grain; its length stops there.",
    "no_grain": "No grain rim in the early frames: probably debris.",
    "front_too_short": "What grew never got longer than 8 px.",
    "degenerate_path": "No usable route for a tube could be traced.",
    "tube_map_without_onset": "Change next to the grain, but no onset.",
    "second_attached_component": "A second region of change touches the grain.",
    "path_by_growth": "Several routes were possible; the one that grew most like a tube was kept.",
    "touches": "Its tube region touches a neighbour's: the length may include part of the other tube.",
    "rotates": "The grain and its tube turn during the movie.",
    "grain_lost_after": "It burst, drifted out of view or was swept off; its numbers are held from then.",
    "settled_from_bin": "The grain was still settling at the start and is read from later on.",
    "tip_continued": "The tube turned back along its grain; the reading followed it on.",
    "onset_lookback": "The onset was walked back to when the tube first showed at the exit.",
    "path_coverage": "The route explains little of the change round the grain: the tube may curl or be shared.",
}


class RunData:
    """One analysed movie, read from its run folder, with the review labels document if there is one."""

    def __init__(self, folder: RunFolder, doc: dict | None = None, model_doc: dict | None = None):
        self.folder = folder
        self.meta = folder.cache_meta()
        self.census = {g["id"]: g for g in json.loads((folder.cache / "grains.json").read_text())["grains"]}
        self.pred = json.loads(folder.predictions.read_text())
        self.fpb = int(self.pred.get("frames_per_bin") or self.meta["frames_per_bin"])
        self.n_bins = int(self.meta["n_bins"])
        self.ref_start = int(self.meta.get("ref_start", 0))
        movie = self.meta.get("movie") or {}
        self.width, self.height = int(movie.get("width", 0)), int(movie.get("height", 0))
        self.model = {g["id"]: g for g in self.pred.get("grains", [])}
        self.order = [g["id"] for g in self.pred.get("grains", [])]
        self.doc = doc
        self.model_doc = model_doc
        self.set_units(folder.load_setup())

    # ---- time and units --------------------------------------------------------------------
    def frame(self, b: int) -> int:
        return int(b) * self.fpb + self.fpb // 2

    def when(self, frame: float | None) -> str:
        """A source frame as a short time: ``143 min`` when the movie's duration is known, else ``frame 17250``."""
        if frame is None:
            return "-"
        m = self.units.minutes(frame)
        return f"{m:.0f} min" if m is not None else f"frame {int(frame)}"

    def when_range(self, after: float | None, by: float | None) -> str:
        """An onset interval: ``119-121 min`` (or ``frame 5250-5550``; ``by 121 min`` when only its end is known)."""
        if by is None:
            return "-"
        if after is None:
            return f"by {self.when(by)}"
        a, b = self.units.minutes(after), self.units.minutes(by)
        return f"{a:.0f}-{b:.0f} min" if a is not None else f"frame {int(after)}-{int(by)}"

    def length_words(self, px: float | None) -> str:
        if px is None:
            return "-"
        um = self.units.um(px)
        return f"{um:.1f} µm" if um is not None else f"{px:.1f} px"

    def set_units(self, setup: dict) -> None:
        self.setup = setup
        self.units = Units.from_setup(setup, self.folder.n_frames())
        self._static = {gid: self._model_part(res) for gid, res in self.model.items()}

    # ---- the model's readings (fixed) -------------------------------------------------------
    def _model_part(self, res: dict) -> dict:
        fpb, nb = self.fpb, self.n_bins
        L = lengths_by_bin(res, fpb, nb)
        rot = by_bin(res.get("rotation_deg") or [], res, fpb, nb)
        drift = by_bin((res.get("drift") or {}).get("xy") or [], res, fpb, nb)
        status = res.get("status")
        fv = None
        if status == "emerged_within" and res.get("onset_frame") is not None:
            fv = int(res["onset_frame"]) // fpb
        elif status == "emerged_at_start":
            fv = int(((res.get("length") or {}).get("frames") or [0])[0]) // fpb
        conf = unsure = None
        if status in EMERGED and L.any():
            plan = trace_bins(fv if status == "emerged_within" else 0, nb) or [nb - 1]
            conf, unsure = min((trace_confidence(L, b), b) for b in plan)
        lost = lost_why = None
        if res.get("observed_until_frame") is not None and any(f.startswith("grain_lost") for f in res.get("flags", [])):
            lost = min(int(res["observed_until_frame"]) // fpb + 1, nb - 1)
            lost_why = res.get("lost_reason") or "gap"
        pivot = None
        if res.get("path"):
            exit_pivot = (self.pred.get("params") or {}).get("rot_pivot") == "exit" and res.get("exit_xy")
            pivot = res["exit_xy"] if exit_pivot else [res["x"], res["y"]]
        flags = list(res.get("flags", []))
        cov = res.get("path_coverage")
        if cov is not None and cov < report.COVERAGE_MIN:
            flags.append(f"path_coverage:{cov:.2f}")
        return {"L": L, "rot": rot if any(abs(a) > 0.05 for a in rot) else None, "drift": drift or None,
                "status": status, "fv": fv, "conf": conf, "unsure": unsure, "lost": lost, "lost_why": lost_why,
                "pivot": pivot, "flags": flags, "path": res.get("path") or [],
                "rate": report.growth_rate([self.frame(b) for b in range(nb)], L) if status in EMERGED else None,
                "census": self.census.get(res["id"], {})}

    # ---- the review ---------------------------------------------------------------------------
    def label(self, gid: str) -> dict:
        return ((self.doc or {}).get("labels") or {}).get(gid) or {}

    def excluded(self, gid: str) -> str | None:
        g = ((self.doc or {}).get("grains") or {}).get(gid) or {}
        return (g.get("exclude_reason") or "other") if g.get("excluded") else None

    @staticmethod
    def human(rec: dict | None) -> bool:
        return bool(rec) and rec.get("review_origin", "human") == "human"

    def human_traces(self, gid: str) -> list[dict]:
        out = []
        for t in (self.label(gid).get("traces") or {}).values():
            if not self.human(t):
                continue
            pts = t.get("path_xy_view") or []
            rec = {"bin": int(t["bin"]), "state": t["state"], "L": _r(t.get("length_px") or 0.0, 2)}
            if t["state"] in ("full", "partial") and len(pts) >= 2:
                rec["pts"] = [[_r(x, 2), _r(y, 2)] for x, y in pts]
            out.append(rec)
        return sorted(out, key=lambda t: t["bin"])

    def review_state(self, gid: str, model_L) -> dict:
        """What a person has done with this grain: onset and lengths checked or changed, or excluded."""
        lab = self.label(gid)
        excl = self.excluded(gid)
        on = lab.get("onset") or {}
        mlab = ((self.model_doc or {}).get("labels") or {}).get(gid) or {}
        mon = mlab.get("onset") or {}
        onset = "model"
        if on and self.human(on):
            same = (on.get("verdict"), on.get("first_visible_bin")) == (mon.get("verdict"), mon.get("first_visible_bin"))
            onset = "checked" if same or not mon else "changed"  # without the model's proposal: no change to tell
        traces = lab.get("traces") or {}
        n_human = changed = 0
        for b, t in traces.items():
            if not self.human(t):
                continue
            n_human += 1
            model_t = (mlab.get("traces") or {}).get(str(b))
            if model_t is None:  # a time the model was not asked at: against its own reading there
                Lm = float(model_L[int(b)]) if int(b) < len(model_L) else 0.0
                model_t = {"state": "full" if Lm >= 2 else "no_tube", "length_px": Lm}
            changed += bool(changed_trace(t, model_t))
        open_ = sum(1 for t in traces.values() if not self.human(t))
        if excl:
            state = "excluded"
        elif onset == "changed" or changed:
            state = "corrected"
        elif onset == "checked" and not open_:
            state = "checked"
        elif onset != "model" or n_human:
            state = "partly checked"
        else:
            state = "model"
        return {"state": state, "onset": onset, "traces_checked": n_human, "traces_changed": changed,
                "traces_open": open_, "excluded": excl}

    # ---- one grain as the app shows it -----------------------------------------------------------
    def grain(self, gid: str) -> dict:
        res = self.model[gid]
        s = self._static[gid]
        nb = self.n_bins
        c = s["census"]
        lab = self.label(gid)
        on = lab.get("onset") or {}
        review = self.review_state(gid, s["L"])
        status, fv = s["status"], s["fv"]
        after = by = None
        if status == "emerged_within" and res.get("onset_interval"):
            after, by = res["onset_interval"]
        L = s["L"]
        lost, lost_why = s["lost"], s["lost_why"]
        human = self.human_traces(gid)
        if on and (self.human(on) or human):  # a person answered: the display follows the review
            status = on.get("verdict")
            fv = on.get("first_visible_bin") if status in EMERGED else None
            if status == "emerged_at_start":
                fv = 0
            after, by = on.get("last_absent_frame"), on.get("first_visible_frame")
            rv = reviewed_series(self.doc, gid, res)
            if rv is None:
                L = np.zeros(nb)
            else:
                L = np.asarray(rv["px"], float)
                L = np.r_[L, np.full(nb - len(L), L[-1] if len(L) else 0.0)]  # held after a burst
            burst = human_burst_bin(lab)
            if burst is not None:
                lost, lost_why = burst, "burst"
        last = lost - 1 if lost is not None else nb - 1
        stall = stall_bin(L, last) if status in EMERGED else None
        final = float(L[max(last, 0)]) if status in EMERGED else 0.0
        rate = s["rate"] if L is s["L"] else (report.growth_rate([self.frame(b) for b in range(nb)], L)
                                             if status in EMERGED else None)
        rec = {
            "id": gid, "x": _r(res["x"], 2), "y": _r(res["y"], 2), "r": _r(res.get("r") or c.get("r") or 12.0, 2),
            "isolated": bool(c.get("isolated", True)), "edge": bool(c.get("border")), "clump": int(c.get("clump_size") or 1),
            "status": status, "model_status": s["status"], "onset": fv, "model_onset": s["fv"],
            "onset_after": after, "onset_by": by,
            "L": [_r(v) for v in L], "path": s["path"], "pivot": s["pivot"], "rot": s["rot"], "drift": s["drift"],
            "human": human, "lost": lost, "lost_why": lost_why, "stall": stall,
            "conf": None if s["conf"] is None else _r(s["conf"], 3), "unsure": s["unsure"],
            "final": _r(final, 2), "rate": rate, "flags": s["flags"], "review": review, "excluded": review["excluded"],
        }
        if L is not s["L"]:
            rec["Lm"] = [_r(v) for v in s["L"]]
        rec["check"] = self.reasons(rec, s)
        rec["done"] = review["state"] != "model"
        return rec

    def reasons(self, rec: dict, s: dict) -> list[dict]:
        """Why a person should look at this grain, most important first: a short phrase, a sentence of detail, and
        the time to look at."""
        flags = s["flags"]
        last = self.n_bins - 2
        onset_bin = rec["onset"] if rec["onset"] is not None else last
        out = []

        def add(code, text, detail, b):
            out.append({"code": code, "text": text, "detail": detail, "bin": int(max(0, min(b, self.n_bins - 1)))})

        has = lambda *names: [f for f in flags if f.startswith(names)]
        if s["lost"] is not None:
            add("lost", f"lost at {self.when(self.frame(s['lost']))}", FLAG_DETAIL["grain_lost_after"], s["lost"] - 1)
        if has("no_grain"):
            add("no_grain", "no grain rim", FLAG_DETAIL["no_grain"], self.ref_start)
        if s["conf"] is not None and s["conf"] < UNSURE:
            add("unsure", "unsure length", f"The model's least sure reading of this tube ({100 * s['conf']:.0f}% "
                                           f"confidence), at {self.when(self.frame(s['unsure']))}.", s["unsure"])
        if has("onset_at_focus_change"):
            add("focus", "onset at focus change", FLAG_DETAIL["onset_at_focus_change"], onset_bin)
        for f in has("onset_moved_to_front", "onset_from_front", "settled_from_bin"):
            add("onset", flag_phrase(f), FLAG_DETAIL[f.partition(":")[0]], onset_bin)
        for f in has("touches:", "shared_change_split", "contact_censored"):
            add("contact", flag_phrase(f), FLAG_DETAIL[f.partition(":")[0]], last)
        if has("drift_rejected"):
            add("unfollowed", "not followed", FLAG_DETAIL["drift_rejected"], last)
        for f in has("path_coverage"):
            add("coverage", flag_phrase(f), FLAG_DETAIL["path_coverage"], last)
        for f in has("degenerate_path", "tube_map_without_onset"):
            add("path", flag_phrase(f), FLAG_DETAIL[f], last)
        if has("front_too_short"):
            add("short", flag_phrase("front_too_short"), FLAG_DETAIL["front_too_short"], last)
        seen, uniq = set(), []
        for r in sorted(out, key=lambda r: REASON_ORDER.index(r["code"])):
            if r["text"] not in seen:
                seen.add(r["text"])
                uniq.append(r)
        return uniq

    # ---- the whole movie -------------------------------------------------------------------------------
    def grains(self) -> list[dict]:
        return [self.grain(gid) for gid in self.order]

    def population(self, grains: list[dict]) -> dict:
        """The germination curve of the isolated grains as shown (reviewed onsets where a person answered):
        Turnbull's estimate for interval-censored onsets, by bin, with T50 (the time by which half had surely
        germinated) and the share germinated by the end."""
        nb, fpb = self.n_bins, self.fpb
        first, end = self.frame(self.ref_start), self.frame(nb - 1)
        iv, counts = [], {"germinated": 0, "not_germinated": 0, "lost_before": 0}
        for g in grains:
            if not g["isolated"] or g["excluded"] or g["status"] not in (*EMERGED, "no_emergence_by_end"):
                continue
            if g["status"] == "emerged_within" and g["onset_by"] is not None:
                by = float(g["onset_by"])
                iv.append((float(g["onset_after"]) if g["onset_after"] is not None else by - fpb, by))
                counts["germinated"] += 1
            elif g["status"] in EMERGED:
                iv.append((-math.inf, float(first)))
                counts["germinated"] += 1
            else:
                until = self.frame(g["lost"] - 1) if g["lost"] is not None else end
                iv.append((float(until), math.inf))
                counts["lost_before" if g["lost"] is not None else "not_germinated"] += 1
        masses = report.turnbull(iv) if iv else []
        frames = np.array([self.frame(b) for b in range(nb)], float)
        certain = np.zeros(nb)
        possible = np.zeros(nb)
        for q, p, m in masses:
            certain += m * (p <= frames)
            possible += m * (q < frames)
        cum, t50 = 0.0, None
        for q, p, m in masses:
            cum += m
            if cum >= 0.5 - 1e-9 and math.isfinite(p):
                t50 = float(p)
                break
        share = float(sum(m for q, p, m in masses if math.isfinite(p))) if masses else None
        return {"n": len(iv), "counts": counts, "certain": certain.round(4).tolist(),
                "possible": possible.round(4).tolist(), "t50_frame": t50,
                "t50_bin": None if t50 is None else int(t50) // fpb, "germinated_share": share}

    def events(self, grains: list[dict], population: dict) -> list[dict]:
        """What happened when (short phrases): germinations, grains lost partway, tubes that stopped growing, the
        readings the model is least sure of, focus changes, T50."""
        out = []
        for g in grains:
            if g["excluded"]:
                continue
            gid = g["id"]
            if g["status"] == "emerged_within" and g["onset"] is not None:
                out.append({"kind": "germination", "bin": g["onset"], "gid": gid, "text": f"{gid} germinates"})
            if g["lost"] is not None:
                out.append({"kind": "lost", "bin": g["lost"], "gid": gid,
                            "text": f"{gid} {'burst' if g['lost_why'] == 'burst' else 'lost'}"})
            if g["stall"] is not None:
                out.append({"kind": "stall", "bin": g["stall"], "gid": gid, "text": f"{gid} stops growing"})
            if g["conf"] is not None and g["conf"] < UNSURE and not g["done"]:
                out.append({"kind": "unsure", "bin": g["unsure"], "gid": gid,
                            "text": f"{gid} unsure ({100 * g['conf']:.0f}%)"})
        for f in self.pred.get("focus_changes") or []:
            out.append({"kind": "focus", "bin": int(f["bin"]), "gid": None, "text": "focus change"})
        if population.get("t50_bin") is not None:
            out.append({"kind": "t50", "bin": population["t50_bin"], "gid": None, "text": "T50"})
        return sorted(out, key=lambda e: (e["bin"], e["kind"]))

    def checks(self, grains: list[dict]) -> list[dict]:
        """The check list: grains with reasons to look; not yet looked at first, isolated grains first, then the
        model's least sure first (the order that fixed the most readings soonest, runs/review_triage)."""
        items = [g for g in grains if g["check"] and not g["excluded"]]
        key = lambda g: (g["done"], not g["isolated"], g["conf"] is None, g["conf"] if g["conf"] is not None else 1.0,
                         g["id"])
        return [{"gid": g["id"], "bin": g["check"][0]["bin"], "reasons": [r["text"] for r in g["check"]],
                 "conf": g["conf"], "done": g["done"], "isolated": g["isolated"]} for g in sorted(items, key=key)]

    def summary(self, grains: list[dict], population: dict) -> dict:
        iso = [g for g in grains if g["isolated"] and not g["excluded"] and g["status"] != "unobservable"]
        germ = [g for g in iso if g["status"] in EMERGED]
        rates = [g["rate"] for g in germ if g["rate"]]
        finals = [g["final"] for g in germ if g["final"] > 0]
        u = self.units
        rate = float(np.median(rates)) if rates else None
        lost = sum(1 for g in iso if g["lost"] is not None)
        parts = [f"{len(iso)} grains", f"{len(germ)} germinated ({100 * len(germ) / max(len(iso), 1):.0f}%)"]
        if population.get("t50_frame") is not None:
            parts.append(f"T50 {self.when(population['t50_frame'])}")
        if rate is not None:
            parts.append(f"growth {u.rate(rate):.3g} {u.rate_unit}")
        if lost:
            parts.append(f"{lost} lost")
        return {"n": len(iso), "germinated": len(germ), "t50_frame": population.get("t50_frame"),
                "median_rate": rate, "median_final": float(np.median(finals)) if finals else None, "tubes": len(rates),
                "lost": lost, "reviewed": sum(1 for g in grains if g["review"]["state"] != "model"),
                "grains": len(grains), "line": "  ·  ".join(parts)}

    def warnings(self) -> list[dict]:
        """Movie-level problems as short phrases (with the sentence as detail)."""
        isolated = {gid for gid, c in self.census.items() if c.get("isolated", True)}
        grains = [g for g in self.pred.get("grains", []) if g["id"] in isolated]
        out = []
        for f in self.pred.get("focus_changes") or []:
            out.append({"text": f"focus change at {self.when(f['frame'])}", "bin": int(f["bin"])})
        unfollowed = sum(1 for g in grains if "drift_rejected" in g.get("flags", []))
        lost = sum(1 for g in grains if any(f.startswith("grain_lost") for f in g.get("flags", [])))
        if grains and unfollowed / len(grains) > 0.25:
            out.append({"text": f"{unfollowed} of {len(grains)} grains not followed", "bin": None})
        if grains and lost / len(grains) > 0.25:
            out.append({"text": f"{lost} of {len(grains)} grains lost partway", "bin": None})
        details = report.movie_warnings(self.pred, isolated)
        for w in out:
            w["detail"] = next((d for d in details if w["text"].split()[0] in d or "focus" in w["text"] and "focus" in d),
                               "")
        return out

    def route(self, gid: str, b: int) -> tuple[list, bool]:
        """The route the tube is drawn along at bin ``b`` (``overlay.route_at``)."""
        return overlay.route_at(self.grain(gid), b)

    def drift_at(self, gid: str, b: int) -> tuple[float, float]:
        d = self._static[gid]["drift"]
        return (float(d[b][0]), float(d[b][1])) if d else (0.0, 0.0)
