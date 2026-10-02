"""Dialogs and secondary windows: the movie's settings, the results, several movies side by side, the keys."""

from __future__ import annotations

import wx

from . import theme
from .charts import Plot
from .overlay import EMERGED
from .panels import bg
from .runfolder import sample_name

PALETTE = theme.PALETTE


def fit_columns(table: wx.ListCtrl, least: int = 50) -> None:
    """Each column as wide as its widest entry or its heading, whichever is wider."""
    for c in range(table.GetColumnCount()):
        table.SetColumnWidth(c, wx.LIST_AUTOSIZE)
        content = table.GetColumnWidth(c)
        table.SetColumnWidth(c, wx.LIST_AUTOSIZE_USEHEADER)
        table.SetColumnWidth(c, max(content, table.GetColumnWidth(c), least))


def _num(text: str) -> float | None:
    text = (text or "").strip().replace(",", ".")
    if not text:
        return None
    v = float(text)
    if v < 0:
        raise ValueError("negative")
    return v


class SetupDialog(wx.Dialog):
    """How long the movie ran, the pixel size, the sample."""

    def __init__(self, parent, name: str, n_frames: int | None, setup: dict, prefs: dict, prepared: bool,
                 analyse: bool):
        super().__init__(parent, title=f"Settings: {name}", style=wx.DEFAULT_DIALOG_STYLE)
        bg(self)
        self.n_frames = n_frames
        s = wx.BoxSizer(wx.VERTICAL)
        head = wx.StaticText(self, label=name)
        head.SetFont(theme.font(14, bold=True))
        s.Add(head, 0, wx.LEFT | wx.RIGHT | wx.TOP, 16)
        sub = wx.StaticText(self, label=f"{n_frames:,} frames" if n_frames else "")
        sub.SetForegroundColour(theme.muted_fg())
        s.Add(sub, 0, wx.LEFT | wx.RIGHT, 16)
        grid = wx.FlexGridSizer(cols=2, vgap=8, hgap=10)
        grid.AddGrowableCol(1)
        dur = setup.get("duration_s")
        if not dur and prefs.get("s_per_frame") and n_frames:
            dur = prefs["s_per_frame"] * n_frames
        self.hours = wx.TextCtrl(self, value=str(int(dur // 3600)) if dur else "", size=(60, -1))
        self.minutes = wx.TextCtrl(self, value=f"{(dur % 3600) / 60:g}" if dur else "", size=(60, -1))
        row = wx.BoxSizer(wx.HORIZONTAL)
        row.Add(self.hours, 0)
        row.Add(wx.StaticText(self, label="h"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT, 5)
        row.Add(self.minutes, 0)
        row.Add(wx.StaticText(self, label="min"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 5)
        self.derived = wx.StaticText(self, label="")
        self.derived.SetForegroundColour(theme.muted_fg())
        row.Add(self.derived, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 14)
        real = wx.StaticText(self, label="Real time")
        tip = ("How long the recording took in real time, from the first frame to the last (not how long the movie "
               "plays): the time between frames follows from it")
        for c in (real, self.hours, self.minutes):
            c.SetToolTip(tip)
        grid.Add(real, 0, wx.ALIGN_CENTER_VERTICAL)
        grid.Add(row, 0, wx.EXPAND)
        um = setup.get("um_per_px") if setup.get("um_per_px") is not None else prefs.get("um_per_px")
        self.um = wx.TextCtrl(self, value="" if um in (None, "") else f"{um:g}", size=(80, -1))
        row = wx.BoxSizer(wx.HORIZONTAL)
        row.Add(self.um, 0)
        row.Add(wx.StaticText(self, label="µm per pixel (optional)"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 5)
        grid.Add(wx.StaticText(self, label="Pixel size"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid.Add(row, 0)
        self.fields = {}
        default_id = sample_name(name)
        for key, lab, default in (("sample_id", "Sample ID", default_id), ("genotype", "Genotype", ""),
                                  ("replicate", "Replicate", ""), ("notes", "Notes", "")):
            ctrl = wx.TextCtrl(self, value=str(setup.get(key) if setup.get(key) is not None else default),
                               size=(320, -1))
            grid.Add(wx.StaticText(self, label=lab), 0, wx.ALIGN_CENTER_VERTICAL)
            grid.Add(ctrl, 1, wx.EXPAND)
            self.fields[key] = ctrl
        s.Add(grid, 0, wx.EXPAND | wx.ALL, 16)
        self.more = wx.CollapsiblePane(self, label="Advanced")
        pane = self.more.GetPane()
        ps = wx.FlexGridSizer(cols=2, vgap=8, hgap=10)
        spf = setup.get("s_per_frame") if not setup.get("duration_s") else None
        self.spf = wx.TextCtrl(pane, value="" if not spf else f"{spf:g}", size=(80, -1))
        r2 = wx.BoxSizer(wx.HORIZONTAL)
        r2.Add(self.spf, 0)
        r2.Add(wx.StaticText(pane, label="s per frame (instead of the real time)"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 5)
        ps.Add(wx.StaticText(pane, label="Frame interval"), 0, wx.ALIGN_CENTER_VERTICAL)
        ps.Add(r2, 0)
        self.flat = wx.CheckBox(pane, label="Even out uneven illumination before finding grains" +
                                ("" if not prepared else " (only before the first analysis)"))
        self.flat.SetValue(bool(setup.get("flatfield")))
        self.flat.Enable(not prepared)
        ps.Add(wx.StaticText(pane, label=""), 0)
        ps.Add(self.flat, 0)
        pane.SetSizer(ps)
        s.Add(self.more, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 16)
        self.error = wx.StaticText(self, label="")
        self.error.SetForegroundColour(wx.Colour(200, 40, 40))
        s.Add(self.error, 0, wx.LEFT | wx.RIGHT | wx.TOP, 16)
        btns = wx.StdDialogButtonSizer()
        ok = wx.Button(self, wx.ID_OK, "Save and Analyse" if analyse else "Save")
        ok.SetDefault()
        btns.AddButton(ok)
        btns.AddButton(wx.Button(self, wx.ID_CANCEL))
        btns.Realize()
        s.Add(btns, 0, wx.EXPAND | wx.ALL, 16)
        self.SetSizerAndFit(s)
        for c in (self.hours, self.minutes, self.spf):
            c.Bind(wx.EVT_TEXT, lambda e: self._derive())
        self.more.Bind(wx.EVT_COLLAPSIBLEPANE_CHANGED, lambda e: self.Fit())
        ok.Bind(wx.EVT_BUTTON, self._ok)
        self._derive()
        self.hours.SetFocus()
        self.CentreOnParent()

    def _derive(self):
        try:
            secs = (_num(self.hours.GetValue()) or 0) * 3600 + (_num(self.minutes.GetValue()) or 0) * 60
            spf = _num(self.spf.GetValue())
        except ValueError:
            self.derived.SetLabel("")
            return
        if secs and self.n_frames:
            self.derived.SetLabel(f"{secs / self.n_frames:.3f} s per frame")
        elif spf and self.n_frames:
            self.derived.SetLabel(f"{spf * self.n_frames / 3600:.2f} h in all")
        else:
            self.derived.SetLabel("blank: times in frames")
        self.Layout()

    def values(self) -> dict:
        out = {"hours": self.hours.GetValue(), "minutes": self.minutes.GetValue(),
               "s_per_frame": self.spf.GetValue(), "um_per_px": self.um.GetValue(), "flatfield": self.flat.GetValue()}
        out.update({k: c.GetValue() for k, c in self.fields.items()})
        return out

    def _ok(self, e):
        try:
            for c in (self.hours, self.minutes, self.spf, self.um):
                _num(c.GetValue())
        except ValueError:
            self.error.SetLabel("Numbers only (use . for decimals).")
            self.Fit()
            return
        e.Skip()


KEYS = (("Space", "Play / pause"), ("Left, Right", "Previous / next time (Shift: 10)"), ("Home, End", "First / last time"),
        ("N, P", "Next / previous grain to check"), ("], [", "Next / previous grain"), ("C", "Movie, contrast, growth"),
        ("G", "Growth view on / off"), ("F", "Fit the field"), ("Z", "Zoom to the grain"), ("I", "Names on / off"),
        ("Enter", "Confirm the grain"), ("O", "Onset here"), ("Shift-O", "Never germinated"), ("T", "Set the tip"),
        ("D", "Draw the tube (Enter saves)"), ("B", "Burst or gone here"), ("X", "Not a grain (again: include)"),
        ("K", "Clump"), ("U", "Back to the model's answer"), ("Cmd-Z", "Undo"),
        ("Backspace", "Take back a point while drawing"), ("=, -", "Zoom in / out"),
        ("Esc", "Cancel the tool, else back to the movie's numbers"), ("F1, ?", "Help"))


class ShortcutsDialog(wx.Dialog):
    def __init__(self, parent):
        super().__init__(parent, title="Keyboard Shortcuts")
        bg(self)
        grid = wx.FlexGridSizer(cols=2, vgap=5, hgap=16)
        for k, v in KEYS:
            key = wx.StaticText(self, label=k)
            key.SetFont(key.GetFont().Bold())
            grid.Add(key, 0, wx.ALIGN_RIGHT)
            grid.Add(wx.StaticText(self, label=v), 0)
        s = wx.BoxSizer(wx.VERTICAL)
        s.Add(grid, 0, wx.ALL, 18)
        btn = wx.Button(self, wx.ID_OK, "Close")
        btn.SetDefault()
        s.Add(btn, 0, wx.ALIGN_RIGHT | wx.RIGHT | wx.BOTTOM, 16)
        self.SetSizerAndFit(s)
        self.CentreOnParent()


class ResultsFrame(wx.Frame):
    """One movie's numbers, germination curve, growth curves and per-grain table; Export writes them."""

    COLS = (("Grain", 60), ("State", 150), ("Onset", 80), ("Final", 80), ("Growth", 90), ("Confidence", 80),
            ("Checked", 90), ("Counted", 70))

    def __init__(self, parent, ctl):
        super().__init__(parent, title="Results", size=(1080, 820))
        self.ctl = ctl
        self.SetIcons(ctl.icons())
        p = wx.Panel(self)
        bg(p)
        s = wx.BoxSizer(wx.VERTICAL)
        self.head = wx.StaticText(p, label="")
        self.head.SetFont(theme.font(14, bold=True))
        s.Add(self.head, 0, wx.LEFT | wx.RIGHT | wx.TOP, 14)
        self.line = wx.StaticText(p, label="")
        s.Add(self.line, 0, wx.LEFT | wx.RIGHT | wx.TOP, 14)
        plots = wx.BoxSizer(wx.HORIZONTAL)
        self.germ = Plot(p, size=(500, 250))
        self.grow = Plot(p, size=(500, 250))
        plots.Add(self.germ, 1, wx.EXPAND | wx.RIGHT, 8)
        plots.Add(self.grow, 1, wx.EXPAND)
        s.Add(plots, 0, wx.EXPAND | wx.ALL, 14)
        self.table = wx.ListCtrl(p, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        s.Add(self.table, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 14)
        self.about = wx.StaticText(p, label="")
        self.about.SetForegroundColour(theme.muted_fg())
        s.Add(self.about, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 14)
        row = wx.BoxSizer(wx.HORIZONTAL)
        self.note = wx.StaticText(p, label="")
        self.note.SetForegroundColour(theme.muted_fg())
        row.Add(self.note, 1, wx.ALIGN_CENTER_VERTICAL)
        self.show_folder = wx.Button(p, label="Show Folder")
        self.show_folder.SetToolTip("Open the results folder in the Finder")
        row.Add(self.show_folder, 0, wx.RIGHT, 8)
        export = wx.Button(p, label="Export")
        export.SetToolTip("Write the tables and figures to the movie's results folder")
        row.Add(export, 0)
        s.Add(row, 0, wx.EXPAND | wx.ALL, 14)
        p.SetSizer(s)
        self.rows = []
        self._sort = (None, 1)
        export.Bind(wx.EVT_BUTTON, lambda e: ctl.on_export())
        self.show_folder.Bind(wx.EVT_BUTTON, lambda e: ctl.show_results_folder())
        self.table.Bind(wx.EVT_LIST_COL_CLICK, self._sort_by)
        self.Bind(wx.EVT_CHAR_HOOK, lambda e: self.Close() if e.GetKeyCode() == wx.WXK_ESCAPE else e.Skip())
        self.table.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self._open)
        self.grow.on_pick = self._pick_curve
        self.refresh()

    def refresh(self):
        ctl, d = self.ctl, self.ctl.data
        if d is None:
            return
        u = d.units
        self.SetTitle(f"Results: {d.setup.get('sample_id') or d.folder.name}")
        self.head.SetLabel(" ".join(x for x in (d.setup.get("sample_id") or d.folder.name, d.setup.get("genotype"),
                                                f"rep {d.setup['replicate']}" if d.setup.get("replicate") else "") if x))
        sm = ctl.summary
        extra = f"  ·  median final length {d.length_words(sm['median_final'])}" if sm["median_final"] else ""
        self.line.SetLabel(sm["line"] + extra + f"  ·  {sm['reviewed']} of {sm['grains']} checked")
        pop, nb = ctl.population, d.n_bins
        t = [u.time(d.frame(b)) for b in range(nb)]
        tl = "min" if u.timed else "frame"
        vl = []
        if pop["t50_frame"] is not None:
            vl.append((u.time(pop["t50_frame"]), theme.SELECTED, f"T50 {d.when(pop['t50_frame'])}"))
        self.germ.set(series=[{"x": t, "y": pop["certain"], "band_y": pop["possible"], "colour": theme.SELECTED,
                               "width": 2.0}], vlines=vl, yrange=(0, 1), xlabel=tl, ylabel="germinated",
                      yfmt=lambda v: f"{100 * v:.0f}%")
        self.curves = [g for g in ctl.grains if g["status"] in EMERGED and not g["excluded"] and g["isolated"]]
        series = []
        for g in sorted(self.curves, key=lambda g: g["id"] == ctl.sel):  # the selected grain on top
            end = g["lost"] if g["lost"] is not None else nb
            sel = g["id"] == ctl.sel
            series.append({"x": t[:end], "y": [u.length(v) for v in g["L"][:end]],
                           "colour": theme.TUBE if sel else (170, 120, 150), "width": 2.6 if sel else 1.0,
                           "alpha": 255 if sel else 170})
        self.curves = sorted(self.curves, key=lambda g: g["id"] == ctl.sel)
        self.grow.set(series=series, xlabel=tl, ylabel=u.length_unit)
        self.rows = []
        for g in ctl.grains:
            grown = g["status"] in EMERGED and not g["excluded"]
            in_curve = g["isolated"] and not g["excluded"] and g["status"] in (*EMERGED, "no_emergence_by_end")
            state = (f"excluded ({g['excluded'].replace('_', ' ')})" if g["excluded"] else
                     {"emerged_within": "germinated", "emerged_at_start": "before start",
                      "no_emergence_by_end": "lost, not germinated" if g["lost"] is not None else "not germinated",
                      "unobservable": "not readable"}.get(g["status"], g["status"]))
            onset = u.time(g["onset_by"]) if g["status"] == "emerged_within" and g["onset_by"] is not None else None
            self.rows.append((g["id"], state, onset, u.length(g["final"]) if grown else None,
                              u.rate(g["rate"]) if grown and g["rate"] else None, g["conf"],
                              {"model": "", "partly checked": "partly", "checked": "yes", "corrected": "corrected",
                               "excluded": "excluded"}[g["review"]["state"]], "yes" if in_curve else "no"))
        self.units = (tl, u.length_unit, u.rate_unit)
        self._fill()
        about = [sm.get("detail") or "", "Confidence: the model's, where it is least sure. Checked: what a person "
                                         "did with the grain."]
        if not u.timed:
            about.append("Give the real time the recording took in Settings (Cmd-,) for times in minutes.")
        if not u.scaled:
            about.append("Give the pixel size in Settings (Cmd-,) for lengths in µm.")
        self.about.SetLabel(" ".join(about))
        self.about.Wrap(max(self.GetClientSize()[0] - 40, 400))
        exists = d.folder.results.exists()
        self.note.SetLabel("Exported to the movie's results folder" if exists else
                           "Export writes the tables and figures to the movie's results folder")
        self.note.SetToolTip(str(d.folder.results))
        self.show_folder.Enable(exists)
        self.Layout()

    def _fill(self):
        self.table.ClearAll()
        tl, lu, ru = getattr(self, "units", ("min", "µm", "µm/min"))
        names = {"Onset": f"Onset ({tl})", "Final": f"Final ({lu})", "Growth": f"Growth ({ru})"}
        for i, (name, width) in enumerate(self.COLS):
            self.table.InsertColumn(i, names.get(name, name), width=width + (30 if name in names else 0),
                                    format=wx.LIST_FORMAT_RIGHT if 2 <= i <= 5 else wx.LIST_FORMAT_LEFT)
        col, direction = self._sort
        rows = list(self.rows)
        if col is not None:
            rows.sort(key=lambda r: ((r[col] is None), r[col] if r[col] is not None else 0), reverse=direction < 0)
        fmt = lambda v, nd: "" if v is None else f"{v:.{nd}f}"
        for i, r in enumerate(rows):
            self.table.InsertItem(i, r[0])
            vals = (r[1], fmt(r[2], 0), fmt(r[3], 1), "" if r[4] is None else f"{r[4]:.3g}",
                    "" if r[5] is None else f"{100 * r[5]:.0f}%", r[6], r[7])
            for c, v in enumerate(vals, start=1):
                self.table.SetItem(i, c, v)
        fit_columns(self.table)
        self._shown = rows

    def _sort_by(self, e):
        col = e.GetColumn()
        self._sort = (col, -self._sort[1] if self._sort[0] == col else 1)
        self._fill()

    def _open(self, e):
        gid = self._shown[e.GetIndex()][0]
        self.ctl.select(gid, zoom=True)
        self.ctl.Raise()

    def _pick_curve(self, i):
        self.ctl.select(self.curves[i]["id"], zoom=True)
        self.refresh()


class CompareFrame(wx.Frame):
    """Several analysed movies side by side: germination curves, growth rates, their numbers; Export writes them."""

    COLS = (("Movie", 210), ("Genotype", 80), ("Rep.", 45), ("Grains", 55), ("Germinated", 80), ("T50", 110),
            ("Growth", 150), ("Final length", 90), ("Lost", 45), ("Checked", 80))

    def __init__(self, parent, ctl, runs: list[dict]):
        super().__init__(parent, title="Compare Movies", size=(1180, 760))
        self.ctl = ctl
        self.SetIcons(ctl.icons())
        self.runs = [r for r in runs if r["analysed"]]
        self._rows, self._stamps = {}, {}  # folder -> its row, and when its analysis and checks last changed
        p = wx.Panel(self)
        bg(p)
        s = wx.BoxSizer(wx.HORIZONTAL)
        left = wx.BoxSizer(wx.VERTICAL)
        left.Add(wx.StaticText(p, label="Movies"), 0, wx.BOTTOM, 6)
        self.pick = wx.CheckListBox(p, choices=[self._label(r) for r in self.runs], size=(230, -1))
        for i in range(min(len(self.runs), 8)):
            self.pick.Check(i)
        left.Add(self.pick, 1, wx.EXPAND)
        export = wx.Button(p, label="Export")
        export.SetToolTip("Write summary.csv and summary.png, as shown here, to the runs folder's summary folder")
        left.Add(export, 0, wx.EXPAND | wx.TOP, 8)
        s.Add(left, 0, wx.EXPAND | wx.ALL, 14)
        right = wx.BoxSizer(wx.VERTICAL)
        self.table = wx.ListCtrl(p, style=wx.LC_REPORT | wx.LC_SINGLE_SEL, size=(-1, 200))
        for i, (name, width) in enumerate(self.COLS):
            self.table.InsertColumn(i, name, width=width)
        right.Add(self.table, 0, wx.EXPAND)
        plots = wx.BoxSizer(wx.HORIZONTAL)
        self.germ = Plot(p, size=(460, 300))
        self.rate = Plot(p, size=(360, 300))
        plots.Add(self.germ, 3, wx.EXPAND | wx.RIGHT, 8)
        plots.Add(self.rate, 2, wx.EXPAND)
        right.Add(plots, 1, wx.EXPAND | wx.TOP, 10)
        self.note = wx.StaticText(p, label="")
        self.note.SetForegroundColour(theme.muted_fg())
        right.Add(self.note, 0, wx.TOP, 6)
        s.Add(right, 1, wx.EXPAND | wx.TOP | wx.RIGHT | wx.BOTTOM, 14)
        p.SetSizer(s)
        self.pick.Bind(wx.EVT_CHECKLISTBOX, lambda e: self.refresh())
        self.Bind(wx.EVT_ACTIVATE, lambda e: (self.refresh() if e.GetActive() else None, e.Skip()))
        export.Bind(wx.EVT_BUTTON, lambda e: ctl.on_export_compare(self.chosen(), [self._rows[f] for f in self.chosen()]))
        self.refresh()

    @staticmethod
    def _label(r):
        return "  ".join(x for x in (r["sample_id"] or r["movie"], r["genotype"],
                                     f"rep {r['replicate']}" if r["replicate"] else "") if x)

    def chosen(self) -> list[str]:
        return [self.runs[i]["folder"] for i in self.pick.GetCheckedItems()]

    @staticmethod
    def _stamp(folder: str) -> tuple:
        from .runfolder import RunFolder
        f = RunFolder(folder)
        return tuple(p.stat().st_mtime if p.exists() else 0.0 for p in (f.predictions, f.review_labels, f.setup_path))

    def refresh(self):
        folders = self.chosen()
        stamps = {f: self._stamp(f) for f in folders}  # checks made meanwhile show when the window comes back
        todo = [f for f in folders if f not in self._rows or self._stamps.get(f) != stamps[f]]
        if todo:
            wx.BeginBusyCursor()
            try:
                self._rows.update(zip(todo, self.ctl.compare_rows(todo)))
                self._stamps.update({f: stamps[f] for f in todo})
            finally:
                wx.EndBusyCursor()
        rows = [self._rows[f] for f in folders]
        self.table.DeleteAllItems()
        f = lambda v, nd=1: "" if v is None else f"{v:.{nd}f}"
        for i, r in enumerate(rows):
            self.table.InsertItem(i, r["sample_id"] or r["name"])
            vals = (r["genotype"], r["replicate"], str(r["grains"]),
                    "" if r["germinated"] is None else f"{100 * r['germinated']:.0f}%",
                    "" if r["t50"] is None else f"{r['t50']:.0f} {r['time_unit']}",
                    "" if r["median_rate"] is None else f"{r['median_rate']:.3g} {r['rate_unit']}",
                    "" if r["median_final"] is None else f"{f(r['median_final'])} {r['length_unit']}",
                    str(r["lost"]), f"{r['checked']} of {r['all']}")
            for c, v in enumerate(vals, start=1):
                self.table.SetItem(i, c, v)
            self.table.SetItemTextColour(i, theme.colour(theme.readable(PALETTE[i % len(PALETTE)])))
        fit_columns(self.table)
        self.table.SetColumnWidth(0, min(self.table.GetColumnWidth(0), 220))
        # one time axis and one rate axis: movies in other units (no duration or pixel size given) are left out
        timed = any(r["timed"] for r in rows)
        plotted = [i for i, r in enumerate(rows) if r["timed"] == timed]
        rate_unit = rows[plotted[0]]["rate_unit"] if plotted else ""
        rated = [i for i in plotted if rows[i]["rate_unit"] == rate_unit]
        series, dots = [], []
        for i in plotted:
            r = rows[i]
            c = PALETTE[i % len(PALETTE)]
            if r["curve"]["t"]:
                series.append({"x": r["curve"]["t"], "y": r["curve"]["y"], "colour": c, "width": 2.0})
            if r["t50"] is not None:
                dots.append((r["t50"], 0.5, c, 4.5))
        self.germ.set(series=series, dots=dots, yrange=(0, 1), xlabel="min" if timed else "frame",
                      ylabel="germinated", yfmt=lambda v: f"{100 * v:.0f}%")
        rdots, meds = [], []
        for x, i in enumerate(rated):
            r = rows[i]
            c = PALETTE[i % len(PALETTE)]
            for k, v in enumerate(v for v in r["rates"] if v is not None):
                rdots.append((x + ((k * 37) % 21 - 10) / 50.0, v, c, 2.6))
            if r["median_rate"] is not None:
                meds.append({"x": [x - 0.3, x + 0.3], "y": [r["median_rate"]] * 2, "colour": (229, 235, 245),
                             "width": 2.4})
        self.rate.set(series=meds, dots=rdots, xrange=(-0.6, max(len(rated) - 0.4, 0.6)),
                      categories=[rows[i]["sample_id"] or rows[i]["name"] for i in rated],
                      ylabel=f"growth ({rate_unit})")
        note = ("Each movie as it shows in the app: your checks where you gave them, the model elsewhere. Dots: T50."
                if rows else "Tick movies to compare.")
        name = lambda i: rows[i]["sample_id"] or rows[i]["name"]
        untimed = [name(i) for i in range(len(rows)) if i not in plotted]
        unscaled = [name(i) for i in plotted if i not in rated]
        if untimed:
            note += f" Not plotted: {', '.join(untimed)} (no real time given: set it in that movie's Settings)."
        if unscaled:
            note += f" No growth plotted for {', '.join(unscaled)} (lengths in other units: check the pixel sizes)."
        self.note.SetLabel(note)
        self.note.Wrap(max(self.GetClientSize()[0] - 300, 400))
        self.Layout()
