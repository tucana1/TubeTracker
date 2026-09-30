"""The window's panels: the start screen (movies), the analysis progress of a movie not analysed yet, and the side
panel (the check list navigator, the selected grain, the check list, the events)."""

from __future__ import annotations

import time
from pathlib import Path

import wx

from . import theme
from .canvas import GrainView
from .charts import GrowthCurve
from .overlay import EMERGED, state_at
from .runfolder import REPO

STATE_WORDS = {"germinated": "germinated", "notyet": "not yet germinated", "never": "never germinated",
               "lost": "lost", "excluded": "excluded", "unobservable": "not readable"}
REVIEW_WORDS = {"model": "model", "partly checked": "partly checked", "checked": "checked", "corrected": "corrected",
                "excluded": "excluded"}


def bg(win, colour=None):
    win.SetBackgroundColour(colour or theme.panel_bg())
    return win


def fmt_secs(s: float) -> str:
    s = int(round(s or 0))
    return f"{s // 3600} h {(s % 3600) // 60} min" if s >= 3600 else f"{s // 60} min {s % 60} s" if s >= 60 else f"{s} s"


class StartPanel(wx.Panel):
    """No movie open: open one, or pick one analysed before."""

    COLS = (("Movie", 280), ("Sample", 110), ("Genotype", 90), ("Rep.", 50), ("Status", 150), ("Changed", 130))

    def __init__(self, parent, ctl):
        super().__init__(parent)
        bg(self)
        self.ctl = ctl
        self.runs = []
        outer = wx.BoxSizer(wx.VERTICAL)
        head = wx.BoxSizer(wx.HORIZONTAL)
        icon = ctl.icon_bitmap(40)
        if icon:
            head.Add(wx.StaticBitmap(self, bitmap=icon), 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 10)
        title = wx.StaticText(self, label="TubeTracker")
        title.SetFont(theme.font(20, bold=True))
        head.Add(title, 0, wx.ALIGN_CENTER_VERTICAL)
        outer.Add(head, 0, wx.LEFT | wx.TOP, 24)
        buttons = wx.BoxSizer(wx.HORIZONTAL)
        self.open_movie = wx.Button(self, label="Open Movie...")
        self.open_movie.SetDefault()
        self.open_folder = wx.Button(self, label="Open Analysis Folder...")
        buttons.Add(self.open_movie, 0, wx.RIGHT, 8)
        buttons.Add(self.open_folder, 0)
        outer.Add(buttons, 0, wx.LEFT | wx.TOP, 24)
        lab = wx.StaticText(self, label="Movies")
        lab.SetFont(theme.font(13, bold=True))
        outer.Add(lab, 0, wx.LEFT | wx.TOP, 24)
        self.list = wx.ListCtrl(self, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        for i, (name, width) in enumerate(self.COLS):
            self.list.InsertColumn(i, name, width=width)
        outer.Add(self.list, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 24)
        row = wx.BoxSizer(wx.HORIZONTAL)
        self.where = wx.StaticText(self, label="")
        self.where.SetForegroundColour(theme.muted_fg())
        row.Add(self.where, 1, wx.ALIGN_CENTER_VERTICAL)
        self.open_btn = wx.Button(self, label="Open")
        self.compare_btn = wx.Button(self, label="Compare Movies...")
        row.Add(self.compare_btn, 0, wx.LEFT, 8)
        row.Add(self.open_btn, 0, wx.LEFT, 8)
        outer.Add(row, 0, wx.EXPAND | wx.ALL, 24)
        self.SetSizer(outer)
        self.open_movie.Bind(wx.EVT_BUTTON, lambda e: ctl.on_open_movie())
        self.open_folder.Bind(wx.EVT_BUTTON, lambda e: ctl.on_open_folder())
        self.open_btn.Bind(wx.EVT_BUTTON, lambda e: self._open_selected())
        self.compare_btn.Bind(wx.EVT_BUTTON, lambda e: ctl.on_compare())
        self.list.Bind(wx.EVT_LIST_ITEM_ACTIVATED, lambda e: self._open_selected())

    def refresh(self, runs: list[dict], jobs: dict, runs_root: str):
        self.runs = runs
        busy = {j["folder"]: j for j in jobs.get("active", [])}
        sel = self.list.GetFirstSelected()
        self.list.DeleteAllItems()
        for i, r in enumerate(runs):
            j = busy.get(r["folder"])
            if j:
                status = (f"analysing {j['k']}/{j['n']}" if j["phase"] == "grains" and j["n"] else
                          "analysing" if j["state"] == "running" else "queued")
            elif r["analysed"]:
                status = "analysed, reviewed" if r["reviewed"] else "analysed"
            else:
                status = "prepared" if r["prepared"] else "not analysed"
            changed = time.strftime("%d %b %H:%M", time.localtime(r["updated"])) if r["updated"] else ""
            self.list.InsertItem(i, r["movie"])
            for c, v in enumerate((r["sample_id"], r["genotype"], r["replicate"], status, changed), start=1):
                self.list.SetItem(i, c, str(v or ""))
        if 0 <= sel < len(runs):
            self.list.Select(sel)
        root = Path(runs_root)
        shown = root.relative_to(REPO) if REPO in root.parents else root
        self.where.SetLabel(f"{len(runs)} in {shown}")
        self.open_btn.Enable(bool(runs))
        self.compare_btn.Enable(sum(1 for r in runs if r["analysed"]) >= 1)

    def _open_selected(self):
        i = self.list.GetFirstSelected()
        if 0 <= i < len(self.runs):
            self.ctl.open_path(self.runs[i]["folder"])


class AnalysisPanel(wx.Panel):
    """A movie that has not been analysed: start the analysis, follow it, cancel it."""

    ROWS = ("probe", "prepare", "register", "census", "maps", "speed", "grains", "finish")

    def __init__(self, parent, ctl):
        super().__init__(parent)
        bg(self)
        self.ctl = ctl
        col = wx.BoxSizer(wx.VERTICAL)
        self.title = wx.StaticText(self, label="")
        self.title.SetFont(theme.font(16, bold=True))
        col.Add(self.title, 0, wx.BOTTOM, 18)
        from .jobs import PHASE_LABELS
        grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=24)
        self.rows = {}
        for key in self.ROWS:
            name = wx.StaticText(self, label=PHASE_LABELS[key])
            state = wx.StaticText(self, label="")
            grid.Add(name, 0)
            grid.Add(state, 0)
            self.rows[key] = (name, state)
        self._font = state.GetFont()  # the labels' own font (the system's GUI font is smaller on macOS)
        col.Add(grid, 0, wx.BOTTOM, 18)
        self.message = wx.StaticText(self, label="")
        col.Add(self.message, 0, wx.BOTTOM, 10)
        self.gauge = wx.Gauge(self, range=1000, size=(460, -1))
        col.Add(self.gauge, 0, wx.EXPAND | wx.BOTTOM, 8)
        self.detail = wx.StaticText(self, label="")
        self.detail.SetForegroundColour(theme.muted_fg())
        col.Add(self.detail, 0, wx.BOTTOM, 18)
        row = wx.BoxSizer(wx.HORIZONTAL)
        self.start = wx.Button(self, label="Analyse")
        self.cancel = wx.Button(self, label="Cancel")
        self.settings = wx.Button(self, label="Settings...")
        for b in (self.start, self.cancel, self.settings):
            row.Add(b, 0, wx.RIGHT, 8)
        col.Add(row, 0, wx.BOTTOM, 18)
        self.logpane = wx.CollapsiblePane(self, label="Log")
        self.log = wx.TextCtrl(self.logpane.GetPane(), style=wx.TE_MULTILINE | wx.TE_READONLY, size=(620, 200))
        self.log.SetFont(wx.Font(wx.FontInfo(10).Family(wx.FONTFAMILY_TELETYPE)))
        ps = wx.BoxSizer(wx.VERTICAL)
        ps.Add(self.log, 1, wx.EXPAND)
        self.logpane.GetPane().SetSizer(ps)
        col.Add(self.logpane, 0, wx.EXPAND)
        outer = wx.BoxSizer(wx.HORIZONTAL)
        outer.AddStretchSpacer(1)
        outer.Add(col, 0, wx.TOP, 60)
        outer.AddStretchSpacer(2)
        self.SetSizer(outer)
        self.start.Bind(wx.EVT_BUTTON, lambda e: ctl.on_analyse())
        self.cancel.Bind(wx.EVT_BUTTON, lambda e: ctl.on_cancel_analysis())
        self.settings.Bind(wx.EVT_BUTTON, lambda e: ctl.on_settings())
        self.logpane.Bind(wx.EVT_COLLAPSIBLEPANE_CHANGED, lambda e: self.Layout())

    def show(self, name: str, job: dict | None):
        self.title.SetLabel(name)
        running = bool(job) and job["state"] in ("running", "queued")
        self.start.Show(not running)
        self.cancel.Show(running)
        self.start.SetLabel("Analyse Again" if job and job["state"] in ("failed", "cancelled") else "Analyse")
        phases = {p["phase"]: p for p in (job or {}).get("phases", [])}
        plain = self._font
        bold = plain.Bold()
        for key, (name_, state) in self.rows.items():
            p = phases.get(key)
            text = {"done": "done", "skipped": "already done", "running": "..."}.get(p["state"], "") if p else ""
            if p and p["state"] == "running" and job["n"]:
                text = f"{job['k']} of {job['n']}"
            state.SetLabel(text)
            now = bool(p) and p["state"] == "running"
            name_.SetFont(bold if now else plain)
            name_.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT) if p
                                      else theme.muted_fg())
        if not job:
            message = "Not analysed yet."
        elif job["state"] == "queued":
            message = "Waiting for another analysis to finish."
        elif job["state"] == "failed":
            message = f"Failed: {job['error'] or 'see the log'}"
        elif job["state"] == "cancelled":
            message = "Cancelled."
        else:
            message = job["label"] or ""
        self.message.SetLabel(message)
        if job and job["n"]:
            self.gauge.SetValue(int(1000 * job["k"] / job["n"]))
        elif running:
            self.gauge.Pulse()
        else:
            self.gauge.SetValue(0)
        eta = f"  ·  about {fmt_secs(job['eta'])} left" if job and job.get("eta") else ""
        self.detail.SetLabel(f"{fmt_secs(job['elapsed'])}{eta}" if job and job["elapsed"] else "")
        text = "\n".join((job or {}).get("log", []))
        if self.log.GetValue() != text:
            self.log.SetValue(text)
            self.log.ShowPosition(self.log.GetLastPosition())
        self.Layout()


class SidePanel(wx.Panel):
    """Check list navigator above three pages: the selected grain, the check list, the events."""

    ACTIONS = (("confirm", "Confirm (Enter)", "Onset and lengths shown are right"),
               ("onset", "Onset here (O)", "Tube first visible at this time"),
               ("tip", "Set tip (T)", "Click the tube tip at this time"),
               ("path", "Draw tube (D)", "Click from the grain along the tube to its tip, then Enter"),
               ("burst", "Burst (B)", "Burst or gone by this time"),
               ("not_a_grain", "Not a grain (X)", None),
               ("clump", "Clump (K)", None),
               ("undo", "Undo", "Take back the last correction"))

    def __init__(self, parent, ctl):
        super().__init__(parent, size=(392, -1))
        bg(self)
        self.ctl = ctl
        s = wx.BoxSizer(wx.VERTICAL)
        nav = wx.BoxSizer(wx.HORIZONTAL)
        self.nav_label = wx.StaticText(self, label="")
        self.nav_label.SetFont(self.nav_label.GetFont().Bold())
        nav.Add(self.nav_label, 1, wx.ALIGN_CENTER_VERTICAL)
        self.prev = wx.Button(self, label="Previous (P)")
        self.next = wx.Button(self, label="Next (N)")
        nav.Add(self.prev, 0, wx.LEFT, 6)
        nav.Add(self.next, 0, wx.LEFT, 6)
        s.Add(nav, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 10)
        self.progress = wx.Gauge(self, range=1000, size=(-1, 6))
        s.Add(self.progress, 0, wx.EXPAND | wx.ALL, 10)
        self.book = wx.Notebook(self)
        self.grain_page = self._grain_page(self.book)
        self.checks = wx.ListCtrl(self.book, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        for i, (name, width) in enumerate((("Grain", 58), ("Why", 232), ("Conf.", 50))):
            self.checks.InsertColumn(i, name, width=width)
        self.events = wx.ListCtrl(self.book, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        for i, (name, width) in enumerate((("Time", 80), ("Event", 260))):
            self.events.InsertColumn(i, name, width=width)
        self.book.AddPage(self.grain_page, "Grain")
        self.book.AddPage(self.checks, "Checks")
        self.book.AddPage(self.events, "Events")
        s.Add(self.book, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)
        self.SetSizer(s)
        self.prev.Bind(wx.EVT_BUTTON, lambda e: ctl.goto_check(-1))
        self.next.Bind(wx.EVT_BUTTON, lambda e: ctl.goto_check(1))
        # the lists are rebuilt when a grain is opened: open it after the list's own event has finished
        self.checks.Bind(wx.EVT_LIST_ITEM_SELECTED,
                         lambda e: None if self._filling else wx.CallAfter(ctl.open_check, e.GetIndex()))
        self.events.Bind(wx.EVT_LIST_ITEM_SELECTED,
                         lambda e: None if self._filling else wx.CallAfter(ctl.goto_event_index, e.GetIndex()))
        self._filling = False

    def _grain_page(self, parent):
        page = wx.ScrolledWindow(parent, style=wx.VSCROLL)
        page.SetScrollRate(0, 12)
        bg(page, theme.window_bg())
        s = wx.BoxSizer(wx.VERTICAL)
        head = wx.BoxSizer(wx.HORIZONTAL)
        self.gid = wx.StaticText(page, label="")
        self.gid.SetFont(theme.font(17, bold=True))
        head.Add(self.gid, 0, wx.ALIGN_BOTTOM)
        self.gstate = wx.StaticText(page, label="")
        self.gstate.SetForegroundColour(theme.muted_fg())
        head.Add(self.gstate, 1, wx.LEFT | wx.ALIGN_BOTTOM, 10)
        s.Add(head, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 10)
        self.why = wx.StaticText(page, label="")
        self.why.SetForegroundColour(theme.colour(theme.WARN_TEXT))
        s.Add(self.why, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 10)
        self.view = GrainView(page, self.ctl, size=300)
        s.Add(self.view, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, 8)
        zrow = wx.BoxSizer(wx.HORIZONTAL)
        self.now = wx.StaticText(page, label="")
        zrow.Add(self.now, 1, wx.ALIGN_CENTER_VERTICAL)
        self.zoom = wx.Choice(page, choices=["Near", "Wide", "Far"])
        self.zoom.SetSelection(0)
        zrow.Add(self.zoom, 0)
        s.Add(zrow, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 8)
        acts = wx.GridSizer(cols=2, vgap=5, hgap=6)
        self.buttons = {}
        for key, text, tip in self.ACTIONS:
            b = wx.Button(page, label=text)
            if tip:
                b.SetToolTip(tip)
            b.Bind(wx.EVT_BUTTON, lambda e, k=key: self.ctl.on_action(k))
            acts.Add(b, 0, wx.EXPAND)
            self.buttons[key] = b
        s.Add(acts, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 8)
        grid = wx.FlexGridSizer(cols=4, vgap=4, hgap=8)
        grid.AddGrowableCol(1)
        grid.AddGrowableCol(3)
        self.stats, self.stat_names = {}, []
        for key, name in (("onset", "Onset"), ("final", "Final"), ("rate", "Growth"), ("conf", "Confidence")):
            lab = wx.StaticText(page, label=name)
            lab.SetForegroundColour(theme.muted_fg())
            val = wx.StaticText(page, label="")
            grid.Add(lab, 0)
            grid.Add(val, 0, wx.EXPAND)
            self.stats[key] = val
            self.stat_names.append(lab)
        s.Add(grid, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 10)
        self.curve = GrowthCurve(page, self.ctl)
        s.Add(self.curve, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 8)
        self.status = wx.StaticText(page, label="")
        self.status.SetForegroundColour(theme.muted_fg())
        s.Add(self.status, 0, wx.ALL, 8)
        self.empty = wx.StaticText(page, label="Click a grain, or press N.")
        self.empty.SetForegroundColour(theme.muted_fg())
        s.Add(self.empty, 0, wx.ALL, 16)
        page.SetSizer(s)
        self.zoom.Bind(wx.EVT_CHOICE, lambda e: self.view.set_zoom(self.zoom.GetSelection()))
        self._page_sizer = s
        return page

    # ---- updates ------------------------------------------------------------------------------------
    def update_nav(self, checks: list[dict], sel: str | None):
        left = sum(1 for c in checks if not c["done"])
        cur = next((i for i, c in enumerate(checks) if c["gid"] == sel), None)
        if not checks:
            self.nav_label.SetLabel("Nothing flagged")
        elif cur is None:
            self.nav_label.SetLabel(f"{left} to check" if left else "All checked")
        else:
            self.nav_label.SetLabel(f"Check {cur + 1} of {len(checks)}" + (f"  ({left} left)" if left else ""))
        self.progress.SetValue(int(1000 * (len(checks) - left) / len(checks)) if checks else 0)
        self.prev.Enable(bool(checks))
        self.next.Enable(bool(checks))
        self.book.SetPageText(1, f"Checks ({left})" if left else "Checks")

    def update_lists(self, checks: list[dict], events: list[dict], data, sel: str | None):
        self._filling = True
        try:
            self.checks.DeleteAllItems()
            muted = theme.muted_fg()
            for i, c in enumerate(checks):
                self.checks.InsertItem(i, c["gid"])
                self.checks.SetItem(i, 1, " · ".join(c["reasons"]))
                self.checks.SetItem(i, 2, "" if c["conf"] is None else f"{100 * c['conf']:.0f}%")
                if c["done"]:
                    self.checks.SetItemTextColour(i, muted)
                if c["gid"] == sel:
                    self.checks.Select(i)
            self.events.DeleteAllItems()
            for i, ev in enumerate(events):
                self.events.InsertItem(i, data.when(data.frame(ev["bin"])))
                self.events.SetItem(i, 1, ev["text"])
        finally:
            self._filling = False

    def update_grain(self, g: dict | None, data, b: int, ready: bool, busy_note: str = ""):
        page = self.grain_page
        shown = g is not None
        for w in (self.gid, self.gstate, self.why, self.view, self.now, self.zoom, self.curve, self.status,
                  *self.stats.values(), *self.stat_names, *self.buttons.values()):
            w.Show(shown)
        self.empty.Show(not shown)
        if shown:
            st = state_at(g, b)
            self.gid.SetLabel(g["id"])
            where = [] if g["isolated"] else ([f"clump of {g['clump']}"] if g["clump"] > 1 else ["near edge"])
            self.gstate.SetLabel("  ·  ".join([STATE_WORDS[st], *where, REVIEW_WORDS[g["review"]["state"]]]))
            reasons = g.get("check") or []
            self.why.SetLabel(" · ".join(r["text"] for r in reasons) if reasons and not g["done"] else "")
            self.why.Wrap(max(self.grain_page.GetClientSize()[0] - 24, 200))
            self.why.SetToolTip("\n".join(f"{r['text']}: {r['detail']}" for r in reasons) if reasons else "")
            self.why.Show(bool(reasons) and not g["done"])
            L = g["L"][b]
            self.now.SetLabel(f"{data.when(data.frame(b))}" + (f"  ·  {data.length_words(L)}" if L > 0.5 else ""))
            u = data.units
            if g["status"] == "emerged_within" and g["onset_by"] is not None:
                onset = data.when_range(g["onset_after"], g["onset_by"])
            else:
                onset = {"emerged_at_start": "before start", "no_emergence_by_end": "none",
                         "unobservable": "-"}.get(g["status"], "-")
            grown = g["status"] in EMERGED and not g["excluded"]
            self.stats["onset"].SetLabel(onset + ("  (checked)" if g["review"]["onset"] != "model" else ""))
            self.stats["final"].SetLabel(data.length_words(g["final"]) if grown else "-")
            rate = u.rate(g["rate"]) if grown and g["rate"] else None
            self.stats["rate"].SetLabel(f"{rate:.3g} {u.rate_unit}" if rate is not None else "-")
            self.stats["conf"].SetLabel("-" if g["conf"] is None else f"{100 * g['conf']:.0f}%")
            for key, btn in self.buttons.items():
                btn.Enable(ready if key != "undo" else ready and self.ctl.can_undo())
            if g["excluded"]:
                self.buttons["not_a_grain"].SetLabel("Include again")
            else:
                self.buttons["not_a_grain"].SetLabel("Not a grain (X)")
            self.status.SetLabel(busy_note)
        self.view.Refresh()
        self.curve.Refresh()
        page.Layout()
        page.FitInside()
