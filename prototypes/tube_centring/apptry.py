"""Open a demo run folder in the app (a scratch copy: the cache linked, the analysis copied, no setup yet), go through
setup, overview, the least sure grains, growth view, results and export, saving a screenshot at each step."""
import shutil
import sys
import traceback
from pathlib import Path

REPO = Path("/Users/joshjiang/Documents/TubeTracker")
sys.path.insert(0, str(REPO))
import wx  # noqa: E402

from tubetracker.app import window  # noqa: E402
from tubetracker.app.dialogs import SetupDialog  # noqa: E402
from tubetracker.app.window import MainFrame  # noqa: E402

NAME = sys.argv[1]
HOURS, MINUTES = sys.argv[2], sys.argv[3]
S = Path(__file__).parent / "apptry"
src = REPO / "runs" / "sparsetrack" / NAME
runs = S / "runs"
dst = runs / NAME
if dst.exists():
    shutil.rmtree(dst)  # scratch copy made by this script
dst.mkdir(parents=True)
(dst / "cache").symlink_to((src / "cache").resolve())
shutil.copytree(src / "analysis", dst / "analysis")
shots = S / "shots" / NAME
shots.mkdir(parents=True, exist_ok=True)
window.reveal = lambda path: None

app = wx.App(False)
frame = MainFrame(runs)
frame.Show()
log = []


def grab(win, path):
    win.Update()
    dc = wx.ClientDC(win)
    w, h = win.GetClientSize()
    bmp = wx.Bitmap(w, h)
    mem = wx.MemoryDC(bmp)
    mem.Blit(0, 0, w, h, dc, 0, 0)
    mem.SelectObject(wx.NullBitmap)
    bmp.SaveFile(str(path), wx.BITMAP_TYPE_PNG)


def step(fn, delay, nxt=None):
    def run():
        try:
            fn()
        except Exception:  # noqa: BLE001
            log.append(traceback.format_exc())
            print(log[-1], flush=True)
            wx.CallLater(200, frame.Close)
            return
        if nxt:
            nxt()
    wx.CallLater(delay, run)


def s_start():
    grab(frame, shots / "0_start.png")
    frame.open_path(str(dst))


def s_dialog():
    dlg = next((w for w in wx.GetTopLevelWindows() if isinstance(w, SetupDialog)), None)
    print("setup dialog shown:", dlg is not None, flush=True)
    if dlg is None:
        return
    grab(dlg, shots / "1_setup_empty.png")
    dlg.hours.SetValue(HOURS)
    dlg.minutes.SetValue(MINUTES)
    wx.GetApp().Yield(True)
    grab(dlg, shots / "1_setup_filled.png")
    dlg.EndModal(wx.ID_OK)


def s_overview():
    frame.reviewer.wait(60)
    print("title:", frame.title.GetLabel(), "|", frame.line.GetLabel(), flush=True)
    print("grains:", len(frame.grains), "bins:", frame.data.n_bins, flush=True)
    frame.snapshot(shots / "2_overview.png")


def s_check():
    frame.goto_check(1)
    wx.GetApp().Yield(True)
    print("first check:", frame.sel, "|", frame.side.gid.GetLabel(), flush=True)
    frame.snapshot(shots / "3_check1.png")


def s_check2():
    frame.goto_check(1)
    wx.GetApp().Yield(True)
    print("second check:", frame.sel, flush=True)
    frame.snapshot(shots / "4_check2.png")
    frame.set_mode("g")


def s_growth():
    frame.snapshot(shots / "5_growth.png")
    frame.set_mode("n")
    frame.on_results()


import os
ZOOMS = [z.split("@") for z in os.environ.get("TT_ZOOM", "").split(",") if z]


def s_zooms():
    for k, (gid, b) in enumerate(ZOOMS):
        frame.select(gid, zoom=True)
        frame.set_bin(int(b))
        wx.GetApp().Yield(True)
        frame.snapshot(shots / f"7_zoom_{gid}_{b}.png")


def s_results():
    wx.GetApp().Yield(True)
    grab(frame.results_win, shots / "6_results.png")
    frame.results_win.Close()
    frame.on_export()
    print("exported:", sorted(p.name for p in (dst / "results").iterdir()), flush=True)
    wx.CallLater(300, frame.Close)


step(s_start, 800, lambda: step(s_dialog, 2500, lambda: step(s_overview, 1500, lambda: step(
    s_check, 1500, lambda: step(s_check2, 1500, lambda: step(s_growth, 1500, lambda: step(
        s_zooms, 1000, lambda: step(s_results, 2000))))))))
app.MainLoop()
print("errors:", len(log))
