"""The help window (Help > TubeTracker Help, F1 or ?): what everything on screen means and how to correct a reading,
in one place, so the window itself can stay short."""

from __future__ import annotations

import wx
import wx.html

from . import theme

SECTIONS = ("start", "movie", "timeline", "checking", "corrections", "numbers", "files", "keys")


def _hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(rgb[:3])


def _swatch(rgb, word: str) -> str:
    return f'<font color="{_hex(theme.readable(rgb))}"><b>{word}</b></font>'


def page_html(fg, bg, accent) -> str:
    """The guide as one HTML page in the given colours (the system's, so it reads in light and dark mode)."""
    from .dialogs import KEYS

    S = theme.STATE
    keys = "".join(f"<tr><td align=right><b>{k}</b></td><td>&nbsp;&nbsp;{v}</td></tr>" for k, v in KEYS)
    return f"""<html><body bgcolor="{bg}" text="{fg}" link="{accent}">
<font size=2>
<p><a href="#start">Getting started</a> &middot; <a href="#movie">The movie</a> &middot;
<a href="#timeline">Timeline</a> &middot; <a href="#checking">Checking</a> &middot;
<a href="#corrections">Corrections</a> &middot; <a href="#numbers">The numbers</a> &middot;
<a href="#files">Files</a> &middot; <a href="#keys">Keys</a></p>

<a name="start"></a><h3>Getting started</h3>
<ol>
<li><b>Open Movie</b> (Cmd-O), or drop a movie on the window. Give the real time the recording took (not how
long the movie plays), and the pixel size if you know it (lengths are in pixels without it). Both can be changed
later in <b>Settings</b> (Cmd-,).</li>
<li>The analysis runs by itself; the window shows how far it is. It can take a while for a long movie, and other
movies can be opened meanwhile.</li>
<li>When it is done, the movie opens with every grain and tube drawn on it, and the side panel shows its numbers.</li>
<li><b>Check</b> the grains the model is least sure of (N), correct what is wrong and confirm what is right
(Enter).</li>
<li><b>Results</b> (Cmd-R) shows the numbers and curves; <b>Export</b> (Cmd-E) writes them to the movie's
results folder.</li>
</ol>

<a name="movie"></a><h3>The movie</h3>
<p>Each grain is circled in the colour of its state at the time shown:
{_swatch(S["germinated"], "germinated")}, {_swatch(S["notyet"], "not yet")} (it germinates later),
{_swatch(S["never"], "never")} (not by the end), {_swatch(S["lost"], "lost")} (dashed: burst, drifted off or swept
away), {_swatch(S["excluded"], "excluded")} (dashed, crossed: not a grain or a clump) and
{_swatch(S["unobservable"], "not readable")} (dashed).</p>
<p>The {_swatch(theme.TUBE, "pink line")} is the tube as measured at that time, from where it leaves the grain
(white dot) to its {_swatch(theme.TIP, "tip")} (yellow dot); a tube shorter than 2 px counts as none. In the
close-up, a faint dashed line shows the path the tube is measured along. An
{_swatch(theme.CHECK, "amber diamond")} marks a grain to check. Grains in a clump or at the edge of the field are
drawn too, but are not counted in the movie's numbers.</p>
<p>Views (above the movie, or C):</p>
<ul>
<li><b>Movie</b>: the movie itself. Each time step averages the movie over a few hundred frames, which makes faint
tubes clearer than in single frames.</li>
<li><b>Contrast</b>: the same, stretched about the background grey so faint tubes show.</li>
<li><b>Growth</b> (G): what changed over the last six time steps. New tube shows dark, so growing tips stand
out.</li>
</ul>
<p>Drag to move the movie, scroll or pinch to zoom (or = and -), F to fit it, Z to zoom to the selected grain;
double-click a grain to zoom to it. Click a grain to select it; click an empty place, or press Esc, to go back to
the movie's numbers. The close-up on the right follows the selected grain as it moves.</p>
<p>The slider, Left and Right (Shift: ten steps), Home and End move in time; Space plays.</p>

<a name="timeline"></a><h3>The timeline</h3>
<p>Under the movie. At the top, the share of counted grains germinated by each time: the line is what had surely
germinated by then, the band what may have (an onset is only known to lie between two time steps). The dashed
line marks <b>T50</b>. Below it, one lane per kind of event:</p>
<ul>
<li><b>Germination</b>: each grain's onset.</li>
<li><b>Lost</b>: a grain burst, drifted out of view or was swept off.</li>
<li><b>Stopped</b>: a tube that stopped growing (no growth for 15 time steps before the end or its grain's loss,
and at least 8 px long).</li>
<li><b>To check</b>: the grains still to check, at the time the check list opens them.</li>
</ul>
<p>Marks of grains that are not counted (in a clump or at the edge) are faint.</p>
<p>Dashed purple lines are focus changes. Click a mark to go to that grain and time; click or drag elsewhere to
move in time. The Events tab lists the same events.</p>

<a name="checking"></a><h3>Checking</h3>
<p>The model marks the grains a person should look at. N and P go through them: the ones not yet looked at first,
counted grains before those in clumps or at the edge, and the least sure first. The Checks tab lists them all
(hover over one for its reasons in full). The side panel says why each was marked; click a reason to go to its
time.</p>
<ul>
<li><b>unsure length</b>: the model's least sure reading of the tube. Is there a tube, and does it reach as far as
drawn?</li>
<li><b>drawn off the tube</b>: most of the tube drawn then does not lie on a tube. The reading may have taken
another tube or lost its own.</li>
<li><b>onset moved back</b>, <b>onset from later growth</b>, <b>onset at focus change</b>, <b>settling at
start</b>: check when the tube first shows.</li>
<li><b>lost at</b>: check that the grain really burst or left.</li>
<li><b>touches</b>, <b>reaches another grain</b>, <b>shares growth with a neighbour</b>: the tube may run into a
neighbour's; check its length.</li>
<li><b>not followed</b>: the grain moved in a way that could not be followed, so it was read at its first
place.</li>
<li><b>growth off the tube</b>, <b>no tube path found</b>: much of what grew round the grain is not on the tube
drawn; the tube may curl or branch, or another may be close.</li>
<li><b>growth but no onset</b>, <b>too short for a tube</b>: something grew next to the grain, but no tube was
seen leaving it, or it never got longer than 8 px.</li>
<li><b>no grain outline</b>: probably debris (Not a grain, X). Confirm does not take such a grain: say what it
is.</li>
</ul>
<p>Look at the grain at a few times (move in time, or click its growth curve). If the onset and the tube are right,
<b>Confirm</b> (Enter): the grain counts as checked and the next one opens. If not, correct it (below), then
confirm or go on with N.</p>
<p>The growth curve under the close-up shows the tube's length over time: pink as shown, dashed grey the model's
before your corrections, white dots the lengths you gave, the green dashed line the onset, the blue line the time
shown, an amber diamond the model's least sure reading, and orange shading after the grain was lost.</p>

<a name="corrections"></a><h3>Corrections</h3>
<p>Corrections are saved at once and can be taken back with <b>Undo</b> (Cmd-Z). Those about the onset or the
tube are about the time shown.</p>
<ul>
<li><b>Onset: Here</b> (O): the tube is first visible at this time. <b>Never</b> (Shift-O): the grain never
germinated.</li>
<li><b>Tube: Set tip</b> (T), then click the tube's tip: its length is read along the route drawn (a click a little
beyond the route's end carries it on straight). <b>Draw</b> (D): click where the tube leaves the grain, then along
it to its tip, and press Enter (Backspace takes a point back, Esc cancels); for a tube the route gets wrong.
<b>None here</b>: there is no tube at this time.</li>
<li><b>Grain: Burst</b> (B): the grain burst or is gone by this time, so nothing is measured after it. <b>Not a
grain</b> (X): debris and the like; press X again to include it. <b>Clump</b> (K): grains stuck together. Both
leave the grain out of the numbers.</li>
<li><b>Back to Model's Answer</b> (U, in the Grain menu) forgets everything said about the grain.</li>
</ul>
<p>Right-click a grain on the movie for the same corrections. The Undo button's tip says what it would take
back. Pressing D again while drawing saves the tube, as Enter does.</p>
<p>A length you give pins the growth curve at that time: between your lengths the curve keeps the shape of the
model's, and after the last one it grows as the model's did. It never shrinks. One or two lengths where the tube is
long are usually enough.</p>
<p>Corrections are kept in the movie's review folder and are there the next time the movie is opened.</p>

<a name="numbers"></a><h3>The numbers</h3>
<ul>
<li><b>Counted grains</b>: the grains on their own (not in a clump or at the edge of the field), readable and not
excluded. The germination numbers, the curve and T50 are of these.</li>
<li><b>Germinated</b>: the counted grains whose tube appeared, during the movie or before it. Their share is
where the germination curve ends: grains lost before they germinated count only until they were lost, so when
there are such grains it is not simply germinated over counted ("by the curve").</li>
<li><b>Onset</b>: when the tube first showed: absent at one time step, visible at the next.</li>
<li><b>T50</b>: the time by which half the counted grains had surely germinated. Grains lost before germinating
count until they were lost.</li>
<li><b>Final length</b>: the tube's length at the end of the movie, or when its grain was lost.</li>
<li><b>Growth</b>: the length a tube gained between reaching 10% and 90% of its final length, over that time
(leaving out the slow start and the stop). The movie's growth is the median over its tubes.</li>
<li><b>Confidence</b>: how likely the model's length is right at its least sure time, judged from the reading's own
history (long tubes and tubes still growing are more often right; a reading that stood still for long is
suspect). It orders the checks; it does not make checking safe to skip.</li>
</ul>
<p>Times are in minutes once the real time the recording took is given (else in frames); lengths in µm once the
pixel size is given (else in pixels).</p>

<a name="files"></a><h3>Files</h3>
<p>Each movie has a folder of its own (File &gt; Open Analysis Folder shows them):</p>
<ul>
<li><b>results</b>, written by Export: grains.csv (one row per grain), growth.csv (every tube's length at every
time), germination.png, growth_curves.png and summary.txt. They follow your corrections where you gave them and
the model elsewhere.</li>
<li><b>review</b>: your corrections.</li>
<li><b>analysis</b>: the model's own readings and tables.</li>
</ul>
<p><b>Compare Movies</b> (File menu) puts several analysed movies side by side, each as the app shows it, and its
Export writes the same table and curves as summary.csv and summary.png. <b>Analyse Again</b> (Movie menu) runs the analysis anew; the current one and its checks are kept in
the folder's <i>earlier</i> folder.</p>

<a name="keys"></a><h3>Keys</h3>
<table cellspacing=2 cellpadding=0>{keys}</table>
</font></body></html>"""


class HelpFrame(wx.Frame):
    """The guide in a window of its own, beside the movie."""

    def __init__(self, parent, section: str | None = None):
        super().__init__(parent, title="TubeTracker Help", size=(620, 760),
                         style=wx.DEFAULT_FRAME_STYLE | wx.FRAME_FLOAT_ON_PARENT)
        icons = getattr(parent, "icons", None)
        if icons:
            self.SetIcons(icons())
        self.html = wx.html.HtmlWindow(self, style=wx.html.HW_SCROLLBAR_AUTO)
        fg = wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT)
        bg = theme.window_bg()
        self.html.SetBackgroundColour(bg)
        self.html.SetStandardFonts(theme.font().GetPointSize() + 1)
        self.html.SetPage(page_html(fg.GetAsString(wx.C2S_HTML_SYNTAX), bg.GetAsString(wx.C2S_HTML_SYNTAX),
                                    _hex(theme.readable(theme.SELECTED))))
        s = wx.BoxSizer(wx.VERTICAL)
        s.Add(self.html, 1, wx.EXPAND)
        self.SetSizer(s)
        self.html.Bind(wx.html.EVT_HTML_LINK_CLICKED, self._link)
        self.Bind(wx.EVT_CHAR_HOOK, lambda e: self.Close() if e.GetKeyCode() == wx.WXK_ESCAPE else e.Skip())
        if section:
            self.show(section)
        self.CentreOnParent()

    def _link(self, e):
        href = e.GetLinkInfo().GetHref()
        if href.startswith("#"):
            self.html.ScrollToAnchor(href[1:])

    def show(self, section: str) -> None:
        if section in SECTIONS:
            self.html.ScrollToAnchor(section)
