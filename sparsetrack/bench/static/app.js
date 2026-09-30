"use strict";
// TubeTracker benchmark labelling UI. All judgements are made on registered bin averages.
const $ = (sel) => document.querySelector(sel);
// C cycles the views: normal, high contrast, growth (this bin minus 6 bins earlier: a growing tip stands
// out, still material cancels)
const NEXT_CONTRAST = { n: "h", h: "g", g: "n" };
const CONTRAST_NAMES = { n: "normal", h: "high contrast", g: "growth" };
const S = {
  st: null, view: "census", gid: null,
  step: "coarse", fineStart: 0, fv: null, la: null, coarseTile: null, consulted: new Set(),
  contrast: "n", traceIdx: 0, pts: [], traceView: "near", overlay: true, marker: true,
  contact: false, smooth: 0, fieldWhich: "early",
  timers: {}, openedAt: Date.now(), retest: null, retestIdx: 0, lretest: null, lretestIdx: 0, peek: 0,
  follow: {}, refindArm: false,
};
const EMERGED = ["emerged_within", "emerged_at_start"];
const VERDICT_TEXT = {
  emerged_within: "emerged during movie", emerged_at_start: "already emerged at start",
  no_emergence_by_end: "no emergence by end", unobservable: "can't tell",
};

// ---------------------------------------------------------------- data helpers
async function refresh() { S.st = await (await fetch("/api/state")).json(); }
async function post(path, body) {
  let r;
  try {
    r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" },
                            body: JSON.stringify(body) });
  } catch (err) {  // no server: its Terminal window was closed, or it is restarting
    flash("NOT SAVED: the tool is not running. Start it again, then repeat this answer.", true); throw err;
  }
  const j = await r.json();
  if (!r.ok) { flash("NOT SAVED: " + (j.error || r.status), true); throw new Error(j.error); }
  flash("saved ✓ " + new Date().toLocaleTimeString());
  return j;
}
function flash(msg, err = false) { const el = $("#saved"); el.textContent = msg; el.className = err ? "error" : ""; }
const grain = (gid = S.gid) => S.st.grains[gid];
const label = (gid = S.gid) => S.st.labels[gid] || {};
const plan = (gid = S.gid) => S.st.trace_plan[gid] || [];
const included = () => S.st.order.filter((g) => !S.st.grains[g].excluded);
// review (a file pre-filled with a model's answers): its answers are asked for until a person checks each
const isModel = (rec) => !!(S.st.review && rec && rec.review_origin === "model");
const needsOnset = (g) => !label(g).onset || isModel(label(g).onset);
const needsTrace = (g) => plan(g).some((b) => { const t = (label(g).traces || {})[String(b)]; return !t || isModel(t); });
// how the views follow a grain (server: Bench.follow_info): view offsets per bin, where the tracker lost it, and the
// annotator's own "the grain is here" answers. Fetched once per grain; a saved trace is drawn from its reference
// coordinates less the view's offset now, so it stays on the tube whatever view it was clicked in.
async function ensureFollow(gid = S.gid) {
  if (!gid || S.follow[gid]) return S.follow[gid];
  try { S.follow[gid] = await (await fetch(`/api/follow/${gid}`)).json(); } catch (err) { S.follow[gid] = null; }
  return S.follow[gid];
}
const viewOffset = (b, gid = S.gid) => { const f = S.follow[gid]; return f && f.offsets && f.offsets[b] ? f.offsets[b] : null; };
function viewPts(saved, b) {
  const off = viewOffset(b);
  if (saved.path_xy_ref && saved.path_xy_ref.length && off)
    return saved.path_xy_ref.map((q) => [Math.round((q[0] - off[0]) * 100) / 100, Math.round((q[1] - off[1]) * 100) / 100]);
  return (saved.path_xy_view || saved.path_xy_ref).map((q) => q.slice());
}
const lostAt = (b, gid = S.gid) => ((S.follow[gid] || {}).lost || []).find((l) => b >= l[0] && b < l[1]);
const frameOfBin = (b) => b * S.st.frames_per_bin + Math.floor(S.st.frames_per_bin / 2);
// time on a grain accumulates across sessions: start from what was saved for it
function spent() { return (S.timers[S.gid] ?? label().time_spent_s ?? 0) + (Date.now() - S.openedAt) / 1000; }

function resetGrainState() {
  S.step = "coarse"; S.fv = null; S.la = null; S.coarseTile = null; S.consulted = new Set();
  const p = plan();
  const tr = label().traces || {};
  const firstOpen = p.findIndex((b) => !tr[String(b)] || isModel(tr[String(b)]));
  S.traceIdx = firstOpen >= 0 ? firstOpen : 0;
  const on = label().onset;
  if (isModel(on) && on.verdict === "emerged_within") {  // review: open on the model's bracket; Enter confirms it
    S.step = "fine"; S.fv = on.first_visible_bin; S.la = on.last_absent_bin ?? null;
    S.fineStart = Math.max(0, Math.min(on.first_visible_bin - 8, S.st.n_bins - S.st.layout.fine.n_tiles));
  }
  loadTrace();
}
function loadTrace() {
  const b = plan()[S.traceIdx];
  const saved = b === undefined ? null : (label().traces || {})[String(b)];
  S.pts = saved ? viewPts(saved, b) : [];
  S.contact = saved ? !!saved.contact : false;
}
async function openGrain(gid, view) {
  if (S.gid) S.timers[S.gid] = spent();
  S.gid = gid; S.openedAt = Date.now(); S.refindArm = false;
  if (view) S.view = view;
  await ensureFollow(gid);
  if (S.gid !== gid) return;  // another grain was opened meanwhile
  resetGrainState(); render();
}
function stepGrain(dir) {
  const list = S.view === "census" ? S.st.order : included();
  let i = list.indexOf(S.gid);
  i = (i + dir + list.length) % list.length;
  openGrain(list[i]);
}
function nextPending() {
  const list = included();
  const start = Math.max(0, list.indexOf(S.gid));
  for (let k = 1; k <= list.length; k++) {
    const g = list[(start + k) % list.length];
    if (needsOnset(g) || needsTrace(g)) return openGrain(g, needsOnset(g) ? "onset" : "trace");
  }
  S.view = "review"; render();
}

// ---------------------------------------------------------------- chrome
function renderChrome() {
  const p = S.st.progress;
  $("#movie").textContent = S.st.movie.name;
  $("#progress").textContent = S.st.review
    ? `review of ${S.st.review.model || "the model"}: onsets checked ${p.onset_checked}/${p.grains} · traces checked ${p.traces_checked}/${p.traces_needed}`
    : `onset ${p.onset_done}/${p.grains} grains · traces ${p.traces_done}/${p.traces_needed}`;
  document.querySelectorAll("nav button").forEach((b) => b.classList.toggle("active", b.dataset.view === S.view));
  const bar = $("#grainbar");
  if (!S.gid || S.view === "review") { bar.innerHTML = ""; return; }
  const g = grain(); const on = label().onset;
  const chips = [];
  chips.push(g.isolated ? `<span class="chip ok">isolated</span>` : "");
  if (g.clump_size > 1) chips.push(`<span class="chip warn">clump of ${g.clump_size}</span>`);
  if (g.border) chips.push(`<span class="chip warn">near edge</span>`);
  if (g.source === "user") chips.push(`<span class="chip">added by you</span>`);
  const fol = S.follow[S.gid];
  if (fol) {  // where the tracker lost the grain (the view stays where it was last seen), and the annotator's answers
    for (const [b0, b1, why] of fol.lost || [])
      chips.push(`<span class="chip bad" title="the view stays where the grain was last seen${why === "frame" ? " (it left the movie)" : ""}">tracker lost it: bin ${b0}${b1 < S.st.n_bins ? `–${b1 - 1}` : " on"}</span>`);
    const mine = (fol.refinds || []).map((r) => r.bin);
    if (mine.length) chips.push(`<span class="chip ok">you placed it at bin ${mine.join(", ")}</span>`);
  }
  if (g.excluded) chips.push(`<span class="chip bad">excluded: ${g.exclude_reason}</span>`);
  const blind = S.view === "retest" || S.view === "lretest";
  if (on && !blind) {
    let t = VERDICT_TEXT[on.verdict];
    if (on.verdict === "emerged_within") t += ` (${on.last_absent_frame ?? "?"}, ${on.first_visible_frame}]`;
    chips.push(`<span class="chip ok">${t}</span>`);
  }
  const pos = S.view === "retest" ? `retest ${S.retestIdx + 1}/${(S.retest || []).length}`
    : S.view === "lretest" ? `length retest ${S.lretestIdx + 1}/${(S.lretest || []).length}`
    : `${included().indexOf(S.gid) + 1}/${included().length}`;
  bar.innerHTML = `<button class="act" id="prevg">◀ grain</button>
    <span class="gid">${blind ? "retest" : S.gid}</span>
    <button class="act" id="nextg">grain ▶</button>
    <button class="act" id="pend">next unfinished ⏭</button>
    <span class="muted">${pos} · centre (${g.x.toFixed(0)}, ${g.y.toFixed(0)}) · r ${g.r.toFixed(1)} px</span>
    ${blind ? "" : chips.join(" ")}`;
  if (S.view === "lretest") {
    $("#prevg").onclick = () => { S.lretestIdx = Math.max(0, S.lretestIdx - 1); openLengthRetest(); };
    $("#nextg").onclick = () => { S.lretestIdx = Math.min(S.lretest.length - 1, S.lretestIdx + 1); openLengthRetest(); };
    $("#pend").style.display = "none";
  } else if (S.view === "retest") {
    $("#prevg").onclick = () => { S.retestIdx = Math.max(0, S.retestIdx - 1); openRetest(); };
    $("#nextg").onclick = () => { S.retestIdx = Math.min(S.retest.length - 1, S.retestIdx + 1); openRetest(); };
    $("#pend").style.display = "none";
  } else {
    $("#prevg").onclick = () => stepGrain(-1);
    $("#nextg").onclick = () => stepGrain(+1);
    $("#pend").onclick = nextPending;
  }
}
function help(html) { $("#help").innerHTML = html; }
function render() {
  $("#menu").classList.add("hidden");
  $("#content").onclick = null;
  renderChrome();
  ({ census: renderCensus, onset: () => renderOnset(false), trace: renderTrace,
     review: renderReview, retest: renderRetestView, lretest: renderLengthRetest })[S.view]();
}

// ---------------------------------------------------------------- 1. grain census
const RING = { included: "#4ade80", excluded: "#f87171", user: "#38bdf8", selected: "#facc15" };
const ringColor = (g) => (g.excluded ? RING.excluded : g.source === "user" ? RING.user : RING.included);
function censusCentre() {
  // the close-up follows the grain in focus, unless empty space on the map was clicked
  if (!S.zoomAt || S.zoomAt.gid !== S.gid) { const g = grain(); S.zoomAt = { x: g.x, y: g.y, gid: S.gid }; }
  return S.zoomAt;
}
async function focusGrain(gid, keepView) {
  if (S.gid !== gid) {
    S.timers[S.gid] = spent(); S.gid = gid; S.openedAt = Date.now(); S.refindArm = false;
    await ensureFollow(gid);
    resetGrainState();
  }
  if (keepView && S.zoomAt) S.zoomAt.gid = gid;
  render();
}
function renderCensus() {
  help(`<b>Click any grain</b> on the map, or step with <kbd>←</kbd>/<kbd>→</kbd>, to bring it into focus: the close-up
    on the right shows it large. <b>In the close-up</b>, click the exact grain you mean (where grains overlap), then use
    the buttons under it to exclude it (clump member, edge, not a grain) or include it again; click an
    <b>uncircled</b> grain there to add it. Clicking empty space on the map moves the close-up there.
    <kbd>E</kbd> early/late.`);
  const c = $("#content");
  const dot = (col, t) => `<span class="legend"><i style="border-color:${col}"></i>${t}</span>`;
  c.innerHTML = `<div><button class="act ${S.fieldWhich === "early" ? "on" : ""}" id="fe">Early (no tubes)</button>
    <button class="act ${S.fieldWhich === "late" ? "on" : ""}" id="fl">Late (end of movie)</button>
    <span class="muted">${included().length} grains included · ${S.st.order.length - included().length} excluded</span>
    ${dot(RING.included, "included")}${dot(RING.excluded, "excluded ✕")}${dot(RING.user, "added by you")}${dot(RING.selected, "in focus")}</div>
    <div class="row wrapping"><div class="wrap"><canvas id="field"></canvas></div>
      <div class="zoomside"><div class="wrap"><canvas id="zoom"></canvas></div><div id="zoomctl"></div></div></div>`;
  $("#fe").onclick = () => { S.fieldWhich = "early"; render(); };
  $("#fl").onclick = () => { S.fieldWhich = "late"; render(); };
  S._field = S._field || {};
  if (S._field[S.fieldWhich]) drawField();
  else {
    const img = new Image();
    img.onload = () => { S._field[S.fieldWhich] = img; drawField(); };
    img.src = `/api/img/field?which=${S.fieldWhich}`;
  }
  drawZoom(); renderZoomControls();
}
// every mark is drawn twice, a dark halo then the colour, so it reads on light background and dark grains
function haloRing(ctx, x, y, r, color, width, dash) {
  ctx.setLineDash(dash || []);
  ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.strokeStyle = "rgba(0,0,0,0.7)"; ctx.lineWidth = width + 3; ctx.stroke();
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke();
  ctx.setLineDash([]);
}
function haloCross(ctx, x, y, d, color) {
  for (const [w, col] of [[5, "rgba(0,0,0,0.7)"], [2.5, color]]) {
    ctx.beginPath(); ctx.moveTo(x - d, y - d); ctx.lineTo(x + d, y + d);
    ctx.moveTo(x + d, y - d); ctx.lineTo(x - d, y + d);
    ctx.strokeStyle = col; ctx.lineWidth = w; ctx.stroke();
  }
}
function haloText(ctx, text, x, y, color, bold) {
  ctx.font = bold ? "bold 13px sans-serif" : "bold 11px sans-serif";
  ctx.lineJoin = "round"; ctx.lineWidth = 3.5; ctx.strokeStyle = "rgba(0,0,0,0.8)"; ctx.strokeText(text, x, y);
  ctx.fillStyle = color; ctx.fillText(text, x, y);
}
function drawField() {
  const img = (S._field || {})[S.fieldWhich], cv = $("#field");
  if (!img || !cv) return;
  cv.width = img.width; cv.height = img.height;
  const ctx = cv.getContext("2d"); ctx.drawImage(img, 0, 0);
  const s = img.width / S.st.movie.width;
  for (const gid of S.st.order) {
    const g = S.st.grains[gid];
    const x = g.x * s, y = g.y * s, r = g.r * s + 5;
    haloRing(ctx, x, y, r, ringColor(g), 2.5, g.excluded ? [5, 4] : null);
    if (g.excluded) haloCross(ctx, x, y, r * 0.55, RING.excluded);
    if (gid === S.gid) haloRing(ctx, x, y, r + 6, RING.selected, 3.5);
    haloText(ctx, gid, x + r + 4, y - r + 2, gid === S.gid ? RING.selected : ringColor(g), gid === S.gid);
  }
  const Z = S.st.layout.zoom, at = censusCentre();  // where the close-up is looking
  ctx.setLineDash([6, 4]); ctx.strokeStyle = RING.selected; ctx.lineWidth = 2;
  ctx.strokeRect((at.x - Z.half) * s, (at.y - Z.half) * s, 2 * Z.half * s, 2 * Z.half * s); ctx.setLineDash([]);
  cv.onclick = (e) => {
    const x = e.offsetX / s, y = e.offsetY / s;
    let best = null, bd = 1e9;
    for (const gid of S.st.order) {
      const g = S.st.grains[gid]; const d = Math.hypot(g.x - x, g.y - y);
      if (d < bd) { bd = d; best = gid; }
    }
    if (best && bd <= S.st.grains[best].r + 8 / s) return focusGrain(best, false);
    S.zoomAt = { x, y, gid: S.gid };  // look here; pick or add grains in the close-up
    drawField(); drawZoom();
  };
}
function drawZoom() {
  const Z = S.st.layout.zoom, at = censusCentre();
  const key = `${S.fieldWhich}:${at.x.toFixed(1)}:${at.y.toFixed(1)}`;
  S._zoom = S._zoom || {};
  const img = S._zoom[key];
  if (!img) {
    const im = new Image();
    im.onload = () => { S._zoom[key] = im; drawZoom(); };
    im.src = `/api/img/zoom?x=${at.x}&y=${at.y}&which=${S.fieldWhich}`;
    return;
  }
  const cv = $("#zoom"); if (!cv) return;
  cv.width = img.width; cv.height = img.height;
  const ctx = cv.getContext("2d"); ctx.drawImage(img, 0, 0);
  const toC = (x, y) => [(x - at.x + Z.half) * Z.zoom, (y - at.y + Z.half) * Z.zoom];
  for (const gid of S.st.order) {
    const g = S.st.grains[gid];
    if (Math.abs(g.x - at.x) > Z.half + g.r + 4 || Math.abs(g.y - at.y) > Z.half + g.r + 4) continue;
    const [x, y] = toC(g.x, g.y), r = g.r * Z.zoom + 6;
    haloRing(ctx, x, y, r, ringColor(g), 3, g.excluded ? [8, 6] : null);
    if (g.excluded) haloCross(ctx, x, y, r * 0.5, RING.excluded);
    if (gid === S.gid) haloRing(ctx, x, y, r + 8, RING.selected, 4);
    haloText(ctx, gid, x + r * 0.72 + 4, y - r * 0.72, gid === S.gid ? RING.selected : ringColor(g), true);
  }
  cv.onclick = (e) => {
    const x = at.x - Z.half + e.offsetX / Z.zoom, y = at.y - Z.half + e.offsetY / Z.zoom;
    let best = null, bd = 1e9;  // the grain whose circle holds the click (nearest centre if several)
    for (const gid of S.st.order) {
      const g = S.st.grains[gid]; const d = Math.hypot(g.x - x, g.y - y);
      if (d <= g.r + 2 && d < bd) { bd = d; best = gid; }
    }
    if (best) return focusGrain(best, true);
    if (confirm(`Add a grain centred here (${x.toFixed(0)}, ${y.toFixed(0)})?`)) {
      post("/api/grain", { x, y }).then(refresh).then(() => {
        const added = S.st.order.filter((g) => S.st.grains[g].source === "user")
          .sort((a, b) => Math.hypot(S.st.grains[a].x - x, S.st.grains[a].y - y) - Math.hypot(S.st.grains[b].x - x, S.st.grains[b].y - y))[0];
        if (added) focusGrain(added, true); else render();
      });
    }
  };
}
function renderZoomControls() {
  const g = grain(), gid = S.gid, el = $("#zoomctl");
  if (!el) return;
  const layout = [g.isolated ? "isolated" : "", g.clump_size > 1 ? `clump of ${g.clump_size}` : "",
                  g.border ? "near edge" : "", g.source === "user" ? "added by you" : ""].filter(Boolean).join(" · ");
  const reasons = S.st.exclude_reasons.filter((r) => r !== "not_sampled")
    .map((r) => `<button class="act" data-r="${r}">${r.replaceAll("_", " ")}</button>`).join("");
  el.innerHTML = `<h3>${gid} <span class="muted">${layout}</span></h3>
    ${g.excluded
      ? `<p><b class="bad">excluded: ${g.exclude_reason.replaceAll("_", " ")}</b> <button class="act" data-inc="1">include again</button></p>`
      : `<p>exclude as: ${reasons}</p>`}
    <p><button class="act primary" data-open="1">open onset ▶</button></p>`;
  el.onclick = async (e) => {
    const b = e.target.closest("button"); if (!b) return;
    if (b.dataset.open) return openGrain(gid, "onset");
    if (b.dataset.inc) await post(`/api/exclude/${gid}`, { excluded: false });
    let j = null;
    if (b.dataset.r) j = await post(`/api/exclude/${gid}`, { excluded: true, reason: b.dataset.r });
    await refresh(); render();
    if (j && j.replaced_by) flash(`saved ✓ ${j.replaced_by} joins the random sample in ${gid}'s place`);
  };
}
function grainMenu(px, py, gid) {
  const g = S.st.grains[gid]; const m = $("#menu");
  const reasons = S.st.exclude_reasons.filter((r) => r !== "not_sampled")
    .map((r) => `<button class="act" data-r="${r}">exclude: ${r.replaceAll("_", " ")}</button>`).join("");
  m.innerHTML = `<b>${gid}</b> ${g.excluded ? "(excluded)" : ""}<br>
    <button class="act primary" data-open="1">open onset</button>
    ${g.excluded ? `<button class="act" data-inc="1">include again</button>` : reasons}
    <button class="act" data-close="1">close</button>`;
  m.style.left = px + 8 + "px"; m.style.top = py + 8 + "px"; m.classList.remove("hidden");
  m.onclick = async (e) => {
    const b = e.target.closest("button"); if (!b) return;
    if (b.dataset.open) return openGrain(gid, "onset");
    if (b.dataset.close) return m.classList.add("hidden");
    if (b.dataset.inc) await post(`/api/exclude/${gid}`, { excluded: false });
    let j = null;
    if (b.dataset.r) j = await post(`/api/exclude/${gid}`, { excluded: true, reason: b.dataset.r });
    await refresh(); render();
    if (j && j.replaced_by) flash(`saved ✓ ${j.replaced_by} joins the random sample in ${gid}'s place`);
  };
}

// ---------------------------------------------------------------- 2. onset (and retest)
function tileGeom(lay) {
  const side = Math.round(2 * lay.half * lay.zoom);
  return { side, cw: side + lay.gap, ch: side + lay.header + lay.gap, header: lay.header };
}
function tileAt(e, lay, nTiles) {
  const t = tileGeom(lay);
  const col = Math.floor(e.offsetX / t.cw), row = Math.floor(e.offsetY / t.ch);
  if (col >= lay.cols || e.offsetX - col * t.cw > t.side) return -1;
  const idx = row * lay.cols + col;
  return idx < nTiles ? idx : -1;
}
function boxAt(idx, lay, cls) {
  const t = tileGeom(lay);
  const r = Math.floor(idx / lay.cols), c = idx % lay.cols;
  return `<div class="box ${cls}" style="left:${c * t.cw}px;top:${r * t.ch + t.header}px;width:${t.side}px;height:${t.side}px"></div>`;
}
function renderOnset(retest) {
  const L = S.st.layout, gid = S.gid, g = grain();
  const prior = retest ? null : label().onset;
  const c = $("#content");
  if (g.excluded && !retest) {
    help(`This grain is excluded (${g.exclude_reason}). Include it again from the Grains view, or move on.`);
    c.innerHTML = `<button class="act primary" id="np">next unfinished ⏭</button>`;
    $("#np").onclick = nextPending; return;
  }
  const nBins = S.st.n_bins, bpt = L.coarse.bins_per_tile;
  if (S.step === "coarse") {
    const nTiles = Math.ceil(nBins / bpt);
    help(`The grain in the <b>centre</b> of each tile is the one being judged; each tile averages
      ${bpt * S.st.frames_per_bin} frames (label = first frame). <b>Click the first tile where a tube is
      clearly visible growing out of this grain.</b> Or: <kbd>N</kbd> no emergence by the end ·
      <kbd>S</kbd> already emerged at the start · <kbd>U</kbd> can't tell · <kbd>X</kbd> exclude ·
      <kbd>C</kbd> view (${CONTRAST_NAMES[S.contrast]}) · <kbd>←</kbd>/<kbd>→</kbd> grain.`);
    let boxes = "";
    if (prior && prior.first_visible_bin != null && prior.verdict === "emerged_within")
      boxes = boxAt(Math.floor(prior.first_visible_bin / bpt), L.coarse, "prior");
    c.innerHTML = `<div class="wrap" id="cw"><img id="coarse" src="/api/img/coarse/${gid}?contrast=${S.contrast}">${boxes}</div>
      <div style="margin-top:8px">
      <button class="act" id="vN">N · no emergence by end</button><button class="act" id="vS">S · emerged at start</button>
      <button class="act" id="vU">U · can't tell</button><button class="act" id="vX">X · exclude…</button>
      ${prior ? `<span class="muted">${isModel(prior) ? "the model's call" : "saved"}: ${VERDICT_TEXT[prior.verdict]}${isModel(prior) ? " — <kbd>Enter</kbd> confirms it" : ""} — dashed blue = previous first-visible tile</span>` : ""}</div>`;
    $("#coarse").onclick = (e) => {
      const idx = tileAt(e, L.coarse, nTiles); if (idx < 0) return;
      S.coarseTile = idx;
      for (let b = idx * bpt; b < Math.min(nBins, (idx + 1) * bpt); b++) S.consulted.add(b);
      S.fineStart = Math.max(0, Math.min(idx * bpt - 8, nBins - L.fine.n_tiles));
      S.fv = null; S.la = null; S.step = "fine"; render();
    };
    $("#vN").onclick = () => saveOnset("no_emergence_by_end", retest);
    $("#vS").onclick = () => saveOnset("emerged_at_start", retest);
    $("#vU").onclick = () => saveOnset("unobservable", retest);
    $("#vX").onclick = (e) => grainMenu(e.pageX, e.pageY, gid);
    return;
  }
  // fine step
  const n = Math.min(L.fine.n_tiles, nBins - S.fineStart);
  help(`Single ${S.st.frames_per_bin}-frame bins. <b>Click the first bin where the tube is clearly
    visible</b> (green). Optionally <b>shift-click</b> the last bin where it is clearly absent (orange);
    otherwise the bin before is used. Unclear bins between them stay unresolved (yellow).
    <kbd>Enter</kbd> save · <kbd>[</kbd>/<kbd>]</kbd> earlier/later · <kbd>B</kbd> back · <kbd>C</kbd> contrast.`);
  let boxes = "";
  for (let i = 0; i < n; i++) {
    const b = S.fineStart + i;
    if (b === S.fv) boxes += boxAt(i, L.fine, "visible");
    else if (b === S.la) boxes += boxAt(i, L.fine, "absent");
    else if (S.fv !== null && S.la !== null && b > S.la && b < S.fv) boxes += boxAt(i, L.fine, "between");
  }
  if (!retest && isModel(prior) && prior.first_visible_bin === S.fv)
    $("#help").insertAdjacentHTML("afterbegin", `<b>Review:</b> the model's bracket is marked. <kbd>Enter</kbd>
      confirms it; or click the bins you see and save.<br>`);
  const fvText = S.fv === null ? "—" : `bin ${S.fv}`;
  const laText = S.la === null ? (S.fv === null ? "—" : `bin ${S.fv - 1} (default)`) : `bin ${S.la}`;
  c.innerHTML = `<div class="wrap"><img id="fine" src="/api/img/fine/${gid}?start=${S.fineStart}&contrast=${S.contrast}">${boxes}</div>
    <div style="margin-top:8px">first clearly visible: <b>${fvText}</b> · last clearly absent: <b>${laText}</b>
    <button class="act" id="fe">◀ earlier</button><button class="act" id="fl">later ▶</button>
    <button class="act" id="fb">back to whole movie</button>
    <button class="act primary" id="fs" ${S.fv === null ? "disabled" : ""}>save (Enter)</button></div>`;
  $("#fine").onclick = (e) => {
    const idx = tileAt(e, L.fine, n); if (idx < 0) return;
    const b = S.fineStart + idx; S.consulted.add(b);
    if (e.shiftKey) {
      if (S.fv !== null && b >= S.fv) { alert("The last-absent bin must come before the first-visible bin."); return; }
      S.la = b;
    } else {
      S.fv = b;
      if (S.la !== null && S.la >= b) S.la = null;
    }
    render();
  };
  $("#fe").onclick = () => shiftFine(-12);
  $("#fl").onclick = () => shiftFine(+12);
  $("#fb").onclick = () => { S.step = "coarse"; render(); };
  $("#fs").onclick = () => saveOnset("emerged_within", retest);
}
function shiftFine(d) {
  const L = S.st.layout;
  S.fineStart = Math.max(0, Math.min(S.fineStart + d, S.st.n_bins - L.fine.n_tiles)); render();
}
async function saveOnset(verdict, retest) {
  const body = { verdict, coarse_tile: S.coarseTile, consulted_bins: [...S.consulted], time_spent_s: spent() };
  if (verdict === "emerged_within") {
    if (S.fv === null) return;
    body.first_visible_bin = S.fv;
    body.last_absent_bin = S.la === null ? (S.fv > 0 ? S.fv - 1 : null) : S.la;
    if (S.fv === 0) { body.verdict = "emerged_at_start"; }
  }
  await post(`/api/${retest ? "retest" : "onset"}/${S.gid}`, body);
  await refresh();
  if (retest) { S.retestIdx += 1; return openRetest(); }
  if (EMERGED.includes(body.verdict)) { resetGrainState(); S.view = "trace"; render(); }
  else nextPending();
}

// ---------------------------------------------------------------- 3. traces
function renderTrace() {
  const c = $("#content"), p = plan();
  const on = label().onset;
  S._img = null;
  if (!p.length) {
    help(on ? `No traces needed for this grain (${VERDICT_TEXT[on.verdict]}).` : `Label this grain's onset first.`);
    c.innerHTML = `<button class="act primary" id="np">next unfinished ⏭</button>
      ${on ? "" : `<button class="act" id="go">go to onset</button>`}`;
    $("#np").onclick = nextPending;
    if (!on) $("#go").onclick = () => { S.view = "onset"; render(); };
    return;
  }
  const b = p[S.traceIdx]; const V = S.st.layout.trace[S.traceView];
  const size = Math.round(2 * V.half * V.zoom);
  const tr = label().traces || {};
  help(`Trace the <b>whole visible tube of the centre grain</b>: click its exit from the grain first,
    then follow the centreline to the apex (as many clicks as the curve needs).
    <kbd>F</kbd> full tube · <kbd>P</kbd> partial (part hidden/out of view) · <kbd>0</kbd> no tube ·
    <kbd>U</kbd> unsure · <kbd>⌫</kbd> undo point · <kbd>T</kbd> touching another tube/grain ·
    ${canBurst() ? "<kbd>B</kbd> burst (nothing left to trace; its later times are skipped) · " : ""}
    <kbd>W</kbd> wide/near · ${S.st.layout.trace.far ? "<kbd>X</kbd> extra wide · " : ""}<kbd>H</kbd> hide marks ·
    <kbd>C</kbd> contrast · <kbd>A</kbd> smoother ·
    <kbd>←</kbd>/<kbd>→</kbd> trace time.`);
  const mark = (t) => !t ? "" : isModel(t) ? " ? model" : t.state === "burst" ? " ✓ burst" : " ✓";
  const chips = p.map((bb, i) => `<span data-i="${i}" class="${tr[String(bb)] && !isModel(tr[String(bb)]) ? "done" : ""} ${i === S.traceIdx ? "cur" : ""}">f${bb * S.st.frames_per_bin + S.st.frames_per_bin / 2}${mark(tr[String(bb)])}</span>`).join("");
  const saved = tr[String(b)];
  const canPlace = S.st.follow_mode !== "review";
  if (canPlace) $("#help").insertAdjacentHTML("beforeend", ` · <kbd>G</kbd> then click: the grain is here
    (the views follow it from this time; <kbd>Shift</kbd>+<kbd>G</kbd> takes it back)`);
  const lost = lostAt(b);
  if (lost) $("#help").insertAdjacentHTML("afterbegin", `<b class="bad">The tracker lost this grain at bin ${lost[0]}
    (frame ${frameOfBin(lost[0])})${lost[2] === "frame" ? ", where it left the movie" : ""}: the view stays where it was
    last seen.</b> ${canPlace ? "If you can see the grain elsewhere (<kbd>X</kbd>: extra wide), press <kbd>G</kbd> and click its centre." : ""}<br>`);
  if (S.refindArm) $("#help").insertAdjacentHTML("afterbegin", `<b>Click the grain's centre</b> (Esc: cancel).<br>`);
  if (isModel(saved)) $("#help").insertAdjacentHTML("afterbegin", `<b>Review:</b> this is the model's answer
    (${saved.state === "full" ? `a ${saved.length_px} px tube` : saved.state.replace("_", " ")}${saved.model_confidence != null ?
    `; model confidence ${Math.round(100 * saved.model_confidence)}%` : ""}). <kbd>Enter</kbd> confirms it as it
    stands; ${saved.model_path ? "<kbd>-</kbd>/<kbd>=</kbd> shorten/lengthen it along the model's route (Shift: 5 px), then <kbd>F</kbd>; " : ""}or fix it and save as usual.<br>`);
  c.innerHTML = `<div class="row"><div class="wrap"><canvas id="tc" width="${size}" height="${size}"></canvas></div>
    <div class="side"><h3>Trace ${S.traceIdx + 1} of ${p.length} · frame ${b * S.st.frames_per_bin + S.st.frames_per_bin / 2} (bin ${b})</h3>
    <div class="chips" id="chips">${chips}</div>
    <p>${S.pts.length} point(s), ${traceLen().toFixed(1)} px ${saved ? `<br><span class="muted">saved: ${saved.state}, ${saved.length_px} px</span>` : ""}</p>
    <button class="act primary" id="sF">F · full tube</button><button class="act" id="sP">P · partial</button><br>
    <button class="act" id="s0">0 · no tube</button><button class="act" id="sU">U · unsure</button>
    ${canBurst() ? `<button class="act" id="sB">B · burst</button>` : ""}<br>
    <button class="act" id="undo">⌫ undo</button><button class="act" id="clr">clear</button><br>
    <button class="act ${S.contact ? "on" : ""}" id="tT">T · touching other tube/grain</button><br>
    <button class="act ${S.traceView === "wide" ? "on" : ""}" id="tW">W · wide view</button>
    ${S.st.layout.trace.far ? `<button class="act ${S.traceView === "far" ? "on" : ""}" id="tX">X · extra wide</button>` : ""}
    <button class="act ${S.contrast !== "n" ? "on" : ""}" id="tC" title="normal, high contrast, or growth: this bin minus 6 bins earlier (a growing tip stands out)">C · ${CONTRAST_NAMES[S.contrast]}</button>
    <button class="act ${S.smooth ? "on" : ""}" id="tA">A · smoother (3 bins)</button>
    ${canPlace ? `<br><button class="act ${S.refindArm ? "on" : ""}" id="tG" title="then click the grain's centre: the views follow it from this time">G · the grain is here…</button>` : ""}
    <p class="muted">Onset: ${on ? VERDICT_TEXT[on.verdict] : "—"} ${on && on.first_visible_frame != null ? "· first visible f" + on.first_visible_frame : ""}</p></div></div>`;
  mountTraceCanvas(b);
  $("#chips").onclick = (e) => { const s = e.target.closest("span"); if (s) { S.traceIdx = +s.dataset.i; loadTrace(); render(); } };
  wireTraceControls(saveTrace);
  if ($("#tG")) $("#tG").onclick = () => { S.refindArm = !S.refindArm; render(); };
}
// "the grain is here": its centre clicked in this bin's view; the tool follows it from there (labelling only)
async function placeGrain(b, x, y, clear = false) {
  const before = viewOffset(b);
  const f = await post(`/api/refind/${S.gid}`, clear ? { bin: b, clear: true } : { bin: b, x, y });
  S.follow[S.gid] = f;
  const after = viewOffset(b);
  if (before && after)  // points clicked so far stay on the tube: from the old view to the new one
    S.pts = S.pts.map((q) => [q[0] + before[0] - after[0], q[1] + before[1] - after[1]]);
  flash(clear ? "saved ✓ your answer at this time was taken back" : "saved ✓ the views follow the grain from here");
  render();
}
function mountTraceCanvas(b, clickable = true) {  // bin b in the current view; clicks add points in movie coordinates
  const V = S.st.layout.trace[S.traceView], [cx, cy] = viewCentre(V);
  const cv = $("#tc"), img = new Image();
  img.onload = () => { S._img = img; drawTrace(); };
  img.src = `/api/img/frame/${S.gid}?bin=${b}&view=${S.traceView}&contrast=${S.contrast}&smooth=${S.smooth}&cx=${cx}&cy=${cy}`;
  cv.onclick = (e) => {
    if (!clickable) { flash("Viewing another time: go back to the trace frame to click.", true); return; }
    const x = cx - V.half + e.offsetX / V.zoom, y = cy - V.half + e.offsetY / V.zoom;
    if (S.refindArm && S.view === "trace") { S.refindArm = false; placeGrain(b, x, y); return; }
    S.pts.push([x, y]); drawTrace(); updateCount();
  };
}
function wireTraceControls(save) {
  $("#sF").onclick = () => save("full");
  $("#sP").onclick = () => save("partial");
  $("#s0").onclick = () => save("no_tube");
  $("#sU").onclick = () => save("unsure");
  if ($("#sB")) $("#sB").onclick = () => save("burst");
  $("#undo").onclick = () => { S.pts.pop(); drawTrace(); updateCount(); };
  $("#clr").onclick = () => { S.pts = []; drawTrace(); updateCount(); };
  $("#tT").onclick = () => { S.contact = !S.contact; render(); };
  $("#tW").onclick = toggleWide;
  if ($("#tX")) $("#tX").onclick = toggleFar;
  $("#tC").onclick = () => { S.contrast = NEXT_CONTRAST[S.contrast] || "n"; render(); };
  $("#tA").onclick = () => { S.smooth = S.smooth ? 0 : 1; render(); };
}
// clicked points are in movie coordinates, so a trace can be continued in another view
function viewCentre(V) {  // a "fit" view slides inward at the movie's edges; image and clicks share this centre
  const g = grain(), m = S.st.movie;
  if (!V.fit) return [g.x, g.y];
  const slide = (c, size) => Math.min(Math.max(c, V.half), size - V.half);
  return [slide(g.x, m.width), slide(g.y, m.height)];
}
function canBurst() { return S.st.trace_states.includes("burst"); }  // an older server would refuse it
function toggleWide() { S.traceView = S.traceView === "wide" ? "near" : "wide"; render(); }
function toggleFar() { if (S.st.layout.trace.far) { S.traceView = S.traceView === "far" ? "wide" : "far"; render(); } }
function traceLen() {
  let L = 0; for (let i = 1; i < S.pts.length; i++) L += Math.hypot(S.pts[i][0] - S.pts[i - 1][0], S.pts[i][1] - S.pts[i - 1][1]);
  return L;
}
// review: the trace's apex slid along the model's whole route (kept with a pre-filled trace), d px at a time
function slideTrace(d) {
  const b = plan()[S.traceIdx], saved = (label().traces || {})[String(b)];
  const mp = saved && saved.model_path;
  if (!mp || mp.length < 2) return;
  let want = Math.max(0, traceLen() + d), acc = 0;
  const pts = [mp[0].slice()];
  for (let i = 1; i < mp.length && want > 0; i++) {
    const seg = Math.hypot(mp[i][0] - mp[i - 1][0], mp[i][1] - mp[i - 1][1]);
    if (acc + seg >= want) {
      const a = (want - acc) / Math.max(seg, 1e-9);
      pts.push([mp[i - 1][0] + a * (mp[i][0] - mp[i - 1][0]), mp[i - 1][1] + a * (mp[i][1] - mp[i - 1][1])]);
      want = 0; break;
    }
    acc += seg; pts.push(mp[i].slice());
  }
  S.pts = pts.map((q) => [Math.round(q[0] * 100) / 100, Math.round(q[1] * 100) / 100]);
  drawTrace(); updateCount();
}
function updateCount() {
  const p = document.querySelector(".side p"); if (p) p.firstChild.textContent = `${S.pts.length} point(s), ${traceLen().toFixed(1)} px `;
}
function drawTrace() {
  const cv = $("#tc"); if (!cv || !S._img) return;
  const ctx = cv.getContext("2d"), g = grain(), V = S.st.layout.trace[S.traceView], [cx, cy] = viewCentre(V);
  ctx.drawImage(S._img, 0, 0);
  if (!S.overlay) return;
  const toC = (x, y) => [(x - cx + V.half) * V.zoom, (y - cy + V.half) * V.zoom];
  if (S.marker) {  // four short ticks just outside the grain, clear of its rim
    ctx.strokeStyle = "rgba(34,211,238,0.8)"; ctx.lineWidth = 2;
    for (const a of [0.25, 0.75, 1.25, 1.75]) {
      const r0 = (g.r + 9) * V.zoom, r1 = (g.r + 13) * V.zoom;
      const [cx, cy] = toC(g.x, g.y);
      ctx.beginPath(); ctx.moveTo(cx + r0 * Math.cos(a * Math.PI), cy + r0 * Math.sin(a * Math.PI));
      ctx.lineTo(cx + r1 * Math.cos(a * Math.PI), cy + r1 * Math.sin(a * Math.PI)); ctx.stroke();
    }
  }
  if (!S.pts.length) return;
  ctx.strokeStyle = "rgba(250,204,21,0.9)"; ctx.lineWidth = 1.5; ctx.beginPath();
  S.pts.forEach((p, i) => { const [u, v] = toC(p[0], p[1]); i ? ctx.lineTo(u, v) : ctx.moveTo(u, v); });
  ctx.stroke();
  S.pts.forEach((p, i) => {
    const [u, v] = toC(p[0], p[1]);
    ctx.fillStyle = i === 0 ? "#ef4444" : i === S.pts.length - 1 ? "#22c55e" : "rgba(250,204,21,0.9)";
    ctx.beginPath(); ctx.arc(u, v, i === 0 || i === S.pts.length - 1 ? 3.5 : 2, 0, 2 * Math.PI); ctx.fill();
  });
}
async function saveTrace(state) {
  const b = plan()[S.traceIdx];
  if ((state === "full" || state === "partial") && S.pts.length < 2) {
    alert("Click at least the exit point and the apex before saving a traced tube."); return;
  }
  await post(`/api/trace/${S.gid}`, { bin: b, state, points: S.pts, contact: S.contact,
                                      view: S.traceView, time_spent_s: spent() });
  await refresh();
  const tr = label().traces || {};
  const nextIdx = plan().findIndex((bb) => !tr[String(bb)] || isModel(tr[String(bb)]));
  if (nextIdx >= 0) { S.traceIdx = nextIdx; loadTrace(); render(); } else nextPending();
}

// ---------------------------------------------------------------- review
function renderReview() {
  help(`All grains. Click a row to open it. Labels are saved to <code>${S.st.labels_path}</code> after every answer.`);
  const rows = S.st.order.map((gid) => {
    const g = S.st.grains[gid], lab = S.st.labels[gid] || {}, on = lab.onset;
    const p = plan(gid), tr = lab.traces || {};
    const done = p.filter((b) => tr[String(b)]).length;
    const flags = [g.isolated ? "isolated" : "", g.clump_size > 1 ? `clump ${g.clump_size}` : "", g.border ? "edge" : "",
                   g.source === "user" ? "added" : ""].filter(Boolean).join(", ");
    const bracket = on && on.verdict === "emerged_within" ? `(${on.last_absent_frame ?? "?"}, ${on.first_visible_frame}]` : "";
    const rv = S.st.review, look = rv && (rv.check_first || {})[gid];
    const checked = rv ? `<td>${on ? (isModel(on) ? "?" : "✓") : ""}</td><td>${p.length ? `${p.filter((b) => tr[String(b)] && !isModel(tr[String(b)])).length}/${p.length}` : ""}</td>
      <td class="muted">${look ? look.join(", ") : ""}</td>
      <td>${rv && (rv.confidence || {})[gid] != null ? Math.round(100 * rv.confidence[gid]) + "%" : ""}</td>` : "";
    return `<tr class="click" data-g="${gid}"><td>${gid}</td><td>${flags}</td><td>${g.excluded ? g.exclude_reason : ""}</td>
      <td>${on ? VERDICT_TEXT[on.verdict] : ""}</td><td>${bracket}</td><td>${p.length ? `${done}/${p.length}${Object.values(tr).some((t) => t.state === "burst") ? " · burst" : ""}` : ""}</td>
      ${checked}<td>${lab.time_spent_s ? Math.round(lab.time_spent_s) + " s" : ""}</td></tr>`;
  }).join("");
  const rvHead = S.st.review ? "<th>onset checked</th><th>traces checked</th><th>the model says look at</th><th>confidence</th>" : "";
  if (S.st.review) help(`Review of ${S.st.review.model || "the model"}'s answers, the model's least sure grains first: ✓ checked, ? still the model's. Click a row to open it. Saved to <code>${S.st.labels_path}</code> after every answer.`);
  $("#content").innerHTML = `<table><tr><th>grain</th><th>layout</th><th>excluded</th><th>onset</th><th>bracket (frames)</th><th>traces</th>${rvHead}<th>time</th></tr>${rows}</table>`;
  $("#content").onclick = (e) => { const tr = e.target.closest("tr[data-g]"); if (tr) openGrain(tr.dataset.g, "onset"); };
}

// ---------------------------------------------------------------- retest (blind repeat of onset)
async function renderRetestView() {
  if (!S.retest) {
    const within = Object.values(S.st.labels).filter((l) => l.onset && l.onset.verdict === "emerged_within").length;
    if (within < 16) {
      help(`The retest repeats 8 onset judgements blind. Label at least 16 grains' onsets first (now ${within}).`);
      $("#content").innerHTML = ""; return;
    }
    S.retest = (await (await fetch("/api/retest")).json()).grains;
    const done = S.st.retest.labels || {};
    const first = S.retest.findIndex((g) => !done[g]);
    S.retestIdx = first < 0 ? S.retest.length : first;
    return openRetest();
  }
  if (S.retestIdx >= S.retest.length) {
    help(`Retest complete — thank you.`); $("#content").innerHTML = ""; return;
  }
  if (S.gid !== S.retest[S.retestIdx]) return openRetest();
  renderOnset(true);
}
function openRetest() {
  S.view = "retest";
  if (S.retestIdx < S.retest.length) {
    if (S.gid) S.timers[S.gid] = spent();
    S.gid = S.retest[S.retestIdx]; S.openedAt = Date.now();
    S.step = "coarse"; S.fv = null; S.la = null; S.coarseTile = null; S.consulted = new Set();
  }
  render();
}

// ---------------------------------------------------------------- length retest (blind repeat of traces)
async function renderLengthRetest() {
  if (!S.lretest) {
    S.lretest = (await (await fetch("/api/trace_retest")).json()).traces;
    if (!S.lretest.length) {
      help(`The length retest repeats full traces blind. Trace some tubes first.`); $("#content").innerHTML = ""; return;
    }
    const done = S.st.retest.trace_labels || {};
    const first = S.lretest.findIndex((t) => !done[`${t.grain}:${t.bin}`]);
    S.lretestIdx = first < 0 ? S.lretest.length : first;
    return openLengthRetest();
  }
  if (S.lretestIdx >= S.lretest.length) {
    help(`Length retest complete — thank you.`); $("#content").innerHTML = ""; return;
  }
  const item = S.lretest[S.lretestIdx];
  if (S.lretestKey !== `${item.grain}:${item.bin}`) return openLengthRetest();
  const V = S.st.layout.trace[S.traceView], size = Math.round(2 * V.half * V.zoom);
  const last = S.st.n_bins - 2, shown = Math.min(Math.max(item.bin + S.peek, 0), last);
  const frameOf = (bb) => bb * S.st.frames_per_bin + S.st.frames_per_bin / 2;
  help(`Length retest: trace this tube again, from scratch, exactly as you would the first time: its exit from
    the centre grain first, then along the centreline to the apex. Your earlier trace is hidden on purpose.
    <kbd>[</kbd>/<kbd>]</kbd> look 10 bins earlier/later (view only) · <kbd>F</kbd> full tube · <kbd>P</kbd> partial ·
    <kbd>0</kbd> no tube · <kbd>U</kbd> unsure · <kbd>⌫</kbd> undo point · <kbd>T</kbd> touching · <kbd>W</kbd> wide/near ·
    ${S.st.layout.trace.far ? "<kbd>X</kbd> extra wide · " : ""}<kbd>H</kbd> hide marks · <kbd>C</kbd> contrast ·
    <kbd>A</kbd> smoother.`);
  const peeking = shown !== item.bin;
  $("#content").innerHTML = `<div class="row"><div class="wrap"><canvas id="tc" width="${size}" height="${size}"></canvas></div>
    <div class="side"><h3>Length retest ${S.lretestIdx + 1} of ${S.lretest.length} · trace frame ${frameOf(item.bin)}</h3>
    ${peeking ? `<p class="error">Viewing frame ${frameOf(shown)} (${shown > item.bin ? "later" : "earlier"}): look only.
      Trace at frame ${frameOf(item.bin)}.</p>` : ""}
    <button class="act" id="pE">◀ [ earlier</button><button class="act" id="pL">later ] ▶</button>
    <button class="act" id="pZ">end of movie</button>
    ${peeking ? `<button class="act primary" id="pB">back to the trace frame</button>` : ""}<br>
    <p>${S.pts.length} point(s), ${traceLen().toFixed(1)} px </p>
    <button class="act primary" id="sF">F · full tube</button><button class="act" id="sP">P · partial</button><br>
    <button class="act" id="s0">0 · no tube</button><button class="act" id="sU">U · unsure</button><br>
    <button class="act" id="undo">⌫ undo</button><button class="act" id="clr">clear</button><br>
    <button class="act ${S.contact ? "on" : ""}" id="tT">T · touching other tube/grain</button><br>
    <button class="act ${S.traceView === "wide" ? "on" : ""}" id="tW">W · wide view</button>
    ${S.st.layout.trace.far ? `<button class="act ${S.traceView === "far" ? "on" : ""}" id="tX">X · extra wide</button>` : ""}
    <button class="act ${S.contrast !== "n" ? "on" : ""}" id="tC" title="normal, high contrast, or growth: this bin minus 6 bins earlier (a growing tip stands out)">C · ${CONTRAST_NAMES[S.contrast]}</button>
    <button class="act ${S.smooth ? "on" : ""}" id="tA">A · smoother (3 bins)</button></div></div>`;
  S._img = null;
  mountTraceCanvas(shown, !peeking);
  wireTraceControls(saveRetestTrace);
  $("#pE").onclick = () => peek(-10);
  $("#pL").onclick = () => peek(+10);
  $("#pZ").onclick = () => { S.peek = last - item.bin; render(); };
  if ($("#pB")) $("#pB").onclick = () => { S.peek = 0; render(); };
}
function peek(d) {  // look at the grain earlier or later without leaving the retest item
  const item = S.lretest[S.lretestIdx];
  S.peek = Math.min(Math.max(item.bin + S.peek + d, 0), S.st.n_bins - 2) - item.bin;
  render();
}
function openLengthRetest() {
  S.view = "lretest";
  if (S.lretestIdx < S.lretest.length) {
    if (S.gid) S.timers[S.gid] = spent();
    const item = S.lretest[S.lretestIdx];
    S.gid = item.grain; S.openedAt = Date.now(); S.lretestKey = `${item.grain}:${item.bin}`;
    S.pts = []; S.contact = false; S.peek = 0;  // blind: nothing of the earlier trace is loaded
  }
  render();
}
async function saveRetestTrace(state) {
  if ((state === "full" || state === "partial") && S.pts.length < 2) {
    alert("Click at least the exit point and the apex before saving a traced tube."); return;
  }
  const item = S.lretest[S.lretestIdx];
  await post(`/api/trace_retest/${item.grain}`, { bin: item.bin, state, points: S.pts, contact: S.contact,
                                                   view: S.traceView, time_spent_s: spent() });
  await refresh();
  S.lretestIdx += 1;
  openLengthRetest();
}

// ---------------------------------------------------------------- keys
document.addEventListener("keydown", (e) => {
  if (["INPUT", "TEXTAREA"].includes(e.target.tagName) || e.metaKey || e.ctrlKey) return;
  const k = e.key;
  const lr = S.view === "lretest" && S.lretest && S.lretestIdx < S.lretest.length;
  if ((S.view === "trace" && plan().length) || lr) {
    const save = lr ? saveRetestTrace : saveTrace;
    if (!lr && k === "ArrowLeft" && !e.shiftKey) { S.traceIdx = Math.max(0, S.traceIdx - 1); loadTrace(); return render(); }
    if (!lr && k === "ArrowRight" && !e.shiftKey) { S.traceIdx = Math.min(plan().length - 1, S.traceIdx + 1); loadTrace(); return render(); }
    const map = { f: "full", p: "partial", 0: "no_tube", u: "unsure" };
    if (map[k.toLowerCase()]) return save(map[k.toLowerCase()]);
    if (k === "Backspace") { e.preventDefault(); S.pts.pop(); drawTrace(); return updateCount(); }
    if (k === "t") { S.contact = !S.contact; return render(); }
    if (!lr && (k === "g" || k === "G") && S.st.follow_mode !== "review") {
      const b = plan()[S.traceIdx];
      if (e.shiftKey) {
        const mine = ((S.follow[S.gid] || {}).refinds || []).some((r) => r.bin === b);
        if (mine) return placeGrain(b, 0, 0, true);
        return flash("No answer of yours at this time to take back.", true);
      }
      S.refindArm = !S.refindArm; return render();
    }
    if (!lr && k === "Escape" && S.refindArm) { S.refindArm = false; return render(); }
    if (!lr && k === "b" && canBurst()) return saveTrace("burst");
    const cur = lr ? null : (label().traces || {})[String(plan()[S.traceIdx])];
    if (!lr && k === "Enter" && isModel(cur)) return saveTrace(cur.state);  // review: confirm the model's answer
    if (!lr && (k === "=" || k === "+" || k === "-" || k === "_") && cur && cur.model_path) {
      e.preventDefault(); return slideTrace((k === "=" || k === "+" ? 1 : -1) * (e.shiftKey ? 5 : 1));
    }
    if (lr && k === "[") return peek(-10);
    if (lr && k === "]") return peek(+10);
    if (k === "w") return toggleWide();
    if (k === "x") return toggleFar();
    if (k === "h") { S.overlay = !S.overlay; return drawTrace(); }
    if (k === "m") { S.marker = !S.marker; return drawTrace(); }
    if (k === "a") { S.smooth = S.smooth ? 0 : 1; return render(); }
  }
  if (k === "c") { S.contrast = NEXT_CONTRAST[S.contrast] || "n"; return render(); }
  if (S.view === "census" && k === "e") { S.fieldWhich = S.fieldWhich === "early" ? "late" : "early"; return render(); }
  if (S.view === "onset" || S.view === "retest") {
    const retest = S.view === "retest";
    if (S.step === "coarse") {
      if (k === "n") return saveOnset("no_emergence_by_end", retest);
      if (k === "s") return saveOnset("emerged_at_start", retest);
      if (k === "u") return saveOnset("unobservable", retest);
      if (k === "x" && !retest) return grainMenu(200, 160, S.gid);
      if (k === "Enter" && !retest && isModel(label().onset)) return saveOnset(label().onset.verdict, false);
    } else {
      if (k === "Enter" && S.fv !== null) return saveOnset("emerged_within", retest);
      if (k === "[") return shiftFine(-12);
      if (k === "]") return shiftFine(+12);
      if (k === "b") { S.step = "coarse"; return render(); }
    }
  }
  if (!["onset", "census", "trace"].includes(S.view)) return;
  if (k === "ArrowLeft") return stepGrain(-1);
  if (k === "ArrowRight") return stepGrain(+1);
});
document.querySelectorAll("nav button").forEach((b) => b.onclick = () => {
  S.view = b.dataset.view;
  if (S.view === "trace" || S.view === "onset") resetGrainState();
  if (S.view === "lretest") S.lretestKey = null;  // re-enter blind, with an empty trace
  render();
});

// ---------------------------------------------------------------- boot
(async () => {
  await refresh();
  const list = included();
  S.gid = list.find((g) => needsOnset(g) || needsTrace(g)) || list[0] || S.st.order[0];
  S.view = Object.keys(S.st.labels).length ? (needsOnset(S.gid) ? "onset" : "trace") : "census";
  await ensureFollow(S.gid);
  resetGrainState(); render();
})();
