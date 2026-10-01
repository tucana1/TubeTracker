"""Per-case numbers: lengths, onset, drift vs the annotator's view offset, how the trace and the drawn / whole
route sit relative to each other, and whether the drawn tube lies on another labelled grain's trace."""
import numpy as np

from common import BAD, CENSUS, G, L, arclen, drawn, drift_at, idx, length_at, pdist, resample, trace

for gid, b, f in BAD:
    g = G[gid]
    tr, t = trace(gid, b)
    trp, _ = resample(tr, 1.0)
    d = drawn(g, b)
    full = drawn(g, b, cut=False)
    Lb = length_at(g, b)
    lab_on = L["labels"][gid]["onset"].get("first_visible_bin")
    mod_on = (g.get("onset_frame") or 0) // 300 if g.get("onset_frame") else None
    print(f"\n=== {gid}@{b}  cov={f:.2f}  traced L={t['length_px']:.1f} (arc {arclen(tr):.1f})  model L={Lb:.1f}"
          f"  final={g.get('final_length_px')}  route arc={arclen(g['path']):.1f}  flags={g['flags']}")
    print(f"  onset: label first visible bin {lab_on}, model onset bin {mod_on}")
    print(f"  view_offset={t['view_offset']}  model drift={drift_at(g, b).round(2).tolist()}"
          f"  has_drift={bool(g.get('drift'))}  exit_xy={g.get('exit_xy')}  census=({g['x']},{g['y']})")
    print(f"  trace start {tr[0].round(1).tolist()} end {tr[-1].round(1).tolist()};"
          f"  drawn start {None if d is None else d[0].round(1).tolist()} end {None if d is None else d[-1].round(1).tolist()}")
    if d is not None and len(d) > 1:
        dp, _ = resample(d, 1.0)
        t2d = pdist(trp, d)
        d2t = pdist(dp, tr)
        t2f = pdist(trp, full)
        print(f"  trace->drawn: median {np.median(t2d):.1f}, frac<=2 {np.mean(t2d <= 2):.2f};  drawn->trace: median"
              f" {np.median(d2t):.1f}, frac<=2 {np.mean(d2t <= 2):.2f}, max {d2t.max():.1f}")
        print(f"  trace->whole route: median {np.median(t2f):.1f}, frac<=2 {np.mean(t2f <= 2):.2f}, frac<=4 {np.mean(t2f <= 4):.2f}")
        # how far along the trace is the drawn tube on it
        s_on = np.nonzero(t2d <= 2.5)[0]
        print(f"  trace points covered (<=2.5 px) at arc positions: {s_on.min() if len(s_on) else None}"
              f"..{s_on.max() if len(s_on) else None} of {len(trp)}")
        # without drift
        nd = drawn(g, b, drift=False)
        if g.get("drift"):
            t2n = pdist(trp, nd)
            print(f"  (no-drift drawn: trace frac<=2 {np.mean(t2n <= 2):.2f})")
        # other labelled grains' traces at this bin
        hits = []
        for oid, olab in L["labels"].items():
            if oid == gid:
                continue
            otr, ot = trace(oid, b)
            if otr is None:
                continue
            op, _ = resample(otr, 1.0)
            dd = pdist(dp, otr)
            if np.mean(dd <= 3) > 0.1:
                hits.append((oid, round(float(np.mean(dd <= 3)), 2), ot["state"]))
        print(f"  drawn tube on other labelled traces at bin {b}: {hits}")
        # other grains' drawn model tubes at this bin that coincide with the trace
        hits2 = []
        for oid, og in G.items():
            if oid == gid:
                continue
            od = drawn(og, b)
            if od is None or len(od) < 2:
                continue
            dd = pdist(trp, od)
            if np.mean(dd <= 2.5) > 0.1:
                hits2.append((oid, round(float(np.mean(dd <= 2.5)), 2)))
        print(f"  the trace is covered by other grains' drawn tubes: {hits2}")
    # model length history around the bin
    i = idx(g, b)
    Ls = g["length"]["px"]
    marks = [b - 40, b - 20, b - 10, b - 5, b, b + 1]
    print("  model L around bin:", {bb: round(Ls[idx(g, bb)], 1) for bb in marks if 9 <= bb <= 350})
    # nearby census grains
    near = sorted(((np.hypot(c["x"] - g["x"], c["y"] - g["y"]), cid) for cid, c in CENSUS.items() if cid != gid))[:4]
    print("  nearest census grains:", [(cid, round(dd, 1)) for dd, cid in near])
