"""Mean cross-section of the tubes about the annotator's trace, each profile flipped so its darkest point is on the
right; the network map and the model path's crossings flipped the same way."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

files = sys.argv[1:-1]
out = sys.argv[-1]
fig, axes = plt.subplots(1, len(files), figsize=(5.2 * len(files), 3.8), squeeze=False)
for ax, f in zip(axes[0], files):
    d = np.load(f)
    offs, prof, pm, model = d["offs"], d["prof"], d["pmap"].astype(float), d["model"]
    prof = prof - np.nanmedian(np.concatenate([prof[:, :6], prof[:, -6:]], 1), axis=1, keepdims=True)
    win = np.abs(offs) <= 6
    dark = offs[win][np.nanargmin(np.where(np.isnan(prof[:, win]), np.inf, prof[:, win]), axis=1)]
    flip = np.where(dark < 0, -1, 1)
    P = np.where(flip[:, None] > 0, prof, prof[:, ::-1])
    M = np.where(flip[:, None] > 0, pm, pm[:, ::-1])
    mod = model * flip
    ax.plot(offs, np.nanmean(P, 0), color="k", lw=2, label="image (mean)")
    ax.fill_between(offs, np.nanpercentile(P, 25, 0), np.nanpercentile(P, 75, 0), color="k", alpha=0.12)
    ax2 = ax.twinx()
    ax2.plot(offs, np.nanmean(M, 0) / 250.0, color="tab:purple", lw=1.5, label="network P")
    ok = np.isfinite(mod)
    h, e = np.histogram(mod[ok], bins=np.arange(-12.25, 12.3, 0.5))
    ax2.bar((e[:-1] + e[1:]) / 2, h / max(h.max(), 1) * 0.8, width=0.45, color="tab:pink", alpha=0.5, label="model path")
    ax.axvline(0, color="tab:green", lw=1.5)
    ax.set_title(f"{f.split('/')[-1]}: {len(P)} pts, model path crosses {ok.sum()}", fontsize=9)
    ax.set_xlabel("px from the annotator's trace (darkest side right)")
    ax2.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(out, dpi=110)
print("wrote", out)
