"""Plot a training run: loss, flip rate, never-changed fraction, per-layer flips.

  python3 plot_run.py checkpoints/run [--out checkpoints/run/run.png]

Reads metrics.jsonl written by ttf/train.py --track_flips.
"""
import argparse, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"          # categorical slots 1-3
SEQ = LinearSegmentedColormap.from_list("blues", [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--out", default=None)
ap.add_argument("--title", default=None)
args = ap.parse_args()

recs = [json.loads(l) for l in open(f"{args.run}/metrics.jsonl")]
tr = [r for r in recs if "loss" in r]
va = [r for r in recs if "val_loss" in r]
if not tr:
    sys.exit("no training records")
tok = np.array([r["tokens"] for r in tr]) / 1e6
loss = np.array([r["loss"] for r in tr])
has_flips = "flip_frac" in tr[-1]

n_panels = 4 if has_flips else 1
fig, axes = plt.subplots(n_panels, 1, figsize=(9, 2.6 * n_panels + 0.8), sharex=True,
                         gridspec_kw=dict(hspace=0.28))
axes = np.atleast_1d(axes)
fig.patch.set_facecolor(SURFACE)


def style(ax, ylabel):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d9d8d3")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)


ax = axes[0]
k = max(1, len(loss) // 200)
sm = np.convolve(loss, np.ones(k) / k, "same") if k > 1 else loss
ax.plot(tok, loss, color=S1, linewidth=0.8, alpha=0.35)
ax.plot(tok[k:-k] if k > 1 else tok, sm[k:-k] if k > 1 else sm, color=S1,
        linewidth=2, label="train")
if va:
    vt = np.array([r["tokens"] for r in tr if r["step"] in {v["step"] for v in va}] or
                  [0]) / 1e6
    vl = np.array([v["val_loss"] for v in va])
    vt = np.interp([v["step"] for v in va], [r["step"] for r in tr], tok)
    ax.plot(vt, vl, color=S2, linewidth=2, marker="o", markersize=5, label="val")
    ax.annotate(f"val ppl {np.exp(vl[-1]):.0f}", (vt[-1], vl[-1]), color=INK,
                fontsize=9, xytext=(4, 6), textcoords="offset points")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
style(ax, "cross-entropy (nats)")
ax.set_title(args.title or args.run, color=INK, fontsize=12, loc="left", pad=10)

if has_flips:
    ff = np.array([r.get("flip_frac", np.nan) for r in tr])
    nf = np.array([r.get("never_frac", np.nan) for r in tr])
    ax = axes[1]
    ax.plot(tok, np.where(ff > 0, ff, np.nan) * 100, color=S2, linewidth=2)
    ax.set_yscale("log")
    style(ax, "flips / step (% of weights)")

    ax = axes[2]
    ax.plot(tok, nf * 100, color=S3, linewidth=2)
    ax.annotate(f"{nf[-1]*100:.1f}% never flipped", (tok[-1], nf[-1] * 100), color=INK,
                fontsize=9, ha="right", xytext=(-4, 8), textcoords="offset points")
    style(ax, "never flipped (%)")
    ax.set_ylim(0, 102)

    ax = axes[3]
    L = np.array([r["flip_frac_layers"] for r in tr if "flip_frac_layers" in r]).T
    lt = np.array([r["tokens"] for r in tr if "flip_frac_layers" in r]) / 1e6
    pos = L[L > 0]
    if pos.size:
        im = ax.pcolormesh(lt, np.arange(L.shape[0]), np.where(L > 0, L, np.nan) * 100,
                           cmap=SEQ, norm=LogNorm(vmin=max(pos.min() * 100, 1e-8),
                                                  vmax=pos.max() * 100), shading="nearest")
        cb = fig.colorbar(im, ax=ax, pad=0.01)
        cb.set_label("% flipped", color=INK2, fontsize=9)
        cb.ax.tick_params(colors=INK2, labelsize=8)
    style(ax, "ternary layer (depth order)")

axes[-1].set_xlabel("tokens (M)", color=INK2, fontsize=10)
out = args.out or f"{args.run}/run.png"
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
print(out)
