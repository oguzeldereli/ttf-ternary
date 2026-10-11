"""Plot the --movediag records (bitnet/movediag.py) of several runs against training step: why does the late-training
gain of the trit moves shrink? The trit gain per step factors as
    realized gain per step = moves per step  x  first-order gain per move (true gradient)  x  survive
and the first-order gain per move as |true gradient| at the moved entries x precision (direction). Each factor is a
panel, master and the TTF arms on the same axes, smoothed over WINDOW diag steps (the per-step values are noisy).
    python scripts/analysis/movediag_plot.py OUT.png NAME=RUN_DIR [NAME=RUN_DIR ...] [--window 5]
"""
import argparse, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("runs", nargs="+"); ap.add_argument("--window", type=int, default=5)
a = ap.parse_args()


def load(d):
    rows = [json.loads(l) for l in open(os.path.join(d, "metrics.jsonl")) if '"movediag"' in l]
    keys = {k for r in rows for k in r if k != "movediag"}      # runs resumed with newer fields: union of keys
    return {k: np.array([r.get(k, np.nan) for r in rows], float) for k in keys} if rows else None


def smooth(y, w):
    if w <= 1 or len(y) < w:
        return y
    k = np.ones(w) / w
    return np.convolve(np.nan_to_num(y), k, mode="same") / np.convolve(np.isfinite(y).astype(float), k, mode="same")


R = {}
for spec in a.runs:
    name, d = spec.split("=", 1)
    r = load(d)
    if r is not None:
        R[name] = r
panels = [("md_move_frac", "moves per step (share of weights)", True),
          ("md_pred_per_move", "first-order gain per move on the true gradient (nats)", True),
          ("md_precision", "precision: share of moves downhill on the true gradient", False),
          ("md_g_ratio", "|true gradient| at moved entries / mean |true gradient|", False),
          ("md_survive", "survive: realized / first-order (curvature, interactions)", False),
          ("md_undo_frac", "share of moves undoing the net change of the last ~1000 steps", False),
          ("md_realized", "realized trit gain per step (nats, probe loss)", False),
          ("md_float", "float-tail gain per step (nats)", False),
          ("md_win_realized", "value of the last ~1000 steps' trit moves (old trits back in, today's floats; nats)", False),
          ("md_win_per_net", "same, per net changed entry (nats)", False)]
fig, axs = plt.subplots(5, 2, figsize=(14, 21)); axs = axs.ravel()
for ax, (k, title, logy) in zip(axs, panels):
    for name, r in R.items():
        if k not in r:
            continue
        y = r[k]
        if k == "md_survive":            # a ratio of two noisy numbers: smooth numerator and denominator separately
            y = smooth(r["md_realized"], a.window) / smooth(r["md_pred"], a.window)
        else:
            y = smooth(y, a.window)
        x = r["step"]
        if k.startswith("md_win"):       # the window needs ~1000 steps of history; its first values are the init
            keep = x >= 2000
            x, y = x[keep], y[keep]
        ax.plot(x, np.abs(y) if logy else y, label=name, lw=1.6)
    ax.set_title(title, fontsize=10); ax.set_xlabel("step"); ax.grid(alpha=0.3)
    if logy:
        ax.set_yscale("log")
    if k == "md_survive":
        ax.set_ylim(-3, 3)
    if k.startswith("md_win"):
        ax.set_yscale("log")
    if k in ("md_precision",):
        ax.axhline(0.5, color="gray", lw=0.8)
    if k in ("md_realized", "md_float", "md_win_realized", "md_win_per_net"):
        ax.axhline(0, color="gray", lw=0.8)
    ax.legend(fontsize=8)
fig.suptitle(f"what one step's trit moves are worth (110M; smoothed over {a.window} diag steps)", fontsize=12)
plt.tight_layout(rect=(0, 0, 1, 0.98)); plt.savefig(a.out, dpi=110); print(a.out)
# the decomposition the late-gain question needs: window value per step = gross moves per step x value per gross move
# (value per gross move = the window's realized gain / (moves per step x window length)); net / gross = how much of the
# moving survives as net change
print("\nwindow decomposition (per phase): value/step = moves/step x value/move")
for name, r in R.items():
    if "md_win_realized" not in r:
        continue
    s = r["step"]
    for lo, hi in ((1000, 3000), (3000, 5000), (5000, 7000), (7000, 9200)):
        m = (s >= lo) & (s < hi) & np.isfinite(r["md_win_realized"]) & (r["md_win_age"] > 0)
        if not m.any():
            continue
        vps = np.nanmean(r["md_win_realized"][m] / r["md_win_age"][m])
        # the window's gross moves: the mean sampled moves per step over (step - age, step], times age
        gw = np.array([np.nanmean(r["md_moves"][(s > s_ - a_) & (s <= s_)]) * a_
                       for s_, a_ in zip(s[m], r["md_win_age"][m])])
        mps = np.nanmean(gw / r["md_win_age"][m])
        net = np.nanmean(r["md_win_net"][m] / gw)
        print(f"{name:8s} {lo}-{hi}: value/step {vps:+.2e}  moves/step {mps:9.0f}  value/move {np.nanmean(r["md_win_realized"][m] / gw):+.2e}"
              f"  net/gross {net:.2f}  precision {np.nanmean(r['md_precision'][m]):.3f}  |g| ratio {np.nanmean(r['md_g_ratio'][m]):.2f}")
for name, r in R.items():
    s = r["step"]
    for lo, hi in ((1000, 3000), (3000, 5000), (5000, 7000), (7000, 9200)):
        m = (s >= lo) & (s < hi)
        if m.any():
            print(f"{name:8s} {lo}-{hi}: moves/step {np.nanmean(r['md_moves'][m]):9.0f}  pred/move {np.nanmean(r['md_pred_per_move'][m]):.2e}"
                  f"  precision {np.nanmean(r['md_precision'][m]):.3f}  |g| ratio {np.nanmean(r['md_g_ratio'][m]):.2f}"
                  f"  survive {np.nansum(r['md_realized'][m]) / np.nansum(r['md_pred'][m]):.2f}"
                  f"  realized/step {np.nanmean(r['md_realized'][m]):+.2e}  float/step {np.nanmean(r['md_float'][m]):+.2e}"
                  f"  undo {np.nanmean(r['md_undo_frac'][m]):.3f}"
                  + (f"  window {np.nanmean(r['md_win_realized'][m]):+.2e} over {np.nanmean(r['md_win_net'][m]):.0f} net"
                     if 'md_win_realized' in r else ""))
