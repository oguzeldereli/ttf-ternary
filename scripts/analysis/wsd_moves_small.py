"""Two-panel figure for the paper: moves per step and the undo share, TTF / master under cosine / WSD (--movediag runs).
    python scripts/analysis/wsd_moves_small.py OUT.png NAME=RUN_DIR [NAME=RUN_DIR ...]
"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt


def load(d):
    rows = [json.loads(l) for l in open(os.path.join(d, "metrics.jsonl")) if '"movediag"' in l]
    return {k: np.array([r.get(k, np.nan) for r in rows], float) for k in ("step", "md_move_frac", "md_undo_frac")}


def smooth(y, w=5):
    k = np.ones(w) / w
    return np.convolve(np.nan_to_num(y), k, mode="same") / np.convolve(np.isfinite(y).astype(float), k, mode="same")


fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for spec in sys.argv[2:]:
    name, d = spec.split("=", 1)
    r = load(d)
    ls = "-" if "cosine" in name else "--"
    col = "C0" if name.startswith("TTF") else "C3"
    ax[0].plot(r["step"], smooth(r["md_move_frac"]), ls, color=col, label=name)
    ax[1].plot(r["step"], smooth(r["md_undo_frac"]), ls, color=col, label=name)
ax[0].set_yscale("log"); ax[0].set_ylabel("moves per step (share of weights)")
ax[1].set_ylabel("undo share (last ~1000 steps)")
for a in ax:
    a.set_xlabel("step"); a.grid(alpha=0.3)
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(sys.argv[1], dpi=150); print(sys.argv[1])
