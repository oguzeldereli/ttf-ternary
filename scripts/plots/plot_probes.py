"""Probe traces (ttf/probe.py) for master vs the fast-ramp flip config, first ~5M tokens.

  python3 plot_probes.py --out docs/probes.png
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
RUNS = [("probe_master", "master weights", "#eb6834"),
        ("probe_fast", "flips: look-ahead + fast ramp + master's tail LR", "#111111")]
UNIGRAM, BIGRAM = 7.18, 5.1


def load(d):
    rows = [json.loads(l) for l in open(f"checkpoints/{d}/metrics.jsonl")]
    P = [r for r in rows if r.get("probe")]
    T = {r["step"]: r for r in rows if "loss" in r and "tokens" in r}
    tok = lambda s: (s + 1) * 32768
    return P, T, tok


ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/figures/archive/probes.png")
a = ap.parse_args()
fig, axs = plt.subplots(2, 3, figsize=(21, 11), facecolor=SURFACE)
ax = axs.ravel()
for d, lab, c in RUNS:
    if not os.path.exists(f"checkpoints/{d}/metrics.jsonl"): continue
    P, T, tok = load(d)
    s = sorted(T); ax[0].plot([tok(i) for i in s], [T[i]["loss"] for i in s], color=c, lw=1.2, alpha=0.8, label=lab)
    x = [tok(r["step"]) for r in P]
    for K, ls in ((1, ":"), (8, "-."), (64, "--"), (512, "-")):
        ax[1].plot(x, [r[f"loss_ctx{K}"] for r in P], color=c, ls=ls, lw=2, label=f"{lab}, context {K}")
    ax[2].plot(x, [r["loss_ctx1"] - r["loss_ctx512"] for r in P], color=c, lw=2.4, label=lab)
    ax[3].plot(x, [r["copy_gain"] for r in P], color=c, lw=2.4, label=lab)
    ax[4].plot(x, [r["bias_std"] for r in P], color=c, lw=2.4, label=f"{lab}: unigram-part logit spread")
    ax[4].plot(x, [r["logit_std"] for r in P], color=c, lw=1.4, ls="--", label=f"{lab}: total logit std")
    ax[5].plot(x, [r["norm_gain"] for r in P], color=c, lw=2.4, label=f"{lab}: final-norm gain")
    ax[5].plot(x, [r["emb_norm"] for r in P], color=c, lw=1.4, ls="--", label=f"{lab}: embedding row norm")
for i in (0, 1):
    ax[i].axhline(UNIGRAM, color=INK2, lw=1, ls=":"); ax[i].axhline(BIGRAM, color=INK2, lw=1, ls=":")
    ax[i].text(4e4, UNIGRAM + 0.05, "unigram 7.18", color=INK2, fontsize=8)
    ax[i].text(4e4, BIGRAM + 0.05, "bigram ~5.1", color=INK2, fontsize=8)
titles = ["Train loss", "Loss given only the last K tokens (val)", "Context use: loss(K=1) - loss(K=512)",
          "In-context copying: loss(first copy) - loss(repeat)", "Logit spread", "Final-norm gain / embedding norm"]
for i, t in enumerate(titles):
    ax[i].set_title(t, loc="left", color=INK, fontsize=11.5)
    ax[i].set_xscale("log"); ax[i].set_xlim(3e4, 6e6)
    ax[i].set_facecolor(SURFACE); ax[i].grid(True, which="both", color=GRID, lw=0.6)
    ax[i].set_xlabel("tokens", color=INK2); ax[i].tick_params(colors=INK2)
    for sp in ax[i].spines.values(): sp.set_color(GRID)
    ax[i].legend(fontsize=8, frameon=False)
ax[0].set_yscale("log"); ax[1].set_yscale("log")
fig.tight_layout()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
fig.savefig(a.out, dpi=120, facecolor=SURFACE)
print("wrote", a.out)
