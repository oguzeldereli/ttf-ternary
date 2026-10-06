"""The mid-training fix: steps decaying like master's lr (p 1) vs the recipe (p 0.5), as TTF minus master + the same extras.
Left: 110M ranks 512 (finished); right: 1.3B, the p 1 branch from step 2000 vs its parent.
    python scripts/plots/plot_midfix.py X1B_CURVES.txt  -> docs/figures/midtraining_fix.png"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
def val(p): return {json.loads(l)["step"]: json.loads(l)["val_loss"] for l in open(p) if '"val_loss"' in l}
X, cur = {}, None
for l in open(sys.argv[1]):
    if l.startswith("== "): cur = l[3:].strip(); X[cur] = {}
    elif '"val_loss"' in l and cur: j = json.loads(l); X[cur][j["step"]] = j["val_loss"]
F = "checkpoints/final"; m = val(f"{F}/mx_extras_s0/metrics.jsonl")
fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
def dif(a, v, ref, lab, c, st="-", lw=2):
    s = sorted(k for k in v if k in ref and k >= 500)
    a.plot(s, [v[k] - ref[k] for k in s], st, color=c, lw=lw, label=lab)
dif(ax[0], val(f"{F}/sw512_base/metrics.jsonl"), m, "recipe: steps x lr ratio^0.5 (final -0.043)", "k")
dif(ax[0], val(f"{F}/sw512_p1/metrics.jsonl"), m, "steps x lr ratio^1 (p 1; final -0.017)", "C2")
dif(ax[0], val(f"{F}/sw512_p075/metrics.jsonl"), m, "p 0.75 (final -0.038)", "C1", lw=1.2)
ax[0].axvspan(2000, 6500, color="C2", alpha=0.07); ax[0].text(2100, 0.04, "mid-training: p 1 stays flat", color="C2")
ax[0].axvspan(6500, 9155, color="C3", alpha=0.06); ax[0].text(6600, 0.04, "late: p 1 loses it", color="C3")
ax[0].set_title("110M ranks 512 (no momentum): TTF minus master + extras")
ref = X["x1b_mx_extras_lr75"]
dif(ax[1], X["x1b_nomom512_lr15"], ref, "recipe (p 0.5), no momentum, ranks 512", "k")
dif(ax[1], X["x1b_nm512_b2k_p1"], ref, "branch at 2000 with p 1", "C2")
dif(ax[1], X["x1b_mx_extras_lr50"], ref, "master + extras at its better lr (5e-4)", "0.5", "--", 1.5)
ax[1].axvline(2000, color="0.6", lw=0.8, ls=":"); ax[1].text(2050, 0.06, "branch point", color="0.4")
ax[1].set_title("1.3B ranks 512: TTF minus master + extras (lr 7.5e-4)")
for a in ax:
    a.axhline(0, color="0.5", lw=0.8); a.set_xlabel("step"); a.set_ylabel("val loss difference (training-time windows)")
    a.set_ylim(-0.16, 0.08); a.legend(fontsize=9, loc="lower right")
fig.tight_layout(); fig.savefig("docs/figures/midtraining_fix.png", dpi=130); print("ok")
