"""Does a small model reproduce the 1.3B curve shape? TTF minus master + the same extras at each evaluation step, for
1.3B (ranks 512 = 0.25 of the width) and 110M at several rank / width ratios, x-axis as a fraction of training.
    python scripts/plots/plot_proxy.py X1B_CURVES.txt     -> docs/figures/proxy_shape.png"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def val_file(p):
    return {json.loads(l)["step"]: json.loads(l)["val_loss"] for l in open(p) if '"val_loss"' in l}
X, cur = {}, None
for l in open(sys.argv[1]):
    if l.startswith("== "): cur = l[3:].strip(); X[cur] = {}
    elif '"val_loss"' in l and cur: j = json.loads(l); X[cur][j["step"]] = j["val_loss"]
F = "checkpoints/final"
m110 = val_file(f"{F}/mx_extras_s0/metrics.jsonl")
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
def plot(a, v, ref, total, lab, st, c):
    s = sorted(k for k in v if k in ref and k >= 500)
    a.plot([k / total for k in s], [v[k] - ref[k] for k in s], st, color=c, label=lab)
plot(ax[0], X["x1b_ts512_la1000an05"], X["x1b_mx_extras_lr75"], 9155, "1.3B ranks 512 (r/w 0.25), momentum, lr 7.5e-4", "-", "C3")
plot(ax[0], X["x1b_ts256_la1000an05"], X["x1b_mx_extras_lr75"], 9155, "1.3B ranks 256 (r/w 0.125)", ":", "C3")
plot(ax[0], X["x1b_nomom512_lr15"], X["x1b_mx_extras_lr75"], 9155, "1.3B ranks 512, no momentum, lr 1.5e-3", "-", "C0")
for r, lab, c in [("ts16rs512tau1000an05_s0", "110M ranks 512 (r/w 0.67)", "C2"), ("ts16r256rs256tau1000an05_s0", "110M ranks 256 (r/w 0.33)", "C1"),
                  ("ts128an05_diag_s0", "110M ranks 128 (r/w 0.17)", "C4"), ("ts64an05_shadows_s0", "110M ranks 64 (r/w 0.08)", "C5"),
                  ("ts512an05_nomom_s0", "110M ranks 512, no momentum", "C0")]:
    plot(ax[1], val_file(f"{F}/{r}/metrics.jsonl"), m110, 9155, lab, "-", c)
for a, t in zip(ax, ["1.3B: TTF minus master + extras", "110M: TTF minus master + extras"]):
    a.axhline(0, color="k", lw=0.8); a.set_xlabel("fraction of training"); a.set_title(t); a.legend(fontsize=8)
    a.set_ylabel("val loss difference (training-time windows)")
ax[0].set_ylim(-0.25, 0.25); ax[1].set_ylim(-0.25, 0.45)
fig.tight_layout(); fig.savefig("docs/figures/proxy_shape.png", dpi=120); print("ok")
