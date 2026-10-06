"""110M proxy sweep (6 Oct): each arm minus master + the same extras (mx_extras_s0) at each evaluation step, by base
(ranks 192 = the 1.3B rank / width, ranks 512 = the 1.3B curve shape).
    python scripts/plots/plot_sweep.py CURVE_DIR      (RUN.jsonl files with val_loss lines) -> docs/figures/sweep_oct6.png"""
import glob, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
D = sys.argv[1]
def val(p):
    return {json.loads(l)["step"]: json.loads(l)["val_loss"] for l in open(p) if '"val_loss"' in l}
m = val("checkpoints/final/mx_extras_s0/metrics.jsonl")
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
for k, base in enumerate(("sw192", "sw512")):
    for f in sorted(glob.glob(os.path.join(D, base + "_*.jsonl"))):
        n = os.path.basename(f)[:-6]; v = val(f)
        s = sorted(x for x in v if x in m and x >= 500)
        if not s: continue
        st = "k-" if n.endswith("_base") else "-"
        ax[k].plot(s, [v[x] - m[x] for x in s], st, lw=2.2 if n.endswith("_base") else 1.2,
                   label=f"{n[len(base)+1:]} ({v[s[-1]] - m[s[-1]]:+.3f} @ {s[-1]})")
    ax[k].axhline(0, color="0.5", lw=0.8); ax[k].set_ylim(-0.25, 0.35)
    ax[k].set_title(f"110M {'ranks 192 (1.3B rank / width)' if base == 'sw192' else 'ranks 512 (1.3B shape)'}: arm minus master + extras")
    ax[k].set_xlabel("step"); ax[k].set_ylabel("val loss difference (training-time windows)"); ax[k].legend(fontsize=8)
fig.tight_layout(); fig.savefig("docs/figures/sweep_oct6.png", dpi=120); print("ok")
