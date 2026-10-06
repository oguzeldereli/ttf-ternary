"""1.3B without the short momentum (6 Oct): val loss curves and the difference from master + the same extras at its
better lr (5e-4) at each evaluation step (training-time windows, the same for every run).
    python scripts/plots/plot_1b_r1024.py CURVE_DIR     (RUN.jsonl metrics files) -> docs/figures/x1b_r1024.png"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1]


def val(n):
    out = {}
    for l in open(os.path.join(D, n + ".jsonl")):
        if '"val_loss"' in l:
            j = json.loads(l); out[j["step"]] = j["val_loss"]
    return out


RUNS = [  # name, label, colour, style, width
    ("x1b_mx_extras_lr50", "master + extras, lr 5e-4 (reference)", "k", "-", 2.2),
    ("x1b_mx_extras_lr75", "master + extras, lr 7.5e-4", "0.45", "--", 1.3),
    ("x1b_master_lr50", "plain master, lr 5e-4", "0.65", ":", 1.3),
    ("x1b_ts512_la1000an05", "TTF rank 512, momentum, lr 7.5e-4", "C3", "-", 1.3),
    ("x1b_nomom512_lr15", "TTF rank 512, no momentum, lr 1.5e-3", "C0", "-", 1.6),
    ("x1b_nomom1024_lr15", "TTF rank 1024, no momentum, lr 1.5e-3", "C2", "-", 2.4),
]
V = {n: val(n) for n, *_ in RUNS}
ref = V["x1b_mx_extras_lr50"]
fig, ax = plt.subplots(1, 2, figsize=(15, 5.6))
for n, lab, c, st, lw in RUNS:
    v = V[n]; s = sorted(k for k in v if k >= 500)
    ax[0].plot(s, [v[k] for k in s], st, color=c, lw=lw, label=f"{lab} ({v[max(v)]:.4f} final)")
    if n != "x1b_mx_extras_lr50":
        s2 = [k for k in s if k in ref]
        ax[1].plot(s2, [v[k] - ref[k] for k in s2], st, color=c, lw=lw, label=f"{lab} ({v[s2[-1]] - ref[s2[-1]]:+.4f})")
v = V["x1b_nomom1024_lr15"]; d = v[9155] - ref[9155]
ax[1].annotate(f"rank 1024 final {d:+.4f}", (9155, d), (6300, -0.07), color="C2", fontsize=10,
               arrowprops=dict(arrowstyle="->", color="C2"))
ax[0].set_ylim(2.45, 3.3); ax[0].set_xlabel("step (32k tokens each; 9155 = 300M tokens)")
ax[0].set_ylabel("val loss (training-time windows)"); ax[0].set_title("1.3B (width 2048), 300M tokens"); ax[0].legend(fontsize=8)
ax[1].axhline(0, color="k", lw=0.8); ax[1].set_ylim(-0.16, 0.12); ax[1].set_xlabel("step")
ax[1].set_ylabel("val loss minus master + extras, lr 5e-4")
ax[1].set_title("Difference from the stronger master (negative: TTF better)"); ax[1].legend(fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig("docs/figures/x1b_r1024.png", dpi=130); print("wrote docs/figures/x1b_r1024.png")
