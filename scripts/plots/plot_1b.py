"""1.3B runs (Myriad): val loss curves and the difference from master + the same extras at each evaluation step
(training-time windows, same for every run: seed 1337, 30 batches of 16).
    python scripts/plots/plot_1b.py CURVES.txt     (lines "== RUN" followed by metrics.jsonl val_loss lines)
Writes docs/figures/x1b_runs.png."""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V, cur = {}, None
for l in open(sys.argv[1]):
    if l.startswith("== "):
        cur = l[3:].strip(); V[cur] = {}
    elif '"val_loss"' in l and cur:
        j = json.loads(l); V[cur][j["step"]] = j["val_loss"]
RUNS = [  # name, label, colour, style
    ("x1b_mx_extras_lr75", "master + extras, lr 7.5e-4", "0.25", "-"),
    ("x1b_mx_extras_lr50", "master + extras, lr 5e-4", "0.25", "--"),
    ("x1b_master_lr50", "plain master, lr 5e-4", "0.6", "-"),
    ("x1b_master_lr75", "plain master, lr 7.5e-4", "0.6", "--"),
    ("x1b_ts512_la1000an05", "TTF ranks 512, momentum, lr 7.5e-4", "C3", "-"),
    ("x1b_ts256_la1000an05", "TTF ranks 256, momentum, lr 7.5e-4", "C3", ":"),
    ("x1b_ts512_lr15", "TTF ranks 512, momentum, lr 1.5e-3", "C1", "-"),
    ("x1b_nomom512_lr15", "TTF ranks 512, no momentum, lr 1.5e-3", "C0", "-"),
    ("x1b_nomom1024_lr15", "TTF ranks 1024, no momentum, lr 1.5e-3", "C2", "-"),
]
ref = V["x1b_mx_extras_lr75"]
fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
for n, lab, c, st in RUNS:
    v = V.get(n, {})
    if not v:
        continue
    s = sorted(k for k in v if k >= 500)
    ax[0].plot(s, [v[k] for k in s], st, color=c, label=f"{lab} ({v[max(v)]:.4f} @ {max(v)})")
    s2 = [k for k in s if k in ref and n != "x1b_mx_extras_lr75"]
    if s2:
        ax[1].plot(s2, [v[k] - ref[k] for k in s2], st, color=c, label=lab)
ax[0].set_ylim(2.45, 3.6); ax[0].set_xlabel("step (32k tokens each)"); ax[0].set_ylabel("val loss (training-time windows)")
ax[0].set_title("1.3B (width 2048), 300M tokens"); ax[0].legend(fontsize=8)
ax[1].axhline(0, color="k", lw=0.8); ax[1].set_ylim(-0.25, 0.25)
ax[1].set_xlabel("step"); ax[1].set_ylabel("val loss minus master + extras (lr 7.5e-4)")
ax[1].set_title("Difference from master + the same extras (negative: better)"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig("docs/figures/x1b_runs.png", dpi=130); print("wrote docs/figures/x1b_runs.png")
