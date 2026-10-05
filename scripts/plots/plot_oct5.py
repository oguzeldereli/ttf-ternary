"""Figures for 5 Oct: the overnight low-rank ideas and speed modes as the difference from their baseline at each
evaluation step (110M, training-time windows of seed 1337), and `--ts_micro` on the same 974 windows.

    python scripts/plots/plot_oct5.py DATA_DIR
DATA_DIR holds RUN.jsonl files ("val_loss" lines) of runs still on the lab PCs; finished runs and baselines are read from
checkpoints/final. Writes docs/figures/oct5_ideas.png and oct5_micro.png.
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1]
FIN, OUT = "checkpoints/final", "docs/figures"


def val(run):
    for p in (os.path.join(D, run + ".jsonl"), os.path.join(FIN, run, "metrics.jsonl")):
        if os.path.exists(p):
            v = {}
            for l in open(p):
                if '"val_loss"' in l:
                    j = json.loads(l)
                    v[j["step"]] = j["val_loss"]
            if v:
                return v
    return {}


BASE = {64: val("ts64an05_shadows_s0"), 128: val("ts128an05_diag_s0"), 512: val("ts512an05_shadows_s0")}


def delta(run, rank):
    v, b = val(run), BASE[rank]
    s = sorted(k for k in v if k in b and k >= 500)
    return s, [v[k] - b[k] for k in s]


fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for run, rank, lab, st in [
        ("ts64an05_nomom_s0", 64, "no momentum, ranks 64", "-"),
        ("ts128an05_nomom_s0", 64, "no momentum, ranks 128 (memory of ranks 64) vs ranks 64", "-"),
        ("ts128an05_nomom_s0", 128, "no momentum, ranks 128 vs ranks 128", "--"),
        ("ts512an05_nomom_s0", 512, "no momentum, ranks 512", "-")]:
    s, d = delta(run, rank)
    ax[0].plot(s, d, st, label=lab)
ax[0].axhline(0, color="k", lw=0.8)
ax[0].set_title("No short momentum (--ts_nomom): loss minus baseline (running)")
ax[0].set_xlabel("step"); ax[0].set_ylabel("val loss - same-rank baseline"); ax[0].legend(fontsize=8)
for run, rank, lab in [
        ("ts64an05_gate_s0", 64, "gate (this batch's sign), r64"), ("ts64an05_sgate1_s0", 64, "sign counter K1, r64"),
        ("ts64an05_init10_s0", 64, "sparse init 10%, r64"), ("ts64an05_init02_s0", 64, "sparse init 2%, r64"),
        ("ts64an05_decay1e4_s0", 64, "trit decay 1e-4, r64"), ("ts64an05_decay3e4_s0", 64, "trit decay 3e-4, r64"),
        ("ts512an05_init10_s0", 512, "sparse init 10%, r512"),
        ("ts512an05_fast_s0", 512, "speed modes (every 4 + defer + int8), r512")]:
    s, d = delta(run, rank)
    ax[1].plot(s, d, label=lab)
ax[1].axhline(0, color="k", lw=0.8)
ax[1].set_title("Other overnight runs: loss minus baseline (finished)")
ax[1].set_xlabel("step"); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "oct5_ideas.png"), dpi=130)

# --ts_micro: same 974 windows at step 1000 (final_eval on the lab PCs), peak memory
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
names = ["16 x 1\n(one pass)", "4 x 4\ndense sum", "4 x 4 micro\n(new sketch / step)", "4 x 4 micro\n(fixed sketch)"]
loss = [3.6126, 3.6103, 3.6078, 3.6201]
peak = [3.75, 1.93, 1.65, 1.65]
ax[0].bar(names, loss, color=["C0", "C7", "C2", "C1"]); ax[0].set_ylim(3.59, 3.625)
for i, x in enumerate(loss):
    ax[0].text(i, x + 0.0005, f"{x:.4f}", ha="center", fontsize=8)
ax[0].set_title("110M, 1000 steps: loss on the same 974 windows\n(axis cut; run-to-run spread ~0.01)", fontsize=10)
ax[0].tick_params(axis="x", labelsize=8)
ax[1].bar(names, peak, color=["C0", "C7", "C2", "C1"]); ax[1].set_title("peak GPU memory (GiB)")
ax[1].tick_params(axis="x", labelsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "oct5_micro.png"), dpi=130)
print("wrote docs/figures/oct5_ideas.png, oct5_micro.png")
