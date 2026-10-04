"""Figures for 4 Oct: width sweep (does the rank TTF needs grow with width?), capacity runs on FineWeb-Edu (low rank with
more compute), memory (27B / 15B on one 16 GB card, host memory, precision of the accumulator).

    python scripts/plots/plot_oct4.py DATA_DIR
DATA_DIR holds the width-sweep metrics (w*_r*.jsonl: "dg_" and "val_loss" lines), the 2x / 4x capacity curves
(fwc_*.jsonl), myriad.txt (batch-16 shadow runs, "== RUN" headers) and optionally qm8a*_s0.txt curves; finished runs
are read from checkpoints/final. Writes docs/figures/oct4_width.png, oct4_capacity.png, oct4_memory.png.
"""
import contextlib, io, json, os, re, runpy, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1]
FIN, OUT = "checkpoints/final", "docs/figures"
WIDTH = {"w110m": 768, "w340m": 1024, "w1b": 2048, "w10b": 5120, "w15b": 6144}
NAME = {"w110m": "110M", "w340m": "340M", "w1b": "1.3B", "w10b": "~10B", "w15b": "~15B"}


def jl(path):
    """metrics lines -> (val {step: loss}, dg {step: record})"""
    val, dg = {}, {}
    if not os.path.exists(path):
        return val, dg
    for l in open(path):
        l = l.strip()
        if not l.startswith("{"):
            continue
        try:
            j = json.loads(l.replace("NaN", "null"))
        except json.JSONDecodeError:
            continue
        if "val_loss" in j:
            val[j["step"]] = j["val_loss"]
        if any(k.startswith("dg_") for k in j):
            dg[j["step"]] = j
    return val, dg


def myriad(run):
    txt = open(os.path.join(D, "myriad.txt")).read() if os.path.exists(os.path.join(D, "myriad.txt")) else ""
    part = txt.split(f"== {run}\n")[1].split("== ")[0] if f"== {run}\n" in txt else ""
    tmp = os.path.join(D, f".{run}.tmp")
    open(tmp, "w").write(part)
    return jl(tmp)


def width_fig():
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    # (a) val loss at step 1000 (batch 2 x 2048, 4M tokens) minus ranks 256 of the same width
    for r, c in [(64, "tab:red"), (512, "tab:green")]:
        xs, ys = [], []
        for p, w in WIDTH.items():
            v, _ = jl(os.path.join(D, f"{p}_r{r}.jsonl")); v0, _ = jl(os.path.join(D, f"{p}_r256.jsonl"))
            if 1000 in v and 1000 in v0:
                xs.append(w); ys.append(v[1000] - v0[1000])
        ax[0].plot(xs, ys, "o-", color=c, label=f"ranks {r} - ranks 256")
    ax[0].axhline(0, color="tab:blue", lw=1)
    ax[0].set_xscale("log"); ax[0].minorticks_off(); ax[0].set_xticks(list(WIDTH.values()))
    ax[0].set_xticklabels([f"{NAME[p]}\n{w}" for p, w in WIDTH.items()], fontsize=8)
    ax[0].set_ylabel("val loss difference at step 1000"); ax[0].set_title("width sweep, batch 2 x 2048 (early: 4M tokens)")
    ax[0].legend(fontsize=8)
    # (b) share of the full-rank accumulator's energy in its top 64 / 512 directions (middle block q)
    for key, c, lab in [("dg_full_top64", "tab:red", "top 64"), ("dg_full_top512", "tab:green", "top 512")]:
        xs, ys = [], []
        for p, w in WIDTH.items():
            _, dg = jl(os.path.join(D, f"{p}_r256.jsonl"))
            s = 750 if 750 in dg else max(dg, default=None)
            if s is not None and key in dg[s]:
                xs.append(w); ys.append(dg[s][key])
        ax[1].plot(xs, ys, "o-", color=c, label=f"{lab}, batch 2 (step 750)")
    b16 = []
    _, dg110 = jl(os.path.join(FIN, "ts512an05_shadows_s0", "metrics.jsonl"))
    for w, dg in [(768, dg110), (1024, myriad("x340_shadows")[1]), (2048, myriad("x1b_shadows")[1])]:
        s = 1000 if 1000 in dg else max(dg, default=None)
        if s is not None and "dg_full_top512" in dg[s]:
            b16.append((w, dg[s]["dg_full_top512"], dg[s].get("dg_full_top64")))
    if b16:
        ax[1].plot([b[0] for b in b16], [b[1] for b in b16], "s--", color="tab:green", mfc="white", label="top 512, batch 16 (step 1000)")
        ax[1].plot([b[0] for b in b16], [b[2] for b in b16], "s--", color="tab:red", mfc="white", label="top 64, batch 16")
    ax[1].set_xscale("log"); ax[1].minorticks_off(); ax[1].set_xticks(list(WIDTH.values())); ax[1].set_xticklabels([str(w) for w in WIDTH.values()], fontsize=8)
    ax[1].set_ylim(0, 1.02); ax[1].set_xlabel("width"); ax[1].set_ylabel("energy share")
    ax[1].set_title("full-rank accumulator: energy in its top directions"); ax[1].legend(fontsize=7)
    # (c) cosine of the rank-r accumulator with the full-rank one (same gradients), q and down of the middle block
    for r, c in [(64, "tab:red"), (256, "tab:blue"), (1024, "tab:green")]:
        for suf, ls in [("", "-"), ("_down", ":")]:
            xs, ys = [], []
            for p, w in WIDTH.items():
                _, dg = jl(os.path.join(D, f"{p}_r256.jsonl"))
                s = 750 if 750 in dg else max(dg, default=None)
                k = f"dg_r{r}_cos{suf}"
                if s is not None and k in dg[s] and r < w:          # r >= width is full rank
                    xs.append(w); ys.append(dg[s][k])
            ax[2].plot(xs, ys, "o" + ls, color=c, label=f"ranks {r} {'down' if suf else 'q'}")
    ax[2].set_xscale("log"); ax[2].minorticks_off(); ax[2].set_xticks(list(WIDTH.values())); ax[2].set_xticklabels([str(w) for w in WIDTH.values()], fontsize=8)
    ax[2].set_ylim(0, 1.02); ax[2].set_xlabel("width"); ax[2].set_ylabel("cosine with full rank")
    ax[2].set_title("rank-r accumulator vs full rank (batch 2, step 750)"); ax[2].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "oct4_width.png"), dpi=130); plt.close(fig)


def capacity_fig():
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    runs = {}
    for r in ("r512_1x", "r128_1x", "r64_1x"):
        runs[r] = jl(os.path.join(FIN, f"fwc_{r}", "metrics.jsonl"))[0]
    for r in ("r128_2x", "r64_2x", "r128_4x", "r64_4x"):
        runs[r] = jl(os.path.join(D, f"fwc_{r}.jsonl"))[0]
    col = {"r512": "tab:green", "r128": "tab:blue", "r64": "tab:red"}
    ls = {"1x": "-", "2x": "--", "4x": ":"}
    for r, v in runs.items():
        if not v:
            continue
        rk, mult = r.split("_")
        s = sorted(k for k in v if k >= 250)
        ax[0].plot(np.array(s) * 32768 / 1e9, [v[k] for k in s], ls[mult], color=col[rk], label=f"ranks {rk[1:]} {mult}")
    ref = runs["r512_1x"]
    if ref:
        ax[0].axhline(ref[max(ref)], color="tab:green", lw=0.8, alpha=0.6)
    ax[0].set_ylim(3.1, 4.2); ax[0].set_xlabel("training tokens (B)"); ax[0].set_ylabel("FineWeb-Edu val loss")
    ax[0].set_title("110M on FineWeb-Edu: low rank with 1x / 2x / 4x compute (own schedules)"); ax[0].legend(fontsize=8)
    # compute multiple of the 1x runs: steps / steps ranks 512 needed for the same loss
    rs = np.array(sorted(k for k in ref if k >= 250)); rl = np.array([ref[k] for k in rs])
    for r in ("r128_1x", "r64_1x"):
        v = runs[r]; xs, ys = [], []
        for s in sorted(k for k in v if k >= 1000):
            i = np.argmax(rl <= v[s])
            if rl[i] <= v[s] and i > 0:
                n = rs[i - 1] + (rl[i - 1] - v[s]) / (rl[i - 1] - rl[i]) * (rs[i] - rs[i - 1])
                xs.append(s); ys.append(s / n)
        ax[1].plot(xs, ys, "-", color=col[r.split("_")[0]], label=f"ranks {r.split('_')[0][1:]}")
    ax[1].set_xlabel("step (1x schedule)"); ax[1].set_ylabel("compute multiple vs ranks 512")
    ax[1].set_title("steps needed / steps ranks 512 needed for the same loss"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "oct4_capacity.png"), dpi=130); plt.close(fig)


def memory_fig():
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    # (a) measured on one RTX 4070 Ti Super 16 GB, micro-batch 1 x 2048 (RUNS.md 3 Oct 07:20, 4 Oct 06:40 / 07:40):
    # torch peak GiB, host max RSS GiB
    meas = [("26.5B r256\n3 Oct", 14.10, 13.7), ("26.5B r256\n4 Oct", 11.08, 11795760 / 2**20),
            ("26.5B r512\n4 Oct", 11.12, 17843912 / 2**20), ("15B r64\n4 Oct", 9.40, 6102528 / 2**20), ("15B r256\n4 Oct", 9.45, 7992312 / 2**20)]
    x = np.arange(len(meas))
    ax[0].bar(x - 0.2, [m[1] for m in meas], 0.4, color="tab:blue", label="GPU peak (torch)")
    ax[0].bar(x + 0.2, [m[2] or 0 for m in meas], 0.4, color="tab:orange", label="host RSS")
    ax[0].axhline(16, color="k", lw=0.8, ls="--"); ax[0].text(-0.4, 16.2, "16 GB card", fontsize=8)
    ax[0].set_xticks(x); ax[0].set_xticklabels([m[0] for m in meas], fontsize=8); ax[0].set_ylabel("GiB")
    ax[0].set_title("measured, one 16 GB GPU (4 Oct: lean tail, async offload, exact pinned memory)", fontsize=9); ax[0].legend(fontsize=8)
    # (b) host memory at 27B by rank: torch's power-of-two pinned blocks vs exact allocation (scripts/analysis/memory_27b.py)
    with contextlib.redirect_stdout(io.StringIO()):
        M = runpy.run_path("scripts/analysis/memory_27b.py")
    rk = [256, 512, 1024]
    ax[1].plot(rk, [M["pinned_pow2_gib"](r, r) + M["host_base"] for r in rk], "o-", color="tab:red", label="torch pinned (int8 / fp16)")
    ax[1].plot(rk, [M["state_gib"](r, r) + M["host_base"] for r in rk], "o-", color="tab:green", label="exact (int8 / fp16)")
    ax[1].plot(rk, [M["state_gib"](r, r, 1, 1.5) + M["host_base"] for r in rk], "o--", color="tab:green", label="exact, 12-bit accumulator")
    ax[1].plot([256, 512], [11795760 / 2**20, 17843912 / 2**20], "k*", ms=10, label="measured (exact)")
    ax[1].plot([256], [13.7], "rx", ms=9, label="measured 3 Oct (torch pinned)")
    ax[1].axhline(30 / 1.074, color="k", lw=0.8, ls="--"); ax[1].text(260, 28.3, "30 GB host", fontsize=8)
    ax[1].set_xscale("log", base=2); ax[1].set_xticks(rk); ax[1].set_xticklabels(rk)
    ax[1].set_xlabel("ranks"); ax[1].set_ylabel("host GiB"); ax[1].set_title("26.5B: host memory by rank"); ax[1].legend(fontsize=7)
    # (c) precision of the accumulator: val loss minus int8 / fp16 (same seed and windows) along training
    base = jl(os.path.join(FIN, "qm8a16_s0", "metrics.jsonl"))[0]
    for r, lab, c in [("qm8abf_s0", "7 bits (bf16)", "tab:green"), ("qm8af5_s0", "5 bits", "tab:blue"),
                      ("qm8af4_s0", "4 bits", "tab:orange"), ("qm8af3_s0", "3 bits", "tab:red"), ("qm8a12_s0", "12-bit store (6 bits)", "tab:purple")]:
        v = jl(os.path.join(FIN, r, "metrics.jsonl"))[0] or jl(os.path.join(D, f"{r}.jsonl"))[0]
        s = sorted(k for k in v if k in base and k >= 1000)
        if s:
            ax[2].plot(s, [v[k] - base[k] for k in s], color=c, label=lab)
    ax[2].axhline(0, color="k", lw=0.8); ax[2].set_ylim(-0.02, 0.16)
    ax[2].set_xlabel("step"); ax[2].set_ylabel("val loss - int8 / fp16 state")
    ax[2].set_title("accumulator mantissa bits (direction int8; 110M, training-time)"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "oct4_memory.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    width_fig(); capacity_fig(); memory_fig()
    print("wrote", ", ".join(os.path.join(OUT, f"oct4_{n}.png") for n in ("width", "capacity", "memory")))
