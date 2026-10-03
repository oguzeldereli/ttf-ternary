"""Figures for 3 Oct: scale + final evaluation, low rank (fixes, shadows, moves), lean memory.

    python scripts/plots/plot_oct3.py CURVES_DIR      (CURVES_DIR: validation curves pulled from the lab PCs / Myriad)
writes docs/figures/oct3_scale_eval.png, oct3_lowrank.png, oct3_memory.png
"""
import glob, json, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CUR = sys.argv[1]
FIN = "checkpoints/final"
OUT = "docs/figures"
TOK = 32768 / 1e6          # million tokens per step


def curve(path):
    d = {}
    if not os.path.exists(path):
        return d
    for l in open(path):
        m = re.search(r'"step": (\d+), "val_loss": ([\d.]+)', l)
        if m:
            d[int(m.group(1))] = float(m.group(2))
    return d


def myr(name):
    txt = open(os.path.join(CUR, "myr_1b.txt")).read()
    blk = txt.split("== " + name)[1].split("== ")[0]
    return {int(a): float(b) for a, b in re.findall(r'"step": (\d+), "val_loss": ([\d.]+)', blk)}


def fe(run):
    p = os.path.join(FIN, run, "final_eval.json")
    return json.load(open(p)) if os.path.exists(p) else None


def plot_curve(ax, d, label, **kw):
    s = sorted(k for k in d if k >= 500)
    if s:
        ax.plot([k * TOK for k in s], [d[k] for k in s], label=label, **kw)


# ---------------------------------------------------------------- figure 1: scale and final evaluation
fig, axs = plt.subplots(2, 2, figsize=(15, 10.5))
ax = axs[0, 0]
for name, lab, kw in (("x1b_ts512_la1000an05", "TTF ranks 512", dict(color="C0", lw=2)),
                      ("x1b_master_lr75", "master weights", dict(color="k", lw=1.5)),
                      ("x1b_dryspend_r512_lr75", "earlier flip rule", dict(color="C7", lw=1.2, ls="--"))):
    plot_curve(ax, myr(name), lab, **kw)
ax.set_ylim(2.5, 3.4); ax.set_title("1.3B: validation loss (training-time eval)")
f_t, f_m = fe("x1b_ts512_la1000an05"), fe("x1b_master_lr75")
if f_t and f_m:
    ax.text(0.98, 0.70, f"fixed windows, Wikipedia: TTF {f_t['wiki_loss']:.4f} vs master {f_m['wiki_loss']:.4f}\n"
            f"FineWeb-Edu: {f_t['fw_loss']:.4f} vs {f_m['fw_loss']:.4f}", transform=ax.transAxes, ha="right", fontsize=9,
            bbox=dict(fc="w", ec="0.8"))
ax.set_xlabel("tokens (M)"); ax.set_ylabel("val loss"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axs[0, 1]
for path, lab, kw in ((f"{FIN}/big_master/metrics.jsonl", "master weights", dict(color="k", lw=1.5)),
                      (f"{CUR}/mx_big_extras_s0.txt", "master + same float extras (running)", dict(color="k", ls="--")),
                      (f"{FIN}/big_ts16rs512tau1000_s0/metrics.jsonl", "TTF ranks 512 (p 1)", dict(color="C0", lw=2)),
                      (f"{CUR}/big_ts512an05_s0.txt", "TTF ranks 512, p 0.5 (running)", dict(color="C2")),
                      (f"{CUR}/big_ts512la1000an05_s0.txt", "TTF ranks 512, p 0.5 + scaled leak (running)", dict(color="C3")),
                      (f"{FIN}/big_ts16r256rs256tau1000_s0/metrics.jsonl", "TTF ranks 256", dict(color="C1", ls=":")),
                      (f"{FIN}/big_ts16r128rs128tau1000_s0/metrics.jsonl", "TTF ranks 128", dict(color="C4", ls=":"))):
    plot_curve(ax, curve(path), lab, **kw)
ax.set_ylim(2.55, 3.3); ax.set_title("340M: validation loss (training-time eval)")
ax.set_xlabel("tokens (M)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axs[1, 0]
groups = [("TTF r512\n(3 seeds)", ["ts16rs512tau1000an05_s0", "ts16rs512tau1000an05_seed2", "ts16rs512tau1000an05_seed3"], "C0"),
          ("TTF int8/fp16\nstate", ["qm8a16_seed2"], "C0"),
          ("master +\nextras", ["mx_extras_s0", "mx_extras_seed2"], "0.3"),
          ("master\n(3 seeds)", ["master_tracked", "mx_seed2", "mx_seed3"], "k"),
          ("TTF k / v\nrank 128", ["ts512an05_kv128_s0"], "C1"),
          ("TTF shared\nbasis", ["ts512an05_sharedV_s0"], "C1"),
          ("earlier\nflip rule", ["gvsharp_dryspend_r512_seed2"], "C7"),
          ("TTF without\nfloat extras", ["tsbare_rs512an05_s0"], "C7")]
x = np.arange(len(groups)); w = 0.38
for j, (key, lab, hatch) in enumerate((("wiki_loss", "Wikipedia val", ""), ("fw_loss", "FineWeb-Edu val (untouched)", "//"))):
    vals, errs, cols = [], [], []
    for _, runs, col in groups:
        v = [fe(r)[key] for r in runs if fe(r) and key in fe(r)]
        vals.append(np.mean(v) if v else np.nan); errs.append(np.std(v) if len(v) > 1 else 0); cols.append(col)
    base = vals[3]
    ax.bar(x + (j - 0.5) * w, np.array(vals) - base, w, yerr=errs, color=cols, hatch=hatch, alpha=0.85, label=lab,
           edgecolor="w")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups], fontsize=8, rotation=20)
ax.set_ylabel("loss minus master (fixed windows)"); ax.set_title("110M, final evaluation on fixed windows (relative to master)")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

ax = axs[1, 1]
sizes = [110, 340, 1300]
m3 = lambda runs, k: np.mean([fe(r)[k] for r in runs])
T3, M3 = ["ts16rs512tau1000an05_s0", "ts16rs512tau1000an05_seed2", "ts16rs512tau1000an05_seed3"], ["master_tracked", "mx_seed2", "mx_seed3"]
ttf = [m3(T3, "wiki_loss") - m3(M3, "wiki_loss"), 2.6268 - 2.6626,
       f_t["wiki_loss"] - f_m["wiki_loss"]]
fw = [m3(T3, "fw_loss") - m3(M3, "fw_loss"), np.nan, f_t["fw_loss"] - f_m["fw_loss"]]
ax.plot(sizes, ttf, "o-", color="C0", lw=2, label="Wikipedia (110M: 3 seeds each; 340M: training-time eval)")
ax.plot([110, 1300], [fw[0], fw[2]], "s--", color="C0", label="FineWeb-Edu (untouched)")
for s_, t_ in zip(sizes, ttf):
    ax.annotate(f"{t_:+.3f}", (s_, t_), textcoords="offset points", xytext=(6, 6), fontsize=9)
ax.axhline(0, color="k", lw=0.8); ax.set_xscale("log"); ax.set_xticks(sizes); ax.set_xticklabels(["110M", "340M", "1.3B"])
ax.set_ylabel("TTF minus master (val loss)"); ax.set_title("TTF vs master weights by model size (300M tokens)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/oct3_scale_eval.png", dpi=110); plt.close(fig)

# ---------------------------------------------------------------- figure 2: low rank
fig, axs = plt.subplots(2, 2, figsize=(15, 10.5))
fixes = [("sketch02", "(a) sketch of u in the basis", "C3"), ("resid", "(b) residual feedback (dense)", "C2"),
         ("th8", "theta 8", "C1"), ("th11", "theta 11", "C5"), ("lr3e3", "lr 3e-3", "C0"), ("typerank", "(c) ranks by input width", "C4")]
for col, rk in enumerate((128, 64)):
    ax = axs[0, col]
    base = curve(f"{FIN}/ts{rk}an05_diag_s0/metrics.jsonl")
    for tag, lab, c in fixes:
        name = f"ts{rk}b_typerank_s0" if tag == "typerank" else f"ts{rk}an05_{tag}_s0"
        d = curve(f"{CUR}/{name}.txt")
        s = sorted(k for k in d if k in base and k >= 500)
        ax.plot([k * TOK for k in s], [d[k] - base[k] for k in s], label=lab, color=c)
    r512 = curve(f"{FIN}/ts512an05_diag_s0/metrics.jsonl")
    s = sorted(k for k in r512 if k in base and k >= 500)
    ax.plot([k * TOK for k in s], [r512[k] - base[k] for k in s], color="k", ls=":", label="ranks 512 (target)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"110M ranks {rk}: loss minus ranks-{rk} baseline"); ax.set_xlabel("tokens (M)")
    ax.set_ylim(-0.3, 0.35); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axs[1, 0]
rows = [json.loads(l) for l in open(os.path.join(CUR, "shadows512.jsonl"))]
last = rows[-1]
names = [("r64", "rank 64"), ("r128", "rank 128"), ("run", "rank 512")]
mets = [("cos", "cosine with\nfull rank"), ("sign_agree_big", "sign agreement\n(|A| >= theta/2)"),
        ("fire_precision", "moves full rank\nwould also make"), ("mom_cos", "momentum cosine\nwith full rank")]
x = np.arange(len(mets)); w = 0.26
for j, (nm, lab) in enumerate(names):
    v = [last.get(f"dg_{nm}_{m}", np.nan) for m, _ in mets]
    ax.bar(x + (j - 1) * w, v, w, label=lab, color=("C3", "C1", "C0")[j])
ax.set_xticks(x); ax.set_xticklabels([m[1] for m in mets], fontsize=9); ax.set_ylim(0, 1.05)
tops = [last.get(f"dg_full_top{k}", np.nan) for k in (64, 128, 512)]
ax.set_title(f"Shadow states vs full rank, same gradients (110M, step {last['step']})\n"
             f"full-rank accumulator energy in top 64 / 128 / 512 directions: {tops[0]:.0%} / {tops[1]:.0%} / {tops[2]:.0%}",
             fontsize=10)
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

ax = axs[1, 1]
TY = ("q", "k", "v", "o", "gate", "up", "down")
def never(run):
    nv = None
    for l in open(f"{FIN}/{run}/metrics.jsonl"):
        if '"never_frac_layers"' in l:
            nv = json.loads(l)["never_frac_layers"]
    return np.array(nv)
x = np.arange(7); w = 0.2
for j, (run, lab, c) in enumerate((("ts16rs512tau1000an05_s0", "TTF ranks 512", "C0"), ("ts512an05_kv128_s0", "k / v at rank 128", "C1"),
                                   ("ts512an05_sharedV_s0", "shared basis", "C3"), ("ts64an05_diag_s0", "ranks 64", "C7"))):
    nv = never(run)
    ax.bar(x + (j - 1.5) * w, [nv[i::7].mean() * 100 for i in range(7)], w, label=lab, color=c)
ax.set_xticks(x); ax.set_xticklabels(TY); ax.set_ylabel("% of weights never moved (end of training)")
ax.set_title("Which layer types stop moving (110M)"); ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(f"{OUT}/oct3_lowrank.png", dpi=110); plt.close(fig)

# ---------------------------------------------------------------- figure 3: lean memory
comp = ["trits", "float params", "float grads", "float optimizer", "TTF state (GPU)"]
data = {"110M": [16, 53, 53, 29, 258], "110M\nstate on host": [16, 53, 53, 29, 0],
        "340M": [59, 78, 78, 46, 696], "340M\nstate on host": [59, 78, 78, 46, 0],
        "1.3B": [235, 156, 156, 92, 1390], "1.3B\nstate on host": [235, 156, 156, 92, 0]}
peak = {"110M": 0.63, "110M\nstate on host": 0.38, "340M": 1.35, "340M\nstate on host": 0.68, "1.3B": 2.97,
        "1.3B\nstate on host": 1.61}
master = {"110M": 110e6 * 16 / 2**30, "340M": 349e6 * 16 / 2**30, "1.3B": 1315e6 * 16 / 2**30}
fig, ax = plt.subplots(figsize=(13, 6))
labels = list(data); x = np.arange(len(labels)); bottom = np.zeros(len(labels))
for i, cname in enumerate(comp):
    v = np.array([data[k][i] for k in labels]) / 1024
    ax.bar(x, v, 0.6, bottom=bottom, label=cname, color=f"C{i}"); bottom += v
ax.plot(x, [peak[k] for k in labels], "kv", ms=9, label="measured peak per step (incl. activations)")
for xx, k in zip(x, labels):
    ax.text(xx, peak[k] + 0.35, f"peak {peak[k]:.2f} GiB\npersistent {sum(data[k]) / 1024:.2f}", ha="center", fontsize=8)
for k, xx in zip(labels, x):
    if "\n" not in k:
        ax.hlines(master[k], xx - 0.35, xx + 1.35, colors="r", linestyles="--")
        ax.text(xx + 0.5, master[k] * 1.03, f"master weights + AdamW: {master[k]:.1f} GiB (before activations)", ha="center",
                fontsize=8, color="r")
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("GiB (PyTorch allocated)")
ax.set_title("Lean TTF training, measured (micro-batch 1 x 2048, ranks 512, int8 / fp16 state, bf16 / fp16 float tail)")
ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(f"{OUT}/oct3_memory.png", dpi=110); plt.close(fig)
print("wrote", [f"{OUT}/oct3_{n}.png" for n in ("scale_eval", "lowrank", "memory")])
