"""Why does TTF fall behind master mid-run (goldbug b3, from ~step 3300)? Everything the training logs can say.

(1) Gap TTF - master at equal steps, validation and training loss (EMA), at 2.9B (goldbug), 1.3B (Myriad, ranks 1024 vs
master + extras lr 5e-4) and 110M (ranks 512 / 192 vs master + extras): is the mid-run slip the same shape at every size,
and does it recover at 1.3B / 110M by the end?
(2) The step-size schedules: both arms share the cosine lr ratio rho(t); master's latent step is rho, TTF's trit step scale
is a = rho^p (p 0.5), so TTF's relative step is rho^(p-1) larger and grows as rho falls. Gap slope against d a / dt.
(3) Flips: per-step move rate and the never-moved share, overall and by matrix type (q k v o gate up down) and depth.
    python scripts/analysis/slip_logs.py      -> docs/figures/slip_logs_oct9.png, printed tables
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
TYPES = ["q", "k", "v", "o", "gate", "up", "down"]


def load(path):
    tr, va = {}, {}
    for l in open(path):
        d = json.loads(l)
        if "val_loss" in d:
            va[d["step"]] = d["val_loss"]
        elif "loss" in d:
            tr[d["step"]] = d
    return tr, va


def ema(steps, x, k=200):
    out, m = [], None
    for v in x:
        m = v if m is None else m + (v - m) / k
        out.append(m)
    return np.array(out)


def gap_curves(ttf, mst):
    tt, tv = load(ttf); mt, mv = load(mst)
    sv = sorted(set(tv) & set(mv))
    st = sorted(set(tt) & set(mt))
    gt = ema(st, [tt[s]["loss"] for s in st]) - ema(st, [mt[s]["loss"] for s in st])
    return np.array(sv), np.array([tv[s] - mv[s] for s in sv]), np.array(st), gt, tt


PAIRS = [
    ("2.9B r1536 (goldbug)", "checkpoints/goldbug/gb3_ttf_r1536", "checkpoints/goldbug/gb3_mx_lr4e4"),
    ("2.9B r1024 (goldbug)", "checkpoints/goldbug/gb3_ttf_r1024", "checkpoints/goldbug/gb3_mx_lr4e4"),
    ("1.3B r1024", "checkpoints/myr1b/x1b_nomom1024_lr15", "checkpoints/myr1b/x1b_mx_extras_lr50"),
    ("110M r512", "checkpoints/final/ts512an05_nomom_s0", "checkpoints/final/mx_extras_s0"),
    ("110M r192", "checkpoints/final/sw192_base", "checkpoints/final/mx_extras_s0"),
]

fig, ax = plt.subplots(2, 3, figsize=(18, 10))
print("== (1) val gap TTF - master at equal steps")
G = {}
for name, a, b in PAIRS:
    pa, pb = os.path.join(R, a, "metrics.jsonl"), os.path.join(R, b, "metrics.jsonl")
    if not (os.path.exists(pa) and os.path.exists(pb)):
        print(f"  {name}: missing"); continue
    sv, gv, st, gt, tt = gap_curves(pa, pb)
    G[name] = (sv, gv, st, gt, tt)
    pick = [s for s in (1000, 2000, 3000, 3500, 4000, 5000, 6000, 7000, 8000, 9000, 9155) if s in set(sv)]
    print(f"  {name:22s} " + "  ".join(f"{s}:{gv[list(sv).index(s)]:+.3f}" for s in pick))
    ax[0, 0].plot(sv, gv, marker=".", label=name)
    ax[0, 1].plot(st, gt, label=name)
for i, t in enumerate(("validation gap TTF - master (equal steps)", "training-loss gap (EMA 200 steps)")):
    ax[0, i].axhline(0, color="k", lw=0.8); ax[0, i].set_title(t); ax[0, i].set_xlabel("step"); ax[0, i].legend(fontsize=8)
    ax[0, i].set_ylim(-0.12, 0.12)

# (2) schedules: rho from master's logged lr, a = rho^0.5
print("== (2) schedule: rho (both arms), TTF step scale a = rho^0.5, relative step a / rho")
_, _, _, _, mt = gap_curves(os.path.join(R, PAIRS[0][2], "metrics.jsonl"), os.path.join(R, PAIRS[0][2], "metrics.jsonl"))
ms = sorted(mt); lr = np.array([mt[s]["lr"] for s in ms]); rho = lr / lr.max(); a = np.sqrt(rho)
ax[0, 2].plot(ms, rho, label="rho = lr / peak (master's step)")
ax[0, 2].plot(ms, a, label="a = rho^0.5 (TTF's trit step scale)")
ax[0, 2].plot(ms, a / np.maximum(rho, 1e-9) / 10, label="a / rho / 10 (TTF's step relative to master's)")
ax[0, 2].legend(fontsize=8); ax[0, 2].set_title("schedules (goldbug)"); ax[0, 2].set_xlabel("step")
for s in (1000, 3000, 4000, 6000, 8000, 9154):
    i = ms.index(s) if s in ms else None
    if i is not None:
        print(f"  step {s}: rho {rho[i]:.3f}  a {a[i]:.3f}  a/rho {a[i] / rho[i]:.2f}")
if "2.9B r1536 (goldbug)" in G:
    sv, gv, *_ = G["2.9B r1536 (goldbug)"]
    m = sv >= 1000
    sl = np.gradient(gv[m], sv[m]) * 1000          # gap change per 1000 steps
    ai = np.interp(sv[m], ms, a); da = np.gradient(ai, sv[m]) * 1000
    ri = np.interp(sv[m], ms, rho)
    print(f"  2.9B r1536 from step 1000: corr(gap slope, da/dt) {np.corrcoef(sl, da)[0, 1]:+.2f}, "
          f"corr(gap, a/rho) {np.corrcoef(gv[m], ai / ri)[0, 1]:+.2f}")

# (3) flips
print("== (3) moves per step and never-moved share (by matrix type, then by depth thirds)")
for name, col in (("2.9B r1536 (goldbug)", 0), ("1.3B r1024", 1)):
    if name not in G:
        continue
    tt = G[name][4]; ss = sorted(tt)
    fr = np.array([tt[s]["flip_frac"] for s in ss]); nv = np.array([tt[s]["never_frac"] for s in ss])
    ax[1, 0].plot(ss, ema(ss, fr, 50) * 100, label=f"{name} moves / step (%)")
    ax[1, 1].plot(ss, nv, label=f"{name} never moved")
    L = np.array([tt[s]["flip_frac_layers"] for s in ss]); N = np.array([tt[s]["never_frac_layers"] for s in ss])
    nl = L.shape[1] // 7
    pick = [s for s in (1000, 2000, 3000, 4000, 6000, 8000, 9000) if s in tt]
    print(f"  {name}: moves / step (%) by type at steps {pick}")
    for j, t in enumerate(TYPES):
        idx = [ss.index(s) for s in pick]
        cols = ema(ss, L[:, j::7].mean(1), 50)
        print(f"    {t:5s} " + " ".join(f"{cols[i] * 100:.4f}" for i in idx) +
              f" | never {N[idx[-1], j::7].mean():.3f}")
        if col == 0:
            ax[1, 2].plot(ss, ema(ss, L[:, j::7].mean(1), 50) * 100, label=t)
    th = [range(0, nl // 3), range(nl // 3, 2 * nl // 3), range(2 * nl // 3, nl)]
    i = ss.index(pick[-1])
    print("    depth thirds, never moved at step", pick[-1], ":",
          " ".join(f"{N[i, [7 * l + j for l in r for j in range(7)]].mean():.3f}" for r in th))
for k, t in ((0, "moves per step (%), EMA 50"), (1, "never-moved share"), (2, "2.9B r1536 moves / step (%) by type")):
    ax[1, k].set_title(t); ax[1, k].set_xlabel("step"); ax[1, k].legend(fontsize=8)
plt.tight_layout()
out = os.path.join(R, "docs/figures/slip_logs_oct9.png"); plt.savefig(out, dpi=110); print("wrote", out)
