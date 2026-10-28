"""Does the state TTF needs grow with the weight count, or slower?

    python scripts/analysis/rank_scaling.py RANK_SCALING.csv [OUT.png]     (the paper's data: rank_scaling.csv)

CSV columns: size,rank,steps,ttf,master,use[,note] (use 0 keeps a row out of the fits). Break-even with master at a
fixed share of the weights fixes one point of each size's curve; the question here is the shape below it. If the loss a
rank costs depends on the rank as a share of the width, the needed state is a fixed share of the weights (linear in
the weight count); if it depends on the absolute rank, the needed state grows only with the width (as the square root
of the weight count at fixed depth / width ratio). The fits put an exponent between the two:

  power: L(r, w) = Linf[size] + c (w / 768)^(g b) (r^-b - w^-b) (r in units of 768; 0 at full rank r = w)
         g = 1: the cost depends on r / w only;  g = 0: on r only
  log:   L(r, w) = Linf[size] + c (w / 768)^-k log2(w / r)   (a fixed cost per halving that may shrink with width)

Linf (the full-rank loss) is free for every size. Each size with two ranks adds one constraint on the shared parameters,
so until a size has three or more ranks, g is set mostly by 110M against one slope per larger size; the profile over g
(sum of squares against g, noise 0.004 per run from the two 110M seeds) shows how well it is pinned down. When a size
has runs of several lengths, a factor (steps / 9155)^e on the rank cost is fitted too (is the cost a matter of size or
of how far training got?). Then: the cost of each halving per size, at fixed rank and at fixed share; the share each
size needs for a given loss; and the predicted cost at the 26.5B shape (width 5120).
"""
import csv
import math
import sys

import numpy as np
from scipy.optimize import least_squares

WIDTH = {"110M": 768, "340M": 1024, "1.3B": 2048, "2.9B": 3072, "26.5B": 5120}
SIGMA = 0.004
rows = [r for r in csv.DictReader(open(sys.argv[1]))]
pts = [dict(size=r["size"], r=float(r["rank"]), w=WIDTH[r["size"]], D=float(r["steps"]), L=float(r["ttf"]),
            M=float(r["master"]), use=r.get("use", "1") == "1", note=r.get("note", "")) for r in rows]
fitp = [p for p in pts if p["use"]]
groups = sorted({(p["size"], p["D"]) for p in fitp}, key=lambda s: (WIDTH[s[0]], s[1]))
gi = {s: i for i, s in enumerate(groups)}
multiD = len({p["D"] for p in fitp}) > 1


def cost(p, th, form, g=None):
    """the rank's cost above the full-rank loss for point p under parameters th = (c, b or k, [g], [e])"""
    w, r, D = p["w"] / 768, p["r"] / 768, p["D"] / 9155
    if form == "power":
        c, b = th[0], th[1]
        e = th[-1] if multiD else 0.0      # e is the last shared parameter, whether g is fitted or fixed
        g = th[2] if g is None else g
        return c * w ** (g * b) * (r ** -b - w ** -b) * D ** e
    c, k = th[0], th[1]
    e = th[2] if multiD else 0.0
    return c * w ** -k * math.log2(w / r) * D ** e


def fit(form, g=None):
    """least squares over the shared parameters and one full-rank loss per (size, steps) group"""
    nsh = (2 if g is not None else 3) + (1 if multiD else 0) if form == "power" else 2 + (1 if multiD else 0)

    def res(x):
        th, Linf = x[:nsh], x[nsh:]
        return np.array([(Linf[gi[(p["size"], p["D"])]] + cost(p, th, form, g) - p["L"]) / SIGMA for p in fitp])
    x0 = ([0.05, 1.0] + ([] if g is not None else [0.5]) if form == "power" else [0.05, 0.0]) + ([0.0] if multiD else [])
    x0 += [min(p["L"] for p in fitp if (p["size"], p["D"]) == s) - 0.01 for s in groups]
    lo = ([0, 0.05] + ([] if g is not None else [-3.0]) if form == "power" else [0, -2.0]) + ([-2.0] if multiD else [])
    hi = ([10, 6] + ([] if g is not None else [2.5]) if form == "power" else [10, 3.0]) + ([2.0] if multiD else [])
    lo += [-np.inf] * len(groups); hi += [np.inf] * len(groups)
    s = least_squares(res, x0, bounds=(lo, hi))
    return s.x[:nsh], dict(zip(groups, s.x[nsh:])), float((s.fun ** 2).sum())


def main():
    print("== cost per halving of the rank (loss difference / log2 of the rank ratio), adjacent ranks per size ==")
    for s in groups:
        P = sorted([p for p in fitp if (p["size"], p["D"]) == s], key=lambda p: p["r"])
        by = {}
        for p in P:
            by.setdefault(p["r"], []).append(p["L"])
        rs = sorted(by)
        for a, b in zip(rs, rs[1:]):
            la, lb = np.mean(by[a]), np.mean(by[b])
            print(f"  {s[0]:5s} {int(s[1]):5d} steps: rank {int(a):4d} -> {int(b):4d} (share {a / WIDTH[s[0]]:.3f} -> "
                  f"{b / WIDTH[s[0]]:.3f}): {(la - lb) / math.log2(b / a):+.4f} per halving")

    print("\n== fits (chi2 with noise 0.004 per run; n points %d) ==" % len(fitp))
    thp, Lp, chip = fit("power")
    thl, Ll, chil = fit("log")
    print(f"  power: c {thp[0]:.4f} b {thp[1]:.3f} g {thp[2]:+.3f}" + (f" e {thp[3]:+.3f}" if multiD else "")
          + f" | chi2 {chip:.1f}")
    print(f"  log:   c {thl[0]:.4f} k {thl[1]:+.3f}" + (f" e {thl[2]:+.3f}" if multiD else "") + f" | chi2 {chil:.1f}")
    for s in groups:
        mk = [p["M"] for p in fitp if (p["size"], p["D"]) == s][0]
        print(f"    full-rank loss {s[0]:5s} {int(s[1])}: power {Lp[s]:.4f} / log {Ll[s]:.4f} (master {mk:.4f}: full-rank "
              f"lead {Lp[s] - mk:+.4f} / {Ll[s] - mk:+.4f})")

    print("\n== profile over g (power form): g = 1 cost set by the share, g = 0 by the rank ==")
    prof = []
    for g in np.round(np.arange(-2.0, 1.81, 0.1), 2):
        th, _, chi = fit("power", g=g)
        prof.append((g, chi, th))
        print(f"  g {g:+.1f}: chi2 {chi:7.2f} (b {th[1]:.2f})")
    cmin = min(c for _, c, _ in prof)
    ok68 = [g for g, c, _ in prof if c - cmin <= 1.0]
    ok95 = [g for g, c, _ in prof if c - cmin <= 3.84]
    print(f"  g within 68%: {min(ok68):+.1f} .. {max(ok68):+.1f}; within 95%: {min(ok95):+.1f} .. {max(ok95):+.1f}")
    print("  needed state for a fixed loss cost: share of the weights ~ width^(g - 1); total ~ weights x width^(g - 1)")

    print("\n== predicted rank cost at the 26.5B shape (width 5120), power form at g from the 95% range, and log form ==")
    W = WIDTH["26.5B"]
    for r in (256, 512, 1024, 1536, 2048, 2560, 3584):
        out = []
        for g, chi, th in prof:
            if g in (min(ok95), round(thp[2], 1), max(ok95)) or abs(g - thp[2]) < 0.05:
                out.append(f"g {g:+.1f}: {cost(dict(w=W, r=r, D=9155), th, 'power', g):+.4f}")
        lg = cost(dict(w=W, r=r, D=9155), thl, "log")
        print(f"  rank {r:4d} (share {r / W:.2f}): " + " | ".join(dict.fromkeys(out)) + f" | log {lg:+.4f}")

    if len(sys.argv) > 2:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(13, 5))
        for s in groups:
            P = sorted([p for p in fitp if (p["size"], p["D"]) == s], key=lambda p: p["r"])
            lab = f"{s[0]}" + (f" {int(s[1])} steps" if multiD else "")
            for k, x in enumerate((lambda p: p["r"], lambda p: p["r"] / p["w"])):
                l, = axs[k].plot([x(p) for p in P], [p["L"] - Lp[s] for p in P], "o", label=lab)
                rr = np.geomspace(min(p["r"] for p in P) * 0.8, P[0]["w"], 60)
                axs[k].plot([x(dict(r=v, w=P[0]["w"])) for v in rr],
                            [cost(dict(w=P[0]["w"], r=v, D=s[1]), thp, "power") for v in rr], "-", color=l.get_color())
        for k, lab in enumerate(("rank", "rank / width")):
            axs[k].set_xscale("log", base=2); axs[k].set_xlabel(lab); axs[k].grid(alpha=0.3); axs[k].legend()
            axs[k].set_ylabel("loss above the fitted full-rank loss")
        fig.suptitle(f"Rank cost across sizes: power fit g = {thp[2]:.2f} (1: share of width, 0: absolute rank)")
        plt.tight_layout(); plt.savefig(sys.argv[2], dpi=110); print(sys.argv[2])


if __name__ == "__main__":
    main()
