"""TTF's gap to master as a function of the accumulator's state, across model sizes.

    python scripts/analysis/rank_curve.py POINTS.csv OUT.png

POINTS.csv columns: size,rank,ttf,master[,note]  (size: 110M | 340M | 1.3B | 2.9B; rank: the accumulator rank or "full";
ttf / master: losses on the same evaluation). State per weight is computed from the model shape: rank min(r, N, K) factors
for every ternary matrix plus the factored second moment, over the ternary weights. Fits, per size and pooled,
gap = a + b log2(floats per weight), and reports the floats per weight where the fit crosses zero (break-even with master),
the slope per doubling, and the same fit against rank / width (is the needed state a share of the weights, or a rank?).
"""
import csv, sys, math
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SHAPES = {"110M": (768, 12, 12, 2048), "340M": (1024, 16, 16, 2816), "1.3B": (2048, 16, 16, 5632),
          "2.9B": (3072, 24, 8, 8192)}


def fpw(size, r):
    d, h, kv, hid = SHAPES[size]
    kvd = d * kv // h
    mats = [(d, d), (kvd, d), (kvd, d), (d, d), (hid, d), (hid, d), (d, hid)]
    r = 10 ** 9 if r == "full" else int(r)
    W = sum(N * K for N, K in mats)
    return sum(min(r, N, K) * (N + K) + N + K for N, K in mats) / W


rows = list(csv.DictReader(open(sys.argv[1])))
pts = []
for x in rows:
    s, r = x["size"], x["rank"]
    w = SHAPES[s][0]
    rr = w if r == "full" else int(r)
    pts.append((s, r, fpw(s, r), rr / w, float(x["ttf"]) - float(x["master"]), x.get("note", "")))


def fit(xs, ys):
    A = np.vstack([np.ones(len(xs)), np.log2(xs)]).T
    (a, b), *_ = np.linalg.lstsq(A, np.array(ys), rcond=None)
    res = np.array(ys) - A @ np.array([a, b])
    return a, b, 2 ** (-a / b) if b else float("nan"), float(np.sqrt(np.mean(res ** 2)))


fig, axs = plt.subplots(1, 2, figsize=(13, 5))
for s in SHAPES:
    P = [p for p in pts if p[0] == s]
    if not P:
        continue
    x = [p[2] for p in P]; y = [p[4] for p in P]; q = [p[3] for p in P]
    axs[0].plot(x, y, "o-", label=s); axs[1].plot(q, y, "o-", label=s)
    for p in P:
        axs[0].annotate(str(p[1]), (p[2], p[4]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    if len(P) >= 2:
        a, b, be, rms = fit(x, y)
        a2, b2, be2, rms2 = fit(q, y)
        print(f"{s:5s}: gap = {a:+.4f} {b:+.4f} log2(floats/weight); per doubling {b:+.4f}; break-even "
              f"{be:.2f} floats/weight (rms {rms:.4f}) | vs rank/width: per doubling {b2:+.4f}, break-even {be2:.2f}")
for k, (i, lab) in enumerate(((2, "floats per weight (fp32 accumulator + factored v)"), (3, "rank / width"))):
    xs = [p[i] for p in pts if p[0] != "110M"]; ys = [p[4] for p in pts if p[0] != "110M"]
    if len(xs) >= 2:
        a, b, be, rms = fit(xs, ys)
        print(f"pooled 340M-2.9B vs {lab}: per doubling {b:+.4f}, break-even {be:.2f}, rms {rms:.4f}")
    xs = [p[i] for p in pts]; ys = [p[4] for p in pts]
    a, b, be, rms = fit(xs, ys)
    print(f"pooled all sizes vs {lab}: per doubling {b:+.4f}, break-even {be:.2f}, rms {rms:.4f}")
    axs[k].set_xscale("log", base=2); axs[k].axhline(0, color="gray", lw=0.8); axs[k].set_xlabel(lab)
    axs[k].set_ylabel("TTF - master (loss)"); axs[k].grid(alpha=0.3); axs[k].legend()
fig.suptitle("TTF's gap to master against the size of its state")
plt.tight_layout(); plt.savefig(sys.argv[2], dpi=110); print(sys.argv[2])
for p in pts:
    print(f"  {p[0]:5s} rank {p[1]:>5s}: {p[2]:.2f} floats/weight, rank/width {p[3]:.2f}, gap {p[4]:+.4f} {p[5]}")
