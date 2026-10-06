"""110M piecewise-p arms (6 Oct): the step scale a, its change a' and the change of its change a'' per step, next to the
gap to master + extras, its slope and its curvature, on one step axis. Right: the gap's slope against a' and the gap's
curvature against a'', one point per evaluation from step 1500 (ranks 512, training-time windows). The gap's slope and
curvature are local quadratic fits over 7 evaluations (1500 steps); a' and a'' are averaged over the same window so both
sides are smoothed alike.
    python scripts/plots/plot_pp_rate2.py CURVE_DIR      (same files as plot_pp.py) -> docs/figures/pp_rate2_oct6.png"""
import json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

D = sys.argv[1]
T, W, MIN, HW = 9155, 305, 0.1, 750          # steps, warmup, final lr ratio, half-width of the smoothing window
ARMS = [  # name, p points, colour, style, width
    ("base", "0:0.5,9155:0.5", "k", "-", 2.4),
    ("p1", "0:1,9155:1", "k", ":", 1.8),
    ("pp_A", "0:1.3,3000:1.5,9155:0.5", "C0", "-", 1.4),
    ("pp_B", "0:1.3,3000:1.5,9155:0.25", "C1", "-", 1.4),
    ("pp_C", "0:1.3,3000:1.5,9155:0", "C2", "-", 1.4),
    ("pp_D", "0:1.3,3000:2.0,9155:0.25", "C3", "-", 1.4),
    ("pp_E", "0:1.0,3000:1.2,9155:0.25", "C4", "-", 1.4),
    ("pp_F", "0:1.3,3000:1.5,6000:1.5,9155:0.25", "C5", "-", 1.4),
]


def val(p):
    out = {}
    for l in open(p):
        if '"val_loss"' in l:
            j = json.loads(l); out[j["step"]] = j["val_loss"]
    return out


def rho(t):
    if t < W:
        return (t + 1) / W
    return MIN + 0.5 * (1 - MIN) * (1 + math.cos(math.pi * (min(t, T) - W) / (T - W)))


def p_at(pts, t):
    pts = sorted((float(a), float(b)) for a, b in (x.split(":") for x in pts.split(",")))
    for (s0, p0), (s1, p1) in zip(pts, pts[1:]):
        if t <= s1:
            return p0 + (p1 - p0) * (t - s0) / (s1 - s0)
    return pts[-1][1]


def scale(pts, t):
    return rho(t) ** p_at(pts, t)


def d1(pts, t, h=HW):        # mean of a' over [t - h, t + h], per 1000 steps
    lo, hi = max(W, t - h), min(T, t + h)
    return (scale(pts, hi) - scale(pts, lo)) / (hi - lo) * 1000


def d2(pts, t, h=HW):        # mean of a'' over [t - h, t + h], per 1000 steps squared
    lo, hi = max(W + 10, t - h), min(T - 10, t + h)
    g = lambda x: (scale(pts, x + 10) - scale(pts, x - 10)) / 20 * 1000
    return (g(hi) - g(lo)) / (hi - lo) * 1000


m = val("checkpoints/final/mx_extras_s0/metrics.jsonl")
fig = plt.figure(figsize=(18, 16))
gs = GridSpec(6, 2, width_ratios=[1.6, 1], figure=fig)
ax = [fig.add_subplot(gs[i, 0]) for i in range(6)]
sc1, sc2 = fig.add_subplot(gs[0:3, 1]), fig.add_subplot(gs[3:6, 1])
ts = np.arange(W + 10, T - 10, 10)
P1, P2 = [], []
for n, pts, c, st, lw in ARMS:
    a = np.array([scale(pts, t) for t in ts])
    da = np.gradient(a, ts) * 1000
    dda = np.gradient(da, ts) * 1000
    ax[0].plot(ts, a, st, color=c, lw=lw, label=n.replace("pp_", ""))
    ax[1].plot(ts, da, st, color=c, lw=lw)
    ax[2].plot(ts, [d2(pts, t, 50) for t in ts], st, color=c, lw=lw)
    f = os.path.join(D, f"sw512_{n}.jsonl")
    if not os.path.exists(f):
        continue
    v = val(f); s = np.array(sorted(x for x in v if x in m and x >= 500))
    gap = np.array([v[x] - m[x] for x in s])
    ax[3].plot(s, gap, st, color=c, lw=lw)
    k = 3; ss, sl, cu = [], [], []
    for i in range(k, len(s) - k):
        x0 = s[i]; q = np.polyfit((s[i - k:i + k + 1] - x0) / 1000, gap[i - k:i + k + 1], 2)
        ss.append(x0); sl.append(q[1]); cu.append(2 * q[0])
    ax[4].plot(ss, sl, st, color=c, lw=lw); ax[5].plot(ss, cu, st, color=c, lw=lw)
    mk = "o" if st == "-" else "s"
    for x, y1, y2 in zip(ss, sl, cu):
        if x >= 1500:
            e1, e2 = d1(pts, x), d2(pts, x)
            sc1.scatter(e1, y1, s=14, color=c, alpha=0.7, marker=mk); P1.append((e1, y1))
            sc2.scatter(e2, y2, s=14, color=c, alpha=0.7, marker=mk); P2.append((e2, y2))
for a_ in ax:
    a_.set_xlim(0, T); a_.axvspan(0.75 * T, T, color="0.5", alpha=0.07)
    if a_ is not ax[0]:
        a_.axhline(0, color="0.5", lw=0.8)
ax[0].set_yscale("log"); ax[0].set_ylabel("step scale a"); ax[0].legend(fontsize=8, ncol=4, loc="lower left")
ax[0].set_title("110M ranks 512: step scale and its first two changes; the gap to master, its slope and curvature")
ax[1].set_ylabel("a' per 1000 steps"); ax[1].set_ylim(-0.25, 0.25)
ax[2].set_ylabel("a'' per 1000 steps²"); ax[2].set_ylim(-0.3, 0.5)
ax[3].set_ylabel("arm minus master + extras"); ax[3].set_ylim(-0.13, 0.07)
ax[4].set_ylabel("gap slope per 1000 steps"); ax[4].set_ylim(-0.03, 0.06)
ax[5].set_ylabel("gap curvature per 1000 steps²"); ax[5].set_ylim(-0.04, 0.06); ax[5].set_xlabel("step")
for sc, P, xl, yl in ((sc1, P1, "a' per 1000 steps (window mean)", "gap slope per 1000 steps"),
                      (sc2, P2, "a'' per 1000 steps² (window mean)", "gap curvature per 1000 steps²")):
    X, Y = map(np.array, zip(*P)); r = np.corrcoef(X, Y)[0, 1]; b = np.polyfit(X, Y, 1)
    xx = np.linspace(X.min(), X.max(), 50); sc.plot(xx, np.polyval(b, xx), "-", color="0.4", lw=1)
    sc.axhline(0, color="0.5", lw=0.8); sc.axvline(0, color="0.5", lw=0.8)
    sc.set_xlabel(xl); sc.set_ylabel(yl); sc.set_title(f"{yl} vs {xl.split(' ')[0]}: r = {r:.2f}, slope {b[0]:.3f}")
    print(f"{yl}: r={r:.3f} slope={b[0]:.4f} intercept={b[1]:.4f} n={len(X)}")
fig.tight_layout(); fig.savefig("docs/figures/pp_rate2_oct6.png", dpi=100); print("wrote docs/figures/pp_rate2_oct6.png")
