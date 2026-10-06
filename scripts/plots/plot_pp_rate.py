"""110M piecewise-p arms (6 Oct): the change in step scale per step next to each arm minus master + extras and its
slope, on one step axis; right, the slope of (arm minus master) against the change in step scale, one point per
evaluation interval from step 1500 (training-time windows, ranks 512).
    python scripts/plots/plot_pp_rate.py CURVE_DIR      (same files as plot_pp.py) -> docs/figures/pp_rate_oct6.png"""
import json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

D = sys.argv[1]
T, W, MIN = 9155, 305, 0.1
ARMS = [  # name, p points ("" = fixed p), colour, style, width
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
    return MIN + 0.5 * (1 - MIN) * (1 + math.cos(math.pi * (t - W) / (T - W)))


def p_at(pts, t):
    pts = sorted((float(a), float(b)) for a, b in (x.split(":") for x in pts.split(",")))
    for (s0, p0), (s1, p1) in zip(pts, pts[1:]):
        if t <= s1:
            return p0 + (p1 - p0) * (t - s0) / (s1 - s0)
    return pts[-1][1]


def scale(pts, t):
    return rho(t) ** p_at(pts, t)


m = val("checkpoints/final/mx_extras_s0/metrics.jsonl")
fig = plt.figure(figsize=(17, 12))
gs = GridSpec(4, 2, width_ratios=[1.6, 1], figure=fig)
ax = [fig.add_subplot(gs[i, 0]) for i in range(4)]
sc = fig.add_subplot(gs[:, 1])
ts = np.arange(W, T + 1, 10)
X, Y = [], []
for n, pts, c, st, lw in ARMS:
    a = np.array([scale(pts, t) for t in ts])
    da = np.gradient(a, ts) * 1000                      # change in step scale per 1000 steps
    ax[0].plot(ts, a, st, color=c, lw=lw, label=n.replace("pp_", ""))
    ax[1].plot(ts, da, st, color=c, lw=lw)
    f = os.path.join(D, f"sw512_{n}.jsonl")
    if not os.path.exists(f):
        continue
    v = val(f); s = np.array(sorted(x for x in v if x in m and x >= 500))
    gap = np.array([v[x] - m[x] for x in s])
    ax[2].plot(s, gap, st, color=c, lw=lw)
    # slope of the gap per 1000 steps, from a 5-point (1000-step) centred window
    k = 2; ss, sl = [], []
    for i in range(k, len(s) - k):
        A = np.polyfit(s[i - k:i + k + 1], gap[i - k:i + k + 1], 1)
        ss.append(s[i]); sl.append(A[0] * 1000)
    ax[3].plot(ss, sl, st, color=c, lw=lw)
    for x, y in zip(ss, sl):
        if x >= 1500:
            dx = np.gradient([scale(pts, t) for t in (x - 10, x, x + 10)], 10)[1] * 1000
            sc.scatter(dx, y, s=14, color=c, alpha=0.7, marker="o" if st == "-" else "s")
            X.append(dx); Y.append(y)
for a_ in ax:
    a_.set_xlim(0, T); a_.axvspan(0.75 * T, T, color="0.5", alpha=0.07)
ax[0].set_yscale("log"); ax[0].set_ylabel("step scale a"); ax[0].legend(fontsize=8, ncol=4, loc="lower left")
ax[0].set_title("110M ranks 512: step scale, its change, the gap to master and the gap's slope (shaded: last 25%)")
ax[1].axhline(0, color="0.5", lw=0.8); ax[1].set_ylabel("change in a per 1000 steps"); ax[1].set_ylim(-0.25, 0.25)
ax[2].axhline(0, color="0.5", lw=0.8); ax[2].set_ylabel("arm minus master + extras"); ax[2].set_ylim(-0.13, 0.07)
ax[3].axhline(0, color="0.5", lw=0.8); ax[3].set_ylabel("slope of the gap per 1000 steps"); ax[3].set_ylim(-0.03, 0.05)
ax[3].set_xlabel("step")
r = np.corrcoef(X, Y)[0, 1]
b = np.polyfit(X, Y, 1); xx = np.linspace(min(X), max(X), 50)
sc.plot(xx, np.polyval(b, xx), "-", color="0.4", lw=1)
sc.axhline(0, color="0.5", lw=0.8); sc.axvline(0, color="0.5", lw=0.8)
sc.set_xlabel("change in step scale per 1000 steps (at that step)")
sc.set_ylabel("slope of (arm minus master) per 1000 steps (positive: losing ground)")
sc.set_title(f"Every arm, every evaluation from step 1500: r = {r:.2f}, slope {b[0]:.3f}")
fig.tight_layout(); fig.savefig("docs/figures/pp_rate_oct6.png", dpi=110); print(f"wrote docs/figures/pp_rate_oct6.png r={r:.3f}")
