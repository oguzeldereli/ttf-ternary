"""110M piecewise-p arms (--ts_anneal_pts, 6 Oct): the step scale each arm uses, and each arm minus the base (p 0.5) and
minus master + the same extras at each evaluation step (training-time windows).
    python scripts/plots/plot_pp.py CURVE_DIR      (RUN.jsonl files: sw512_pp_A..F, sw192_pp_*, sw{512,192}_base)
Writes docs/figures/pp_oct6.png."""
import json, math, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1]
T, W, MIN = 9155, 305, 0.1          # steps, warmup, final lr ratio (lr 1.5e-3 -> 1.5e-4)
ARMS = {  # name: (p points, colour)
    "A": ("0:1.3,3000:1.5,9155:0.5", "C0"),
    "B": ("0:1.3,3000:1.5,9155:0.25", "C1"),
    "C": ("0:1.3,3000:1.5,9155:0", "C2"),
    "D": ("0:1.3,3000:2.0,9155:0.25", "C3"),
    "E": ("0:1.0,3000:1.2,9155:0.25", "C4"),
    "F": ("0:1.3,3000:1.5,6000:1.5,9155:0.25", "C5"),
}


def val(p):
    out = {}
    for l in open(p):
        if '"val_loss"' in l:
            j = json.loads(l); out[j["step"]] = j["val_loss"]
    return out


def rho(t):
    if t < W:
        return (t + 1) / W
    r = (t - W) / (T - W)
    return MIN + 0.5 * (1 - MIN) * (1 + math.cos(math.pi * r))


def p_at(pts, t):
    pts = sorted((float(a), float(b)) for a, b in (x.split(":") for x in pts.split(",")))
    for (s0, p0), (s1, p1) in zip(pts, pts[1:]):
        if t <= s1:
            return p0 + (p1 - p0) * (t - s0) / (s1 - s0)
    return pts[-1][1]


def get(n):
    f = os.path.join(D, n + ".jsonl")
    return val(f) if os.path.exists(f) else {}


m = val("checkpoints/final/mx_extras_s0/metrics.jsonl")
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
ts = range(0, T + 1, 25)
a = ax[0, 0]
a.plot(ts, [rho(t) ** 0.5 for t in ts], "k-", lw=2.4, label="base: p 0.5")
a.plot(ts, [rho(t) for t in ts], "k:", lw=1.2, label="p 1 (= master's lr ratio)")
for k, (pts, c) in ARMS.items():
    a.plot(ts, [rho(t) ** p_at(pts, t) for t in ts], "-", color=c, lw=1.4, label=f"{k}: p {pts.replace(',', ', ')}")
a.set_yscale("log"); a.set_ylim(0.004, 1.2); a.set_xlabel("step"); a.set_ylabel("step scale  a = (lr ratio)^p  (log)")
a.set_title("Step scale of each arm (smaller = smaller moves)"); a.legend(fontsize=8, loc="lower left")


def panel(a, base, ref, title, ylim):
    b = get(base + "_base")
    for k, (pts, c) in ARMS.items():
        v = get(f"{base}_pp_{k}")
        s = sorted(x for x in v if x in ref and x >= 500)
        if not s:
            continue
        fin = " final" if max(v) == T else f" @ {s[-1]}"
        a.plot(s, [v[x] - ref[x] for x in s], "-", color=c, lw=1.4, label=f"{k} ({v[s[-1]] - ref[s[-1]]:+.3f}{fin})")
    if ref is not b:
        s = sorted(x for x in b if x in ref and x >= 500)
        a.plot(s, [b[x] - ref[x] for x in s], "k-", lw=2.4, label=f"base p 0.5 ({b[s[-1]] - ref[s[-1]]:+.3f} final)")
    a.axhline(0, color="0.5", lw=0.8); a.axvspan(0.75 * T, T, color="0.5", alpha=0.08)
    a.set_ylim(*ylim); a.set_xlabel("step"); a.set_ylabel("val loss difference (training-time windows)")
    a.set_title(title); a.legend(fontsize=8, loc="upper left")


panel(ax[0, 1], "sw512", m, "110M ranks 512: arm minus master + extras", (-0.2, 0.12))
panel(ax[1, 0], "sw512", get("sw512_base"), "110M ranks 512: arm minus base (p 0.5); shaded: last 25%", (-0.08, 0.12))
panel(ax[1, 1], "sw192", get("sw192_base"), "110M ranks 192: arm minus base (p 0.5); shaded: last 25%", (-0.08, 0.12))
fig.tight_layout(); fig.savefig("docs/figures/pp_oct6.png", dpi=120); print("wrote docs/figures/pp_oct6.png")
