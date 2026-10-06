"""110M cooldown schedules (--ts_step_sched HOLD:END, 6-7 Oct) against the losing piecewise-p arms: the step scale of
each arm, each arm minus master + the same extras, and each arm minus the base (p 0.5), at each evaluation step
(training-time windows). Cooldown arms in colour, piecewise-p arms in grey, p 1 dotted.
    python scripts/plots/plot_ss.py RUN_DIR      (RUN_DIR/RUN/metrics.jsonl) -> docs/figures/ss_oct7.png"""
import json, math, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1]
T, W, MIN = 9155, 305, 0.1
SS = [("ss4000_032", 4000, 0.32, "C0"), ("ss5000_02", 5000, 0.2, "C3"), ("ss4000_05", 4000, 0.5, "C1")]
PP = {"A": "0:1.3,3000:1.5,9155:0.5", "B": "0:1.3,3000:1.5,9155:0.25", "C": "0:1.3,3000:1.5,9155:0",
      "D": "0:1.3,3000:2.0,9155:0.25", "E": "0:1.0,3000:1.2,9155:0.25", "F": "0:1.3,3000:1.5,6000:1.5,9155:0.25"}


def val(n):
    f = os.path.join(D, n, "metrics.jsonl")
    if not os.path.exists(f):
        return {}
    return {j["step"]: j["val_loss"] for j in map(json.loads, open(f)) if "val_loss" in j}


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


def ss_scale(hold, end, t):
    if t < W:
        return (t + 1) / W
    if t < hold:
        return 1.0
    return end + 0.5 * (1 - end) * (1 + math.cos(math.pi * min(1.0, (t - hold) / (T - hold))))


m = val("../final/mx_extras_s0")
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
ts = range(0, T + 1, 25)
a = ax[0, 0]
for k, pts in PP.items():
    a.plot(ts, [rho(t) ** p_at(pts, t) for t in ts], "-", color="0.7", lw=1, label="piecewise p A-F" if k == "A" else None)
a.plot(ts, [rho(t) for t in ts], "k:", lw=1.5, label="p 1 (master's lr ratio)")
a.plot(ts, [rho(t) ** 0.5 for t in ts], "k-", lw=2.4, label="base: p 0.5")
for n, h, e, c in SS:
    a.plot(ts, [ss_scale(h, e, t) for t in ts], "-", color=c, lw=2, label=f"{n}: 1 until {h}, cosine to {e}")
a.set_yscale("log"); a.set_ylim(0.08, 1.2); a.set_xlabel("step"); a.set_ylabel("step scale a (log)")
a.set_title("Step scale: cooldown schedules (colour) vs the losing piecewise arms (grey)"); a.legend(fontsize=8)


def panel(a, rk, ref, title, ylim):
    base = val(f"sw{rk}_base")
    for k in PP:
        v = val(f"sw{rk}_pp_{k}"); s = sorted(x for x in v if x in ref and x >= 500)
        if s:
            a.plot(s, [v[x] - ref[x] for x in s], "-", color="0.75", lw=1, label="piecewise p A-F" if k == "A" else None)
    v = val(f"sw{rk}_p1"); s = sorted(x for x in v if x in ref and x >= 500)
    a.plot(s, [v[x] - ref[x] for x in s], "k:", lw=1.6, label=f"p 1 ({v[s[-1]] - ref[s[-1]]:+.3f} final)")
    if ref is not base:
        s = sorted(x for x in base if x in ref and x >= 500)
        a.plot(s, [base[x] - ref[x] for x in s], "k-", lw=2.4, label=f"base p 0.5 ({base[s[-1]] - ref[s[-1]]:+.3f} final)")
    for n, h, e, c in SS:
        v = val(f"sw{rk}_{n}"); s = sorted(x for x in v if x in ref and x >= 500)
        if s:
            fin = "final" if max(v) == T else f"@ {s[-1]}"
            a.plot(s, [v[x] - ref[x] for x in s], "-", color=c, lw=2, label=f"{n} ({v[s[-1]] - ref[s[-1]]:+.3f} {fin})")
    a.axhline(0, color="0.5", lw=0.8); a.axvspan(0.75 * T, T, color="0.5", alpha=0.07)
    a.set_ylim(*ylim); a.set_xlabel("step"); a.set_ylabel("val loss difference (training-time windows)")
    a.set_title(title); a.legend(fontsize=8, loc="upper left")


panel(ax[0, 1], "512", m, "110M ranks 512: arm minus master + extras", (-0.15, 0.12))
panel(ax[1, 0], "512", val("sw512_base"), "110M ranks 512: arm minus base (p 0.5); shaded: last 25%", (-0.08, 0.12))
panel(ax[1, 1], "192", val("sw192_base"), "110M ranks 192: arm minus base (p 0.5); shaded: last 25%", (-0.08, 0.12))
fig.tight_layout(); fig.savefig("docs/figures/ss_oct7.png", dpi=120); print("wrote docs/figures/ss_oct7.png")
