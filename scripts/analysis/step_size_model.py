"""Why does changing TTF's step scale a matter, and not only its size? (110M schedule arms, 6 Oct)

Model: the step scale does two things. (1) Progress: the loss falls with the total step taken so far, S(t) = sum of a.
(2) A noise floor: at step scale a the loss sits c * n(t) above the progress curve, where n is a relaxed by an EMA with
time constant kappa steps (the state needs time to adjust). Fitted jointly to every arm of one rank:
    L(t) = L0 + B * (S(t) + s0)^(-alpha) + c * n(t)
Compared against the same model without the noise term (c = 0) and with progress in steps instead of S (a does not
speed learning). Also: the per-step move rate against the current a, to see whether moves follow the current step
scale or the history.
    python scripts/analysis/step_size_model.py RUN_DIR [ranks]   (RUN_DIR/RUN/{metrics.jsonl}) -> docs/figures/step_model_oct6.png
"""
import json, math, os, sys
import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1]; RK = sys.argv[2] if len(sys.argv) > 2 else "512"
T, W, MIN = 9155, 305, 0.1
pre = f"sw{RK}_"
ARMS = {  # run suffix: schedule spec (kind, args)
    "base": ("p", 0.5), "p025": ("p", 0.25), "p075": ("p", 0.75), "p1": ("p", 1.0), "p15": ("p", 1.5), "p2": ("p", 2.0),
    "p1_f02": ("floor", 1.0, 0.2), "p1_f03": ("floor", 1.0, 0.3),
    "ps1to05_s05": ("sched", 1, 0.5, 0.5), "ps1to05": ("sched", 1, 0.5, 0.0), "ps1to025": ("sched", 1, 0.25, 0.0),
    "ps1to025_s06": ("sched", 1, 0.25, 0.6),
    "pp_A": ("pts", "0:1.3,3000:1.5,9155:0.5"), "pp_B": ("pts", "0:1.3,3000:1.5,9155:0.25"),
    "pp_C": ("pts", "0:1.3,3000:1.5,9155:0"), "pp_D": ("pts", "0:1.3,3000:2.0,9155:0.25"),
    "pp_E": ("pts", "0:1.0,3000:1.2,9155:0.25"), "pp_F": ("pts", "0:1.3,3000:1.5,6000:1.5,9155:0.25"),
}


def rho(t):
    if t < W:
        return (t + 1) / W
    return MIN + 0.5 * (1 - MIN) * (1 + math.cos(math.pi * (min(t, T) - W) / (T - W)))


def a_of(spec, t):
    k = spec[0]
    if k == "p":
        return rho(t) ** spec[1]
    if k == "floor":
        return max(rho(t) ** spec[1], spec[2] if t >= W else 0.0)
    if k == "sched":
        p0, p1, st = spec[1:]
        pr = min(max(t / T, 0), 1); p = p0 if pr <= st else p0 + (p1 - p0) * (pr - st) / (1 - st)
        return rho(t) ** p
    pts = sorted((float(x), float(y)) for x, y in (q.split(":") for q in spec[1].split(",")))
    for (s0, p0), (s1, p1) in zip(pts, pts[1:]):
        if t <= s1:
            return rho(t) ** (p0 + (p1 - p0) * (t - s0) / (s1 - s0))
    return rho(t) ** pts[-1][1]


runs = {}
for n, spec in ARMS.items():
    f = os.path.join(D, pre + n, "metrics.jsonl")
    if not os.path.exists(f):
        continue
    val, mv = {}, {}
    for l in open(f):
        j = json.loads(l)
        if "val_loss" in j:
            val[j["step"]] = j["val_loss"]
        elif "ts_frac" in j:
            mv[j["step"]] = j["ts_frac"]
    ts = np.arange(T + 1)
    a = np.array([a_of(spec, t) for t in ts])
    runs[n] = dict(spec=spec, val=val, mv=mv, a=a, S=np.cumsum(a))
print(f"ranks {RK}: {len(runs)} arms: {' '.join(runs)}")
T0 = 1000


def relaxed(a, kappa):
    out = np.empty_like(a); e = a[0]; k = 1.0 / max(kappa, 1.0)
    for i, x in enumerate(a):
        e += k * (x - e); out[i] = e
    return out


def data():
    X = []
    for n, r in runs.items():
        for s, v in sorted(r["val"].items()):
            if s >= T0:
                X.append((n, s, v))
    return X


X = data()


def predict(th, kind, kappa, n_cache):
    L0, lB, al, s0, c = th
    out = []
    for n, s, _ in X:
        r = runs[n]
        prog = (r["S"][s] if kind != "steps" else s) + abs(s0)
        out.append(L0 + math.exp(min(lB, 30.0)) * prog ** (-min(abs(al), 5.0)) + (c * n_cache[n][s] if kind != "nonoise" else 0.0))
    return np.array(out)


def fit(kind, kappa):
    nc = {n: relaxed(r["a"], kappa) for n, r in runs.items()}
    y = np.array([v for *_, v in X])
    f = lambda th: predict(th, kind, kappa, nc) - y
    best = None
    for al0 in (0.3, 0.6, 1.0):
        res = least_squares(f, [2.0, math.log(30.0), al0, 300.0, 0.1], max_nfev=4000)
        if best is None or res.cost < best.cost:
            best = res
    return best, nc


results = {}
for kind in ("full", "nonoise", "steps"):
    for kappa in ((1, 100, 300, 1000, 3000) if kind != "nonoise" else (1,)):
        res, nc = fit(kind, kappa)
        rmse = math.sqrt(2 * res.cost / len(X))
        results[(kind, kappa)] = (rmse, res.x, nc)
        print(f"{kind:8s} kappa {kappa:5d}: rmse {rmse:.4f}  L0 {res.x[0]:.3f} alpha {abs(res.x[2]):.3f} s0 {abs(res.x[3]):.0f} c {res.x[4]:+.4f}")
bk = min((k for k in results if k[0] == "full"), key=lambda k: results[k][0])
rmse, th, nc = results[bk]
print(f"best full model: kappa {bk[1]}, c {th[4]:+.4f} nats per unit of step scale")

# leave-one-arm-out: refit without an arm, predict its final
print("leave-one-out final predictions (full model at the best kappa vs no-noise model):")
fin = {}
for hold in [n for n in runs if max(runs[n]["val"]) == T]:
    keep = X
    for kind in ("full", "nonoise"):
        globals()["X"] = [x for x in keep if x[0] != hold]
        res, nc2 = fit(kind, bk[1])
        globals()["X"] = [(hold, T, runs[hold]["val"][T])]
        p = predict(res.x, kind, bk[1], nc2)[0]
        fin[(hold, kind)] = p
    globals()["X"] = keep
    v = runs[hold]["val"][T]
    print(f"  {hold:14s} actual {v:.4f}  full {fin[(hold, 'full')]:.4f} ({fin[(hold, 'full')] - v:+.4f})  "
          f"no-noise {fin[(hold, 'nonoise')]:.4f} ({fin[(hold, 'nonoise')] - v:+.4f})")
e_full = [fin[(h, "full")] - runs[h]["val"][T] for h in runs if (h, "full") in fin]
e_nn = [fin[(h, "nonoise")] - runs[h]["val"][T] for h in runs if (h, "nonoise") in fin]
print(f"  final-loss rmse, leave-one-out: full {np.sqrt(np.mean(np.square(e_full))):.4f}, "
      f"no-noise {np.sqrt(np.mean(np.square(e_nn))):.4f}")

# figure
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
pred = predict(th, "full", bk[1], nc)
for i, n in enumerate(runs):
    c = "k" if n == "base" else cols[i % 10]
    idx = [j for j, x in enumerate(X) if x[0] == n]
    s = [X[j][1] for j in idx]; y = [X[j][2] for j in idx]; p = pred[idx]
    ax[0, 0].plot(s, np.array(y) - np.array(p), "-", color=c, lw=2 if n == "base" else 1, label=n)
    # decomposition at each step: noise term c * n(t)
    ax[0, 1].plot(np.arange(T + 1), th[4] * nc[n], "-", color=c, lw=2 if n == "base" else 1)
    mv = runs[n]["mv"]; ms = np.array(sorted(k for k in mv if k >= 1500))
    if len(ms):
        ax[1, 0].scatter(runs[n]["a"][ms], np.array([mv[k] for k in ms]) * 100, s=3, color=c, alpha=0.4)
        ax[1, 1].plot(ms, np.array([mv[k] for k in ms]) * 100, "-", color=c, lw=0.8)
ax[0, 0].axhline(0, color="0.5", lw=0.8); ax[0, 0].set_ylim(-0.03, 0.03); ax[0, 0].legend(fontsize=7, ncol=3)
ax[0, 0].set_title(f"ranks {RK}: residual of the joint fit L0 + B (sum a + s0)^-alpha + c a_relaxed (rmse {rmse:.4f})")
ax[0, 0].set_xlabel("step"); ax[0, 0].set_ylabel("val loss minus model")
ax[0, 1].set_title(f"fitted noise term c x a relaxed over {bk[1]} steps (c = {th[4]:+.3f})")
ax[0, 1].set_xlabel("step"); ax[0, 1].set_ylabel("nats above the progress curve")
ax[1, 0].set_xscale("log"); ax[1, 0].set_yscale("log"); ax[1, 0].set_xlabel("current step scale a")
ax[1, 0].set_ylabel("moves per step (% of weights)"); ax[1, 0].set_title("move rate vs current a (every 10 steps, from 1500)")
ax[1, 1].set_yscale("log"); ax[1, 1].set_xlabel("step"); ax[1, 1].set_ylabel("moves per step (%)")
ax[1, 1].set_title("move rate over training")
fig.tight_layout(); out = f"docs/figures/step_model_r{RK}_oct6.png"; fig.savefig(out, dpi=110); print("wrote", out)
