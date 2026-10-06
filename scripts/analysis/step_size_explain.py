"""Why changing the step scale matters (110M ranks 512, 18 schedule arms, 6 Oct). Uses the joint fit of
step_size_model.py (L = L0 + B (sum a + s0)^-alpha + c a_relaxed, relaxation 300 steps). Left: the move rate is one
function of the current step scale for every arm, with the threshold-crossing law log m = A - k / a^2 fitted at small a.
Middle / right: each arm minus the base, actual, and the model's split into the progress part (total step taken) and
the noise part (current step scale).
    python scripts/analysis/step_size_explain.py RUN_DIR      -> docs/figures/step_explain_oct6.png"""
import json, math, sys
import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.argv = [sys.argv[0], sys.argv[1], "512"]
exec(open("scripts/analysis/step_size_model.py").read().split("T0 = 1000")[0])   # ARMS, runs, a_of, rho
KAPPA = 300


def relaxed(a, k):
    out = np.empty_like(a); e = a[0]
    for i, x in enumerate(a):
        e += (x - e) / k; out[i] = e
    return out


NC = {n: relaxed(r["a"], KAPPA) for n, r in runs.items()}
X = [(n, s, v) for n, r in runs.items() for s, v in sorted(r["val"].items()) if s >= 1000]
y = np.array([v for *_, v in X])
parts = lambda th, n, s: (th[0] + math.exp(min(th[1], 30)) * (runs[n]["S"][s] + abs(th[3])) ** (-min(abs(th[2]), 5)),
                          th[4] * NC[n][s])
best = None
for al0 in (0.3, 0.6, 1.0):
    r_ = least_squares(lambda th: np.array([sum(parts(th, n, s)) for n, s, _ in X]) - y,
                       [2.0, math.log(30.0), al0, 300.0, 0.1], max_nfev=4000)
    if best is None or r_.cost < best.cost:
        best = r_
th = best.x; rmse = math.sqrt(2 * best.cost / len(X))
print(f"fit: rmse {rmse:.4f}, c {th[4]:.3f}, alpha {abs(th[2]):.3f}, s0 {abs(th[3]):.0f}, L0 {th[0]:.3f}")

fig, ax = plt.subplots(1, 3, figsize=(19, 6))
A, M = [], []
for n, r in runs.items():
    for s, m in r["mv"].items():
        if s >= 1500 and m > 0:
            A.append(r["a"][s]); M.append(m)
A, M = np.array(A), np.array(M)
ax[0].scatter(A, M * 100, s=2, color="0.35", alpha=0.25, label="18 arms, every 10 steps from 1500")
k = (A >= 0.12) & (A < 0.4); b = np.polyfit(1 / A[k] ** 2, np.log(M[k]), 1)
aa = np.linspace(0.1, 0.5, 100)
ax[0].plot(aa, np.exp(b[1] + b[0] / aa ** 2) * 100, "C3-", lw=2,
           label=f"threshold crossings: log m = {b[1]:.1f} - {-b[0]:.3f} / a²\n(noise per step sigma ≈ {math.sqrt(0.256 / -b[0]):.2f})")
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_ylim(5e-7, 0.5)
ax[0].set_xlabel("current step scale a"); ax[0].set_ylabel("weights moved per step (%)")
ax[0].set_title("The move rate is set by the current step scale alone"); ax[0].legend(fontsize=9, loc="lower right")

show = [("pp_A", "C0"), ("pp_D", "C3"), ("p1", "k"), ("pp_F", "C5")]
for j, (n, c) in enumerate(show):
    a_ = ax[1] if j < 2 else ax[2]
    s = np.array(sorted(x for x in runs[n]["val"] if x >= 1000 and x in runs["base"]["val"]))
    act = np.array([runs[n]["val"][x] - runs["base"]["val"][x] for x in s])
    pg = np.array([parts(th, n, x)[0] - parts(th, "base", x)[0] for x in s])
    nz = np.array([parts(th, n, x)[1] - parts(th, "base", x)[1] for x in s])
    a_.plot(s, act, "-", color=c, lw=2.4, label=f"{n}: actual minus base")
    a_.plot(s, pg, "--", color=c, lw=1.3, label=f"{n}: model, progress part (less total step: +)")
    a_.plot(s, nz, ":", color=c, lw=1.6, label=f"{n}: model, noise part (smaller current a: -)")
for a_ in ax[1:]:
    a_.axhline(0, color="0.5", lw=0.8); a_.set_xlabel("step"); a_.set_ylabel("val loss minus base (training-time windows)")
    a_.set_ylim(-0.12, 0.14); a_.legend(fontsize=8, loc="upper left"); a_.axvspan(0.75 * T, T, color="0.5", alpha=0.07)
ax[1].set_title(f"Smaller steps mid-training: quick noise gain, slow progress loss (fit rmse {rmse:.3f})")
ax[2].set_title("p 1 (too small late) and F (rises late)")
fig.tight_layout(); fig.savefig("docs/figures/step_explain_oct6.png", dpi=110); print("wrote docs/figures/step_explain_oct6.png")
