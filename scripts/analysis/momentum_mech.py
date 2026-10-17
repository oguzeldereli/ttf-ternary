"""Two mechanisms for momentum on a landscape that moves under its own flips, tested separately and together at the
step-4000 bench (float tail frozen, 66 steps of flips at the bench's rate, same batches for every arm; momentum kept
full-size here because test 1 showed rank 256 loses nothing).

  V0 plain      F = beta F + g, flip from F                                              (control)
  V1 target     D estimates the displacement to the minimum (trit units): each step subtract the move just made,
                then blend in -g / h (h: one curvature number per layer, measured once after step 1); flip toward D
  V2 transport  after each step's flips, delta = g_same_batch(after) - g_same_batch(before) (the change our move
                caused; the batch noise cancels); every remembered gradient is moved to the new point:
                F <- F + S delta, S = total weight of the remembered gradients; flip from F   (1 extra backward)
  V3 both       D as in V1 but fed with V2's transported average gradient, and h re-measured every step from
                the same-batch difference, so a change of geometry changes the target
  V4 measured   D <- beta (D - delta / h) + (1 - beta)(-g / h): the target estimate is corrected by the MEASURED
                gradient change of our move (delta, same batch), which includes how each flip shifted the other
                weights' targets, instead of assuming the move is covered exactly (V1's D - move)

At 10 / 33 / 66 steps: cosine and top-1% sign precision of the flip signal against the true gradient at that moment
(64 batches), held-out loss, trits changed.
Bench: env BENCH_CKPT (default: the step-4000 look-ahead bench) and BENCH_DIR (where the 256-batch true gradient
is cached; computed if missing). A checkpoint with saved low-rank momentum starts every arm from it (warm, S at its
steady-state 1/(1-beta)); otherwise momentum starts empty.
  python -m scripts.analysis.momentum_mech [ARMS, e.g. V0,V1,V2,V3] [STEPS]
"""
import os
import sys, json, numpy as np, torch
from scripts.analysis.testbench import Bench, batches, G_REF
from ttf.kernel import fused_flip, unpack_rows

OUT = os.environ.get("BENCH_DIR", "checkpoints/testbench_4000")
ARMS = (sys.argv[1] if len(sys.argv) > 1 else "V0,V1,V2,V3").split(",")
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 66
EVAL_AT = [k for k in (10, 33, 66) if k <= STEPS] or [STEPS]
NTRUTH = int(os.environ.get("EVAL_TRUTH", 64 if STEPS >= 33 else 4))
BETA = 0.97
CKPT = os.environ.get("BENCH_CKPT", "checkpoints/r4090_replay_11M_205M/ckpt_4000.pt")
RATE = float(os.environ.get("BENCH_RATE", 0.01202))
os.makedirs(OUT, exist_ok=True)
B = Bench(CKPT)
meta = {"step": B.b["step"]}
names, Ls = B.names, B.Ls


def cat(xs): return torch.cat([x.flatten().float() for x in xs])


def tern_grad(batch):
    g = B.grad(batch)
    return [g[n] for n in names]


def trits(): return [unpack_rows(l.wpacked, l.K).float() for l in Ls]


def held_out():
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return float(np.mean([B.m(x, y)[1].item() for x, y in B.VB]))


def truth(seed):
    s = batches(seed); acc = None
    for _ in range(NTRUTH):
        g = tern_grad(next(s)); acc = g if acc is None else [a + b for a, b in zip(acc, g)]
    return cat([a / NTRUTH for a in acc])


def score(sig_grad_convention, ref):
    """sig in gradient convention (a flip moves against its sign); ref = true gradient now"""
    s = cat(sig_grad_convention); c = float(s @ ref / (s.norm() * ref.norm()))
    a = s.abs(); thr = a[torch.randint(0, a.numel(), (2_000_000,), device=a.device)].quantile(0.99)
    top = a >= thr
    return c, float(((s > 0) == (ref > 0))[top].float().mean())


def flip(sig, k):
    """flip from a gradient-convention signal (moves against its sign), trainer rule; returns the move per layer"""
    before = trits()
    for i, (l, s) in enumerate(zip(Ls, sig)):
        fused_flip(l.wpacked, s.contiguous(), RATE, G_REF, 5000 + 131 * k + i, gmean=s.abs().mean().clamp_min(1e-12))
    return [a - b for a, b in zip(trits(), before)]


def secant_h(delta, move):
    """per-layer curvature from the same-batch gradient change along the move: (delta . move) / |move|^2"""
    out = []
    for d, m in zip(delta, move):
        n = float((m * m).sum())
        out.append(max(float((d * m).sum()) / n, 1e-12) if n > 0 else None)
    return out


if os.path.exists(f"{OUT}/gbar.pt"):
    Gs = torch.load(f"{OUT}/gbar.pt"); gbar0 = cat([Gs[n].cuda() for n in names])
else:
    ss = batches(4242); acc = None
    NB = int(os.environ.get("BENCH_TRUTH", 256))
    for i in range(NB):
        g = tern_grad(next(ss)); acc = g if acc is None else [a + b for a, b in zip(acc, g)]
        if (i + 1) % 64 == 0: print(f"  true gradient {i + 1}/{NB}", flush=True)
    torch.save({n: (a / NB).cpu() for n, a in zip(names, acc)}, f"{OUT}/gbar.pt")
    gbar0 = cat([a / NB for a in acc]); del acc
WARM = [(U.cuda().float(), V.cuda().float()) for U, V in B.b["lowrank"]] if B.b.get("lowrank") else None
L0 = held_out()
print(f"bench {CKPT} step {meta['step']}, rate {RATE}, momentum {'warm' if WARM else 'empty'}, held-out loss at "
      f"start {L0:.4f}; arms {ARMS}, {STEPS} steps", flush=True)
if WARM:
    c0, p0 = score([U @ V.T for U, V in WARM], gbar0)
    print(f"  saved momentum vs true gradient at the checkpoint: cos {c0:+.3f}  top-1% precision {p0:.3f}", flush=True)
for arm in ARMS:
    B.set_trits([torch.zeros_like(t) for t in B.T0])
    s = batches(888)
    F = [U @ V.T for U, V in WARM] if WARM else None; S = 1.0 / (1 - BETA) if WARM else 0.0
    D = None; h = None; nch = 0; rows = []
    prev_batch = None; prev_g = None; prev_move = None
    for k in range(1, STEPS + 1):
        batch = next(s)
        g = tern_grad(batch)
        if arm in ("V2", "V3", "V4") and prev_batch is not None:
            g_same = tern_grad(prev_batch)                      # the previous batch at the new point
            delta = [a - b for a, b in zip(g_same, prev_g)]     # the change our last move caused
            if arm != "V4":
                gain = float(os.environ.get("MECH_GAIN", S))     # default S (~33): every remembered gradient shifted
                F = [f + gain * d for f, d in zip(F, delta)]    # transport every remembered gradient
            if arm == "V3":
                hn = secant_h(delta, prev_move)
                h = [hn_i if hn_i is not None else h_i for hn_i, h_i in zip(hn, h)]
        F = g if F is None else [BETA * f + x for f, x in zip(F, g)]
        S = BETA * S + 1.0
        if k == 1:
            move = flip(F, k)                                   # every arm takes the same first step
            if arm in ("V1", "V3", "V4"):
                g_same = tern_grad(batch)
                h = secant_h([a - b for a, b in zip(g_same, g)], move)
                h = [x if x is not None else 1e-6 for x in h]
                D = [-f / S / hh for f, hh in zip(F, h)]
        elif arm in ("V0", "V2"):
            move = flip(F, k)
        elif arm == "V4":                                       # target corrected by the measured change
            D = [BETA * (d - dl / hh) + (1 - BETA) * (-x / hh) for d, dl, x, hh in zip(D, delta, g, h)]
            move = flip([-d for d in D], k)
        else:                                                   # V1 / V3: target point
            gbar_now = [f / S for f in F] if arm == "V3" else g
            D = [BETA * (d - m) + (1 - BETA) * (-x / hh) for d, m, x, hh in zip(D, prev_move, gbar_now, h)]
            move = flip([-d for d in D], k)                     # flip toward D (gradient convention = -D)
        nch += int(sum(float(m.abs().sum()) for m in move))
        prev_batch, prev_g, prev_move = batch, g, move
        if k in EVAL_AT:
            sig = F if arm in ("V0", "V2") else [-d for d in D]
            tn = truth(9000 + k)
            c, p = score(sig, tn)
            rot = float(tn @ gbar0 / (tn.norm() * gbar0.norm()))
            rows.append((k, c, p, held_out() - L0, nch))
            print(f"  {arm} step {k:3d}: truth now vs at start {rot:+.3f} | cos(signal, truth now) {c:+.3f}  top-1% precision {p:.3f}  "
                  f"held-out dL {rows[-1][3]:+.4f}  trits changed {nch / 1e6:.2f}M", flush=True)
B.set_trits([torch.zeros_like(t) for t in B.T0])
