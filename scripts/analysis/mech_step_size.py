"""Is the flip rate too big a step for momentum's direction? At a no-look-ahead snapshot with its saved momentum
(float tail frozen, plain momentum F = beta F + g, the trainer's flip rule):
  1 line search   run 33 steps at the snapshot's rate, keep the total move; apply a random fraction alpha of the
                  changed trits (alpha 0 .. 1) and measure held-out loss. A minimum below alpha = 1 = overshoot.
  2 rate sweep    from the same start, 33 steps at 1, 1/2, 1/4, 1/8 of the rate; then the signal's cosine and
                  top-1% precision against a fresh 64-batch true gradient, held-out loss change, trits changed.
  python -m scripts.analysis.mech_step_size RUN STEP
"""
import sys, math, numpy as np, torch
from scripts.analysis.testbench import Bench, G_REF
from ttf.kernel import fused_flip, unpack_rows, pack_rows
from ttf.train import get_batch

RUN, ST = sys.argv[1], int(sys.argv[2])
BETA, NT, K = 0.97, 64, 33
train = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")
prog = min(1.0, (ST - 30) / (9155 - 30)); RATE = 0.02 * 0.5 * (1 + math.cos(math.pi * prog))
B = Bench(f"checkpoints/{RUN}/ckpt_{ST}.pt")
Ls, names = B.Ls, B.names
T0 = [unpack_rows(l.wpacked, l.K).to(torch.int8).clone() for l in Ls]
warm = [(U.cuda().float(), V.cuda().float()) for U, V in B.b["lowrank"]]


def batches(seed):
    g = torch.Generator().manual_seed(seed)
    while True:
        yield [get_batch(train, 16, 2048, "cuda", g)]


def grad(b):
    g = B.grad(b); return [g[n] for n in names]


def set_trits(T):
    for l, t in zip(Ls, T): l.wpacked.copy_(pack_rows(t.to(torch.int8)))


def held():
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return float(np.mean([B.m(x, y)[1].item() for x, y in B.VB]))


def run(rate):
    set_trits(T0); F = [U @ V.T for U, V in warm]; s = batches(5151)
    for k in range(1, K + 1):
        g = grad(next(s)); F = [BETA * f + x for f, x in zip(F, g)]
        for i, (l, f) in enumerate(zip(Ls, F)):
            fused_flip(l.wpacked, f.contiguous(), rate, G_REF, 3000 + 131 * k + i, gmean=f.abs().mean().clamp_min(1e-12))
    return F, [unpack_rows(l.wpacked, l.K).to(torch.int8) for l in Ls]


L0 = held()
print(f"{RUN} @{ST}: rate {RATE:.4f}, held-out loss {L0:.4f}", flush=True)
print("\n== 1: line search along the 33-step move (random fraction alpha of the changed trits applied)", flush=True)
F, T1 = run(RATE)
moved = sum(int((a != b).sum()) for a, b in zip(T1, T0))
gen = torch.Generator(device="cuda").manual_seed(3)
for alpha in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
    Ta = [torch.where((torch.rand(t0.shape, device="cuda", generator=gen) < alpha) & (t1 != t0), t1, t0) for t0, t1 in zip(T0, T1)]
    set_trits(Ta)
    print(f"  alpha {alpha:4.2f}: held-out loss change {held() - L0:+.4f}", flush=True)
print(f"  ({moved / 1e6:.2f}M trits changed over the 33 steps)", flush=True)
print("\n== 2: flip rate sweep, 33 steps each", flush=True)
for mult in (1.0, 0.5, 0.25, 0.125):
    F, T1 = run(RATE * mult)
    dL = held() - L0
    s2 = batches(777); acc = None
    for _ in range(NT):
        gg = grad(next(s2)); acc = gg if acc is None else [a + b for a, b in zip(acc, gg)]
    T = torch.cat([(a / NT).flatten() for a in acc]); Fv = torch.cat([f.flatten() for f in F])
    c = float(Fv @ T / (Fv.norm() * T.norm()))
    a = Fv.abs(); thr = a[torch.randint(0, a.numel(), (2_000_000,), device=a.device)].quantile(0.99)
    p = float(((Fv > 0) == (T > 0))[a >= thr].float().mean())
    ch = sum(int((x != y).sum()) for x, y in zip(T1, T0))
    print(f"  rate x{mult:<5}: cos(momentum, truth now) {c:+.3f}  top-1% precision {p:.3f}  held-out dL {dL:+.4f}  "
          f"trits changed {ch / 1e6:.2f}M", flush=True)
set_trits(T0)
