"""How much is a perfect direction worth, at the right amount? At a no-look-ahead snapshot (float tail frozen), one
step's worth of flip candidates from four signals, each applied at several fractions of its candidate set (the
entries with the largest |signal| first; each flip moves one trit in -sign(signal), trits bound to [-1, 1]):
  true gradient (64 batches)          the oracle direction
  saved momentum (beta 0.97)          what training uses
  short momentum (beta 0.8)           built over 20 fresh batches at the snapshot
  one batch's gradient
Candidate set = the entries the trainer's flip rule would propose at 4x the snapshot's rate (so fractions up to 4x
of a normal step's count). Held-out loss change per fraction.
  python -m scripts.analysis.oracle_steps RUN STEP
"""
import sys, math, numpy as np, torch
from scripts.analysis.testbench import Bench
from ttf.kernel import unpack_rows, pack_rows
from ttf.train import get_batch

RUN, ST = sys.argv[1], int(sys.argv[2])
NT = 64
train = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")
prog = min(1.0, (ST - 30) / (9155 - 30)); RATE = 0.02 * 0.5 * (1 + math.cos(math.pi * prog))
B = Bench(f"checkpoints/{RUN}/ckpt_{ST}.pt")
Ls, names = B.Ls, B.names
T0 = [unpack_rows(l.wpacked, l.K).to(torch.int8).clone() for l in Ls]


def batches(seed):
    g = torch.Generator().manual_seed(seed)
    while True:
        yield [get_batch(train, 16, 2048, "cuda", g)]


def grad(b):
    g = B.grad(b); return [g[n] for n in names]


def held():
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return float(np.mean([B.m(x, y)[1].item() for x, y in B.VB]))


def set_trits(T):
    for l, t in zip(Ls, T): l.wpacked.copy_(pack_rows(t.to(torch.int8)))


s = batches(4242); acc = None
for _ in range(NT):
    g = grad(next(s)); acc = g if acc is None else [a + b for a, b in zip(acc, g)]
truth = [a / NT for a in acc]
one = grad(next(batches(11)))
short = None
for k, b in enumerate(batches(22)):
    if k == 20: break
    g = grad(b); short = g if short is None else [0.8 * a + x for a, x in zip(short, g)]
saved = [(U.cuda().float() @ V.cuda().float().T) for U, V in B.b["lowrank"]]
L0 = held()
n_step = RATE * 0.25 * sum(t.numel() for t in T0)            # rough count of one normal step's flips (rule: ~rate/4 of weights)
print(f"{RUN} @{ST}: held-out loss {L0:.4f}; rate {RATE:.4f}", flush=True)
for name, sig in (("true gradient", truth), ("saved momentum (0.97)", saved), ("short momentum (0.8)", short),
                  ("one batch", one)):
    # movable candidates: the move -sign(signal) stays inside [-1, 1]; rank all candidates by |signal| per layer-normalized
    scores, moves = [], []
    for t0, x in zip(T0, sig):
        mv = (-torch.sign(x)).to(torch.int8)
        ok = (t0 + mv).abs() <= 1
        sc = torch.where(ok, x.abs() / x.abs().mean().clamp_min(1e-12), torch.zeros_like(x))
        scores.append(sc); moves.append(mv)
    allsc = torch.cat([s_.flatten() for s_ in scores])
    row = []
    for mult in (0.25, 0.5, 1, 2, 4, 8):
        n = int(n_step * mult)
        thr = allsc.topk(n).values[-1]
        set_trits([torch.where(sc >= thr, (t0 + mv).clamp(-1, 1), t0) for t0, sc, mv in zip(T0, scores, moves)])
        row.append(f"{mult:g}x: {held() - L0:+.4f}")
    set_trits(T0)
    print(f"  {name:22s} " + "  ".join(row), flush=True)
print(f"(1x = ~{n_step / 1e6:.2f}M flips, about one normal step)", flush=True)
