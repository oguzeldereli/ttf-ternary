"""Which weights to flip along a given direction? At a no-look-ahead snapshot (float tail frozen), the direction is the
true gradient (64 batches); each flip moves one trit in -sign(T), trits bound to [-1, 1]. The flips are chosen by
different per-weight scores, taking the top n (fractions of a normal step), and the held-out loss change measured:
  size          |T|                                   (what "largest gradient" picks)
  consistency   |T| / std over the 64 batches          (signal-to-noise per weight)
  adam          |T| / sqrt(mean over batches of g^2)   (what AdamW's normalisation does to master's latent)
  adam factored |T| / sqrt(R_i C_j / mean R)           (row x column approximation of the same, N + K numbers)
  random        uniform over movable weights
  python -m scripts.analysis.flip_choice RUN STEP
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


s = batches(4242); S1 = S2 = None
for _ in range(NT):
    g = grad(next(s))
    S1 = g if S1 is None else [a + b for a, b in zip(S1, g)]
    S2 = [x * x for x in g] if S2 is None else [a + x * x for a, x in zip(S2, g)]
mean = [a / NT for a in S1]; msq = [b / NT for b in S2]
std = [(q - m * m).clamp_min(0).sqrt() for m, q in zip(mean, msq)]
fact = []
for q in msq:
    R, C = q.mean(1), q.mean(0)
    fact.append((R[:, None] * C[None, :] / R.mean().clamp_min(1e-30)).sqrt())
L0 = held()
n_step = RATE * 0.25 * sum(t.numel() for t in T0)
gen = torch.Generator(device="cuda").manual_seed(5)
scores = {"size": [m.abs() for m in mean],
          "consistency": [m.abs() / s_.clamp_min(1e-12) for m, s_ in zip(mean, std)],
          "adam": [m.abs() / q.sqrt().clamp_min(1e-12) for m, q in zip(mean, msq)],
          "adam factored": [m.abs() / f.clamp_min(1e-12) for m, f in zip(mean, fact)],
          "random": [torch.rand(m.shape, device="cuda", generator=gen) for m in mean]}
moves = [(-torch.sign(m)).to(torch.int8) for m in mean]
print(f"{RUN} @{ST}: held-out loss {L0:.4f}; direction = true gradient ({NT} batches); 1x = ~{n_step / 1e6:.2f}M flips", flush=True)
for name, sc in scores.items():
    sc = [torch.where((t0 + mv).abs() <= 1, x, torch.full_like(x, -1.0)) for t0, mv, x in zip(T0, moves, sc)]
    # rank within each layer (scores differ in scale across layers), then take the same share per layer
    row = []
    for mult in (0.25, 0.5, 1, 2, 4):
        frac = n_step * mult / sum(t.numel() for t in T0)
        new = []
        for t0, mv, x in zip(T0, moves, sc):
            k = max(1, int(frac * x.numel()))
            thr = x.flatten().topk(k).values[-1]
            new.append(torch.where(x >= thr, (t0 + mv).clamp(-1, 1), t0))
        set_trits(new); row.append(f"{mult:g}x: {held() - L0:+.4f}")
    set_trits(T0)
    print(f"  {name:14s} " + "  ".join(row), flush=True)
