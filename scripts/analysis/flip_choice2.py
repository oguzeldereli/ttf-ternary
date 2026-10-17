"""Follow-up to flip_choice (random sign-aligned flips lower loss, every top-k choice raises it): is it concentration
(top-k piles flips into a few rows / columns) or size itself? Arms:
  size per row      top |T| within each row, the same count in every row (no row concentration)
  size per column   the same by column
  proportional      sample with p ~ min(|T| / (3 mean|T|), 1), the trainer's rule, scaled to the count
  low size          the smallest |T| entries (sign-aligned)
  random            uniform, also at 8x and 16x to find its best amount
Original docstring:
Which weights to flip along a given direction? At a no-look-ahead snapshot (float tail frozen), the direction is the
true gradient (64 batches); each flip moves one trit in -sign(T), trits bound to [-1, 1]. The flips are chosen by
different per-weight scores, taking the top n (fractions of a normal step), and the held-out loss change measured:
  size          |T|                                   (what "largest gradient" picks)
  consistency   |T| / std over the 64 batches          (signal-to-noise per weight)
  adam          |T| / sqrt(mean over batches of g^2)   (what AdamW's normalisation does to master's latent)
  adam factored |T| / sqrt(R_i C_j / mean R)           (row x column approximation of the same, N + K numbers)
  random        uniform over movable weights
  python -m scripts.analysis.flip_choice2 RUN STEP
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
N_ALL = sum(t.numel() for t in T0)
moves = [(-torch.sign(m)).to(torch.int8) for m in mean]
print(f"{RUN} @{ST}: held-out loss {L0:.4f}; direction = true gradient ({NT} batches); 1x = ~{n_step / 1e6:.2f}M flips", flush=True)
MULTS = (0.25, 0.5, 1, 2, 4, 8, 16)


def run_arm(name, pick):
    """pick(i, t0, mv, m, ok, k) -> bool mask of the k weights to flip in layer i"""
    row = []
    for mult in MULTS:
        frac = n_step * mult / N_ALL; new = []
        for i, (t0, mv, m) in enumerate(zip(T0, moves, mean)):
            ok = (t0 + mv).abs() <= 1
            k = max(1, int(frac * t0.numel()))
            sel = pick(i, t0, mv, m, ok, k) & ok
            new.append(torch.where(sel, (t0 + mv).clamp(-1, 1), t0))
        set_trits(new); row.append(f"{mult:g}x: {held() - L0:+.4f}")
    set_trits(T0)
    print(f"  {name:16s} " + "  ".join(row), flush=True)


def topk_mask(x, k):
    thr = x.flatten().topk(k).values[-1]; return x >= thr


def per_row(x, k, dim):
    kk = max(1, k // x.shape[1 - dim])
    if dim == 1: thr = x.topk(min(kk, x.shape[1]), dim=1).values[:, -1:]
    else: thr = x.topk(min(kk, x.shape[0]), dim=0).values[-1:, :]
    return x >= thr


def masked(x, ok): return torch.where(ok, x, torch.full_like(x, -1.0))


gen = torch.Generator(device="cuda").manual_seed(5)
run_arm("size", lambda i, t0, mv, m, ok, k: topk_mask(masked(m.abs(), ok), k))
run_arm("size per row", lambda i, t0, mv, m, ok, k: per_row(masked(m.abs(), ok), k, 1))
run_arm("size per column", lambda i, t0, mv, m, ok, k: per_row(masked(m.abs(), ok), k, 0))


def prop(i, t0, mv, m, ok, k):
    p = (m.abs() / (3 * m.abs().mean())).clamp(max=1) * ok
    p = (p * k / p.sum().clamp_min(1e-12)).clamp(max=1)
    return torch.rand(p.shape, device="cuda", generator=gen) < p


run_arm("proportional", prop)
run_arm("low size", lambda i, t0, mv, m, ok, k: topk_mask(masked(-m.abs() + 1e9, ok), k))
run_arm("random", lambda i, t0, mv, m, ok, k: topk_mask(masked(torch.rand(m.shape, device="cuda", generator=gen), ok), k))
# where do top-k flips land: share of flips in the busiest 1% of rows, 1x amount
frac = n_step / N_ALL; shares = []
for t0, mv, m in zip(T0, moves, mean):
    ok = (t0 + mv).abs() <= 1; k = max(1, int(frac * t0.numel()))
    sel = (topk_mask(masked(m.abs(), ok), k) & ok).float().sum(1)
    top = sel.sort(descending=True).values[:max(1, sel.numel() // 100)].sum()
    shares.append(float(top / sel.sum().clamp_min(1)))
print(f"top-|T| flips at 1x: busiest 1% of rows hold {100 * np.mean(shares):.1f}% of the flips (uniform = 1%)", flush=True)
