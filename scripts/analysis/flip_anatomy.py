"""Which flips go uphill, and what do they cost? At a mid-training checkpoint with its saved momentum, the current
recipe's expected flips (Adam step from row / column g^2 of NB batches, sign gate with a fresh batch, p = rate * min(|S| /
(3 mean|S|), 1), moves blocked at +-1) are grouped by features of the weight:
  - signal strength: p / rate (how sure the rule is)
  - |T|: size of the true gradient at that weight, in quartiles within its layer (T = mean of NT training batches)
  - move: 0 -> +-1 (leaving zero) or +-1 -> 0 (towards zero)
  - layer type: attention q / k / v / o, MLP gate / up / down
For each group: share of expected flips, share going uphill on T (move * T > 0), first-order change sum(p * move * T).
Then the real cost: per |T| quartile and per move type, sample that group's flips (scaled to the same count), apply them,
and measure the held-out loss change per 100k flips (second order included), against random-direction flips of the
same weights as a control.
  python -m scripts.analysis.flip_anatomy RUN STEP
"""
import sys, math, numpy as np, torch
from scripts.analysis.testbench import Bench
from ttf.train import get_batch
from ttf.kernel import unpack_rows

RUN, ST = sys.argv[1], int(sys.argv[2])
NT, NB = 16, 8
tr = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")
prog = min(1.0, (ST - 30) / (9155 - 30)); RATE = 0.02 * 0.5 * (1 + math.cos(math.pi * prog))
B = Bench(f"checkpoints/{RUN}/ckpt_{ST}.pt")
Ls, names = B.Ls, B.names
gen = torch.Generator().manual_seed(4242)
S1 = S2 = None
for _ in range(NT):
    gd = B.grad([get_batch(tr, 16, 2048, "cuda", gen)]); g_ = [gd[n].float() for n in names]
    S1 = g_ if S1 is None else [a + b for a, b in zip(S1, g_)]
    S2 = [x * x for x in g_] if S2 is None else [a + x * x for a, x in zip(S2, g_)]
T = [a / NT for a in S1]; G2 = [b / NT for b in S2]; del S1, S2
gb = B.grad([get_batch(tr, 16, 2048, "cuda", torch.Generator().manual_seed(99))]); gb = [gb[n].float() for n in names]
W = [unpack_rows(l.wpacked, l.K).to(torch.int8) for l in Ls]
P, MV = [], []
for i, (U, V) in enumerate(B.b["lowrank"]):
    M = U.cuda().float() @ V.cuda().float().T
    R, C = G2[i].mean(1), G2[i].mean(0)
    S = M / (R[:, None] * C[None, :] / R.mean().clamp_min(1e-30)).sqrt().clamp_min(1e-30)
    gm = S.abs().mean().clamp_min(1e-12)
    S = S * (S.sign() == gb[i].sign())
    mv = -S.sign()
    ok = ((W[i].float() + mv).abs() <= 1) & (S != 0)
    P.append(RATE * (S.abs() / (3 * gm)).clamp(max=1) * ok); MV.append(mv)
    del M, S
kind = lambda n: n.split(".")[-1]
tot = sum(float(p.sum()) for p in P)
print(f"{RUN} @{ST}: rate {RATE:.4f}, expected flips {tot / 1e3:.0f}k per step; T from {NT} batches")


def table(title, groups):
    print(f"\n{title}\n{'group':22s} {'share of flips':>14s} {'uphill':>8s} {'first-order dL':>15s}")
    for g, masks in groups:
        f = up = dl = 0.0
        for i, msk in masks:
            p = P[i] * msk; f += float(p.sum())
            up += float((p * ((MV[i] * T[i]) > 0)).sum()); dl += float((p * MV[i] * T[i]).sum())
        print(f"{g:22s} {100 * f / tot:13.1f}% {100 * up / max(f, 1e-12):7.1f}% {dl:+15.3e}")


qs = [0, 0.25, 0.5, 0.75, 1.0]
def tq(i, k):
    a = T[i].abs(); lo, hi = torch.quantile(a.flatten()[::97], torch.tensor([qs[k], qs[k + 1]], device=a.device))
    return (a >= lo) & (a <= hi) if k == 3 else (a >= lo) & (a < hi)
def sq(i, k):
    r = P[i] / RATE
    edges = [0, 0.1, 0.25, 0.5, 0.999, 1.01]
    return (r > edges[k]) & (r <= edges[k + 1])
table("by signal strength (p / rate)", [(f"{e}", [(i, sq(i, k)) for i in range(len(Ls))])
      for k, e in enumerate(["<=0.1", "0.1-0.25", "0.25-0.5", "0.5-1", "saturated (=1)"])])
table("by |T| quartile within its layer (true-gradient size)", [(f"|T| Q{k + 1}", [(i, tq(i, k)) for i in range(len(Ls))]) for k in range(4)])
table("by move", [("0 -> +-1 (leave zero)", [(i, W[i] == 0) for i in range(len(Ls))]),
                  ("+-1 -> 0 (to zero)", [(i, W[i] != 0) for i in range(len(Ls))])])
table("by layer type", [(t, [(i, torch.ones_like(W[i], dtype=torch.bool)) for i, n in enumerate(names) if kind(n) == t])
                        for t in ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down")])

# real cost: apply sampled flips of one group (scaled to N flips), measure held-out loss change
L0 = B.held_out()
NF = 100_000
def apply(masks, random_dir=False, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    tot_g = sum(float((P[i] * m).sum()) for i, m in masks)
    sc = NF / max(tot_g, 1e-9)
    D = [torch.zeros_like(w) for w in W]
    for i, m in masks:
        p = (P[i] * m * sc).clamp(max=1)
        fire = torch.rand(p.shape, generator=g, device="cuda") < p
        mv = MV[i].to(torch.int8)
        if random_dir: mv = torch.where(torch.rand(p.shape, generator=g, device="cuda") < 0.5, 1, -1).to(torch.int8)
        nt = (W[i] + mv).clamp(-1, 1)
        D[i] = torch.where(fire, nt - W[i], torch.zeros_like(W[i])).to(torch.int8)
    n = sum(int((d != 0).sum()) for d in D)
    return (B.held_out(D) - L0) / n * 1e5, n
print(f"\nreal cost: held-out loss change per 100k applied flips (held-out base {L0:.4f}); rule's direction vs random direction")
for nm, masks in ([(f"|T| Q{k + 1}", [(i, tq(i, k)) for i in range(len(Ls))]) for k in range(4)]
                  + [("0 -> +-1", [(i, W[i] == 0) for i in range(len(Ls))]), ("+-1 -> 0", [(i, W[i] != 0) for i in range(len(Ls))]),
                     ("all", [(i, torch.ones_like(W[i], dtype=torch.bool)) for i in range(len(Ls))])]):
    d, n = apply(masks); dr, _ = apply(masks, random_dir=True, seed=1)
    print(f"{nm:12s} rule {d:+.4f}   random {dr:+.4f}   ({n / 1e3:.0f}k flips)", flush=True)
