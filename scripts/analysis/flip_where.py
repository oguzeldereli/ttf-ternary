"""Where do the flips land under each rule, and are they right there? At one checkpoint, one momentum (the saved
rank-256 U V^T), one true gradient T (32 batches) and one fresh batch gradient g, apply each flip rule and compare
the expected flips p_ij (no sampling):
  plain       p = rate * min(|M| / (3 mean|M|), 1), move -sign(M)
  vnorm       M / sqrt(R_i C_j / mean R), R, C = row / column means of g^2 over the 32 batches (the factored second
              moment the trainer keeps as an EMA)
  gate        plain, but only where sign(M) = sign(g) for the fresh batch
  gate+vnorm  both (gate after vnorm, as the trainer)
Reports per rule: expected flips, share moving uphill on T, first-order loss change sum p * move * T, the share of
flips in the busiest 1% of rows, and by row steepness (row mean of g^2, in fifths): share of flips and uphill share.
  python -m scripts.analysis.flip_where RUN STEP
"""
import sys, math, numpy as np, torch
from scripts.analysis.testbench import Bench
from ttf.kernel import unpack_rows
from ttf.train import get_batch

RUN, ST = sys.argv[1], int(sys.argv[2])
NT = 32
train = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")
prog = min(1.0, (ST - 30) / (9155 - 30)); RATE = 0.02 * 0.5 * (1 + math.cos(math.pi * prog))
B = Bench(f"checkpoints/{RUN}/ckpt_{ST}.pt")
Ls, names = B.Ls, B.names
gen = torch.Generator().manual_seed(4242)
S1 = S2 = None
for _ in range(NT):
    gd = B.grad([get_batch(train, 16, 2048, "cuda", gen)]); g_ = [gd[n].float() for n in names]
    S1 = g_ if S1 is None else [a + b for a, b in zip(S1, g_)]
    S2 = [x * x for x in g_] if S2 is None else [a + x * x for a, x in zip(S2, g_)]
T = [a / NT for a in S1]; G2 = [b / NT for b in S2]; del S1, S2
gd = B.grad([get_batch(train, 16, 2048, "cuda", torch.Generator().manual_seed(99))]); gb = [gd[n].float() for n in names]
Ms = [(U.cuda().float() @ V.cuda().float().T) for U, V in B.b["lowrank"]]
W = [unpack_rows(l.wpacked, l.K).to(torch.int8) for l in Ls]


def rule(name, i):
    M = Ms[i]
    if "vnorm" in name:
        R, C = G2[i].mean(1), G2[i].mean(0)
        M = M / (R[:, None] * C[None, :] / R.mean().clamp_min(1e-30)).sqrt().clamp_min(1e-30)
    gm = M.abs().mean().clamp_min(1e-12)
    if "gate" in name:
        M = M * (M.sign() == gb[i].sign())
    mv = -M.sign()
    ok = ((W[i].float() + mv).abs() <= 1) & (M != 0)
    p = RATE * (M.abs() / (3 * gm)).clamp(max=1) * ok
    return p, mv


print(f"{RUN} @{ST}, rate {RATE:.4f}; true gradient {NT} batches; rows split into fifths by steepness (row mean g^2)")
print(f"{'rule':11s} {'flips':>7s} {'uphill':>7s} {'dL (1st order)':>15s} {'top-1% rows':>12s}   "
      "share of flips / uphill share by steepness fifth (flat -> steep)")
for name in ("plain", "vnorm", "gate", "gate+vnorm"):
    tot = up = dl = top = 0.0; qf = np.zeros(5); qu = np.zeros(5)
    for i in range(len(Ls)):
        p, mv = rule(name, i)
        u = p * ((mv * T[i]) > 0)
        tot += float(p.sum()); up += float(u.sum()); dl += float((p * mv * T[i]).sum())
        rows = p.sum(1); k = max(1, rows.numel() // 100)
        top += float(rows.topk(k).values.sum())
        steep = G2[i].mean(1); q = torch.bucketize(steep, steep.quantile(torch.tensor([.2, .4, .6, .8], device=steep.device)))
        for j in range(5):
            m = q == j; qf[j] += float(rows[m].sum()); qu[j] += float(u.sum(1)[m].sum())
    fifths = "  ".join(f"{100 * qf[j] / tot:4.1f}%/{100 * qu[j] / max(qf[j], 1e-9):4.1f}%" for j in range(5))
    print(f"{name:11s} {tot / 1e3:6.0f}k {100 * up / tot:6.1f}% {dl:+15.4e} {100 * top / tot:11.1f}%   {fifths}", flush=True)
