"""Why is the low-rank momentum at the noise level? Two tests at the step-4000 bench (true gradient = mean of 256
batches, checkpoints/testbench_4000/gbar.pt).

1 (compression)  weights frozen. Over 66 fresh batches accumulate, with beta = 0.97:
     full   F = beta F + g, no compression (diagnostic; this is what master's latent weights integrate)
     ours   the trainer's rank-256 momentum (one subspace-iteration step per batch, same code path)
     svd    the best rank-256 approximation of F (truncated SVD per layer)
   and score each against the true gradient: cosine, and sign precision on the 1% of weights with the largest
   |signal| (the ones a flip rule picks).
2 (moving target)  same start, flips applied every step from the rank-256 momentum as a no-look-ahead run does (rate
   of the bench); the true gradient is re-estimated from 64 batches after 33 and 66 steps. How far it rotates says
   how stale a 33-step momentum is.
  python -m scripts.analysis.momentum_test
"""
import json, torch
from scripts.analysis.testbench import Bench, batches, G_REF
from ttf.kernel import fused_flip

OUT = "checkpoints/testbench_4000"
BETA, R = 0.97, 256
meta = json.load(open(f"{OUT}/meta.json"))
B = Bench(meta["ckpt"])
names = B.names
Gs = torch.load(f"{OUT}/gbar.pt")
gbar0 = [Gs[n].cuda() for n in names]


def cat(xs): return torch.cat([x.flatten().float() for x in xs])


gb0 = cat(gbar0)


def score(sig, ref):
    s = cat(sig); c = float(s @ ref / (s.norm() * ref.norm()))
    a = s.abs(); thr = a[torch.randint(0, a.numel(), (2_000_000,), device=a.device)].quantile(0.99)
    top = a >= thr
    return c, float(((s > 0) == (ref > 0))[top].float().mean())


def lr_update(st, i, g):
    if i not in st:
        U0 = torch.linalg.qr(torch.randn(g.shape[0], R, device=g.device))[0]
        V = torch.linalg.qr(g.T @ U0)[0]; st[i] = (g @ V, V)
    else:
        U, V = st[i]
        Vn = torch.linalg.qr(BETA * V @ (U.T @ U) + g.T @ U)[0]
        st[i] = (BETA * U @ (V.T @ Vn) + g @ Vn, Vn)


def svd_r(F):
    out = []
    for f in F:
        U, S, Vh = torch.linalg.svd(f, full_matrices=False)
        out.append((U[:, :R] * S[:R]) @ Vh[:R])
    return out


def tern_grad(batch):
    g = B.grad(batch)
    return [g[n] for n in names]


def truth(nb, seed):
    s = batches(seed); acc = None
    for _ in range(nb):
        g = tern_grad(next(s)); acc = g if acc is None else [a + b for a, b in zip(acc, g)]
    return [a / nb for a in acc]


print(f"bench {meta['ckpt']} step {meta['step']}; true gradient = 256 batches", flush=True)
print("\n== 1: weights frozen (no flips)")
print(f"{'batches':>7s} | {'full: cos':>9s} {'prec@1%':>7s} | {'ours r256: cos':>14s} {'prec@1%':>7s} | {'best r256 (svd): cos':>20s} {'prec@1%':>7s}")
F = None; st = {}; s = batches(777)
for k in range(1, 67):
    g = tern_grad(next(s))
    F = g if F is None else [BETA * f + x for f, x in zip(F, g)]
    for i, x in enumerate(g): lr_update(st, i, x)
    if k in (1, 2, 5, 10, 20, 33, 50, 66):
        M = [st[i][0] @ st[i][1].T for i in range(len(g))]
        cf, pf = score(F, gb0); cm, pm = score(M, gb0); cs, ps = score(svd_r(F), gb0)
        print(f"{k:7d} | {cf:9.3f} {pf:7.3f} | {cm:14.3f} {pm:7.3f} | {cs:20.3f} {ps:7.3f}", flush=True)
        del M

g64 = truth(64, 9999); c64 = cat(g64)
ceil64 = float(c64 @ gb0 / (c64.norm() * gb0.norm()))
print(f"\nreference: a 64-batch estimate at the frozen start vs the 256-batch truth: cos {ceil64:.3f} "
      f"(the ceiling for the 64-batch truths below)", flush=True)
print("\n== 2: flips on, as a no-look-ahead run (rate %.4f, signal = rank-256 momentum)" % meta["rate"])
B.set_trits([torch.zeros_like(t) for t in B.T0])
F = None; st = {}; s = batches(888); nflip = 0
for k in range(1, 67):
    g = tern_grad(next(s))
    F = g if F is None else [BETA * f + x for f, x in zip(F, g)]
    for i, x in enumerate(g): lr_update(st, i, x)
    for i, l in enumerate(B.Ls):
        M = st[i][0] @ st[i][1].T
        before = l.wpacked.clone()
        fused_flip(l.wpacked, M, meta["rate"], G_REF, 5000 + 131 * k + i, gmean=M.abs().mean().clamp_min(1e-12))
        nflip += int((before != l.wpacked).sum())
    if k in (33, 66):
        gt = truth(64, 9000 + k); gtc = cat(gt)
        M = [st[i][0] @ st[i][1].T for i in range(len(g))]
        cf, pf = score(F, gtc); cm, pm = score(M, gtc)
        print(f"after {k} steps: cos(true gradient now, true gradient at start) = {float(gtc @ gb0 / (gtc.norm() * gb0.norm())):.3f}"
              f"  | full sum vs truth now: cos {cf:.3f} prec {pf:.3f} | ours r256 vs truth now: cos {cm:.3f} prec {pm:.3f}", flush=True)
print(f"trits changed over the 66 steps: {nflip}")
B.set_trits([torch.zeros_like(t) for t in B.T0])
