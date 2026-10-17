"""How good is a rank-r momentum? At a checkpoint with its saved momentum M = U V^T (rank r per layer, at most min(N, K)):
the true held-out gradient G (64 held-out sequences, two disjoint halves; energy = <G1, G2> so batch noise cancels),
for all positions and for target pairs by their training-set count (0-999 / 1e3-1e4 / >1e4):
  - in M's subspace: share of G's signal inside span(U) x span(V)
  - best rank r: share the top-r singular directions of G itself would hold (the ceiling for that rank)
  - random r: share a random rank-r subspace would hold
  - cos(M, G): does the momentum point along the true gradient
and, for the flip rule on M (plain: p ~ |M| / mean|M|, move -sign(M); gated: only where a fresh training batch agrees in
sign), the share of expected flips that go uphill on G and the first-order loss change sum(p * move * G) (negative =
the flips lower the held-out loss). The Adam step's row / column scaling is left out here.
With RUN:STEP:trunc=32,64,... the checkpoint's momentum is first truncated to its top-r singular directions per layer
(what a rank-r memory of the same momentum would hold; an upper bound for a run trained at rank r).
"Best rank r" is cross-validated: the top-r directions of the gradient from two quarters of the sequences, scored on
the other two quarters (so the estimate is not inflated by fitting the noise).
  python -m scripts.analysis.rank_quality RUN:STEP [RUN:STEP:trunc=64,128 ...]
"""
import sys, numpy as np, torch
from scripts.analysis.testbench import Bench
from ttf.train import get_batch

val = np.memmap("data/wiki32k_val.bin", dtype=np.uint16, mode="r")
tr = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")
NSEQ = 64
starts = np.random.default_rng(11).integers(0, len(val) - 2049, NSEQ)
X = np.stack([val[s:s + 2049].astype(np.int64) for s in starts])
pid = (X[:, :-1] * 65536 + X[:, 1:]).ravel()
up, inv = np.unique(pid, return_inverse=True); cnt = np.zeros(len(up), np.int64)
for a in range(0, len(tr) - 1, 50_000_000):
    c = np.asarray(tr[a:a + 50_000_001]).astype(np.int64); q = c[:-1] * 65536 + c[1:]
    i = np.searchsorted(up, q); i[i >= len(up)] = 0; hit = up[i] == q
    cnt += np.bincount(i[hit], minlength=len(up))
PC = cnt[inv].reshape(NSEQ, 2048)
BK = {"all positions": (-1, 1 << 62), "pairs 0-999x": (0, 1000), "pairs 1e3-1e4x": (1000, 10000), "pairs >1e4x": (10000, 1 << 62)}
if "--all-only" in sys.argv:          # bigger models: only the all-positions gradient (6 gradient copies instead of 24)
    BK = {"all positions": BK["all positions"]}
    sys.argv.remove("--all-only")
Xt = torch.tensor(X)
print(f"{'run':34s} {'r':>5s} {'bucket':16s} {'in M sub':>9s} {'best r':>7s} {'random':>7s} {'cos(M,G)':>9s}")
def views(UV0, trunc):
    """[(label, rank, UV, Q)]: the saved momentum, or its top-r truncations"""
    out = []
    if not trunc:
        UV = UV0; out.append(("", int(np.median([U.shape[1] for U, _ in UV])), UV))
    else:
        full = [torch.linalg.svd(U @ V.T, full_matrices=False) for U, V in UV0]
        for r in trunc:
            UV = [(Us[:, :r] * S[:r], Vh[:r].T) for Us, S, Vh in full]
            out.append((f" top{r}", r, UV))
    return [(lab, r, UV, [(torch.linalg.qr(U)[0], torch.linalg.qr(V)[0]) for U, V in UV]) for lab, r, UV in out]


for arg in sys.argv[1:]:
    parts = arg.split(":"); run, st = parts[0], parts[1]
    trunc = [int(x) for x in parts[2].split("=")[1].split(",")] if len(parts) > 2 else []
    B = Bench(f"checkpoints/{run}/ckpt_{st}.pt")
    names = B.names
    UV0 = [(U.cuda().float(), V.cuda().float()) for U, V in B.b["lowrank"]]

    def bucket_grad(lo, hi, part, nparts=2):
        acc, n = None, 0
        idx = np.arange(part, NSEQ, nparts)
        for j in range(0, len(idx), 8):
            s = idx[j:j + 8]
            x = Xt[s, :-1].cuda(); y = Xt[s, 1:].clone()
            keep = torch.tensor((PC[s] >= lo) & (PC[s] < hi)); y[~keep] = -1
            k = int(keep.sum())
            if k == 0: continue
            gd = B.grad([(x, y.cuda())])
            g = [gd[nm].float() * k for nm in names]
            acc = g if acc is None else [a + b for a, b in zip(acc, g)]; n += k
        return [a / n for a in acc]

    GR = {}
    for name, (lo, hi) in BK.items():
        GR[name] = ([bucket_grad(lo, hi, h) for h in (0, 1)], [bucket_grad(lo, hi, q, 4) for q in range(4)])
    VIEWS = views(UV0, trunc)
    for lab, rk, UV, Q in VIEWS:
        for name in BK:
            (G1, G2), Gq = GR[name]
            sig = cap = rnd = 0.0; bnum = bden = 0.0; mg = mm = gg = 0.0
            for li, ((Qu, Qv), (U, V), a, b) in enumerate(zip(Q, UV, G1, G2)):
                r = U.shape[1]
                sig += float((a * b).sum())
                cap += float(((Qu.T @ a @ Qv) * (Qu.T @ b @ Qv)).sum())
                Us, _, Vh = torch.linalg.svd(Gq[0][li] + Gq[1][li], full_matrices=False)   # fit on two quarters
                Us, Vs = Us[:, :r], Vh[:r].T
                c, d = Gq[2][li], Gq[3][li]                                               # score on the other two
                bnum += float(((Us.T @ c @ Vs) * (Us.T @ d @ Vs)).sum()); bden += float((c * d).sum())
                rnd += float((a * b).sum()) * min(1.0, (r / a.shape[0]) * (r / a.shape[1]))
                M = U @ V.T; Gm = a + b
                mg += float((M * Gm).sum()); mm += float((M * M).sum()); gg += float((Gm * Gm).sum())
            print(f"{run + '@' + st + lab:34s} {rk:5d} {name:16s} {100 * cap / sig:8.1f}% {100 * bnum / bden:6.1f}% "
                  f"{100 * rnd / sig:6.1f}% {mg / (mm * gg) ** .5:+9.3f}", flush=True)
    Gall = [a + b for a, b in zip(*GR["all positions"][0])]
    # flip precision of the rule on M against the all-positions true gradient
    gb = B.grad([get_batch(tr, 16, 2048, "cuda", torch.Generator().manual_seed(99))]); gb = [gb[nm].float() for nm in names]
    for lab, rk, UV, _ in VIEWS:
      for gated in (False, True):
        tot = upw = dl = 0.0
        for (U, V), G, g in zip(UV, Gall, gb):
            M = U @ V.T
            p = (M.abs() / (3 * M.abs().mean())).clamp(max=1)
            if gated: p = p * (M.sign() == g.sign())
            mv = -M.sign()
            tot += float(p.sum()); upw += float((p * ((mv * G) > 0)).sum()); dl += float((p * mv * G).sum())
        print(f"{run + '@' + st + lab:34s} {rk:5d} flips {'gated' if gated else 'plain':5s}: uphill {100 * upw / tot:5.1f}%, "
              f"first-order dL per unit rate {dl:+.3e}", flush=True)
    del B, UV0, VIEWS, GR; torch.cuda.empty_cache()
