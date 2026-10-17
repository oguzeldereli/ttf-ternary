"""Learn which of momentum's proposed flips to keep. At a no-look-ahead snapshot (saved momentum, float tail frozen),
plain momentum proposes flips each step (the trainer's rule); every proposal is applied so the trajectory is the
real one. At decision steps the proposals are recorded with features available at flip time (all per weight from
M = the low-rank momentum, this batch's gradient g, the trit, row/column statistics; nothing stored per weight):
  |M|/mean|M| (layer), |g|/mean|g|, sign(M) == sign(g), trit before (-1/0/+1), move goes to 0 or away from 0,
  row and column rms of M relative to the layer's, layer type (7) and depth
Two selectors (small MLPs), trained on decision steps TRAIN and judged on held-out ones EVAL:
  A aligned      label: the flip is downhill on the true gradient at that moment (64 batches), BCE
  B loss change  48 random halves of the proposals are applied one at a time and their held-out loss change measured;
                 the MLP gives each flip a contribution c(x) fitted so that the sum over a half matches its change
Eval at a held-out decision step: keep the top 50% by each score vs all proposals vs a random 50%: held-out loss
change and the share of kept flips that are downhill on the true gradient.
  python -m scripts.analysis.flip_selector RUN STEP
"""
import sys, math, numpy as np, torch, torch.nn as nn
from scripts.analysis.testbench import Bench, G_REF
from ttf.kernel import fused_flip, unpack_rows, pack_rows
from ttf.train import get_batch

RUN, ST = sys.argv[1], int(sys.argv[2])
BETA, NT, NSUB = 0.97, 64, 48
TRAIN, EVAL = (10, 20, 30), (25, 33)
train = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")
prog = min(1.0, (ST - 30) / (9155 - 30)); RATE = 0.02 * 0.5 * (1 + math.cos(math.pi * prog))
B = Bench(f"checkpoints/{RUN}/ckpt_{ST}.pt")
Ls, names = B.Ls, B.names
torch.manual_seed(0)


def batches(seed):
    g = torch.Generator().manual_seed(seed)
    while True:
        yield [get_batch(train, 16, 2048, "cuda", g)]


def grad(b):
    g = B.grad(b); return [g[n] for n in names]


def held():
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return float(np.mean([B.m(x, y)[1].item() for x, y in B.VB]))


def trits(): return [unpack_rows(l.wpacked, l.K).to(torch.int8) for l in Ls]


def set_trits(T):
    for l, t in zip(Ls, T): l.wpacked.copy_(pack_rows(t.to(torch.int8)))


def truth(seed):
    s = batches(seed); acc = None
    for _ in range(NT):
        g = grad(next(s)); acc = g if acc is None else [a + b for a, b in zip(acc, g)]
    return [a / NT for a in acc]


def features(F, g, T0, mv, i):
    """per proposed flip in layer i (mv != 0): feature rows"""
    m = mv != 0
    f, gg, t = F[m], g[m], T0[m].float()
    fm, gm = F.abs().mean().clamp_min(1e-12), g.abs().mean().clamp_min(1e-12)
    rr = F.pow(2).mean(1).sqrt(); cr = F.pow(2).mean(0).sqrt(); lr = F.pow(2).mean().sqrt().clamp_min(1e-12)
    rows, cols = m.nonzero(as_tuple=True)
    typ = torch.zeros(len(f), 7, device=f.device); typ[:, i % 7] = 1
    to_zero = (T0[m] != 0).float()
    X = torch.stack([f.abs() / fm, gg.abs() / gm, ((f > 0) == (gg > 0)).float(), (t == -1).float(), (t == 0).float(), (t == 1).float(), to_zero,
                     rr[rows] / lr, cr[cols] / lr, torch.full_like(f, (i // 7) / 11.0)], 1).float()
    return torch.cat([X, typ], 1)


# run the real trajectory; at decision steps record proposals, labels and subset effects
F = [(U.cuda().float() @ V.cuda().float().T) for U, V in B.b["lowrank"]]
s = batches(5151); L_start = held(); data = {}
for k in range(1, max(TRAIN + EVAL) + 1):
    g = grad(next(s)); F = [BETA * f + x for f, x in zip(F, g)]
    T0 = trits()
    for i, (l, f) in enumerate(zip(Ls, F)):
        fused_flip(l.wpacked, f.contiguous(), RATE, G_REF, 3000 + 131 * k + i, gmean=f.abs().mean().clamp_min(1e-12))
    T1 = trits(); mv = [(a - b) for a, b in zip(T1, T0)]
    if k in TRAIN + EVAL:
        set_trits(T0)
        Tg = truth(9000 + k); L0 = held()
        X = torch.cat([features(F[i], g[i], T0[i], mv[i], i) for i in range(len(Ls))])
        down = torch.cat([((mv[i][mv[i] != 0].float() * Tg[i][mv[i] != 0]) < 0).float() for i in range(len(Ls))])
        n = X.shape[0]; subs, dls = [], []
        gen = torch.Generator(device="cuda").manual_seed(k)
        idx_layer = [(mv[i] != 0) for i in range(len(Ls))]
        for j in range(NSUB):                                    # random halves of the proposals
            keep = torch.rand(n, device="cuda", generator=gen) < 0.5
            off = 0; Tj = []
            for i in range(len(Ls)):
                c = int(idx_layer[i].sum()); kk = keep[off:off + c]; off += c
                t = T0[i].clone(); mm = idx_layer[i].clone(); mm[mm.clone()] = kk
                t[mm] = T1[i][mm]; Tj.append(t)
            set_trits(Tj); dls.append(held() - L0); subs.append(keep)
        full = T1; set_trits(full); L_all = held() - L0
        data[k] = dict(X=X, down=down, subs=torch.stack(subs), dls=torch.tensor(dls, device="cuda"), T0=T0, T1=T1,
                       mask=idx_layer, L_all=L_all, n=n)
        print(f"step {k}: {n} proposals, downhill share {float(down.mean()):.3f}, all applied dL {L_all:+.4f}, "
              f"random halves dL mean {float(np.mean(dls)):+.4f}", flush=True)
        set_trits(T1)

# selectors
Xtr = torch.cat([data[k]["X"] for k in TRAIN]); mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
norm = lambda X: (X - mu) / sd
mlp = lambda: nn.Sequential(nn.Linear(Xtr.shape[1], 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)).cuda()
A = mlp(); opt = torch.optim.Adam(A.parameters(), 1e-3)
ytr = torch.cat([data[k]["down"] for k in TRAIN])
for it in range(3000):
    bi = torch.randint(0, len(Xtr), (8192,), device="cuda")
    loss = nn.functional.binary_cross_entropy_with_logits(A(norm(Xtr[bi])).squeeze(1), ytr[bi])
    opt.zero_grad(); loss.backward(); opt.step()
Bm = mlp(); opt = torch.optim.Adam(Bm.parameters(), 1e-3)
for it in range(1500):
    tot = 0.0
    for k in TRAIN:
        d = data[k]; c = Bm(norm(d["X"])).squeeze(1)             # contribution per flip
        pred = d["subs"].float() @ c                              # sum over each half
        tot = tot + ((pred - d["dls"]) ** 2).mean()
    opt.zero_grad(); tot.backward(); opt.step()
print(f"trained: A (aligned) final BCE {float(loss):.4f}; B (loss change) final fit mse {float(tot) / len(TRAIN):.2e}", flush=True)


def apply_keep(d, keep):
    off = 0; Tj = []
    for i in range(len(Ls)):
        c = int(d["mask"][i].sum()); kk = keep[off:off + c]; off += c
        t = d["T0"][i].clone(); mm = d["mask"][i].clone(); mm[mm.clone()] = kk; t[mm] = d["T1"][i][mm]; Tj.append(t)
    set_trits(Tj)


for k in EVAL + TRAIN:
    d = data[k]; set_trits(d["T0"]); L0 = held(); n = d["n"]; half = n // 2
    with torch.no_grad():
        sa = A(norm(d["X"])).squeeze(1); sb = -Bm(norm(d["X"])).squeeze(1)
    rows = {}
    for name, score in (("A aligned, top 50%", sa), ("B loss change, top 50%", sb)):
        keep = torch.zeros(n, dtype=torch.bool, device="cuda"); keep[score.topk(half).indices] = True
        apply_keep(d, keep); rows[name] = (held() - L0, float(d["down"][keep].mean()))
    gen = torch.Generator(device="cuda").manual_seed(99)
    keep = torch.rand(n, device="cuda", generator=gen) < 0.5; apply_keep(d, keep)
    rows["random 50%"] = (held() - L0, float(d["down"][keep].mean()))
    rows["all proposals"] = (d["L_all"], float(d["down"].mean()))
    tag = "held out" if k in EVAL else "train"
    print(f"\ndecision step {k} ({tag}), {n} proposals", flush=True)
    for name, (dl, dn) in rows.items():
        print(f"  {name:24s} held-out dL {dl:+.4f}   downhill share of kept flips {dn:.3f}", flush=True)
set_trits(data[max(TRAIN + EVAL)]["T1"])
