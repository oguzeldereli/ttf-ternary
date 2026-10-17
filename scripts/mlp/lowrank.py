"""Is the accumulated gradient low-rank? MLP master run: at steps 300 and 2000, sum the
gradients over the next 200 steps (sum_g) and record master's trit change over that window
(dT). Per layer, rank-r truncated SVD of sum_g; report (layer average)
  cos(rank-r sum_g, sum_g)      how much of the summed direction rank r keeps
  cos(-rank-r sum_g, dT)        how well it points along master's actual discrete moves
  (reference: cos(-sum_g, dT) and a single batch gradient's rank-r cosine)

  python -m scripts.mlp.lowrank
"""
import math, numpy as np, torch, torch.nn.functional as F
from ttf.master import MasterTernaryLinear
from scripts.mlp.mlp_lab import MLPLM, windows

dev, BATCH, LR, WARM, TOTAL, WIN = "cuda", 4096, 1.5e-3, 100, 6000, 200
RANKS = (1, 4, 16, 64, 256)
train = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")
cos = lambda a, b: F.cosine_similarity(a.flatten().float(), b.flatten().float(), 0).item()


def lr_at(s):
    if s < WARM: return LR * (s + 1) / WARM
    return 1.5e-4 + 0.5 * (LR - 1.5e-4) * (1 + math.cos(math.pi * (s - WARM) / (TOTAL - WARM)))


def lowrank(g, r):
    U, S, Vh = torch.linalg.svd(g.float(), full_matrices=False)
    return (U[:, :r] * S[:r]) @ Vh[:r]


torch.manual_seed(0)
m = MLPLM(lambda i, o: MasterTernaryLinear(i, o)).to(dev).train()
opt = torch.optim.AdamW([{"params": [p for n, p in m.named_parameters() if "norm" not in n], "weight_decay": 0.1},
                         {"params": [p for n, p in m.named_parameters() if "norm" in n], "weight_decay": 0.0}],
                        lr=LR, betas=(0.9, 0.95))
Ls = [l for l in m.modules() if isinstance(l, MasterTernaryLinear)]
gen = torch.Generator().manual_seed(1234)
trits = lambda: [(l.weight.detach().float() / l.weight.detach().abs().mean().clamp_min(1e-5)).round().clamp(-1, 1) for l in Ls]


def step(s):
    for g in opt.param_groups: g["lr"] = lr_at(s)
    x, y = windows(train, BATCH, 8, gen, dev)
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        m(x, y)[1].backward()
    gr = [l.weight.grad.float().clone() for l in Ls]
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return gr


s = 0
for ck in (300, 2000):
    while s < ck:
        step(s); s += 1
    T0 = trits(); gsum = None; g_one = None
    for i in range(WIN):
        gr = step(s); s += 1
        g_one = g_one or gr
        gsum = gr if gsum is None else [a + b for a, b in zip(gsum, gr)]
    dT = [t - t0 for t, t0 in zip(trits(), T0)]
    print(f"\n=== window {ck}-{ck + WIN} ({ck * BATCH / 1e6:.1f}M examples) ===")
    print(f"  reference: cos(-sum_g, master trit change) = {np.mean([cos(-g, d) for g, d in zip(gsum, dT)]):.3f}")
    print(f"  {'rank':>5s} {'keeps of sum_g':>15s} {'vs master moves':>16s} {'1-batch g keeps':>16s}")
    for r in RANKS:
        k = np.mean([cos(lowrank(g, r), g) for g in gsum])
        mv = np.mean([cos(-lowrank(g, r), d) for g, d in zip(gsum, dT)])
        k1 = np.mean([cos(lowrank(g, r), g) for g in g_one])
        print(f"  {r:5d} {k:15.3f} {mv:16.3f} {k1:16.3f}")
