"""Snapshot of the low-rank momentum against the gradient at a checkpoint of a --lowrank run.

Loads the ternary net, the float tail and M = U V^T from the checkpoint. Takes K training
batches (each the full 16 x 2048 batch, accumulated in micro-batches so it fits beside a
running job) and reports per layer, averaged over layers:
  |g|        mean |dL/dT| of one batch          |gbar|  of the K-batch mean
  |M|        mean |M|; |M| / |gbar| (33 if every step's gradient had pointed the same way)
  cos(g, gbar)   how much of a single batch is signal (noise reference)
  cos(g, M), cos(gbar, M)   agreement of the momentum with one batch / the mean gradient
  sign agree     share of weights where sign(M) = sign(gbar), all and on the top 1% |M|
                 (the ones the flip rule proposes)
  in-subspace    share of |gbar| inside M's rank-r row x column subspace

  python -m scripts.analysis.m_snapshot [ckpt] [K]
"""
import sys, numpy as np, torch
import torch.nn.functional as F
from ttf.flip import build_kernel_transformer, KernelTernaryLinear
from ttf.train import get_batch

dev = "cuda"
CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/lm_lowrank256_xb2_100M/ckpt.pt"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 8
BS, MICRO, SEQ = 16, 4, 2048

b = torch.load(CKPT, map_location="cpu", weights_only=False)
m = build_kernel_transformer(b["cfg"], grad_checkpoint=True, beta=b.get("beta", True),
                             int8=b.get("int8", True), dw_mode=b.get("dw_mode", "dense"), g_ref=3.0)
for p in m.float_tail_parameters():
    p.data = p.data.float()
m.load_state_dict(b["model"], strict=False)
m = m.to(dev).train()
Ls = [l for l in m.modules() if isinstance(l, KernelTernaryLinear)]
UV = b["lowrank"]
print(f"checkpoint step {b['step']} ({(b['step'] + 1) * 32768 / 1e6:.0f}M tokens), {len(Ls)} layers, K={K}")
train = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")
gen = torch.Generator().manual_seed(31337)


def batch_grad():
    acc = [torch.zeros(l.N, l.K, device=dev) for l in Ls]
    for _ in range(BS // MICRO):
        x, y = get_batch(train, MICRO, SEQ, dev, gen)
        for l in Ls: l.capture = True
        for p in m.parameters(): p.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = m(x, y)[1]
        loss.backward()
        for a, l in zip(acc, Ls):
            a += l.gw.float(); l.gw = None
    for p in m.parameters(): p.grad = None
    return [a / (BS // MICRO) for a in acc]


cos = lambda a, b: F.cosine_similarity(a.flatten(), b.flatten(), 0).item()
g1 = batch_grad()
gbar = [g.clone() for g in g1]
for _ in range(K - 1):
    for a, g in zip(gbar, batch_grad()): a += g
gbar = [a / K for a in gbar]

rows = []
for l, g, gb, (U, V) in zip(Ls, g1, gbar, UV):
    U, V = U.to(dev).float(), V.to(dev).float()
    M = U @ V.T
    Qu = torch.linalg.qr(U)[0]
    inside = (Qu @ (Qu.T @ gb @ V) @ V.T).norm() / gb.norm()
    top = M.abs().flatten() >= M.abs().flatten().quantile(0.99) if M.numel() < 2 ** 24 else \
        M.abs().flatten() >= M.abs().flatten()[torch.randperm(M.numel(), device=dev)[:2 ** 20]].quantile(0.99)
    agree = (M.sign() == gb.sign()).float()
    rows.append([g.abs().mean().item(), gb.abs().mean().item(), M.abs().mean().item(),
                 (M.abs().mean() / gb.abs().mean()).item(),
                 cos(g, gb), cos(g, M), cos(gb, M),
                 agree.mean().item(), agree.flatten()[top].mean().item(), inside.item()])
R = np.array(rows)
names = ["|g| 1 batch", f"|gbar| {K} batches", "|M|", "|M| / |gbar|", "cos(g, gbar)",
         "cos(g, M)", "cos(gbar, M)", "sign agree all", "sign agree top1% |M|", "gbar in M subspace"]
print(f"{'':24s} {'mean':>10s} {'min':>10s} {'max':>10s}")
for i, n in enumerate(names):
    print(f"{n:24s} {R[:, i].mean():10.4g} {R[:, i].min():10.4g} {R[:, i].max():10.4g}")
print("\nby layer type (mean over blocks): cos(g,gbar) cos(gbar,M) agree-top1% in-subspace |M|/|gbar|")
for j, t in enumerate(["wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down"]):
    s = R[j::7]
    print(f"  {t:7s} {s[:, 4].mean():7.3f} {s[:, 6].mean():7.3f} {s[:, 8].mean():7.3f} {s[:, 9].mean():7.3f} {s[:, 3].mean():8.2f}")
