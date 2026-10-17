"""True per-weight curvature for the ternary body, via Hutchinson.

The packed-trit layers are not autograd parameters, so we build a dense surrogate
whose weight is W = beta * q (a real nn.Parameter) and which computes the same
forward. Then diag(H) ~ E_v[v * Hv] with Rademacher v, each sample one
Hessian-vector product.

Units: the flip moves a weight by beta, so in "flip units"
    g_q = beta * dL/dW,    H_q = beta^2 * d2L/dW2
and a flip is justified when |g_q| > H_q/2 (Newton displacement > half a step).
"""
import argparse, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from ttf.config import TrainConfig
from ttf.layers import build_kernel_transformer, KernelTernaryLinear
from ttf.bitlinear import _act_quant_ste
from ttf.kernel import unpack_rows, trit_beta
from ttf.train import get_batch
from ttf.model import BitTransformer

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/armA_cosine/ckpt.pt")
ap.add_argument("--samples", type=int, default=8)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--seq", type=int, default=512)
args = ap.parse_args()
dev = "cuda"


class DenseTern(nn.Module):
    """y = act_quant_ste(x) @ W^T, with W = beta * q  (same forward as the kernel path)."""
    def __init__(self, q, beta, act_bits=8):
        super().__init__()
        self.weight = nn.Parameter((beta * q).float())
        self.act_bits = act_bits
        self.beta = beta

    def forward(self, x):
        return F.linear(_act_quant_ste(x, self.act_bits).float(), self.weight)


def load(ckpt):
    b = torch.load(ckpt, map_location="cpu", weights_only=False)
    torch.manual_seed(1337)
    m = build_kernel_transformer(b["cfg"], grad_checkpoint=False, beta=True, int8=True)
    for p in m.float_tail_parameters():
        p.data = p.data.to(torch.bfloat16)
    m.load_state_dict(b["model"], strict=False)
    m = m.cuda()
    trits, betas = {}, {}
    for n, l in m.named_modules():
        if isinstance(l, KernelTernaryLinear):
            trits[n] = unpack_rows(l.wpacked, l.K).float()
            betas[n] = trit_beta(l.wpacked, l.K).item()
    return b, m, trits, betas


def densify(m, trits, betas):
    """Swap every ternary layer for its dense surrogate, in place."""
    for n, l in list(m.named_modules()):
        if isinstance(l, KernelTernaryLinear):
            parent = m.get_submodule(n.rsplit(".", 1)[0])
            setattr(parent, n.rsplit(".", 1)[1], DenseTern(trits[n], betas[n], l.act_bits))
    for p in m.parameters():
        p.data = p.data.float()
    return m.cuda()


blob, kern, TRITS, BETAS = load(args.ckpt)
tc = TrainConfig(); tc.batch_size, tc.seq_len = args.batch, args.seq
BitTransformer.loss_chunk = 512
train = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")
g0 = torch.Generator().manual_seed(555)
x, y = get_batch(train, tc.batch_size, tc.seq_len, dev, g0)

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    l_kernel = kern(x, y)[1].item()
del kern
torch.cuda.empty_cache()

torch.manual_seed(1337)
m = build_kernel_transformer(blob["cfg"], grad_checkpoint=False, beta=True, int8=True)
for p in m.float_tail_parameters():
    p.data = p.data.to(torch.bfloat16)
m.load_state_dict(blob["model"], strict=False)
m = densify(m.cuda(), TRITS, BETAS).train()
names = [n for n, l in m.named_modules() if isinstance(l, DenseTern)]
Ws = [m.get_submodule(n).weight for n in names]

def grads_at(scale=0.0, vs=None):
    """dL/dW at W (+ scale*v), same batch, no graph retained."""
    if vs is not None and scale != 0.0:
        with torch.no_grad():
            for w, v in zip(Ws, vs): w.add_(v, alpha=scale)
    for w in Ws:
        if w.grad is not None: w.grad = None
    with sdpa_kernel(SDPBackend.MATH):
        loss = m(x, y)[1]
        loss.backward()
    out = [w.grad.detach().clone() for w in Ws]
    if vs is not None and scale != 0.0:
        with torch.no_grad():
            for w, v in zip(Ws, vs): w.sub_(v, alpha=scale)
    return loss.item(), out

l0, g0 = grads_at()
print(f"loss: kernel path {l_kernel:.4f} | dense surrogate {l0:.4f} "
      f"(diff {abs(l0-l_kernel):.4f})", flush=True)
G = {n: (BETAS[n] * g).float() for n, g in zip(names, g0)}                   # dL/dq
diag = {n: torch.zeros_like(g) for n, g in zip(names, g0)}
EPS = {n: 1e-3 * BETAS[n] for n in names}                # small vs the weight scale beta
for s_i in range(args.samples):
    torch.manual_seed(1000 + s_i)
    vs = [torch.randint(0, 2, w.shape, device=dev, dtype=torch.float32) * 2 - 1 for w in Ws]
    eps = EPS[names[0]]
    _, g1 = grads_at(eps, vs)                            # finite-difference Hessian-vector
    for n, v, a, b_ in zip(names, vs, g0, g1):
        diag[n] += v * (b_ - a) / eps
    del vs, g1
    torch.cuda.empty_cache()
    print(f"  hutchinson sample {s_i+1}/{args.samples}", flush=True)
H = {n: (BETAS[n] ** 2) * diag[n] / args.samples for n in names}             # d2L/dq2

torch.save({"G": {n: G[n].cpu() for n in names}, "H": {n: H[n].cpu() for n in names},
            "betas": BETAS, "loss": l0}, "checkpoints/hess_probe.pt")

allg = torch.cat([G[n].abs().flatten()[::37] for n in names]).float()
allh = torch.cat([H[n].flatten()[::37] for n in names]).float()
pos = allh > 0
ratio = allg[pos] / allh[pos]
print(f"\ncurvature H_q: {100*pos.float().mean():.1f}% positive | "
      f"median {allh[pos].median():.3e}")
print(f"Newton displacement |g|/H in flip units: median {ratio.median():.4f}  "
      f"p90 {torch.quantile(ratio, 0.9):.3f}  p99 {torch.quantile(ratio, 0.99):.3f}")
print(f"fraction of weights where a flip is JUSTIFIED (|g| > H/2): "
      f"{100*(allg > 0.5*allh.clamp_min(0)).float().mean():.2f}%")
print("\nsaved -> checkpoints/hess_probe.pt")
