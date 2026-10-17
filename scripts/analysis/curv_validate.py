"""Validate cheap curvature estimators against a Hutchinson reference.

All quantities in flip units q (weight = beta*q):  g_q = beta*dL/dW,  H_q = beta^2*d2L/dW2.
A flip helps iff |g_q| > H_q/2.
"""
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from ttf.config import TrainConfig
from ttf.flip import build_kernel_transformer, KernelTernaryLinear
from ttf.bitlinear import _act_quant_ste
from ttf.kernel import unpack_rows, trit_beta, pack_rows
from ttf.train import get_batch
from ttf.model import BitTransformer
dev = "cuda"
HUT, LABELS, SEQ = 64, 8, 1024
BitTransformer.loss_chunk = 1024


class DenseTern(nn.Module):
    def __init__(self, q, beta, act_bits=8):
        super().__init__()
        self.weight = nn.Parameter((beta * q).float()); self.act_bits = act_bits
        self.beta = beta; self.cap = False; self.xq = None; self.gy = None
    def forward(self, x):
        xq = _act_quant_ste(x, self.act_bits).float()
        y = F.linear(xq, self.weight)
        if self.cap:
            self.xq = xq.detach().reshape(-1, xq.shape[-1])
            y.register_hook(lambda g: setattr(self, "gy", g.detach().reshape(-1, g.shape[-1])))
        return y


b = torch.load("checkpoints/armA_cosine/ckpt.pt", map_location="cpu", weights_only=False)
torch.manual_seed(1337)
m = build_kernel_transformer(b["cfg"], grad_checkpoint=False, beta=True, int8=True)
for p in m.float_tail_parameters(): p.data = p.data.to(torch.bfloat16)
m.load_state_dict(b["model"], strict=False); m = m.cuda()
TR, BT = {}, {}
for n, l in list(m.named_modules()):
    if isinstance(l, KernelTernaryLinear):
        TR[n] = unpack_rows(l.wpacked, l.K).float(); BT[n] = trit_beta(l.wpacked, l.K).item()
        par = m.get_submodule(n.rsplit(".", 1)[0])
        setattr(par, n.rsplit(".", 1)[1], DenseTern(TR[n], BT[n], l.act_bits))
for p in m.parameters(): p.data = p.data.float()
m = m.cuda().train()
names = [n for n, l in m.named_modules() if isinstance(l, DenseTern)]
L = {n: m.get_submodule(n) for n in names}
Ws = [L[n].weight for n in names]

train = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")
g0g = torch.Generator().manual_seed(777)
x, y = get_batch(train, 1, SEQ, dev, g0g)
M = x.numel()

def grads(targets, scale=0.0, vs=None, cap=False):
    if vs is not None and scale:
        with torch.no_grad():
            for w, v in zip(Ws, vs): w.add_(v, alpha=scale)
    for w in Ws: w.grad = None
    for n in names: L[n].cap = cap
    with sdpa_kernel(SDPBackend.MATH):
        loss = m(x, targets)[1]; loss.backward()
    out = [w.grad.detach().clone() for w in Ws]
    for n in names: L[n].cap = False
    if vs is not None and scale:
        with torch.no_grad():
            for w, v in zip(Ws, vs): w.sub_(v, alpha=scale)
    return loss.item(), out

# --- gradient (real labels) + empirical Fisher (per position) -----------------
l0, g0 = grads(y, cap=True)
G = {n: BT[n] * g for n, g in zip(names, g0)}
EF = {n: (BT[n]**2) * M * ((L[n].gy**2).t() @ (L[n].xq**2)) for n in names}
print(f"loss {l0:.4f} on {M} tokens", flush=True)

# --- Hutchinson reference ------------------------------------------------------
HH = {n: torch.zeros_like(G[n]) for n in names}
for s in range(HUT):
    torch.manual_seed(5000 + s)
    vs = [torch.randint(0, 2, w.shape, device=dev, dtype=torch.float32)*2-1 for w in Ws]
    eps = 1e-3 * BT[names[0]]
    _, g1 = grads(y, eps, vs)
    for n, v, a, c in zip(names, vs, g0, g1): HH[n] += (BT[n]**2) * v * (c - a) / eps
    del vs, g1
for n in names: HH[n] /= HUT
print(f"hutchinson reference: {HUT} samples", flush=True)

# --- MC-Fisher with labels sampled from the model ------------------------------
KF = {n: torch.zeros_like(G[n]) for n in names}      # per-position (K-FAC diagonal)
SQ = {n: torch.zeros_like(G[n]) for n in names}      # squared sampled gradient
with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
    logits, _ = m(x)
probs = logits.float().softmax(-1).reshape(-1, logits.shape[-1])
for s in range(LABELS):
    torch.manual_seed(9000 + s)
    ys = torch.multinomial(probs, 1).view_as(y)
    _, gs = grads(ys, cap=True)
    for n, gg in zip(names, gs):
        KF[n] += (BT[n]**2) * M * ((L[n].gy**2).t() @ (L[n].xq**2))
        SQ[n] += (BT[n]**2) * M * gg**2
for n in names: KF[n] /= LABELS; SQ[n] /= LABELS
del probs, logits
print(f"MC-Fisher: {LABELS} label samples\n", flush=True)

torch.save({"G": {n: G[n].cpu() for n in names}, "HUT": {n: HH[n].cpu() for n in names},
            "KF": {n: KF[n].cpu() for n in names}, "SQ": {n: SQ[n].cpu() for n in names},
            "EF": {n: EF[n].cpu() for n in names}, "betas": BT}, "checkpoints/curv_validate.pt")

def cat(D): return torch.cat([D[n].flatten()[::53] for n in names]).float()
g = cat(G).abs(); ref = cat(HH)
def spearman(a, c):
    ra = a.argsort().argsort().float(); rc = c.argsort().argsort().float()
    return float(torch.corrcoef(torch.stack([ra, rc]))[0, 1])
print(f"Hutchinson reference: {100*(ref>0).float().mean():.1f}% positive, "
      f"median |g|/H (pos) {(g[ref>0]/ref[ref>0]).median():.4f}, "
      f"justified {100*(g > 0.5*ref.clamp_min(0)).float().mean():.1f}% / "
      f"{100*((ref>0)&(g>0.5*ref)).float().mean():.1f}% (strict)\n")
print(f"{'estimator':<28}{'spearman vs Hut':>16}{'median H / Hut':>16}{'median |g|/H':>14}{'justified %':>13}")
pos = ref > 0
for name, D in [("MC-Fisher per-position", KF), ("MC-Fisher squared grad", SQ),
                ("empirical Fisher", EF)]:
    h = cat(D)
    print(f"{name:<28}{spearman(h[pos], ref[pos]):>16.3f}"
          f"{float((h[pos]/ref[pos]).median()):>16.3f}{float((g/h.clamp_min(1e-30)).median()):>14.4f}"
          f"{100*(g > 0.5*h).float().mean():>12.2f}%")
