"""Per-weight curvature, measured one weight at a time, vs cheap estimators.

For each sampled weight: flip it by one trit level in the signSGD direction (and the
opposite direction) on the dense surrogate and measure the loss change directly.
All quantities in flip units (weight = beta*q):  g = beta*dL/dW,  H = beta^2*d2L/dW2.
  H_central = L(+1) + L(-1) - 2 L0          (no gradient needed)
  d = |g| / H                                (Newton displacement, in flips)
A lone flip helps iff d > 1/2 and is "right-sized" around d ~ 1.
"""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from ttf.flip import build_kernel_transformer, KernelTernaryLinear
from ttf.bitlinear import _act_quant_ste
from ttf.kernel import unpack_rows, trit_beta
from ttf.train import get_batch
from ttf.model import BitTransformer
dev = "cuda"
import os
NOQ = os.environ.get("NOQ") == "1"
CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/lr_ctl/ckpt.pt"
B, SEQ, LABELS, PER_BAND = 4, 1024, 4, 80
BitTransformer.loss_chunk = 1024
BANDS = [(0, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 0.1), (0.1, 0.5), (0.5, 1.0)]


@staticmethod
def _loss_chunk64(h, w, tgt):      # float64 accumulation: single-flip effects are ~1e-7
    return F.cross_entropy(F.linear(h, w).float(), tgt, ignore_index=-1,
                           reduction="none").double().sum()
BitTransformer._loss_chunk = _loss_chunk64


class DenseTern(nn.Module):
    def __init__(self, q, beta, act_bits=8):
        super().__init__()
        self.weight = nn.Parameter((beta * q).float()); self.act_bits = act_bits
        self.cap = None                    # (dict, key, scale): accumulate (gy^2)^T (xq^2) in backward
    def forward(self, x):
        # NOQ=1: no activation rounding. The rounded loss is piecewise constant with ~1e-4
        # jumps per weight nudge, larger than a single flip's effect.
        xq = x.float() if NOQ else _act_quant_ste(x, self.act_bits).float()
        y = F.linear(xq, self.weight)
        if self.cap is not None:
            acc, key, sc = self.cap
            x2 = xq.detach().reshape(-1, xq.shape[-1]) ** 2
            def hook(g):
                g2 = g.detach().reshape(-1, g.shape[-1]) ** 2
                acc[key] = acc.get(key, 0) + sc * (g2.t() @ x2)
            y.register_hook(hook)
        return y


b = torch.load(CKPT, map_location="cpu", weights_only=False)
torch.manual_seed(1337)
m = build_kernel_transformer(b["cfg"], grad_checkpoint=False, beta=True, int8=True)
m.load_state_dict(b["model"], strict=False)
BT = {}
for n, l in list(m.named_modules()):
    if isinstance(l, KernelTernaryLinear):
        BT[n] = trit_beta(l.wpacked, l.K).item()
        par = m.get_submodule(n.rsplit(".", 1)[0])
        setattr(par, n.rsplit(".", 1)[1], DenseTern(unpack_rows(l.wpacked, l.K).float(), BT[n], l.act_bits))
for p in m.parameters(): p.data = p.data.float()
m = m.cuda().train()
names = [n for n, l in m.named_modules() if isinstance(l, DenseTern)]
L = {n: m.get_submodule(n) for n in names}

train = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")
x, y = get_batch(train, B, SEQ, dev, torch.Generator().manual_seed(777))
M = x.numel()


def loss_only():
    with torch.no_grad():
        return m(x, y)[1].item()


def grads(targets, cap=None, keep=True):
    for n in names: L[n].weight.grad = None; L[n].cap = None if cap is None else (cap, n, BT[n] ** 2 * M)
    loss = m(x, targets)[1]; loss.backward()
    for n in names: L[n].cap = None
    out = {n: L[n].weight.grad for n in names} if keep else None
    for p in m.parameters(): p.grad = None
    return loss.item(), out


# gradient + empirical Fisher diag (per position), real labels
EF = {}
l0, g = grads(y, cap=EF)
G = {n: BT[n] * g[n] for n in names}; del g
# MC-Fisher (per position), labels sampled from the model
KF = {}
with torch.no_grad():
    logits = m(x)[0].float()
    probs = logits.softmax(-1).reshape(-1, logits.shape[-1]); del logits
for s in range(LABELS):
    torch.manual_seed(9000 + s)
    grads(torch.multinomial(probs, 1).view_as(y), cap=KF, keep=False)
for n in names: KF[n] /= LABELS
del probs
for n in names: L[n].weight.grad = None
torch.cuda.empty_cache()
print(f"loss {l0:.6f} on {M} tokens; repeat {loss_only():.9f} {loss_only():.9f}", flush=True)

# sample weights by global |g| quantile band
absg = torch.cat([G[n].abs().flatten() for n in names])
offs = np.cumsum([0] + [G[n].numel() for n in names])
order = torch.argsort(absg, descending=True)
N = absg.numel()
gen = torch.Generator().manual_seed(4242)
rows = []
L0 = loss_only()
for bi, (lo, hi) in enumerate(BANDS):
    a, z = int(lo * N), max(int(hi * N), int(lo * N) + PER_BAND)
    pick = order[a:z][torch.randperm(z - a, generator=gen)[:PER_BAND].to(order.device)].cpu().numpy()
    for gi in pick:
        li = int(np.searchsorted(offs, gi, side="right") - 1); n = names[li]
        r, c = divmod(int(gi - offs[li]), G[n].shape[1])
        gq = G[n][r, c].item(); s = -np.sign(gq); w = L[n].weight
        with torch.no_grad():
            w[r, c] += s * BT[n]; lp = loss_only()
            w[r, c] -= 2 * s * BT[n]; lm = loss_only()
            w[r, c] += s * BT[n]
        rows.append((bi, li, r, c, gq, lp - L0, lm - L0, EF[n][r, c].item(), KF[n][r, c].item(),
                     float(absg.mean())))
    print(f"band {bi} done", flush=True)

A = np.array(rows)
torch.save({"rows": A, "bands": BANDS, "names": names, "ckpt": CKPT, "M": M}, f"checkpoints/curv_single{'_noq' if NOQ else ''}.pt")

band, gq, dp, dm, ef, kf = A[:, 0], np.abs(A[:, 4]), A[:, 5], A[:, 6], A[:, 7], A[:, 8]
H = dp + dm                      # central second difference (flip units)
Hone = 2 * (dp + gq)             # one-sided, using the autograd gradient
gfd = (dm - dp) / 2              # finite-difference |g| along the flip direction
print("\nband        |g|med   fd/|g|  dL_flip   1st-order   H_c med   d=|g|/H_c med  [q25,q75]   frac d<0.5  frac H<0")
for bi, (lo, hi) in enumerate(BANDS):
    k = band == bi
    d = gq[k] / np.where(H[k] > 0, H[k], np.nan)
    print(f"{lo*100:>7.3g}-{hi*100:<5.3g}% {np.median(gq[k]):8.2e} {np.median(gfd[k]/gq[k]):6.2f} "
          f"{np.median(dp[k]):9.2e} {np.median(-gq[k]):9.2e} {np.median(H[k]):9.2e} "
          f"{np.nanmedian(d):8.2f}   [{np.nanpercentile(d,25):.2f},{np.nanpercentile(d,75):.2f}] "
          f"{np.mean(gq[k]/np.maximum(H[k],1e-30) < 0.5):8.2f} {np.mean(H[k] < 0):8.2f}")


def spear(a, c):
    ra = np.argsort(np.argsort(a)); rc = np.argsort(np.argsort(c)); return np.corrcoef(ra, rc)[0, 1]


print("\nestimator vs measured H_c (weights with resolvable curvature, top 3 bands and all):")
for lab, est in (("EF", ef), ("MC-Fisher", kf), ("g^2", gq ** 2)):
    for sel_lab, sel in (("top-3 bands", band <= 2), ("all", band >= 0)):
        ok = sel & (H > 0)
        rat = np.median(H[ok] / est[ok])
        print(f"  {lab:10s} {sel_lab:12s} spearman {spear(est[ok], H[ok]):+.3f}   median H/est {rat:.3g}")
# does the estimator pick out right-sized flips? (calibrated by one global constant)
print("\nclassification: true d in [0.5,2] vs estimated d in [0.5,2] (estimator scaled by median ratio on all):")
dtrue = gq / np.where(H > 0, H, np.inf)
for lab, est in (("EF", ef), ("MC-Fisher", kf), ("g^2", gq ** 2)):
    ok = H > 0; c = np.median(H[ok] / est[ok]); dest = gq / (c * est)
    t = (dtrue >= 0.5) & (dtrue <= 2); e = (dest >= 0.5) & (dest <= 2)
    print(f"  {lab:10s} true-right-sized {t.sum():3d}  est-right-sized {e.sum():3d}  both {(t&e).sum():3d}  "
          f"precision {(t&e).sum()/max(e.sum(),1):.2f}  recall {(t&e).sum()/max(t.sum(),1):.2f}")
