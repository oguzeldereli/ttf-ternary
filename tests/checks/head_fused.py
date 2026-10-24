"""--head-fused (ttf/headloss.py): the fused frozen-head loss matches the autograd chunks (qwen._ce_chunk,
BitTransformer._loss_chunk_kd) in loss and dL/dh, with ignored targets, at a 248k vocabulary."""
import torch, torch.nn.functional as F
from ttf.headloss import head_loss_chunk
from ttf.model import BitTransformer
from ttf.qwen import _ce_chunk

torch.manual_seed(0)
dev = "cuda"
n, d, V = 256, 1024, 248320
w = (torch.randn(V, d, device=dev) * 0.02).to(torch.bfloat16)
h0 = torch.randn(n, d, device=dev).to(torch.bfloat16)
tgt = torch.randint(0, V, (n,), device=dev); tgt[:7] = -1
tlp = F.log_softmax(torch.randn(n, V, device=dev) * 2, -1).to(torch.bfloat16)
for name, ref, kd, a in (("kd0.75", lambda h: BitTransformer._loss_chunk_kd(h, w, tgt, tlp, 0.75), tlp, 0.75),
                         ("kd1", lambda h: BitTransformer._loss_chunk_kd(h, w, tgt, tlp, 1.0), tlp, 1.0),
                         ("ce", lambda h: _ce_chunk(h, w, tgt), None, 0.0)):
    h1 = h0.clone().requires_grad_(); l1 = ref(h1); (2.5 * l1).backward()
    h2 = h0.clone().requires_grad_(); l2 = head_loss_chunk(h2, w, tgt, kd, a); (2.5 * l2).backward()
    # fp64 reference of dL/dh (both bf16 paths round dz and the GEMM output once each, at different points)
    z = (h0.double() @ w.double().T); lp = torch.log_softmax(z, -1); okm = (tgt != -1).double()[:, None]
    if kd is None:
        dz = lp.exp() * okm; dz[torch.arange(n, device=dev), tgt.clamp_min(0)] -= okm[:, 0]
    else:
        pt = kd.double().exp(); ca = 0.0 if a >= 1 else 1 - a
        dz = lp.exp() * (ca * okm + a * pt.sum(-1, keepdim=True)) - a * pt
        dz[torch.arange(n, device=dev), tgt.clamp_min(0)] -= ca * okm[:, 0]
    g64 = 2.5 * (dz @ w.double())
    el = abs(float(l1 - l2)) / abs(float(l1))
    e1 = float((h1.grad.double() - g64).norm() / g64.norm()); e2 = float((h2.grad.double() - g64).norm() / g64.norm())
    print(f"{name}: loss rel {el:.1e}; dL/dh error vs fp64: autograd {e1:.1e}, fused {e2:.1e}")
    assert el < 1e-5 and e2 < 1.5 * e1 + 1e-4, name
print("ok")
