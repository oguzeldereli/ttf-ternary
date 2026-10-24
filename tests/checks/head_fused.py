"""--head-fused (ttf/headloss.py): the fused frozen-head loss matches the autograd chunks (qwen._ce_chunk,
BitTransformer._loss_chunk_kd) in loss and dL/dh, with ignored targets, at a 248k vocabulary."""
import torch, torch.nn.functional as F
from ttf.headloss import head_loss_chunk
from ttf.model import BitTransformer
from ttf.qwen import _ce_chunk

torch.manual_seed(0)
dev = "cuda"
n, d, V = 512, 1024, 248320
w = (torch.randn(V, d, device=dev) * 0.02).to(torch.bfloat16)
h0 = torch.randn(n, d, device=dev).to(torch.bfloat16)
tgt = torch.randint(0, V, (n,), device=dev); tgt[:7] = -1
tlp = F.log_softmax(torch.randn(n, V, device=dev) * 2, -1).to(torch.bfloat16)
for name, ref, kd, a in (("kd0.75", lambda h: BitTransformer._loss_chunk_kd(h, w, tgt, tlp, 0.75), tlp, 0.75),
                         ("kd1", lambda h: BitTransformer._loss_chunk_kd(h, w, tgt, tlp, 1.0), tlp, 1.0),
                         ("ce", lambda h: _ce_chunk(h, w, tgt), None, 0.0)):
    h1 = h0.clone().requires_grad_(); l1 = ref(h1); (2.5 * l1).backward()
    h2 = h0.clone().requires_grad_(); l2 = head_loss_chunk(h2, w, tgt, kd, a); (2.5 * l2).backward()
    el = abs(float(l1 - l2)) / abs(float(l1)); eg = float((h1.grad.float() - h2.grad.float()).norm() / h1.grad.float().norm())
    print(f"{name}: loss rel {el:.1e}, grad rel {eg:.1e}")
    assert el < 1e-5 and eg < 1.5e-2, name   # dL/dh is rounded to bf16 before the upstream scale (one extra bf16 rounding)
print("ok")
