"""--head-fused (ttf/headloss.py): the fused frozen-head loss matches the autograd chunks (qwen._ce_chunk,
BitTransformer._loss_chunk_kd) in loss and dL/dh, with ignored targets, at a 248k vocabulary."""
import torch, torch.nn.functional as F
from ttf.headloss import head_loss_chunk
from ttf.model import BitTransformer
from ttf.qwen import _ce_chunk, _kd_topk_chunk
from ttf.headloss import head_loss_topk_chunk

torch.manual_seed(0)
dev = "cuda"
n, d, V = 256, 1024, 248320
w = (torch.randn(V, d, device=dev) * 0.02).to(torch.bfloat16)
h0 = torch.randn(n, d, device=dev).to(torch.bfloat16)
tgt = torch.randint(0, V, (n,), device=dev); tgt[:7] = -1
tlp = F.log_softmax(torch.randn(n, V, device=dev) * 2, -1).to(torch.bfloat16)
for name, ref, kd, a in (("kd1.5", lambda h: BitTransformer._loss_chunk_kd(h, w, tgt, tlp, 1.5), tlp, 1.5),   # a > 1: KL alone
                         ("kd0.75", lambda h: BitTransformer._loss_chunk_kd(h, w, tgt, tlp, 0.75), tlp, 0.75),
                         ("kd1", lambda h: BitTransformer._loss_chunk_kd(h, w, tgt, tlp, 1.0), tlp, 1.0),
                         ("ce", lambda h: _ce_chunk(h, w, tgt), None, 0.0)):
    h1 = h0.clone().requires_grad_(); l1 = ref(h1); (2.5 * l1).backward()
    h2 = h0.clone().requires_grad_(); l2 = head_loss_chunk(h2, w, tgt, kd, a); (2.5 * l2).backward()
    # fp64 reference of dL/dh (both bf16 paths round dz and the GEMM output once each, at different points)
    z = (h0.double() @ w.double().T); lp = torch.log_softmax(z, -1); okm = (tgt != -1).double()[:, None]
    if kd is None:
        dz = lp.exp() * okm; dz[torch.arange(n, device=dev), tgt.clamp_min(0)] -= okm[:, 0]
    else:
        pt = kd.double().exp(); ca = 0.0 if a >= 1 else 1 - a; ak = min(a, 1.0)
        dz = lp.exp() * (ca * okm + ak * pt.sum(-1, keepdim=True)) - ak * pt
        dz[torch.arange(n, device=dev), tgt.clamp_min(0)] -= ca * okm[:, 0]
    g64 = 2.5 * (dz @ w.double())
    el = abs(float(l1 - l2)) / abs(float(l1))
    e1 = float((h1.grad.double() - g64).norm() / g64.norm()); e2 = float((h2.grad.double() - g64).norm() / g64.norm())
    print(f"{name}: loss rel {el:.1e}; dL/dh error vs fp64: autograd {e1:.1e}, fused {e2:.1e}")
    assert el < 1e-5 and e2 < 1.5 * e1 + 1e-4, name
# the top-k cache losses: loss and dL/dh against the autograd chunk (fp64 reference not repeated: same roundings)
ids = torch.topk(tlp.float(), 32, -1).indices; tk = tlp.gather(1, ids)
for mode in ("rest", "renorm"):
    for a in (0.75, 1.0):
        h1 = h0.clone().requires_grad_(); l1 = _kd_topk_chunk(h1, w, tgt, ids, tk, a, mode); (2.5 * l1).backward()
        h2 = h0.clone().requires_grad_(); l2 = head_loss_topk_chunk(h2, w, tgt, ids, tk, a, mode); (2.5 * l2).backward()
        el = abs(float(l1 - l2)) / abs(float(l1)); eg = float((h1.grad.float() - h2.grad.float()).norm() / h1.grad.float().norm())
        print(f"topk {mode} a={a}: loss rel {el:.1e}, dL/dh rel to autograd {eg:.1e}")
        assert el < 1e-5 and eg < 2e-2, (mode, a)
print("ok")
# a trained head: dL/dW and dL/dh over 3 chunks summed and scaled, against autograd
from ttf.headloss import head_loss_train_chunk
wt = w.clone().float().requires_grad_()
hs = [torch.randn(n, d, device=dev).to(torch.bfloat16) for _ in range(3)]
h1 = [x.clone().requires_grad_() for x in hs]
l1 = sum(BitTransformer._loss_chunk_kd(x, wt.to(torch.bfloat16), tgt, tlp, 0.75) for x in h1) / 1000.0; l1.backward()
g1 = wt.grad.clone(); wt.grad = None
h2 = [x.clone().requires_grad_() for x in hs]; acc = {}
wc = wt.to(torch.bfloat16)
l2 = sum(head_loss_train_chunk(x, wc, tgt, tlp, 0.75, acc) for x in h2) / 1000.0; l2.backward()
g2 = wt.grad
ew = float((g1 - g2).norm() / g1.norm()); eh = max(float((a.grad.float() - b.grad.float()).norm() / a.grad.float().norm()) for a, b in zip(h1, h2))
print(f"trained head: loss rel {abs(float(l1 - l2)) / abs(float(l1)):.1e}, dW rel {ew:.1e}, dh rel {eh:.1e}, acc cleared {acc == {}}")
assert ew < 2e-2 and eh < 2e-2 and acc == {}
print("ok2")
