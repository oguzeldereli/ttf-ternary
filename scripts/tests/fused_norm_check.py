"""kernel.rms_norm_qwen against transformers' Qwen3_5RMSNorm (17 Oct): output and gradients (x, weight) for bf16
inputs at the 27B's shapes (hidden 5120; the q / k norms over head_dim 256), max differences in units of bf16's
precision, and the time of forward + backward.
  python scripts/tests/fused_norm_check.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as Q
from bitnet.kernel import rms_norm_qwen

dev = "cuda"; torch.manual_seed(0)
for shp, d in (((2, 2048, 5120), 5120), ((2, 2048, 24, 256), 256)):
    n = Q.Qwen3_5RMSNorm(d, eps=1e-6).to(dev); n.weight.data.normal_(0, 0.1)
    x0 = (torch.randn(*shp, device=dev) * 3).to(torch.bfloat16); g = torch.randn(*shp, device=dev).to(torch.bfloat16)
    res = {}
    for fused in (False, True):
        x = x0.clone().requires_grad_(); n.weight.grad = None
        y = rms_norm_qwen(x, n.weight, n.eps) if fused else n(x)
        y.backward(g); res[fused] = (y.detach().float(), x.grad.float(), n.weight.grad.float())
        ts = []
        for _ in range(5):
            x = x0.clone().requires_grad_(); torch.cuda.synchronize(); t0 = time.time()
            y = rms_norm_qwen(x, n.weight, n.eps) if fused else n(x)
            y.backward(g); torch.cuda.synchronize(); ts.append(time.time() - t0)
        res[fused] += (min(ts),)
    a, b = res[False], res[True]
    ulp = lambda u, v: float(((u - v).abs() / (u.abs().clamp_min(1e-3) * 2 ** -8)).max())
    print(f"{shp}: y max diff {ulp(a[0], b[0]):.2f} bf16 ulp ({float((a[0] != b[0]).float().mean()):.2e} differ), gx {ulp(a[1], b[1]):.2f} ulp, "
          f"gw rel {float((a[2] - b[2]).norm() / a[2].norm()):.1e}; time {a[3] * 1e3:.2f} -> {b[3] * 1e3:.2f} ms", flush=True)

# SwiGLU (the MLP's silu(gate) * up), [2 x 2048, 17408]
from bitnet.kernel import swiglu
import torch.nn.functional as F
g0 = (torch.randn(4096, 17408, device=dev) * 2).to(torch.bfloat16); u0 = torch.randn(4096, 17408, device=dev).to(torch.bfloat16)
gy = torch.randn(4096, 17408, device=dev).to(torch.bfloat16)
res = {}
for fused in (False, True):
    g = g0.clone().requires_grad_(); u = u0.clone().requires_grad_()
    y = swiglu(g, u) if fused else F.silu(g) * u
    y.backward(gy); res[fused] = (y.detach(), g.grad, u.grad)
    ts = []
    for _ in range(5):
        g = g0.clone().requires_grad_(); u = u0.clone().requires_grad_(); torch.cuda.synchronize(); t0 = time.time()
        y = swiglu(g, u) if fused else F.silu(g) * u
        y.backward(gy); torch.cuda.synchronize(); ts.append(time.time() - t0)
    res[fused] += (min(ts),)
a, b = res[False], res[True]
print(f"swiglu: y differ {float((a[0] != b[0]).float().mean()):.2e}, gg differ {float((a[1] != b[1]).float().mean()):.2e}, "
      f"gu differ {float((a[2] != b[2]).float().mean()):.2e} (shares of entries); time {a[3] * 1e3:.2f} -> {b[3] * 1e3:.2f} ms", flush=True)
