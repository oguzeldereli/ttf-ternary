"""--fused-scales against the unfused layer (17 Oct): one KernelTernaryLinear with row / column scales (fp16, as
--tail lean) and a rank-16 'add' adapter, bf16 input; forward and backward with and without the fused kernels. The
output and the input gradient must be bit-identical, the scale gradients equal to rounding (their sums run in another
order); then the time of forward + backward of both.
  python scripts/tests/fused_scales_check.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
import ttf.layers as FL
from ttf.layers import KernelTernaryLinear, enable_rc_scales

dev = "cuda"
for (K, N, T) in ((512, 384, 1000), (5120, 17408, 4096), (17408, 5120, 4096)):
    torch.manual_seed(0)
    l = KernelTernaryLinear(K, N, int8=True).to(dev)
    l.row_scale = torch.nn.Parameter((torch.rand(N, device=dev) + 0.5).half())
    l.col_scale = torch.nn.Parameter((torch.rand(K, device=dev) + 0.5).half())
    l.enable_lowrank_mag("add", 16); l.mag_A.data.normal_(0, 0.02)
    l.capture = True; l.on_grad = None
    x0 = torch.randn(T, K, device=dev, dtype=torch.bfloat16)
    g0 = torch.randn(T, N, device=dev, dtype=torch.bfloat16)
    out = {}
    for fz in (False, True):
        FL.FUSED_SCALES[0] = fz
        for p in (l.row_scale, l.col_scale, l.mag_A, l.mag_B):
            p.grad = None
        x = x0.clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = l(x)
        y.backward(g0)
        out[fz] = (y.detach(), x.grad, l.row_scale.grad.float(), l.col_scale.grad.float(), l.mag_A.grad.float(), l.mag_B.grad.float())
        ts = []
        for _ in range(3):
            x = x0.clone().requires_grad_(); torch.cuda.synchronize(); t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = l(x)
            y.backward(g0); torch.cuda.synchronize(); ts.append(time.time() - t0)
        out[fz] += (min(ts),)
    a, b = out[False], out[True]
    rel = lambda u, v: float((u.float() - v.float()).norm() / u.float().norm().clamp_min(1e-30))
    print(f"{K}->{N} T {T}: y identical {torch.equal(a[0], b[0])}, gx identical {torch.equal(a[1], b[1])}; rel diff gr {rel(a[2], b[2]):.1e} "
          f"gc {rel(a[3], b[3]):.1e} gA {rel(a[4], b[4]):.1e} gB {rel(a[5], b[5]):.1e}; time {a[6] * 1e3:.1f} -> {b[6] * 1e3:.1f} ms", flush=True)
